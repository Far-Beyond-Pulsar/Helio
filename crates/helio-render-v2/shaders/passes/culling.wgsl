//! GPU frustum culling compute shader.
//!
//! One workgroup thread per draw input.  Reads each object's bounding sphere
//! from the GPU Scene buffer, tests it against 6 frustum half-planes, and
//! appends an `DrawIndexedIndirect` command to either the opaque or transparent
//! output buffers using atomic counters.
//!
//! Dispatch: `ceil(draw_count / 64)` workgroups of 64 threads each.

// ── Shared structs (must match Rust GpuPrimitive and mesh.rs structs) ─────────

struct GpuPrimitive {
    transform:     mat4x4<f32>,
    inv_trans_c0:  vec4<f32>,
    inv_trans_c1:  vec4<f32>,
    inv_trans_c2:  vec4<f32>,
    bounds_center: vec3<f32>,
    bounds_radius: f32,
    material_id:   u32,
    flags:         u32,
    _pad:          vec2<u32>,
}

// One input record per potentially-visible object.
struct DrawInput {
    index_count:    u32,
    first_index:    u32,
    primitive_id:   u32,   // index into gpu_primitives[]
    _pad:           u32,
}

// matches wgpu's DrawIndexedIndirectArgs layout
struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,   // = primitive_id, read by vs_main
}

// ── Bindings ──────────────────────────────────────────────────────────────────

struct CullingUniforms {
    planes:     array<vec4<f32>, 6>,  // frustum half-planes (normal.xyz, d)
    draw_count: u32,
    _pad:       array<u32, 3>,
}

@group(0) @binding(0) var<storage, read>            primitives:        array<GpuPrimitive>;
@group(0) @binding(1) var<uniform>                  culling_uniforms:  CullingUniforms;
@group(0) @binding(2) var<storage, read>            draw_inputs:       array<DrawInput>;
@group(0) @binding(3) var<storage, read_write>      opaque_cmds:       array<DrawIndexedIndirect>;
@group(0) @binding(4) var<storage, read_write>      transparent_cmds:  array<DrawIndexedIndirect>;
@group(0) @binding(5) var<storage, read_write>      opaque_count:      atomic<u32>;
@group(0) @binding(6) var<storage, read_write>      transparent_count: atomic<u32>;

// ── PRIM flags ────────────────────────────────────────────────────────────────

const PRIM_CAST_SHADOW: u32  = 1u;
const PRIM_TRANSPARENT: u32  = 2u;

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns true if the sphere (centre, radius) is fully outside any plane,
/// i.e. the object is culled.  `bounds_radius == f32::MAX` disables culling.
fn sphere_outside_frustum(centre: vec3<f32>, radius: f32) -> bool {
    if radius >= 3.4028235e+38 { return false; }   // skip culling sentinel
    for (var i = 0u; i < 6u; i++) {
        let plane = culling_uniforms.planes[i];
        let dist  = dot(plane.xyz, centre) + plane.w;
        if dist < -radius {
            return true;
        }
    }
    return false;
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= culling_uniforms.draw_count { return; }

    let di   = draw_inputs[idx];
    let prim = primitives[di.primitive_id];

    // Frustum test against world-space bounding sphere.
    if sphere_outside_frustum(prim.bounds_center, prim.bounds_radius) { return; }

    // Object is visible — emit a DrawIndexedIndirect record.
    let cmd = DrawIndexedIndirect(
        di.index_count,
        1u,              // instance_count = 1 (each prim has unique primitive_id)
        di.first_index,
        0,               // base_vertex
        di.primitive_id, // first_instance = primitive_id → read by vs_main via @location(5)
    );

    let transparent = (prim.flags & PRIM_TRANSPARENT) != 0u;
    if transparent {
        let slot = atomicAdd(&transparent_count, 1u);
        transparent_cmds[slot] = cmd;
    } else {
        let slot = atomicAdd(&opaque_count, 1u);
        opaque_cmds[slot] = cmd;
    }
}
