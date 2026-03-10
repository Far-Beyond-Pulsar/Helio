/// Compute shader: build GPU-driven indirect draw buffers
///
/// For each draw call, tests its bounding sphere against the camera frustum
/// (proper 6-plane bounds test, NOT a centre-point check) and only writes a
/// DrawIndexedIndirect command if the draw is potentially visible.  Draws are
/// separated into opaque and transparent indirect buffers, grouped by material
/// for efficient batching.

struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

/// GPU representation of a DrawCall (uploaded from CPU each frame)
struct GpuDrawCall {
    vertex_offset:   u32,
    index_offset:    u32,
    index_count:     u32,
    vertex_count:    u32,
    material_id:     u32,
    transparent_blend: u32,   // bool packed as u32
    _pad0:           u32,
    _pad1:           u32,
    // World-space bounding sphere (set from cpu_data world-space bounds)
    bounds_center:   vec3f,
    bounds_radius:   f32,
}

// Camera uniform — must exactly match the Rust `Camera` struct (144 bytes).
struct CameraUniforms {
    view_proj:     mat4x4f,  // offset   0  (64 bytes)
    position:      vec3f,    // offset  64  (12 bytes)
    time:          f32,      // offset  76  ( 4 bytes)
    view_proj_inv: mat4x4f,  // offset  80  (64 bytes)
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

// Input: draw list uploaded from CPU
@group(1) @binding(0) var<storage, read>       draw_calls:          array<GpuDrawCall>;

// Output: indirect command buffers
@group(1) @binding(1) var<storage, read_write> opaque_indirect:      array<DrawIndexedIndirect>;
@group(1) @binding(2) var<storage, read_write> transparent_indirect: array<DrawIndexedIndirect>;

// Output: atomic draw counts
@group(1) @binding(3) var<storage, read_write> opaque_count:      atomic<u32>;
@group(1) @binding(4) var<storage, read_write> transparent_count: atomic<u32>;

// ── Frustum plane extraction (Gribb-Hartmann, wgpu/Vulkan NDC z∈[0,1]) ───────
//
// Extracts 6 planes from the combined view-projection matrix.
// Each vec4 is (nx, ny, nz, d): inside satisfies dot(n,p)+d >= 0.
fn extract_frustum_planes(m: mat4x4f) -> array<vec4f, 6> {
    // Access matrix columns: m[c][r]
    // Row i = (m[0][i], m[1][i], m[2][i], m[3][i])
    let r0 = vec4f(m[0][0], m[1][0], m[2][0], m[3][0]);
    let r1 = vec4f(m[0][1], m[1][1], m[2][1], m[3][1]);
    let r2 = vec4f(m[0][2], m[1][2], m[2][2], m[3][2]);
    let r3 = vec4f(m[0][3], m[1][3], m[2][3], m[3][3]);

    var planes: array<vec4f, 6>;
    planes[0] = r3 + r0;  // left   (x >= -w)
    planes[1] = r3 - r0;  // right  (x <=  w)
    planes[2] = r3 + r1;  // bottom (y >= -w)
    planes[3] = r3 - r1;  // top    (y <=  w)
    planes[4] = r2;        // near   (z >= 0 in Vulkan NDC)
    planes[5] = r3 - r2;  // far    (z <= w)
    return planes;
}

// Bounds-based sphere-vs-frustum test.
// Returns true if the sphere is potentially visible (may have false positives
// near corners but NEVER culls a visible object).
fn frustum_cull_sphere(
    planes:  array<vec4f, 6>,
    centre:  vec3f,
    radius:  f32,
) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane  = planes[i];
        let normal = plane.xyz;
        let dist   = dot(normal, centre) + plane.w;
        if dist < -(radius * length(normal)) {
            return false;
        }
    }
    return true;
}

@compute @workgroup_size(256)
fn build_indirect_buffers(
    @builtin(global_invocation_id) gid: vec3u,
) {
    let draw_idx   = gid.x;
    let total_draws = arrayLength(&draw_calls);

    if draw_idx >= total_draws { return; }

    let dc = draw_calls[draw_idx];

    // ── Frustum cull using world-space bounding sphere ────────────────────
    let planes = extract_frustum_planes(camera.view_proj);
    if !frustum_cull_sphere(planes, dc.bounds_center, dc.bounds_radius) {
        return;  // Object is outside the camera frustum — skip
    }

    // Build indirect command
    let cmd = DrawIndexedIndirect(
        dc.index_count,
        1u,                   // instance_count
        dc.index_offset,
        i32(dc.vertex_offset),
        0u,                   // first_instance
    );

    // Write to appropriate buffer based on transparency
    if dc.transparent_blend != 0u {
        let idx = atomicAdd(&transparent_count, 1u);
        transparent_indirect[idx] = cmd;
    } else {
        let idx = atomicAdd(&opaque_count, 1u);
        opaque_indirect[idx] = cmd;
    }
}
