// GPU per-face shadow frustum culling.
// Each thread tests one draw_call against all active dirty shadow faces.
// Visible draws are atomically appended to a per-face compacted indirect list.

const MAX_FACES: u32 = 256u;

struct GpuShadowMatrix {
    mat: mat4x4<f32>,
}

struct GpuInstance {
    model_0:     vec4<f32>,
    model_1:     vec4<f32>,
    model_2:     vec4<f32>,
    model_3:     vec4<f32>,
    normal_0:    vec4<f32>,
    normal_1:    vec4<f32>,
    normal_2:    vec4<f32>,
    bounds:      vec4<f32>,
    mesh_id:     u32,
    material_id: u32,
    flags:       u32,
    _pad:        u32,
}

struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

struct CullUniforms {
    instance_count: u32,
    max_draws_per_face: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform>             uniforms:         CullUniforms;
@group(0) @binding(1) var<storage, read>       shadow_matrices:  array<GpuShadowMatrix>;
@group(0) @binding(2) var<storage, read>       instances:        array<GpuInstance>;
@group(0) @binding(3) var<storage, read>       src_indirect:     array<DrawIndexedIndirect>;
@group(0) @binding(4) var<storage, read_write> dst_indirect:     array<DrawIndexedIndirect>;
@group(0) @binding(5) var<storage, read_write> face_counts:      array<atomic<u32>>;
@group(0) @binding(6) var<storage, read>       face_dirty:       array<u32>;

fn normalize_plane(p: vec4<f32>) -> vec4<f32> {
    let len = length(p.xyz);
    if len > 1e-10 {
        return vec4<f32>(p.xyz / len, p.w / len);
    }
    return p;
}

fn sphere_in_frustum(vp: mat4x4<f32>, center: vec3<f32>, radius: f32) -> bool {
    let p0 = normalize_plane(vp[3] + vp[0]);
    if dot(p0.xyz, center) + p0.w + radius < 0.0 { return false; }
    let p1 = normalize_plane(vp[3] - vp[0]);
    if dot(p1.xyz, center) + p1.w + radius < 0.0 { return false; }
    let p2 = normalize_plane(vp[3] + vp[1]);
    if dot(p2.xyz, center) + p2.w + radius < 0.0 { return false; }
    let p3 = normalize_plane(vp[3] - vp[1]);
    if dot(p3.xyz, center) + p3.w + radius < 0.0 { return false; }
    let p4 = normalize_plane(vp[2]);
    if dot(p4.xyz, center) + p4.w + radius < 0.0 { return false; }
    let p5 = normalize_plane(vp[3] - vp[2]);
    if dot(p5.xyz, center) + p5.w + radius < 0.0 { return false; }
    return true;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let draw_idx = gid.x;
    if draw_idx >= uniforms.instance_count { return; }

    let draw = src_indirect[draw_idx];
    let inst = instances[draw.first_instance];
    let center = inst.bounds.xyz;
    let radius = inst.bounds.w;
    if radius <= 0.0 { return; }

    for (var face = 0u; face < MAX_FACES; face++) {
        if face_dirty[face] == 0u { continue; }

        let vp = shadow_matrices[face].mat;
        if sphere_in_frustum(vp, center, radius) {
            let slot = atomicAdd(&face_counts[face], 1u);
            if slot < uniforms.max_draws_per_face {
                let base = face * uniforms.max_draws_per_face;
                dst_indirect[base + slot] = draw;
            }
        }
    }
}
