// Virtual geometry culling compute shader.
//
// One thread per meshlet. Tests each meshlet against the view frustum and the
// backface cone. Visible meshlets write a full DrawIndexedIndirect command into
// the indirect buffer; invisible meshlets write instance_count = 0.
//
// Because every meshlet slot is unconditionally written, there is no need for
// atomic counters or separate compaction passes. The GPU simply skips draw
// commands where instance_count == 0.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

/// Mirrors GpuMeshletEntry (Rust, 64 bytes).
struct MeshletEntry {
    center:         vec3<f32>,
    radius:         f32,
    cone_apex:      vec3<f32>,
    cone_cutoff:    f32,
    cone_axis:      vec3<f32>,
    lod_error:      f32,
    first_index:    u32,
    index_count:    u32,
    vertex_offset:  i32,
    instance_index: u32,
}

/// Mirrors GpuInstanceData (Rust, 144 bytes).
struct InstanceData {
    transform:    mat4x4<f32>,
    normal_mat_0: vec4<f32>,
    normal_mat_1: vec4<f32>,
    normal_mat_2: vec4<f32>,
    bounds:       vec4<f32>,
    mesh_id:      u32,
    material_id:  u32,
    flags:        u32,
    _pad:         u32,
}

/// Mirrors wgpu::util::DrawIndexedIndirectArgs (20 bytes, but aligned to 4).
struct DrawIndexedIndirect {
    index_count:    u32,
    instance_count: u32,
    first_index:    u32,
    base_vertex:    i32,
    first_instance: u32,
}

struct CullUniforms {
    meshlet_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform>          camera:    Camera;
@group(0) @binding(1) var<uniform>          cull_uni:  CullUniforms;
@group(0) @binding(2) var<storage, read>    meshlets:  array<MeshletEntry>;
@group(0) @binding(3) var<storage, read>    instances: array<InstanceData>;
@group(0) @binding(4) var<storage, read_write> indirect: array<DrawIndexedIndirect>;

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Backface cone test.
/// Returns true if the meshlet might be *front-facing* for the current camera.
/// cone_cutoff > 1.0 disables cone culling (mixed-winding or nearly flat).
fn cone_visible(
    apex_ws: vec3<f32>,
    axis_ws: vec3<f32>,
    cutoff:  f32,
    cam_pos: vec3<f32>,
) -> bool {
    if cutoff > 1.0 {
        return true;
    }
    let view_dir = normalize(cam_pos - apex_ws);
    return dot(view_dir, -axis_ws) < cutoff;
}

// ─── Main ────────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= cull_uni.meshlet_count {
        return;
    }

    let m    = meshlets[idx];
    let inst = instances[m.instance_index];

    let model     = inst.transform;
    let center_ws = (model * vec4<f32>(m.center, 1.0)).xyz;

    let scale_x = length(model[0].xyz);
    let scale_y = length(model[1].xyz);
    let scale_z = length(model[2].xyz);
    let world_radius = m.radius * max(scale_x, max(scale_y, scale_z));

    // ── Frustum cull (inline — avoids array<T,N> as return / param type) ─────
    // Each plane: vec4(normal.xyz, d)  with  sign convention  normal·p + d >= 0 = inside.
    let vp = camera.view_proj;
    let pl0 = vec4<f32>(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]); // left
    let pl1 = vec4<f32>(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]); // right
    let pl2 = vec4<f32>(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]); // bottom
    let pl3 = vec4<f32>(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]); // top
    let pl4 = vec4<f32>(vp[0][3] + vp[0][2], vp[1][3] + vp[1][2], vp[2][3] + vp[2][2], vp[3][3] + vp[3][2]); // near
    let pl5 = vec4<f32>(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]); // far

    var visible = (dot(pl0.xyz, center_ws) + pl0.w >= -world_radius)
               && (dot(pl1.xyz, center_ws) + pl1.w >= -world_radius)
               && (dot(pl2.xyz, center_ws) + pl2.w >= -world_radius)
               && (dot(pl3.xyz, center_ws) + pl3.w >= -world_radius)
               && (dot(pl4.xyz, center_ws) + pl4.w >= -world_radius)
               && (dot(pl5.xyz, center_ws) + pl5.w >= -world_radius);

    if visible {
        let apex_ws = (model * vec4<f32>(m.cone_apex, 1.0)).xyz;
        let norm_mat = mat3x3<f32>(
            inst.normal_mat_0.xyz,
            inst.normal_mat_1.xyz,
            inst.normal_mat_2.xyz,
        );
        let axis_ws = normalize(norm_mat * m.cone_axis);
        let cam_pos = camera.position_near.xyz;
        visible = cone_visible(apex_ws, axis_ws, m.cone_cutoff, cam_pos);
    }

    var cmd: DrawIndexedIndirect;
    cmd.index_count    = m.index_count;
    cmd.instance_count = select(0u, 1u, visible);
    cmd.first_index    = m.first_index;
    cmd.base_vertex    = m.vertex_offset;
    cmd.first_instance = m.instance_index;
    indirect[idx] = cmd;
}
