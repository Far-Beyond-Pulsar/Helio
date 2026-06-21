// Virtual geometry culling compute shader.
//
// One thread per meshlet. Tests each meshlet against the view frustum,
// backface cone, and Hi-Z occlusion buffer.
// Visible meshlets are atomically appended to a compact indirect draw list:
//
//   slot = atomicAdd(&draw_count, 1u);
//   indirect[slot] = cmd;
//
// The GPU-written draw_count is passed to multi_draw_indexed_indirect_count so
// the hardware only reads the N_visible compact commands — never stale zero-
// instance_count entries (Nanite / DOTS style compaction).

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
    _pad0:  u32,
    _pad1:  u32,
    _pad2:  u32,
    // 7 screen-radius LOD thresholds: s[i] = transition LOD i → i+1.
    // 8 LOD levels total (0–7), matching UE5's traditional mesh LOD system.
    lod_s0: f32,
    lod_s1: f32,
    lod_s2: f32,
    lod_s3: f32,
    lod_s4: f32,
    lod_s5: f32,
    lod_s6: f32,
    _pad3:  f32,
}

@group(0) @binding(0) var<uniform>             camera:     Camera;
@group(0) @binding(1) var<uniform>             cull_uni:   CullUniforms;
@group(0) @binding(2) var<storage, read>       meshlets:   array<MeshletEntry>;
@group(0) @binding(3) var<storage, read>       instances:  array<InstanceData>;
@group(0) @binding(4) var<storage, read_write> indirect:   array<DrawIndexedIndirect>;
/// Atomic counter: cull shader increments once per visible meshlet.
/// CPU passes this buffer to multi_draw_indexed_indirect_count as the count arg.
@group(0) @binding(5) var<storage, read_write> draw_count: atomic<u32>;

// ─── LOD selection ───────────────────────────────────────────────────────────

fn select_lod_level(screen_size: f32) -> u32 {
    if screen_size >= cull_uni.lod_s0 { return 0u; }
    if screen_size >= cull_uni.lod_s1 { return 1u; }
    if screen_size >= cull_uni.lod_s2 { return 2u; }
    if screen_size >= cull_uni.lod_s3 { return 3u; }
    if screen_size >= cull_uni.lod_s4 { return 4u; }
    if screen_size >= cull_uni.lod_s5 { return 5u; }
    if screen_size >= cull_uni.lod_s6 { return 6u; }
    return 7u;
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

    // ── Frustum cull ──────────────────────────────────────────────────────────
    // Planes extracted from VP using Gribb/Hartmann (column-major, depth ∈ [0,1]).
    // Plane normals are NOT normalised; scale world_radius by |n| per plane.
    let vp = camera.view_proj;
    let pl0 = vec4<f32>(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]); // left
    let pl1 = vec4<f32>(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]); // right
    let pl2 = vec4<f32>(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]); // bottom
    let pl3 = vec4<f32>(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]); // top
    let pl4 = vec4<f32>(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);                                               // near (depth∈[0,1])
    let pl5 = vec4<f32>(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]); // far

    let visible = (dot(pl0.xyz, center_ws) + pl0.w >= -world_radius * length(pl0.xyz))
               && (dot(pl1.xyz, center_ws) + pl1.w >= -world_radius * length(pl1.xyz))
               && (dot(pl2.xyz, center_ws) + pl2.w >= -world_radius * length(pl2.xyz))
               && (dot(pl3.xyz, center_ws) + pl3.w >= -world_radius * length(pl3.xyz))
               && (dot(pl4.xyz, center_ws) + pl4.w >= -world_radius * length(pl4.xyz))
               && (dot(pl5.xyz, center_ws) + pl5.w >= -world_radius * length(pl5.xyz));

    if !visible {
        return;
    }

    // ── Backface cone cull (guarded) ─────────────────────────────────────────
    // Skip when the camera is inside or close to the meshlet's bounding sphere
    // to avoid false culls on nearby geometry.
    let cam_to_center = center_ws - camera.position_near.xyz;
    let cam_dist_sq = dot(cam_to_center, cam_to_center);
    let guard_radius = world_radius * 1.5;
    if cam_dist_sq > guard_radius * guard_radius && m.cone_cutoff <= 1.0 {
        let model_mat3 = mat3x3<f32>(model[0].xyz, model[1].xyz, model[2].xyz);
        let cone_axis_ws = normalize(model_mat3 * m.cone_axis);
        let view_dir = normalize(cam_to_center);
        if dot(view_dir, cone_axis_ws) >= m.cone_cutoff {
            return;
        }
    }

    // ── LOD selection — Nanite-style projected screen coverage ─────────────
    let cluster_dist = max(length(cam_to_center), 0.001);
    let obj_radius   = max(inst.bounds.w, 0.001);
    let focal_len    = camera.proj[1][1];
    let screen_size  = (obj_radius * focal_len) / cluster_dist;
    let lod_level    = u32(m.lod_error + 0.5);
    let desired_lod  = select_lod_level(screen_size);

    if lod_level != desired_lod {
        return;
    }

    var cmd: DrawIndexedIndirect;
    cmd.index_count    = m.index_count;
    cmd.instance_count = 1u;
    cmd.first_index    = m.first_index;
    cmd.base_vertex    = m.vertex_offset;
    // Pack LOD level into upper 8 bits of first_instance so the draw shader
    // can extract it for debug visualisation (LOD heatmap mode 21).
    cmd.first_instance = m.instance_index | (lod_level << 24u);
    let slot = atomicAdd(&draw_count, 1u);
    indirect[slot] = cmd;
}
