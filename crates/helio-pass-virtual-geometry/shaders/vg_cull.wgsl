// Virtual geometry culling compute shader.
//
// One thread per meshlet. Tests each meshlet against the view frustum,
// backface cone, and Hi-Z occlusion buffer.
// Visible meshlets are atomically appended to a compact indirect draw list:
//
//   slot = atomicAdd(&draw_count, 1u);
//   indirect[slot] = cmd;
//
// WebGPU has no indirect-count draw. The CPU records a bounded indirect slot per
// meshlet; culled slots have instance_count=0, so no readback is required.

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
    screen_width:  u32,
    screen_height: u32,
    hiz_mip_count: u32,
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
/// Diagnostic count of emitted commands; rendering does not read it back.
@group(0) @binding(5) var<storage, read_write> draw_count: atomic<u32>;

@group(0) @binding(6) var hiz_tex:  texture_2d<f32>;
@group(0) @binding(7) var hiz_samp: sampler;
/// One entry per instance: max(|col0|, |col1|, |col2|) of its transform.
/// Precomputed CPU-side once per instance (see instance_scale_buf in lib.rs) —
/// identical for every meshlet of a given instance, so there's no reason for
/// every meshlet thread to redo the 3 sqrt()s that produced it.
@group(0) @binding(8) var<storage, read> instance_scale: array<f32>;

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

/// Pre-normalised frustum planes (left, right, bottom, top, near, far), shared
/// across the whole workgroup. Planes depend only on `camera.view_proj` — the
/// same for every thread in the entire dispatch, not just within one workgroup
/// — so computing them (and the length() normalisation) separately per meshlet
/// thread bought nothing. One thread per workgroup computes them into workgroup
/// memory instead of all 64 redoing the same math.
var<workgroup> wg_planes: array<vec4<f32>, 6>;

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lid: u32) {
    // Must run before any thread can `return` below, so the barrier stays in
    // uniform control flow (reached by every invocation in the workgroup).
    if lid == 0u {
        let vp = camera.view_proj;
        // Gribb/Hartmann (column-major, depth ∈ [0,1]); normalised so the
        // per-meshlet test is a plain dot product (no per-thread length()).
        let p0 = vec4<f32>(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]); // left
        let p1 = vec4<f32>(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]); // right
        let p2 = vec4<f32>(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]); // bottom
        let p3 = vec4<f32>(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]); // top
        let p4 = vec4<f32>(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);                                               // near
        let p5 = vec4<f32>(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]); // far
        wg_planes[0] = p0 / length(p0.xyz);
        wg_planes[1] = p1 / length(p1.xyz);
        wg_planes[2] = p2 / length(p2.xyz);
        wg_planes[3] = p3 / length(p3.xyz);
        wg_planes[4] = p4 / length(p4.xyz);
        wg_planes[5] = p5 / length(p5.xyz);
    }
    workgroupBarrier();

    let idx = gid.x;
    if idx >= cull_uni.meshlet_count {
        return;
    }

    let m    = meshlets[idx];
    let inst = instances[m.instance_index];

    let model     = inst.transform;
    let center_ws = (model * vec4<f32>(m.center, 1.0)).xyz;

    let world_radius = m.radius * instance_scale[m.instance_index];

    // ── LOD selection — Nanite-style projected screen coverage ──────────────
    // Done first and cheaply (distance + radius only) so the ~7/8 of meshlets
    // whose baked LOD isn't the one currently wanted bail out here, instead of
    // paying for frustum planes, cone culling, and a Hi-Z texture sample only
    // to be discarded at the end.
    let cam_to_center = center_ws - camera.position_near.xyz;
    let cluster_dist  = max(length(cam_to_center), 0.001);
    let obj_radius    = max(world_radius, 0.001);
    let focal_len     = camera.proj[1][1];
    let screen_size   = (obj_radius * focal_len) / cluster_dist;
    let lod_level     = u32(m.lod_error + 0.5);
    let desired_lod   = select_lod_level(screen_size);

    if lod_level != desired_lod {
        return;
    }

    // ── Frustum cull ──────────────────────────────────────────────────────────
    // Planes were pre-normalised once per workgroup above (see wg_planes) —
    // just dot-product against them here, no per-thread length() needed.
    let pl0 = wg_planes[0];
    let pl1 = wg_planes[1];
    let pl2 = wg_planes[2];
    let pl3 = wg_planes[3];
    let pl4 = wg_planes[4];
    let pl5 = wg_planes[5];

    let visible = (dot(pl0.xyz, center_ws) + pl0.w >= -world_radius)
               && (dot(pl1.xyz, center_ws) + pl1.w >= -world_radius)
               && (dot(pl2.xyz, center_ws) + pl2.w >= -world_radius)
               && (dot(pl3.xyz, center_ws) + pl3.w >= -world_radius)
               && (dot(pl4.xyz, center_ws) + pl4.w >= -world_radius)
               && (dot(pl5.xyz, center_ws) + pl5.w >= -world_radius);

    if !visible {
        return;
    }

    // ── Backface cone cull (guarded) ─────────────────────────────────────────
    // Skip when the camera is inside or close to the meshlet's bounding sphere
    // to avoid false culls on nearby geometry.
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

    // ── Hi-Z occlusion cull ──────────────────────────────────────────────────
    let cull_clip = camera.view_proj * vec4<f32>(center_ws, 1.0);
    if cull_clip.w > 0.0 {
        let cull_ndc = cull_clip.xyz / cull_clip.w;
        let cull_uv_x = cull_ndc.x * 0.5 + 0.5;
        let cull_uv_y = cull_ndc.y * -0.5 + 0.5;
        let cull_uv = vec2<f32>(cull_uv_x, cull_uv_y);

        let ndc_r = max(
            abs(world_radius * camera.proj[0][0] / cull_clip.w),
            abs(world_radius * camera.proj[1][1] / cull_clip.w),
        );

        if cull_uv.x + ndc_r > -1.0 && cull_uv.x - ndc_r < 1.0 &&
           cull_uv.y + ndc_r > -1.0 && cull_uv.y - ndc_r < 1.0 {
            let cam_pos = camera.position_near.xyz;
            let to_meshlet = center_ws - cam_pos;
            let dist_sq = dot(to_meshlet, to_meshlet);
            var near_z = 0.0;
            if dist_sq > world_radius * world_radius {
                let dir = to_meshlet * (1.0 / sqrt(dist_sq));
                let near_ws = center_ws - dir * world_radius;
                let near_clip = camera.view_proj * vec4<f32>(near_ws, 1.0);
                near_z = clamp(near_clip.z / near_clip.w, 0.0, 1.0);
            }

            let half_h = f32(cull_uni.screen_height) * 0.5;
            let r_px = abs(world_radius / cull_clip.w * camera.proj[1][1] * half_h);
            let diameter = max(r_px * 2.0, 1.0);
            let mip = clamp(u32(ceil(log2(diameter))), 0u, cull_uni.hiz_mip_count - 1u);

            let sample_uv = clamp(cull_uv, vec2<f32>(0.0), vec2<f32>(1.0));
            let hiz_depth = textureSampleLevel(hiz_tex, hiz_samp, sample_uv, f32(mip)).r;

            let depth_bias = 1.0 / 65536.0;
            if near_z > hiz_depth + depth_bias {
                return;
            }
        }
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
