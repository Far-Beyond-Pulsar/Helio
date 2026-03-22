// Virtual geometry culling compute shader.
//
// One thread per meshlet. Tests each meshlet against the view frustum.
// Visible meshlets are atomically appended to a compact indirect draw list:
//
//   slot = atomicAdd(&draw_count, 1u);
//   indirect[slot] = cmd;
//
// The GPU-written draw_count is passed to multi_draw_indexed_indirect_count so
// the hardware only reads the N_visible compact commands — never stale zero-
// instance_count entries (Nanite / DOTS style AAA compaction).

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
    lod_d0: f32,   // LOD 0→1 transition in units of object-radii (dist / bounds.w)
    lod_d1: f32,   // LOD 1→2 transition in units of object-radii
    _pad2:  u32,
}

@group(0) @binding(0) var<uniform>             camera:     Camera;
@group(0) @binding(1) var<uniform>             cull_uni:   CullUniforms;
@group(0) @binding(2) var<storage, read>       meshlets:   array<MeshletEntry>;
@group(0) @binding(3) var<storage, read>       instances:  array<InstanceData>;
@group(0) @binding(4) var<storage, read_write> indirect:   array<DrawIndexedIndirect>;
/// Atomic counter: cull shader increments once per visible meshlet.
/// CPU passes this buffer to multi_draw_indexed_indirect_count as the count arg.
@group(0) @binding(5) var<storage, read_write> draw_count: atomic<u32>;

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

    // Correct sphere-vs-plane test: scale the radius by the plane normal magnitude.
    let visible = (dot(pl0.xyz, center_ws) + pl0.w >= -world_radius * length(pl0.xyz))
               && (dot(pl1.xyz, center_ws) + pl1.w >= -world_radius * length(pl1.xyz))
               && (dot(pl2.xyz, center_ws) + pl2.w >= -world_radius * length(pl2.xyz))
               && (dot(pl3.xyz, center_ws) + pl3.w >= -world_radius * length(pl3.xyz))
               && (dot(pl4.xyz, center_ws) + pl4.w >= -world_radius * length(pl4.xyz))
               && (dot(pl5.xyz, center_ws) + pl5.w >= -world_radius * length(pl5.xyz));

    // NOTE: No backface cone cull here. Per-triangle backface culling is already
    // performed by the GPU rasterizer (cull_mode = Back). The meshlet cone test
    // is a pure optimisation and is disabled because the apex-centroid approximation
    // produces false culls when the camera is close to the surface.

    if visible {
        // ── LOD selection (per-object, scale-invariant) ───────────────────────
        // Distance is normalised by the object's world-space bounding radius so
        // that large rocks maintain full detail at proportionally larger distances
        // and small rocks downgrade sooner.  All meshlets of the same object share
        // the same bounding sphere, so the LOD level is identical across all of
        // them — no mixed-LOD seams or holes.
        //
        // lod_error encodes the LOD level: 0.0 = full, 1.0 = medium, 2.0 = coarse.
        // lod_d0 / lod_d1 are in units of (object radii):
        //   dist_radii < lod_d0   → LOD 0  (full detail)
        //   lod_d0 ≤ dist_radii < lod_d1 → LOD 1  (medium)
        //   dist_radii ≥ lod_d1   → LOD 2  (coarse)
        let obj_offset = inst.bounds.xyz - camera.position_near.xyz;
        let obj_dist_sq = dot(obj_offset, obj_offset);
        let obj_radius  = max(inst.bounds.w, 0.5);  // guard against degenerate bounds
        // Use squared comparison to avoid a sqrt on the hot path.
        let d0_sq = cull_uni.lod_d0 * cull_uni.lod_d0 * obj_radius * obj_radius;
        let d1_sq = cull_uni.lod_d1 * cull_uni.lod_d1 * obj_radius * obj_radius;
        let lod_level = u32(m.lod_error + 0.5);
        let lod_ok    = (lod_level == 0u && obj_dist_sq <  d0_sq)
                     || (lod_level == 1u && obj_dist_sq >= d0_sq && obj_dist_sq < d1_sq)
                     || (lod_level == 2u && obj_dist_sq >= d1_sq);
        if !lod_ok {
            return;
        }

        var cmd: DrawIndexedIndirect;
        cmd.index_count    = m.index_count;
        cmd.instance_count = 1u;
        cmd.first_index    = m.first_index;
        cmd.base_vertex    = m.vertex_offset;
        cmd.first_instance = m.instance_index;
        // Atomically claim a compact slot and write directly — no zero-instance
        // padding entries, no wasted GPU reads.
        let slot = atomicAdd(&draw_count, 1u);
        indirect[slot] = cmd;
    }
}
