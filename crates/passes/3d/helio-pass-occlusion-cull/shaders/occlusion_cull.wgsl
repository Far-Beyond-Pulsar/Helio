//! Hi-Z occlusion culling — fully GPU-driven, O(1) CPU.
//!
//! Each thread evaluates one DRAW CALL slot by testing the bounding sphere
//! of that draw call's first (representative) instance against the Hi-Z pyramid.
//! Occluded draws get instance_count=0 in the indirect buffer.
//!
//! IMPORTANT: this pass runs AFTER IndirectDispatchPass (frustum cull). It does
//! NOT re-do frustum culling — only tests occlusion. The indirect buffer is
//! shared: frustum cull writes initial instance_count, then we may zero it.
//!
//! Uses TEMPORAL Hi-Z: the pyramid was built from the PREVIOUS frame's depth,
//! so the OcclusionCullPass runs BEFORE DepthPrepass each frame.
//! Frame 0 is skipped entirely (no pyramid yet).

// ──────────────────────────────────────────────────────────────────────────────
// Bind group 0
// ──────────────────────────────────────────────────────────────────────────────

struct Camera {
    view:          mat4x4<f32>,   // bytes  0 – 63
    proj:          mat4x4<f32>,   // bytes 64 – 127
    view_proj:     mat4x4<f32>,   // bytes 128 – 191
    inv_view_proj: mat4x4<f32>,   // bytes 192 – 255
    position_near: vec4<f32>,     // bytes 256 – 271
    direction_far: vec4<f32>,     // bytes 272 – 287
}
@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;

struct CullParams {
    screen_width:       u32,
    screen_height:      u32,
    draw_count:         u32,
    hiz_mip_count:      u32,
    // Baked PVS grid (helio-bake, published as `baked_pvs`); 0 = none.
    pvs_available:      u32,
    pvs_grid_x:         u32,
    pvs_grid_y:         u32,
    pvs_grid_z:         u32,
    pvs_min_x:          f32,
    pvs_min_y:          f32,
    pvs_min_z:          f32,
    pvs_cell_size:      f32,
    /// u32 words per source cell (the baker's u64 words × 2).
    pvs_words_per_cell: u32,
    _pad0:              u32,
    _pad1:              u32,
    _pad2:              u32,
}
@group(0) @binding(1) var<uniform> params: CullParams;

// GpuInstanceData: 208 bytes, must match libhelio/src/instance.rs exactly.
struct GpuInstanceData {
    model_col0:      vec4<f32>,  //   0 – 15
    model_col1:      vec4<f32>,  //  16 – 31
    model_col2:      vec4<f32>,  //  32 – 47
    model_col3:      vec4<f32>,  //  48 – 63
    normal_col0:     vec4<f32>,  //  64 – 79   (w = padding)
    normal_col1:     vec4<f32>,  //  80 – 95
    normal_col2:     vec4<f32>,  //  96 – 111
    bounds:          vec4<f32>,  // 112 – 127  (xyz = world-space sphere center, w = radius)
    prev_model_col0: vec4<f32>,  // 128 – 143
    prev_model_col1: vec4<f32>,  // 144 – 159
    prev_model_col2: vec4<f32>,  // 160 – 175
    prev_model_col3: vec4<f32>,  // 176 – 191
    mesh_id:         u32,        // 192
    material_id:     u32,        // 196
    flags:           u32,        // 200
    _pad:            u32,        // 204
}
@group(0) @binding(2) var<storage, read> instances: array<GpuInstanceData>;

/// GpuDrawCall: 20 bytes, matches DrawCall in indirect_dispatch.wgsl.
struct GpuDrawCall {
    index_count:    u32,
    first_index:    u32,
    vertex_offset:  i32,
    first_instance: u32,  // base index into instances[] for this batch
    instance_count: u32,  // number of consecutive instances in this draw
}
@group(0) @binding(3) var<storage, read> draw_calls: array<GpuDrawCall>;

@group(0) @binding(4) var hiz_tex:  texture_2d<f32>;
@group(0) @binding(5) var hiz_samp: sampler;

// Baked potentially-visible-set bitfield, the baker's u64 words as u32 pairs:
// bit `to` of source cell `from` is word `from * pvs_words_per_cell + to / 32`,
// bit `to % 32`. Set means "cell `to` may be visible from cell `from`".
@group(0) @binding(7) var<storage, read> pvs_bits: array<u32>;

// Indirect draw buffer as raw u32 array.
// DrawIndexedIndirect stride = 20 bytes = 5 × u32:
//   [i*5 + 0] index_count
//   [i*5 + 1] instance_count  ← we write 0 (occluded) or keep original value
//   [i*5 + 2] first_index
//   [i*5 + 3] base_vertex     (i32 reinterpreted as u32 for array access)
//   [i*5 + 4] first_instance
@group(0) @binding(6) var<storage, read_write> indirect: array<u32>;
@group(0) @binding(9) var<storage, read_write> stats:   array<atomic<u32>>;

// Per-group compacted original instance slots surviving frustum culling
// (written by IndirectDispatchPass) and, after this pass, surviving BOTH
// frustum AND Hi-Z occlusion. Draw-consuming passes must read
// `compacted_indices_2`, not `compacted_indices` (frustum-only intermediate).
@group(0) @binding(10) var<storage, read>       compacted_indices:   array<u32>;
@group(0) @binding(11) var<storage, read_write> compacted_indices_2: array<u32>;

// Coordinate-space transforms (current frame). Slot 0 = identity. An
// instance's `bounds` center is authored pre-space (same frame as `model`),
// so it must be mapped through this before any occlusion test — mirrors the
// frustum-stage handling in indirect_dispatch.wgsl.
@group(0) @binding(12) var<storage, read> coordinate_spaces: array<mat4x4<f32>>;

// One workgroup handles one draw-call group, cooperatively Hi-Z-testing only
// the instances that already survived frustum culling and re-compacting the
// survivors — mirrors IndirectDispatchPass's per-instance compaction so a
// group's final `instance_count` reflects real per-instance occlusion instead
// of an all-or-nothing "is the whole batch occluded" test.
var<workgroup> wg_counter: atomic<u32>;

// Stats layout (shared with IndirectDispatchPass):
// 4: occlusion_culled  (we only write to slot 4)
// 7: shadow_occlusion_culled

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Project NDC xy to texture UV.
/// wgpu NDC: x∈[-1,+1] left→right, y∈[-1,+1] bottom→top.
/// UV:       u∈[0,1]   left→right, v∈[0,1]   top→bottom.
fn ndc_to_uv(ndc_xy: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        ndc_xy.x *  0.5 + 0.5,
        ndc_xy.y * -0.5 + 0.5,
    );
}

/// Estimate screen-space radius (in pixels) of a sphere.
/// proj[1][1] = cot(fovY/2) = 2n/h for a standard perspective matrix.
fn screen_radius_px(world_radius: f32, clip_w: f32) -> f32 {
    let half_h = f32(params.screen_height) * 0.5;
    return abs(world_radius / clip_w * cameras[0].proj[1][1] * half_h);
}

/// Select HiZ mip level for a sphere footprint of `r_px` pixels.
fn pick_mip(r_px: f32) -> u32 {
    let diameter = max(r_px * 2.0, 1.0);
    let mip = u32(ceil(log2(diameter)));
    return clamp(mip, 0u, params.hiz_mip_count - 1u);
}

/// Conservative sphere near depth in NDC [0,1].
/// Projects the point on the sphere nearest to the camera into NDC depth.
fn sphere_near_depth(center: vec3<f32>, radius: f32) -> f32 {
    let cam_pos = cameras[0].position_near.xyz;
    let to_center = center - cam_pos;
    let dist_sq = dot(to_center, to_center);
    if dist_sq <= radius * radius {
        // Camera inside sphere — near depth is 0 (on the near plane)
        return 0.0;
    }
    let dir = to_center * (1.0 / sqrt(dist_sq));
    let near_ws = center - dir * radius;
    let near_clip = cameras[0].view_proj * vec4<f32>(near_ws, 1.0);
    // Protect against near_clip.w <= 0 (shouldn't happen since camera is outside)
    if near_clip.w <= 0.0 {
        return 0.0;
    }
    return clamp(near_clip.z / near_clip.w, 0.0, 1.0);
}

// ──────────────────────────────────────────────────────────────────────────────
// Main kernel  (64 threads × 1 × 1 workgroup)
// ──────────────────────────────────────────────────────────────────────────────

/// Test a single instance against the Hi-Z pyramid. `center` is the bounds
/// center already mapped through the instance's coordinate space (see caller).
fn instance_hiz_occluded(inst: GpuInstanceData, center: vec3<f32>) -> bool {
    let radius = inst.bounds.w;
    if radius <= 0.0 {
        return false;
    }

    let clip = cameras[0].view_proj * vec4<f32>(center, 1.0);
    if clip.w <= 0.0 {
        return false;
    }

    let ndc_r = max(
        abs(radius * cameras[0].proj[0][0] / clip.w),
        abs(radius * cameras[0].proj[1][1] / clip.w),
    );
    let ndc = clip.xyz / clip.w;
    let uv = ndc_to_uv(ndc.xy);

    if ndc.x + ndc_r < -1.0 || ndc.x - ndc_r > 1.0 ||
       ndc.y + ndc_r < -1.0 || ndc.y - ndc_r > 1.0 {
        return false;
    }

    let near_z = sphere_near_depth(center, radius);
    let r_px = screen_radius_px(radius, clip.w);
    let mip = pick_mip(r_px);

    let uv_half = ndc_r * 0.5;
    let uv_00 = clamp(uv - vec2<f32>(uv_half, uv_half), vec2<f32>(0.0), vec2<f32>(1.0));
    let uv_11 = clamp(uv + vec2<f32>(uv_half, uv_half), vec2<f32>(0.0), vec2<f32>(1.0));

    let hiz_00 = textureSampleLevel(hiz_tex, hiz_samp, uv_00, f32(mip)).r;
    let hiz_01 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_11.x, uv_00.y), f32(mip)).r;
    let hiz_10 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_00.x, uv_11.y), f32(mip)).r;
    let hiz_11 = textureSampleLevel(hiz_tex, hiz_samp, uv_11, f32(mip)).r;
    let hiz_depth = max(max(hiz_00, hiz_01), max(hiz_10, hiz_11));

    let depth_bias = 1.0 / 65536.0;
    return near_z > hiz_depth + depth_bias;
}

/// Largest cell block an instance's bounds may cover before the PVS test is
/// skipped (the instance is then kept, never culled on a guess).
const PVS_MAX_CELLS: i32 = 64;

/// Cell coordinates of `p` in the PVS grid (may be outside it).
fn pvs_cell_coords(p: vec3<f32>) -> vec3<i32> {
    let origin = vec3<f32>(params.pvs_min_x, params.pvs_min_y, params.pvs_min_z);
    return vec3<i32>(floor((p - origin) / params.pvs_cell_size));
}

fn pvs_grid() -> vec3<i32> {
    return vec3<i32>(i32(params.pvs_grid_x), i32(params.pvs_grid_y), i32(params.pvs_grid_z));
}

fn pvs_inside(c: vec3<i32>) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < pvs_grid());
}

fn pvs_cell_index(c: vec3<i32>) -> u32 {
    let g = pvs_grid();
    return u32(c.x + c.y * g.x + c.z * g.x * g.y);
}

/// Whether cell `dst_cell` may be visible from cell `src_cell`. Out-of-range words
/// count as visible, matching `BakedPvsRef::is_visible`.
fn pvs_visible(src_cell: u32, dst_cell: u32) -> bool {
    let word = src_cell * params.pvs_words_per_cell + dst_cell / 32u;
    if word >= arrayLength(&pvs_bits) {
        return true;
    }
    return ((pvs_bits[word] >> (dst_cell % 32u)) & 1u) != 0u;
}

/// Test a single instance against the baked PVS (Helio#256). `center` is the
/// bounds center already mapped through the instance's coordinate space.
///
/// Occluded only when no cell the bounding sphere touches is potentially
/// visible from the camera's cell. Anything the grid cannot answer exactly
/// (camera or bounds outside it, bounds spanning too many cells) is visible.
fn instance_pvs_occluded(inst: GpuInstanceData, center: vec3<f32>, cam_pos: vec3<f32>) -> bool {
    let cam_cell = pvs_cell_coords(cam_pos);
    if !pvs_inside(cam_cell) {
        return false;
    }
    let src_cell = pvs_cell_index(cam_cell);
    let radius = max(inst.bounds.w, 0.0);
    let lo = pvs_cell_coords(center - vec3<f32>(radius));
    let hi = pvs_cell_coords(center + vec3<f32>(radius));
    if !pvs_inside(lo) || !pvs_inside(hi) {
        return false;
    }
    let span = hi - lo + vec3<i32>(1);
    if span.x * span.y * span.z > PVS_MAX_CELLS {
        return false;
    }
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                if pvs_visible(src_cell, pvs_cell_index(vec3<i32>(x, y, z))) {
                    return false;
                }
            }
        }
    }
    return true;
}

/// Returns true when an instance is occluded by the Hi-Z pyramid or, when a
/// bake provided one, the static PVS.
/// Mirrors `libhelio::INSTANCE_FLAG_ALWAYS_VISIBLE`.
const INSTANCE_FLAG_ALWAYS_VISIBLE: u32 = 4u;

fn instance_is_occluded(inst: GpuInstanceData, center: vec3<f32>, cam_pos: vec3<f32>) -> bool {
    // Per-object cull opt-out — must be honoured here as well as in the frustum stage,
    // or an object marked always-visible still vanishes behind the Hi-Z test.
    if (inst.flags & INSTANCE_FLAG_ALWAYS_VISIBLE) != 0u {
        return false;
    }
    if instance_hiz_occluded(inst, center) {
        return true;
    }
    if params.pvs_available != 0u {
        if instance_pvs_occluded(inst, center, cam_pos) {
            return true;
        }
    }
    return false;
}

@compute @workgroup_size(64, 1, 1)
fn main(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = wg_id.x;
    let active_draw = idx < params.draw_count;
    var visible_count = 0u;

    // Cooperatively Hi-Z-test only the instances that already survived
    // frustum culling (`visible_count` of them, packed in `compacted_indices`
    // starting at `dc.first_instance`), compacting survivors into
    // `compacted_indices_2` via a workgroup-shared atomic counter.
    //
    // Every invocation must reach the barrier below. Although `idx` and the
    // indirect count are uniform for a workgroup, FXC cannot prove uniformity
    // through storage-buffer reads and rejects a barrier reached after the old
    // early returns. Keep all draw-dependent work inside the branch instead.
    if active_draw {
        visible_count = indirect[idx * 5u + 1u];
        let dc = draw_calls[idx];
        let cam_pos = cameras[0].position_near.xyz;
        for (var i = lid.x; i < visible_count; i += 64u) {
            let original_idx = compacted_indices[dc.first_instance + i];
            let inst = instances[original_idx];
            // Common case (space 0, identity) stays exactly as before; only a
            // sublevel/portal-tagged instance pays the extra matrix multiply.
            let space_id = (inst.flags >> 8u) & 0xFFu;
            let center = select(
                inst.bounds.xyz,
                (coordinate_spaces[space_id] * vec4<f32>(inst.bounds.xyz, 1.0)).xyz,
                space_id != 0u,
            );
            if !instance_is_occluded(inst, center, cam_pos) {
                let slot = atomicAdd(&wg_counter, 1u);
                compacted_indices_2[dc.first_instance + slot] = original_idx;
            }
        }
    }

    workgroupBarrier();

    if lid.x == 0u && active_draw && visible_count > 0u {
        let final_count = atomicLoad(&wg_counter);
        indirect[idx * 5u + 1u] = final_count;
        if final_count == 0u {
            atomicAdd(&stats[4u], 1u);
        }
    }
}
