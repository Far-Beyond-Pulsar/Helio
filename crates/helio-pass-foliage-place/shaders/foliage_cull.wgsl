// Foliage tile cull, cluster cull + compaction, and indirect finalize — stages 2 and 3
// of `FoliagePlacePass`.
//
// Three entry points, dispatched in order on the *main render encoder* with an implicit
// barrier between them:
//
//   cs_tile_cull     one lane per ring slot   -> tile_visible[slot]
//   cs_cluster_cull  one lane per cluster     -> visible_blades[lod region] + counters
//   cs_finalize      four lanes               -> foliage_indirect[0..4]
//
// ── Why the main encoder and not `chain_transparent` ────────────────────────────
//
// `chain_transparent` passes must record exclusively on the separate compute encoder,
// and the two encoders are submitted as `[compute_encoder, encoder]`, so *all* compute-
// encoder work runs before *all* render-encoder work. A `chain_transparent` foliage cull
// would therefore Hi-Z-test against the **previous** frame's pyramid. `HiZBuildPass` and
// `OcclusionCullPass` both use `ctx.encoder_ptr` for exactly this reason and neither opts
// in. See the plan's §6.2 (marked [audit] — the first draft had this wrong).

const WG_SIZE: u32 = 64u;

const FOLIAGE_LOD_COUNT: u32 = 4u;
const FOLIAGE_LOD_NONE: u32 = 4u;

/// The clump-card LOD: one instance per 4x4 cluster rather than per blade. Must agree
/// with `LOD_WIDTH_SCALE` / `LOD_HEIGHT_SCALE` in `helio-pass-foliage-gbuffer`, which
/// size that card to cover sixteen blades.
const FOLIAGE_LOD_CLUMP: u32 = 3u;

// Mirrors helio_foliage_core::TileState. `Placing` is excluded on purpose: its slab
// contents are undefined until the placement dispatch's final barrier, so drawing it
// renders whatever the previous tenant left behind.
const TILE_STATE_RESIDENT: u32 = 2u;
const TILE_STATE_EVICTING: u32 = 3u;

// Mirrors the COUNTER_* constants in src/lib.rs.
const COUNTER_VISIBLE_OVERFLOW: u32 = 4u;

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

/// Mirrors `FoliageCullUniforms` (Rust, 64 bytes).
struct FoliageCullUniforms {
    tile_count:             u32,
    screen_width:           u32,
    screen_height:          u32,
    hiz_mip_count:          u32,

    // 0 on frame 0 and right after a resize rebuilds the graph: the Hi-Z pyramid has not
    // been built from any real depth yet, so its texture reads back as 0.0 — which this
    // engine's near-is-0.0 convention would otherwise read as "everything is behind the
    // (nonexistent) occluder", culling the entire world for that frame. Copied verbatim
    // from `CullUniforms::hiz_valid` in vg_cull.wgsl, and it is load-bearing.
    hiz_valid:              u32,
    cluster_size:           u32,
    clusters_per_tile:      u32,
    per_lod_capacity:       u32,

    tile_size:              f32,
    lod_quality_scale:      f32,
    type_count:             u32,
    cluster_dispatch_width: u32,

    max_foliage_height:     f32,
    wpo_extent:             f32,
    /// Width of the LOD cross-fade overlap band, in metres. Must match the consumer's
    /// `lod_fade_band`: this pass decides which clusters get emitted into *two* LODs,
    /// and the consumer decides their blend weights. A mismatch leaves instances that
    /// blend to less (gap) or more (double-density ring) than one blade's coverage.
    lod_fade_band:          f32,
    _pad0:                  u32,
}

/// Mirrors `helio_foliage_core::GpuFoliageType` (Rust, 96 bytes). All scalars — see the
/// note in foliage_place.wgsl.
struct FoliageType {
    density:               f32,
    height_min:            f32,
    height_max:            f32,
    width_min:             f32,
    width_max:             f32,
    slope_min:             f32,
    slope_max:             f32,
    altitude_min:          f32,
    altitude_max:          f32,
    lod0:                  f32,
    lod1:                  f32,
    lod2:                  f32,
    lod3:                  f32,
    wind_trunk:            f32,
    wind_branch:           f32,
    wind_leaf:             f32,
    interaction_stiffness: f32,
    material_id:           u32,
    density_layer:         u32,
    kind_and_flags:        u32,
    mesh_or_impostor_id:   u32,
    _pad0:                 u32,
    _pad1:                 u32,
    _pad2:                 u32,
}

/// Mirrors `helio_foliage_core::GpuFoliageTile` (Rust, 32 bytes).
struct FoliageTile {
    tile_coord_x:    i32,
    tile_coord_z:    i32,
    blade_offset:    u32,
    blade_count:     u32,
    bounds_center_y: f32,
    bounds_half_y:   f32,
    state:           u32,
    generation:      u32,
}

/// Mirrors `helio_foliage_core::GpuBladeInstance` (Rust, 16 bytes).
struct BladeInstance {
    packed_pos:        u32,
    packed_height_yaw: u32,
    packed_scale_type: u32,
    packed_tint_seed:  u32,
}

/// Mirrors `wgpu::util::DrawIndirectArgs` (16 bytes).
struct DrawIndirect {
    vertex_count:   u32,
    instance_count: u32,
    first_vertex:   u32,
    first_instance: u32,
}

@group(0) @binding(0)  var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1)  var<uniform> cull: FoliageCullUniforms;
@group(0) @binding(2)  var<storage, read> tiles: array<FoliageTile>;
@group(0) @binding(3)  var<storage, read> blades: array<BladeInstance>;
@group(0) @binding(4)  var<storage, read> types: array<FoliageType>;
@group(0) @binding(5)  var<storage, read_write> visible_blades: array<u32>;
@group(0) @binding(6)  var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(7)  var<storage, read_write> indirect: array<DrawIndirect>;
@group(0) @binding(8)  var<storage, read_write> tile_visible: array<u32>;
@group(0) @binding(9)  var hiz_tex: texture_2d<f32>;
@group(0) @binding(10) var hiz_samp: sampler;

var<workgroup> wg_planes: array<vec4<f32>, 6>;

// ── Shared helpers ──────────────────────────────────────────────────────────────

/// True for finite values only. WGSL has no `isnan`/`isinf`, and `x == x` is legal for a
/// compiler to fold away under fast-math, so this tests the exponent field directly.
fn is_finite_f32(value: f32) -> bool {
    return (bitcast<u32>(value) & 0x7f800000u) != 0x7f800000u;
}

fn publish_frustum_planes() {
    let vp = cameras[0].view_proj;
    let p0 = vec4<f32>(vp[0][3] + vp[0][0], vp[1][3] + vp[1][0], vp[2][3] + vp[2][0], vp[3][3] + vp[3][0]);
    let p1 = vec4<f32>(vp[0][3] - vp[0][0], vp[1][3] - vp[1][0], vp[2][3] - vp[2][0], vp[3][3] - vp[3][0]);
    let p2 = vec4<f32>(vp[0][3] + vp[0][1], vp[1][3] + vp[1][1], vp[2][3] + vp[2][1], vp[3][3] + vp[3][1]);
    let p3 = vec4<f32>(vp[0][3] - vp[0][1], vp[1][3] - vp[1][1], vp[2][3] - vp[2][1], vp[3][3] - vp[3][1]);
    let p4 = vec4<f32>(vp[0][2], vp[1][2], vp[2][2], vp[3][2]);
    let p5 = vec4<f32>(vp[0][3] - vp[0][2], vp[1][3] - vp[1][2], vp[2][3] - vp[2][2], vp[3][3] - vp[3][2]);
    wg_planes[0] = p0 / length(p0.xyz);
    wg_planes[1] = p1 / length(p1.xyz);
    wg_planes[2] = p2 / length(p2.xyz);
    wg_planes[3] = p3 / length(p3.xyz);
    wg_planes[4] = p4 / length(p4.xyz);
    wg_planes[5] = p5 / length(p5.xyz);
}

fn sphere_visible(center: vec3<f32>, radius: f32) -> bool {
    return (dot(wg_planes[0].xyz, center) + wg_planes[0].w >= -radius)
        && (dot(wg_planes[1].xyz, center) + wg_planes[1].w >= -radius)
        && (dot(wg_planes[2].xyz, center) + wg_planes[2].w >= -radius)
        && (dot(wg_planes[3].xyz, center) + wg_planes[3].w >= -radius)
        && (dot(wg_planes[4].xyz, center) + wg_planes[4].w >= -radius)
        && (dot(wg_planes[5].xyz, center) + wg_planes[5].w >= -radius);
}

/// Conservative max-depth Hi-Z test, transcribed from `vg_cull.wgsl` (the block around
/// lines 199-252) including the `hiz_valid` frame-0 guard.
///
/// Rejects only when the *complete* projected sphere is on screen and all four corners of
/// its footprint are behind existing depth at the chosen mip. Every early `return false`
/// below is "cannot prove occlusion" — the conservative answer — not "visible".
fn hiz_occluded(center_ws: vec3<f32>, world_radius: f32) -> bool {
    if cull.hiz_valid == 0u {
        return false;
    }
    let cull_clip = cameras[0].view_proj * vec4<f32>(center_ws, 1.0);
    if cull_clip.w <= 0.0 {
        return false;
    }
    let cull_ndc = cull_clip.xyz / cull_clip.w;
    let cull_uv = vec2<f32>(cull_ndc.x * 0.5 + 0.5, cull_ndc.y * -0.5 + 0.5);
    let nearest_view_depth = cull_clip.w - world_radius;
    if nearest_view_depth <= cameras[0].position_near.w {
        return false;
    }

    let ndc_r = max(
        abs(world_radius * cameras[0].proj[0][0] / nearest_view_depth),
        abs(world_radius * cameras[0].proj[1][1] / nearest_view_depth),
    );
    let uv_radius = ndc_r * 0.5;
    let uv_min = cull_uv - vec2<f32>(uv_radius);
    let uv_max = cull_uv + vec2<f32>(uv_radius);
    if !(all(uv_min >= vec2<f32>(0.0)) && all(uv_max <= vec2<f32>(1.0))) {
        return false;
    }

    let cam_to_center = center_ws - cameras[0].position_near.xyz;
    let dist_sq = dot(cam_to_center, cam_to_center);
    var near_z = 0.0;
    if dist_sq > world_radius * world_radius {
        let direction = cam_to_center / sqrt(dist_sq);
        let near_ws = center_ws - direction * world_radius;
        let near_clip = cameras[0].view_proj * vec4<f32>(near_ws, 1.0);
        if near_clip.w > 0.0 {
            near_z = clamp(near_clip.z / near_clip.w, 0.0, 1.0);
        }
    }

    let half_height = f32(cull.screen_height) * 0.5;
    let diameter_px = max(ndc_r * half_height * 2.0, 1.0);
    let mip = clamp(u32(ceil(log2(diameter_px))), 0u, max(cull.hiz_mip_count, 1u) - 1u);
    let hiz_00 = textureSampleLevel(hiz_tex, hiz_samp, uv_min, f32(mip)).r;
    let hiz_01 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_max.x, uv_min.y), f32(mip)).r;
    let hiz_10 = textureSampleLevel(hiz_tex, hiz_samp, vec2<f32>(uv_min.x, uv_max.y), f32(mip)).r;
    let hiz_11 = textureSampleLevel(hiz_tex, hiz_samp, uv_max, f32(mip)).r;
    let hiz_depth = max(max(hiz_00, hiz_01), max(hiz_10, hiz_11));
    return near_z > hiz_depth + 1.0 / 65536.0;
}

/// Transcription of `helio_foliage_core::select_blade_lod`.
///
/// Bands are half-open and lower-inclusive: level `n` covers `[threshold[n-1],
/// threshold[n])`, so a blade exactly on a boundary belongs to the *coarser* level. The
/// CPU reference and this function must agree bit-for-bit at the boundary or the two
/// disagree on a hairline of the ring, and `FoliageGBufferPass` reimplements the same
/// function again for its cross-fade — three copies that must not drift.
///
/// Non-finite input drops to `FOLIAGE_LOD_NONE` rather than LOD 0: this runs once per
/// cluster, and a poisoned uniform that promoted the whole ring to the 11-vertex strip
/// would blow the raster budget in a single frame. Dropping to terrain shading is the
/// bounded failure.
/// Upper distance bound of `level`, with the same non-decreasing repair
/// `select_blade_lod` applies. Mirrors `foliage_lod_threshold` in the consumer's shader —
/// the two must agree or a cluster is emitted into a band the consumer does not blend.
fn lod_upper_threshold(ty: FoliageType, level: u32, quality_scale: f32) -> f32 {
    var scale = 1.0;
    if is_finite_f32(quality_scale) && quality_scale > 0.0 {
        scale = quality_scale;
    }
    var ladder = array<f32, 4>(ty.lod0, ty.lod1, ty.lod2, ty.lod3);
    var threshold = 0.0;
    for (var i = 0u; i < FOLIAGE_LOD_COUNT; i = i + 1u) {
        let scaled = ladder[i] * scale;
        if scaled > threshold {
            threshold = scaled;
        }
        if i >= level {
            break;
        }
    }
    return threshold;
}

fn select_blade_lod(
    distance: f32,
    lod0: f32,
    lod1: f32,
    lod2: f32,
    lod3: f32,
    quality_scale: f32,
) -> u32 {
    if !is_finite_f32(distance) {
        return FOLIAGE_LOD_NONE;
    }
    var scale = 1.0;
    if is_finite_f32(quality_scale) && quality_scale > 0.0 {
        scale = quality_scale;
    }
    var clamped_distance = distance;
    if clamped_distance < 0.0 {
        clamped_distance = 0.0;
    }

    var threshold = 0.0;
    var level = 0u;
    loop {
        if level >= FOLIAGE_LOD_COUNT {
            break;
        }
        var raw = lod0;
        if level == 1u {
            raw = lod1;
        } else if level == 2u {
            raw = lod2;
        } else if level == 3u {
            raw = lod3;
        }
        if !is_finite_f32(raw) {
            return FOLIAGE_LOD_NONE;
        }
        let scaled = raw * scale;
        if scaled > threshold {
            threshold = scaled;
        }
        if clamped_distance < threshold {
            return level;
        }
        level = level + 1u;
    }
    return FOLIAGE_LOD_NONE;
}

fn blade_world_position(tile: FoliageTile, blade: BladeInstance) -> vec3<f32> {
    let origin_x = f32(tile.tile_coord_x) * cull.tile_size;
    let origin_z = f32(tile.tile_coord_z) * cull.tile_size;
    let local_u = f32(blade.packed_pos & 0xffffu) * (1.0 / 65535.0);
    let local_v = f32(blade.packed_pos >> 16u) * (1.0 / 65535.0);
    let height_offset = unpack2x16float(blade.packed_height_yaw).x;
    return vec3<f32>(
        origin_x + local_u * cull.tile_size,
        tile.bounds_center_y + height_offset,
        origin_z + local_v * cull.tile_size,
    );
}

// ── Stage 2a: tile cull ─────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn cs_tile_cull(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if lane == 0u {
        publish_frustum_planes();
    }
    workgroupBarrier();

    let slot = workgroup_id.x * WG_SIZE + lane;
    if slot >= cull.tile_count || slot >= arrayLength(&tiles) || slot >= arrayLength(&tile_visible) {
        return;
    }

    let tile = tiles[slot];
    var visible = 0u;
    if (tile.state == TILE_STATE_RESIDENT || tile.state == TILE_STATE_EVICTING)
        && tile.blade_count > 0u
    {
        let origin_x = f32(tile.tile_coord_x) * cull.tile_size;
        let origin_z = f32(tile.tile_coord_z) * cull.tile_size;
        let center = vec3<f32>(
            origin_x + cull.tile_size * 0.5,
            tile.bounds_center_y,
            origin_z + cull.tile_size * 0.5,
        );
        // Dilate by the tallest foliage plus its wind displacement before testing.
        // Under-dilating here is the classic wind-culling bug: blades blow outside the
        // tile bounds and the tile is culled while its geometry is still on screen.
        let half = vec3<f32>(
            cull.tile_size * 0.5,
            tile.bounds_half_y + cull.max_foliage_height + cull.wpo_extent,
            cull.tile_size * 0.5,
        );
        let radius = length(half);
        if sphere_visible(center, radius) && !hiz_occluded(center, radius) {
            visible = 1u;
        }
    }
    tile_visible[slot] = visible;
}

// ── Stage 2b: cluster cull + per-LOD compaction ─────────────────────────────────

@compute @workgroup_size(64)
fn cs_cluster_cull(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if lane == 0u {
        publish_frustum_planes();
    }
    workgroupBarrier();

    let group = workgroup_id.x + workgroup_id.y * max(cull.cluster_dispatch_width, 1u);
    let index = group * WG_SIZE + lane;
    let clusters_per_tile = max(cull.clusters_per_tile, 1u);
    let slot = index / clusters_per_tile;
    let cluster = index - slot * clusters_per_tile;

    if slot >= cull.tile_count || slot >= arrayLength(&tiles) || slot >= arrayLength(&tile_visible) {
        return;
    }
    if tile_visible[slot] == 0u {
        return;
    }

    let tile = tiles[slot];
    let cluster_size = max(cull.cluster_size, 1u);
    let first = cluster * cluster_size;
    if first >= tile.blade_count {
        return;
    }
    let count = min(cluster_size, tile.blade_count - first);

    // Real per-cluster bounds, read from the blades. The alternative — reusing the tile's
    // bounds — makes cluster culling exactly as precise as tile culling, i.e. useless. At
    // the reference tier this is 16 loads per lane over ~98 k lanes, ~25 MiB of reads.
    var bounds_min = vec3<f32>(1.0e30);
    var bounds_max = vec3<f32>(-1.0e30);
    var sampled = 0u;
    for (var i = 0u; i < count; i = i + 1u) {
        let arena_index = tile.blade_offset + first + i;
        if arena_index >= arrayLength(&blades) {
            break;
        }
        let position = blade_world_position(tile, blades[arena_index]);
        bounds_min = min(bounds_min, position);
        bounds_max = max(bounds_max, position);
        sampled = sampled + 1u;
    }
    if sampled == 0u {
        return;
    }

    let center = (bounds_min + bounds_max) * 0.5;
    let dilate = cull.max_foliage_height + cull.wpo_extent;
    let radius = length(bounds_max - center) + dilate;

    if !sphere_visible(center, radius) {
        return;
    }
    if hiz_occluded(center, radius) {
        return;
    }

    // LOD is classified per *cluster*, per the plan's §6.2, using the ladder of the
    // cluster's first blade. Candidates pick their type independently, so a cluster can
    // straddle two types with different ladders; when that happens the whole cluster
    // follows the first blade's. Types in a layer overwhelmingly share a ladder, and the
    // fix if that stops being true is to move the classification into the append loop
    // below and emit up to four `atomicAdd`s per cluster instead of one.
    let head = blades[tile.blade_offset + first];
    let type_id = min(
        (head.packed_scale_type >> 16u) & 0xffu,
        max(min(cull.type_count, arrayLength(&types)), 1u) - 1u,
    );
    let foliage = types[type_id];
    let distance = length(center - cameras[0].position_near.xyz);
    let lod = select_blade_lod(
        distance,
        foliage.lod0,
        foliage.lod1,
        foliage.lod2,
        foliage.lod3,
        cull.lod_quality_scale,
    );
    if lod >= FOLIAGE_LOD_COUNT {
        // Not a cull: past the last band the type contributes as a terrain-material
        // perturbation instead of geometry (plan §2.7). This pass simply stops emitting
        // draws for it.
        return;
    }

    emit_cluster(lod, count, slot, first);

    // ── Cross-fade needs the cluster in BOTH bands ────────────────────────────
    //
    // The consumer's `foliage_cross_fade` gives the near LOD weight `f` and the far LOD
    // `1 - f` across the overlap band, so the two sum to one blade's coverage. That only
    // works if both instances actually exist. Classifying each cluster into exactly one
    // LOD — which is what this pass used to do — means the near LOD fades *out* over the
    // band and nothing fades *in*: coverage collapses toward the boundary and then jumps
    // back at it. That is the hard-edged LOD banding, and no amount of tuning the fade
    // curve fixes it, because the second half of the blend was never submitted.
    //
    // So a cluster inside the band is emitted twice, once per adjacent LOD, and the
    // fragment shader's complementary weights do the rest. The extra instances exist only
    // within `lod_fade_band` metres of a boundary, so the cost is a thin annulus, not a
    // doubling.
    let upper = lod_upper_threshold(foliage, lod, cull.lod_quality_scale);
    if lod + 1u < FOLIAGE_LOD_COUNT && distance > upper - cull.lod_fade_band {
        emit_cluster(lod + 1u, count, slot, first);
    }
}

/// Append one cluster's blade references into `visible_blades[]` for a single LOD.
fn emit_cluster(lod: u32, count: u32, slot: u32, first: u32) {
    // L3 is a *clump* card: one card standing in for the whole 4x4 cluster, not one card
    // per blade. The consumer sizes it to cover the cluster (`sqrt(cluster_granularity)`
    // times a blade's width), so emitting one per blade puts sixteen oversized cards
    // where one belongs and the far ring reads denser than the near field — the density
    // falloff looks inverted. The near LODs stay one instance per blade.
    let emit = select(count, 1u, lod == FOLIAGE_LOD_CLUMP);

    // Bounded append. The reservation can overshoot the region, so the finalize stage
    // clamps `instance_count` — and the partial write below guarantees every slot below
    // that clamp was actually written, which a "reserve then drop the whole cluster"
    // policy would not: it would leave a hole of uninitialised indices inside the drawn
    // range and render blades from wherever those indices happened to point.
    let capacity = cull.per_lod_capacity;
    let base = atomicAdd(&counters[lod], emit);
    if base >= capacity {
        atomicAdd(&counters[COUNTER_VISIBLE_OVERFLOW], emit);
        return;
    }
    let writable = min(emit, capacity - base);
    if writable < emit {
        atomicAdd(&counters[COUNTER_VISIBLE_OVERFLOW], emit - writable);
    }

    let region = lod * capacity;
    for (var i = 0u; i < writable; i = i + 1u) {
        let out_index = region + base + i;
        if out_index < arrayLength(&visible_blades) {
            // Packed reference, NOT a flat arena index: tile slot in the high 16 bits,
            // tile-local blade index in the low 16. `FoliageGBufferPass` needs the tile
            // anyway (for `tile_coord` and `bounds_center_y`, which is how a blade's
            // world position is reconstructed), so handing it the slot directly saves a
            // division per *vertex* in the hottest shader in the foliage path. The
            // consumer mirrors this as `VISIBLE_TILE_SHIFT` / `VISIBLE_LOCAL_MASK`; the
            // two must change together or blades render from the wrong tile's origin.
            //
            // Both halves are comfortably in range: the ring is 4096 slots and a tile's
            // arena slab is a few hundred blades, against a 65 536 ceiling each.
            visible_blades[out_index] = (slot << 16u) | ((first + i) & 0xffffu);
        }
    }
}

// ── Stage 3: counters -> DrawIndirectArgs ───────────────────────────────────────

@compute @workgroup_size(4)
fn cs_finalize(@builtin(local_invocation_index) lane: u32) {
    if lane >= FOLIAGE_LOD_COUNT || lane >= arrayLength(&indirect) {
        return;
    }

    // 11 / 7 / 4 / 4, matching FOLIAGE_LOD_VERTEX_COUNTS: a 5-segment blade, a 3-segment
    // blade, a card and a clump card, all `TriangleStrip` with no vertex or index buffer.
    // Written as a branch rather than a const array because indexing a `const` array with
    // a runtime value is not portable across WGSL front-ends.
    var vertex_count = 4u;
    if lane == 0u {
        vertex_count = 11u;
    } else if lane == 1u {
        vertex_count = 7u;
    }

    let raw = atomicLoad(&counters[lane]);
    let instance_count = min(raw, cull.per_lod_capacity);

    indirect[lane].vertex_count = vertex_count;
    indirect[lane].instance_count = instance_count;
    indirect[lane].first_vertex = 0u;
    // Always 0. The per-LOD region base is applied by the consumer through
    // `lod_region_offset`, not by the draw call, so that a single vertex shader can index
    // `visible_blades` with `instance_index` in every LOD.
    indirect[lane].first_instance = 0u;
}
