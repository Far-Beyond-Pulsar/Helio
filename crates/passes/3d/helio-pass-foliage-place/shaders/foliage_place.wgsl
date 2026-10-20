// Foliage placement — stage 1 of `FoliagePlacePass`, on any ground (`contract::ground`).
//
// A tile is placed in two halves a frame apart:
//
//  1. `cs_query` (the frame a tile is queued): one workgroup per tile writes one
//     `GroundQuery` per stratified candidate — a segment down the tile's up through the
//     candidate — and lane 0 of the first workgroup the answering dispatch's header.
//     The ground answers before the next frame: `cs_ground_plane` here for a
//     heightfield (the terrain capture, else the plane y = 0), or another pass (a voxel
//     world's) reading the published `foliage_ground_*` resources.
//  2. `cs_place` (the next frame): one workgroup per answered tile reads its hits, takes
//     the tile's height range along its up, accepts each candidate by the types that grow
//     on its ground, and compacts the survivors into the tile's fixed arena slab.
//
// ── Determinism is the contract, not a nice-to-have ─────────────────────────────
//
// The same (tile_coord, generation, lane) over the same ground must produce a
// byte-identical blade list on every GPU, in every build, forever. Residency caching, the
// stable per-blade seed that wind phase and the dithered LOD cross-fade key off, and the
// CPU reference this shader is tested against all rest on it. Two consequences that look
// like over-engineering and are not:
//
//  1. **Nothing frame-dependent may enter the hash.** No frame index, no time, no
//     `atomicAdd` result. A seed that changes between frames turns the stochastic
//     cross-fade into full-screen static that TAA cannot resolve.
//
//  2. **Compaction is a workgroup prefix sum, not `atomicAdd`.** Atomic ordering is
//     unspecified, so the arena contents would differ run to run. A 64-lane Hillis-Steele
//     scan makes the slab a pure function of the candidate index.
//
// The packing helpers below are hand-rolled rather than using `pack2x16unorm` /
// `pack4x8unorm` on purpose: the builtins round to nearest even, while
// `crate::packing` rounds half away from zero (`x * 65535.0 + 0.5` then
// truncate). `pack2x16float` *is* used for the f16 height offset because both sides round
// to nearest even there.

const WG_SIZE: u32 = 64u;
const TAU: f32 = 6.283185307179586;

// Mirrors crate::TileState.
const TILE_STATE_PLACING: u32 = 1u;
const TILE_STATE_RESIDENT: u32 = 2u;

// Mirrors the COUNTER_* constants in src/lib.rs. Slots 0..4 are the per-LOD visible
// counts owned by the cull shader; placement only touches 5 and 6.
const COUNTER_PLACEMENT_OVERFLOW: u32 = 5u;
const COUNTER_PLACED_BLADES: u32 = 6u;

// Mirrors contract::ground::GROUND_NONE.
const GROUND_NONE: u32 = 0xffffffffu;
// A queue entry's slot (low half) and its tile's place in last frame's queries (high).
const QUEUE_SLOT_MASK: u32 = 0xffffu;

/// Mirrors `PlaceUniforms` (Rust, 80 bytes).
struct PlaceUniforms {
    tile_size:          f32,
    candidate_grid:     u32,
    /// Edge length of one cluster block in cells. Candidates are mapped block-linearly
    /// so a cluster is a square patch, not a 1-cell-tall strip — see `candidate_uv`.
    cluster_edge:       u32,
    slab_capacity:      u32,

    /// Tiles to place this frame (`tile_queue[0..]`), then tiles to query
    /// (`tile_queue[place_tile_count..]`).
    place_tile_count:   u32,
    query_tile_count:   u32,
    density_multiplier: f32,
    /// Candidates per square metre of the grid (`candidate_grid² / tile area`): the
    /// denominator of the acceptance probability.
    grid_density:       f32,

    type_count:         u32,
    max_foliage_height: f32,
    // 0 until a terrain capture and its ring transform are published; then the built-in
    // ground samples it (`ground_plane_height`).
    terrain_valid:      u32,
    terrain_origin_x:   f32,

    terrain_origin_z:   f32,
    terrain_extent:     f32,
    // Number of valid entries in the layer table. Zero = carpet everything.
    layer_count:        u32,
    /// How far over the eye's height along a tile's up a query starts, and how far under
    /// it it ends.
    search_above:       f32,

    search_below:       f32,
    eye_x:              f32,
    eye_y:              f32,
    eye_z:              f32,
}

/// Mirrors `crate::GpuFoliageType` (Rust, 96 bytes). **Every field is a scalar** — see the
/// struct doc in `src/contract/gpu_types.rs`.
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
    base_color:            u32,
    roughness_metallic:    u32,
    ground_mask:           u32,
}

/// Mirrors `crate::GpuFoliageTile` (Rust, 32 bytes).
struct FoliageTile {
    tile_coord_x:  i32,
    tile_coord_z:  i32,
    blade_offset:  u32,
    blade_count:   u32,
    bounds_center: f32,
    bounds_half:   f32,
    state:         u32,
    generation:    u32,
}

/// Mirrors `crate::GpuTileFrame` (Rust, 64 bytes).
struct TileFrame {
    origin: vec3<f32>,
    axis_u: vec3<f32>,
    axis_v: vec3<f32>,
    up:     vec3<f32>,
}

/// Mirrors `crate::GpuFoliageLayer` (Rust, 32 bytes).
struct FoliageLayer {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}

/// Mirrors `crate::GpuBladeInstance` (Rust, 16 bytes).
struct BladeInstance {
    packed_pos:        u32,
    packed_height_yaw: u32,
    packed_scale_type: u32,
    packed_tint_seed:  u32,
}

/// Mirrors `crate::GpuGroundQuery` (Rust, 32 bytes).
struct GroundQuery {
    start:  vec3<f32>,
    length: f32,
    up:     vec3<f32>,
    pad:    u32,
}

/// Mirrors `crate::GpuGroundHit` (Rust, 16 bytes).
struct GroundHit {
    distance: u32,
    normal:   u32,
    ground:   u32,
    pad:      u32,
}

/// Mirrors `crate::GpuGroundArgs` (Rust, 16 bytes).
struct GroundArgs {
    groups: vec3<u32>,
    count:  u32,
}

@group(0) @binding(0)  var<uniform> place: PlaceUniforms;
@group(0) @binding(1)  var<storage, read> types: array<FoliageType>;
@group(0) @binding(2)  var<storage, read_write> tiles: array<FoliageTile>;
@group(0) @binding(3)  var<storage, read_write> blades: array<BladeInstance>;
@group(0) @binding(4)  var<storage, read> tile_queue: array<u32>;
@group(0) @binding(5)  var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6)  var terrain_height_slope: texture_2d<f32>;
@group(0) @binding(7)  var terrain_samp: sampler;
@group(0) @binding(8)  var<storage, read> layers: array<FoliageLayer>;
@group(0) @binding(9)  var<storage, read> frames: array<TileFrame>;
@group(0) @binding(10) var<storage, read_write> queries: array<GroundQuery>;
@group(0) @binding(11) var<storage, read_write> hits: array<GroundHit>;
@group(0) @binding(12) var<storage, read_write> ground_args: GroundArgs;

var<workgroup> wg_scan: array<u32, WG_SIZE>;
var<workgroup> wg_base: u32;
var<workgroup> wg_low: atomic<i32>;
var<workgroup> wg_high: atomic<i32>;
var<workgroup> wg_live_types: atomic<u32>;

// ── Hash, transcribed from crate::placement ────────────────────────

fn rotl(value: u32, amount: u32) -> u32 {
    return (value << amount) | (value >> (32u - amount));
}

/// Byte-for-byte transcription of `crate::blade_seed`.
///
/// `bitcast<u32>` on the coordinates matches Rust's `i32 as u32`, i.e. two's-complement
/// reinterpretation. Do not "fix" this with `abs()`: that aliases tiles across the chart
/// origin, and grass west of it becomes a mirror image of grass east of it.
fn blade_seed(coord_x: i32, coord_z: i32, lane: u32, generation: u32) -> u32 {
    var h = bitcast<u32>(coord_x) * 374761393u
        + bitcast<u32>(coord_z) * 668265263u
        + lane * 2654435761u
        + generation * 2246822519u;
    h = (h ^ (h >> 15u)) * 2246822519u;
    h = (h ^ (h >> 13u)) * 3266489917u;
    h = h ^ (h >> 16u);
    return h;
}

/// Transcription of `crate::hash_to_unit`: top 24 bits over 2^24, strictly below 1.0.
fn hash_to_unit(hash: u32) -> f32 {
    return f32(hash >> 8u) * (1.0 / 16777216.0);
}

// ── Packing, matching crate::packing exactly ───────────────────────

fn pack_unorm16_u(value: f32) -> u32 {
    return u32(clamp(value, 0.0, 1.0) * 65535.0 + 0.5);
}

fn pack_unorm8_u(value: f32) -> u32 {
    return u32(clamp(value, 0.0, 1.0) * 255.0 + 0.5);
}

/// 16 bits of *turn*, not unorm: 0 and 2π are the same orientation.
fn pack_yaw_u(radians: f32) -> u32 {
    let turns = radians * (1.0 / TAU);
    let wrapped = turns - floor(turns);
    return u32(wrapped * 65536.0 + 0.5) & 0xffffu;
}

fn pack_pos(u: f32, v: f32) -> u32 {
    return pack_unorm16_u(u) | (pack_unorm16_u(v) << 16u);
}

fn pack_height_yaw(height_offset: f32, yaw: f32) -> u32 {
    return (pack2x16float(vec2<f32>(height_offset, 0.0)) & 0xffffu) | (pack_yaw_u(yaw) << 16u);
}

fn pack_scale_type(height: f32, width: f32, type_id: u32, variant: u32) -> u32 {
    return pack_unorm8_u(height)
        | (pack_unorm8_u(width) << 8u)
        | ((type_id & 0xffu) << 16u)
        | ((variant & 0xffu) << 24u);
}

fn pack_tint_seed(tint_x: f32, tint_y: f32, seed: u32) -> u32 {
    return pack_unorm8_u(tint_x) | (pack_unorm8_u(tint_y) << 8u) | ((seed & 0xffffu) << 16u);
}

/// An `f32` as an `i32` ordered like the float (for `atomicMin`/`atomicMax`).
fn ordered(x: f32) -> i32 {
    let i = bitcast<i32>(x);
    return i ^ ((i >> 31u) & 0x7fffffff);
}

fn unordered(i: i32) -> f32 {
    return bitcast<f32>(i ^ ((i >> 31u) & 0x7fffffff));
}

/// Transcription of `contract::ground::unpack_octahedral`.
fn unpack_octahedral(packed: u32) -> vec3<f32> {
    let p = unpack2x16snorm(packed);
    var n = vec3<f32>(p, 1.0 - abs(p.x) - abs(p.y));
    if n.z < 0.0 {
        n = vec3<f32>((1.0 - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0)), n.z);
    }
    let l = dot(n, n);
    return select(vec3<f32>(0.0, 1.0, 0.0), n * inverseSqrt(l), l > 1.0e-12);
}

// ── Candidates ──────────────────────────────────────────────────────────────────

/// Candidate `candidate` of a tile: its stratified, jittered `(u, v)` in the tile, on the
/// unorm16 grid the blade stores. The ground is asked at exactly the stored point: a
/// candidate a hair from a voxel step's edge would otherwise be answered by one column
/// and stored over the other, a whole step off.
///
/// Stratified: one candidate per grid cell, jittered inside it — pure rejection sampling
/// clumps. Block-linear, not row-major: `cluster_size` consecutive candidates form one
/// cluster, and the L3 clump card is anchored on its cluster, so a row-major mapping
/// renders the far field as straight rows.
fn candidate_uv(candidate: u32, seed: u32) -> vec2<f32> {
    let grid = max(place.candidate_grid, 1u);
    let edge = max(place.cluster_edge, 1u);
    let per_block = edge * edge;
    let blocks_across = max(grid / edge, 1u);
    let block = candidate / per_block;
    let within = candidate % per_block;
    let cell_x = (block % blocks_across) * edge + (within % edge);
    let cell_z = (block / blocks_across) * edge + (within / edge);
    let inv_grid = 1.0 / f32(grid);
    let uv = vec2<f32>(
        (f32(cell_x) + hash_to_unit(seed)) * inv_grid,
        (f32(cell_z) + hash_to_unit(rotl(seed, 11u))) * inv_grid,
    );
    return vec2<f32>(f32(pack_unorm16_u(uv.x)), f32(pack_unorm16_u(uv.y))) * (1.0 / 65535.0);
}

fn frame_point(frame: TileFrame, uv: vec2<f32>, h: f32) -> vec3<f32> {
    return frame.origin + frame.axis_u * uv.x + frame.axis_v * uv.y + frame.up * h;
}

// ── 1. Queries ──────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn cs_query(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let cells = max(place.candidate_grid, 1u) * max(place.candidate_grid, 1u);
    if workgroup_id.x == 0u && lane == 0u {
        let count = place.query_tile_count * cells;
        ground_args.groups = vec3<u32>((count + WG_SIZE - 1u) / WG_SIZE, 1u, 1u);
        ground_args.count = count;
    }
    if workgroup_id.x >= place.query_tile_count { return; }
    let at = place.place_tile_count + workgroup_id.x;
    if at >= arrayLength(&tile_queue) { return; }
    let slot = tile_queue[at] & QUEUE_SLOT_MASK;
    if slot >= arrayLength(&tiles) { return; }
    let tile = tiles[slot];
    let frame = frames[slot];
    // Every query of the tile starts the same height over its datum: the eye's, raised.
    let eye_h = dot(vec3<f32>(place.eye_x, place.eye_y, place.eye_z) - frame.origin, frame.up);
    let start_h = eye_h + place.search_above;
    let length = place.search_above + place.search_below;
    let first = workgroup_id.x * cells;
    for (var candidate = lane; candidate < cells; candidate += WG_SIZE) {
        let seed = blade_seed(tile.tile_coord_x, tile.tile_coord_z, candidate, tile.generation);
        let uv = candidate_uv(candidate, seed);
        let index = first + candidate;
        if index < arrayLength(&queries) {
            queries[index] = GroundQuery(frame_point(frame, uv, start_h), length, frame.up, 0u);
        }
    }
}

// ── The built-in ground: a heightfield over world XZ ────────────────────────────

/// `(height, cos(slope))` of the heightfield at world XZ: the terrain capture where one
/// is published (`terrain_valid`), else the plane y = 0 with a straight-up normal.
fn ground_plane_height(world_xz: vec2<f32>) -> vec2<f32> {
    if place.terrain_valid == 0u {
        return vec2<f32>(0.0, 1.0);
    }
    let uv = (world_xz - vec2<f32>(place.terrain_origin_x, place.terrain_origin_z))
        / max(place.terrain_extent, 1.0e-3);
    let texel = textureSampleLevel(terrain_height_slope, terrain_samp, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0);
    return texel.rg;
}

/// The built-in ground's answers (+Y up, ground id 0): a thread a query, over the header
/// `cs_query` wrote (dispatched indirectly from it).
@compute @workgroup_size(64)
fn cs_ground_plane(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= ground_args.count || id.x >= arrayLength(&queries) || id.x >= arrayLength(&hits) { return; }
    let q = queries[id.x];
    let terrain = ground_plane_height(q.start.xz);
    let c = clamp(terrain.y, -1.0, 1.0);
    // The slope's direction is not captured, only its steepness: a normal tilted by it.
    let normal = vec3<f32>(sqrt(max(1.0 - c * c, 0.0)), c, 0.0);
    let p = vec2<f32>(normal.x, normal.y) / (abs(normal.x) + abs(normal.y) + abs(normal.z));
    var hit: GroundHit;
    hit.distance = bitcast<u32>(q.start.y - terrain.x);
    hit.normal = pack2x16snorm(p);
    hit.ground = 0u;
    hits[id.x] = hit;
}

// ── 2. Placement ────────────────────────────────────────────────────────────────

/// Whether a render-space position lies inside at least one authored layer. An empty
/// layer table carpets everything; an infinite-extent layer accepts everything.
fn within_any_layer(p: vec3<f32>) -> bool {
    let layer_limit = min(place.layer_count, arrayLength(&layers));
    if layer_limit == 0u {
        return true;
    }
    for (var i = 0u; i < layer_limit; i = i + 1u) {
        let layer = layers[i];
        if layer.bounds_max.w > 0.5 {
            return true;
        }
        if all(p >= layer.bounds_min.xyz) && all(p <= layer.bounds_max.xyz) {
            return true;
        }
    }
    return false;
}

/// Whether type `t` grows on ground `ground` at slope `slope_cos` and height `h`.
fn type_grows(t: FoliageType, ground: u32, slope_cos: f32, h: f32) -> bool {
    let on_ground = t.ground_mask == 0u || (ground < 32u && ((t.ground_mask >> ground) & 1u) != 0u);
    return t.density > 0.0 && on_ground
        && slope_cos >= t.slope_min && slope_cos <= t.slope_max
        && h >= t.altitude_min && h <= t.altitude_max;
}

/// A candidate's answered ground: whether there is any, its height along the tile's up
/// over the frame's datum, the normal and the ground id.
struct Ground {
    ok: bool,
    h: f32,
    normal: vec3<f32>,
    ground: u32,
}

fn answered_ground(index: u32, frame: TileFrame) -> Ground {
    var g = Ground(false, 0.0, frame.up, GROUND_NONE);
    if index >= arrayLength(&hits) || index >= arrayLength(&queries) { return g; }
    let hit = hits[index];
    let q = queries[index];
    let distance = bitcast<f32>(hit.distance);
    if hit.ground == GROUND_NONE || !(distance >= 0.0 && distance <= q.length) { return g; }
    g.ok = true;
    g.h = dot(q.start - frame.origin, frame.up) - distance;
    g.normal = unpack_octahedral(hit.normal);
    g.ground = hit.ground;
    return g;
}

@compute @workgroup_size(64)
fn cs_place(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // Resolve the tile *without* letting the result gate any barrier: `tile_queue` is a
    // storage load, so uniformity analysis cannot prove it uniform. Every lane runs the
    // full candidate loops and only the *writes* are guarded.
    var slot = 0u;
    var asked = 0u;
    var valid = false;
    if workgroup_id.x < place.place_tile_count && workgroup_id.x < arrayLength(&tile_queue) {
        let entry = tile_queue[workgroup_id.x];
        slot = entry & QUEUE_SLOT_MASK;
        asked = entry >> 16u;
        valid = slot < arrayLength(&tiles);
    }
    var tile: FoliageTile;
    var frame: TileFrame;
    if valid {
        tile = tiles[slot];
        frame = frames[slot];
    }
    let cells = max(place.candidate_grid, 1u) * max(place.candidate_grid, 1u);
    let first = asked * cells;

    if lane == 0u {
        atomicStore(&wg_low, ordered(3.0e38));
        atomicStore(&wg_high, ordered(-3.0e38));
        atomicStore(&wg_live_types, 0u);
        wg_base = 0u;
    }
    workgroupBarrier();

    // The tile's height range along its up: blades store a height *offset* from its
    // centre, so it is known before any blade is packed.
    for (var candidate = lane; candidate < cells; candidate += WG_SIZE) {
        let g = answered_ground(first + candidate, frame);
        if g.ok {
            atomicMin(&wg_low, ordered(g.h));
            atomicMax(&wg_high, ordered(g.h));
        }
    }
    // Types are the leading rows up to the last one with a density: a publisher keeping
    // a fixed-capacity, packed table must not pay for its empty rows.
    let type_rows = min(place.type_count, arrayLength(&types));
    for (var row = lane; row < type_rows; row = row + WG_SIZE) {
        if types[row].density > 0.0 {
            atomicMax(&wg_live_types, row + 1u);
        }
    }
    workgroupBarrier();
    let low = unordered(atomicLoad(&wg_low));
    let high = unordered(atomicLoad(&wg_high));
    let any_ground = low <= high;
    let center = select(0.0, (low + high) * 0.5, any_ground);
    let type_limit = atomicLoad(&wg_live_types);
    if lane == 0u && valid {
        tiles[slot].bounds_center = center;
        tiles[slot].bounds_half = select(0.0, (high - low) * 0.5, any_ground) + place.max_foliage_height;
        tiles[slot].blade_count = 0u;
        tiles[slot].state = TILE_STATE_PLACING;
    }
    workgroupBarrier();

    // `cells` comes from the uniform buffer, so this loop and every barrier inside it are
    // in uniform control flow for the whole workgroup.
    var chunk = 0u;
    loop {
        if chunk >= cells {
            break;
        }

        let candidate = chunk + lane;
        var accepted = 0u;
        var record = BladeInstance(0u, 0u, 0u, 0u);

        if valid && candidate < cells {
            let seed = blade_seed(tile.tile_coord_x, tile.tile_coord_z, candidate, tile.generation);
            let uv = candidate_uv(candidate, seed);
            let g = answered_ground(first + candidate, frame);
            if g.ok && within_any_layer(frame_point(frame, uv, g.h)) {
                // The types that grow here, by their densities: the candidate grid holds
                // `grid_density` candidates a square metre, so the ground grows a blade
                // with probability `sum / grid_density` and the blade is each type's by
                // its share of the sum. Types keyed to other grounds cost nothing here.
                let slope_cos = dot(g.normal, frame.up);
                var sum = 0.0;
                for (var t = 0u; t < type_limit; t++) {
                    if type_grows(types[t], g.ground, slope_cos, g.h) { sum += types[t].density; }
                }
                let wanted = sum * place.density_multiplier;
                let roll = hash_to_unit(rotl(seed, 17u)) * max(place.grid_density, 1.0e-6);
                if roll < wanted {
                    // Which type: the roll's place in the cumulative densities.
                    let pick = hash_to_unit(rotl(seed, 5u)) * sum;
                    var type_id = 0u;
                    var running = 0.0;
                    for (var t = 0u; t < type_limit; t++) {
                        if !type_grows(types[t], g.ground, slope_cos, g.h) { continue; }
                        type_id = t;
                        running += types[t].density;
                        if pick < running { break; }
                    }
                    accepted = 1u;
                    record.packed_pos = pack_pos(uv.x, uv.y);
                    record.packed_height_yaw = pack_height_yaw(g.h - center, hash_to_unit(rotl(seed, 23u)) * TAU);
                    record.packed_scale_type = pack_scale_type(
                        hash_to_unit(rotl(seed, 29u)),
                        hash_to_unit(rotl(seed, 3u)),
                        type_id,
                        (seed >> 30u) & 3u,
                    );
                    record.packed_tint_seed = pack_tint_seed(
                        hash_to_unit(rotl(seed, 7u)),
                        hash_to_unit(rotl(seed, 13u)),
                        seed & 0xffffu,
                    );
                }
            }
        }

        // Hillis-Steele inclusive scan over the 64 acceptance flags.
        wg_scan[lane] = accepted;
        workgroupBarrier();
        var offset = 1u;
        loop {
            if offset >= WG_SIZE {
                break;
            }
            var addend = 0u;
            if lane >= offset {
                addend = wg_scan[lane - offset];
            }
            workgroupBarrier();
            wg_scan[lane] = wg_scan[lane] + addend;
            workgroupBarrier();
            offset = offset << 1u;
        }

        let inclusive = wg_scan[lane];
        let chunk_total = wg_scan[WG_SIZE - 1u];
        let base = wg_base;

        if accepted == 1u {
            let index = base + inclusive - 1u;
            let arena_index = tile.blade_offset + index;
            if index < place.slab_capacity && arena_index < arrayLength(&blades) {
                blades[arena_index] = record;
            } else {
                // Hard ceiling, never silent truncation: counted. The CPU clamps the
                // candidate grid so `cells <= slab_capacity`, so a non-zero value here
                // means the two sides have drifted.
                atomicAdd(&counters[COUNTER_PLACEMENT_OVERFLOW], 1u);
            }
        }

        // Every lane must have consumed `wg_base` before lane 0 advances it.
        workgroupBarrier();
        if lane == 0u {
            wg_base = base + chunk_total;
        }
        workgroupBarrier();

        chunk = chunk + WG_SIZE;
    }

    workgroupBarrier();
    if lane == 0u && valid {
        let count = min(wg_base, place.slab_capacity);
        tiles[slot].blade_count = count;
        // Placing -> Resident makes the tile drawable; until this line the slab holds
        // whatever its previous tenant (or its previous placement) left.
        tiles[slot].state = TILE_STATE_RESIDENT;
        atomicAdd(&counters[COUNTER_PLACED_BLADES], count);
    }
}
