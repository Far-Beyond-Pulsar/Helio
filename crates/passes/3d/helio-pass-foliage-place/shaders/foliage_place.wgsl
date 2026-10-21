//!use helio_foliage_common
// Foliage placement — stage 1 of `FoliagePlacePass`, on any ground (`contract::ground`),
// in every band (`contract::bands`).
//
// A tile is placed in two halves a frame apart:
//
//  1. `cs_query` (the frame a tile is queued): one workgroup per tile writes one
//     `GroundQuery` per candidate — a segment down the tile's up through the candidate.
//     The ground answers before the next frame: `cs_ground_plane` here for a heightfield
//     (the terrain capture, else the plane y = 0), or another pass (a voxel world's)
//     reading the published `foliage_ground_*` resources.
//  2. `cs_place` (the next frame): one workgroup per answered tile reads its hits, takes
//     the tile's height range along its up, accepts each candidate as a tuft of the types
//     growing on its ground (their densities, thinned by their patches), and compacts the
//     tufts into the tile's slab: a record per blade in the near band, one per tuft past it.
//
// A tile's band (`foliage_band_of_slot`) sets its size, candidate grid, density share and
// record kind. Candidates are ordered by probe block (the 16 m blocks big tiles ask the
// ground in, each from its own reference point), then by cluster block within one.
//
// ── Determinism is the contract, not a nice-to-have ─────────────────────────────
//
// The same (band, tile_coord, generation, lane) over the same ground must produce a
// byte-identical record list on every GPU, in every build. Residency caching, the stable
// per-tuft seed that wind phase and the dithered fades key off, and the CPU reference all
// rest on it:
//
//  1. **Nothing frame-dependent may enter the hash.** No frame index, no time, no
//     `atomicAdd` result.
//  2. **Compaction is a workgroup prefix sum, not `atomicAdd`.** Atomic ordering is
//     unspecified; a 64-lane Hillis-Steele scan of the records each candidate writes makes
//     the slab a pure function of the candidate index.
//
// The packing helpers round half away from zero as `crate::packing` does (the builtins
// round to nearest even); `pack2x16float` is used for the f16 height because both sides
// round to nearest even there.

const WG_SIZE: u32 = 64u;

// Mirrors crate::TileState.
const TILE_STATE_PLACING: u32 = 1u;
const TILE_STATE_RESIDENT: u32 = 2u;

// Mirrors the COUNTER_* constants in src/lib.rs. Slots 0..4 are the per-LOD visible
// counts owned by the cull shader; placement only touches 5 and 6.
const COUNTER_PLACEMENT_OVERFLOW: u32 = 5u;
const COUNTER_PLACED_BLADES: u32 = 6u;

// Mirrors contract::ground::GROUND_NONE.
const GROUND_NONE: u32 = 0xffffffffu;

/// Mirrors `PlaceUniforms` (Rust, 80 bytes).
struct PlaceUniforms {
    /// Edge of one cluster block in candidates: `cluster_size` consecutive candidates are
    /// a square patch (the cull's clusters), not a strip.
    cluster_edge:       u32,
    /// Tiles to place this frame (`tile_queue[0..]`), then tiles to query
    /// (`tile_queue[place_tile_count..]`), and the queries those ask in all.
    place_tile_count:   u32,
    query_tile_count:   u32,
    query_count:        u32,

    density_multiplier: f32,
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
    _pad0:              u32,
    _pad1:              u32,
    _pad2:              u32,
}

/// Mirrors `crate::GpuFoliageLayer` (Rust, 32 bytes).
struct FoliageLayer {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
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
    colour:   u32,
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
/// Queued tiles: their slot (x) and their first query (y).
@group(0) @binding(4)  var<storage, read> tile_queue: array<vec2<u32>>;
@group(0) @binding(5)  var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(6)  var terrain_height_slope: texture_2d<f32>;
@group(0) @binding(7)  var terrain_samp: sampler;
@group(0) @binding(8)  var<storage, read> layers: array<FoliageLayer>;
@group(0) @binding(9)  var<storage, read> frames: array<TileFrame>;
@group(0) @binding(10) var<storage, read_write> queries: array<GroundQuery>;
@group(0) @binding(11) var<storage, read_write> hits: array<GroundHit>;
@group(0) @binding(12) var<storage, read_write> ground_args: GroundArgs;
@group(0) @binding(13) var<uniform> foliage_bands: FoliageBands;

var<workgroup> wg_scan: array<u32, WG_SIZE>;
var<workgroup> wg_base: u32;
var<workgroup> wg_low: atomic<i32>;
var<workgroup> wg_high: atomic<i32>;
var<workgroup> wg_live_types: atomic<u32>;
var<workgroup> wg_cells: u32;

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

/// The seed generation of band `band`'s tiles: the bands' tiles are distinct draws.
fn band_generation(generation: u32, band: u32) -> u32 {
    return generation ^ (band * 0x9e3779b9u);
}

// ── Packing, matching crate::packing exactly ───────────────────────

fn pack_unorm16_u(value: f32) -> u32 {
    return u32(clamp(value, 0.0, 1.0) * 65535.0 + 0.5);
}

fn pack_unorm8_u(value: f32) -> u32 {
    return u32(clamp(value, 0.0, 1.0) * 255.0 + 0.5);
}

fn pack_pos(u: f32, v: f32) -> u32 {
    return pack_unorm16_u(u) | (pack_unorm16_u(v) << 16u);
}

fn pack_height_tuft(height_offset: f32, blade: u32, blades: u32, rank: f32) -> u32 {
    return (pack2x16float(vec2<f32>(height_offset, 0.0)) & 0xffffu)
        | ((blade & 15u) << 16u)
        | ((blades & 15u) << 20u)
        | (pack_unorm8_u(rank) << 24u);
}

fn pack_scale_type(height: f32, aux: u32, type_id: u32, variant: u32) -> u32 {
    return pack_unorm8_u(height)
        | ((aux & 0xffu) << 8u)
        | ((type_id & 0xffu) << 16u)
        | ((variant & 0xffu) << 24u);
}

/// `crate::pack_rgb565` of an RGBA8 ground colour (0: none).
fn pack_ground_colour(rgba8: u32) -> u32 {
    if rgba8 == 0u {
        return 0u;
    }
    let c = unpack4x8unorm(rgba8);
    let packed = u32(c.r * 31.0 + 0.5) | (u32(c.g * 63.0 + 0.5) << 5u) | (u32(c.b * 31.0 + 0.5) << 11u);
    return max(packed, 1u);
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

/// The cubes across a tile's two axes that blocky types stand on (`FoliageKind::Block`,
/// whose cube is the type's `width_min`: the smallest of them), or zero without any.
/// Whole cubes a tile, so the grid runs on unbroken from tile to tile.
fn cube_cells(frame: TileFrame) -> vec2<u32> {
    var edge = 0.0;
    let rows = min(place.type_count, arrayLength(&types));
    for (var t = 0u; t < rows; t++) {
        let ty = types[t];
        if (ty.kind_and_flags & 0xffu) == FOLIAGE_KIND_BLOCK && ty.density > 0.0 && ty.width_min > 0.0 {
            edge = select(min(edge, ty.width_min), ty.width_min, edge == 0.0);
        }
    }
    if edge <= 0.0 {
        return vec2<u32>(0u);
    }
    return vec2<u32>(
        u32(max(round(length(frame.axis_u) / edge), 1.0)),
        u32(max(round(length(frame.axis_v) / edge), 1.0)),
    );
}

/// The cube a candidate stands on along one axis: one of the cubes whose centres fall in
/// its grid cell `cell` of `grid` (picked by `pick` in 0..1), of `cubes` across. Integer
/// ownership, so no two candidates share a cube; -1 when the cell holds no centre (cubes
/// coarser than the grid).
fn owned_cube(cell: u32, grid: u32, cubes: u32, pick: f32) -> i32 {
    // Cube k's centre (k + 1/2) / cubes is in cell floor((2k + 1) grid / (2 cubes)).
    let g = i32(grid);
    let n2 = 2 * i32(cubes);
    let a = (2 * i32(cell) * i32(cubes) + g - 1) / g;
    let b = (n2 * (i32(cell) + 1) + g - 1) / g;
    let lo = a / 2;
    let hi = (b - 2) / 2 + select(0, -1, b < 2);
    if hi < lo {
        return -1;
    }
    return min(lo + i32(pick * f32(hi - lo + 1)), hi);
}

/// Candidate `candidate`'s cell in a tile of `band`: probe block first (the blocks a big
/// tile asks the ground in), then cluster block within it (`cluster_edge`² consecutive
/// candidates are a square patch, the cull's cluster), then row. Mirror of
/// `reference::candidate_cell`.
fn candidate_cell(candidate: u32, band: FoliageBand) -> vec2<u32> {
    let grid = max(band.candidate_grid, 1u);
    let blocks = max(band.probe_blocks, 1u);
    let per_side = max(grid / blocks, 1u);
    let per_probe = per_side * per_side;
    let probe = candidate / per_probe;
    let within = candidate % per_probe;
    let edge = max(place.cluster_edge, 1u);
    let clusters_across = max(per_side / edge, 1u);
    let cluster = within / (edge * edge);
    let inside = within % (edge * edge);
    return vec2<u32>(
        (probe % blocks) * per_side + (cluster % clusters_across) * edge + inside % edge,
        (probe / blocks) * per_side + (cluster / clusters_across) * edge + inside / edge,
    );
}

/// Candidate `candidate`'s `(u, v)` in its tile, on the unorm16 grid the record stores:
/// the ground is asked at exactly the stored point (a candidate a hair from a voxel step's
/// edge would otherwise be answered by one column and stored over the other). With a cube
/// grid (`cubes`) the candidate is the centre of a cube its cell owns, or `-1` (none);
/// else it is jittered in its cell (stratified: one a cell — pure rejection sampling
/// clumps).
fn candidate_uv(candidate: u32, seed: u32, cubes: vec2<u32>, band: FoliageBand) -> vec2<f32> {
    let grid = max(band.candidate_grid, 1u);
    let cell = candidate_cell(candidate, band);
    let pick = vec2<f32>(hash_to_unit(seed), hash_to_unit(rotl(seed, 11u)));
    if cubes.x > 0u {
        let ku = owned_cube(cell.x, grid, cubes.x, pick.x);
        let kv = owned_cube(cell.y, grid, cubes.y, pick.y);
        if ku < 0 || kv < 0 {
            return vec2<f32>(-1.0);
        }
        let at = (vec2<f32>(f32(ku), f32(kv)) + 0.5) / vec2<f32>(cubes);
        return vec2<f32>(f32(pack_unorm16_u(at.x)), f32(pack_unorm16_u(at.y))) * (1.0 / 65535.0);
    }
    let uv = (vec2<f32>(cell) + pick) / f32(grid);
    return vec2<f32>(f32(pack_unorm16_u(uv.x)), f32(pack_unorm16_u(uv.y))) * (1.0 / 65535.0);
}

fn frame_point(frame: TileFrame, uv: vec2<f32>, h: f32) -> vec3<f32> {
    return frame.origin + frame.axis_u * uv.x + frame.axis_v * uv.y + frame.up * h;
}

// ── Patches ─────────────────────────────────────────────────────────────────────

/// A lattice hash in `[0, 1)`.
fn lattice(i: vec2<i32>, salt: u32) -> f32 {
    return foliage_unit(foliage_hash(bitcast<u32>(i.x) * 73856093u ^ bitcast<u32>(i.y) * 19349663u ^ salt));
}

/// Value noise in `[0, 1]` over the chart at the point `uv` of the tile at band `coord`
/// (`span` base tiles across), with lattice cells `2^e` base tiles (8 m) across. The
/// lattice index is integer arithmetic on the tile coordinate, so patches are as sharp at
/// a planet's millionth tile as at its first (a float position there is half a metre off).
fn chart_noise(coord: vec2<i32>, uv: vec2<f32>, span: u32, e: i32, salt: u32) -> f32 {
    let base = coord * i32(span);
    let local = uv * f32(span);
    var i: vec2<i32>;
    var f: vec2<f32>;
    if e <= 0 {
        let at = local * f32(1u << u32(-e));
        i = base * i32(1u << u32(-e)) + vec2<i32>(floor(at));
        f = fract(at);
    } else {
        let mask = i32((1u << u32(e)) - 1u);
        let at = (vec2<f32>(base & vec2<i32>(mask)) + local) / f32(1u << u32(e));
        i = (base >> vec2<u32>(u32(e))) + vec2<i32>(floor(at));
        f = fract(at);
    }
    let s = f * f * (3.0 - 2.0 * f);
    let a = mix(lattice(i, salt), lattice(i + vec2<i32>(1, 0), salt), s.x);
    let b = mix(lattice(i + vec2<i32>(0, 1), salt), lattice(i + vec2<i32>(1, 1), salt), s.x);
    return mix(a, b, s.y);
}

/// Type `t`'s patch weight at the candidate (`0..1`): bare where its noise is low, by its
/// strength. Two octaves, so patches have ragged edges.
fn patch_weight(ty: FoliageType, t: u32, coord: vec2<i32>, uv: vec2<f32>, span: u32) -> f32 {
    if ty.patch_scale <= 0.0 || ty.patch_strength <= 0.0 {
        return 1.0;
    }
    let e = clamp(i32(round(log2(ty.patch_scale / 8.0))), -8, 16);
    let salt = t * 0x27d4eb2fu + 0x165667b1u;
    let n = 0.7 * chart_noise(coord, uv, span, e, salt) + 0.3 * chart_noise(coord, uv, span, e - 2, salt ^ 0x9e3779b9u);
    let f = smoothstep(0.38, 0.62, n);
    return mix(1.0 - clamp(ty.patch_strength, 0.0, 1.0), 1.0, f);
}

// ── 1. Queries ──────────────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn cs_query(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    if workgroup_id.x == 0u && lane == 0u {
        ground_args.groups = vec3<u32>((place.query_count + WG_SIZE - 1u) / WG_SIZE, 1u, 1u);
        ground_args.count = place.query_count;
    }
    if workgroup_id.x >= place.query_tile_count { return; }
    let at = place.place_tile_count + workgroup_id.x;
    if at >= arrayLength(&tile_queue) { return; }
    let entry = tile_queue[at];
    let slot = entry.x;
    if slot >= arrayLength(&tiles) { return; }
    let tile = tiles[slot];
    let frame = frames[slot];
    let band_index = foliage_band_of_slot(slot);
    let band = foliage_bands.bands[band_index];
    let generation = band_generation(tile.generation, band_index);
    // Every query of the tile starts the same height over its datum: the eye's, raised.
    let eye_h = dot(vec3<f32>(place.eye_x, place.eye_y, place.eye_z) - frame.origin, frame.up);
    let start_h = eye_h + place.search_above;
    let length = place.search_above + place.search_below;
    let cells = band.candidate_grid * band.candidate_grid;
    let cubes = cube_cells(frame);
    for (var candidate = lane; candidate < cells; candidate += WG_SIZE) {
        let seed = blade_seed(tile.tile_coord_x, tile.tile_coord_z, candidate, generation);
        let uv = max(candidate_uv(candidate, seed, cubes, band), vec2<f32>(0.0));
        let index = entry.y + candidate;
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

/// The built-in ground's answers (+Y up, ground id 0, no colour): a thread a query.
@compute @workgroup_size(64)
fn cs_ground_plane(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= place.query_count || id.x >= arrayLength(&queries) || id.x >= arrayLength(&hits) { return; }
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
    hit.colour = 0u;
    hits[id.x] = hit;
}

// ── 2. Placement ────────────────────────────────────────────────────────────────

/// True when `p` lies inside at least one layer's AABB, or the layer table is empty.
fn within_any_layer(p: vec3<f32>) -> bool {
    let count = min(place.layer_count, arrayLength(&layers));
    if count == 0u {
        return true;
    }
    for (var i = 0u; i < count; i++) {
        let layer = layers[i];
        if layer.bounds_max.w >= 0.5 {
            return true;
        }
        if all(p >= layer.bounds_min.xyz) && all(p <= layer.bounds_max.xyz) {
            return true;
        }
    }
    return false;
}

/// Whether type `t` grows on ground id `ground` with `slope_cos` at height `h`.
fn type_grows(t: FoliageType, ground: u32, slope_cos: f32, h: f32) -> bool {
    let on_ground = t.ground_mask == 0u || (ground < 32u && ((t.ground_mask >> ground) & 1u) != 0u);
    return t.density > 0.0 && on_ground
        && slope_cos >= t.slope_min && slope_cos <= t.slope_max
        && h >= t.altitude_min && h <= t.altitude_max;
}

/// A candidate's answered ground: whether there is any, its height along the tile's up
/// over the frame's datum, the normal, the ground id and colour.
struct Ground {
    ok: bool,
    h: f32,
    normal: vec3<f32>,
    ground: u32,
    colour: u32,
}

fn answered_ground(index: u32, frame: TileFrame, start_h: f32, length: f32) -> Ground {
    var g = Ground(false, 0.0, frame.up, GROUND_NONE, 0u);
    if index >= arrayLength(&hits) {
        return g;
    }
    let hit = hits[index];
    let d = bitcast<f32>(hit.distance);
    if hit.ground == GROUND_NONE || !(d >= 0.0) || d > length {
        return g;
    }
    g.ok = true;
    g.h = start_h - d;
    g.normal = unpack_octahedral(hit.normal);
    g.ground = hit.ground;
    g.colour = hit.colour;
    return g;
}

/// The blades a tuft of type `ty` holds, by `pick` in 0..1.
fn tuft_blades(ty: FoliageType, pick: f32) -> u32 {
    let lo = clamp(ty.blades_per_tuft & 0xffffu, 1u, 15u);
    let hi = clamp(ty.blades_per_tuft >> 16u, lo, 15u);
    return min(lo + u32(pick * f32(hi - lo + 1u)), hi);
}

@compute @workgroup_size(64)
fn cs_place(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    // The whole workgroup takes the same path up to the scan: a workgroup whose tile is
    // not valid still runs the loops (and their barriers) and only guards the writes.
    var slot = 0u;
    var first = 0u;
    var valid = false;
    if workgroup_id.x < place.place_tile_count && workgroup_id.x < arrayLength(&tile_queue) {
        let entry = tile_queue[workgroup_id.x];
        slot = entry.x;
        first = entry.y;
        valid = slot < arrayLength(&tiles);
    }
    var tile: FoliageTile;
    var frame: TileFrame;
    if valid {
        tile = tiles[slot];
        frame = frames[slot];
    }
    let band_index = foliage_band_of_slot(slot);
    let band = foliage_bands.bands[band_index];
    let generation = band_generation(tile.generation, band_index);
    let eye_h = dot(vec3<f32>(place.eye_x, place.eye_y, place.eye_z) - frame.origin, frame.up);
    let start_h = eye_h + place.search_above;
    let length = place.search_above + place.search_below;
    let cubes = cube_cells(frame);
    let coord = vec2<i32>(tile.tile_coord_x, tile.tile_coord_z);
    let blade_records = band.lod == FOLIAGE_BAND_LOD_BLADES;

    // The candidate count is the same for the whole workgroup (one tile), but it comes
    // from storage: broadcast it so the loops around the scan's barriers are uniform.
    if lane == 0u {
        wg_cells = band.candidate_grid * band.candidate_grid;
    }
    let cells = workgroupUniformLoad(&wg_cells);

    if lane == 0u {
        atomicStore(&wg_low, ordered(3.0e38));
        atomicStore(&wg_high, ordered(-3.0e38));
        atomicStore(&wg_live_types, 0u);
        wg_base = 0u;
    }
    workgroupBarrier();

    // The tile's height range along its up: records store a height *offset* from its
    // centre, so it is known before any is packed.
    for (var candidate = lane; candidate < cells; candidate += WG_SIZE) {
        let g = answered_ground(first + candidate, frame, start_h, length);
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

    // `cells` is a workgroup-uniform load, so this loop and every barrier inside it are in
    // uniform control flow for the whole workgroup.
    var chunk = 0u;
    loop {
        if chunk >= cells {
            break;
        }

        let candidate = chunk + lane;
        var count = 0u;
        var record = BladeInstance(0u, 0u, 0u, 0u);

        if valid && candidate < cells {
            let seed = blade_seed(coord.x, coord.y, candidate, generation);
            let uv = candidate_uv(candidate, seed, cubes, band);
            let g = answered_ground(first + candidate, frame, start_h, length);
            if g.ok && uv.x >= 0.0 && within_any_layer(frame_point(frame, uv, g.h)) {
                // The types that grow here, by their densities and patches: the band's
                // grid holds `grid_density` candidates a square metre for `share` of the
                // full density, so the ground grows a tuft with probability
                // `sum · share / grid_density`, of each type by its share of the sum.
                let slope_cos = dot(g.normal, frame.up);
                var sum = 0.0;
                for (var t = 0u; t < type_limit; t++) {
                    let ty = types[t];
                    if type_grows(ty, g.ground, slope_cos, g.h) {
                        sum += ty.density * patch_weight(ty, t, coord, uv, band.tile_span);
                    }
                }
                let wanted = sum * place.density_multiplier * band.share;
                let roll = hash_to_unit(rotl(seed, 17u)) * max(band.grid_density, 1.0e-6);
                if roll < wanted {
                    // Which type: the roll's place in the cumulative densities.
                    let pick = hash_to_unit(rotl(seed, 5u)) * sum;
                    var type_id = 0u;
                    var running = 0.0;
                    for (var t = 0u; t < type_limit; t++) {
                        let ty = types[t];
                        if !type_grows(ty, g.ground, slope_cos, g.h) { continue; }
                        type_id = t;
                        running += ty.density * patch_weight(ty, t, coord, uv, band.tile_span);
                        if pick < running { break; }
                    }
                    let blades_in_tuft = tuft_blades(types[type_id], hash_to_unit(rotl(seed, 3u)));
                    count = select(1u, blades_in_tuft, blade_records);
                    record.packed_pos = pack_pos(uv.x, uv.y);
                    record.packed_height_tuft = pack_height_tuft(g.h - center, 0u, blades_in_tuft, hash_to_unit(rotl(seed, 23u)));
                    record.packed_scale_type = pack_scale_type(
                        hash_to_unit(rotl(seed, 29u)),
                        (seed >> 9u) & 0xffu,
                        type_id,
                        (seed >> 30u) & 3u,
                    );
                    record.packed_colour_seed = pack_ground_colour(g.colour) | ((seed & 0xffffu) << 16u);
                }
            }
        }

        // Hillis-Steele inclusive scan over the 64 lanes' record counts.
        wg_scan[lane] = count;
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
        let exclusive = inclusive - count;
        let base = wg_base;
        if valid && count > 0u {
            for (var j = 0u; j < count; j++) {
                let local = base + exclusive + j;
                if local < band.slab && tile.blade_offset + local < arrayLength(&blades) {
                    var r = record;
                    r.packed_height_tuft = r.packed_height_tuft | (j << 16u);
                    blades[tile.blade_offset + local] = r;
                } else {
                    atomicAdd(&counters[COUNTER_PLACEMENT_OVERFLOW], 1u);
                }
            }
        }
        workgroupBarrier();
        if lane == WG_SIZE - 1u {
            wg_base = base + inclusive;
        }
        workgroupBarrier();
        chunk = chunk + WG_SIZE;
    }

    if lane == 0u && valid {
        let placed = min(wg_base, band.slab);
        tiles[slot].blade_count = placed;
        tiles[slot].state = TILE_STATE_RESIDENT;
        atomicAdd(&counters[COUNTER_PLACED_BLADES], placed);
    }
}
