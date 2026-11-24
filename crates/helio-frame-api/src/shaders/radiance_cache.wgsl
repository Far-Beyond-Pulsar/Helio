// The radiance cache (`helio_frame_api::rays::radiance_cache`,
// `//!use frame_radiance_cache`, with `//!use frame_ray_stream` for its
// packings): irradiance arriving at surface cells of the world, direct and
// indirect apart, kept by its publisher (the GI plugin's cache) and read by
// every ray that needs a lit hit.
//
// A shader that opts in also opts into `frame_surface_stops` (whether a
// reader lies on a cell's surface) and binds, under these names:
//   radiance_cache_frame:   var<uniform> RadianceCacheFrame
//   radiance_cache_slots:   var<storage, read_write> array<RadianceCacheSlot>
//   radiance_cache_entries: var<storage, read_write> array<RadianceCacheEntry>
// the publisher's buffers (`RADIANCE_CACHE_FRAME`, `_SLOTS`, `_ENTRIES`), or
// the stand-in (`radiance_cache::stand_in`), whose frame holds no slots: a
// read then finds nothing and a mark does nothing.
//
// A reader with a hit calls `radiance_cache_read(p, n, space, footprint)`
// (what the cache holds there, or nothing, and why) and
// `radiance_cache_mark(p, n, space, footprint)` (the cell is in use; a cell
// first used takes its point and normal from the hit), or
// `radiance_cache_read_mark(p, n, space, footprint)` for both with one find
// a level, with the hit's frame-space point, the unit normal of the side
// the ray arrived on (the face turned toward the ray), its coordinate space
// (`scene_ray_space`) and its ray's width there (`scene_ray_footprint`; 0
// for a reader as sharp as the eye angle). Every reader reads a cell alike: a
// cell's light is the prefiltered irradiance of its surface about its
// point, which is what a diffuse bounce gathers from there however far the
// reader's ray came (as a surface cache supplies it), and a reader beside an
// occluder the cell spans reads a finer cell. A Lambertian hit of albedo `a`
// then leaves `a * (direct + indirect) / PI`; one that lights itself with
// light samples (a mirror's hit, a wide reader near small lights) takes
// `indirect` alone: a cell's `direct` is its point's.
//
// Who marks is the cache's working set: a mark claims a cell where none is,
// and a cell no mark used for a while is evicted. A reader that only reads
// keeps nothing alive and claims nothing (its misses say ABSENT), so a
// publisher bounds its working set by which readers mark (GI: the pixels'
// gather alone, so the cells in use are what the screen's rays reach).
//
// A read's level is the coarser of the surface's (the eye angle's cell at
// its distance from the eye, `radiance_cache_level`, as SHaRC and GI-1.0
// key their hash grids) and the reader's footprint at the hit
// (`radiance_cache_level_at`: a wide ray takes the light of the patch its
// cone covers, as Lumen reads its surface cache at the mip of a cone's
// footprint). A cell finer than a reader's footprint is detail its estimate
// averages away, and a cell a wide reader's rays would claim and rarely meet
// again; a sharp reader (a short ray's hit) reads the eye angle's. A surface
// point holds a cell at each level its readers' footprints round to: at
// most log2(longest ray's footprint / eye cell) + 1 levels, and the levels
// coarser than the eye angle's hold a quarter of the cells of the one
// finer, so the finest, which only short rays reach, bounds the count.
// Finer cells exist only next to an occluder that splits a cell
// (RADIANCE_CACHE_DESCENT): a coarser cell spans thin walls and floors under
// them and holds its light on its point's side.
//
// A cell is a slab lying on its surface: as wide as its level along the two
// axes across its side, a RADIANCE_CACHE_LAYERS-th of that along its side's
// own axis. Relief thinner than a layer (a step, a bump, a stone) is the
// cell's surface, prefiltered; a parallel surface farther off along the
// side's axis (a slab's top and the floor under it, a wall's top and the
// floor beside it) is in another layer, another cell at the same level. A
// surface's relief over a cell's width claims a cell a layer it spans, at
// its own level.

// Levels of cells: level `l` is `2^(l + RADIANCE_CACHE_FINEST)` metres wide,
// 1/64 m to 1024 m. The coarsest is the frame origin's step (a multiple of
// 1024 m), so the origin is a whole number of cells at every level.
const RADIANCE_CACHE_LEVELS: u32 = 17u;
const RADIANCE_CACHE_FINEST: i32 = -6;
// Slots a bucket holds. A key sits in the bucket its hash names or, where
// that was full when it was claimed, the next one: a key whose two buckets
// are both full at half load is about one in a million (one bucket alone
// turned away 0.4% of keys, and a surface has one level, so a key turned
// away is a place no reader finds light at).
const RADIANCE_CACHE_BUCKET: u32 = 16u;
// No slot.
const RADIANCE_CACHE_NONE: u32 = 0xffffffffu;
// How many levels finer than a surface's own a reader may read (and its mark
// claim) where the surface's cell does not hold for it (an occluder between
// the cell's point and the hit, a surface turned away from the cell's): the
// cell a blocked mark claims is seated at its own hit, so one level finer
// holds unless a reader past the same occluder seated it first; three levels
// (an eighth of the cell) is as finely as a cell's clear distances place an
// occluder (a way is 45 degrees wide, its bound within 1 - cos 22.5
// degrees, 8%, of the distance, at most a cell's).
const RADIANCE_CACHE_DESCENT: u32 = 3u;
// The layers a cell's width holds along its side's axis (a cell's thickness
// is its width over this): as thin as the finest cell a read descends to,
// the finest the cell's own rays resolve, so relief thinner than that is
// the surface prefiltered and a surface farther off is another.
const RADIANCE_CACHE_LAYERS: u32 = 8u;
// Where a layer starts within its thickness: layers start at
// `(k - RADIANCE_CACHE_PHASE)` thicknesses along the side's axis, so a
// surface on the grid the levels are cut from (a floor at 0 m or 2 m: 0.38
// of a thickness inside) or on a decimal one (10 cm voxel faces: 0.07% to 2%
// of a thickness from a boundary for thicknesses of 16 m to 1/64 m, far past
// its hits' rounding at the distances whose cells are that thick) sits
// inside a layer, not on a boundary where its hits' rounding
// would split it between two cells (2 - the golden ratio: no dyadic or
// decimal fraction lands on it).
const RADIANCE_CACHE_PHASE: f32 = 0.381966;
// The new cells' queue, after the hash's `slots` slots: its count in
// `radiance_cache_slots[slots].state`, then a place a new cell
// (`radiance_cache_slots[slots + 1 + i].state`: the cell's slot, or
// RADIANCE_CACHE_NONE where the mark that reserved it claimed nothing),
// `slots / RADIANCE_CACHE_QUEUE_SHARE` places. The publisher takes the
// cells in and empties it once a frame, so it walks the cells in use and
// never the slots; a mark that finds it full claims nothing (its read finds
// nothing until a later frame's mark claims it).
const RADIANCE_CACHE_QUEUE_SHARE: u32 = 4u;
// A cell's clear distance (`RadianceCacheEntry::clear`) along a way its
// rays found open: as far as f16 holds.
const RADIANCE_CACHE_FAR: f32 = 65504.0;
// How near its point, in cells, a cell's light holds along a way its rays
// have not explored yet: the point's own neighbourhood, thinner than any
// wall a cell is wide enough to span.
const RADIANCE_CACHE_NEAR: f32 = 0.03125;
// The directions along a cell's surface its clear distances are kept in,
// each 45 degrees about its axis: a wall at an angle to a way bounds it no
// nearer than 0.92 (cos 22.5 degrees) of its distance.
const RADIANCE_CACHE_WAYS: u32 = 8u;
// A cell's `point` while it has none: its rays found the point it was seated
// at inside a solid (the surface it was taken on is no longer there: the
// terrain streamed in finer, an edit, a finer level of detail drawn), so it
// holds no value and the next mark seats it at that mark's hit.
const RADIANCE_CACHE_UNSEATED: u32 = 0xffffffffu;

// The frame's cache. `origin_xy`, `origin_z`: the frame's origin in whole
// 1024 m (64-bit two's complement a component: x lo, x hi, y lo, y hi; z lo,
// z hi), then the slots (0: no cache) and the frame's number. `offset`: the
// origin's rest below 1024 m (0 for an engine frame, whose origin is a
// multiple of 1024 m), and in w the eye angle (a cell is at least that
// angle wide seen from the eye). `eye`: the eye in the frame's space.
struct RadianceCacheFrame {
    origin_xy: vec4<u32>,
    origin_z: vec4<u32>,
    offset: vec4<f32>,
    eye: vec4<f32>,
}

// A slot of the hash: `state` the key's checksum (0: empty), `used` the
// frame a reader last marked it.
struct RadianceCacheSlot {
    state: atomic<u32>,
    used: atomic<u32>,
}

// A cell: `tag` its level (bits 0..5), the dominant side of its normal
// (5..8: axis * 2 + negative) and the coordinate space (8..17,
// `SceneRay::space`'s encoding); the world cell, 64 bits an axis: in the
// level's cells across its side, in its layers along the side's axis
// (`radiance_cache_key`).
struct RadianceCacheKey {
    tag: u32,
    x_lo: u32,
    x_hi: u32,
    y_lo: u32,
    y_hi: u32,
    z_lo: u32,
    z_hi: u32,
}

// A slot's cell, 96 bytes: its key; the point its light is gathered at, as a
// 15-bit unorm an axis in the cell (its layer along the side's axis: the
// key's units, `radiance_cache_place`; `point` its top 10 bits an axis, 10:10:10;
// `history` bits 16..31 its low 5, 5:5:5) and that side's normal
// (`scene_pack_unit`); `history` bits 0..16: the updates folded into the
// direct part (0..8) and the indirect part (8..16), 0 for a cell without a
// value yet; the irradiance arriving there, W / m^2: from the lights
// (`direct`), from what the cell's rays meet as they meet it (`indirect`:
// what surfaces emit and reflect of the lights' light, the sky) and from
// the light those surfaces reflect of the cache's own (`bounced`: the later
// bounces); `bounced_history`: the updates folded into `bounced` (0..8);
// `clear`: how far from the cell's point, along its surface, its rays found
// nothing in the way, in RADIANCE_CACHE_WAYS directions
// (`radiance_cache_holds`: f16 pairs, metres; RADIANCE_CACHE_FAR open, 0 not
// explored yet). A read's indirect light
// is `indirect + bounced`.
struct RadianceCacheEntry {
    key: RadianceCacheKey,
    point: u32,
    normal: u32,
    history: u32,
    direct: array<f32, 3>,
    indirect: array<f32, 3>,
    bounced: array<f32, 3>,
    bounced_history: u32,
    clear: array<u32, 4>,
}

// What a read found: the irradiance at the cell, the level it came from (for
// a miss, the level the read stopped at) and the cell's history (its updates folded: direct, indirect; how long its
// value has held still); where it found nothing, why (`miss`,
// RADIANCE_CACHE_MISS_*); `claimed`, this read's mark claimed the cell it
// stopped at (a new cell of the cache's working set).
struct RadianceCacheIrradiance {
    direct: vec3<f32>,
    indirect: vec3<f32>,
    level: u32,
    found: bool,
    history: vec2<u32>,
    miss: u32,
    claimed: bool,
}

// Why a read found nothing (`RadianceCacheIrradiance::miss`). The first
// three pass: the read's own mark claims the cell, the cache lights it, its
// rays explore its ways, so a later frame's read of the same place finds
// light. The last stays as long as the cells do: the cell of the place's
// level and every finer one up to RADIANCE_CACHE_DESCENT hold their light on
// the other side of what is between (`OCCLUDED`: a thin wall, or a contact
// finer than the finest of them). A reader stands in for it nothing that
// cannot tell an occluder's two sides apart.
const RADIANCE_CACHE_MISS_NONE: u32 = 0u;
// No slot holds the cell: not marked yet, or its key turned away (two full
// buckets, a full queue) until a slot frees.
const RADIANCE_CACHE_MISS_ABSENT: u32 = 1u;
// The cell holds no value yet (new, or unseated).
const RADIANCE_CACHE_MISS_NEW: u32 = 2u;
// The cell's rays have not explored the way to the point yet.
const RADIANCE_CACHE_MISS_UNEXPLORED: u32 = 3u;
const RADIANCE_CACHE_MISS_OCCLUDED: u32 = 4u;

// Whether miss `m` stays (RADIANCE_CACHE_MISS_OCCLUDED).
fn radiance_cache_miss_stays(m: u32) -> bool {
    return m == RADIANCE_CACHE_MISS_OCCLUDED;
}

// Whether the frame has a cache.
fn radiance_cache_present() -> bool {
    return radiance_cache_frame.origin_z.z != 0u;
}

// The width of level `level`'s cells (metres) and its inverse.
fn radiance_cache_size(level: u32) -> f32 {
    return exp2(f32(i32(level) + RADIANCE_CACHE_FINEST));
}

fn radiance_cache_inv_size(level: u32) -> f32 {
    return exp2(-f32(i32(level) + RADIANCE_CACHE_FINEST));
}

// The level of frame point `p`'s surface (every reader's): the cell as wide
// as the eye angle at its distance from the eye, the nearest power of two.
fn radiance_cache_level(p: vec3<f32>) -> u32 {
    let d = length(p - radiance_cache_frame.eye.xyz);
    let size = radiance_cache_frame.offset.w * d;
    let l = round(log2(max(size, 1e-30))) - f32(RADIANCE_CACHE_FINEST);
    return u32(clamp(l, 0.0, f32(RADIANCE_CACHE_LEVELS - 1u)));
}

// The level a reader whose ray is `footprint` wide at frame point `p` reads
// there: the surface's (`radiance_cache_level`), or the footprint's, the
// nearest power of two, where that is coarser.
fn radiance_cache_level_at(p: vec3<f32>, footprint: f32) -> u32 {
    let l = round(log2(max(footprint, 1e-30))) - f32(RADIANCE_CACHE_FINEST);
    return max(radiance_cache_level(p), u32(clamp(l, 0.0, f32(RADIANCE_CACHE_LEVELS - 1u))));
}

// The dominant side of unit `n`: axis * 2, plus 1 toward the negative.
fn radiance_cache_side(n: vec3<f32>) -> u32 {
    let a = abs(n);
    if a.x >= a.y && a.x >= a.z { return select(0u, 1u, n.x < 0.0); }
    if a.y >= a.z { return select(2u, 3u, n.y < 0.0); }
    return select(4u, 5u, n.z < 0.0);
}

// A 64-bit `lo`, `hi` shifted left by `s` (0..31).
fn radiance_cache_shl(lo: u32, hi: u32, s: u32) -> vec2<u32> {
    if s == 0u { return vec2<u32>(lo, hi); }
    return vec2<u32>(lo << s, (hi << s) | (lo >> (32u - s)));
}

// A 64-bit `a` plus 32-bit signed `b`.
fn radiance_cache_add(a: vec2<u32>, b: i32) -> vec2<u32> {
    let lo = a.x + bitcast<u32>(b);
    let carry = select(0u, 1u, lo < a.x);
    return vec2<u32>(lo, a.y + select(0u, 0xffffffffu, b < 0) + carry);
}

// The frame origin's place in a key of `level` on side `side`: whole
// 1024 m times the cells in 1024 m across the side, times the layers along
// its axis.
fn radiance_cache_origin_cell(level: u32, side: u32) -> array<vec2<u32>, 3> {
    let s = RADIANCE_CACHE_LEVELS - 1u - level;
    let a = side >> 1u;
    let layer = countTrailingZeros(RADIANCE_CACHE_LAYERS);
    let o = radiance_cache_frame.origin_xy;
    let z = radiance_cache_frame.origin_z;
    return array<vec2<u32>, 3>(
        radiance_cache_shl(o.x, o.y, s + select(0u, layer, a == 0u)),
        radiance_cache_shl(o.z, o.w, s + select(0u, layer, a == 1u)),
        radiance_cache_shl(z.x, z.y, s + select(0u, layer, a == 2u)),
    );
}

// Frame point `p` in level `level`'s cells (frame-local).
fn radiance_cache_cells(p: vec3<f32>, level: u32) -> vec3<f32> {
    return (p + radiance_cache_frame.offset.xyz) * radiance_cache_inv_size(level);
}

// Frame point `p` in a key's units at `level` on side `side` (frame-local):
// the level's cells across the side, its layers along the side's axis
// (from RADIANCE_CACHE_PHASE).
fn radiance_cache_place(p: vec3<f32>, level: u32, side: u32) -> vec3<f32> {
    let q = radiance_cache_cells(p, level);
    let a = side >> 1u;
    let l = f32(RADIANCE_CACHE_LAYERS);
    return vec3<f32>(
        select(q.x, q.x * l + RADIANCE_CACHE_PHASE, a == 0u),
        select(q.y, q.y * l + RADIANCE_CACHE_PHASE, a == 1u),
        select(q.z, q.z * l + RADIANCE_CACHE_PHASE, a == 2u),
    );
}

// The key of the cell holding frame point `p` at `level`, on the side of
// unit `n`, in coordinate space `space`.
fn radiance_cache_key(p: vec3<f32>, n: vec3<f32>, level: u32, space: u32) -> RadianceCacheKey {
    let side = radiance_cache_side(n);
    let c = vec3<i32>(floor(radiance_cache_place(p, level, side)));
    let o = radiance_cache_origin_cell(level, side);
    let x = radiance_cache_add(o[0], c.x);
    let y = radiance_cache_add(o[1], c.y);
    let z = radiance_cache_add(o[2], c.z);
    let tag = level | (side << 5u) | ((space & 0x1ffu) << 8u);
    return RadianceCacheKey(tag, x.x, x.y, y.x, y.y, z.x, z.y);
}

fn radiance_cache_mix(x0: u32) -> u32 {
    var x = x0;
    x ^= x >> 16u;
    x *= 0x7feb352du;
    x ^= x >> 15u;
    x *= 0x846ca68bu;
    x ^= x >> 16u;
    return x;
}

fn radiance_cache_hash(k: RadianceCacheKey, seed: u32) -> u32 {
    var h = radiance_cache_mix(seed ^ k.tag);
    h = radiance_cache_mix(h ^ k.x_lo);
    h = radiance_cache_mix(h ^ k.x_hi);
    h = radiance_cache_mix(h ^ k.y_lo);
    h = radiance_cache_mix(h ^ k.y_hi);
    h = radiance_cache_mix(h ^ k.z_lo);
    return radiance_cache_mix(h ^ k.z_hi);
}

// The first slot of `k`'s bucket.
fn radiance_cache_bucket(k: RadianceCacheKey) -> u32 {
    let buckets = radiance_cache_frame.origin_z.z / RADIANCE_CACHE_BUCKET;
    return (radiance_cache_hash(k, 0x9e3779b9u) & (buckets - 1u)) * RADIANCE_CACHE_BUCKET;
}

// `k`'s checksum, a slot's state (never 0).
fn radiance_cache_checksum(k: RadianceCacheKey) -> u32 {
    return max(radiance_cache_hash(k, 0x85ebca6bu), 1u);
}

fn radiance_cache_same(a: RadianceCacheKey, b: RadianceCacheKey) -> bool {
    return a.tag == b.tag && a.x_lo == b.x_lo && a.x_hi == b.x_hi && a.y_lo == b.y_lo
        && a.y_hi == b.y_hi && a.z_lo == b.z_lo && a.z_hi == b.z_hi;
}

// The slot holding cell `k` (in its bucket or the next), or
// RADIANCE_CACHE_NONE.
fn radiance_cache_find(k: RadianceCacheKey) -> u32 {
    let base = radiance_cache_bucket(k);
    let check = radiance_cache_checksum(k);
    let wrap = radiance_cache_frame.origin_z.z - 1u;
    for (var i = 0u; i < 2u * RADIANCE_CACHE_BUCKET; i++) {
        let s = (base + i) & wrap;
        if atomicLoad(&radiance_cache_slots[s].state) == check && radiance_cache_same(radiance_cache_entries[s].key, k) {
            return s;
        }
    }
    return RADIANCE_CACHE_NONE;
}

// Whether entry `e` holds a value.
fn radiance_cache_has_value(e: RadianceCacheEntry) -> bool {
    return (e.history & 0xffffu) != 0u;
}

// An orthonormal frame about unit `n` (Duff et al. 2017): the directions
// along the surface a cell's clear distances are kept in.
fn radiance_cache_tangents(n: vec3<f32>) -> array<vec3<f32>, 2> {
    let sign = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    return array<vec3<f32>, 2>(
        vec3<f32>(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x),
        vec3<f32>(b, sign + n.y * n.y * a, -n.y),
    );
}

// Offset `v` along the surface of unit normal `n`: its way (the nearest of
// RADIANCE_CACHE_WAYS directions about `radiance_cache_tangents`' first,
// counterclockwise) in x, its length along the surface in y.
fn radiance_cache_along(n: vec3<f32>, v: vec3<f32>) -> vec2<f32> {
    let t = radiance_cache_tangents(n);
    let a = vec2<f32>(dot(v, t[0]), dot(v, t[1]));
    let turn = atan2(a.y, a.x) * (f32(RADIANCE_CACHE_WAYS) / 6.2831853);
    let way = u32(round(turn) + f32(RADIANCE_CACHE_WAYS)) % RADIANCE_CACHE_WAYS;
    return vec2<f32>(f32(way), length(a));
}

// Entry `e`'s clear distance along way `w`.
fn radiance_cache_clear(e: RadianceCacheEntry, w: u32) -> f32 {
    let pair = unpack2x16float(e.clear[w >> 1u]);
    return select(pair.x, pair.y, (w & 1u) != 0u);
}

// What entry `e`'s light is at frame point `p` (the reader's side `n`). A
// cell wider than an occluder beside it (a wall on a floor, a slab against
// a wall) gathers its light on one side of it; a surface parallel to the
// cell's farther off than a layer is in another cell (its layer), so a
// reader of this cell lies on its surface to within a layer:
// - RADIANCE_CACHE_HOLDS: `p` lies on the cell's surface (the frame's
//   surface stops between its point and `p`, `surface_stops_weight`: the
//   normal stop, and the plane stop to a layer's thickness, the relief a
//   cell prefilters), and the way from the cell's point to `p` along it is
//   shorter than its rays found clear that way (a way they have not
//   explored: `p` within RADIANCE_CACHE_NEAR of its point);
// - RADIANCE_CACHE_BLOCKED: `p` is on a surface turned away from the
//   cell's, or past what its rays met that way: a finer cell is `p`'s;
// - RADIANCE_CACHE_UNKNOWN: its rays have not explored that way yet: it has
//   no light for `p` until they have (a few updates).
const RADIANCE_CACHE_HOLDS: u32 = 0u;
const RADIANCE_CACHE_BLOCKED: u32 = 1u;
const RADIANCE_CACHE_UNKNOWN: u32 = 2u;

fn radiance_cache_light_at(e: RadianceCacheEntry, p: vec3<f32>, n: vec3<f32>) -> u32 {
    let point = radiance_cache_entry_point(e);
    let normal = scene_unpack_unit(e.normal);
    let along = radiance_cache_along(normal, p - point);
    let size = radiance_cache_size(e.key.tag & 31u);
    let near = RADIANCE_CACHE_NEAR * size;
    if surface_stops_weight(point, normal, p, n, size / f32(RADIANCE_CACHE_LAYERS)) <= 0.0 { return RADIANCE_CACHE_BLOCKED; }
    let clear = radiance_cache_clear(e, u32(along.x));
    if clear == 0.0 {
        return select(RADIANCE_CACHE_UNKNOWN, RADIANCE_CACHE_HOLDS, along.y <= near);
    }
    return select(RADIANCE_CACHE_BLOCKED, RADIANCE_CACHE_HOLDS, along.y <= clear);
}

// Whether entry `e`'s light holds at frame point `p` (side `n`,
// `radiance_cache_light_at`).
fn radiance_cache_holds(e: RadianceCacheEntry, p: vec3<f32>, n: vec3<f32>) -> bool {
    return radiance_cache_light_at(e, p, n) == RADIANCE_CACHE_HOLDS;
}

// The walk a read and a mark share, at frame point `p` on the side of unit
// `n` (the face turned toward the ray) in coordinate space `space` for a
// reader `footprint` wide there: the cell of the read's level
// (`radiance_cache_level_at`), then, while a cell's light is blocked at `p`
// (`radiance_cache_light_at`: an occluder between its point and `p`, a
// surface turned away from its own), the next finer, at most
// RADIANCE_CACHE_DESCENT levels finer. Each level's slot is found (`mark`
// false) or found and marked used, claimed where no slot holds it (`mark`
// true, `radiance_cache_mark_key`): one find a level either way.
//
// What it found: the cell whose light holds at `p`; `found` false, and
// `miss` why, where there is no cell (ABSENT; for a mark, its claim turned
// away), it has no value yet (NEW: a cell this walk claimed, or one claimed
// this frame), its rays have not explored the way to `p` (UNEXPLORED: the
// cell's light comes with it, which holds at its point; whether anything
// along its surface stands between that and `p` is not known for a few of
// its updates), or no cell up to the finest holds (OCCLUDED).
fn radiance_cache_walk(p: vec3<f32>, n: vec3<f32>, space: u32, footprint: f32, mark: bool) -> RadianceCacheIrradiance {
    return radiance_cache_walk_from(p, n, space, radiance_cache_level_at(p, footprint), mark);
}

// `radiance_cache_walk` from level `level`.
fn radiance_cache_walk_from(p: vec3<f32>, n: vec3<f32>, space: u32, level: u32, mark: bool) -> RadianceCacheIrradiance {
    var none = RadianceCacheIrradiance(vec3<f32>(0.0), vec3<f32>(0.0), 0u, false, vec2<u32>(0u), RADIANCE_CACHE_MISS_ABSENT, false);
    if !radiance_cache_present() { return none; }
    let finest = level - min(level, RADIANCE_CACHE_DESCENT);
    var at = level;
    loop {
        let key = radiance_cache_key(p, n, at, space);
        none.level = at;
        var s = RADIANCE_CACHE_NONE;
        if mark {
            let m = radiance_cache_mark_key(key, p, n);
            s = m.x;
            none.claimed = m.y != 0u;
        } else {
            s = radiance_cache_find(key);
        }
        if s == RADIANCE_CACHE_NONE { return none; }
        let e = radiance_cache_entries[s];
        // A key another reader claims this frame may not be written yet.
        if !radiance_cache_same(e.key, key) || !radiance_cache_has_value(e) {
            none.miss = RADIANCE_CACHE_MISS_NEW;
            return none;
        }
        let light = radiance_cache_light_at(e, p, n);
        if light != RADIANCE_CACHE_BLOCKED {
            let miss = select(RADIANCE_CACHE_MISS_NONE, RADIANCE_CACHE_MISS_UNEXPLORED, light == RADIANCE_CACHE_UNKNOWN);
            return RadianceCacheIrradiance(
                vec3<f32>(e.direct[0], e.direct[1], e.direct[2]),
                vec3<f32>(e.indirect[0] + e.bounced[0], e.indirect[1] + e.bounced[1], e.indirect[2] + e.bounced[2]),
                at,
                miss == RADIANCE_CACHE_MISS_NONE,
                vec2<u32>(e.history & 0xffu, (e.history >> 8u) & 0xffu),
                miss,
                false,
            );
        }
        if at == finest {
            none.miss = RADIANCE_CACHE_MISS_OCCLUDED;
            return none;
        }
        at -= 1u;
    }
    return none;
}

// The irradiance the cache holds at frame point `p` on the side of unit `n`
// (the face turned toward the ray) in coordinate space `space`, for a
// reader `footprint` wide there, or nothing and why (`radiance_cache_walk`);
// it marks nothing. Where no slot holds the cell of the reader's level
// (ABSENT), the next finer down to the surface's (`radiance_cache_level`):
// the readers that mark, whose cells these are, may be sharper than this
// one there (a finer cell is the same light, less prefiltered), and one
// that only reads claims none of its own.
fn radiance_cache_read(p: vec3<f32>, n: vec3<f32>, space: u32, footprint: f32) -> RadianceCacheIrradiance {
    let surface = radiance_cache_level(p);
    var at = radiance_cache_level_at(p, footprint);
    loop {
        let r = radiance_cache_walk_from(p, n, space, at, false);
        if r.miss != RADIANCE_CACHE_MISS_ABSENT || at <= surface { return r; }
        at -= 1u;
    }
    return radiance_cache_walk_from(p, n, space, surface, false);
}

// Marks the cells a read of frame point `p` (side of unit `n`, coordinate
// space `space`, a reader `footprint` wide) goes to as used this frame
// (`radiance_cache_walk`): the read's level's and, where a cell's light is
// blocked at `p`, the finer ones, down to the first that holds or has no
// value yet.
fn radiance_cache_mark(p: vec3<f32>, n: vec3<f32>, space: u32, footprint: f32) {
    _ = radiance_cache_walk(p, n, space, footprint, true);
}

// `radiance_cache_read` and `radiance_cache_mark` at once, with one find a
// level: what a reader that does both calls. A cell this read claims holds
// no value yet (RADIANCE_CACHE_MISS_NEW, where a read before the mark
// would find it ABSENT): ABSENT is then only a key the hash turned away.
fn radiance_cache_read_mark(p: vec3<f32>, n: vec3<f32>, space: u32, footprint: f32) -> RadianceCacheIrradiance {
    return radiance_cache_walk(p, n, space, footprint, true);
}

// Frame point `p`'s place in its cell at `level` on side `side` (a key's
// units, `radiance_cache_place`), 15 bits an axis: the entry's `point` (the
// top 10 bits, 10:10:10) and the bits of its `history` from 16 (the low 5,
// 5:5:5).
fn radiance_cache_pack_point(p: vec3<f32>, level: u32, side: u32) -> vec2<u32> {
    let q = radiance_cache_place(p, level, side);
    let u = vec3<u32>(clamp((q - floor(q)) * 32768.0, vec3<f32>(0.0), vec3<f32>(32767.0)));
    let high = u >> vec3<u32>(5u);
    let low = u & vec3<u32>(31u);
    return vec2<u32>(high.x | (high.y << 10u) | (high.z << 20u), (low.x | (low.y << 5u) | (low.z << 10u)) << 16u);
}

// Entry `e`'s point, in the frame's space (within 1/32768 of its cell an
// axis, of its layer along its side's axis, of where the hit was).
fn radiance_cache_entry_point(e: RadianceCacheEntry) -> vec3<f32> {
    let level = e.key.tag & 31u;
    let side = (e.key.tag >> 5u) & 7u;
    let o = radiance_cache_origin_cell(level, side);
    let c = vec3<f32>(vec3<i32>(bitcast<i32>(e.key.x_lo - o[0].x), bitcast<i32>(e.key.y_lo - o[1].x), bitcast<i32>(e.key.z_lo - o[2].x)));
    let high = vec3<u32>(e.point & 1023u, (e.point >> 10u) & 1023u, (e.point >> 20u) & 1023u);
    let fine = e.history >> 16u;
    let low = vec3<u32>(fine & 31u, (fine >> 5u) & 31u, (fine >> 10u) & 31u);
    let q = c + (vec3<f32>((high << vec3<u32>(5u)) | low) + 0.5) / 32768.0;
    // Back from the key's units to the level's cells.
    let a = side >> 1u;
    let l = f32(RADIANCE_CACHE_LAYERS);
    let cells = vec3<f32>(
        select(q.x, (q.x - RADIANCE_CACHE_PHASE) / l, a == 0u),
        select(q.y, (q.y - RADIANCE_CACHE_PHASE) / l, a == 1u),
        select(q.z, (q.z - RADIANCE_CACHE_PHASE) / l, a == 2u),
    );
    return cells * radiance_cache_size(level) - radiance_cache_frame.offset.xyz;
}

// Marks cell `key` as used this frame by a hit at frame point `p` (side
// `n`): its slot, found in its bucket or the next, or claimed where no slot
// holds it (an empty slot of its bucket, or of the next where that is full;
// its point and normal this hit's, a place in the new cells' queue).
// RADIANCE_CACHE_NONE where two full buckets or a full queue keep it out (a
// read then finds nothing); in y, 1 where this mark claimed it.
fn radiance_cache_mark_key(key: RadianceCacheKey, p: vec3<f32>, n: vec3<f32>) -> vec2<u32> {
    let base = radiance_cache_bucket(key);
    let check = radiance_cache_checksum(key);
    let frame = radiance_cache_frame.origin_z.w;
    let slots = radiance_cache_frame.origin_z.z;
    let wrap = slots - 1u;
    // A key claimed in the next bucket while its own was full stays there
    // once its own has room again: both are looked through before a claim.
    for (var k = 0u; k < 2u * RADIANCE_CACHE_BUCKET; k++) {
        let s = (base + k) & wrap;
        if atomicLoad(&radiance_cache_slots[s].state) == check {
            radiance_cache_used(s, key, p, n, frame);
            return vec2<u32>(s, 0u);
        }
    }
    var i = 0u;
    var retry = 0u;
    // This mark's place in the queue, once reserved.
    var place = RADIANCE_CACHE_NONE;
    var claimed = RADIANCE_CACHE_NONE;
    loop {
        if i >= 2u * RADIANCE_CACHE_BUCKET { break; }
        let s = (base + i) & wrap;
        var state = atomicLoad(&radiance_cache_slots[s].state);
        if state == 0u {
            if place == RADIANCE_CACHE_NONE {
                let queued = slots / RADIANCE_CACHE_QUEUE_SHARE;
                if atomicLoad(&radiance_cache_slots[slots].state) >= queued { return vec2<u32>(RADIANCE_CACHE_NONE, 0u); }
                let q = atomicAdd(&radiance_cache_slots[slots].state, 1u);
                if q >= queued { return vec2<u32>(RADIANCE_CACHE_NONE, 0u); }
                place = slots + 1u + q;
            }
            let claim = atomicCompareExchangeWeak(&radiance_cache_slots[s].state, 0u, check);
            if claim.exchanged {
                let at = radiance_cache_pack_point(p, key.tag & 31u, (key.tag >> 5u) & 7u);
                radiance_cache_entries[s] = RadianceCacheEntry(key, at.x, scene_pack_unit(n), at.y, array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0), 0u, array<u32, 4>(0u, 0u, 0u, 0u));
                atomicStore(&radiance_cache_slots[s].used, frame);
                atomicStore(&radiance_cache_slots[place].state, s);
                return vec2<u32>(s, 1u);
            }
            // A weak exchange may fail with the slot still empty: once more.
            if claim.old_value == 0u && retry == 0u {
                retry = 1u;
                continue;
            }
            state = claim.old_value;
        }
        retry = 0u;
        if state == check {
            // Another mark claimed it meanwhile.
            radiance_cache_used(s, key, p, n, frame);
            claimed = s;
            break;
        }
        i++;
    }
    // A place reserved for a claim another mark won: nothing in it.
    if place != RADIANCE_CACHE_NONE { atomicStore(&radiance_cache_slots[place].state, RADIANCE_CACHE_NONE); }
    return vec2<u32>(claimed, 0u);
}

// Slot `s`'s cell (`key`) used on frame `frame` by a mark of frame point `p`
// (side `n`). An unseated cell takes this hit's point and normal, with no
// value and its ways unexplored (two marks at once may mix their points'
// bits: a point still in the cell, which its rays then judge again).
fn radiance_cache_used(s: u32, key: RadianceCacheKey, p: vec3<f32>, n: vec3<f32>, frame: u32) {
    if radiance_cache_entries[s].point == RADIANCE_CACHE_UNSEATED {
        let at = radiance_cache_pack_point(p, key.tag & 31u, (key.tag >> 5u) & 7u);
        radiance_cache_entries[s].normal = scene_pack_unit(n);
        radiance_cache_entries[s].history = at.y;
        radiance_cache_entries[s].clear = array<u32, 4>(0u, 0u, 0u, 0u);
        radiance_cache_entries[s].point = at.x;
    }
    if atomicLoad(&radiance_cache_slots[s].used) != frame {
        atomicStore(&radiance_cache_slots[s].used, frame);
    }
}
