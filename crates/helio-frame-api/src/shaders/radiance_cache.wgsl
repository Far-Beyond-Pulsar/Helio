// The radiance cache (`helio_frame_api::rays::radiance_cache`,
// `//!use frame_radiance_cache`, with `//!use frame_ray_stream` for its
// packings): irradiance arriving at surface cells of the world, direct and
// indirect apart, kept by its publisher (the GI plugin's cache) and read by
// every ray that needs a lit hit.
//
// A shader that opts in binds, under these names:
//   radiance_cache_frame:   var<uniform> RadianceCacheFrame
//   radiance_cache_slots:   var<storage, read_write> array<RadianceCacheSlot>
//   radiance_cache_entries: var<storage, read_write> array<RadianceCacheEntry>
// the publisher's buffers (`RADIANCE_CACHE_FRAME`, `_SLOTS`, `_ENTRIES`), or
// the stand-in (`radiance_cache::stand_in`), whose frame holds no slots: a
// read then finds nothing and a mark does nothing.
//
// A reader with a hit calls `radiance_cache_read(p, n, footprint, space,
// coarser, reach)` (what the cache holds there, or nothing) and
// `radiance_cache_mark(p, n, footprint, space, reach)` (the cell is in use; a
// cell first used takes its point and normal from the hit), with the hit's
// frame-space point, the unit normal of the side the ray arrived on (the face
// turned toward the ray), the ray's footprint at the hit
// (`scene_ray_footprint`), its coordinate space (`scene_ray_space`), how many
// levels coarser than the hit's own cell may answer while it has no value (a
// coarser cell spans more of the scene, and across a thin occluder its light
// is both sides'; RADIANCE_CACHE_LEVELS: any) and the reader's reach: how far
// from the hit, along its surface, a cell's point may stand for it (metres).
// A ray that left a surface passes its length there (the hit distance): the
// surface it left is in the hit's way at that distance, so light gathered
// farther off is light the hit does not see from where it is (the floor
// beyond a contact, the lit side of a thin wall); a reader in the air (a
// probe) passes RADIANCE_CACHE_FAR. A Lambertian hit of albedo `a` then
// leaves `a * (direct + indirect) / PI`; one that lights itself with light
// samples (a mirror's hit) takes `indirect` alone.

// Levels of cells: level `l` is `2^(l + RADIANCE_CACHE_FINEST)` metres wide,
// 1/64 m to 1024 m. The coarsest is the frame origin's step (a multiple of
// 1024 m), so the origin is a whole number of cells at every level.
const RADIANCE_CACHE_LEVELS: u32 = 17u;
const RADIANCE_CACHE_FINEST: i32 = -6;
// Slots a key may sit in: the bucket its hash names.
const RADIANCE_CACHE_BUCKET: u32 = 16u;
// No slot.
const RADIANCE_CACHE_NONE: u32 = 0xffffffffu;
// A mark also marks the cell this many levels coarser, so a fine cell's
// read has a coarser one to answer from while it has no value of its own.
const RADIANCE_CACHE_ANCESTOR: u32 = 3u;
// The new cells' queue, after the hash's `slots` slots: its count in
// `radiance_cache_slots[slots].state`, then a place a new cell
// (`radiance_cache_slots[slots + 1 + i].state`: the cell's slot, or
// RADIANCE_CACHE_NONE where the mark that reserved it claimed nothing),
// `slots / RADIANCE_CACHE_QUEUE_SHARE` places. The publisher takes the
// cells in and empties it once a frame, so it walks the cells in use and
// never the slots; a mark that finds it full claims nothing (its read
// answers from a coarser cell until a later frame's mark claims it).
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
// `SceneRay::space`'s encoding); the world cell, 64 bits an axis.
struct RadianceCacheKey {
    tag: u32,
    x_lo: u32,
    x_hi: u32,
    y_lo: u32,
    y_hi: u32,
    z_lo: u32,
    z_hi: u32,
}

// A slot's cell, 80 bytes: its key; the point its light is gathered at, as a
// 15-bit unorm an axis in the cell (`point` its top 10 bits an axis, 10:10:10;
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

// What a read found: the irradiance at the cell, the level it came from and
// the cell's history (its updates folded: direct, indirect; how long its
// value has held still).
struct RadianceCacheIrradiance {
    direct: vec3<f32>,
    indirect: vec3<f32>,
    level: u32,
    found: bool,
    history: vec2<u32>,
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

// The level a read of frame point `p` with footprint `footprint` (metres)
// keys: the cell at least as wide as the footprint and as the eye angle at
// its distance, whichever is coarser, the nearest power of two.
fn radiance_cache_level(p: vec3<f32>, footprint: f32) -> u32 {
    let d = length(p - radiance_cache_frame.eye.xyz);
    let size = max(footprint, radiance_cache_frame.offset.w * d);
    let l = round(log2(max(size, 1e-30))) - f32(RADIANCE_CACHE_FINEST);
    return u32(clamp(l, 0.0, f32(RADIANCE_CACHE_LEVELS - 1u)));
}

// The dominant side of unit `n`: axis * 2, plus 1 toward the negative.
fn radiance_cache_side(n: vec3<f32>) -> u32 {
    let a = abs(n);
    if a.x >= a.y && a.x >= a.z { return select(0u, 1u, n.x < 0.0); }
    if a.y >= a.z { return select(2u, 3u, n.y < 0.0); }
    return select(4u, 5u, n.z < 0.0);
}

// A 64-bit `lo`, `hi` shifted left by `s` (0..16).
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

// The frame origin's cell at `level`: whole 1024 m times the cells in
// 1024 m.
fn radiance_cache_origin_cell(level: u32) -> array<vec2<u32>, 3> {
    let s = RADIANCE_CACHE_LEVELS - 1u - level;
    let o = radiance_cache_frame.origin_xy;
    let z = radiance_cache_frame.origin_z;
    return array<vec2<u32>, 3>(radiance_cache_shl(o.x, o.y, s), radiance_cache_shl(o.z, o.w, s), radiance_cache_shl(z.x, z.y, s));
}

// Frame point `p` in level `level`'s cells (frame-local).
fn radiance_cache_cells(p: vec3<f32>, level: u32) -> vec3<f32> {
    return (p + radiance_cache_frame.offset.xyz) * radiance_cache_inv_size(level);
}

// The key of the cell holding frame point `p` at `level`, on the side of
// unit `n`, in coordinate space `space`.
fn radiance_cache_key(p: vec3<f32>, n: vec3<f32>, level: u32, space: u32) -> RadianceCacheKey {
    let c = vec3<i32>(floor(radiance_cache_cells(p, level)));
    let o = radiance_cache_origin_cell(level);
    let x = radiance_cache_add(o[0], c.x);
    let y = radiance_cache_add(o[1], c.y);
    let z = radiance_cache_add(o[2], c.z);
    let tag = level | (radiance_cache_side(n) << 5u) | ((space & 0x1ffu) << 8u);
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

// The slot holding cell `k`, or RADIANCE_CACHE_NONE.
fn radiance_cache_find(k: RadianceCacheKey) -> u32 {
    let base = radiance_cache_bucket(k);
    let check = radiance_cache_checksum(k);
    for (var i = 0u; i < RADIANCE_CACHE_BUCKET; i++) {
        let s = base + i;
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

// What entry `e`'s light is at frame point `p` for a reader of reach `reach`
// (metres along the surface). A cell wider than an occluder beside it (a
// wall on a floor, a slab against a wall) gathers its light on one side:
// - RADIANCE_CACHE_HOLDS: the way from the cell's point to `p` along its
//   surface is shorter than its rays found clear that way and than the
//   reader's reach (a way they have not explored: `p` within
//   RADIANCE_CACHE_NEAR of its point);
// - RADIANCE_CACHE_BLOCKED: `p` is past what its rays met that way, or past
//   the reader's reach (the reader's own surface is in the hit's way there):
//   a finer cell is `p`'s;
// - RADIANCE_CACHE_UNKNOWN: its rays have not explored that way yet: it has
//   no light for `p` until they have (a few updates).
const RADIANCE_CACHE_HOLDS: u32 = 0u;
const RADIANCE_CACHE_BLOCKED: u32 = 1u;
const RADIANCE_CACHE_UNKNOWN: u32 = 2u;

fn radiance_cache_light_at(e: RadianceCacheEntry, p: vec3<f32>, reach: f32) -> u32 {
    let along = radiance_cache_along(scene_unpack_unit(e.normal), p - radiance_cache_entry_point(e));
    if along.y > reach { return RADIANCE_CACHE_BLOCKED; }
    let clear = radiance_cache_clear(e, u32(along.x));
    if clear == 0.0 {
        return select(RADIANCE_CACHE_UNKNOWN, RADIANCE_CACHE_HOLDS, along.y <= RADIANCE_CACHE_NEAR * radiance_cache_size(e.key.tag & 31u));
    }
    return select(RADIANCE_CACHE_BLOCKED, RADIANCE_CACHE_HOLDS, along.y <= clear);
}

// Whether entry `e`'s light holds at frame point `p` for a reader of reach
// `reach` (`radiance_cache_light_at`).
fn radiance_cache_holds(e: RadianceCacheEntry, p: vec3<f32>, reach: f32) -> bool {
    return radiance_cache_light_at(e, p, reach) == RADIANCE_CACHE_HOLDS;
}

// The finest cell below `level` whose light holds at `p` (side of `n`,
// coordinate space `space`, a reader of reach `reach`), starting from the cell at `level` whose light
// does not, no finer than the eye angle's cells there (the cache's
// resolution, its capacity's: the finer cells near occluders cost no more
// cells than the eye angle allows a surface): RADIANCE_CACHE_NONE when none
// does.
fn radiance_cache_finer(p: vec3<f32>, n: vec3<f32>, level: u32, space: u32, reach: f32) -> u32 {
    let finest = radiance_cache_level(p, 0.0);
    for (var at = level; at > finest; at--) {
        let s = radiance_cache_find(radiance_cache_key(p, n, at - 1u, space));
        if s == RADIANCE_CACHE_NONE { break; }
        let e = radiance_cache_entries[s];
        if !radiance_cache_has_value(e) { break; }
        let light = radiance_cache_light_at(e, p, reach);
        if light == RADIANCE_CACHE_HOLDS { return s; }
        if light == RADIANCE_CACHE_UNKNOWN { break; }
    }
    return RADIANCE_CACHE_NONE;
}

// The irradiance the cache holds at frame point `p` on the side of unit `n`
// (the face turned toward the ray), for a ray of footprint `footprint` there
// in coordinate space `space`: its cell's, or the next coarser cell's that
// has a value, at most `coarser` levels coarser; where that cell's light does
// not hold at `p` for a reader of reach `reach` (`radiance_cache_holds`: an
// occluder between its point and `p`, or its point past the reach), the
// finer cell that holds, or the next coarser; `found` false when none has.
fn radiance_cache_read(p: vec3<f32>, n: vec3<f32>, footprint: f32, space: u32, coarser: u32, reach: f32) -> RadianceCacheIrradiance {
    if !radiance_cache_present() { return RadianceCacheIrradiance(vec3<f32>(0.0), vec3<f32>(0.0), 0u, false, vec2<u32>(0u)); }
    let first = radiance_cache_level(p, footprint);
    let last = min(first + min(coarser, RADIANCE_CACHE_LEVELS), RADIANCE_CACHE_LEVELS - 1u);
    for (var level = first; level <= last; level++) {
        var s = radiance_cache_find(radiance_cache_key(p, n, level, space));
        if s == RADIANCE_CACHE_NONE { continue; }
        if !radiance_cache_has_value(radiance_cache_entries[s]) { continue; }
        var at = level;
        let light = radiance_cache_light_at(radiance_cache_entries[s], p, reach);
        if light == RADIANCE_CACHE_UNKNOWN { continue; }
        if light == RADIANCE_CACHE_BLOCKED {
            s = radiance_cache_finer(p, n, level, space, reach);
            if s == RADIANCE_CACHE_NONE { continue; }
            at = radiance_cache_entries[s].key.tag & 31u;
        }
        let e = radiance_cache_entries[s];
        return RadianceCacheIrradiance(
            vec3<f32>(e.direct[0], e.direct[1], e.direct[2]),
            vec3<f32>(e.indirect[0] + e.bounced[0], e.indirect[1] + e.bounced[1], e.indirect[2] + e.bounced[2]),
            at,
            true,
            vec2<u32>(e.history & 0xffu, (e.history >> 8u) & 0xffu),
        );
    }
    return RadianceCacheIrradiance(vec3<f32>(0.0), vec3<f32>(0.0), 0u, false, vec2<u32>(0u));
}

// Frame point `p`'s place in its cell at `level`, 15 bits an axis: the
// entry's `point` (the top 10 bits, 10:10:10) and the bits of its `history`
// from 16 (the low 5, 5:5:5).
fn radiance_cache_pack_point(p: vec3<f32>, level: u32) -> vec2<u32> {
    let q = radiance_cache_cells(p, level);
    let u = vec3<u32>(clamp((q - floor(q)) * 32768.0, vec3<f32>(0.0), vec3<f32>(32767.0)));
    let high = u >> vec3<u32>(5u);
    let low = u & vec3<u32>(31u);
    return vec2<u32>(high.x | (high.y << 10u) | (high.z << 20u), (low.x | (low.y << 5u) | (low.z << 10u)) << 16u);
}

// Entry `e`'s point, in the frame's space (within 1/32768 of its cell an
// axis of where the hit was).
fn radiance_cache_entry_point(e: RadianceCacheEntry) -> vec3<f32> {
    let level = e.key.tag & 31u;
    let o = radiance_cache_origin_cell(level);
    let c = vec3<f32>(vec3<i32>(bitcast<i32>(e.key.x_lo - o[0].x), bitcast<i32>(e.key.y_lo - o[1].x), bitcast<i32>(e.key.z_lo - o[2].x)));
    let high = vec3<u32>(e.point & 1023u, (e.point >> 10u) & 1023u, (e.point >> 20u) & 1023u);
    let fine = e.history >> 16u;
    let low = vec3<u32>(fine & 31u, (fine >> 5u) & 31u, (fine >> 10u) & 31u);
    let f = (vec3<f32>((high << vec3<u32>(5u)) | low) + 0.5) / 32768.0;
    return (c + f) * radiance_cache_size(level) - radiance_cache_frame.offset.xyz;
}

// Marks the cell of frame point `p` at `level` on the side of `n` in
// `space` as used this frame, claiming an empty slot of its bucket for it
// (its point and normal this hit's, a place in the new cells' queue) when
// no slot holds it. A full bucket or a full queue keeps it out (a read then
// answers from a coarser cell).
fn radiance_cache_mark_level(p: vec3<f32>, n: vec3<f32>, level: u32, space: u32) {
    let key = radiance_cache_key(p, n, level, space);
    let base = radiance_cache_bucket(key);
    let check = radiance_cache_checksum(key);
    let frame = radiance_cache_frame.origin_z.w;
    let slots = radiance_cache_frame.origin_z.z;
    var i = 0u;
    var retry = 0u;
    // This mark's place in the queue, once reserved.
    var place = RADIANCE_CACHE_NONE;
    loop {
        if i >= RADIANCE_CACHE_BUCKET { break; }
        let s = base + i;
        var state = atomicLoad(&radiance_cache_slots[s].state);
        if state == 0u {
            if place == RADIANCE_CACHE_NONE {
                let queued = slots / RADIANCE_CACHE_QUEUE_SHARE;
                if atomicLoad(&radiance_cache_slots[slots].state) >= queued { return; }
                let q = atomicAdd(&radiance_cache_slots[slots].state, 1u);
                if q >= queued { return; }
                place = slots + 1u + q;
            }
            let claim = atomicCompareExchangeWeak(&radiance_cache_slots[s].state, 0u, check);
            if claim.exchanged {
                let at = radiance_cache_pack_point(p, level);
                radiance_cache_entries[s] = RadianceCacheEntry(key, at.x, scene_pack_unit(n), at.y, array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0), array<f32, 3>(0.0, 0.0, 0.0), 0u, array<u32, 4>(0u, 0u, 0u, 0u));
                atomicStore(&radiance_cache_slots[s].used, frame);
                atomicStore(&radiance_cache_slots[place].state, s);
                return;
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
            // An unseated cell takes this hit's point and normal, with no
            // value and its ways unexplored (two marks at once may mix
            // their points' bits: a point still in the cell, which its rays
            // then judge again).
            if radiance_cache_entries[s].point == RADIANCE_CACHE_UNSEATED {
                let at = radiance_cache_pack_point(p, level);
                radiance_cache_entries[s].normal = scene_pack_unit(n);
                radiance_cache_entries[s].history = at.y;
                radiance_cache_entries[s].clear = array<u32, 4>(0u, 0u, 0u, 0u);
                radiance_cache_entries[s].point = at.x;
            }
            if atomicLoad(&radiance_cache_slots[s].used) != frame {
                atomicStore(&radiance_cache_slots[s].used, frame);
            }
            break;
        }
        i++;
    }
    // A place reserved for a claim another mark won: nothing in it.
    if place != RADIANCE_CACHE_NONE { atomicStore(&radiance_cache_slots[place].state, RADIANCE_CACHE_NONE); }
}

// Marks the cell a read of frame point `p` (side of unit `n`, footprint
// `footprint`, coordinate space `space`, reach `reach`) keys as used this
// frame, the cell RADIANCE_CACHE_ANCESTOR levels coarser and, where a cell's
// light does not hold at `p` (`radiance_cache_holds`), the finer ones a read
// goes to, down to the first that holds or is new, and no finer than the eye
// angle's.
fn radiance_cache_mark(p: vec3<f32>, n: vec3<f32>, footprint: f32, space: u32, reach: f32) {
    if !radiance_cache_present() { return; }
    let level = radiance_cache_level(p, footprint);
    radiance_cache_mark_level(p, n, level, space);
    if level + RADIANCE_CACHE_ANCESTOR < RADIANCE_CACHE_LEVELS {
        radiance_cache_mark_level(p, n, level + RADIANCE_CACHE_ANCESTOR, space);
    }
    let finest = radiance_cache_level(p, 0.0);
    var at = level;
    loop {
        if at <= finest { break; }
        let s = radiance_cache_find(radiance_cache_key(p, n, at, space));
        if s == RADIANCE_CACHE_NONE { break; }
        let e = radiance_cache_entries[s];
        // Finer only past an occluder the cell's rays met, or the reach.
        if !radiance_cache_has_value(e) || radiance_cache_light_at(e, p, reach) != RADIANCE_CACHE_BLOCKED { break; }
        at -= 1u;
        radiance_cache_mark_level(p, n, at, space);
    }
}
