// Environment join: one source row per component instance (fog volumes,
// post-process volumes, camera post-process baselines, water volumes,
// foliage, atmospheres, decals, particle emitters; see `environment_join.rs`). A placed row is copied into its pass's buffer; a
// spatial row gets its world AABB from the owner's transform first. Every
// other output row stays zero, which each pass treats as inert (`enabled`
// 0, `blend_weight` 0, zero extent).
//
// `cs_join_rows` keeps the source row index (passes that scan the whole
// buffer). `cs_compact_rows` packs placed rows into the first `capacity`
// output rows in source row order, for passes that read a fixed number of
// leading rows (water, foliage). A table may read a slice of a wider source
// row (`source_offset`, `copy_words`): the three foliage tables share one.

struct JoinUniforms {
    rows: u32,
    source_words: u32,
    output_words: u32,
    /// bit 0: spatial (source starts with a local size vec4, output with
    /// AABB min/max vec4s); bit 1: a hidden owner turns the row off;
    /// bit 2: the output's `bounds_max.w` is the owner's Y plus the source
    /// size's `w`, scaled like the box (a water surface height); bit 3: a
    /// foliage layer (see `write_layer`); bit 4: an atmosphere, whose
    /// centre (words 0..2) moves to the owner's position when its
    /// placement (word 3) is `PLACEMENT_CENTER`; bit 5: a decal (see
    /// `write_decal_transform`); bit 6: a particle emitter (see
    /// `write_emitter_transform` and `cs_compact_rows`); bit 7: a water
    /// row, whose `sun_direction` is the scene's sun (see `write_sun`);
    /// bit 8: the global wind over the water rows (see `write_water_wind`);
    /// bit 9: a water hitbox (see `write_hitbox`); bit 10: a 2D sprite (see
    /// `write_sprite_placement`).
    flags: u32,
    /// Output rows `cs_compact_rows` may fill.
    capacity: u32,
    /// The source word this table's slice starts at.
    source_offset: u32,
    /// Words copied after the headers; 0 copies as many as both rows hold.
    copy_words: u32,
    /// A source word (from the row's start) that must be non-zero for the
    /// row to be placed; `NO_GATE_WORD` for none.
    gate_word: u32,
    /// Particles the emitter ranges are allocated from.
    pool: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const SPATIAL: u32 = 1u;
const GATE_HIDDEN: u32 = 2u;
const SURFACE: u32 = 4u;
const LAYER: u32 = 8u;
const CENTERED: u32 = 16u;
const DECAL: u32 = 32u;
const EMITTER: u32 = 64u;
const SUN: u32 = 128u;
const WATER_WIND: u32 = 256u;
const HITBOX: u32 = 512u;
const SPRITE: u32 = 1024u;
// `helio_pass_water_sim::GpuWaterVolume`: `sun_direction` and `wind_params`
// (`w`: the volume opts out of the global wind).
const WATER_SUN_WORD: u32 = 44u;
const WATER_WIND_WORD: u32 = 56u;
const WATER_OWN_WIND_WORD: u32 = 59u;
// `environment_join.rs`'s `WATER_WIND_STRENGTH_PER_SPEED`.
const WATER_WIND_PER_SPEED: f32 = 0.5;
// `scene_lights` rows (`helio_pass_forward_lit::GpuLight`): 32 words.
const LIGHT_ROW_WORDS: u32 = 32u;
// `helio_pass_corona::GpuCoronaEmitter`: `particle_offset`, `particle_count`
// (the requested range in the source row) and `spawn_cursor`, which the
// pass reads as the emitter's identity.
const EMITTER_OFFSET_WORD: u32 = 45u;
const EMITTER_COUNT_WORD: u32 = 46u;
const EMITTER_IDENTITY_WORD: u32 = 47u;
// `helio_pass_corona::CORONA_RANGE_ALIGNMENT`.
const EMITTER_RANGE_ALIGNMENT: u32 = 256u;
// `helio_pass_sky::atmosphere::placement::CENTER`.
const PLACEMENT_CENTER: u32 = 1u;
const NO_GATE_WORD: u32 = 0xffffffffu;
const WORKGROUP: u32 = 64u;

@group(0) @binding(0) var<uniform> u: JoinUniforms;
@group(0) @binding(1) var<storage, read> owners: array<Owner>;
@group(0) @binding(2) var<storage, read> generations: array<u32>;
@group(0) @binding(3) var<storage, read> hidden: array<u32>;
@group(0) @binding(4) var<storage, read> transforms: array<ObjectTransform>;
@group(0) @binding(5) var<storage, read> sources: array<u32>;
@group(0) @binding(6) var<storage, read_write> rows_out: array<u32>;
@group(0) @binding(7) var<storage, read> lights: array<u32>;
// Mesh instances' local bounding spheres, keyed like `owners`.
@group(0) @binding(8) var<storage, read> mesh_bounds: array<vec4<f32>>;

fn source_base(row: u32) -> u32 {
    return row * u.source_words + u.source_offset;
}

fn source_size(row: u32) -> vec3<f32> {
    let source = source_base(row);
    return vec3<f32>(
        bitcast<f32>(sources[source]),
        bitcast<f32>(sources[source + 1u]),
        bitcast<f32>(sources[source + 2u]),
    );
}

/// Whether source `row` becomes a pass row: attached, enabled, its owner
/// live (and visible, when gated), its gate word set, and, for a volume, a
/// non-empty box.
fn placed(row: u32) -> bool {
    if row >= u.rows || row >= arrayLength(&owners) {
        return false;
    }
    if (row + 1u) * u.source_words > arrayLength(&sources) {
        return false;
    }
    if u.gate_word != NO_GATE_WORD && sources[row * u.source_words + u.gate_word] == 0u {
        return false;
    }
    let owner = owners[row];
    if owner.enabled == 0u {
        return false;
    }
    let index = owner.owner_index;
    if index >= arrayLength(&generations) || generations[index] != owner.owner_generation {
        return false;
    }
    if (u.flags & GATE_HIDDEN) != 0u && index < arrayLength(&hidden) && hidden[index] != 0u {
        return false;
    }
    if (u.flags & (SPATIAL | LAYER | CENTERED | DECAL | EMITTER | HITBOX | SPRITE)) != 0u && index >= arrayLength(&transforms) {
        return false;
    }
    if (u.flags & DECAL) != 0u && any(source_size(row) == vec3<f32>(0.0)) {
        return false;
    }
    if (u.flags & SPATIAL) != 0u && all(source_size(row) == vec3<f32>(0.0)) {
        return false;
    }
    // A body whose owner holds no mesh has nothing to bound.
    if (u.flags & HITBOX) != 0u && body_bounds(row).lo.w == 0.0 {
        return false;
    }
    return true;
}

/// A foliage layer: a world-aligned square of half extent `source[0]`
/// (scaled by the owner's X and Z scale) centred on the owner, spanning the
/// authored altitudes `source[2]..source[3]` in Y; `source[1]` is the
/// infinite-extent flag, which the placement pass reads from `bounds_max.w`.
fn write_layer(row: u32, output: u32) {
    let source = source_base(row);
    let t = transforms[owners[row].owner_index];
    let center = object_position(t);
    let scale = abs(object_scale(t));
    let half = bitcast<f32>(sources[source]);
    rows_out[output] = bitcast<u32>(center.x - half * scale.x);
    rows_out[output + 1u] = sources[source + 2u];
    rows_out[output + 2u] = bitcast<u32>(center.z - half * scale.z);
    rows_out[output + 3u] = 0u;
    rows_out[output + 4u] = bitcast<u32>(center.x + half * scale.x);
    rows_out[output + 5u] = sources[source + 3u];
    rows_out[output + 6u] = bitcast<u32>(center.z + half * scale.z);
    rows_out[output + 7u] = sources[source + 1u];
}

/// A decal's world-to-decal transform (`mat4x4`, column-major, words 0..15
/// of the output row): world space into the owner-placed box, whose full
/// local extent is the source size, so the box spans -1..1 on each axis.
/// Row `i` of the matrix is the box's world axis `i` over its world half
/// extent; the translation takes the owner's position to the origin.
fn write_decal_transform(row: u32, output: u32) {
    let t = transforms[owners[row].owner_index];
    let r = object_rotation(t);
    let half = source_size(row) * object_scale(t) * 0.5;
    let center = object_position(t);
    for (var i = 0u; i < 3u; i++) {
        let axis = r[i] / half[i];
        for (var j = 0u; j < 3u; j++) {
            rows_out[output + j * 4u + i] = bitcast<u32>(axis[j]);
        }
        rows_out[output + 12u + i] = bitcast<u32>(-dot(axis, center));
        rows_out[output + i * 4u + 3u] = 0u;
    }
    rows_out[output + 15u] = bitcast<u32>(1.0);
}

/// A particle emitter's transform (`mat4x4`, column-major, words 0..15 of
/// the output row): its owner's, so particles spawn at the owner.
fn write_emitter_transform(row: u32, output: u32) {
    let t = transforms[owners[row].owner_index];
    let r = object_rotation(t);
    let scale = object_scale(t);
    let center = object_position(t);
    for (var i = 0u; i < 3u; i++) {
        let axis = r[i] * scale[i];
        for (var j = 0u; j < 3u; j++) {
            rows_out[output + i * 4u + j] = bitcast<u32>(axis[j]);
        }
        rows_out[output + i * 4u + 3u] = 0u;
        rows_out[output + 12u + i] = bitcast<u32>(center[i]);
    }
    rows_out[output + 15u] = bitcast<u32>(1.0);
}

/// The pool particles source `row` takes: its request, rounded up to the
/// range alignment so the next range starts on it.
fn emitter_request(row: u32) -> u32 {
    let requested = sources[row * u.source_words + EMITTER_COUNT_WORD];
    return (requested + EMITTER_RANGE_ALIGNMENT - 1u) / EMITTER_RANGE_ALIGNMENT * EMITTER_RANGE_ALIGNMENT;
}

/// Gives output row `slot` (from source `row`) the pool range starting at
/// `offset`: its request, clamped to what is left of the pool (none past
/// its end), and the emitter's identity (its source row + 1): when another
/// emitter takes the slot, the pass restarts the slot's particles.
fn write_emitter_range(row: u32, slot: u32, offset: u32) {
    let output = slot * u.output_words;
    if output + u.output_words > arrayLength(&rows_out) {
        return;
    }
    let requested = sources[row * u.source_words + EMITTER_COUNT_WORD];
    var count = 0u;
    if offset < u.pool {
        count = min(requested, u.pool - offset);
    }
    rows_out[output + EMITTER_OFFSET_WORD] = min(offset, u.pool);
    rows_out[output + EMITTER_COUNT_WORD] = count;
    rows_out[output + EMITTER_IDENTITY_WORD] = row + 1u;
}

/// A water row's sun: the direction toward the scene's first directional
/// light with any intensity (`direction_outer` is the direction the light
/// travels), as the sky's atmosphere picks its sun; `w` is 1. Straight up
/// with `w` 0 when the scene has none.
fn write_sun(output: u32) {
    var sun = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    let rows = arrayLength(&lights) / LIGHT_ROW_WORDS;
    for (var i = 0u; i < rows; i++) {
        let base = i * LIGHT_ROW_WORDS;
        let intensity = bitcast<f32>(lights[base + 11u]);
        let light_type = lights[base + 13u];
        let direction = vec3<f32>(
            bitcast<f32>(lights[base + 4u]),
            bitcast<f32>(lights[base + 5u]),
            bitcast<f32>(lights[base + 6u]),
        );
        if light_type == 0u && intensity > 0.0 && dot(direction, direction) > 0.0 {
            sun = vec4<f32>(-normalize(direction), 1.0);
            break;
        }
    }
    for (var i = 0u; i < 4u; i++) {
        rows_out[output + WATER_SUN_WORD + i] = bitcast<u32>(sun[i]);
    }
}

/// The global wind (source `row`, a foliage wind row: normalised direction
/// and speed) as the wind of every placed water row that does not opt out
/// of it: its direction's XZ, and a strength proportional to its speed.
fn write_water_wind(row: u32) {
    let source = row * u.source_words;
    let direction = vec2<f32>(bitcast<f32>(sources[source]), bitcast<f32>(sources[source + 2u]));
    let speed = max(bitcast<f32>(sources[source + 3u]), 0.0);
    let slots = min(u.capacity, arrayLength(&rows_out) / u.output_words);
    for (var slot = 0u; slot < slots; slot++) {
        let output = slot * u.output_words;
        // An unplaced row is zero; a placed one has a non-empty box.
        var placed_volume = false;
        for (var i = 0u; i < 3u; i++) {
            placed_volume = placed_volume || rows_out[output + i] != rows_out[output + 4u + i];
        }
        if !placed_volume || bitcast<f32>(rows_out[output + WATER_OWN_WIND_WORD]) != 0.0 {
            continue;
        }
        rows_out[output + WATER_WIND_WORD] = bitcast<u32>(direction.x);
        rows_out[output + WATER_WIND_WORD + 1u] = bitcast<u32>(direction.y);
        rows_out[output + WATER_WIND_WORD + 2u] = bitcast<u32>(speed * WATER_WIND_PER_SPEED);
    }
}

/// A body's world bounds: the AABB of the bounding spheres of every mesh
/// its owner holds (attached and enabled), placed by the owner's
/// transform. `w` of `lo` is 1 when the owner holds any.
struct BodyBounds {
    lo: vec4<f32>,
    hi: vec3<f32>,
}

fn body_bounds(row: u32) -> BodyBounds {
    let body = owners[row];
    let t = transforms[body.owner_index];
    let r = object_rotation(t);
    let scale = object_scale(t);
    let position = object_position(t);
    let largest_scale = max(abs(scale.x), max(abs(scale.y), abs(scale.z)));
    var lo = vec3<f32>(3.0e38);
    var hi = vec3<f32>(-3.0e38);
    var found = 0.0;
    let instances = min(arrayLength(&mesh_bounds), arrayLength(&owners));
    for (var i = 0u; i < instances; i++) {
        let sphere = mesh_bounds[i];
        let mesh = owners[i];
        if sphere.w <= 0.0 || mesh.enabled == 0u || mesh.owner_index != body.owner_index
            || mesh.owner_generation != body.owner_generation {
            continue;
        }
        let center = position + r * (sphere.xyz * scale);
        let radius = sphere.w * largest_scale;
        lo = min(lo, center - vec3<f32>(radius));
        hi = max(hi, center + vec3<f32>(radius));
        found = 1.0;
    }
    return BodyBounds(vec4<f32>(lo, found), hi);
}

/// A water hitbox (`helio_pass_water_sim::GpuWaterHitbox`, 20 words): the
/// body's new bounds are its `body_bounds`; its old bounds an empty box at
/// their centre (the simulation replaces them with where the body was last
/// applied while the row holds the same body); `params` the source's edge
/// softness and strength and the body's identity (source row + 1).
fn write_hitbox(row: u32, output: u32) {
    let bounds = body_bounds(row);
    let lo = bounds.lo.xyz;
    let hi = bounds.hi;
    let center = (lo + hi) * 0.5;
    let source = source_base(row);
    for (var i = 0u; i < 3u; i++) {
        rows_out[output + i] = bitcast<u32>(center[i]);
        rows_out[output + 4u + i] = bitcast<u32>(center[i]);
        rows_out[output + 8u + i] = bitcast<u32>(lo[i]);
        rows_out[output + 12u + i] = bitcast<u32>(hi[i]);
    }
    rows_out[output + 16u] = sources[source];
    rows_out[output + 17u] = sources[source + 1u];
    rows_out[output + 18u] = bitcast<u32>(f32(row + 1u));
}

/// A 2D sprite (`helio_pass_sprite_batch::SpriteComponent`): at its owner's
/// X and Y, turned by its roll (rotation about Z) and its size scaled by
/// its X and Y scale. The owner's Z and other rotations do not apply.
fn write_sprite_placement(row: u32, output: u32) {
    let t = transforms[owners[row].owner_index];
    let scale = abs(object_scale(t));
    let roll = t.rotation[2] * 0.017453292519943295;
    rows_out[output] = bitcast<u32>(t.position[0]);
    rows_out[output + 1u] = bitcast<u32>(t.position[1]);
    rows_out[output + 2u] = bitcast<u32>(bitcast<f32>(rows_out[output + 2u]) * scale.x);
    rows_out[output + 3u] = bitcast<u32>(bitcast<f32>(rows_out[output + 3u]) * scale.y);
    rows_out[output + 4u] = bitcast<u32>(bitcast<f32>(rows_out[output + 4u]) + roll);
}

/// Writes placed source `row` as output row `slot`.
fn write_row(row: u32, slot: u32) {
    let source = source_base(row);
    let output = slot * u.output_words;
    if output + u.output_words > arrayLength(&rows_out) {
        return;
    }
    if (u.flags & WATER_WIND) != 0u {
        // The first placed global wind (a level has one).
        if slot == 0u {
            write_water_wind(row);
        }
        return;
    }
    if (u.flags & LAYER) != 0u {
        write_layer(row, output);
        return;
    }
    if (u.flags & HITBOX) != 0u {
        write_hitbox(row, output);
        return;
    }
    var source_header = 0u;
    var output_header = 0u;
    if (u.flags & SPATIAL) != 0u {
        let t = transforms[owners[row].owner_index];
        let scale = object_scale(t);
        // The world AABB of the owner-oriented box.
        let half = abs(source_size(row) * scale) * 0.5;
        let r = object_rotation(t);
        let extent = abs(r[0]) * half.x + abs(r[1]) * half.y + abs(r[2]) * half.z;
        let center = object_position(t);
        let lo = center - extent;
        let hi = center + extent;
        var w = 0.0;
        if (u.flags & SURFACE) != 0u {
            w = center.y + bitcast<f32>(sources[source + 3u]) * scale.y;
        }
        rows_out[output] = bitcast<u32>(lo.x);
        rows_out[output + 1u] = bitcast<u32>(lo.y);
        rows_out[output + 2u] = bitcast<u32>(lo.z);
        rows_out[output + 3u] = 0u;
        rows_out[output + 4u] = bitcast<u32>(hi.x);
        rows_out[output + 5u] = bitcast<u32>(hi.y);
        rows_out[output + 6u] = bitcast<u32>(hi.z);
        rows_out[output + 7u] = bitcast<u32>(w);
        source_header = 4u;
        output_header = 8u;
    }
    if (u.flags & DECAL) != 0u {
        source_header = 4u;
    }
    var count = min(
        u.source_words - u.source_offset - source_header,
        u.output_words - output_header,
    );
    if u.copy_words != 0u {
        count = min(count, u.copy_words);
    }
    for (var word = 0u; word < count; word++) {
        rows_out[output + output_header + word] = sources[source + source_header + word];
    }
    if (u.flags & DECAL) != 0u {
        write_decal_transform(row, output);
    }
    if (u.flags & EMITTER) != 0u {
        write_emitter_transform(row, output);
    }
    if (u.flags & SUN) != 0u {
        write_sun(output);
    }
    if (u.flags & SPRITE) != 0u {
        write_sprite_placement(row, output);
    }
    if (u.flags & CENTERED) != 0u && sources[source + 3u] == PLACEMENT_CENTER {
        let center = object_position(transforms[owners[row].owner_index]);
        rows_out[output] = bitcast<u32>(bitcast<f32>(sources[source]) + center.x);
        rows_out[output + 1u] = bitcast<u32>(bitcast<f32>(sources[source + 1u]) + center.y);
        rows_out[output + 2u] = bitcast<u32>(bitcast<f32>(sources[source + 2u]) + center.z);
    }
}

@compute @workgroup_size(64)
fn cs_join_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if placed(row) {
        write_row(row, row);
    }
}

var<workgroup> wg_placed: array<u32, 64>;
var<workgroup> wg_base: u32;
var<workgroup> wg_request: array<u32, 64>;
var<workgroup> wg_pool_base: u32;

/// One workgroup walks the source rows in order, 64 at a time; each placed
/// row takes the next output slot until `capacity` is reached. The order is
/// the source row order, so a volume keeps its slot from frame to frame
/// while the rows before it are unchanged. Particle emitters also take the
/// next range of the particle pool (a prefix sum of the slotted rows'
/// requests), so their ranges are contiguous, disjoint and stable the same
/// way.
@compute @workgroup_size(64)
fn cs_compact_rows(@builtin(local_invocation_index) lid: u32) {
    if lid == 0u {
        wg_base = 0u;
        wg_pool_base = 0u;
    }
    workgroupBarrier();
    let chunks = (u.rows + WORKGROUP - 1u) / WORKGROUP;
    for (var chunk = 0u; chunk < chunks; chunk++) {
        let row = chunk * WORKGROUP + lid;
        let keep = placed(row);
        wg_placed[lid] = select(0u, 1u, keep);
        workgroupBarrier();
        var before = 0u;
        for (var i = 0u; i < lid; i++) {
            before += wg_placed[i];
        }
        let base = wg_base;
        let slot = base + before;
        let slotted = keep && slot < u.capacity;
        var request = 0u;
        if slotted && (u.flags & EMITTER) != 0u {
            request = emitter_request(row);
        }
        wg_request[lid] = request;
        workgroupBarrier();
        var pool_before = 0u;
        for (var i = 0u; i < lid; i++) {
            pool_before += wg_request[i];
        }
        let pool_base = wg_pool_base;
        workgroupBarrier();
        if slotted {
            write_row(row, slot);
            if (u.flags & EMITTER) != 0u {
                write_emitter_range(row, slot, pool_base + pool_before);
            }
        }
        if lid == WORKGROUP - 1u {
            wg_base = slot + wg_placed[lid];
            wg_pool_base = pool_base + pool_before + request;
        }
        workgroupBarrier();
    }
}
