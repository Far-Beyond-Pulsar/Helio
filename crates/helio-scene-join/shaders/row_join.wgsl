// Row join: one source row per component instance (fog media, post-process
// volumes, water volumes, foliage, atmospheres, decals, particle emitters,
// sprites: each kind builds its own join). A placed row is copied into its
// pass's buffer; an `OWNER_AABB` row gets its world AABB from the owner's
// transform first. Every other output row stays zero, which each pass treats
// as inert (`enabled` 0, `blend_weight` 0, zero extent).
//
// `cs_join_rows` keeps the source row index (passes that scan the whole
// buffer). `cs_compact_rows` packs placed rows into the first `capacity`
// output rows in source row order, for passes that read a fixed number of
// leading rows (water, foliage). A table may read a slice of a wider source
// row (`source_offset`, `copy_words`): the three foliage tables share one.
//
// What a kind's rows need beyond that (a decal's box transform, an emitter's
// pool range, a sprite's 2D placement, water's sun and wind) is the kind's
// own placement: WGSL hooks the kind's crate supplies
// (`helio_scene_join::Placement`), compiled after this file, each with a
// default for the hooks it leaves out:
//
//   fn placement_placed(row: u32) -> bool        // further conditions
//   fn placement_write_row(row: u32, slot: u32, output: u32) -> bool
//                                                // writes the whole row (true: no copy)
//   fn placement_source_header() -> u32          // source words skipped before the copy
//   fn placement_write(row: u32, output: u32)    // after the copy
//   fn placement_request(row: u32) -> u32        // pool units a packed row takes
//   fn placement_range(row: u32, slot: u32, offset: u32) // its range of the pool
//
// Hooks read the bindings and helpers below (`u`, `owners`, `transforms`,
// `sources`, `rows_out`, `directional_light`, `owner_bounds`, `source_base`,
// `source_size`, `placed_position`, `placed_altitude`, the `object_*`
// helpers of `scene_join_common.wgsl`).

struct JoinUniforms {
    rows: u32,
    source_words: u32,
    output_words: u32,
    /// `OWNER_AABB`, `GATE_HIDDEN`, `READS_TRANSFORMS` (the placement places
    /// from the owner's transform: an owner without one is not placed).
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
    /// Units of the pool `placement_request`/`placement_range` allocate.
    pool: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // The frame's world origin (hi + lo): placed positions are written
    // relative to it, the frame's space every pass draws in.
    origin_hi: vec4<f32>,
    origin_lo: vec4<f32>,
    // The world origin's X and Z modulo the table's phase period (`xy`).
    phase: vec4<f32>,
}

const OWNER_AABB: u32 = 1u;
const GATE_HIDDEN: u32 = 2u;
const READS_TRANSFORMS: u32 = 4u;
const NO_GATE_WORD: u32 = 0xffffffffu;
const WORKGROUP: u32 = 64u;

@group(0) @binding(0) var<uniform> u: JoinUniforms;
@group(0) @binding(1) var<storage, read> owners: array<Owner>;
@group(0) @binding(2) var<storage, read> generations: array<u32>;
@group(0) @binding(3) var<storage, read> hidden: array<u32>;
@group(0) @binding(4) var<storage, read> transforms: array<ObjectTransform>;
@group(0) @binding(5) var<storage, read> sources: array<u32>;
@group(0) @binding(6) var<storage, read_write> rows_out: array<u32>;
// The directional light a placement takes (`RowJoinKeys::directional_light`:
// its first `vec4`, the direction toward it and 1).
@group(0) @binding(7) var<storage, read> directional_light: array<vec4<f32>>;
// The owners' content's local bounding spheres, keyed like `owners`.
@group(0) @binding(8) var<storage, read> owner_bounds: array<vec4<f32>>;

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
    if (u.flags & (OWNER_AABB | READS_TRANSFORMS)) != 0u && index >= arrayLength(&transforms) {
        return false;
    }
    if (u.flags & OWNER_AABB) != 0u && all(source_size(row) == vec3<f32>(0.0)) {
        return false;
    }
    return placement_placed(row);
}

/// A world altitude in the frame's space: less the origin's Y. Authored
/// altitudes are f32: far from the world origin they hold only their own
/// step (0.5 m at Earth's radius), which this keeps.
fn placed_altitude(y: f32) -> f32 {
    return y - (u.origin_hi.y + u.origin_lo.y);
}

/// The owner's position in the frame's space: relative to the world origin,
/// subtracted pair-wise, so it is exact however far the owner is.
fn placed_position(t: ObjectTransform) -> vec3<f32> {
    return object_relative(t, u.origin_hi, u.origin_lo);
}

/// Writes placed source `row` as output row `slot`.
fn write_row(row: u32, slot: u32) {
    let source = source_base(row);
    let output = slot * u.output_words;
    if output + u.output_words > arrayLength(&rows_out) {
        return;
    }
    if placement_write_row(row, slot, output) {
        return;
    }
    var source_header = placement_source_header();
    var output_header = 0u;
    if (u.flags & OWNER_AABB) != 0u {
        let t = transforms[owners[row].owner_index];
        let scale = object_scale(t);
        // The world AABB of the owner-oriented box.
        let half = abs(source_size(row) * scale) * 0.5;
        let r = object_rotation(t);
        let extent = abs(r[0]) * half.x + abs(r[1]) * half.y + abs(r[2]) * half.z;
        let center = placed_position(t);
        let lo = center - extent;
        let hi = center + extent;
        rows_out[output] = bitcast<u32>(lo.x);
        rows_out[output + 1u] = bitcast<u32>(lo.y);
        rows_out[output + 2u] = bitcast<u32>(lo.z);
        rows_out[output + 3u] = 0u;
        rows_out[output + 4u] = bitcast<u32>(hi.x);
        rows_out[output + 5u] = bitcast<u32>(hi.y);
        rows_out[output + 6u] = bitcast<u32>(hi.z);
        rows_out[output + 7u] = 0u;
        source_header = 4u;
        output_header = 8u;
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
    placement_write(row, output);
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
/// while the rows before it are unchanged. A placement that allocates (a
/// particle emitter) also gives each slotted row the next range of the
/// table's pool (a prefix sum of the slotted rows' `placement_request`s), so
/// ranges are contiguous, disjoint and stable the same way.
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
        if slotted {
            request = placement_request(row);
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
            placement_range(row, slot, pool_base + pool_before);
        }
        if lid == WORKGROUP - 1u {
            wg_base = slot + wg_placed[lid];
            wg_pool_base = pool_base + pool_before + request;
        }
        workgroupBarrier();
    }
}
