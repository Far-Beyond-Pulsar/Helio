// Environment join: one source row per component instance (fog volumes,
// post-process volumes, camera post-process baselines, water volumes; see
// `environment_join.rs`). A placed row is copied into its pass's buffer; a
// spatial row gets its world AABB from the owner's transform first. Every
// other output row stays zero, which each pass treats as inert (`enabled`
// 0, `blend_weight` 0, zero extent).
//
// `cs_join_rows` keeps the source row index (passes that scan the whole
// buffer). `cs_compact_rows` packs placed rows into the first `capacity`
// output rows in source row order, for passes that read a fixed number of
// leading rows (water).

struct JoinUniforms {
    rows: u32,
    source_words: u32,
    output_words: u32,
    /// bit 0: spatial (source starts with a local size vec4, output with
    /// AABB min/max vec4s); bit 1: a hidden owner turns the row off;
    /// bit 2: the output's `bounds_max.w` is the owner's Y plus the source
    /// size's `w`, scaled like the box (a water surface height).
    flags: u32,
    /// Output rows `cs_compact_rows` may fill.
    capacity: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const SPATIAL: u32 = 1u;
const GATE_HIDDEN: u32 = 2u;
const SURFACE: u32 = 4u;
const WORKGROUP: u32 = 64u;

@group(0) @binding(0) var<uniform> u: JoinUniforms;
@group(0) @binding(1) var<storage, read> owners: array<Owner>;
@group(0) @binding(2) var<storage, read> generations: array<u32>;
@group(0) @binding(3) var<storage, read> hidden: array<u32>;
@group(0) @binding(4) var<storage, read> transforms: array<ObjectTransform>;
@group(0) @binding(5) var<storage, read> sources: array<u32>;
@group(0) @binding(6) var<storage, read_write> rows_out: array<u32>;

fn source_size(row: u32) -> vec3<f32> {
    let source = row * u.source_words;
    return vec3<f32>(
        bitcast<f32>(sources[source]),
        bitcast<f32>(sources[source + 1u]),
        bitcast<f32>(sources[source + 2u]),
    );
}

/// Whether source `row` becomes a pass row: attached, enabled, its owner
/// live (and visible, when gated), and, for a volume, a non-empty box.
fn placed(row: u32) -> bool {
    if row >= u.rows || row >= arrayLength(&owners) {
        return false;
    }
    if (row + 1u) * u.source_words > arrayLength(&sources) {
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
    if (u.flags & SPATIAL) != 0u {
        if index >= arrayLength(&transforms) {
            return false;
        }
        if all(source_size(row) == vec3<f32>(0.0)) {
            return false;
        }
    }
    return true;
}

/// Writes placed source `row` as output row `slot`.
fn write_row(row: u32, slot: u32) {
    let source = row * u.source_words;
    let output = slot * u.output_words;
    if output + u.output_words > arrayLength(&rows_out) {
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
    let count = min(u.source_words - source_header, u.output_words - output_header);
    for (var word = 0u; word < count; word++) {
        rows_out[output + output_header + word] = sources[source + source_header + word];
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

/// One workgroup walks the source rows in order, 64 at a time; each placed
/// row takes the next output slot until `capacity` is reached. The order is
/// the source row order, so a volume keeps its slot from frame to frame
/// while the rows before it are unchanged.
@compute @workgroup_size(64)
fn cs_compact_rows(@builtin(local_invocation_index) lid: u32) {
    if lid == 0u {
        wg_base = 0u;
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
        workgroupBarrier();
        let slot = base + before;
        if keep && slot < u.capacity {
            write_row(row, slot);
        }
        if lid == WORKGROUP - 1u {
            wg_base = slot + wg_placed[lid];
        }
        workgroupBarrier();
    }
}
