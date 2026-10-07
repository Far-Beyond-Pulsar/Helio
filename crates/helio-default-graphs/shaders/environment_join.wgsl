// Environment join: one thread per authored source row (fog volumes,
// post-process volumes, camera post-process baselines; see
// `environment_join.rs`). A placed row is copied into the same row of its
// pass's buffer; a spatial row gets its world AABB from the owner's
// transform first. Every other row stays zero, which each pass treats as
// inert (`enabled` 0, `blend_weight` 0).

struct JoinUniforms {
    rows: u32,
    source_words: u32,
    output_words: u32,
    /// bit 0: spatial (source starts with a local size vec4, output with
    /// AABB min/max vec4s); bit 1: a hidden owner turns the row off.
    flags: u32,
}

const SPATIAL: u32 = 1u;
const GATE_HIDDEN: u32 = 2u;

@group(0) @binding(0) var<uniform> u: JoinUniforms;
@group(0) @binding(1) var<storage, read> owners: array<Owner>;
@group(0) @binding(2) var<storage, read> generations: array<u32>;
@group(0) @binding(3) var<storage, read> hidden: array<u32>;
@group(0) @binding(4) var<storage, read> transforms: array<ObjectTransform>;
@group(0) @binding(5) var<storage, read> sources: array<u32>;
@group(0) @binding(6) var<storage, read_write> rows_out: array<u32>;

@compute @workgroup_size(64)
fn cs_join_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= u.rows || row >= arrayLength(&owners) {
        return;
    }
    let source = row * u.source_words;
    let output = row * u.output_words;
    if source + u.source_words > arrayLength(&sources) || output + u.output_words > arrayLength(&rows_out) {
        return;
    }
    let owner = owners[row];
    if owner.enabled == 0u {
        return;
    }
    let index = owner.owner_index;
    if index >= arrayLength(&generations) || generations[index] != owner.owner_generation {
        return;
    }
    if (u.flags & GATE_HIDDEN) != 0u && index < arrayLength(&hidden) && hidden[index] != 0u {
        return;
    }

    var source_header = 0u;
    var output_header = 0u;
    if (u.flags & SPATIAL) != 0u {
        if index >= arrayLength(&transforms) {
            return;
        }
        let t = transforms[index];
        let size = vec3<f32>(
            bitcast<f32>(sources[source]),
            bitcast<f32>(sources[source + 1u]),
            bitcast<f32>(sources[source + 2u]),
        );
        // The world AABB of the owner-oriented box.
        let half = abs(size * object_scale(t)) * 0.5;
        let r = object_rotation(t);
        let extent = abs(r[0]) * half.x + abs(r[1]) * half.y + abs(r[2]) * half.z;
        let center = object_position(t);
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
    let count = min(u.source_words - source_header, u.output_words - output_header);
    for (var word = 0u; word < count; word++) {
        rows_out[output + output_header + word] = sources[source + source_header + word];
    }
}
