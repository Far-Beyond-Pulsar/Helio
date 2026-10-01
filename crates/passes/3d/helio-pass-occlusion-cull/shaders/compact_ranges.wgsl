// Compact visible indirect draw records within each material/shading range.
// One workgroup owns one range, so its atomic allocator is deterministic with
// respect to range boundaries and never races another workgroup's output.

struct Range {
    material_class: u32,
    graph_hash_lo: u32,
    graph_hash_hi: u32,
    start: u32,
    count: u32,
}

struct RangeCapacity {
    slots: u32,
    bucket: u32,
}

@group(0) @binding(0) var<storage, read> source_indirect: array<u32>;
@group(0) @binding(1) var<storage, read_write> compacted_indirect: array<u32>;
@group(0) @binding(2) var<storage, read> range_counts: array<u32>;
@group(0) @binding(3) var<storage, read> opaque_ranges: array<Range>;
@group(0) @binding(4) var<storage, read> transparent_ranges: array<Range>;
@group(0) @binding(5) var<storage, read> forward_ranges: array<Range>;
@group(0) @binding(6) var<storage, read_write> draw_counts: array<u32>;
@group(0) @binding(7) var<uniform> capacity: RangeCapacity;

var<workgroup> survivor_count: atomic<u32>;

@compute @workgroup_size(64, 1, 1)
fn compact_ranges(
    @builtin(workgroup_id) workgroup: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let bucket = capacity.bucket;
    let range_index = workgroup.x + workgroup.y * 65535u;
    let bucket_count = range_counts[bucket];
    if range_index >= bucket_count {
        return;
    }

    var range: Range;
    if bucket == 0u {
        range = opaque_ranges[range_index];
    } else if bucket == 1u {
        range = transparent_ranges[range_index];
    } else {
        range = forward_ranges[range_index];
    }

    if local.x == 0u {
        atomicStore(&survivor_count, 0u);
    }
    workgroupBarrier();

    // Order within one material range has no semantic meaning. Atomic
    // allocation packs live records without a serial per-draw CPU loop.
    for (var i = local.x; i < range.count; i += 64u) {
        let source_slot = range.start + i;
        if source_indirect[source_slot * 5u + 1u] != 0u {
            let destination = range.start + atomicAdd(&survivor_count, 1u);
            for (var word = 0u; word < 5u; word += 1u) {
                compacted_indirect[destination * 5u + word] =
                    source_indirect[source_slot * 5u + word];
            }
        }
    }

    workgroupBarrier();
    let count = atomicLoad(&survivor_count);

    // Clear the unused tail. This keeps the non-count multi-draw fallback
    // correct while count-capable adapters skip the same empty slots.
    for (var i = count + local.x; i < range.count; i += 64u) {
        compacted_indirect[(range.start + i) * 5u + 1u] = 0u;
    }

    if local.x == 0u {
        let slot = 4u + bucket * capacity.slots + range_index;
        draw_counts[slot] = count;
    }
}
