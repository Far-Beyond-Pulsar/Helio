// Stable GPU LSD radix sort of (key, index) pairs: 32-bit keys, 4-bit digits,
// eight passes of three kernels (`helio_core::radix_sort`).
//
// The caller's own prepare step writes `frame` (the live count and its blocks
// of WG elements) and the indirect dispatch size of `cs_histogram` and
// `cs_scatter` (one workgroup a block); `cs_scan` is one workgroup.
//
//   cs_histogram  each block's count of each digit, digit-major:
//                 `hist[digit * num_blocks + block]`.
//   cs_scan       the exclusive prefix sum of `hist` in that order: each
//                 block's first output slot for each digit (all of digit 0's
//                 blocks first, in block order, then digit 1's, ...).
//   cs_scatter    each element to its block's slot for its digit plus its
//                 rank among the block's earlier elements of that digit.
//
// Stability (an LSD sort is only correct if every pass keeps equal digits in
// order) comes from that rank: an exclusive prefix sum over the block's
// threads in `local_invocation_id` order, never an atomic (whose order is the
// threads' execution order). The sum counts the sixteen digits at once, eight
// bits each, in a vec4<u32>: an exclusive count within a block of 256 is at
// most 255, so no lane carries into the next.

const WG: u32 = 256u;
const RADIX: u32 = 16u;

struct SortPass {
    shift: u32,
}

struct Frame {
    count: u32,
    num_blocks: u32,
}

@group(0) @binding(0) var<uniform> sort_pass: SortPass;
@group(0) @binding(1) var<storage, read> frame: Frame;
@group(0) @binding(2) var<storage, read> src_keys: array<u32>;
@group(0) @binding(3) var<storage, read> src_indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst_keys: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst_indices: array<u32>;
@group(0) @binding(6) var<storage, read_write> hist: array<u32>;

var<workgroup> digit_counts: array<atomic<u32>, RADIX>;

@compute @workgroup_size(WG)
fn cs_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    if lid.x < RADIX {
        atomicStore(&digit_counts[lid.x], 0u);
    }
    workgroupBarrier();
    if gid.x < frame.count {
        atomicAdd(&digit_counts[(src_keys[gid.x] >> sort_pass.shift) & (RADIX - 1u)], 1u);
    }
    workgroupBarrier();
    if lid.x < RADIX {
        hist[lid.x * frame.num_blocks + wg.x] = atomicLoad(&digit_counts[lid.x]);
    }
}

var<workgroup> chunk_sums: array<u32, WG>;

@compute @workgroup_size(WG)
fn cs_scan(@builtin(local_invocation_id) lid: vec3<u32>) {
    // Each thread a contiguous run of `hist`, then a scan of the runs' sums.
    let n = RADIX * frame.num_blocks;
    let per = (n + WG - 1u) / WG;
    let start = min(lid.x * per, n);
    let end = min(start + per, n);
    var total = 0u;
    for (var i = start; i < end; i++) {
        total += hist[i];
    }
    chunk_sums[lid.x] = total;
    workgroupBarrier();
    for (var offset = 1u; offset < WG; offset *= 2u) {
        var v = 0u;
        if lid.x >= offset {
            v = chunk_sums[lid.x - offset];
        }
        workgroupBarrier();
        chunk_sums[lid.x] += v;
        workgroupBarrier();
    }
    var running = chunk_sums[lid.x] - total;
    for (var i = start; i < end; i++) {
        let c = hist[i];
        hist[i] = running;
        running += c;
    }
}

var<workgroup> ranks: array<vec4<u32>, WG>;

// Digit `d` as one count in the packed vec4 (eight bits a digit).
fn one_hot(d: u32) -> vec4<u32> {
    let bit = 1u << ((d & 3u) * 8u);
    return select(vec4<u32>(0u), vec4<u32>(bit), vec4<u32>(0u, 1u, 2u, 3u) == vec4<u32>(d >> 2u));
}

@compute @workgroup_size(WG)
fn cs_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
) {
    let has = gid.x < frame.count;
    var key = 0u;
    var d = 0u;
    if has {
        key = src_keys[gid.x];
        d = (key >> sort_pass.shift) & (RADIX - 1u);
    }
    // Exclusive: each thread starts from the one before it (elements past
    // the count are only ever at a block's end and count nothing).
    let mine = select(vec4<u32>(0u), one_hot(d), has);
    ranks[lid.x] = vec4<u32>(0u);
    workgroupBarrier();
    if lid.x + 1u < WG {
        ranks[lid.x + 1u] = mine;
    }
    workgroupBarrier();
    for (var offset = 1u; offset < WG; offset *= 2u) {
        var v = vec4<u32>(0u);
        if lid.x >= offset {
            v = ranks[lid.x - offset];
        }
        workgroupBarrier();
        ranks[lid.x] += v;
        workgroupBarrier();
    }
    if !has {
        return;
    }
    let rank = (ranks[lid.x][d >> 2u] >> ((d & 3u) * 8u)) & 0xffu;
    let dst = hist[d * frame.num_blocks + wg.x] + rank;
    dst_keys[dst] = key;
    dst_indices[dst] = src_indices[gid.x];
}
