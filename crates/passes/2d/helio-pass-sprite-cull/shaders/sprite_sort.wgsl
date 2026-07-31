// ── GPU LSD Radix Sort (32 single-bit passes) ──────────────────────────────
//
// Three kernels per bit, dispatched in this order:
//
//   1. cs_histogram — one thread per element. Each workgroup (WG_SIZE=256)
//      counts how many of its elements have bit=0 vs bit=1, writing
//      `block_hist[workgroup_id * 2 + bit]`.
//
//   2. cs_scan — a single thread. Turns `block_hist` (per-block 0/1 counts)
//      into per-block-per-bit *global* output offsets, in place: total 0s
//      and total 1s summed across all blocks give the two global bucket
//      bases (all 0s first, then all 1s — this is what makes ascending-key
//      order fall out of doing this once per bit, LSB first), then a second
//      pass distributes each block's share within its bucket. Sequential
//      rather than parallel — see `helio-pass-sprite-cull`'s module doc
//      comment for why (small `num_blocks`, correctness over throughput).
//
//   3. cs_scatter — one thread per element, same workgroup layout as the
//      histogram pass. This is the one that has to be *stable*: LSD radix
//      sort's correctness proof requires every single-digit pass to
//      preserve the relative order of elements that share that digit's
//      value, because later (higher-order) passes are only correct if
//      earlier (lower-order) passes' relative ordering survived intact. An
//      8-bit-digit version of this kernel that used a workgroup-shared
//      *atomic* counter per bucket (`atomicAdd`) — the previous version of
//      this file — is NOT stable: GPU threads within a workgroup don't
//      execute in `local_invocation_id` order, so whichever thread's atomic
//      happened to run first claimed the lower slot, silently reordering
//      same-bucket elements and, worse, corrupting the sort of elements
//      that only *tied* on a higher digit but differed on a lower one (this
//      was caught by `tests/gpu_sort_validation.rs`, not by inspection).
//      The fix: a 1-bit split has exactly two buckets, so "how many earlier
//      threads in my block share my bucket" is answerable with a plain
//      Hillis-Steele inclusive prefix sum of a 0/1 predicate over
//      `local_invocation_id` — deterministic, not execution-order-dependent,
//      and small enough (8 barrier-synced steps for 256 threads) to not be
//      the bottleneck even at 32 passes instead of 4.

struct SortUniforms {
    bit: u32,
    num_blocks: u32,
}
struct CountUniform {
    count: u32,
}

const WG: u32 = 256u;

// ── cs_histogram ────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> su_h: SortUniforms;
@group(0) @binding(1) var<uniform> cu_h: CountUniform;
@group(0) @binding(2) var<storage, read> src_keys_h: array<u32>;
@group(0) @binding(3) var<storage, read_write> block_hist_h: array<u32>;

var<workgroup> hist_ones: atomic<u32>;
var<workgroup> hist_total: atomic<u32>;

@compute @workgroup_size(WG)
fn cs_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    if lid.x == 0u {
        atomicStore(&hist_ones, 0u);
        atomicStore(&hist_total, 0u);
    }
    workgroupBarrier();

    if gid.x < cu_h.count {
        atomicAdd(&hist_total, 1u);
        let bit = (src_keys_h[gid.x] >> su_h.bit) & 1u;
        if bit == 1u {
            atomicAdd(&hist_ones, 1u);
        }
    }
    workgroupBarrier();

    if lid.x == 0u {
        let ones = atomicLoad(&hist_ones);
        let total = atomicLoad(&hist_total);
        block_hist_h[wgid.x * 2u + 0u] = total - ones;
        block_hist_h[wgid.x * 2u + 1u] = ones;
    }
}

// ── cs_scan ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> su_s: SortUniforms;
@group(0) @binding(1) var<storage, read_write> block_hist_s: array<u32>;

@compute @workgroup_size(1)
fn cs_scan() {
    var total_zeros = 0u;
    var total_ones = 0u;
    for (var blk = 0u; blk < su_s.num_blocks; blk++) {
        total_zeros += block_hist_s[blk * 2u + 0u];
        total_ones += block_hist_s[blk * 2u + 1u];
    }
    let base_zero = 0u;
    let base_one = total_zeros;

    var running_zero = base_zero;
    var running_one = base_one;
    for (var blk = 0u; blk < su_s.num_blocks; blk++) {
        let cz = block_hist_s[blk * 2u + 0u];
        let co = block_hist_s[blk * 2u + 1u];
        block_hist_s[blk * 2u + 0u] = running_zero;
        block_hist_s[blk * 2u + 1u] = running_one;
        running_zero += cz;
        running_one += co;
    }
}

// ── cs_scatter ──────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> su_c: SortUniforms;
@group(0) @binding(1) var<uniform> cu_c: CountUniform;
@group(0) @binding(2) var<storage, read> src_keys_c: array<u32>;
@group(0) @binding(3) var<storage, read> src_indices_c: array<u32>;
@group(0) @binding(4) var<storage, read_write> dst_keys_c: array<u32>;
@group(0) @binding(5) var<storage, read_write> dst_indices_c: array<u32>;
@group(0) @binding(6) var<storage, read> block_offsets_c: array<u32>;

// Hillis-Steele inclusive scan buffer: `scan_buf[lid.x]` starts as "does my
// element have bit=1", and after the loop holds "count of bit=1 elements at
// local indices `0..=lid.x`" — a stable, index-order-derived rank, not a
// race-derived one.
var<workgroup> scan_buf: array<u32, 256>;

@compute @workgroup_size(WG)
fn cs_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let has_elem = gid.x < cu_c.count;
    let key = select(0u, src_keys_c[gid.x], has_elem);
    let bit = select(0u, (key >> su_c.bit) & 1u, has_elem);

    scan_buf[lid.x] = bit;
    workgroupBarrier();

    var offset = 1u;
    loop {
        if offset >= WG {
            break;
        }
        var v = 0u;
        if lid.x >= offset {
            v = scan_buf[lid.x - offset];
        }
        workgroupBarrier();
        scan_buf[lid.x] += v;
        workgroupBarrier();
        offset = offset * 2u;
    }
    let inclusive_ones = scan_buf[lid.x];

    if !has_elem {
        return;
    }

    var local_pos: u32;
    if bit == 1u {
        local_pos = inclusive_ones - 1u; // 0-based rank among this block's 1s
    } else {
        // Elements up to and including me: lid.x + 1. Of those, `inclusive_ones`
        // are 1s, so the rest are 0s; 0-based rank among 0s is that count minus one.
        local_pos = (lid.x + 1u - inclusive_ones) - 1u;
    }

    let base = block_offsets_c[wgid.x * 2u + bit];
    let dst = base + local_pos;
    dst_keys_c[dst] = key;
    dst_indices_c[dst] = src_indices_c[gid.x];
}
