// ── GPU LSD Radix Sort (4 passes over 8-bit digits of a 32-bit key) ────────
//
// Three kernels, dispatched in this order per digit pass:
//
//   1. cs_histogram — one thread per element. Each workgroup (WG_SIZE=256,
//      chosen to equal the bucket count) builds a local 256-bucket histogram
//      in workgroup-shared memory, then writes it out to
//      `block_hist[workgroup_id * 256 + bucket]`.
//
//   2. cs_scan — a single thread. Turns `block_hist` (per-block counts) into
//      per-block-per-bucket *global* output offsets, in place: bucket totals
//      summed across all blocks, exclusive-prefix-summed into per-bucket base
//      offsets, then a second pass distributes each block's share within its
//      bucket. Deliberately sequential rather than a parallel scan — at the
//      scale this is meant for (sorting a *culled, visible* sprite count,
//      not the whole pool; see `SpriteCullPass`), `num_blocks * 256` is small
//      enough that single-thread throughput is not the bottleneck, and a
//      sequential scan is far less likely to hide a subtle correctness bug
//      than a parallel work-efficient one.
//
//   3. cs_scatter — one thread per element, same workgroup layout as the
//      histogram pass. Each workgroup loads its block's global base offsets
//      (from step 2) into workgroup-shared *atomic* counters, then every
//      thread with a live element atomically claims the next free slot in
//      its own bucket's counter and writes there. This is what turns
//      "known base offset per (block, bucket)" + "local claim order" into an
//      exact final position without a second local scan.
//
// Not a stable sort: two elements with the *equal* keys can end up in either
// relative order (GPU thread scheduling within a workgroup isn't required to
// match `global_invocation_id` order, unlike the CPU version in `src/lib.rs`
// which explicitly preserves input order on ties). Keys that differ are
// still placed in strictly correct ascending order — only genuine ties are
// affected, which for a depth-sort of alpha-blended sprites is an
// acceptably rare, low-consequence case (see `SpriteInstance::depth`'s doc
// comment on why sorting matters at all here).

struct SortUniforms {
    shift: u32,
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

var<workgroup> local_hist: array<atomic<u32>, 256>;

@compute @workgroup_size(WG)
fn cs_histogram(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    atomicStore(&local_hist[lid.x], 0u);
    workgroupBarrier();

    if gid.x < cu_h.count {
        let bucket = (src_keys_h[gid.x] >> su_h.shift) & 0xFFu;
        atomicAdd(&local_hist[bucket], 1u);
    }
    workgroupBarrier();

    block_hist_h[wgid.x * 256u + lid.x] = atomicLoad(&local_hist[lid.x]);
}

// ── cs_scan ─────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> su_s: SortUniforms;
@group(0) @binding(1) var<storage, read_write> block_hist_s: array<u32>;

@compute @workgroup_size(1)
fn cs_scan() {
    var bucket_total: array<u32, 256>;
    for (var b = 0u; b < 256u; b++) {
        bucket_total[b] = 0u;
    }
    for (var blk = 0u; blk < su_s.num_blocks; blk++) {
        for (var b = 0u; b < 256u; b++) {
            bucket_total[b] += block_hist_s[blk * 256u + b];
        }
    }

    var running_per_bucket: array<u32, 256>;
    var running = 0u;
    for (var b = 0u; b < 256u; b++) {
        running_per_bucket[b] = running;
        running += bucket_total[b];
    }

    for (var blk = 0u; blk < su_s.num_blocks; blk++) {
        for (var b = 0u; b < 256u; b++) {
            let idx = blk * 256u + b;
            let cnt = block_hist_s[idx];
            block_hist_s[idx] = running_per_bucket[b];
            running_per_bucket[b] += cnt;
        }
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

var<workgroup> local_base: array<atomic<u32>, 256>;

@compute @workgroup_size(WG)
fn cs_scatter(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    atomicStore(&local_base[lid.x], block_offsets_c[wgid.x * 256u + lid.x]);
    workgroupBarrier();

    if gid.x < cu_c.count {
        let key = src_keys_c[gid.x];
        let bucket = (key >> su_c.shift) & 0xFFu;
        let pos = atomicAdd(&local_base[bucket], 1u);
        dst_keys_c[pos] = key;
        dst_indices_c[pos] = src_indices_c[gid.x];
    }
}
