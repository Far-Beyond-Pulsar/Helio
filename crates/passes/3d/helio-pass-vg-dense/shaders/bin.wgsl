// Tier-3-NonRT stage 1: bin visible clusters into the cluster lists of the
// dense tiles their bounding sphere can touch. One thread per visible cluster.
// The tile rectangle is `cluster_tile_rect` (bounds.wgsl).
//
// Three kernels, so every tile's list is exactly as long as it needs to be
// (a fixed per-tile capacity dropped geometry on crowds):
//   bin_count   per (cluster, tile): count[slot] += 1
//   bin_scan    one thread: start[slot] = running sum; counts are clamped to
//               what is left of the shared pool (the excess is counted in
//               STAT_BIN_OVERFLOW: the only way geometry is dropped)
//   bin_fill    per (cluster, tile): pool[start + cursor++] = entry
// `slot_lists` layout: LIST_HDR words per slot [count, start, cursor], then the
// pool at LIST_POOL_BASE.
@group(0) @binding(0) var<uniform> params: DenseParams;
@group(0) @binding(6) var<storage, read> tile_slot: array<u32>;
@group(0) @binding(7) var<storage, read_write> slot_lists: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> bin_args: array<u32>;

fn bin_visit(i: u32, fill: bool) {
    // Locate range r with prefix[r] <= i < prefix[r + 1] (args[20..33] hold the
    // prefix, args[8..20] the `first` of each cull range, see args.wgsl).
    let total = bin_args[ARGS_PREFIX + CULL_RANGE_COUNT];
    if i >= total { return; }
    var rg = 0u;
    for (var k = 1u; k < CULL_RANGE_COUNT; k++) {
        if bin_args[ARGS_PREFIX + k] <= i && bin_args[ARGS_PREFIX + k + 1u] > i { rg = k; }
    }
    let u = bin_args[8u + rg] + (i - bin_args[ARGS_PREFIX + rg]);
    // Range `rg` is `phase * BIN_COUNT + bin`: bins from CULL_FIRST_MASKED hold
    // alpha-tested clusters.
    let tagged = select(u, u | LIST_MASKED_BIT, (rg % (CULL_RANGE_COUNT / 2u)) >= CULL_FIRST_MASKED);
    let v = visible[u];
    let c = vgd_load_cluster(v.cluster_slot);
    let rect = cluster_tile_rect(v, c);
    if !rect.ok { return; }
    let tiles_x = params.viewport.z;
    for (var ty = rect.ty0; ty <= rect.ty1; ty++) {
        for (var tx = rect.tx0; tx <= rect.tx1; tx++) {
            let s = tile_slot[ty * tiles_x + tx];
            if s == SLOT_INVALID { continue; }
            if fill {
                let k = atomicAdd(&slot_lists[s * LIST_HDR + 2u], 1u);
                if k < atomicLoad(&slot_lists[s * LIST_HDR]) {
                    let start = atomicLoad(&slot_lists[s * LIST_HDR + 1u]);
                    atomicStore(&slot_lists[LIST_POOL_BASE + start + k], tagged);
                }
            } else {
                atomicAdd(&slot_lists[s * LIST_HDR], 1u);
            }
        }
    }
}

@compute @workgroup_size(64)
fn bin(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    bin_visit((wid.y * BIN_ROW_GROUPS + wid.x) * 64u + l.x, false);
}

@compute @workgroup_size(64)
fn bin_fill(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    bin_visit((wid.y * BIN_ROW_GROUPS + wid.x) * 64u + l.x, true);
}

// Serial prefix over the dense slots (at most `max_dense_slots`, typically tens).
@compute @workgroup_size(1)
fn bin_scan() {
    let n = min(atomicLoad(&state[STATE_DENSE_COUNT]), params.limits.y);
    var acc = 0u;
    for (var s = 0u; s < n; s++) {
        let c = atomicLoad(&slot_lists[s * LIST_HDR]);
        let eff = min(c, LIST_POOL_CAP - acc);
        atomicStore(&slot_lists[s * LIST_HDR], eff);
        atomicStore(&slot_lists[s * LIST_HDR + 1u], acc);
        atomicStore(&slot_lists[s * LIST_HDR + 2u], 0u);
        acc += eff;
        if c > eff { atomicAdd(&state[STAT_BIN_OVERFLOW], c - eff); }
    }
}
