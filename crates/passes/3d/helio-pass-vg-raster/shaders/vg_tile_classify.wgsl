// Tile overlap classification (plan 5.3).
//
// Compares the per-tile counters written by the software tier to thresholds
// and writes the dense-tile mask for this frame, plus a compact list of dense
// tiles and indirect dispatch args for Tier 3.
//
// A tile is dense when `tris >= tri_thresh` or `hits >= hit_thresh`
// (hits = pixel hits that passed the depth test, i.e. total depth complexity;
// 256 pixels per tile, so hit_thresh = 256 * K means average overdraw K).
// Hysteresis: a tile that was dense in the previous frame stays dense until
// both counters fall below `hysteresis * threshold` (hysteresis in (0, 1]).
//
// Dispatch `classify_main` over ceil(tile_count / 64) workgroups, then
// `dense_args_main` with one workgroup.
//
// Mask format: one u32 per tile, 1 = dense (row-major tile index).
// `dense_args` (u32 x 4): [0..3] = DispatchIndirect (x, y, z) for one workgroup
// per dense tile (2D: x <= 65535), [3] = dense tile count. Clear [3] each frame.

struct ClassifyParams {
    tile_count:  u32,
    tri_thresh:  u32,
    hit_thresh:  u32,
    hysteresis:  f32,
}

@group(0) @binding(0) var<uniform> cp: ClassifyParams;
@group(0) @binding(1) var<storage, read> tile_counters: array<u32>;
@group(0) @binding(2) var<storage, read> prev_mask: array<u32>;
@group(0) @binding(3) var<storage, read_write> cur_mask: array<u32>;
@group(0) @binding(4) var<storage, read_write> dense_list: array<u32>;
@group(0) @binding(5) var<storage, read_write> dense_args: array<atomic<u32>, 4>;

@compute @workgroup_size(64)
fn classify_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= cp.tile_count) { return; }
    let tris = f32(tile_counters[t * 2u]);
    let hits = f32(tile_counters[t * 2u + 1u]);
    let k = select(1.0, cp.hysteresis, prev_mask[t] != 0u);
    let dense = tris >= f32(cp.tri_thresh) * k || hits >= f32(cp.hit_thresh) * k;
    cur_mask[t] = select(0u, 1u, dense);
    if (dense) {
        let i = atomicAdd(&dense_args[3], 1u);
        dense_list[i] = t;
    }
}

@compute @workgroup_size(1)
fn dense_args_main() {
    let c = atomicLoad(&dense_args[3]);
    atomicStore(&dense_args[0], min(c, 65535u));
    atomicStore(&dense_args[1], max((c + 65534u) / 65535u, 1u));
    atomicStore(&dense_args[2], 1u);
}
