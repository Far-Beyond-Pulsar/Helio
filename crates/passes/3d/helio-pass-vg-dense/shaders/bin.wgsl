// Tier-3-NonRT stage 1: bin visible clusters into the cluster lists of the
// dense tiles their bounding sphere can touch. One thread per visible cluster.
//
// The sphere bound is conservative (see `sphere_tile_bounds` in
// `src/peel_ref.rs`, the CPU mirror that tests it). A cluster straddling the
// near plane is binned into every tile. Lists are capped at CLUSTER_CAP per
// tile; past the cap clusters are dropped and `STAT_BIN_OVERFLOW` counts them.
@group(0) @binding(0) var<uniform> params: DenseParams;
@group(0) @binding(6) var<storage, read> tile_slot: array<u32>;
@group(0) @binding(7) var<storage, read_write> slot_lists: array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write> state: array<atomic<u32>>;
@group(0) @binding(11) var<storage, read> bin_args: array<u32>;

@compute @workgroup_size(64)
fn bin(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let i = (wid.y * BIN_ROW_GROUPS + wid.x) * 64u + l.x;
    // Locate range r with prefix[r] <= i < prefix[r + 1] (args[20..33] hold the
    // prefix, args[8..20] the `first` of each cull range, see args.wgsl).
    let total = bin_args[20u + CULL_RANGE_COUNT];
    if i >= total { return; }
    var rg = 0u;
    for (var k = 1u; k < CULL_RANGE_COUNT; k++) {
        if bin_args[20u + k] <= i && bin_args[20u + k + 1u] > i { rg = k; }
    }
    let u = bin_args[8u + rg] + (i - bin_args[20u + rg]);
    let v = visible[u];
    let c = vgd_load_cluster(v.cluster_slot);
    let m = dense_xform(v);
    let wc = (m * vec4<f32>(c.sphere.xyz, 1.0)).xyz;
    let sc = max(length(m[0].xyz), max(length(m[1].xyz), length(m[2].xyz)));
    let r = c.sphere.w * sc;
    let vp = params.view_proj;
    let cc = vp * vec4<f32>(wc, 1.0);
    let lx = length(vec3<f32>(vp[0][0], vp[1][0], vp[2][0]));
    let ly = length(vec3<f32>(vp[0][1], vp[1][1], vp[2][1]));
    let lw = length(vec3<f32>(vp[0][3], vp[1][3], vp[2][3]));
    let wmin = cc.w - r * lw;
    let wmax = cc.w + r * lw;
    if wmax <= 0.0 { return; }
    let tiles_x = params.viewport.z;
    let tiles_y = params.viewport.w;
    var tx0 = 0u;
    var ty0 = 0u;
    var tx1 = tiles_x - 1u;
    var ty1 = tiles_y - 1u;
    if wmin > 1.0e-4 {
        let xlo = cc.x - r * lx;
        let xhi = cc.x + r * lx;
        let ylo = cc.y - r * ly;
        let yhi = cc.y + r * ly;
        let nx0 = min(xlo / wmin, xlo / wmax);
        let nx1 = max(xhi / wmin, xhi / wmax);
        let ny0 = min(ylo / wmin, ylo / wmax);
        let ny1 = max(yhi / wmin, yhi / wmax);
        let fw = f32(params.viewport.x);
        let fh = f32(params.viewport.y);
        let px0 = (nx0 * 0.5 + 0.5) * fw;
        let px1 = (nx1 * 0.5 + 0.5) * fw;
        let py0 = (0.5 - ny1 * 0.5) * fh;
        let py1 = (0.5 - ny0 * 0.5) * fh;
        if px1 < 0.0 || px0 >= fw || py1 < 0.0 || py0 >= fh { return; }
        let ft = f32(TILE);
        tx0 = u32(clamp(floor(px0 / ft), 0.0, f32(tiles_x - 1u)));
        tx1 = u32(clamp(floor(px1 / ft), 0.0, f32(tiles_x - 1u)));
        ty0 = u32(clamp(floor(py0 / ft), 0.0, f32(tiles_y - 1u)));
        ty1 = u32(clamp(floor(py1 / ft), 0.0, f32(tiles_y - 1u)));
    }
    for (var ty = ty0; ty <= ty1; ty++) {
        for (var tx = tx0; tx <= tx1; tx++) {
            let s = tile_slot[ty * tiles_x + tx];
            if s == SLOT_INVALID { continue; }
            let k = atomicAdd(&slot_lists[s * SLOT_STRIDE], 1u);
            if k < CLUSTER_CAP {
                atomicStore(&slot_lists[s * SLOT_STRIDE + 1u + k], u);
            } else {
                atomicAdd(&state[STAT_BIN_OVERFLOW], 1u);
            }
        }
    }
}
