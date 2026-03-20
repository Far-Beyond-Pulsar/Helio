// ─────────────────────────────────────────────────────────────────────────────
// Screen-Space SDF — Jump Flooding Algorithm (JFA)
//
// Three compute passes, zero CPU iteration:
//
//   Pass 0 — seed_pass      : mark foreground (surface) pixels from depth
//   Pass 1 — jfa_pass       : JFA flood-fill, dispatched log2(max_dim) times
//   Pass 2 — resolve_pass   : convert nearest-seed coords → signed distance
//
// Output: Rg16Float texture
//   .r = unsigned distance to nearest surface edge (screen pixels, normalised to [0,1])
//   .g = sign: +1.0 = outside surface (background), -1.0 = inside / on surface
//
// Coordinate convention: (0,0) = top-left, texel centres at (x+0.5, y+0.5).
// ─────────────────────────────────────────────────────────────────────────────

// ── Uniforms ─────────────────────────────────────────────────────────────────

struct SdfUniforms {
    // Render-target dimensions
    width:        u32,
    height:       u32,
    // JFA step size for this dispatch (halved each pass: max_dim/2, .../4, ...)
    step:         u32,
    // Linearise depth: near/far clip planes
    near:         f32,
    far:          f32,
    // Surface detection threshold (view-space units). Pixels whose linear
    // depth differs from their neighbours by more than this are "edges".
    edge_thresh:  f32,
    _pad0:        u32,
    _pad1:        u32,
}

@group(0) @binding(0) var<uniform> u: SdfUniforms;

// ── Textures / storage ───────────────────────────────────────────────────────

// Depth buffer (read-only in seed + resolve passes)
@group(0) @binding(1) var t_depth : texture_2d<f32>;

// Seed / JFA ping-pong buffers.
//   Each texel stores the (x, y) coordinates (as f32) of the nearest known
//   seed.  (−1, −1) means "no seed found yet".
@group(0) @binding(2) var t_seed_src : texture_storage_2d<rg32float, read>;
@group(0) @binding(3) var t_seed_dst : texture_storage_2d<rg32float, write>;

// Final output written by resolve_pass.
@group(0) @binding(4) var t_sdf_out  : texture_storage_2d<rg16float, write>;

// Sampler used to read raw depth (non-filtering, matching depth precision).
@group(0) @binding(5) var s_depth    : sampler;

// ── Helpers ───────────────────────────────────────────────────────────────────

const NO_SEED : vec2<f32> = vec2<f32>(-1.0, -1.0);
const UNORM_SCALE : f32 = 1.0 / 65504.0; // f16 max — used to normalise dist

// Linearise a raw [0,1] depth to view-space distance.
fn linear_depth(raw: f32) -> f32 {
    // Reversed-Z: far=0, near=1 —— standard formula still works when near/far
    // are passed correctly for the projection in use.
    return (u.near * u.far) / (u.far - raw * (u.far - u.near));
}

fn load_raw_depth(coord: vec2<i32>) -> f32 {
    return textureLoad(t_depth, coord, 0).r;
}

fn is_surface(coord: vec2<i32>) -> bool {
    let d = load_raw_depth(coord);
    // Anything outside [near, far) is considered sky / empty
    return d > 0.0 && d < 1.0;
}

// Detect silhouette / edge by comparing a pixel's depth to its 4-neighbours.
fn is_edge(coord: vec2<i32>) -> bool {
    let dim = vec2<i32>(i32(u.width), i32(u.height));
    let d   = linear_depth(load_raw_depth(coord));
    let offsets = array<vec2<i32>, 4>(
        vec2<i32>( 1, 0), vec2<i32>(-1, 0),
        vec2<i32>( 0, 1), vec2<i32>( 0,-1),
    );
    for (var k = 0u; k < 4u; k++) {
        let nb = coord + offsets[k];
        if nb.x < 0 || nb.y < 0 || nb.x >= dim.x || nb.y >= dim.y {
            // Boundary pixels are always seeds
            return true;
        }
        let dn = linear_depth(load_raw_depth(nb));
        if abs(d - dn) > u.edge_thresh { return true; }
    }
    return false;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 0 — Seed initialisation
//
// Each thread writes to t_seed_dst:
//   • On-surface edge pixels  → seed at their own coordinate
//   • Everything else         → NO_SEED (−1, −1)
//
// Workgroup: 8×8
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn seed_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    if coord.x >= i32(u.width) || coord.y >= i32(u.height) { return; }

    var seed = NO_SEED;
    if is_surface(coord) && is_edge(coord) {
        seed = vec2<f32>(f32(coord.x), f32(coord.y));
    }
    textureStore(t_seed_dst, coord, vec4<f32>(seed, 0.0, 0.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 1 — Jump Flooding
//
// Classic JFA: for each pixel sample 8 neighbours at ±step offsets and keep
// the nearest seed.  Repeat with step = max_dim/2 → 1 (log2 dispatches).
//
// Workgroup: 8×8
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn jfa_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord   = vec2<i32>(i32(gid.x), i32(gid.y));
    let dim     = vec2<i32>(i32(u.width), i32(u.height));
    if coord.x >= dim.x || coord.y >= dim.y { return; }

    let pos   = vec2<f32>(f32(coord.x), f32(coord.y));
    let step  = i32(u.step);

    // Start with our own current best seed
    var best_seed = textureLoad(t_seed_src, coord).xy;
    var best_dist = select(
        distance(pos, best_seed),
        1e30,
        all(best_seed == NO_SEED),
    );

    // Sample 3×3 neighbourhood at jump-step offsets
    for (var dy = -1; dy <= 1; dy++) {
        for (var dx = -1; dx <= 1; dx++) {
            let nb = coord + vec2<i32>(dx, dy) * step;
            if nb.x < 0 || nb.y < 0 || nb.x >= dim.x || nb.y >= dim.y { continue; }

            let nb_seed = textureLoad(t_seed_src, nb).xy;
            if all(nb_seed == NO_SEED) { continue; }

            let d = distance(pos, nb_seed);
            if d < best_dist {
                best_dist = d;
                best_seed = nb_seed;
            }
        }
    }

    textureStore(t_seed_dst, coord, vec4<f32>(best_seed, 0.0, 0.0));
}

// ─────────────────────────────────────────────────────────────────────────────
// Pass 2 — Resolve: nearest-seed → signed distance
//
// Reads final seed buffer, computes Euclidean distance, then signs it:
//   outside surface (background / sky) → positive
//   on or inside surface               → negative
//
// Writes t_sdf_out:
//   .r = distance [0, 1] (normalised to half of max screen dimension)
//   .g = sign    (+1.0 / −1.0)
//
// Workgroup: 8×8
// ─────────────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8)
fn resolve_pass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = vec2<i32>(i32(gid.x), i32(gid.y));
    let dim   = vec2<i32>(i32(u.width), i32(u.height));
    if coord.x >= dim.x || coord.y >= dim.y { return; }

    let pos       = vec2<f32>(f32(coord.x), f32(coord.y));
    let seed      = textureLoad(t_seed_src, coord).xy;
    let max_dim   = f32(max(u.width, u.height));

    var dist_px : f32;
    if all(seed == NO_SEED) {
        // Unflooded pixel — maximum distance
        dist_px = max_dim;
    } else {
        dist_px = distance(pos, seed);
    }

    // Normalise to [0, 1] using half the screen diagonal as reference
    let dist_norm = clamp(dist_px / (max_dim * 0.5), 0.0, 1.0);

    // Sign: surface pixels are negative (inside), background is positive
    let on_surface = is_surface(coord);
    let sign_val   = select(1.0, -1.0, on_surface);

    textureStore(t_sdf_out, coord, vec4<f32>(dist_norm, sign_val, 0.0, 0.0));
}
