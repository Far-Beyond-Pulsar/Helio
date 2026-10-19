//!use helio_prelude
// Temporal Super-Resolution: one resolve draw and one sharpening draw.
//
// Resolve, per output pixel:
//   1. Reconstruct this frame's colour from the jittered input samples around
//      it with a Gaussian of the output pixel, and keep the summed kernel
//      weight: how much this frame actually says about the pixel.
//   2. Dilate depth (nearest of the 3x3) and reproject with it, so an edge
//      moves with its foreground surface.
//   3. Reject history on disocclusion (reprojected vs stored view depth).
//   4. Sample history with Catmull-Rom, de-ringed to its own 2x2.
//   5. Rectify: clip history toward the mean of the input neighbourhood,
//      against mean +/- one standard deviation (YCoCg, tonemapped). How far
//      outside that box history was scales down its accumulated weight, so
//      a changed pixel restarts accumulation where it changed.
//   6. Detect change the box cannot see. Old and new can both fit in it (an
//      edge whose two sides both brighten), and aliasing makes single frames
//      unreliable judges. So compare this frame and history low-passed at the
//      same sample positions: for a still scene that residual is the jitter's
//      noise and averages to zero; a change (a light switching, an animated
//      material, an object without motion vectors) biases it. Its running
//      mean, against the running deviation, drops history once the bias is
//      significant, within a few frames even where per-frame noise is large.
//   7. Thin features: a sub-pixel ridge that this frame's samples happened to
//      miss would be clipped away and come back when the jitter hits it
//      again, every few frames. A ridge seen in the input locks the pixel for
//      a short lifetime, during which the clip yields to history. A detected
//      change (6) releases the lock, so a thin object that moved away does
//      not stay behind; so does the locked ridge seen again with another
//      luminance (a light changing on a wire, where the residual is too
//      noisy to tell).
//   8. Accumulate as a weighted running mean: history weight vs this frame's
//      kernel weight, capped at MAX_HISTORY_WEIGHT. Converges at 1/n from a
//      disocclusion and integrates the whole jitter sequence once stable.
//      Blending happens in a reversible max-channel tonemap, so a single
//      bright sample cannot dominate (anti-firefly weighting).
// Sharpen: RCAS on the accumulated history. History itself stays
// unsharpened (no feedback), and sharpening only the converged image cannot
// put the jitter back.
//
// References:
//   FSR 2 — https://gpuopen.com/fidelityfx-super-resolution-2 (MIT): dilation,
//     weight accumulation, thin-feature locks, RCAS
//   UE5 TSR — https://dev.epicgames.com/documentation/en-us/unreal-engine/temporal-super-resolution-in-unreal-engine
//     (rejection by comparing input and history low-passed on one grid)
//   Salvi, "An Excursion in Temporal Supersampling" (GDC 2016): variance clipping
//   Karis, "High Quality Temporal Supersampling" (SIGGRAPH 2014): tonemapped blend
//   Playdead temporal — https://github.com/playdeadgames/temporal (MIT): clip toward mean

// ── Constants ─────────────────────────────────────────────────────────────────

// Reconstruction kernel: Gaussian sigma in output pixels. Converged, the
// image is the scene filtered by this kernel (weights are accumulated, not
// renormalized per frame), so it sets the trade between sharpness and
// residual aliasing; 0.4 is close to a 1-pixel box with smooth tails.
const KERNEL_SIGMA: f32 = 0.4;
// Weighted running mean cap, in units of a sample landing on the pixel
// centre. At native resolution a frame contributes about 1, so this is the
// number of frames integrated once stable; upscaled frames contribute less
// and integrate proportionally more of them. Change detection, not this
// cap, bounds how long stale history survives.
const MAX_HISTORY_WEIGHT: f32 = 64.0;
// Rectification box half-size in standard deviations of the neighbourhood.
const VARIANCE_CLIP_GAMMA: f32 = 1.0;
// Box floor in tonemapped YCoCg: below half-float noise of the input.
const MIN_BOX_EXTENT: f32 = 1.0 / 1024.0;
// Low-pass sigma (output pixels) of the change detector: wide enough to
// average the jitter, narrow enough for the 3x3 the clip box reads anyway.
const LOW_PASS_SIGMA: f32 = 1.0;
// Rate of the detector's running mean (about 4 frames), and the slower one
// at which it learns the residual's variance, from residuals clamped to
// CHANGE_OUTLIER deviations: a change acts within frames, before it can teach
// the detector that residuals that large are noise.
const CHANGE_RATE: f32 = 0.25;
const NOISE_RATE: f32 = 0.0625;
const CHANGE_OUTLIER: f32 = 3.0;
// The running mean must exceed this many of its own standard deviations...
const CHANGE_SIGMAS: f32 = 3.0;
// ...and this fraction of the luma (tonemapped; absolute below the floor).
const CHANGE_RELATIVE: f32 = 0.05;
const CHANGE_FLOOR: f32 = 1.0 / 64.0;
// Frames a thin-feature lock survives without seeing its ridge again: longer
// than the jitter takes to hit a sub-pixel feature again.
const LOCK_FRAMES: f32 = 8.0;
// Relative change of a locked ridge's tonemapped luma that marks what history
// holds of it stale (absolute below the change floor).
const RIDGE_CHANGE: f32 = 0.1;
// Luminance ratio between neighbours treated as the same surface when
// looking for thin ridges (FSR 2's 1.05).
const RIDGE_SIMILARITY: f32 = 1.05;
// RCAS strength in stops below its maximum (0 = strongest).
const SHARPEN_STOPS: f32 = 0.5;
// RCAS lobe limit, 0.25 - 1/16.
const RCAS_LIMIT: f32 = 0.1875;

// ── Bindings ──────────────────────────────────────────────────────────────────

@group(0) @binding(0) var current_frame:  texture_2d<f32>;  // pre-AA at internal res
@group(0) @binding(1) var history_frame:  texture_2d<f32>;  // accumulated linear HDR + coverage
@group(0) @binding(2) var depth_tex:      texture_depth_2d; // depth at internal res
@group(0) @binding(3) var linear_sampler: sampler;
@group(0) @binding(5) var<storage, read> cameras: array<Camera, 2>;

struct TsrUniform {
    jitter_offset:  vec2<f32>, // this frame's jitter in input pixels
    reactivity:     f32,       // fraction of history confidence dropped per frame
    flags:          u32,       // bit 0: reset, bit 1: transparency coverage
    previous_jitter_uv: vec2<f32>,
    _pad: vec2<f32>,
    previous_view: mat4x4<f32>,
}
@group(0) @binding(6) var<uniform> tsr: TsrUniform;
@group(0) @binding(7) var history_depth_tex: texture_2d<f32>; // dilated view depth, 0 = sky
@group(0) @binding(8) var transparency_reactivity: texture_2d<f32>;
// x: accumulated weight, y: thin-feature lock lifetime (0..1), z: the locked
// ridge's tonemapped luma (negative for a dark ridge), w: the change
// detector's running mean residual.
@group(0) @binding(9) var history_state: texture_2d<f32>;
// The change detector's learnt residual variance.
@group(0) @binding(11) var history_noise: texture_2d<f32>;

// Sharpening pass input: this frame's resolve.
@group(0) @binding(10) var resolved: texture_2d<f32>;

// ── Vertex passthrough ────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

// ── Colour space ──────────────────────────────────────────────────────────────

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(
        dot(rgb, vec3<f32>( 0.25, 0.5,  0.25)),
        dot(rgb, vec3<f32>( 0.5,  0.0, -0.5 )),
        dot(rgb, vec3<f32>(-0.25, 0.5, -0.25)),
    );
}

fn ycocg_to_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

fn max3(v: vec3<f32>) -> f32 { return max(v.r, max(v.g, v.b)); }

// Reversible max-channel Reinhard: [0, inf) -> [0, 1). Averaging here weights
// each colour by 1 / (1 + max channel).
fn tonemap(c: vec3<f32>) -> vec3<f32> { return c / (max3(c) + 1.0); }
fn reverse_tonemap(c: vec3<f32>) -> vec3<f32> {
    // Half-float history holds at most 65504; keep the inverse finite.
    let t = min(c, vec3<f32>(0.9999));
    return t / (1.0 - max3(t));
}

fn perceptual(rgb: vec3<f32>) -> vec3<f32> { return rgb_to_ycocg(tonemap(max(rgb, vec3<f32>(0.0)))); }
fn linear_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    return reverse_tonemap(clamp(ycocg_to_rgb(ycocg), vec3<f32>(0.0), vec3<f32>(1.0)));
}

// ── History sampling ──────────────────────────────────────────────────────────

fn catmull_rom_weights(f: vec2<f32>) -> array<vec2<f32>, 4> {
    return array<vec2<f32>, 4>(
        f * (-0.5 + f * (1.0 - 0.5 * f)),
        1.0 + f * f * (-2.5 + 1.5 * f),
        f * (0.5 + f * (2.0 - 1.5 * f)),
        f * f * (-0.5 + 0.5 * f),
    );
}

// Catmull-Rom over the 4x4 texels around `uv`, clamped to the range of the
// inner 2x2 so its negative lobes cannot ring (a bright texel next to a dark
// one would otherwise leave a dark halo that only rectification removes).
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let dims = vec2<i32>(textureDimensions(history_frame));
    let p = uv * vec2<f32>(dims) - 0.5;
    let base = vec2<i32>(floor(p));
    let w = catmull_rom_weights(p - floor(p));
    var sum = vec3<f32>(0.0);
    var lo = vec3<f32>(3.0e38);
    var hi = vec3<f32>(0.0);
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let at = clamp(base + vec2<i32>(x - 1, y - 1), vec2<i32>(0), dims - 1);
            let s = textureLoad(history_frame, at, 0).rgb;
            sum += s * (w[x].x * w[y].y);
            if (x == 1 || x == 2) && (y == 1 || y == 2) {
                lo = min(lo, s);
                hi = max(hi, s);
            }
        }
    }
    return clamp(sum, lo, hi);
}

// ── Reprojection ──────────────────────────────────────────────────────────────

// Color and depth refer to the same jittered raster position. History is
// resolved into an unjittered display grid, so remove the prior projection's
// jitter after reprojection. Keep clip W for behind-camera rejection.
fn reproject_history(raster_uv: vec2<f32>, depth: f32) -> vec4<f32> {
    let ndc = raster_uv * vec2<f32>(2.0,-2.0) + vec2<f32>(-1.0,1.0);
    let world_h = cameras[0].view_proj_inv * vec4<f32>(ndc,depth,1.0);
    // Sky texels reproject as directions: the homogeneous point minus the
    // eye, exact for any w including the w = 0 of depth 0 (infinity).
    // Sky has no history depth.
    let sky = helio_is_sky(depth);
    let point = select(vec4<f32>(world_h.xyz/world_h.w,1.0),
        vec4<f32>(world_h.xyz - world_h.w*cameras[0].position_near.xyz,0.0), sky);
    let prev_clip = cameras[0].prev_view_proj * point;
    let uv = prev_clip.xy/prev_clip.w * vec2<f32>(0.5,-0.5)+0.5;
    let previous_position = tsr.previous_view * point;
    return vec4<f32>(uv-tsr.previous_jitter_uv,prev_clip.w,select(-previous_position.z,0.0,sky));
}

fn history_depth_matches(expected: f32, stored: f32, tolerance: f32) -> bool {
    return expected > 0.0 && stored > 0.0 && abs(expected-stored) <= tolerance;
}

// Positive view depth of a raw depth sample, 0 for sky (matching history).
fn view_depth(raster_uv: vec2<f32>, depth: f32) -> f32 {
    let world_h = cameras[0].view_proj_inv * vec4<f32>(helio_uv_to_ndc(raster_uv), depth, 1.0);
    let view_position = cameras[0].view * vec4<f32>(world_h.xyz / world_h.w, 1.0);
    return select(max(-view_position.z, 0.0), 0.0, helio_is_sky(depth));
}

// ── Input neighbourhood ───────────────────────────────────────────────────────

struct Neighbourhood {
    colour: vec3<f32>,      // kernel-weighted reconstruction, perceptual YCoCg
    weight: f32,            // summed kernel weight (1 = a sample on the centre)
    mean: vec3<f32>,        // of the 3x3 around the nearest texel
    deviation: vec3<f32>,
    nearest_depth: f32,     // dilated: the nearest raw depth of that 3x3
    // A thin feature crossing a texel this pixel draws on, the nearest such:
    // its tonemapped luma, negative for a dark one, 0 for none.
    ridge: f32,
    low_luma: f32,          // luma low-passed by the change detector's kernel
}

// Gaussian weights, per axis (the kernels are separable), of the four
// texels from `first` on, `pos` the pixel centre in input texels.
fn gaussian4(first: f32, pos: f32, to_output: f32, sigma: f32) -> vec4<f32> {
    let d = (vec4<f32>(first) + vec4<f32>(0.5, 1.5, 2.5, 3.5) - pos) * to_output;
    return exp(-0.5 * d * d / (sigma * sigma));
}

// `in_pos` is the output pixel centre in input texels, jitter included. The
// 4x4 texels around it hold every sample the kernel weighs, and the 3x3 of
// each of the 2x2 nearest, which the thin-feature test needs.
fn gather(in_pos: vec2<f32>) -> Neighbourhood {
    let dims = vec2<i32>(textureDimensions(current_frame));
    let origin = vec2<i32>(floor(in_pos - 0.5)) - 1;
    let nearest = vec2<i32>(floor(in_pos)) - origin; // 1 or 2 per axis
    let to_output = vec2<f32>(textureDimensions(history_frame)) / vec2<f32>(dims);
    let kx = gaussian4(f32(origin.x), in_pos.x, to_output.x, KERNEL_SIGMA);
    let ky = gaussian4(f32(origin.y), in_pos.y, to_output.y, KERNEL_SIGMA);
    let lx = gaussian4(f32(origin.x), in_pos.x, to_output.x, LOW_PASS_SIGMA);
    let ly = gaussian4(f32(origin.y), in_pos.y, to_output.y, LOW_PASS_SIGMA);
    var n: Neighbourhood;
    // Luma rows of the 4x4, kept in registers for the ridge test.
    var rows = mat4x4<f32>();
    var sum = vec3<f32>(0.0);
    var low = vec2<f32>(0.0);
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var depth = 0.0;
    for (var y = 0; y < 4; y++) {
        var row = vec4<f32>(0.0);
        for (var x = 0; x < 4; x++) {
            let p = clamp(origin + vec2<i32>(x, y), vec2<i32>(0), dims - 1);
            let q = perceptual(textureLoad(current_frame, p, 0).rgb);
            sum += kx[x] * ky[y] * q;
            row[x] = q.x;
            if all(abs(vec2<i32>(x, y) - nearest) <= vec2<i32>(1)) {
                m1 += q;
                m2 += q * q;
                low += lx[x] * ly[y] * vec2<f32>(q.x, 1.0);
                depth = helio_depth_nearest(depth, textureLoad(depth_tex, p, 0));
            }
        }
        rows[y] = row;
    }
    n.weight = dot(kx, vec4<f32>(1.0)) * dot(ky, vec4<f32>(1.0));
    n.colour = sum / max(n.weight, 1.0e-6);
    n.mean = m1 / 9.0;
    n.deviation = sqrt(max(m2 / 9.0 - n.mean * n.mean, vec3<f32>(0.0)));
    n.nearest_depth = depth;
    n.low_luma = low.x / low.y;
    // The ridge nearest the pixel centre among the four texels it draws on.
    let ridges = vec4<f32>(
        thin_ridge_polarity(rows[0].xyz, rows[1].xyz, rows[2].xyz) * rows[1].y,
        thin_ridge_polarity(rows[0].yzw, rows[1].yzw, rows[2].yzw) * rows[1].z,
        thin_ridge_polarity(rows[1].xyz, rows[2].xyz, rows[3].xyz) * rows[2].y,
        thin_ridge_polarity(rows[1].yzw, rows[2].yzw, rows[3].yzw) * rows[2].z,
    );
    let weights = vec4<f32>(kx.yz * ky.y, kx.yz * ky.z);
    var best = 0.0;
    for (var i = 0; i < 4; i++) {
        if ridges[i] != 0.0 && weights[i] > best {
            best = weights[i];
            n.ridge = ridges[i];
        }
    }
    return n;
}

// FSR 2's thin-feature test on a 3x3 of luma, by rows: the centre is
// brighter or darker than every neighbour unlike it, and no 2x2 quad around
// the centre is all alike (the feature is under two texels thick). 1 for a
// bright ridge, -1 for a dark one, 0 for none.
fn thin_ridge_polarity(top: vec3<f32>, middle: vec3<f32>, bottom: vec3<f32>) -> f32 {
    let c = middle.y;
    let st = max(top, vec3<f32>(c)) <= RIDGE_SIMILARITY * min(top, vec3<f32>(c));
    let sm = max(middle, vec3<f32>(c)) <= RIDGE_SIMILARITY * min(middle, vec3<f32>(c));
    let sb = max(bottom, vec3<f32>(c)) <= RIDGE_SIMILARITY * min(bottom, vec3<f32>(c));
    // The centre is similar to itself and takes no part in lo/hi.
    let lo = min(min(select(top, vec3<f32>(3.0e38), st), select(bottom, vec3<f32>(3.0e38), sb)),
        vec3<f32>(select(middle.x, 3.0e38, sm.x), 3.0e38, select(middle.z, 3.0e38, sm.z)));
    let hi = max(max(select(top, vec3<f32>(0.0), st), select(bottom, vec3<f32>(0.0), sb)),
        vec3<f32>(select(middle.x, 0.0, sm.x), 0.0, select(middle.z, 0.0, sm.z)));
    let low = min(lo.x, min(lo.y, lo.z));
    let high = max(hi.x, max(hi.y, hi.z));
    let quad = (st.x && st.y && sm.x) || (st.y && st.z && sm.z)
        || (sm.x && sb.x && sb.y) || (sm.z && sb.y && sb.z);
    if quad || !(c > high || c < low) { return 0.0; }
    return select(-1.0, 1.0, c > high);
}

// History's luma low-passed exactly like `Neighbourhood::low_luma`: the same
// weights at the same 3x3 sample positions, carried back to `history_uv`.
// Sharing the sampling pattern leaves in the residual only what the scene
// did, plus zero-mean noise, not where this frame's jitter put the samples.
fn history_low_luma(in_pos: vec2<f32>, history_uv: vec2<f32>) -> f32 {
    let in_dims = vec2<f32>(textureDimensions(current_frame));
    let to_output = vec2<f32>(textureDimensions(history_frame)) / in_dims;
    let first = floor(in_pos) - 1.0;
    let lx = gaussian4(first.x, in_pos.x, to_output.x, LOW_PASS_SIGMA);
    let ly = gaussian4(first.y, in_pos.y, to_output.y, LOW_PASS_SIGMA);
    var sum = vec2<f32>(0.0);
    for (var y = 0; y < 3; y++) {
        for (var x = 0; x < 3; x++) {
            let offset = first + vec2<f32>(f32(x), f32(y)) + 0.5 - in_pos;
            let q = perceptual(textureSampleLevel(history_frame, linear_sampler, history_uv + offset / in_dims, 0.0).rgb);
            sum += lx[x] * ly[y] * vec2<f32>(q.x, 1.0);
        }
    }
    return sum.x / sum.y;
}

// Clip `h` toward `mean` onto the box mean +/- extent; returns the clipped
// colour and how many box half-sizes outside it `h` was (<= 1 inside).
fn clip_toward_mean(h: vec3<f32>, mean: vec3<f32>, extent: vec3<f32>) -> vec4<f32> {
    let d = h - mean;
    let outside = max3(abs(d) / extent);
    return vec4<f32>(select(h, mean + d / outside, outside > 1.0), outside);
}

// ── Resolve ───────────────────────────────────────────────────────────────────

struct TsrOutput {
    @location(0) color: vec4<f32>,
    @location(1) view_depth: f32,
    @location(2) state: vec4<f32>,
    @location(3) noise: f32,
}

@fragment
fn fs_main(in: VertexOutput) -> TsrOutput {
    let in_dims = vec2<f32>(textureDimensions(current_frame));
    // The scene point under this output pixel lies at this raster position.
    let cur_uv = in.uv + tsr.jitter_offset * vec2<f32>(1.0, -1.0) / in_dims;
    let in_pos = cur_uv * in_dims;
    let use_coverage = (tsr.flags & 2u) != 0u;
    var coverage = 0.0;
    if use_coverage {
        coverage = clamp(textureSampleLevel(transparency_reactivity, linear_sampler, cur_uv, 0.0).r, 0.0, 1.0);
    }
    // History alpha holds coverage only in the opt-in transparency path.
    let output_alpha = select(1.0, coverage, use_coverage);

    let n = gather(in_pos);
    let current_depth = view_depth(cur_uv, n.nearest_depth);
    var fresh: TsrOutput;
    fresh.color = vec4<f32>(linear_rgb(n.colour), output_alpha);
    fresh.view_depth = current_depth;
    fresh.state = vec4<f32>(n.weight, select(0.0, 1.0, n.ridge != 0.0), n.ridge, 0.0);
    // Until it has been measured, expect the detector's residual to vary as
    // much as the neighbourhood does.
    fresh.noise = n.deviation.x * n.deviation.x;

    let history = reproject_history(cur_uv, n.nearest_depth);
    let history_uv = history.xy;
    // Derivatives before any divergent return. The slope term admits the
    // depth change across one pixel of a surface seen at a grazing angle.
    let depth_tolerance = max(0.002, abs(history.w) * 0.004)
        + 1.5 * (abs(dpdx(history.w)) + abs(dpdy(history.w)));

    // Full reactivity promises current-frame-only output, as a reset does.
    var reactivity = max(tsr.reactivity, coverage);
    if (tsr.flags & 1u) != 0u || reactivity >= 1.0 {
        return fresh;
    }
    if history.z <= 0.0 || any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0)) {
        return fresh;
    }
    let out_dims = vec2<i32>(textureDimensions(history_frame));
    // Both depths are dilated the same way, so a still silhouette compares
    // its foreground with itself whichever side the jitter lands on.
    let stored_depth = textureLoad(history_depth_tex,
        clamp(vec2<i32>(history_uv * vec2<f32>(out_dims)), vec2<i32>(0), out_dims - 1), 0).r;
    let sky_continues = helio_is_sky(n.nearest_depth) && stored_depth == 0.0;
    if !sky_continues && !history_depth_matches(history.w, stored_depth, depth_tolerance) {
        return fresh;
    }

    if use_coverage {
        let previous_coverage = clamp(textureSampleLevel(history_frame, linear_sampler, history_uv, 0.0).a, 0.0, 1.0);
        reactivity = max(reactivity, previous_coverage);
    }
    if reactivity >= 1.0 {
        return fresh;
    }

    let previous = textureSampleLevel(history_state, linear_sampler, history_uv, 0.0);
    var history_weight = previous.x * (1.0 - reactivity);
    var lock = previous.y * (1.0 - reactivity);
    var lock_luma = previous.z;

    // Change detection (step 6). The running mean of a zero-mean residual
    // with deviation sigma has deviation sigma * sqrt(rate / (2 - rate)).
    let residual = n.low_luma - history_low_luma(in_pos, history_uv);
    var residual_mean = mix(previous.w, residual, CHANGE_RATE);
    let noise = textureSampleLevel(history_noise, linear_sampler, history_uv, 0.0).r;
    let deviation = sqrt(noise);
    let slack = CHANGE_RELATIVE * max(n.low_luma, CHANGE_FLOOR);
    let tolerance = CHANGE_SIGMAS * deviation * sqrt(CHANGE_RATE / (2.0 - CHANGE_RATE)) + CHANGE_RATE * slack;
    var unchanged = saturate(2.0 - abs(residual_mean) / tolerance);
    let typical = clamp(residual, -CHANGE_OUTLIER * deviation - slack, CHANGE_OUTLIER * deviation + slack);
    let learnt_noise = mix(noise, typical * typical, NOISE_RATE);

    // A locked ridge seen again with another luminance: the scene changed
    // under it (a light, a material) and what history holds of it is stale.
    // Only same-polarity ridges compare; a dark gap between two bright
    // features is not the feature that was locked.
    if n.ridge != 0.0 {
        if lock == 0.0 || sign(n.ridge) == sign(lock_luma) {
            let a = abs(n.ridge);
            let b = abs(lock_luma);
            if lock > 0.0 && abs(a - b) > RIDGE_CHANGE * max(max(a, b), CHANGE_FLOOR) {
                unchanged = 0.0;
            }
            lock_luma = n.ridge;
        }
    }
    // What a change drops restarts as a fresh pixel: weight, lock and the
    // change acted on. The learnt noise stays: it describes the place.
    history_weight *= unchanged;
    lock = select(max(lock * unchanged - 1.0 / LOCK_FRAMES, 0.0), 1.0, n.ridge != 0.0);
    residual_mean *= unchanged;

    // Full protection for most of the lifetime, fading over its last frames.
    let protection = saturate(lock * 4.0);

    let history_colour = perceptual(sample_history(history_uv));
    let extent = max(VARIANCE_CLIP_GAMMA * n.deviation, vec3<f32>(MIN_BOX_EXTENT));
    let clip = clip_toward_mean(history_colour, n.mean, extent);
    // History outside the box by k box sizes keeps 1/k^2 of its weight: a
    // small excursion is sampling noise, a large one is a change.
    history_weight *= mix(min(1.0, 1.0 / (clip.w * clip.w)), 1.0, protection);
    let rectified = mix(clip.xyz, history_colour, protection);

    let total = history_weight + n.weight;
    let blended = mix(rectified, n.colour, n.weight / max(total, 1.0e-6));

    var out: TsrOutput;
    out.color = vec4<f32>(linear_rgb(blended), output_alpha);
    out.view_depth = current_depth;
    out.state = vec4<f32>(min(total, MAX_HISTORY_WEIGHT), lock, lock_luma, residual_mean);
    out.noise = learnt_noise;
    return out;
}

// ── Sharpening (RCAS) ─────────────────────────────────────────────────────────

// FidelityFX RCAS on the resolve, in the tonemapped domain it is defined
// for: the strongest negative lobe that keeps the 5-tap cross in range, so
// edges sharpen without ringing.
@fragment
fn fs_sharpen(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(resolved));
    let p = vec2<i32>(in.position.xy);
    let centre = textureLoad(resolved, p, 0);
    let e = tonemap(centre.rgb);
    let b = tonemap(textureLoad(resolved, clamp(p + vec2<i32>(0, -1), vec2<i32>(0), dims - 1), 0).rgb);
    let d = tonemap(textureLoad(resolved, clamp(p + vec2<i32>(-1, 0), vec2<i32>(0), dims - 1), 0).rgb);
    let f = tonemap(textureLoad(resolved, clamp(p + vec2<i32>(1, 0), vec2<i32>(0), dims - 1), 0).rgb);
    let h = tonemap(textureLoad(resolved, clamp(p + vec2<i32>(0, 1), vec2<i32>(0), dims - 1), 0).rgb);
    let lo = min(min(b, d), min(f, h));
    let hi = max(max(b, d), max(f, h));
    let hit_min = lo / (4.0 * hi + 1.0e-6);
    let hit_max = (1.0 - hi) / (4.0 * lo - 4.0 - 1.0e-6);
    let lobe_rgb = max(-hit_min, hit_max);
    let lobe = max(-RCAS_LIMIT, min(max3(lobe_rgb), 0.0)) * exp2(-SHARPEN_STOPS);
    let sharpened = (lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0);
    // RCAS only keeps the result inside [0, 1], the display range; here 1 is
    // infinitely bright, so also keep it inside the cross's own range: no
    // overshoot can turn a bright edge into a firefly.
    let limited = clamp(sharpened, min(lo, e), max(hi, e));
    return vec4<f32>(reverse_tonemap(limited), centre.a);
}
