//!use helio_prelude
// Temporal Super-Resolution: one resolve dispatch and one sharpening draw.
//
// The resolve runs over 8x8 tiles of output pixels. A tile first converts
// the input texels its pixels read (perceptual colour, log luminance, view
// depth) once into workgroup memory, then runs the thin-feature test of
// step 7 once per texel a kernel can centre on; each pixel then reads both
// from there. The steps below are per pixel and unchanged by the tiling.
//
// Resolve, per output pixel:
//   1. Reconstruct this frame's colour from the jittered input samples around
//      it with a Gaussian of the output pixel, and keep the summed kernel
//      weight: how much this frame actually says about the pixel.
//   2. Dilate depth (nearest of the 3x3) and reproject with it, so an edge
//      moves with its foreground surface.
//   3. Reject history on disocclusion: no stored depth in the 3x3 around the
//      reprojected pixel near the dilated depth or the pixel's own surface
//      (history stores both). The dilated depth keeps a still silhouette
//      matching whichever side the jitter puts the sample on; the pixel's own
//      keeps ground at a grazing angle matching while the jitter moves the
//      3x3 between rows. The tolerance follows the surface's slope measured
//      on the input's depth texels, one-sided so a silhouette is not slope.
//   4. Sample history with Catmull-Rom, de-ringed to its own 2x2.
//   5. Detect changes the per-pixel box cannot tell from noise, two ways,
//      each a fast running mean against a slow one (a bias-free test: a
//      static offset between input and history is not a change), against
//      the measured spread of that difference:
//      - fine: this frame and history low-passed at the same sample
//        positions (UE5 TSR compares low-passed input and history);
//      - coarse: the mean log luminance of the 4x4 input texels, FSR 2's
//        shading-change luma, which aliasing on wires and fine texture
//        barely moves, so a light switching there is still seen. A shading
//        change over FSR 2's 10 % (in luma^(1/6)) is decisive.
//      A change drops the accumulated weight and what the pixel learnt of
//      its own variation, and restarts the detectors from this frame.
//   6. Rectify: clip history toward the neighbourhood mean (YCoCg,
//      tonemapped), against a box that is the larger of:
//      - mean +/- one standard deviation of the 3x3 (Salvi's variance clip);
//      - twice the pixel's own measured temporal deviation, while still: the
//        input's spread about history, learnt as a running variance. Jitter
//        aliasing on texel-sized detail and stochastic per-frame lighting
//        (rotating shadow and sky samples) make single frames disagree with
//        the converged value far beyond the frame's neighbourhood; history
//        may oscillate within what the pixel is known to do (UE5 TSR's
//        flicker amplitude and FSR 2's luma instability serve the same end).
//        Real changes still reject: step 5 forgets the variance first;
//      - with a stochastic-shading mask (lib.rs STOCHASTIC_MASK), the
//        neighbourhood's whole range (FSR 2's box on still pixels).
//      How far outside history was scales its weight down (1/k^2), keeping
//      at least two samples' worth of the clipped value (UE5 TSR).
//   7. Thin features (FSR 2 locks): a ridge seen in the input locks the
//      pixel for a short lifetime, during which the clip yields to history,
//      so a sub-pixel wire the jitter missed this frame is not clipped away.
//      A detected change releases it.
//   8. Accumulate as a weighted running mean: history weight vs this frame's
//      kernel weight, capped at MAX_HISTORY_WEIGHT while still and toward
//      MOVING_HISTORY_WEIGHT at a pixel per frame of motion, so repeated
//      reprojection does not blur (UE5 TSR, FSR 2). Blending happens in a
//      reversible max-channel tonemap, so a single bright sample cannot
//      dominate (Karis).
// Sharpen: RCAS on the accumulated history. History itself stays
// unsharpened (no feedback), and sharpening only the converged image cannot
// put the jitter back.
//
// References:
//   FSR 2 — https://github.com/GPUOpen-Effects/FidelityFX-FSR2 (MIT):
//     ffx_fsr2_accumulate.h, _upsample.h, _depth_clip.h,
//     _postprocess_lock_status.h — dilation, weight accumulation, box scale
//     on still pixels, luma instability, locks, shading change, RCAS
//   UE5 TSR — https://dev.epicgames.com/documentation/en-us/unreal-engine/temporal-super-resolution-in-unreal-engine
//     (low-passed shading rejection, flicker detection, rejection keeping
//     two samples, sample count limited by velocity)
//   Salvi, "An Excursion in Temporal Supersampling" (GDC 2016): variance clipping
//   Karis, "High Quality Temporal Supersampling" (SIGGRAPH 2014): tonemapped blend
//   Pedersen, "Temporal Reprojection Anti-Aliasing in INSIDE" (GDC 2016),
//     https://github.com/playdeadgames/temporal (MIT): clip toward mean
//   Yang, Liu, Salvi, "A Survey of Temporal Antialiasing Techniques" (2020)
//   Finch, "Incremental calculation of weighted mean and variance" (2009)

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
// The cap at a pixel per frame of motion and beyond (UE5 TSR's 4 samples).
const MOVING_HISTORY_WEIGHT: f32 = 4.0;
// What the box rejects keeps at least this much weight (of its clipped
// value): UE5 TSR's total shading rejection keeps about two samples.
const REJECTED_HISTORY_WEIGHT: f32 = 2.0;
// Rectification box half-size in standard deviations of the neighbourhood...
const VARIANCE_CLIP_GAMMA: f32 = 1.0;
// ...and in the pixel's own temporal deviations.
const TEMPORAL_GAMMA: f32 = 2.0;
// Motion (output pixels per frame) at which the temporal widening and the
// stochastic mask stop applying: FSR 2 scales its box down by velocity / 20,
// UE5 TSR has a maximum parallax velocity for flicker detection.
const PARALLAX_VELOCITY: f32 = 20.0;
// Box floor in tonemapped YCoCg: below half-float noise of the input.
const MIN_BOX_EXTENT: f32 = 1.0 / 1024.0;
// Disocclusion: the stored depth may differ by this many texels of the
// surface's slope (the jitter moves the sample by up to one, the dilation by
// up to one more).
const DEPTH_SLOPE_TEXELS: f32 = 2.0;
// Low-pass sigma (output pixels) of the fine detector: wide enough to
// average the jitter, narrow enough for the 3x3 the clip box reads anyway.
const LOW_PASS_SIGMA: f32 = 1.0;
// The detectors' running means: fast (about 4 frames) and slow (about 16),
// and the rate at which they learn their difference's spread, from values
// limited to LEARN_SIGMAS deviations so a change acts before it can teach
// the detector that differences that large are noise.
const FAST_RATE: f32 = 0.25;
const SLOW_RATE: f32 = 0.0625;
const NOISE_RATE: f32 = 0.0625;
const LEARN_SIGMAS: f32 = 2.0;
// The input's temporal variance learns from deviations within this many box
// sizes.
const LEARN_LIMIT: f32 = 3.0;
// The difference must exceed this many of its own deviations...
const CHANGE_SIGMAS: f32 = 3.0;
// ...and this fraction of the luma (tonemapped; absolute below the floor;
// for the coarse detector, of luminance: 0.05 in log).
const CHANGE_RELATIVE: f32 = 0.05;
const CHANGE_FLOOR: f32 = 1.0 / 64.0;
// FSR 2's decisive shading change, in luminance^(1/6) (a factor of 1.9),
// measured against a reference that follows slowly and restarts only when
// a change is acted on in full.
const SHADING_CHANGE: f32 = 0.1;
const REFERENCE_RATE: f32 = 0.03125;
// Frames a thin-feature lock survives without seeing its ridge again: longer
// than the jitter takes to hit a sub-pixel feature again.
const LOCK_FRAMES: f32 = 8.0;
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
// xy: view depths, 0 = sky: x dilated, y the pixel's own surface's.
// zw: the coarse detector's slow and fast mean log luminance.
@group(0) @binding(7) var history_depth_coarse: texture_2d<f32>;
@group(0) @binding(8) var transparency_reactivity: texture_2d<f32>;
// x: accumulated weight, y: thin-feature lock lifetime (0..1), z: the fine
// detector's fast mean residual, w: the deviation of its fast - slow.
@group(0) @binding(9) var history_state: texture_2d<f32>;
// x: the input's temporal deviation about history, y: the deviation of the
// coarse detector's fast - slow, z: the fine detector's slow mean residual,
// w: the shading-change reference (log luminance). Deviations are stored,
// not variances: half floats flush small variances to zero.
@group(0) @binding(11) var history_noise: texture_2d<f32>;
// Fraction of the pixel's shading sampled stochastically per frame (lib.rs
// STOCHASTIC_MASK); zero when the graph does not provide it.
@group(0) @binding(13) var stochastic_mask: texture_2d<f32>;

// The resolve's outputs: next frame's history_frame, history_depth_coarse,
// history_state and history_noise.
@group(0) @binding(16) var out_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(17) var out_depth_coarse: texture_storage_2d<rgba16float, write>;
@group(0) @binding(18) var out_state: texture_storage_2d<rgba16float, write>;
@group(0) @binding(19) var out_noise: texture_storage_2d<rgba16float, write>;

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

// A colour with its non-finite channels (NaN, infinity: all exponent bits
// set) as 0. A NaN that reached history would survive every blend after
// it, and one frame's would stay for good; tested on the bits, which no
// fast-math assumption folds away.
fn finite_rgb(c: vec3<f32>) -> vec3<f32> {
    let exponent = bitcast<vec3<u32>>(c) & vec3<u32>(0x7f800000u);
    return select(c, vec3<f32>(0.0), exponent == vec3<u32>(0x7f800000u));
}

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
            let s = finite_rgb(textureLoad(history_frame, at, 0).rgb);
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
    lo: vec3<f32>,
    hi: vec3<f32>,
    nearest_depth: f32,     // dilated: the nearest raw depth of that 3x3
    centre_depth: f32,      // raw depth of the texel under the pixel
    // View depth change across one input texel of the centre's own surface:
    // per axis the smaller of the two one-sided differences, so a silhouette
    // next to the pixel does not count as slope.
    slope: f32,
    // A thin feature crossing a texel this pixel draws on, the nearest such:
    // its tonemapped luma, negative for a dark one, 0 for none.
    ridge: f32,
    low_luma: f32,          // luma low-passed by the change detector's kernel
    log_luma: f32,          // mean log luminance of the 4x4 (FSR 2's shading-change luma)
    log_variance: f32,      // its variance over the 4x4
}

// Gaussian weights, per axis (the kernels are separable), of the four
// texels from `first` on, `pos` the pixel centre in input texels.
fn gaussian4(first: f32, pos: f32, to_output: f32, sigma: f32) -> vec4<f32> {
    let d = (vec4<f32>(first) + vec4<f32>(0.5, 1.5, 2.5, 3.5) - pos) * to_output;
    return exp(-0.5 * d * d / (sigma * sigma));
}

// ── Input tile ────────────────────────────────────────────────────────────────

// Output pixels per tile (lib.rs RESOLVE_TILE).
const TILE: vec2<u32> = vec2<u32>(8u, 8u);
// Input texels a tile reads at most per axis: the 4x4 around each pixel,
// whose origins lie at most one texel apart per pixel for input no larger
// than the output (the renderer's render scale is at most 1), plus one for
// rounding.
const FOOTPRINT: vec2<i32> = vec2<i32>(12, 12);
const FOOTPRINT_TEXELS: u32 = 144u;
// Per texel of the tile, by the texel's own (unclamped) position, rows
// FOOTPRINT.x apart: perceptual YCoCg and log luminance of the clamped
// texel; its raw depth and the view depth that has at the unclamped texel's
// centre.
var<workgroup> tile_colour: array<vec4<f32>, FOOTPRINT_TEXELS>;
var<workgroup> tile_depth: array<vec2<f32>, FOOTPRINT_TEXELS>;
// Per texel a pixel's kernel centres on (the inner 2x2 of its 4x4): a thin
// feature crossing it, as `Neighbourhood::ridge`.
var<workgroup> tile_ridge: array<f32, FOOTPRINT_TEXELS>;

fn pixel_uv(pixel: vec2<u32>) -> vec2<f32> {
    return (vec2<f32>(pixel) + 0.5) / vec2<f32>(textureDimensions(history_frame));
}

// The scene point under this output pixel lies at this raster position.
fn current_uv(pixel: vec2<u32>) -> vec2<f32> {
    let in_dims = vec2<f32>(textureDimensions(current_frame));
    return pixel_uv(pixel) + tsr.jitter_offset * vec2<f32>(1.0, -1.0) / in_dims;
}

// The first texel of the 4x4 a pixel at `in_pos` reads.
fn footprint_origin(in_pos: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(floor(in_pos - 0.5)) - 1;
}

// Converts the tile's input texels from `tile_origin` on, `span` per axis,
// all threads of the workgroup sharing the work.
fn load_tile(tile_origin: vec2<i32>, span: vec2<i32>, thread: u32) {
    let dims = vec2<i32>(textureDimensions(current_frame));
    let texel = 1.0 / vec2<f32>(dims);
    for (var j = i32(thread); j < span.x * span.y; j += i32(TILE.x * TILE.y)) {
        let local = vec2<i32>(j % span.x, j / span.x);
        let i = local.y * FOOTPRINT.x + local.x;
        let at = tile_origin + local;
        let p = clamp(at, vec2<i32>(0), dims - 1);
        let rgb = finite_rgb(textureLoad(current_frame, p, 0).rgb);
        let l = log(max(dot(rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 1.0e-4));
        tile_colour[i] = vec4<f32>(perceptual(rgb), l);
        let d = textureLoad(depth_tex, p, 0);
        tile_depth[i] = vec2<f32>(d, view_depth((vec2<f32>(at) + 0.5) * texel, d));
    }
}

// Tests the tile's texels from `first` on, `span` per axis, for thin
// features, all threads of the workgroup sharing the work.
fn find_ridges(first: vec2<i32>, span: vec2<i32>, thread: u32) {
    let row = FOOTPRINT.x;
    for (var i = i32(thread); i < span.x * span.y; i += i32(TILE.x * TILE.y)) {
        let c = (first.y + i / span.x) * row + first.x + i % span.x;
        let top = vec3<f32>(tile_colour[c - row - 1].x, tile_colour[c - row].x, tile_colour[c - row + 1].x);
        let middle = vec3<f32>(tile_colour[c - 1].x, tile_colour[c].x, tile_colour[c + 1].x);
        let bottom = vec3<f32>(tile_colour[c + row - 1].x, tile_colour[c + row].x, tile_colour[c + row + 1].x);
        tile_ridge[c] = thin_ridge_polarity(top, middle, bottom) * middle.y;
    }
}

// `in_pos` is the output pixel centre in input texels, jitter included. The
// 4x4 texels around it hold every sample the kernel weighs, the 3x3 of each
// of the 2x2 nearest, which the thin-feature test needs, and the coarse luma.
// They are read from the tile loaded from `tile_origin`.
fn gather(in_pos: vec2<f32>, tile_origin: vec2<i32>) -> Neighbourhood {
    let dims = vec2<i32>(textureDimensions(current_frame));
    let origin = footprint_origin(in_pos);
    let local = clamp(origin - tile_origin, vec2<i32>(0), FOOTPRINT - 4);
    let row = FOOTPRINT.x;
    let nearest = vec2<i32>(floor(in_pos)) - origin; // 1 or 2 per axis
    let to_output = vec2<f32>(textureDimensions(history_frame)) / vec2<f32>(dims);
    let kx = gaussian4(f32(origin.x), in_pos.x, to_output.x, KERNEL_SIGMA);
    let ky = gaussian4(f32(origin.y), in_pos.y, to_output.y, KERNEL_SIGMA);
    let lx = gaussian4(f32(origin.x), in_pos.x, to_output.x, LOW_PASS_SIGMA);
    let ly = gaussian4(f32(origin.y), in_pos.y, to_output.y, LOW_PASS_SIGMA);
    var n: Neighbourhood;
    // The 4x4: the kernel's samples and the coarse luma.
    let first = local.y * row + local.x;
    var sum = vec3<f32>(0.0);
    var log_sum = vec2<f32>(0.0);
    for (var y = 0; y < 4; y++) {
        for (var x = 0; x < 4; x++) {
            let texel = tile_colour[first + y * row + x];
            sum += kx[x] * ky[y] * texel.xyz;
            log_sum += vec2<f32>(texel.w, texel.w * texel.w);
        }
    }
    // The 3x3 around the nearest texel, in the same order.
    let centre = first + nearest.y * row + nearest.x;
    let lx3 = select(lx.xyz, lx.yzw, nearest.x == 2);
    let ly3 = select(ly.xyz, ly.yzw, nearest.y == 2);
    var low = vec2<f32>(0.0);
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    var depth = 0.0;
    n.lo = vec3<f32>(3.0e38);
    n.hi = vec3<f32>(-3.0e38);
    for (var y = 0; y < 3; y++) {
        for (var x = 0; x < 3; x++) {
            let i = centre + (y - 1) * row + x - 1;
            let q = tile_colour[i].xyz;
            m1 += q;
            m2 += q * q;
            n.lo = min(n.lo, q);
            n.hi = max(n.hi, q);
            low += lx3[x] * ly3[y] * vec2<f32>(q.x, 1.0);
            depth = helio_depth_nearest(depth, tile_depth[i].x);
        }
    }
    n.centre_depth = tile_depth[centre].x;
    let centre_view = tile_depth[centre].y;
    // View depths of the centre's four neighbours: -x, +x, -y, +y.
    let around = vec4<f32>(tile_depth[centre - 1].y, tile_depth[centre + 1].y,
        tile_depth[centre - row].y, tile_depth[centre + row].y);
    n.weight = dot(kx, vec4<f32>(1.0)) * dot(ky, vec4<f32>(1.0));
    n.colour = sum / max(n.weight, 1.0e-6);
    n.mean = m1 / 9.0;
    n.deviation = sqrt(max(m2 / 9.0 - n.mean * n.mean, vec3<f32>(0.0)));
    n.nearest_depth = depth;
    // Sky (view depth 0) next to the pixel is a silhouette, not slope.
    let steps = select(abs(around - centre_view), vec4<f32>(3.0e38), around == vec4<f32>(0.0));
    n.slope = max(min(steps.x, steps.y), min(steps.z, steps.w));
    n.low_luma = low.x / low.y;
    n.log_luma = log_sum.x / 16.0;
    n.log_variance = max(log_sum.y / 16.0 - n.log_luma * n.log_luma, 0.0);
    // The ridge nearest the pixel centre among the four texels it draws on.
    let inner = first + row + 1;
    let ridges = vec4<f32>(tile_ridge[inner], tile_ridge[inner + 1],
        tile_ridge[inner + row], tile_ridge[inner + row + 1]);
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
            let q = perceptual(finite_rgb(textureSampleLevel(history_frame, linear_sampler, history_uv + offset / in_dims, 0.0).rgb));
            sum += lx[x] * ly[y] * vec2<f32>(q.x, 1.0);
        }
    }
    return sum.x / sum.y;
}

// FSR 2's shading change between two log luminances: 1 - min / max of
// luminance to the 1/6 power (0.1 is a factor of about 1.9).
fn shading_change(a: f32, b: f32) -> f32 {
    return 1.0 - exp(-abs(a - b) / 6.0);
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
    color: vec4<f32>,
    view_depth: vec2<f32>,
    state: vec4<f32>,
    noise: vec4<f32>,
    coarse: vec2<f32>,
}

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) group: vec3<u32>,
           @builtin(local_invocation_index) thread: u32) {
    // The 4x4s of the tile's pixels, from its first pixel's to its last's.
    let in_dims = vec2<f32>(textureDimensions(current_frame));
    let tile_origin = footprint_origin(current_uv(group.xy * TILE) * in_dims);
    let last = min(group.xy * TILE + TILE - 1u, textureDimensions(history_frame) - 1u);
    let last_origin = min(footprint_origin(current_uv(last) * in_dims) - tile_origin, FOOTPRINT - 4);
    load_tile(tile_origin, last_origin + 4, thread);
    workgroupBarrier();
    // Their inner 2x2s.
    find_ridges(vec2<i32>(1), last_origin + 2, thread);
    workgroupBarrier();
    if any(id.xy >= textureDimensions(history_frame)) {
        return;
    }
    let out = resolve(id.xy, tile_origin);
    textureStore(out_color, id.xy, out.color);
    textureStore(out_depth_coarse, id.xy, vec4<f32>(out.view_depth, out.coarse));
    textureStore(out_state, id.xy, out.state);
    textureStore(out_noise, id.xy, out.noise);
}

fn resolve(pixel: vec2<u32>, tile_origin: vec2<i32>) -> TsrOutput {
    let in_dims = vec2<f32>(textureDimensions(current_frame));
    let cur_uv = current_uv(pixel);
    let in_pos = cur_uv * in_dims;
    let use_coverage = (tsr.flags & 2u) != 0u;
    var coverage = 0.0;
    if use_coverage {
        coverage = clamp(textureSampleLevel(transparency_reactivity, linear_sampler, cur_uv, 0.0).r, 0.0, 1.0);
    }
    // History alpha holds coverage only in the opt-in transparency path.
    let output_alpha = select(1.0, coverage, use_coverage);

    let n = gather(in_pos, tile_origin);
    let current_depth = vec2<f32>(view_depth(cur_uv, n.nearest_depth), view_depth(cur_uv, n.centre_depth));
    var fresh: TsrOutput;
    fresh.color = vec4<f32>(linear_rgb(n.colour), output_alpha);
    fresh.view_depth = current_depth;
    fresh.state = vec4<f32>(n.weight, select(0.0, 1.0, n.ridge != 0.0), 0.0,
        n.deviation.x * sqrt(FAST_RATE / (2.0 - FAST_RATE)));
    // Until they have been measured, expect the input and the detectors to
    // vary as much as the neighbourhood does.
    fresh.noise = vec4<f32>(length(n.deviation), sqrt(n.log_variance * FAST_RATE / (2.0 - FAST_RATE)), 0.0,
        n.log_luma);
    fresh.coarse = vec2<f32>(n.log_luma);

    let history = reproject_history(cur_uv, n.nearest_depth);
    let history_uv = history.xy;
    let expected_own = reproject_history(cur_uv, n.centre_depth).w;
    let depth_tolerance = max(0.002, abs(expected_own) * 0.004) + DEPTH_SLOPE_TEXELS * n.slope;

    // Full reactivity promises current-frame-only output, as a reset does.
    var reactivity = max(tsr.reactivity, coverage);
    if (tsr.flags & 1u) != 0u || reactivity >= 1.0 {
        return fresh;
    }
    if history.z <= 0.0 || any(history_uv < vec2<f32>(0.0)) || any(history_uv > vec2<f32>(1.0)) {
        return fresh;
    }
    // Disocclusion (step 3).
    let out_dims = vec2<i32>(textureDimensions(history_frame));
    let depth_texel = clamp(vec2<i32>(history_uv * vec2<f32>(out_dims)), vec2<i32>(0), out_dims - 1);
    // Any match decides, so the centre is tried first and the search stops
    // there; sky only counts when nothing matched.
    var matched = false;
    var sky_stored = false;
    for (var i = 0; i < 9 && !matched; i++) {
        let offset = vec2<i32>((i + 1) % 3, (i / 3 + 1) % 3) - 1;
        let stored = textureLoad(history_depth_coarse,
            clamp(depth_texel + offset, vec2<i32>(0), out_dims - 1), 0).xy;
        matched = history_depth_matches(history.w, stored.x, depth_tolerance)
            || history_depth_matches(history.w, stored.y, depth_tolerance)
            || history_depth_matches(expected_own, stored.x, depth_tolerance)
            || history_depth_matches(expected_own, stored.y, depth_tolerance);
        sky_stored = sky_stored || stored.x == 0.0;
    }
    let sky_continues = helio_is_sky(n.nearest_depth) && sky_stored;
    if !sky_continues && !matched {
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
    let moments = textureSampleLevel(history_noise, linear_sampler, history_uv, 0.0);
    let means = textureSampleLevel(history_depth_coarse, linear_sampler, history_uv, 0.0).zw;
    var history_weight = previous.x * (1.0 - reactivity);
    var lock = previous.y * (1.0 - reactivity);

    // Fine change detector (step 5): the residual's fast mean against its
    // slow one.
    let residual = n.low_luma - history_low_luma(in_pos, history_uv);
    let residual_fast = mix(previous.z, residual, FAST_RATE);
    let residual_slow = moments.z;
    let deviation = previous.w;
    let slack = CHANGE_RELATIVE * max(n.low_luma, CHANGE_FLOOR);
    let statistic = residual_fast - residual_slow;
    var unchanged = saturate(2.0 - abs(statistic) / (CHANGE_SIGMAS * deviation + FAST_RATE * slack));
    let learnt = clamp(statistic, -LEARN_SIGMAS * deviation - slack, LEARN_SIGMAS * deviation + slack);
    let learnt_deviation = sqrt(mix(deviation * deviation, learnt * learnt, NOISE_RATE));

    // Coarse change detector: the mean log luminance's fast mean against its
    // slow one, and FSR 2's decisive shading change against the reference.
    let coarse = n.log_luma;
    let slow = means.x;
    let fast = mix(means.y, coarse, FAST_RATE);
    let coarse_deviation = moments.y;
    let coarse_tolerance = CHANGE_SIGMAS * coarse_deviation + CHANGE_RELATIVE;
    let coarse_unchanged = saturate(2.0 - abs(fast - slow) / coarse_tolerance);
    let coarse_learnt = clamp(fast - slow, -LEARN_SIGMAS * coarse_deviation - CHANGE_RELATIVE,
        LEARN_SIGMAS * coarse_deviation + CHANGE_RELATIVE);
    let learnt_coarse_deviation = sqrt(mix(coarse_deviation * coarse_deviation, coarse_learnt * coarse_learnt,
        NOISE_RATE));
    let reference = moments.w;
    let change = shading_change(reference, fast);
    // Within the coarse luma's own noise, even 10 % is not evidence.
    let shading_changed = change > SHADING_CHANGE && abs(fast - reference) > coarse_tolerance;
    unchanged = select(min(unchanged, coarse_unchanged), 0.0, shading_changed);

    // What a change drops restarts from this frame: weight, the detectors'
    // means (acted on, a change is not detected again) and what the pixel
    // learnt of its own variation. The detectors' learnt spreads stay: they
    // describe the place.
    history_weight *= unchanged;
    let kept_weight = history_weight;
    let new_residual_fast = mix(residual_fast, residual, 1.0 - unchanged);
    let new_residual_slow = mix(mix(residual_slow, residual, SLOW_RATE), residual, 1.0 - unchanged);
    // The coarse means restart only for what the coarse detectors acted on,
    // so a partial drop by the fine one does not use up their evidence.
    let restart = select(1.0 - coarse_unchanged, 1.0, shading_changed);
    let new_slow = mix(mix(slow, coarse, SLOW_RATE), coarse, restart);
    let new_fast = mix(fast, coarse, restart);
    let new_reference = select(mix(reference, coarse, REFERENCE_RATE), coarse, restart >= 1.0);
    let variance = moments.x * moments.x * unchanged;

    // Thin-feature locks (step 7), released by a detected change. Full
    // protection for most of the lifetime, fading over its last frames.
    lock = select(max(lock - 1.0 / LOCK_FRAMES, 0.0), 1.0, n.ridge != 0.0) * unchanged;
    let protection = saturate(lock * 4.0);

    // Rectification (step 6).
    let history_colour = perceptual(sample_history(history_uv));
    let velocity = length((history_uv - pixel_uv(pixel)) * vec2<f32>(out_dims));
    let still = 1.0 - saturate(velocity / PARALLAX_VELOCITY);
    // A shading change short of decisive weakens the widening, as it does
    // FSR 2's lock protection.
    let relit = select(1.0 - change, 0.0, change > SHADING_CHANGE);
    let temporal = TEMPORAL_GAMMA * sqrt(variance) * still * relit;
    let stochastic = saturate(textureSampleLevel(stochastic_mask, linear_sampler, cur_uv, 0.0).r) * still;
    let range = max(n.hi - n.mean, n.mean - n.lo);
    let extent = max(max(VARIANCE_CLIP_GAMMA * n.deviation, vec3<f32>(temporal)),
        max(stochastic * range, vec3<f32>(MIN_BOX_EXTENT)));
    let clip = clip_toward_mean(history_colour, n.mean, extent);
    // History outside the box by k box sizes keeps 1/k^2 of its weight: a
    // small excursion is sampling noise, a large one is a change. What is
    // left is clipped into the box, so a couple of samples of it may stay.
    history_weight *= mix(min(1.0, 1.0 / (clip.w * clip.w)), 1.0, protection);
    history_weight = max(history_weight, min(kept_weight, REJECTED_HISTORY_WEIGHT));
    // Accumulation (step 8).
    history_weight = min(history_weight, mix(MAX_HISTORY_WEIGHT, MOVING_HISTORY_WEIGHT, saturate(velocity)));
    let rectified = mix(clip.xyz, history_colour, protection);
    let total = history_weight + n.weight;
    let alpha = n.weight / max(total, 1.0e-6);
    let blended = mix(rectified, n.colour, alpha);
    // The input's variance about history (Finch's incremental weighted
    // variance), from deviations limited to a few box sizes so that a change
    // the box rejects does not teach it.
    let spread = clamp(n.colour - history_colour, -LEARN_LIMIT * extent, LEARN_LIMIT * extent);

    var out: TsrOutput;
    out.color = vec4<f32>(linear_rgb(blended), output_alpha);
    out.view_depth = current_depth;
    out.state = vec4<f32>(min(total, MAX_HISTORY_WEIGHT), lock, new_residual_fast, learnt_deviation);
    out.noise = vec4<f32>(sqrt((1.0 - alpha) * (variance + alpha * dot(spread, spread))), learnt_coarse_deviation,
        new_residual_slow, new_reference);
    out.coarse = vec2<f32>(new_slow, new_fast);
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
