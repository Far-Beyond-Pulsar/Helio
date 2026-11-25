// The irradiance signal (`helio_frame_api::denoise::IRRADIANCE_SNIPPET`,
// `//!use frame_denoise_irradiance`): which pixel of a block is traced, a
// producer's samples, a reader's decode. The tables are in
// `helio_frame_api::denoise`.

// YCoCg (Y = (r + 2g + b) / 4).
fn denoise_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(dot(rgb, vec3<f32>(0.25, 0.5, 0.25)), 0.5 * (rgb.r - rgb.b), dot(rgb, vec3<f32>(-0.25, 0.5, -0.25)));
}

fn denoise_rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// The Bayer rank of `p` within a `side x side` group (side 1, 2 or 4).
fn denoise_bayer_rank(p: vec2<u32>, side: u32) -> u32 {
    let a = p.x ^ p.y;
    if side == 2u {
        return ((a & 1u) << 1u) | (p.y & 1u);
    }
    if side == 4u {
        return ((a & 1u) << 3u) | ((p.y & 1u) << 2u) | (((a >> 1u) & 1u) << 1u) | ((p.y >> 1u) & 1u);
    }
    return 0u;
}

// The place of Bayer rank `r` within a `side x side` group.
fn denoise_bayer_place(r: u32, side: u32) -> vec2<u32> {
    var a = 0u;
    var y = 0u;
    if side == 2u {
        a = (r >> 1u) & 1u;
        y = r & 1u;
    } else if side == 4u {
        a = ((r >> 3u) & 1u) | (((r >> 1u) & 1u) << 1u);
        y = ((r >> 2u) & 1u) | ((r & 1u) << 1u);
    }
    return vec2<u32>(a ^ y, y);
}

// The pixel of block `block` traced on frame `frame`, within the block
// (`helio_frame_api::denoise::traced_offset`).
fn denoise_traced_offset(block: vec2<u32>, frame: u32, side: u32) -> vec2<u32> {
    let phase = denoise_bayer_rank(block % vec2<u32>(side), side);
    return denoise_bayer_place((frame + phase) % (side * side), side);
}

// The internal pixel block `block` traces on frame `frame` (its last row and
// column of blocks clamped into an internal image of `size`).
fn denoise_traced_pixel(block: vec2<u32>, frame: u32, side: u32, size: vec2<u32>) -> vec2<u32> {
    return min(block * side + denoise_traced_offset(block, frame, side), size - 1u);
}

// A traced pixel's inputs, summed over its samples.
struct DenoiseIrradianceInput {
    // Input 0: luminance, Co, Cg of the irradiance estimate at the traced
    // normal; the hit distance.
    signal0: vec4<f32>,
    // Input 1: the luminance-weighted direction; the confidence.
    signal1: vec4<f32>,
}

// Adds a sample to `input`: `contribution` the RGB irradiance it estimates
// at the traced normal (incident radiance times the cosine over the
// sample's density), `dir` the unit direction it arrived from. `count`
// samples summed are divided by `count` by the producer (or weighted as
// its estimator says); the hit distance and confidence are its own.
fn denoise_irradiance_add(input: DenoiseIrradianceInput, contribution: vec3<f32>, dir: vec3<f32>) -> DenoiseIrradianceInput {
    let c = denoise_ycocg(max(contribution, vec3<f32>(0.0)));
    var out = input;
    out.signal0 = vec4<f32>(input.signal0.xyz + c, input.signal0.w);
    out.signal1 = vec4<f32>(input.signal1.xyz + c.x * dir, input.signal1.w);
    return out;
}

// The share of the luminance-weighted direction that is the irradiance's
// first-order term: over a uniformly lit hemisphere the luminance-weighted
// mean direction is 2/3 of the normal, and the harmonics' cosine
// convolution gives the irradiance a gradient of 1/2 of its value along it.
const DENOISE_IRRADIANCE_SLOPE: f32 = 0.75;

// A traced sample as the linear irradiance E(n) = a + b . n, exact at its
// traced normal `n0`: x a, yzw b.
fn denoise_irradiance_linear(signal0: vec4<f32>, signal1: vec4<f32>, n0: vec3<f32>) -> vec4<f32> {
    let b = DENOISE_IRRADIANCE_SLOPE * signal1.xyz;
    return vec4<f32>(signal0.x - dot(b, n0), b);
}

// RGB irradiance at normal `n` from luminance `y` and chroma ratios.
fn denoise_irradiance_rgb(y: f32, chroma: vec2<f32>) -> vec3<f32> {
    return max(denoise_rgb(vec3<f32>(y, chroma * y)), vec3<f32>(0.0));
}

// The denoiser's output at a pixel of normal `n`: RGB irradiance.
fn denoise_irradiance(out0: vec4<f32>, out1: vec4<f32>, n: vec3<f32>) -> vec3<f32> {
    return denoise_irradiance_rgb(max(out0.x + dot(out1.xyz, n), 0.0), out0.yz);
}

// The samples' worth at which an estimate is settled
// (`helio_frame_api::denoise::SETTLED_SAMPLES`).
const DENOISE_SETTLED_SAMPLES: f32 = 16.0;

// The samples' worth behind the denoiser's output (below 0: no light yet).
fn denoise_irradiance_samples(out0: vec4<f32>) -> f32 {
    return out0.w;
}

// The denoiser's output at a pixel of normal `n` leaning on the reader's
// prior `prior` (RGB irradiance, a low-variance estimate of the same light)
// where it rests on fewer than DENOISE_SETTLED_SAMPLES samples: the two
// weighted by the samples each is worth, the prior the deficit times
// `hold` (in [0, 1], how much of the prior is there). The weights depend on
// counts alone, never on either value's light.
fn denoise_irradiance_with_prior(out0: vec4<f32>, out1: vec4<f32>, n: vec3<f32>, prior: vec3<f32>, hold: f32) -> vec3<f32> {
    let e = denoise_irradiance(out0, out1, n);
    let samples = max(denoise_irradiance_samples(out0), 0.0);
    let k = saturate(hold) * max(DENOISE_SETTLED_SAMPLES - samples, 0.0);
    if samples + k <= 0.0 { return e; }
    return (samples * e + k * max(prior, vec3<f32>(0.0))) / (samples + k);
}

// A pass-through signal's texel at a pixel of normal `n`, traced at `n0`.
fn denoise_irradiance_input(signal0: vec4<f32>, signal1: vec4<f32>, n0: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let l = denoise_irradiance_linear(signal0, signal1, n0);
    let y = max(l.x + dot(l.yzw, n), 0.0);
    return denoise_irradiance_rgb(y, signal0.yz / max(signal0.x, 1e-20));
}
