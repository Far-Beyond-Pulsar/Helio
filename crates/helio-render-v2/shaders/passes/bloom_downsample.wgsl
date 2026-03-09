//! Bloom downsample pass — dual-Kawase filter.
//!
//! Each dispatch halves the resolution.  The first dispatch also extracts
//! the bright threshold so only HDR pixels above `threshold` contribute.
//! Subsequent dispatches operate on the already-thresholded mip chain.
//!
//! The filter kernel is a 13-tap cross (UE5 / COD dual-Kawase variant):
//!   centre weight 0.5, ring-4 neighbour weight 0.125 each → sum = 1.0.

struct BloomUniforms {
    threshold:   f32,
    knee:        f32,   // soft knee width (same as THRESHOLD × 0.5 typically)
    intensity:   f32,
    is_first:    u32,   // 1 on the first downsample mip (threshold extraction)
}

@group(0) @binding(0) var src_texture: texture_2d<f32>;
@group(0) @binding(1) var src_sampler: sampler;
@group(0) @binding(2) var <uniform> bu: BloomUniforms;

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)      uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var uvs = array<vec2<f32>, 3>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(2.0, 1.0),
        vec2<f32>(0.0,-1.0),
    );
    var out: VSOut;
    out.clip_pos = vec4<f32>(pos[vi], 0.0, 1.0);
    out.uv       = uvs[vi];
    return out;
}

// Soft-knee threshold: preserves energy below the knee while hard-clipping
// bright values above it — avoids flickering on high-frequency details.
fn quadratic_threshold(color: vec3<f32>, threshold: f32, knee: f32) -> vec3<f32> {
    let br   = max(max(color.r, color.g), color.b);
    let rq   = clamp(br - threshold + knee, 0.0, 2.0 * knee);
    let rq2  = (rq * rq) / (4.0 * knee + 0.00001);
    let w    = max(rq2, br - threshold) / max(br, 0.00001);
    return color * w;
}

// 13-tap dual-Kawase downsample kernel:
//   s = texel_size (1.0 / src_resolution) × 0.5
//   Sample 4 inner quad centres + 4 outer quad corners + 1 centre = 13 samples
fn downsample_13tap(uv: vec2<f32>, ts: vec2<f32>) -> vec3<f32> {
    let a = textureSample(src_texture, src_sampler, uv + vec2<f32>(-2.0, -2.0) * ts).rgb;
    let b = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0, -2.0) * ts).rgb;
    let c = textureSample(src_texture, src_sampler, uv + vec2<f32>( 2.0, -2.0) * ts).rgb;
    let d = textureSample(src_texture, src_sampler, uv + vec2<f32>(-1.0, -1.0) * ts).rgb;
    let e = textureSample(src_texture, src_sampler, uv + vec2<f32>( 1.0, -1.0) * ts).rgb;
    let f = textureSample(src_texture, src_sampler, uv + vec2<f32>(-2.0,  0.0) * ts).rgb;
    let g = textureSample(src_texture, src_sampler, uv).rgb;
    let h = textureSample(src_texture, src_sampler, uv + vec2<f32>( 2.0,  0.0) * ts).rgb;
    let i = textureSample(src_texture, src_sampler, uv + vec2<f32>(-1.0,  1.0) * ts).rgb;
    let j = textureSample(src_texture, src_sampler, uv + vec2<f32>( 1.0,  1.0) * ts).rgb;
    let k = textureSample(src_texture, src_sampler, uv + vec2<f32>(-2.0,  2.0) * ts).rgb;
    let l = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,  2.0) * ts).rgb;
    let m = textureSample(src_texture, src_sampler, uv + vec2<f32>( 2.0,  2.0) * ts).rgb;
    // Staggered weights: inner 2×2 quads get 0.5, outer ring 0.125, centre 0.125
    return (d + e + i + j) * 0.5 / 4.0
        + (a + b + g + f) * 0.125 / 4.0
        + (b + c + h + g) * 0.125 / 4.0
        + (f + g + l + k) * 0.125 / 4.0
        + (g + h + m + l) * 0.125 / 4.0
        + g * 0.125 / 1.0;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let dim = textureDimensions(src_texture);
    let ts  = vec2<f32>(0.5) / vec2<f32>(f32(dim.x), f32(dim.y));   // half-texel steps
    var col = downsample_13tap(in.uv, ts);

    // On the first pass extract bright pixels via soft-knee threshold.
    if bu.is_first != 0u {
        col = quadratic_threshold(col, bu.threshold, bu.knee);
    }
    return vec4<f32>(col, 1.0);
}
