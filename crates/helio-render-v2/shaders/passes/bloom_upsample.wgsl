//! Bloom upsample pass — dual-Kawase tent filter.
//!
//! Each dispatch doubles the resolution.  The final dispatch composites the
//! accumulated bloom back onto the HDR scene colour (additive blend).
//!
//! The 9-tap tent filter provides smooth, artefact-free upsampling

@group(0) @binding(0) var src_texture: texture_2d<f32>;   // current bloom mip
@group(0) @binding(1) var src_sampler: sampler;

struct UpsampleUniforms {
    filter_radius: f32,   // typically 0.001 * window_height / mip_resolution
    blend_factor:  f32,   // 1.0 for pure bloom; < 1.0 when compositing onto scene
}
@group(0) @binding(2) var <uniform> uu: UpsampleUniforms;

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

// 9-tap tent filter:  weight layout (×/9 total)
//  1 2 1
//  2 4 2
//  1 2 1
fn upsample_tent(uv: vec2<f32>, r: f32) -> vec3<f32> {
    let a = textureSample(src_texture, src_sampler, uv + vec2<f32>(-r, -r)).rgb;
    let b = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0, -r)).rgb;
    let c = textureSample(src_texture, src_sampler, uv + vec2<f32>( r, -r)).rgb;
    let d = textureSample(src_texture, src_sampler, uv + vec2<f32>(-r, 0.0)).rgb;
    let e = textureSample(src_texture, src_sampler, uv).rgb;
    let f = textureSample(src_texture, src_sampler, uv + vec2<f32>( r, 0.0)).rgb;
    let g = textureSample(src_texture, src_sampler, uv + vec2<f32>(-r,  r)).rgb;
    let h = textureSample(src_texture, src_sampler, uv + vec2<f32>( 0.0,  r)).rgb;
    let i = textureSample(src_texture, src_sampler, uv + vec2<f32>( r,  r)).rgb;
    return (a + c + g + i) * (1.0/16.0)
         + (b + d + f + h) * (2.0/16.0)
         +  e               * (4.0/16.0);
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let col = upsample_tent(in.uv, uu.filter_radius);
    return vec4<f32>(col * uu.blend_factor, 1.0);
}
