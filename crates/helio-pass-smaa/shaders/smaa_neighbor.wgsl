// SMAA Neighborhood Blending Pass

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var linear_sampler: sampler;
@group(0) @binding(2) var point_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32((vertex_index << 1u) & 2u);
    let y = f32(vertex_index & 2u);
    out.position = vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, y);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(input_tex);
    let texel_size = 1.0 / vec2<f32>(dimensions);
    
    // Sample center pixel
    let center = textureSample(input_tex, linear_sampler, in.uv);
    
    // Simple 4-tap bilinear filter for anti-aliasing
    let tl = textureSample(input_tex, linear_sampler, in.uv + vec2<f32>(-0.5, 0.5) * texel_size);
    let tr = textureSample(input_tex, linear_sampler, in.uv + vec2<f32>(0.5, 0.5) * texel_size);
    let bl = textureSample(input_tex, linear_sampler, in.uv + vec2<f32>(-0.5, -0.5) * texel_size);
    let br = textureSample(input_tex, linear_sampler, in.uv + vec2<f32>(0.5, -0.5) * texel_size);
    
    let result = (center + tl + tr + bl + br) * 0.2;
    
    return vec4<f32>(result.rgb, 1.0);
}
