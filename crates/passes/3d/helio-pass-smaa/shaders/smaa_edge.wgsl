// SMAA Edge Detection Pass

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

fn rgb2luma(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

const EDGE_THRESHOLD: f32 = 0.1;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec2<f32> {
    let dimensions = textureDimensions(input_tex);
    let texel_size = 1.0 / vec2<f32>(dimensions);
    
    // Sample center and neighbors
    let center = textureSample(input_tex, linear_sampler, in.uv).rgb;
    let left = textureSample(input_tex, linear_sampler, in.uv + vec2<f32>(-texel_size.x, 0.0)).rgb;
    let right = textureSample(input_tex, linear_sampler, in.uv + vec2<f32>(texel_size.x, 0.0)).rgb;
    let top = textureSample(input_tex, linear_sampler, in.uv + vec2<f32>(0.0, texel_size.y)).rgb;
    let bottom = textureSample(input_tex, linear_sampler, in.uv + vec2<f32>(0.0, -texel_size.y)).rgb;
    
    // Convert to luma
    let luma_center = rgb2luma(center);
    let luma_left = rgb2luma(left);
    let luma_right = rgb2luma(right);
    let luma_top = rgb2luma(top);
    let luma_bottom = rgb2luma(bottom);
    
    // Calculate deltas
    let delta_left = abs(luma_center - luma_left);
    let delta_right = abs(luma_center - luma_right);
    let delta_top = abs(luma_center - luma_top);
    let delta_bottom = abs(luma_center - luma_bottom);
    
    // Detect edges
    var edges = vec2<f32>(0.0);
    
    let max_horizontal = max(delta_left, delta_right);
    let max_vertical = max(delta_top, delta_bottom);
    
    if max_horizontal > EDGE_THRESHOLD {
        edges.x = 1.0;
    }
    
    if max_vertical > EDGE_THRESHOLD {
        edges.y = 1.0;
    }
    
    return edges;
}
