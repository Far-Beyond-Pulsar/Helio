// FXAA (Fast Approximate Anti-Aliasing) shader
// Based on NVIDIA FXAA 3.11 algorithm

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var input_sampler: sampler;

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

const EDGE_THRESHOLD_MIN: f32 = 0.0312;
const EDGE_THRESHOLD_MAX: f32 = 0.125;
const SUBPIXEL_QUALITY: f32 = 0.75;
const ITERATIONS: i32 = 12;

fn rgb2luma(rgb: vec3<f32>) -> f32 {
    return dot(rgb, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(input_tex);
    let texel_size = 1.0 / vec2<f32>(dimensions);
    let texel = vec2<i32>(in.uv * vec2<f32>(dimensions));
    
    // Sample center and neighbors
    let rgb_center = textureSample(input_tex, input_sampler, in.uv).rgb;
    let luma_center = rgb2luma(rgb_center);
    
    let luma_down = rgb2luma(textureLoad(input_tex, texel + vec2<i32>(0, -1), 0).rgb);
    let luma_up = rgb2luma(textureLoad(input_tex, texel + vec2<i32>(0, 1), 0).rgb);
    let luma_left = rgb2luma(textureLoad(input_tex, texel + vec2<i32>(-1, 0), 0).rgb);
    let luma_right = rgb2luma(textureLoad(input_tex, texel + vec2<i32>(1, 0), 0).rgb);
    
    // Find min/max luma
    let luma_min = min(luma_center, min(min(luma_down, luma_up), min(luma_left, luma_right)));
    let luma_max = max(luma_center, max(max(luma_down, luma_up), max(luma_left, luma_right)));
    
    let luma_range = luma_max - luma_min;
    
    // Early exit if no edge
    if luma_range < max(EDGE_THRESHOLD_MIN, luma_max * EDGE_THRESHOLD_MAX) {
        return vec4<f32>(rgb_center, 1.0);
    }
    
    // Sample corners
    let luma_down_left = rgb2luma(textureLoad(input_tex, texel + vec2<i32>(-1, -1), 0).rgb);
    let luma_up_right = rgb2luma(textureLoad(input_tex, texel + vec2<i32>(1, 1), 0).rgb);
    let luma_up_left = rgb2luma(textureLoad(input_tex, texel + vec2<i32>(-1, 1), 0).rgb);
    let luma_down_right = rgb2luma(textureLoad(input_tex, texel + vec2<i32>(1, -1), 0).rgb);
    
    // Compute gradient
    let luma_down_up = luma_down + luma_up;
    let luma_left_right = luma_left + luma_right;
    let luma_left_corners = luma_down_left + luma_up_left;
    let luma_down_corners = luma_down_left + luma_down_right;
    let luma_right_corners = luma_down_right + luma_up_right;
    let luma_up_corners = luma_up_right + luma_up_left;
    
    let edge_horizontal = abs(-2.0 * luma_left + luma_left_corners) +
                         abs(-2.0 * luma_center + luma_down_up) * 2.0 +
                         abs(-2.0 * luma_right + luma_right_corners);
    let edge_vertical = abs(-2.0 * luma_up + luma_up_corners) +
                       abs(-2.0 * luma_center + luma_left_right) * 2.0 +
                       abs(-2.0 * luma_down + luma_down_corners);
    
    let is_horizontal = edge_horizontal >= edge_vertical;
    
    // Select edge direction
    var luma1: f32;
    var luma2: f32;
    var gradient1: f32;
    var gradient2: f32;
    
    if is_horizontal {
        luma1 = luma_down;
        luma2 = luma_up;
        gradient1 = edge_horizontal;
        gradient2 = edge_vertical;
    } else {
        luma1 = luma_left;
        luma2 = luma_right;
        gradient1 = edge_vertical;
        gradient2 = edge_horizontal;
    }
    
    // Compute gradient in perpendicular direction
    let gradient_scaled = gradient1 * 0.25;
    
    // Subpixel shift
    let luma_local_average = (luma1 + luma2) * 0.5;
    let luma_local_average_delta = abs(luma_local_average - luma_center);
    
    var pixel_offset: vec2<f32>;
    if is_horizontal {
        pixel_offset = vec2<f32>(0.0, texel_size.y);
    } else {
        pixel_offset = vec2<f32>(texel_size.x, 0.0);
    }
    
    var uv_offset = in.uv;
    if luma_local_average_delta >= gradient_scaled {
        if luma_local_average < luma_center {
            uv_offset = uv_offset - pixel_offset * SUBPIXEL_QUALITY;
        } else {
            uv_offset = uv_offset + pixel_offset * SUBPIXEL_QUALITY;
        }
    }
    
    return vec4<f32>(textureSample(input_tex, input_sampler, uv_offset).rgb, 1.0);
}
