// TAA (Temporal Anti-Aliasing) shader

@group(0) @binding(0) var current_frame: texture_2d<f32>;
@group(0) @binding(1) var history_frame: texture_2d<f32>;
@group(0) @binding(2) var velocity_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var linear_sampler: sampler;
@group(0) @binding(5) var point_sampler: sampler;

struct TaaUniform {
    feedback_min: f32,
    feedback_max: f32,
    jitter_offset: vec2<f32>,
}

@group(0) @binding(6) var<uniform> taa: TaaUniform;

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

fn rgb_to_ycocg(rgb: vec3<f32>) -> vec3<f32> {
    let y = dot(rgb, vec3<f32>(0.25, 0.5, 0.25));
    let co = dot(rgb, vec3<f32>(0.5, 0.0, -0.5));
    let cg = dot(rgb, vec3<f32>(-0.25, 0.5, -0.25));
    return vec3<f32>(y, co, cg);
}

fn ycocg_to_rgb(ycocg: vec3<f32>) -> vec3<f32> {
    let y = ycocg.x;
    let co = ycocg.y;
    let cg = ycocg.z;
    return vec3<f32>(
        y + co - cg,
        y + cg,
        y - co - cg
    );
}

fn sample_catmull_rom(tex: texture_2d<f32>, samp: sampler, uv: vec2<f32>) -> vec3<f32> {
    let dimensions = vec2<f32>(textureDimensions(tex));
    let sample_pos = uv * dimensions;
    let tex_pos = floor(sample_pos - 0.5) + 0.5;
    let f = sample_pos - tex_pos;
    
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    
    let w12 = w1 + w2;
    let offset12 = w2 / w12;
    
    let texel_size = 1.0 / dimensions;
    let uv0 = (tex_pos - 1.0) * texel_size;
    let uv12 = (tex_pos + offset12) * texel_size;
    let uv3 = (tex_pos + 2.0) * texel_size;
    
    var result = vec3<f32>(0.0);
    result = result + textureSample(tex, samp, vec2<f32>(uv0.x, uv0.y)).rgb * w0.x * w0.y;
    result = result + textureSample(tex, samp, vec2<f32>(uv12.x, uv0.y)).rgb * w12.x * w0.y;
    result = result + textureSample(tex, samp, vec2<f32>(uv3.x, uv0.y)).rgb * w3.x * w0.y;
    
    result = result + textureSample(tex, samp, vec2<f32>(uv0.x, uv12.y)).rgb * w0.x * w12.y;
    result = result + textureSample(tex, samp, vec2<f32>(uv12.x, uv12.y)).rgb * w12.x * w12.y;
    result = result + textureSample(tex, samp, vec2<f32>(uv3.x, uv12.y)).rgb * w3.x * w12.y;
    
    result = result + textureSample(tex, samp, vec2<f32>(uv0.x, uv3.y)).rgb * w0.x * w3.y;
    result = result + textureSample(tex, samp, vec2<f32>(uv12.x, uv3.y)).rgb * w12.x * w3.y;
    result = result + textureSample(tex, samp, vec2<f32>(uv3.x, uv3.y)).rgb * w3.x * w3.y;
    
    return max(result, vec3<f32>(0.0));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dimensions = textureDimensions(current_frame);
    let texel_size = 1.0 / vec2<f32>(dimensions);
    
    // Sample current frame
    let current_color = textureSample(current_frame, linear_sampler, in.uv).rgb;
    
    // Sample velocity (use point sampler for non-filterable texture)
    let velocity = textureSample(velocity_tex, point_sampler, in.uv).xy;
    
    // Calculate history UV
    let history_uv = in.uv - velocity;
    
    // Check if history UV is valid
    if history_uv.x < 0.0 || history_uv.x > 1.0 || history_uv.y < 0.0 || history_uv.y > 1.0 {
        return vec4<f32>(current_color, 1.0);
    }
    
    // Sample history with high-quality filter
    let history_color = sample_catmull_rom(history_frame, linear_sampler, history_uv);
    
    // Neighborhood clamping - sample 3x3 grid
    var color_min = vec3<f32>(1e10);
    var color_max = vec3<f32>(-1e10);
    var m1 = vec3<f32>(0.0);
    var m2 = vec3<f32>(0.0);
    
    for (var x = -1; x <= 1; x = x + 1) {
        for (var y = -1; y <= 1; y = y + 1) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let neighbor = textureSample(current_frame, linear_sampler, in.uv + offset).rgb;
            let neighbor_ycocg = rgb_to_ycocg(neighbor);
            
            color_min = min(color_min, neighbor_ycocg);
            color_max = max(color_max, neighbor_ycocg);
            
            m1 = m1 + neighbor_ycocg;
            m2 = m2 + neighbor_ycocg * neighbor_ycocg;
        }
    }
    
    // Variance clipping
    let sample_count = 9.0;
    let mean = m1 / sample_count;
    let variance = (m2 / sample_count) - (mean * mean);
    let std_dev = sqrt(max(variance, vec3<f32>(0.0)));
    
    let box_min = mean - 1.25 * std_dev;
    let box_max = mean + 1.25 * std_dev;
    
    // Clamp history color
    let history_ycocg = rgb_to_ycocg(history_color);
    let clamped_history_ycocg = clamp(history_ycocg, box_min, box_max);
    let clamped_history = ycocg_to_rgb(clamped_history_ycocg);
    
    // Calculate blend factor based on motion
    let velocity_len = length(velocity);
    let blend_factor = mix(taa.feedback_max, taa.feedback_min, saturate(velocity_len * 100.0));
    
    // Blend current and history
    let result = mix(current_color, clamped_history, blend_factor);
    
    return vec4<f32>(result, 1.0);
}
