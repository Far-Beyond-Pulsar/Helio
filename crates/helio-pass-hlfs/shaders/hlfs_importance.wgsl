//! HLFS Importance Sampling Compute Shader
//!
//! Generates K importance-weighted light samples per pixel based on
//! the current clip-stack state. Uses the hierarchical field to build
//! a PDF for light selection.

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

struct HlfsGlobals {
    frame:            u32,
    sample_count:     u32,
    light_count:      u32,
    screen_width:     u32,
    screen_height:    u32,
    near_field_size:  f32,
    cascade_scale:    f32,
    temporal_blend:   f32,
    camera_position:  vec3<f32>,
    _pad0:            u32,
    camera_forward:   vec3<f32>,
    _pad1:            u32,
}

struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
}

struct LightSample {
    position:  vec3<f32>,
    _pad0:     f32,
    direction: vec3<f32>,
    _pad1:     f32,
    radiance:  vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera:    Camera;
@group(0) @binding(1) var<uniform> globals:   HlfsGlobals;
@group(0) @binding(2) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(3) var<storage, read_write> samples: array<LightSample>;
@group(0) @binding(4) var clip_stack_level0: texture_storage_3d<rgba16float, write>;

// Simple hash function for random sampling
fn hash13(p3: vec3<f32>) -> f32 {
    var p = fract(p3 * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

// Sample light using importance from clip-stack
fn importance_sample_light(pixel_pos: vec2<u32>, sample_idx: u32) -> LightSample {
    let pixel_coord = vec2<f32>(pixel_pos);
    let seed = vec3<f32>(pixel_coord, f32(sample_idx + globals.frame));
    let rnd = hash13(seed);

    // Simple uniform light selection (in production: use clip-stack PDF)
    let light_idx = u32(rnd * f32(globals.light_count)) % globals.light_count;
    let light = lights[light_idx];

    // Reconstruct world position from pixel
    let ndc = vec2<f32>(
        (pixel_coord.x / f32(globals.screen_width)) * 2.0 - 1.0,
        1.0 - (pixel_coord.y / f32(globals.screen_height)) * 2.0
    );

    // Sample light contribution
    let light_pos = light.position_range.xyz;
    let light_color = light.color_intensity.xyz;
    let light_intensity = light.color_intensity.w;

    var sample: LightSample;
    sample.position = light_pos;
    sample.direction = normalize(light_pos - camera.position_near.xyz);
    sample.radiance = vec4<f32>(light_color * light_intensity, 1.0);

    return sample;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_pos = global_id.xy;

    if (pixel_pos.x >= globals.screen_width || pixel_pos.y >= globals.screen_height) {
        return;
    }

    // Generate K samples for this pixel
    let pixel_idx = pixel_pos.y * globals.screen_width + pixel_pos.x;
    let base_sample_idx = pixel_idx * globals.sample_count;

    for (var i = 0u; i < globals.sample_count; i++) {
        let sample = importance_sample_light(pixel_pos, i);
        samples[base_sample_idx + i] = sample;
    }
}
