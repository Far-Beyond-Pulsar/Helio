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
@group(0) @binding(4) var clip_stack_level0: texture_3d<f32>;
@group(0) @binding(5) var gbuf_normal: texture_2d<f32>;
@group(0) @binding(6) var gbuf_depth: texture_depth_2d;
@group(0) @binding(7) var clip_stack_sampler: sampler;

// Simple hash function for random sampling
fn hash13(p3: vec3<f32>) -> f32 {
    var p = fract(p3 * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

fn reconstruct_world_pos(pixel_pos: vec2<u32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(
        (f32(pixel_pos.x) + 0.5) / f32(globals.screen_width) * 2.0 - 1.0,
        1.0 - (f32(pixel_pos.y) + 0.5) / f32(globals.screen_height) * 2.0,
        depth * 2.0 - 1.0,
        1.0,
    );
    let world_h = camera.view_proj_inv * ndc;
    return world_h.xyz / world_h.w;
}

fn world_to_clipstack_uv(world_pos: vec3<f32>, level: u32) -> vec3<f32> {
    let half_extent = globals.near_field_size * 0.5 * pow(globals.cascade_scale, f32(level));
    let local = world_pos - globals.camera_position;
    let uv = (local / (2.0 * half_extent)) + vec3<f32>(0.5);
    return clamp(uv, vec3<f32>(0.0), vec3<f32>(1.0));
}

fn evaluate_light_sample(world_pos: vec3<f32>, normal: vec3<f32>, light: GpuLight) -> vec3<f32> {
    let light_pos = light.position_range.xyz;
    let delta = light_pos - world_pos;
    let dist = max(length(delta), 0.001);
    let dir = normalize(delta);
    let ndotl = max(dot(normal, dir), 0.0);
    let attenuation = light.position_range.w / (dist * dist + 1.0);
    let light_color = light.color_intensity.xyz;
    let intensity = light.color_intensity.w;
    return light_color * intensity * ndotl * attenuation;
}

// Sample light using importance from clip-stack
fn importance_sample_light(pixel_pos: vec2<u32>, sample_idx: u32) -> LightSample {
    var sample: LightSample;

    if (globals.light_count == 0u) {
        sample.position = camera.position_near.xyz;
        sample.direction = camera.forward_far.xyz;
        sample.radiance = vec4<f32>(0.0);
        return sample;
    }

    let depth = textureLoad(gbuf_depth, vec2<i32>(pixel_pos), 0);
    let world_pos = reconstruct_world_pos(pixel_pos, depth);
    let surf_normal = normalize(textureLoad(gbuf_normal, vec2<i32>(pixel_pos), 0).xyz * 2.0 - 1.0);

    let pixel_coord = vec2<f32>(pixel_pos);
    let seed = vec3<f32>(pixel_coord, f32(sample_idx + globals.frame));
    let rnd = hash13(seed);

    // Choose a light with importance sampling based on geometry + intensity
    var best_radiance = vec3<f32>(0.0);
    var best_weight: f32 = 0.0;
    let max_trials = min(globals.light_count, 8u);
    for (var i = 0u; i < max_trials; i = i + 1u) {
        let local_rnd = hash13(seed + vec3<f32>(f32(i), f32(i * 3u), f32(i * 7u)));
        let candidate = lights[(u32(local_rnd * f32(globals.light_count))) % globals.light_count];
        let radiance = evaluate_light_sample(world_pos, surf_normal, candidate);
        let weight = length(radiance);
        if (weight > best_weight) {
            best_weight = weight;
            best_radiance = radiance;
            sample.position = world_pos;
            sample.direction = normalize(candidate.position_range.xyz - world_pos);
        }
    }

    // Add clip-stack indirect contribution from nearest level
    let field_uv = world_to_clipstack_uv(world_pos, 0u);
    let field_radiance = textureSampleLevel(clip_stack_level0, clip_stack_sampler, field_uv, 0).rgb;

    sample.radiance = vec4<f32>(best_radiance + field_radiance * 0.4, 1.0);
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
