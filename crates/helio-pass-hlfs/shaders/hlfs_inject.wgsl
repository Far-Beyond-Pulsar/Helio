//! HLFS Radiance Injection Compute Shader (Simplified)
//!
//! Proof-of-concept that demonstrates the injection phase.
//! In production, this would use atomic operations or double-buffering.

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

struct LightSample {
    position:  vec3<f32>,
    _pad0:     f32,
    direction: vec3<f32>,
    _pad1:     f32,
    radiance:  vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera:    Camera;
@group(0) @binding(1) var<uniform> globals:   HlfsGlobals;
@group(0) @binding(3) var<storage, read_write> samples: array<LightSample>;
@group(0) @binding(8) var clip_stack_level0: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var clip_stack_level1: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var clip_stack_level2: texture_storage_3d<rgba16float, write>;
@group(0) @binding(11) var clip_stack_level3: texture_storage_3d<rgba16float, write>;

const VOXEL_RESOLUTION: u32 = 128u;

fn world_to_voxel(world_pos: vec3<f32>, level: u32) -> vec3<i32> {
    let half_extent = globals.near_field_size * 0.5 * pow(globals.cascade_scale, f32(level));
    let local = world_pos - globals.camera_position;
    let uv = (local / (2.0 * half_extent)) + vec3<f32>(0.5);
    let coord = clamp(uv, vec3<f32>(0.0), vec3<f32>(0.999)) * vec3<f32>(f32(VOXEL_RESOLUTION));
    return vec3<i32>(coord);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_pos = global_id.xy;

    if (pixel_pos.x >= globals.screen_width || pixel_pos.y >= globals.screen_height) {
        return;
    }

    let pixel_idx = pixel_pos.y * globals.screen_width + pixel_pos.x;
    let base_sample_idx = pixel_idx * globals.sample_count;

    for (var i = 0u; i < globals.sample_count; i = i + 1u) {
        let sample = samples[base_sample_idx + i];
        let world_pos = sample.position;

        let radiance = sample.radiance;

        // Inject into multilevel clip-stack with decreasing influence.
        let levels = 4u;
        for (var level = 0u; level < levels; level = level + 1u) {
            let voxel = world_to_voxel(world_pos, level);
            let color = vec4<f32>(radiance.rgb * (1.0 - f32(level) * 0.2), 1.0);
            if (level == 0u) {
                textureStore(clip_stack_level0, voxel, color);
            } else if (level == 1u) {
                textureStore(clip_stack_level1, voxel, color * 0.8);
            } else if (level == 2u) {
                textureStore(clip_stack_level2, voxel, color * 0.6);
            } else {
                textureStore(clip_stack_level3, voxel, color * 0.4);
            }
        }
    }
}
