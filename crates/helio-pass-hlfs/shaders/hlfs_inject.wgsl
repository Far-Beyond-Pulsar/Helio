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
@group(0) @binding(4) var output_tex: texture_storage_3d<rgba16float, write>;

const VOXEL_RESOLUTION: u32 = 128u;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_pos = global_id.xy;

    if (pixel_pos.x >= globals.screen_width || pixel_pos.y >= globals.screen_height) {
        return;
    }

    let pixel_idx = pixel_pos.y * globals.screen_width + pixel_pos.x;
    let base_sample_idx = pixel_idx * globals.sample_count;

    // Simplified: just write first sample to a test voxel
    if (base_sample_idx < arrayLength(&samples)) {
        let sample = samples[base_sample_idx];

        // Map to voxel coordinate (simplified)
        let voxel_x = pixel_pos.x % VOXEL_RESOLUTION;
        let voxel_y = pixel_pos.y % VOXEL_RESOLUTION;
        let voxel_z = u32(sample.radiance.r * 128.0) % VOXEL_RESOLUTION;

        textureStore(output_tex, vec3<i32>(i32(voxel_x), i32(voxel_y), i32(voxel_z)), sample.radiance);
    }
}
