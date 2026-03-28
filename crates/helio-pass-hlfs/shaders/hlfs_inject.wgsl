//! HLFS Radiance Injection Compute Shader
//!
//! Injects sampled radiance values into the appropriate voxels
//! of the clip-stack hierarchy. This "seeds" the field with
//! direct lighting information.

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
@group(0) @binding(3) var<storage, read> samples: array<LightSample>;
@group(0) @binding(4) var clip_stack_level0: texture_storage_3d<rgba16float, read_write>;

const VOXEL_RESOLUTION: u32 = 128u;

// Convert world position to voxel coordinates for level 0
fn world_to_voxel_l0(world_pos: vec3<f32>) -> vec3<i32> {
    let camera_pos = camera.position_near.xyz;
    let rel_pos = world_pos - camera_pos;

    // Map [-near_field_size/2, +near_field_size/2] to [0, VOXEL_RESOLUTION]
    let half_size = globals.near_field_size * 0.5;
    let normalized = (rel_pos + vec3<f32>(half_size)) / globals.near_field_size;
    let voxel_coord = vec3<i32>(normalized * f32(VOXEL_RESOLUTION));

    return voxel_coord;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pixel_pos = global_id.xy;

    if (pixel_pos.x >= globals.screen_width || pixel_pos.y >= globals.screen_height) {
        return;
    }

    let pixel_idx = pixel_pos.y * globals.screen_width + pixel_pos.x;
    let base_sample_idx = pixel_idx * globals.sample_count;

    // Inject all K samples for this pixel into clip-stack
    for (var i = 0u; i < globals.sample_count; i++) {
        let sample = samples[base_sample_idx + i];
        let voxel_coord = world_to_voxel_l0(sample.position);

        // Bounds check
        if (all(voxel_coord >= vec3<i32>(0)) &&
            all(voxel_coord < vec3<i32>(i32(VOXEL_RESOLUTION)))) {

            // Atomic add radiance to voxel (simplified: using textureStore with temporal blend)
            let current = textureLoad(clip_stack_level0, voxel_coord);
            let blended = current * globals.temporal_blend + sample.radiance * (1.0 - globals.temporal_blend);
            textureStore(clip_stack_level0, voxel_coord, blended);
        }
    }
}
