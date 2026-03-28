//! HLFS Hierarchical Propagation Compute Shader (Simplified)
//!
//! Demonstrates the concept of propagating energy through the hierarchy.

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

@group(0) @binding(1) var<uniform> globals: HlfsGlobals;
@group(0) @binding(4) var output_tex: texture_storage_3d<rgba16float, write>;

const VOXEL_RESOLUTION: u32 = 128u;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id >= vec3<u32>(VOXEL_RESOLUTION))) {
        return;
    }

    // Simplified propagation: write gradient pattern to demonstrate concept
    let normalized = vec3<f32>(global_id) / f32(VOXEL_RESOLUTION);
    let value = vec4<f32>(normalized, 1.0);

    textureStore(output_tex, vec3<i32>(global_id), value);
}
