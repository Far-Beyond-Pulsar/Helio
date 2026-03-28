//! HLFS Hierarchical Propagation Compute Shader
//!
//! Propagates energy from fine levels to coarse levels in a mip-map style.
//! This creates the hierarchical structure that enables efficient queries.

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
@group(0) @binding(4) var clip_stack_level0: texture_storage_3d<rgba16float, read_write>;

const VOXEL_RESOLUTION: u32 = 128u;

// Average 2x2x2 neighborhood from fine level (simplified single-level propagation)
@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (any(global_id >= vec3<u32>(VOXEL_RESOLUTION / 2u))) {
        return;
    }

    // Sample 8 neighbors from fine level
    let base_coord = global_id * 2u;
    var sum = vec4<f32>(0.0);
    var count = 0.0;

    for (var dx = 0u; dx < 2u; dx++) {
        for (var dy = 0u; dy < 2u; dy++) {
            for (var dz = 0u; dz < 2u; dz++) {
                let coord = base_coord + vec3<u32>(dx, dy, dz);
                if (all(coord < vec3<u32>(VOXEL_RESOLUTION))) {
                    sum += textureLoad(clip_stack_level0, vec3<i32>(coord));
                    count += 1.0;
                }
            }
        }
    }

    // In full implementation, write to coarser level
    // For now, this is a placeholder showing the propagation concept
    if (count > 0.0) {
        let avg = sum / count;
        // Would write to level1, level2, etc.
    }
}
