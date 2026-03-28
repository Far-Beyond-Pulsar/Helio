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
@group(0) @binding(4) var clip_stack_level0: texture_3d<f32>;
@group(0) @binding(8) var out_level1: texture_storage_3d<rgba16float, write>;
@group(0) @binding(9) var out_level2: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var out_level3: texture_storage_3d<rgba16float, write>;

const VOXEL_RESOLUTION: u32 = 128u;

@compute @workgroup_size(8, 8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    // Level 0 smoothing
    if (all(global_id < vec3<u32>(VOXEL_RESOLUTION))) {
        let uv = vec3<f32>(global_id) / f32(VOXEL_RESOLUTION);
        let center = textureLoad(clip_stack_level0, vec3<i32>(global_id), 0).rgb;
        let neighbors = vec3<f32>(0.0);

        // 6-neighborhood blur for dynamic response
        var sum = center;
        var count = 1.0;
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
                for (var dz: i32 = -1; dz <= 1; dz = dz + 1) {
                    if (abs(dx) + abs(dy) + abs(dz) == 1) {
                        let sample_coord = vec3<i32>(i32(global_id.x) + dx, i32(global_id.y) + dy, i32(global_id.z) + dz);
                        if (all(sample_coord >= vec3<i32>(0)) && all(sample_coord < vec3<i32>(i32(VOXEL_RESOLUTION)))) {
                            sum = sum + textureLoad(clip_stack_level0, sample_coord, 0).rgb;
                            count = count + 1.0;
                        }
                    }
                }
            }
        }

        let color = vec4<f32>(sum / count, 1.0);
        // optional: can write level0 smoothing into level1 as a blended factor
        // textureStore(out_level1, vec3<i32>(global_id), color);
    }

    // Level 1 downsample from level0
    if (all(global_id < vec3<u32>(VOXEL_RESOLUTION / 2u))) {
        let src_coord = global_id * 2u;
        var sum = vec3<f32>(0.0);
        for (var x = 0u; x < 2u; x = x + 1u) {
            for (var y = 0u; y < 2u; y = y + 1u) {
                for (var z = 0u; z < 2u; z = z + 1u) {
                    let pos = vec3<i32>(i32(src_coord.x + x), i32(src_coord.y + y), i32(src_coord.z + z));
                    sum = sum + textureLoad(clip_stack_level0, pos, 0).rgb;
                }
            }
        }
        textureStore(out_level1, vec3<i32>(global_id), vec4<f32>(sum / 8.0, 1.0));
    }

    // Level 2 downsample from level0
    if (all(global_id < vec3<u32>(VOXEL_RESOLUTION / 4u))) {
        let src_coord = global_id * 4u;
        var sum = vec3<f32>(0.0);
        for (var x = 0u; x < 4u; x = x + 1u) {
            for (var y = 0u; y < 4u; y = y + 1u) {
                for (var z = 0u; z < 4u; z = z + 1u) {
                    let pos = vec3<i32>(i32(src_coord.x + x), i32(src_coord.y + y), i32(src_coord.z + z));
                    sum = sum + textureLoad(clip_stack_level0, pos, 0).rgb;
                }
            }
        }
        textureStore(out_level2, vec3<i32>(global_id), vec4<f32>(sum / 64.0, 1.0));
    }

    // Level 3 downsample from level0
    if (all(global_id < vec3<u32>(VOXEL_RESOLUTION / 8u))) {
        let src_coord = global_id * 8u;
        var sum = vec3<f32>(0.0);
        for (var x = 0u; x < 8u; x = x + 1u) {
            for (var y = 0u; y < 8u; y = y + 1u) {
                for (var z = 0u; z < 8u; z = z + 1u) {
                    let pos = vec3<i32>(i32(src_coord.x + x), i32(src_coord.y + y), i32(src_coord.z + z));
                    sum = sum + textureLoad(clip_stack_level0, pos, 0).rgb;
                }
            }
        }
        textureStore(out_level3, vec3<i32>(global_id), vec4<f32>(sum / 512.0, 1.0));
    }
}
