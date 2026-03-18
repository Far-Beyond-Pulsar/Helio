//! Depth-only prepass shader.
//!
//! Transforms vertices through the camera view-projection and writes to the depth buffer.
//! No fragment output — depth writes are implicit.

struct Camera {
    view_proj: mat4x4<f32>,
    position:  vec3<f32>,
    _pad:      f32,
}

/// Per-instance GPU data.  Must match `GpuInstanceData` in libhelio.
struct GpuInstanceData {
    transform:     mat4x4<f32>,
    normal_mat_0:  vec4<f32>,
    normal_mat_1:  vec4<f32>,
    normal_mat_2:  vec4<f32>,
    bounds_center: vec3<f32>,
    bounds_radius: f32,
}

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var<storage, read> instance_data: array<GpuInstanceData>;

@vertex
fn vs_main(
    @location(0)             position:    vec3<f32>,
    @location(2)             _tex_coords: vec2<f32>,  // kept for vertex layout compatibility
    @builtin(instance_index) slot:        u32,
) -> @builtin(position) vec4<f32> {
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(position, 1.0);
    return camera.view_proj * world_pos;
}
