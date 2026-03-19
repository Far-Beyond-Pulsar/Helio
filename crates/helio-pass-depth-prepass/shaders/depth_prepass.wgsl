//! Depth-only prepass shader.
//!
//! Transforms vertices through the camera view-projection and writes to the depth buffer.
//! No fragment output — depth writes are implicit.

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

/// Per-instance GPU data.  Must match `GpuInstanceData` in libhelio.
struct GpuInstanceData {
    transform:     mat4x4<f32>,
    normal_mat_0:  vec4<f32>,
    normal_mat_1:  vec4<f32>,
    normal_mat_2:  vec4<f32>,
    bounds:        vec4<f32>,
    mesh_id:       u32,
    material_id:   u32,
    flags:         u32,
    _pad:          u32,
}

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var<storage, read> instance_data: array<GpuInstanceData>;

@vertex
fn vs_main(
    @location(0)             position:    vec3<f32>,
    @location(2)             _tex_coords: vec2<f32>,  // kept for vertex layout compatibility
    @builtin(instance_index) slot:        u32,
) -> @invariant @builtin(position) vec4<f32> {
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(position, 1.0);
    return camera.view_proj * world_pos;
}
