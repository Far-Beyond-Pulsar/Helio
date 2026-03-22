//! Transparent geometry pass shader.
//!
//! Adapted from geometry.wgsl for transparent/alpha-blended objects.
//! Uses Group 0 (camera, globals, instances) only.
//! Alpha blending is handled by the render pipeline blend state.
//! A full implementation would add Group 1 for per-material colors/textures.

struct Camera {
    view_proj: mat4x4<f32>,
    position:  vec3<f32>,
    time:      f32,
}

struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec4<f32>,
    rc_world_min:      vec4<f32>,
    rc_world_max:      vec4<f32>,
    csm_splits:        vec4<f32>,
}

/// Must match `GpuInstanceData` in libhelio.
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
@group(0) @binding(1) var<uniform>       globals:       Globals;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(vertex: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(vertex.position, 1.0);
    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );
    var out: VertexOutput;
    out.clip_position  = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat * decode_snorm8x4(vertex.normal));
    out.tex_coords     = vertex.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple ambient + normal-based shading with translucent alpha.
    // A full implementation would sample per-material textures from Group 1.
    let ambient = globals.ambient_color.rgb * globals.ambient_intensity;
    let normal_shade = in.world_normal * 0.5 + 0.5;
    let color = ambient + normal_shade * 0.3;
    let alpha = 0.5; // Fixed 50% alpha; full impl reads per-material alpha
    return vec4<f32>(color, alpha);
}
