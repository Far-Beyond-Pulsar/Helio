//! Shadow pass shader — GPU-driven (reads instance transform from storage buffer).

struct LightMatrix {
    mat: mat4x4<f32>,
}

struct Material {
    base_color:      vec4<f32>,
    metallic:        f32,
    roughness:       f32,
    emissive_factor: f32,
    ao:              f32,
    emissive_color:  vec3<f32>,
    alpha_cutoff:    f32,
    workflow:        u32,
    workflow_flags:  u32,
    _pad0:           vec2<u32>,
    specular_color:  vec3<f32>,
    specular_weight: f32,
    ior:             f32,
    dielectric_f0:   f32,
    _reserved:       vec2<f32>,
}

/// Per-instance GPU data.  Must match `GpuInstanceData` in gpu_scene.rs.
struct GpuInstanceData {
    transform:     mat4x4<f32>,
    normal_mat_0:  vec4<f32>,
    normal_mat_1:  vec4<f32>,
    normal_mat_2:  vec4<f32>,
    bounds_center: vec3<f32>,
    bounds_radius: f32,
}

@group(0) @binding(0) var<storage, read> light_matrices:   array<LightMatrix>;
@group(0) @binding(1) var<uniform>       shadow_layer_idx: u32;
@group(0) @binding(2) var<storage, read> instance_data:    array<GpuInstanceData>;

@group(1) @binding(0) var<uniform> material:           Material;
@group(1) @binding(1) var          base_color_texture: texture_2d<f32>;
@group(1) @binding(3) var          material_sampler:   sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       tex_coords:    vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position:   vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @builtin(instance_index) slot: u32,
) -> VertexOutput {
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(position, 1.0);
    var out: VertexOutput;
    out.clip_position = light_matrices[shadow_layer_idx].mat * world_pos;
    out.tex_coords = tex_coords;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) {
    let tex_sample = textureSample(base_color_texture, material_sampler, input.tex_coords);
    let alpha = material.base_color.a * tex_sample.a;
    if alpha <= 0.001 { discard; }
    if alpha < material.alpha_cutoff { discard; }
}
