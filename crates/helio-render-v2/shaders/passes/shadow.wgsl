//! Shadow pass shader with alpha testing support
//!
//! Renders all shadow casters into a single shadow atlas layer.
//! The shadow atlas layer index is delivered as a push constant (var<immediate>),
//! set once per face before encoding the draw bundle.  Geometry instancing uses
//! @location(5-8) instance model matrices so every mesh is transformed to world
//! space before the light-space projection is applied.
//!
//! Supports transparent textures: samples the material's base color alpha
//! and discards fragments where alpha is below the cutoff threshold.

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
}

/// Which shadow atlas layer (light_idx * 6 + face) this draw encodes into.
/// Stored in a tiny per-face uniform buffer so the RenderBundle can bake it
/// without needing push-constant/immediates device support.
@group(0) @binding(1) var<uniform> shadow_layer_idx: u32;

@group(0) @binding(0) var<storage, read> light_matrices: array<LightMatrix>;

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(3) var material_sampler: sampler;

/// Per-instance model matrix, split into four vec4 columns (locations 5-8).
struct Instance {
    @location(5) model_0: vec4<f32>,
    @location(6) model_1: vec4<f32>,
    @location(7) model_2: vec4<f32>,
    @location(8) model_3: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position:   vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    inst: Instance,
) -> VertexOutput {
    let model     = mat4x4<f32>(inst.model_0, inst.model_1, inst.model_2, inst.model_3);
    let world_pos = model * vec4<f32>(position, 1.0);
    var out: VertexOutput;
    out.clip_position = light_matrices[shadow_layer_idx].mat * world_pos;
    out.tex_coords = tex_coords;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) {
    let tex_sample = textureSample(base_color_texture, material_sampler, input.tex_coords);
    let alpha = material.base_color.a * tex_sample.a;
    
    // Discard fully transparent pixels (PNG cutouts)
    if alpha <= 0.001 {
        discard;
    }
    
    // Discard pixels below alpha cutoff (masked materials like foliage)
    if alpha < material.alpha_cutoff {
        discard;
    }
}
