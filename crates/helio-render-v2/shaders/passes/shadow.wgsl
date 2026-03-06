//! Shadow pass shader with alpha testing support
//!
//! Renders all shadow casters into a single shadow atlas layer.
//! The light index is carried via @builtin(instance_index), which the CPU
//! sets by issuing one instance per light (draw_indexed(…, light_idx..light_idx+1)).
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

@group(0) @binding(0) var<storage, read> light_matrices: array<LightMatrix>;

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(3) var material_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position:   vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @builtin(instance_index) light_idx: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = light_matrices[light_idx].mat * vec4<f32>(position, 1.0);
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
