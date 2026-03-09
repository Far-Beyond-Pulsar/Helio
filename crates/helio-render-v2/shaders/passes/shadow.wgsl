//! Shadow pass shader with alpha testing support.
//!
//! Renders all shadow casters into the shadow atlas.  The `shadow_layer_idx`
//! uniform (group 0, binding 1) selects which atlas layer (light × face) each
//! draw bundle writes to.  Model transforms are read from the GPU Scene storage
//! buffer (group 2, binding 0) via a 4-byte `primitive_id` instance attribute
//! — replacing the old 64-byte mat4 per-instance stream.
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

// GPU Scene: persistent per-object transforms (group 2, binding 0).
// Shadow pass uses group 0 for light matrices, group 1 for material,
// so GPU Scene lives at group 2 here (no lighting group needed for shadows).
struct GpuPrimitive {
    transform:     mat4x4<f32>,
    inv_trans_c0:  vec4<f32>,
    inv_trans_c1:  vec4<f32>,
    inv_trans_c2:  vec4<f32>,
    bounds_center: vec3<f32>,
    bounds_radius: f32,
    material_id:   u32,
    flags:         u32,
    _pad:          vec2<u32>,
}
@group(2) @binding(0) var<storage, read> gpu_primitives: array<GpuPrimitive>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position:   vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(5) primitive_id: u32,
) -> VertexOutput {
    let world_pos = gpu_primitives[primitive_id].transform * vec4<f32>(position, 1.0);
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
