//! PBR material mapping from SolidRS to Helio GPU material data.

use crate::Result;
use helio::GpuMaterial;
use libhelio::MaterialWorkflow;
use solid_rs::scene::{AlphaMode, Material as SolidMaterial, Scene};

const MATERIAL_WORKFLOW_EPSILON: f32 = 1.0e-6;

fn uses_explicit_specular_ior_workflow(material: &SolidMaterial) -> bool {
    let specular_color_is_default =
        (material.specular_color.x - 1.0).abs() <= MATERIAL_WORKFLOW_EPSILON
            && (material.specular_color.y - 1.0).abs() <= MATERIAL_WORKFLOW_EPSILON
            && (material.specular_color.z - 1.0).abs() <= MATERIAL_WORKFLOW_EPSILON;
    let specular_weight_is_default =
        (material.specular_weight - 1.0).abs() <= MATERIAL_WORKFLOW_EPSILON;
    let ior_is_default = (material.ior - 1.5).abs() <= MATERIAL_WORKFLOW_EPSILON;

    !specular_color_is_default
        || material.specular_color_texture.is_some()
        || !specular_weight_is_default
        || material.specular_weight_texture.is_some()
        || !ior_is_default
}

/// Convert a SolidRS material to Helio's GPU material layout.
///
/// The current `helio` desktop wrapper uses constant factors only, so texture slots
/// are left unbound for now. This keeps the asset bridge compatible with the v3
/// examples while the texture system is reintroduced on top of the new facade.
pub fn convert_material(
    material: &SolidMaterial,
    scene: &Scene,
    base_dir: &std::path::Path,
) -> Result<GpuMaterial> {
    let _ = (scene, base_dir);

    let mut workflow = MaterialWorkflow::Metallic as u32;
    let mut metallic = material.metallic_factor;
    let mut roughness = material.roughness_factor;
    let mut ior = 1.5;
    let mut specular = 0.5;

    if uses_explicit_specular_ior_workflow(material) {
        workflow = MaterialWorkflow::Specular as u32;
        metallic = material.specular_weight;
        roughness = material.roughness_factor;
        ior = material.ior;
        specular = material.specular_weight;
    }

    let mut flags = 0u32;
    match material.alpha_mode {
        AlphaMode::Opaque => {}
        AlphaMode::Mask => {
            flags |= 1 << 2;
        }
        AlphaMode::Blend => {
            flags |= 1 << 1;
        }
    }
    if material.double_sided {
        flags |= 1;
    }

    Ok(GpuMaterial {
        base_color: [
            material.base_color_factor.x,
            material.base_color_factor.y,
            material.base_color_factor.z,
            material.base_color_factor.w,
        ],
        emissive: [
            material.emissive_factor.x,
            material.emissive_factor.y,
            material.emissive_factor.z,
            1.0,
        ],
        roughness_metallic: [roughness, metallic, ior, specular],
        tex_base_color: GpuMaterial::NO_TEXTURE,
        tex_normal: GpuMaterial::NO_TEXTURE,
        tex_roughness: GpuMaterial::NO_TEXTURE,
        tex_emissive: GpuMaterial::NO_TEXTURE,
        tex_occlusion: GpuMaterial::NO_TEXTURE,
        workflow,
        flags,
        _pad: 0,
    })
}
