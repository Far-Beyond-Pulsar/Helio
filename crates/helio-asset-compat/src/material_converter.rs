//! PBR material mapping from SolidRS to Helio

use crate::{Result, AssetError, texture_loader};
use helio_render_v2::material::{Material, TextureData};
use solid_rs::scene::{Material as SolidMaterial, Scene, AlphaMode};

/// Convert a SolidRS material to Helio's PBR material format
///
/// Note: This function doesn't load GPU textures directly - it extracts TextureData
/// that Helio's material system will upload to the GPU.
pub fn convert_material(
    material: &SolidMaterial,
    scene: &Scene,
) -> Result<Material> {
    let mut helio_mat = Material::new();

    // Base color factor (RGBA)
    helio_mat.base_color = [
        material.base_color_factor.x,
        material.base_color_factor.y,
        material.base_color_factor.z,
        material.base_color_factor.w,
    ];

    // Metallic and roughness factors
    helio_mat.metallic = material.metallic_factor;
    helio_mat.roughness = material.roughness_factor;

    // Occlusion strength (Helio uses ao factor)
    helio_mat.ao = material.occlusion_strength;

    // Emissive
    helio_mat.emissive_color = [
        material.emissive_factor.x,
        material.emissive_factor.y,
        material.emissive_factor.z,
    ];
    helio_mat.emissive_factor = 1.0; // SolidRS bakes intensity into emissive_factor

    // Alpha mode
    match material.alpha_mode {
        AlphaMode::Opaque => {
            helio_mat.alpha_cutoff = 0.0;
            helio_mat.transparent_blend = false;
        }
        AlphaMode::Mask => {
            helio_mat.alpha_cutoff = material.alpha_cutoff;
            helio_mat.transparent_blend = false;
        }
        AlphaMode::Blend => {
            helio_mat.alpha_cutoff = 0.0;
            helio_mat.transparent_blend = true;
        }
    }

    // Load textures
    if let Some(tex_ref) = &material.base_color_texture {
        helio_mat.base_color_texture = Some(load_texture_data(tex_ref, scene, true)?);
    }

    if let Some(tex_ref) = &material.normal_texture {
        helio_mat.normal_map = Some(load_texture_data(tex_ref, scene, false)?);
    }

    // For metallic-roughness texture, we need to pack it into ORM format
    // glTF: metallic_roughness has G=roughness, B=metallic
    // Helio: ORM has R=occlusion, G=roughness, B=metallic
    // TODO: In Phase 3, implement proper texture packing/merging
    if let Some(tex_ref) = &material.metallic_roughness_texture {
        // For now, just load it as-is and log a warning
        log::warn!("Metallic-roughness texture requires packing into ORM format - not yet implemented");
        // helio_mat.orm_texture = Some(load_texture_data(tex_ref, scene, false)?);
    }

    if let Some(tex_ref) = &material.occlusion_texture {
        // TODO: Pack with metallic-roughness into ORM
        log::warn!("Occlusion texture requires packing into ORM format - not yet implemented");
    }

    if let Some(tex_ref) = &material.emissive_texture {
        helio_mat.emissive_texture = Some(load_texture_data(tex_ref, scene, true)?);
    }

    Ok(helio_mat)
}

/// Load texture data from a SolidRS TextureRef
fn load_texture_data(
    tex_ref: &solid_rs::scene::TextureRef,
    scene: &Scene,
    _srgb: bool,
) -> Result<TextureData> {
    // Get the texture
    let texture = scene.textures.get(tex_ref.texture_index)
        .ok_or_else(|| AssetError::InvalidData(format!(
            "Texture index {} out of bounds", tex_ref.texture_index
        )))?;

    // Get the image
    let image = scene.images.get(texture.image_index)
        .ok_or_else(|| AssetError::InvalidData(format!(
            "Image index {} out of bounds", texture.image_index
        )))?;

    // Extract image data
    use solid_rs::scene::ImageSource;
    let data_vec: Vec<u8>;
    let image_data = match &image.source {
        ImageSource::Embedded { data, .. } => data.as_slice(),
        ImageSource::Uri(path) => {
            // Load external image file
            data_vec = std::fs::read(path)
                .map_err(|e| AssetError::InvalidData(format!(
                    "Failed to read image file '{}': {}",
                    path, e
                )))?;
            data_vec.as_slice()
        }
    };

    // Decode image to RGBA8
    let decoded = image::load_from_memory(image_data)
        .map_err(|e| AssetError::InvalidData(format!("Failed to decode image: {}", e)))?;

    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();

    Ok(TextureData::new(rgba.into_raw(), width, height))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Vec3, Vec4};

    #[test]
    fn test_convert_basic_material() {
        let mut scene = Scene::default();
        let solid_mat = SolidMaterial {
            name: "TestMaterial".to_string(),
            base_color_factor: Vec4::new(1.0, 0.5, 0.25, 1.0),
            metallic_factor: 0.8,
            roughness_factor: 0.6,
            occlusion_strength: 1.0,
            emissive_factor: Vec3::ZERO,
            alpha_mode: AlphaMode::Opaque,
            alpha_cutoff: 0.5,
            base_color_texture: None,
            normal_texture: None,
            normal_scale: 1.0,
            metallic_roughness_texture: None,
            occlusion_texture: None,
            emissive_texture: None,
            extensions: Default::default(),
        };

        let helio_mat = convert_material(&solid_mat, &scene).unwrap();

        assert_eq!(helio_mat.base_color, [1.0, 0.5, 0.25, 1.0]);
        assert_eq!(helio_mat.metallic, 0.8);
        assert_eq!(helio_mat.roughness, 0.6);
        assert_eq!(helio_mat.alpha_cutoff, 0.0); // Opaque mode
        assert!(!helio_mat.transparent_blend);
    }
}
