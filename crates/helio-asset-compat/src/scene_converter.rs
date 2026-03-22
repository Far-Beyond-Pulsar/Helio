//! SolidRS Scene → Helio GPU structures conversion orchestrator.

use std::collections::HashMap;

use helio::{MaterialAsset, MaterialTextureRef, MaterialTextures, PackedVertex, TextureUpload};
use solid_rs::Scene;

use crate::material_converter::{
    convert_material, ConvertedMaterial, ConvertedTextureRef,
};
use crate::texture_loader::{load_texture_upload, TextureSemantic};
use crate::{camera_converter, light_converter, mesh_converter, CameraData, Result};

/// Converted scene data ready for GPU upload.
pub struct ConvertedScene {
    pub name: String,
    pub meshes: Vec<ConvertedMesh>,
    pub textures: Vec<TextureUpload>,
    pub materials: Vec<ConvertedMaterial>,
    pub lights: Vec<helio::GpuLight>,
    pub cameras: Vec<CameraData>,
}

pub struct ConvertedMesh {
    pub name: String,
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
    pub material_index: Option<usize>,
}

pub fn convert_scene(
    scene: &Scene,
    base_dir: &std::path::Path,
    config: &crate::LoadConfig,
) -> Result<ConvertedScene> {
    log::info!(
        "Converting SolidRS scene '{}' with {} meshes, {} materials, {} textures",
        scene.name,
        scene.meshes.len(),
        scene.materials.len(),
        scene.textures.len()
    );

    let mut textures = Vec::new();
    let mut texture_cache = HashMap::<(usize, TextureSemantic), usize>::new();
    let materials: Result<Vec<ConvertedMaterial>> = scene
        .materials
        .iter()
        .map(|material| {
            convert_material(material, |texture_ref, semantic| {
                let key = (texture_ref.texture_index, semantic);
                let converted_index = if let Some(&index) = texture_cache.get(&key) {
                    index
                } else {
                    let upload = load_texture_upload(scene, texture_ref.texture_index, semantic, base_dir)?;
                    let index = textures.len();
                    textures.push(upload);
                    texture_cache.insert(key, index);
                    index
                };

                Ok(ConvertedTextureRef {
                    texture_index: converted_index,
                    ..texture_ref
                })
            })
        })
        .collect();
    let materials = materials?;

    let mut meshes = Vec::new();
    for (mesh_idx, mesh) in scene.meshes.iter().enumerate() {
        if mesh.primitives.is_empty() {
            log::warn!("Mesh '{}' has no primitives, skipping", mesh.name);
            continue;
        }

        for (prim_idx, primitive) in mesh.primitives.iter().enumerate() {
            let (vertices, indices) = mesh_converter::convert_primitive(mesh, primitive, config)?;

            let mesh_name = if mesh.name.is_empty() {
                if mesh.primitives.len() > 1 {
                    format!("Mesh_{}_{}", mesh_idx, prim_idx)
                } else {
                    format!("Mesh_{}", mesh_idx)
                }
            } else if mesh.primitives.len() > 1 {
                format!("{}_{}", mesh.name, prim_idx)
            } else {
                mesh.name.clone()
            };

            meshes.push(ConvertedMesh {
                name: mesh_name,
                vertices,
                indices,
                material_index: primitive.material_index,
            });
        }
    }

    let lights = scene
        .lights
        .iter()
        .filter_map(light_converter::convert_light)
        .collect::<Vec<_>>();
    let cameras = scene
        .cameras
        .iter()
        .map(camera_converter::extract_camera_data)
        .collect::<Vec<_>>();

    Ok(ConvertedScene {
        name: scene.name.clone(),
        meshes,
        textures,
        materials,
        lights,
        cameras,
    })
}

fn remap_texture_slot(
    texture: Option<ConvertedTextureRef>,
    texture_ids: &[helio::TextureId],
) -> Option<MaterialTextureRef> {
    texture.map(|texture| MaterialTextureRef {
        texture: texture_ids[texture.texture_index],
        uv_channel: texture.uv_channel,
        transform: texture.transform,
    })
}

pub(crate) fn material_asset_from_converted(
    material: &ConvertedMaterial,
    texture_ids: &[helio::TextureId],
) -> MaterialAsset {
    MaterialAsset {
        gpu: material.gpu,
        textures: MaterialTextures {
            base_color: remap_texture_slot(material.textures.base_color, texture_ids),
            normal: remap_texture_slot(material.textures.normal, texture_ids),
            roughness_metallic: remap_texture_slot(material.textures.roughness_metallic, texture_ids),
            emissive: remap_texture_slot(material.textures.emissive, texture_ids),
            occlusion: remap_texture_slot(material.textures.occlusion, texture_ids),
            specular_color: remap_texture_slot(material.textures.specular_color, texture_ids),
            specular_weight: remap_texture_slot(material.textures.specular_weight, texture_ids),
            normal_scale: material.textures.normal_scale,
            occlusion_strength: material.textures.occlusion_strength,
            alpha_cutoff: material.textures.alpha_cutoff,
        },
    }
}
