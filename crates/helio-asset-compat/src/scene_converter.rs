//! SolidRS Scene → Helio GPU structures conversion orchestrator.

use std::collections::HashMap;

use helio::{MaterialAsset, MaterialTextureRef, MaterialTextures, PackedVertex, TextureUpload};
use solid_rs::Scene;

use crate::material_converter::{convert_material, ConvertedMaterial, ConvertedTextureRef};
use crate::texture_loader::{load_texture_upload, TextureSemantic};
use crate::{camera_converter, light_converter, mesh_converter, CameraData, Result};

/// Build a noisy checkerboard fallback texture for a missing image
fn fallback_texture(semantic: TextureSemantic) -> helio::TextureUpload {
    const SIZE: u32 = 64;
    const TILE: u32 = 8; // checker tile size in pixels
    let srgb = semantic.is_srgb();

    // For non-colour channels produce a flat neutral value so shading isn't broken.
    match semantic {
        TextureSemantic::Normal => {
            // Flat upward-pointing normal map — 1×1 is fine.
            return helio::TextureUpload::rgba8(
                "fallback-normal".into(),
                1, 1, false,
                vec![128, 128, 255, 255],
                helio::TextureSamplerDesc::default(),
            );
        }
        TextureSemantic::MetallicRoughness | TextureSemantic::Occlusion
        | TextureSemantic::SpecularWeight => {
            // Fully rough, non-metallic, full occlusion.
            return helio::TextureUpload::rgba8(
                format!("fallback-{}", semantic.suffix()),
                1, 1, false,
                vec![255, 255, 255, 255],
                helio::TextureSamplerDesc::default(),
            );
        }
        _ => {}
    }

    // Tiny deterministic LCG — no external dep needed.
    let mut rng: u32 = 0x9e37_79b9;
    let mut rand_u8 = move || -> u8 {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        (rng >> 24) as u8
    };

    let mut data = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let checker = ((x / TILE) ^ (y / TILE)) & 1 == 0;
            // Dark tile: ~45, Light tile: ~190  (matches UE4 look)
            let base: u8 = if checker { 45 } else { 190 };
            // ±18 noise on top, clamped to [0, 255]
            let noise = (rand_u8() & 0x24) as i16 - 18; // range roughly −18..+18
            let v = (base as i16 + noise).clamp(0, 255) as u8;
            data.extend_from_slice(&[v, v, v, 255]);
        }
    }

    helio::TextureUpload::rgba8(
        format!("fallback-{}", semantic.suffix()),
        SIZE, SIZE, srgb,
        data,
        helio::TextureSamplerDesc::default(),
    )
}

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
                    let upload = match load_texture_upload(
                        scene,
                        texture_ref.texture_index,
                        semantic,
                        base_dir,
                    ) {
                        Ok(u) => u,
                        Err(e) => {
                            log::warn!(
                                "Texture {} ({:?}) not found, using fallback: {}",
                                texture_ref.texture_index,
                                semantic,
                                e
                            );
                            fallback_texture(semantic)
                        }
                    };
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
            roughness_metallic: remap_texture_slot(
                material.textures.roughness_metallic,
                texture_ids,
            ),
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

