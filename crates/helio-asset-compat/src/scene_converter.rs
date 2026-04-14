//! SolidRS Scene → Helio GPU structures conversion orchestrator.

use std::collections::HashMap;

use helio::{MaterialAsset, MaterialTextureRef, MaterialTextures, PackedVertex, TextureUpload};
use solid_rs::Scene;

use crate::material_converter::{convert_material, ConvertedMaterial, ConvertedTextureRef};
use crate::texture_loader::{load_texture_upload, TextureSemantic};
use crate::{camera_converter, light_converter, mesh_converter, CameraData, Result};

/// Build a high-res checkerboard fallback matching Unreal Engine's style:
///   • 512×512, 8×8 grid of 64px tiles, 1px hair-thin grout
///   • 4px chamfered bevel at tile edges — normal map encodes the actual depth
///   • Light tiles ~195 (metallic silver), ±3 noise
///   • Dark tiles ~55 (graphite), ±15 fine grain noise
///   • Normal semantic gets a full 512×512 normal map matching the tile geometry
fn fallback_texture(semantic: TextureSemantic) -> helio::TextureUpload {
    let srgb = semantic.is_srgb();

    // Flat neutral for channels that don't need a visual pattern.
    match semantic {
        TextureSemantic::MetallicRoughness | TextureSemantic::Occlusion
        | TextureSemantic::SpecularWeight => {
            return helio::TextureUpload::rgba8(
                format!("fallback-{}", semantic.suffix()),
                1, 1, false,
                vec![255, 255, 255, 255],
                helio::TextureSamplerDesc::default(),
            );
        }
        _ => {}
    }

    const SIZE: u32 = 512;
    const TILE: u32 = 64;  // 8×8 grid
    const GROUT: u32 = 1;  // 1px hairline grout
    const BEVEL: u32 = 4;  // chamfer width (pixels), used for normal map tilt

    let is_normal = semantic == TextureSemantic::Normal;

    // Deterministic position-keyed hash (no sequential state).
    let hash = |x: u32, y: u32| -> u8 {
        let mut h = x.wrapping_mul(2246822519_u32).wrapping_add(y.wrapping_mul(3266489917_u32));
        h ^= h >> 13;
        h = h.wrapping_mul(1274126177_u32);
        h ^= h >> 16;
        (h & 0xFF) as u8
    };

    let mut data = Vec::with_capacity((SIZE * SIZE * 4) as usize);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let tx = x % TILE;
            let ty = y % TILE;
            let checker = ((x / TILE) + (y / TILE)) % 2 == 0;

            let edge_x = tx.min(TILE - 1 - tx);
            let edge_y = ty.min(TILE - 1 - ty);
            let edge = edge_x.min(edge_y);

            if is_normal {
                // ── Normal map ────────────────────────────────────────────────
                // Grout is the recessed groove; bevel region tilts the normal
                // outward (toward tile center) creating a chamfered raised-tile look.
                let pixel: [u8; 4] = if edge < GROUT {
                    // Grout groove — shallow downward tilt.
                    [128, 128, 200, 255]
                } else if edge < GROUT + BEVEL {
                    let t = (edge - GROUT) as f32 / BEVEL as f32; // 0=grout edge, 1=flat
                    let tilt = (1.0_f32 - t) * 0.65;
                    // Tilt toward the closer axis edge.
                    let (nx, ny) = if edge_x <= edge_y {
                        let sign = if tx < TILE / 2 { 1.0_f32 } else { -1.0 };
                        (sign * tilt, 0.0_f32)
                    } else {
                        let sign = if ty < TILE / 2 { 1.0_f32 } else { -1.0 };
                        (0.0_f32, sign * tilt)
                    };
                    let nz = (1.0 - nx * nx - ny * ny).max(0.001).sqrt();
                    [
                        (128.0 + nx * 127.0) as u8,
                        (128.0 + ny * 127.0) as u8,
                        (nz * 255.0).clamp(0.0, 255.0) as u8,
                        255,
                    ]
                } else {
                    [128, 128, 255, 255] // flat tile centre
                };
                data.extend_from_slice(&pixel);
            } else {
                // ── Base colour / emissive / specular colour ──────────────────
                let v: u8 = if edge < GROUT {
                    10 // very thin dark hairline grout
                } else if checker {
                    // Light tile — metallic silver, near-clean.
                    let n = (hash(x, y) & 0x07) as i16 - 3;
                    (195_i16 + n).clamp(0, 255) as u8
                } else {
                    // Dark tile — graphite with fine grain noise.
                    let n = (hash(x ^ 0xA5A5, y ^ 0x5A5A) & 0x1F) as i16 - 15;
                    (55_i16 + n).clamp(0, 255) as u8
                };
                data.extend_from_slice(&[v, v, v, 255]);
            }
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

