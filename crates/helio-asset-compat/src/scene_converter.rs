//! SolidRS Scene → Helio GPU structures conversion orchestrator
//!
//! This module coordinates the full conversion pipeline from a CPU-side SolidRS
//! scene to GPU-resident Helio structures.

use crate::{Result, AssetError, mesh_converter, material_converter, light_converter, camera_converter, SceneAsset, CameraData};
use helio_render_v2::mesh::PackedVertex;
use helio_render_v2::material::Material;
use helio_render_v2::scene::SceneLight;
use solid_rs::Scene;

/// Converted scene data ready for GPU upload
///
/// This holds all converted CPU-side data from a SolidRS scene.
/// The actual GPU upload and object registration happens separately.
pub struct ConvertedScene {
    /// Scene name
    pub name: String,
    /// Converted meshes (vertices + indices)
    pub meshes: Vec<ConvertedMesh>,
    /// Converted PBR materials
    pub materials: Vec<Material>,
    /// Converted lights
    pub lights: Vec<SceneLight>,
    /// Camera data (informational only - not auto-applied)
    pub cameras: Vec<CameraData>,
    // TODO: Animations (Phase 5)
    // TODO: Skins (Phase 5)
}

/// A single converted mesh with its primitives
pub struct ConvertedMesh {
    /// Mesh name from SolidRS
    pub name: String,
    /// Converted vertices
    pub vertices: Vec<PackedVertex>,
    /// Indices for all primitives
    pub indices: Vec<u32>,
    /// Material index (into ConvertedScene::materials)
    /// Note: For now we merge all primitives, in Phase 3 we'll separate by material
    pub material_index: Option<usize>,
}

/// Convert a SolidRS scene to Helio-compatible structures
pub fn convert_scene(scene: &Scene, base_dir: &std::path::Path) -> Result<ConvertedScene> {
    log::info!("Converting SolidRS scene '{}' with {} meshes, {} materials",
        scene.name, scene.meshes.len(), scene.materials.len());

    // Convert all materials first
    let materials: Result<Vec<Material>> = scene.materials.iter()
        .map(|mat| material_converter::convert_material(mat, scene, base_dir))
        .collect();
    let materials = materials?;

    log::debug!("Converted {} materials", materials.len());

    // Convert all meshes
    let meshes: Result<Vec<ConvertedMesh>> = scene.meshes.iter()
        .enumerate()
        .map(|(idx, mesh)| {
            let (vertices, indices) = mesh_converter::convert_mesh(mesh)?;

            // Find the material index for this mesh
            // Note: SolidRS meshes can have multiple primitives with different materials
            // For now, we take the first primitive's material
            let material_index = mesh.primitives.first()
                .and_then(|p| p.material_index);

            Ok(ConvertedMesh {
                name: if mesh.name.is_empty() {
                    format!("Mesh_{}", idx)
                } else {
                    mesh.name.clone()
                },
                vertices,
                indices,
                material_index,
            })
        })
        .collect();
    let meshes = meshes?;

    log::debug!("Converted {} meshes ({} total vertices)",
        meshes.len(),
        meshes.iter().map(|m| m.vertices.len()).sum::<usize>());

    // Convert all lights
    let lights: Vec<SceneLight> = scene.lights.iter()
        .filter_map(|light| light_converter::convert_light(light))
        .collect();

    log::debug!("Converted {} lights", lights.len());

    // Extract camera data
    let cameras: Vec<CameraData> = scene.cameras.iter()
        .map(|camera| camera_converter::extract_camera_data(camera))
        .collect();

    log::debug!("Extracted {} cameras", cameras.len());

    Ok(ConvertedScene {
        name: scene.name.clone(),
        meshes,
        materials,
        lights,
        cameras,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_empty_scene() {
        let scene = Scene {
            name: "EmptyScene".to_string(),
            ..Default::default()
        };

        let converted = convert_scene(&scene, std::path::Path::new(".")).unwrap();
        assert_eq!(converted.name, "EmptyScene");
        assert_eq!(converted.meshes.len(), 0);
        assert_eq!(converted.materials.len(), 0);
    }
}
