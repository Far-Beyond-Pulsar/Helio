//! 3D asset loading integration with SolidRS
//!
//! This crate provides a bridge between SolidRS (comprehensive 3D model loader)
//! and Helio's GPU-driven rendering pipeline. It handles conversion of CPU-side
//! scene data to GPU buffers while maintaining performance standards.

mod animation_system;
mod camera_converter;
mod light_converter;
mod material_converter;
mod mesh_converter;
mod scene_converter;
mod texture_loader;

use helio::MeshId;
use helio::{LightId, MaterialId, MultiMeshId, ObjectId, Renderer, SectionedMeshUpload, TextureId};
use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;

pub use camera_converter::{extract_camera_data, CameraData};
pub use light_converter::convert_light;
pub use material_converter::{
    convert_material, ConvertedMaterial, ConvertedMaterialTextures, ConvertedTextureRef,
};
pub use mesh_converter::{convert_primitive, convert_vertex};
pub use scene_converter::{
    convert_scene, ConvertedMesh, ConvertedMeshSection, ConvertedScene, ConvertedSectionedMesh,
};

use std::path::Path;

/// Configuration for asset loading
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Flip UV Y-axis (1.0 - v)
    /// - true: DirectX convention (0,0 at top-left) → OpenGL (0,0 at bottom-left)
    /// - false: Use UVs as-is
    pub flip_uv_y: bool,
    /// Merge all sub-meshes into a single mesh with vertex positions baked into
    /// world space.  Useful when you want to treat the whole asset as one draw
    /// call or one physics body.  The merged mesh gets `node_transform` = IDENTITY.
    pub merge_meshes: bool,
    /// Scale applied to the entire imported asset.  Applied before any other
    /// transform so it acts as a unit-conversion factor (e.g. `Vec3::splat(0.01)`
    /// to convert centimetres → metres).  Defaults to `Vec3::ONE` (no change).
    pub import_scale: glam::Vec3,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            flip_uv_y: false,
            merge_meshes: false,
            import_scale: glam::Vec3::ONE,
        }
    }
}

impl LoadConfig {
    pub fn with_uv_flip(mut self, flip: bool) -> Self {
        self.flip_uv_y = flip;
        self
    }

    pub fn with_merge_meshes(mut self, merge: bool) -> Self {
        self.merge_meshes = merge;
        self
    }

    pub fn with_import_scale(mut self, scale: glam::Vec3) -> Self {
        self.import_scale = scale;
        self
    }
}

/// Load a 3D scene file (FBX, glTF, OBJ, etc.) and convert to Helio structures
///
/// This is the main entry point for loading 3D assets. It:
/// 1. Detects the file format from the extension
/// 2. Loads the file using the appropriate SolidRS loader
/// 3. Converts the scene to Helio-compatible structures
///
/// # Example
/// ```no_run
/// use helio_asset_compat::load_scene_file;
///
/// let scene = load_scene_file("models/character.fbx").unwrap();
/// println!("Loaded {} meshes, {} materials", scene.meshes.len(), scene.materials.len());
/// ```
pub fn load_scene_file<P: AsRef<Path>>(path: P) -> Result<ConvertedScene> {
    load_scene_file_with_config(path, LoadConfig::default())
}

/// Load with custom configuration (e.g., UV flipping)
pub fn load_scene_file_with_config<P: AsRef<Path>>(
    path: P,
    config: LoadConfig,
) -> Result<ConvertedScene> {
    let path = path.as_ref();

    // Detect format from extension
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| AssetError::UnsupportedFormat("File has no extension".to_string()))?;

    log::info!(
        "Loading 3D model: {} (UV flip: {})",
        path.display(),
        config.flip_uv_y
    );
    log::info!("Detected extension: {}", extension);

    // Create SolidRS registry and register loaders
    let mut registry = solid_rs::registry::Registry::new();
    registry.register_loader(solid_fbx::FbxLoader);
    registry.register_loader(solid_gltf::GltfLoader);
    registry.register_loader(solid_obj::ObjLoader);
    registry.register_loader(solid_usd::UsdLoader); // supports usda/usdc/usdz

    // Load the scene
    let solid_scene = registry.load_file(path).map_err(|e| AssetError::Solid(e))?;

    log::info!(
        "Loaded SolidRS scene '{}' - {} meshes, {} materials, {} lights",
        solid_scene.name,
        solid_scene.meshes.len(),
        solid_scene.materials.len(),
        solid_scene.lights.len()
    );

    // Get the directory containing the model file for resolving relative texture paths
    let base_dir = path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    // Convert to Helio structures
    convert_scene(&solid_scene, &base_dir, &config)
}

/// Load a 3D scene from embedded bytes using a known format identifier.
///
/// This is useful for examples or applications that bundle assets with
/// `include_bytes!` but still want Helio's normal scene conversion pipeline.
pub fn load_scene_bytes(
    bytes: &[u8],
    format_id: &str,
    base_dir: Option<&Path>,
) -> Result<ConvertedScene> {
    load_scene_bytes_with_config(bytes, format_id, base_dir, LoadConfig::default())
}

/// Load embedded scene bytes with custom configuration (e.g., UV flipping).
pub fn load_scene_bytes_with_config(
    bytes: &[u8],
    format_id: &str,
    base_dir: Option<&Path>,
    config: LoadConfig,
) -> Result<ConvertedScene> {
    log::info!(
        "Loading embedded 3D model as '{}' (UV flip: {})",
        format_id,
        config.flip_uv_y
    );

    let mut registry = solid_rs::registry::Registry::new();
    registry.register_loader(solid_fbx::FbxLoader);
    registry.register_loader(solid_gltf::GltfLoader);
    registry.register_loader(solid_obj::ObjLoader);
    registry.register_loader(solid_usd::UsdLoader);

    let mut options = solid_rs::traits::LoadOptions::default();
    options.base_dir = base_dir.map(Path::to_path_buf);

    let solid_scene = registry
        .load_from(Cursor::new(bytes), format_id, &options)
        .map_err(AssetError::Solid)?;

    log::info!(
        "Loaded embedded SolidRS scene '{}' - {} meshes, {} materials, {} lights",
        solid_scene.name,
        solid_scene.meshes.len(),
        solid_scene.materials.len(),
        solid_scene.lights.len()
    );

    let conversion_base_dir = base_dir
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));

    convert_scene(&solid_scene, &conversion_base_dir, &config)
}

/// GPU handles for a scene that has been fully uploaded to the renderer.
///
/// `mesh_ids[i]` corresponds to `ConvertedScene::meshes[i]`.
/// `material_ids[i]` corresponds to `ConvertedScene::materials[i]`.
/// Use `mesh_material(mesh_index)` to look up the material for a given mesh.
#[derive(Debug, Clone)]
pub struct UploadedScene {
    pub mesh_ids: Vec<MeshId>,
    pub material_ids: Vec<MaterialId>,
}

impl UploadedScene {
    /// Convenience: return the `MaterialId` that should be used for mesh at `mesh_index`.
    ///
    /// Falls back to `material_ids[0]` when the mesh has no material index, and
    /// returns `None` when `material_ids` is empty.
    pub fn mesh_material(
        &self,
        converted: &scene_converter::ConvertedMesh,
    ) -> Option<MaterialId> {
        let idx = converted.material_index?;
        self.material_ids
            .get(idx)
            .copied()
            .or_else(|| self.material_ids.first().copied())
    }
}

/// Load a scene file **and** upload all its meshes + materials in a single pass.
///
/// This is a convenience wrapper around [`load_scene_file_with_config`] +
/// [`upload_scene`] that avoids loading the file twice when you need both.
pub fn load_and_upload_scene<P: AsRef<Path>>(
    path: P,
    config: LoadConfig,
    renderer: &mut Renderer,
) -> Result<UploadedScene> {
    let scene = load_scene_file_with_config(path, config)?;
    upload_scene(renderer, &scene)
}

/// Upload a already-converted scene (meshes **and** materials) to the renderer
/// in a single pass, returning stable GPU handles for both.
///
/// Prefer this over calling `upload_scene_materials` + a manual mesh loop so
/// the `ConvertedScene` is only traversed once.
pub fn upload_scene(renderer: &mut Renderer, scene: &ConvertedScene) -> Result<UploadedScene> {
    let material_ids = upload_scene_materials(renderer, scene)?;
    let mesh_ids = scene
        .meshes
        .iter()
        .filter_map(|mesh| {
            let actor_id = renderer.scene_mut().insert_actor(helio::SceneActor::mesh(helio::MeshUpload {
                vertices: mesh.vertices.clone(),
                indices: mesh.indices.clone(),
            }));
            match actor_id {
                helio::SceneActorId::Mesh(id) => Some(id),
                _ => None,
            }
        })
        .collect::<Vec<_>>() ;
    Ok(UploadedScene {
        mesh_ids,
        material_ids,
    })
}

pub fn upload_scene_materials(
    renderer: &mut Renderer,
    scene: &ConvertedScene,
) -> Result<Vec<MaterialId>> {
    let texture_ids: Result<Vec<TextureId>> = scene
        .textures
        .iter()
        .cloned()
        .map(|texture| {
            renderer
                .scene_mut()
                .insert_texture(texture)
                .map_err(|err: helio::SceneError| AssetError::InvalidData(err.to_string()))
        })
        .collect();
    let texture_ids = texture_ids?;

    scene
        .materials
        .iter()
        .map(|material| {
            let asset = scene_converter::material_asset_from_converted(material, &texture_ids);
            renderer
                .scene_mut()
                .insert_material_asset(asset)
                .map_err(|err: helio::SceneError| AssetError::InvalidData(err.to_string()))
        })
        .collect()
}

/// Upload the materials/textures and sectioned mesh from a [`ConvertedScene`] that
/// was loaded with [`LoadConfig::merge_meshes`] = true.
///
/// Returns the [`MultiMeshId`] asset handle and the list of [`MaterialId`]s (one per
/// section, in the same order as [`ConvertedSectionedMesh::sections`]).
///
/// Panics if `scene.sectioned_mesh` is `None` (i.e. the scene was not loaded with
/// `merge_meshes = true`).
pub fn upload_sectioned_scene(
    renderer: &mut Renderer,
    scene: &ConvertedScene,
) -> Result<(MultiMeshId, Vec<MaterialId>)> {
    let sm = scene
        .sectioned_mesh
        .as_ref()
        .expect("upload_sectioned_scene called on a scene without sectioned_mesh; use merge_meshes=true");

    // Upload textures and materials (same as upload_scene_materials).
    let texture_ids: Result<Vec<TextureId>> = scene
        .textures
        .iter()
        .cloned()
        .map(|t| {
            renderer
                .scene_mut()
                .insert_texture(t)
                .map_err(|e: helio::SceneError| AssetError::InvalidData(e.to_string()))
        })
        .collect();
    let texture_ids = texture_ids?;

    let all_material_ids: Result<Vec<MaterialId>> = scene
        .materials
        .iter()
        .map(|mat| {
            let asset = scene_converter::material_asset_from_converted(mat, &texture_ids);
            renderer
                .scene_mut()
                .insert_material_asset(asset)
                .map_err(|e: helio::SceneError| AssetError::InvalidData(e.to_string()))
        })
        .collect();
    let all_material_ids = all_material_ids?;

    // Build the SectionedMeshUpload: shared vertices + per-section index lists.
    let upload = SectionedMeshUpload {
        vertices: sm.vertices.clone(),
        sections: sm.sections.iter().map(|s| s.indices.clone()).collect(),
    };
    let multi_mesh_id = renderer.scene_mut().insert_sectioned_mesh(upload);

    // Resolve per-section material IDs (fall back to a unit material when None).
    let section_material_ids: Vec<MaterialId> = sm
        .sections
        .iter()
        .map(|sec| {
            sec.material_index
                .and_then(|idx| all_material_ids.get(idx).copied())
                .unwrap_or_else(|| {
                    renderer.scene_mut().insert_material(helio::GpuMaterial {
                        base_color: [0.7, 0.65, 0.55, 1.0],
                        emissive: [0.0, 0.0, 0.0, 0.0],
                        roughness_metallic: [0.6, 0.0, 1.5, 0.0],
                        tex_base_color: helio::GpuMaterial::NO_TEXTURE,
                        tex_normal: helio::GpuMaterial::NO_TEXTURE,
                        tex_roughness: helio::GpuMaterial::NO_TEXTURE,
                        tex_emissive: helio::GpuMaterial::NO_TEXTURE,
                        tex_occlusion: helio::GpuMaterial::NO_TEXTURE,
                        workflow: 0,
                        flags: 0,
                        _pad: 0,
                    })
                })
        })
        .collect();

    Ok((multi_mesh_id, section_material_ids))
}

/// Result type for asset loading operations
pub type Result<T> = std::result::Result<T, AssetError>;

/// Errors that can occur during asset loading
#[derive(Debug, thiserror::Error)]
pub enum AssetError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("SolidRS error: {0}")]
    Solid(#[from] solid_rs::SolidError),

    #[error("Unsupported format: {0}")]
    UnsupportedFormat(String),

    #[error("Invalid data: {0}")]
    InvalidData(String),
}

/// Handle to a loaded 3D scene
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SceneHandle(pub(crate) u64);

/// Identifier for a skeletal skin
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct SkinId(pub(crate) u32);

/// Identifier for an animation instance
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct AnimationId(pub(crate) u32);

/// Metadata about a loaded scene asset
#[derive(Debug, Clone)]
pub struct SceneAsset {
    /// Scene name from the file
    pub name: String,
    /// Mesh objects registered with the renderer
    pub object_ids: Vec<ObjectId>,
    /// Lights registered with the renderer
    pub light_ids: Vec<LightId>,
    /// Skinned mesh controllers
    pub skin_ids: Vec<SkinId>,
    /// Available animation clip names
    pub animation_names: Vec<String>,
}