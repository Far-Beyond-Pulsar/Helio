//! 3D asset loading integration with SolidRS
//!
//! This crate provides a bridge between SolidRS (comprehensive 3D model loader)
//! and Helio's GPU-driven rendering pipeline. It handles conversion of CPU-side
//! scene data to GPU buffers while maintaining AAA performance standards.

mod scene_converter;
mod mesh_converter;
mod material_converter;
mod texture_loader;
mod light_converter;
mod camera_converter;
mod animation_system;

use std::io::Cursor;
use std::path::PathBuf;
use std::collections::HashMap;
use helio::{LightId, MaterialId, ObjectId, Renderer, TextureId};
use helio::MeshId;

pub use mesh_converter::{convert_vertex, convert_primitive};
pub use material_converter::{convert_material, ConvertedMaterial, ConvertedMaterialTextures, ConvertedTextureRef};
pub use light_converter::convert_light;
pub use camera_converter::{extract_camera_data, CameraData};
pub use scene_converter::{convert_scene, ConvertedScene, ConvertedMesh};

use std::path::Path;

/// Configuration for asset loading
#[derive(Debug, Clone)]
pub struct LoadConfig {
    /// Flip UV Y-axis (1.0 - v)
    /// - true: DirectX convention (0,0 at top-left) → OpenGL (0,0 at bottom-left)
    /// - false: Use UVs as-is
    pub flip_uv_y: bool,
}

impl Default for LoadConfig {
    fn default() -> Self {
        Self {
            // Default: no flip - most modern exporters use OpenGL convention
            flip_uv_y: false,
        }
    }
}

impl LoadConfig {
    pub fn with_uv_flip(mut self, flip: bool) -> Self {
        self.flip_uv_y = flip;
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
pub fn load_scene_file_with_config<P: AsRef<Path>>(path: P, config: LoadConfig) -> Result<ConvertedScene> {
    let path = path.as_ref();

    // Detect format from extension
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| AssetError::UnsupportedFormat(
            "File has no extension".to_string()
        ))?;

    log::info!("Loading 3D model: {} (UV flip: {})", path.display(), config.flip_uv_y);
    log::info!("Detected extension: {}", extension);

    // Create SolidRS registry and register loaders
    let mut registry = solid_rs::registry::Registry::new();
    registry.register_loader(solid_fbx::FbxLoader);
    registry.register_loader(solid_gltf::GltfLoader);
    registry.register_loader(solid_obj::ObjLoader);
    registry.register_loader(solid_usd::UsdLoader); // supports usda/usdc/usdz

    // Load the scene
    let solid_scene = registry.load_file(path)
        .map_err(|e| AssetError::Solid(e))?;

    log::info!("Loaded SolidRS scene '{}' - {} meshes, {} materials, {} lights",
        solid_scene.name,
        solid_scene.meshes.len(),
        solid_scene.materials.len(),
        solid_scene.lights.len());

    // Get the directory containing the model file for resolving relative texture paths
    let base_dir = path.parent()
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
    pub fn mesh_material(&self, mesh_index: usize, converted: &scene_converter::ConvertedMesh) -> Option<MaterialId> {
        let idx = converted.material_index?;
        self.material_ids.get(idx).copied().or_else(|| self.material_ids.first().copied())
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
pub fn upload_scene(
    renderer: &mut Renderer,
    scene: &ConvertedScene,
) -> Result<UploadedScene> {
    let material_ids = upload_scene_materials(renderer, scene)?;
    let mesh_ids = scene
        .meshes
        .iter()
        .map(|mesh| {
            renderer.insert_mesh(helio::MeshUpload {
                vertices: mesh.vertices.clone(),
                indices:  mesh.indices.clone(),
            })
        })
        .collect();
    Ok(UploadedScene { mesh_ids, material_ids })
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
                .insert_texture(texture)
                .map_err(|err| AssetError::InvalidData(err.to_string()))
        })
        .collect();
    let texture_ids = texture_ids?;

    scene
        .materials
        .iter()
        .map(|material| {
            let asset = scene_converter::material_asset_from_converted(material, &texture_ids);
            renderer
                .insert_material_asset(asset)
                .map_err(|err| AssetError::InvalidData(err.to_string()))
        })
        .collect()
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

/// Internal registry of loaded scenes
pub(crate) struct SceneRegistry {
    assets: HashMap<SceneHandle, SceneAsset>,
    next_handle: u64,
}

impl SceneRegistry {
    pub fn new() -> Self {
        Self {
            assets: HashMap::new(),
            next_handle: 1, // 0 reserved for invalid handle
        }
    }

    pub fn allocate_handle(&mut self) -> SceneHandle {
        let handle = SceneHandle(self.next_handle);
        self.next_handle += 1;
        handle
    }

    pub fn register(&mut self, handle: SceneHandle, asset: SceneAsset) {
        self.assets.insert(handle, asset);
    }

    pub fn get(&self, handle: SceneHandle) -> Option<&SceneAsset> {
        self.assets.get(&handle)
    }

    pub fn remove(&mut self, handle: SceneHandle) -> Option<SceneAsset> {
        self.assets.remove(&handle)
    }
}
