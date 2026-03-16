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

use std::collections::HashMap;
use helio_render_v2::scene::{ObjectId, LightId};

pub use mesh_converter::{convert_vertex, convert_mesh};
pub use material_converter::convert_material;
pub use light_converter::convert_light;
pub use camera_converter::{extract_camera_data, CameraData};
pub use scene_converter::{convert_scene, ConvertedScene, ConvertedMesh};

use std::path::Path;

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
    let path = path.as_ref();

    // Detect format from extension
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .ok_or_else(|| AssetError::UnsupportedFormat(
            "File has no extension".to_string()
        ))?;

    log::info!("Loading 3D model: {}", path.display());

    // Create SolidRS registry and register loaders
    let mut registry = solid_rs::registry::Registry::new();
    registry.register_loader(solid_fbx::FbxLoader);
    registry.register_loader(solid_gltf::GltfLoader);
    registry.register_loader(solid_obj::ObjLoader);

    // Load the scene
    let solid_scene = registry.load_file(path)
        .map_err(|e| AssetError::Solid(e))?;

    log::info!("Loaded SolidRS scene '{}' - {} meshes, {} materials, {} lights",
        solid_scene.name,
        solid_scene.meshes.len(),
        solid_scene.materials.len(),
        solid_scene.lights.len());

    // Convert to Helio structures
    convert_scene(&solid_scene)
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

    #[error("Helio renderer error: {0}")]
    Renderer(#[from] helio_render_v2::Error),
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
