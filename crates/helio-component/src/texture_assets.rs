//! Image files as SceneDB assets (SceneDB#84): what a sprite's `texture` and
//! a decal's `albedo_texture` name.
//!
//! A component names its image by path and nothing loads it on the caller's
//! side. Its handle-ledger collector turns the path into an [`AssetId`] of
//! [`TEXTURE_ASSET_KIND`], so the world counts the reference however the
//! value got there (a typed insert, a decoded level, a reflected write, a
//! script), loads the file on a worker thread and binds the decoded image
//! into the component's [`BoundTexture`]. The GPU mirror registers that
//! image in the scene's texture store when it derives the component's row;
//! it never reads the file.
//!
//! Reload is the world's: [`texture_source_for_file`] maps a changed file to
//! the source its asset was declared under, for
//! `World::asset_source_changed`, which rebinds every component showing it.

use pulsar_scenedb::{AssetId, AssetPayload, AssetView};
use std::sync::Arc;

/// The SceneDB asset kind of image files. A texture asset's source is the
/// file's absolute path, normalized lexically ([`texture_source`]).
pub const TEXTURE_ASSET_KIND: &str = "texture";

/// A decoded image: what every component showing the file is bound to.
pub struct LoadedTexture {
    pub width: u32,
    pub height: u32,
    /// Straight-alpha sRGB RGBA8 texels, row by row from the top left.
    pub rgba: Vec<u8>,
}

/// The image a component is bound to, as the world loaded it: the asset it
/// came from and that asset's content version, so the GPU copy is replaced
/// when the file is reloaded. Empty until the world has loaded the image, or
/// when it failed to load. Never authored or saved.
#[derive(Clone, Default)]
pub struct BoundTexture(Option<BoundImage>);

#[derive(Clone)]
struct BoundImage {
    id: AssetId,
    version: u64,
    image: Arc<LoadedTexture>,
}

impl BoundTexture {
    /// The bound image, if it is the asset `expected` (a component whose
    /// path changed shows nothing until the new image is bound).
    pub fn image_for(&self, expected: AssetId) -> Option<&Arc<LoadedTexture>> {
        self.0
            .as_ref()
            .filter(|bound| !expected.is_zero() && bound.id == expected)
            .map(|bound| &bound.image)
    }
}

impl std::fmt::Debug for BoundTexture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(bound) => write!(
                f,
                "BoundTexture({:032x} v{}, {}x{})",
                bound.id.0, bound.version, bound.image.width, bound.image.height
            ),
            None => f.write_str("BoundTexture(none)"),
        }
    }
}

/// Registers the texture asset kind with SceneDB: its loader and payload
/// size, and how sprites and decals take in a loaded image. Idempotent; done
/// on the first [`texture_asset_id`].
pub fn register_texture_assets() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        pulsar_scenedb::register_loader(TEXTURE_ASSET_KIND, load_texture_asset);
        pulsar_scenedb::register_payload_size(TEXTURE_ASSET_KIND, |payload| {
            payload
                .downcast_ref::<LoadedTexture>()
                .map_or(0, |image| image.rgba.len())
        });
        crate::components::register_texture_bindings();
    });
}

/// The source string of the image at `path` (project-relative, or
/// absolute): the absolute path, with `.` and `..` resolved lexically. It
/// does not touch the file system, so the same path always gives the same
/// source whether or not the file exists. `None` for an empty path.
pub fn texture_source(path: &str) -> Option<String> {
    let path = path.trim();
    if path.is_empty() {
        return None;
    }
    let mut full = std::path::PathBuf::from(path);
    if full.is_relative() {
        if let Some(root) = engine_state::get_project_path() {
            full = std::path::Path::new(&root).join(full);
        }
    }
    Some(texture_source_for_file(&full))
}

/// The source string of the image file at `abs` (see [`texture_source`]).
pub fn texture_source_for_file(abs: &std::path::Path) -> String {
    use std::path::Component;
    let mut normal = std::path::PathBuf::new();
    for component in abs.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normal.pop() {
                    normal.push(component);
                }
            }
            other => normal.push(other),
        }
    }
    normal.to_string_lossy().into_owned()
}

/// The asset id of the image `path` names, without declaring it.
/// [`AssetId::ZERO`] for an empty path.
pub fn texture_asset_id_of(path: &str) -> AssetId {
    texture_source(path).map_or(AssetId::ZERO, |source| {
        AssetId::of(TEXTURE_ASSET_KIND, &source)
    })
}

/// The asset id of the image `path` names, declared so the world can load
/// it. [`AssetId::ZERO`] for an empty path.
pub fn texture_asset_id(path: &str) -> AssetId {
    let Some(source) = texture_source(path) else {
        return AssetId::ZERO;
    };
    register_texture_assets();
    let id = AssetId::of(TEXTURE_ASSET_KIND, &source);
    // Collectors run on every write: take the sources' write lock only for
    // a source no one declared yet.
    if id.source().is_none() {
        AssetId::declare(TEXTURE_ASSET_KIND, &source);
    }
    id
}

/// The texture kind's loader: runs on a SceneDB worker thread.
fn load_texture_asset(source: &str) -> Result<AssetPayload, String> {
    Ok(Arc::new(decode_image_file(std::path::Path::new(source))?))
}

/// Read and decode the image file at `path`.
pub(crate) fn decode_image_file(path: &std::path::Path) -> Result<LoadedTexture, String> {
    let shown = path.display();
    let bytes =
        std::fs::read(path).map_err(|error| format!("could not read '{shown}': {error}"))?;
    let decoded = image::load_from_memory(&bytes)
        .map_err(|error| format!("could not decode '{shown}': {error}"))?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    if width == 0 || height == 0 {
        return Err(format!("'{shown}' has empty dimensions"));
    }
    Ok(LoadedTexture {
        width,
        height,
        rgba: decoded.into_raw(),
    })
}

/// Bind the asset `view` into `bound` when it is the image `path` names: the
/// decoded image when it loaded, nothing when it failed (`what` names the
/// component in the warning).
pub(crate) fn bind_texture(bound: &mut BoundTexture, path: &str, view: &AssetView, what: &str) {
    if view.id.is_zero() || view.id != texture_asset_id_of(path) {
        return;
    }
    match view.payload::<LoadedTexture>() {
        Some(image) => {
            bound.0 = Some(BoundImage {
                id: view.id,
                version: view.version,
                image,
            });
        }
        None => {
            tracing::warn!("{what} texture '{path}' is unavailable ({:?})", view.state);
            bound.0 = None;
        }
    }
}

/// The handle-ledger reference of the image `path` names: exactly one entry,
/// [`AssetId::ZERO`] for none (the ledger diffs a value's handles by
/// position, as it does a derived collector's fields).
pub(crate) fn collect_texture(path: &str, out: &mut Vec<pulsar_scenedb::handle_ledger::HandleId>) {
    out.push(texture_asset_id(path).handle());
}

/// The scene texture store slot of `bound`'s image when it is the one
/// `path` names, uploading it on first use and again after a reload;
/// `u32::MAX` for none (the tint alone).
pub(crate) fn texture_slot(
    bound: &BoundTexture,
    path: &str,
    mirror: &pulsar_scenedb::gpu::GpuMirrorHandle,
    what: &str,
) -> u32 {
    let Some(image) = bound.image_for(texture_asset_id_of(path)) else {
        return u32::MAX;
    };
    let current = bound.0.as_ref().expect("image_for found it");
    match resident_slot(current.id, current.version, image, mirror) {
        Ok(slot) => slot,
        Err(error) => {
            tracing::warn!("{what} texture '{path}': {error}");
            u32::MAX
        }
    }
}

struct ResidentTexture {
    owner: std::sync::Weak<std::sync::RwLock<pulsar_scenedb::gpu::TextureStore>>,
    version: u64,
    slot: u32,
}

/// Each texture asset's slot in each scene texture store (keyed by the
/// store's address), with the content version it holds.
type Residency = std::collections::HashMap<(usize, AssetId), ResidentTexture>;

fn resident() -> &'static std::sync::Mutex<Residency> {
    static RESIDENT: std::sync::OnceLock<std::sync::Mutex<Residency>> = std::sync::OnceLock::new();
    RESIDENT.get_or_init(Default::default)
}

/// The slot `image` (content `version` of the texture `id`) holds in
/// `mirror`'s texture store, uploading it when the store does not hold that
/// version yet.
pub(crate) fn resident_slot(
    id: AssetId,
    version: u64,
    image: &LoadedTexture,
    mirror: &pulsar_scenedb::gpu::GpuMirrorHandle,
) -> Result<u32, String> {
    let texture_store = mirror
        .texture_store()
        .ok_or_else(|| "SceneDB has no texture store".to_string())?;
    let key = (Arc::as_ptr(&texture_store) as usize, id);
    let mut resident = resident()
        .lock()
        .map_err(|_| "texture residency lock poisoned".to_string())?;
    resident.retain(|_, entry| entry.owner.strong_count() != 0);
    if let Some(entry) = resident.get(&key) {
        if entry.version == version {
            return Ok(entry.slot);
        }
    }
    let mut store = texture_store
        .write()
        .map_err(|_| "SceneDB texture store lock poisoned".to_string())?;
    if let Some(old) = resident.remove(&key) {
        let _ = store.unregister(old.slot);
    }
    let slot = crate::material_textures::upload_rgba(
        &mut store,
        mirror,
        image.width,
        image.height,
        &image.rgba,
    )?;
    resident.insert(
        key,
        ResidentTexture {
            owner: Arc::downgrade(&texture_store),
            version,
            slot,
        },
    );
    Ok(slot)
}
