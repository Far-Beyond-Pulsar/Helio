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
///
/// The texels are rebuildable data, not owned data: they sit in SceneDB's
/// budgeted residency cache ([`pulsar_scenedb::Residency`]), so what stays in
/// memory is bounded by the cache budget rather than by how many images
/// components reference. A dropped image is decoded again from its file the
/// next time it is needed (a new scene's texture store, a device rebuild).
pub struct LoadedTexture {
    /// Dimensions after the [`max_texture_size`] cap.
    pub width: u32,
    pub height: u32,
    /// Straight-alpha sRGB RGBA8 texels, row by row from the top left.
    texels: pulsar_scenedb::Residency<Vec<u8>>,
}

impl LoadedTexture {
    /// The texels, decoded again from the file if the cache dropped them.
    /// Hold the result only while using it: the cache can drop its own
    /// reference, but not yours.
    pub fn rgba(&self) -> Result<Arc<Vec<u8>>, String> {
        self.texels.get()
    }

    /// Whether the texels are in memory right now.
    pub fn is_resident(&self) -> bool {
        self.texels.is_resident()
    }
}

/// The largest side, in texels, an image is decoded at; larger images are
/// scaled down (aspect kept) as they load. Project setting
/// `streaming.max_texture_size`; 0 disables the cap.
pub const DEFAULT_MAX_TEXTURE_SIZE: u32 = 2048;

pub fn max_texture_size() -> u32 {
    let configured = engine_state::settings::global_config()
        .get(engine_state::settings::NS_PROJECT, "streaming", "max_texture_size")
        .ok();
    match configured {
        Some(engine_state::settings::ConfigValue::Int(0)) => 0,
        Some(engine_state::settings::ConfigValue::Int(size)) => size.clamp(64, 16_384) as u32,
        _ => DEFAULT_MAX_TEXTURE_SIZE,
    }
}

/// `(width, height)` after fitting the larger side to `cap` (0 = no cap).
pub(crate) fn capped_dimensions(width: u32, height: u32, cap: u32) -> (u32, u32) {
    let longest = width.max(height);
    if cap == 0 || longest <= cap {
        return (width, height);
    }
    let scale = f64::from(cap) / f64::from(longest);
    (
        ((f64::from(width) * scale).round() as u32).max(1),
        ((f64::from(height) * scale).round() as u32).max(1),
    )
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
            // The texels are budgeted by SceneDB's residency cache; the
            // payload itself is a handle.
            payload
                .downcast_ref::<LoadedTexture>()
                .map_or(0, |_| std::mem::size_of::<LoadedTexture>())
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

/// Read and decode the image file at `path`, scaled down to
/// [`max_texture_size`]. The decoded texels start out resident in SceneDB's
/// budgeted cache and are decoded again from the file if it drops them.
pub(crate) fn decode_image_file(path: &std::path::Path) -> Result<LoadedTexture, String> {
    let cap = max_texture_size();
    let (width, height, rgba) = decode_rgba(path, cap)?;
    let path = path.to_path_buf();
    Ok(LoadedTexture {
        width,
        height,
        texels: pulsar_scenedb::Residency::with_value(
            rgba,
            |texels: &Vec<u8>| texels.len(),
            move || decode_rgba(&path, cap).map(|(_, _, rgba)| rgba),
        ),
    })
}

/// Decode `path` to RGBA8 with its larger side at most `cap` (0 = no cap).
/// The scale-down happens before the RGBA conversion so a 4K RGB source never
/// becomes a 64 MiB RGBA buffer on the way to a 16 MiB one.
fn decode_rgba(path: &std::path::Path, cap: u32) -> Result<(u32, u32, Vec<u8>), String> {
    let shown = path.display();
    let bytes =
        std::fs::read(path).map_err(|error| format!("could not read '{shown}': {error}"))?;
    let mut decoded = image::load_from_memory(&bytes)
        .map_err(|error| format!("could not decode '{shown}': {error}"))?;
    drop(bytes);
    let (width, height) = (decoded.width(), decoded.height());
    if width == 0 || height == 0 {
        return Err(format!("'{shown}' has empty dimensions"));
    }
    let (scaled_width, scaled_height) = capped_dimensions(width, height, cap);
    if (scaled_width, scaled_height) != (width, height) {
        decoded = decoded.resize_exact(
            scaled_width,
            scaled_height,
            image::imageops::FilterType::Triangle,
        );
    }
    let rgba = decoded.to_rgba8();
    Ok((rgba.width(), rgba.height(), rgba.into_raw()))
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
    {
        let mut resident = resident()
            .lock()
            .map_err(|_| "texture residency lock poisoned".to_string())?;
        resident.retain(|_, entry| entry.owner.strong_count() != 0);
        if let Some(entry) = resident.get(&key) {
            if entry.version == version {
                return Ok(entry.slot);
            }
        }
    }
    // Not uploaded to this store yet. The texels may have to be decoded
    // again, which takes a while: fetch them with no lock held.
    let rgba = image.rgba()?;
    let mut resident = resident()
        .lock()
        .map_err(|_| "texture residency lock poisoned".to_string())?;
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
        &rgba,
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

#[cfg(test)]
mod residency_tests {
    use super::*;

    fn write_png(dir: &std::path::Path, name: &str, width: u32, height: u32) -> std::path::PathBuf {
        let path = dir.join(name);
        let mut image = image::RgbImage::new(width, height);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x % 251) as u8, (y % 241) as u8, 7]);
        }
        image.save(&path).unwrap();
        path
    }

    #[test]
    fn dimensions_fit_the_cap_and_keep_their_aspect() {
        assert_eq!(capped_dimensions(4096, 4096, 2048), (2048, 2048));
        assert_eq!(capped_dimensions(8192, 4096, 2048), (2048, 1024));
        assert_eq!(capped_dimensions(1000, 500, 2048), (1000, 500), "small images are untouched");
        assert_eq!(capped_dimensions(4096, 4096, 0), (4096, 4096), "0 disables the cap");
        assert_eq!(capped_dimensions(4096, 1, 64), (64, 1), "never collapses to zero");
    }

    #[test]
    fn decoding_applies_the_cap_before_the_rgba_conversion() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_png(dir.path(), "big.png", 300, 150);
        let (width, height, rgba) = decode_rgba(&path, 100).unwrap();
        assert_eq!((width, height), (100, 50));
        assert_eq!(rgba.len(), 100 * 50 * 4);
        let (width, height, rgba) = decode_rgba(&path, 0).unwrap();
        assert_eq!((width, height, rgba.len()), (300, 150, 300 * 150 * 4));
    }

    #[test]
    fn dropped_texels_are_decoded_again_identically() {
        let dir = tempfile::tempdir().unwrap();
        let first = decode_image_file(&write_png(dir.path(), "a.png", 64, 64)).unwrap();
        let before = first.rgba().unwrap();
        assert!(first.is_resident());

        // A budget below one image makes every newer image evict the older.
        let old_budget = pulsar_scenedb::residency_budget();
        pulsar_scenedb::set_residency_budget(1024);
        let second = decode_image_file(&write_png(dir.path(), "b.png", 64, 64)).unwrap();
        assert!(second.is_resident());
        assert!(!first.is_resident(), "the older image was dropped for room");

        let again = first.rgba().unwrap();
        assert_eq!(*again, *before, "the rebuilt texels match the originals");
        assert!(first.is_resident());
        pulsar_scenedb::set_residency_budget(old_budget);
    }
}
