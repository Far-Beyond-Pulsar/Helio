//! Bounded background import/cook pipeline for the standalone museum.
//! Preview meshes are published before any expensive VG builds. Warm starts
//! read the preview and VG caches without parsing the source FBXs again.

use super::vg_stress_data::{
    layout::Placement,
    materials::Mat,
    textures::{self, GltfMaterial},
};
use super::vg_stress_prep::{catalog::Config, polyhaven};
use super::{classic_proxy, load_or_build_vgeo_cache, stable_content_id};
use helio::{MeshUpload, PackedVertex};
use helio_asset_compat::{ConvertedMesh, LoadConfig, MaterialSource, load_scene_file_with_config};
use helio_vg_stream::{VgeoBytes, VgeoSource, content_words};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashSet,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver, SyncSender},
    },
};

/// Alpha-masked parts (twig / leaf cards) go through the classic path until
/// virtual geometry supports alpha-tested materials. Flip this when it does:
/// nothing else in the example changes.
pub const MASKED_PARTS_USE_VG: bool = false;
/// Vertex budget of the classic stand-in of a masked part. These stay
/// on screen permanently for now, so they are much smaller than the
/// temporary proxies of VG parts.
pub const MASKED_PROXY_VERTICES: usize = 60_000;

pub struct PreparedPart {
    pub asset: String,
    pub meta: PartMeta,
    pub mesh: MeshUpload,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct PartMeta {
    pub slot: String,
    pub transform: [f32; 16],
    pub radius: f32,
    pub content: Option<[u64; 2]>,
    pub source_triangles: u64,
    /// The part has an alpha-masked material (see [`MASKED_PARTS_USE_VG`]).
    #[serde(default)]
    pub masked: bool,
    vertices: usize,
    indices: usize,
}

pub enum Event {
    Part(PreparedPart),
    Vgeo(u128, Arc<VgeoBytes>),
    /// One decoded texture (full mip chain) and every material channel it feeds.
    Texture {
        bindings: Vec<(Mat, u8)>,
        srgb: bool,
        width: u32,
        height: u32,
        mips: Vec<Vec<u8>>,
    },
    Status(String),
    Error(String),
    Finished,
}

pub struct Loader {
    pub events: Receiver<Event>,
    cancelled: Arc<AtomicBool>,
}

impl Drop for Loader {
    fn drop(&mut self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

pub fn start(root: PathBuf, placements: &[Placement], vg_enabled: bool, config: Config) -> Loader {
    let mut seen = HashSet::new();
    let assets: Vec<_> = placements
        .iter()
        .filter(|p| seen.insert(p.asset.clone()))
        .map(|p| (p.asset.clone(), p.virtual_geometry))
        .collect();
    let (tx, events) = mpsc::sync_channel(2);
    let cancelled = Arc::new(AtomicBool::new(false));
    let stop = cancelled.clone();
    std::thread::Builder::new()
        .name("museum-loader".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run(&root, assets, vg_enabled, &config, &tx, &stop)
            }));
            clear_scene_cache();
            if result.is_err() {
                let _ = tx.send(Event::Error(
                    "Asset loader panicked; loaded scene remains usable".into(),
                ));
            }
            let _ = tx.send(Event::Finished);
        })
        .expect("start museum asset worker");
    Loader { events, cancelled }
}

fn send(tx: &SyncSender<Event>, event: Event, stop: &AtomicBool) -> bool {
    !stop.load(Ordering::Relaxed) && tx.send(event).is_ok()
}

fn content_id(words: [u64; 2]) -> u128 {
    u128::from(words[0]) | (u128::from(words[1]) << 64)
}

fn read_vgeo(cache: &Path, content: u128) -> Option<Arc<VgeoBytes>> {
    let bytes = std::fs::read(cache.join(format!("{content:032x}.vgeo"))).ok()?;
    let source = VgeoBytes::new(bytes).ok()?;
    (source.header().content_id == content_words(content)
        && helio_vg_build::is_current_cook(&source.header()))
    .then(|| Arc::new(source))
}

/// Split `path#variant` asset keys (tree variants).
pub fn split_key(key: &str) -> (&str, Option<usize>) {
    match key.rsplit_once('#') {
        Some((path, v)) => match v.parse() {
            Ok(v) => (path, Some(v)),
            Err(_) => (key, None),
        },
        None => (key, None),
    }
}

/// Where the file of an asset key is: normally under the crate root, but a
/// tree provisioned at a lower resolution has a different glTF name.
fn resolve_path(root: &Path, key: &str) -> PathBuf {
    let (rel, _) = split_key(key);
    let direct = root.join(rel);
    if direct.exists() {
        return direct;
    }
    let assets = root.join("assets/vg_stress");
    rel.strip_prefix("assets/vg_stress/trees/")
        .and_then(|r| r.split('/').next())
        .and_then(|id| polyhaven::tree_gltf(&assets, id))
        .unwrap_or(direct)
}

/// One imported source file, shared by every variant of it.
struct LoadedScene {
    meshes: Vec<ConvertedMesh>,
    sources: Vec<MaterialSource>,
    info: std::collections::HashMap<String, GltfMaterial>,
}

static SCENE_CACHE: Mutex<Option<(PathBuf, Arc<LoadedScene>)>> = Mutex::new(None);

/// Import `path` once and keep the last file around: the variants of a tree
/// package are processed back to back and must not re-read a 1 GB glTF each.
/// glTF `KHR_texture_transform` UV scale/offset is baked into the vertices.
fn load_scene(path: &Path) -> Result<Arc<LoadedScene>, String> {
    let mut cache = SCENE_CACHE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((p, scene)) = cache.as_ref() {
        if p == path {
            return Ok(scene.clone());
        }
    }
    *cache = None;
    let mut scene =
        load_scene_file_with_config(path, LoadConfig::default()).map_err(|e| e.to_string())?;
    let info = if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("gltf"))
    {
        textures::read_gltf_materials(path)
    } else {
        Default::default()
    };
    let mut meshes = std::mem::take(&mut scene.meshes);
    for mesh in &mut meshes {
        let uv = mesh
            .material_index
            .and_then(|i| scene.material_sources.get(i))
            .and_then(|s| info.get(&s.name))
            .and_then(|m| m.uv);
        if let Some((scale, offset)) = uv {
            for v in &mut mesh.vertices {
                v.tex_coords0 = [
                    v.tex_coords0[0] * scale[0] + offset[0],
                    v.tex_coords0[1] * scale[1] + offset[1],
                ];
            }
        }
    }
    let loaded = Arc::new(LoadedScene {
        meshes,
        sources: std::mem::take(&mut scene.material_sources),
        info,
    });
    *cache = Some((path.to_path_buf(), loaded.clone()));
    Ok(loaded)
}

fn clear_scene_cache() {
    *SCENE_CACHE.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

impl LoadedScene {
    fn slot(&self, mesh: &ConvertedMesh) -> String {
        mesh.material_index
            .and_then(|i| self.sources.get(i))
            .map(|s| s.name.clone())
            .unwrap_or_default()
    }

    fn masked(&self, mesh: &ConvertedMesh) -> bool {
        self.info.get(&self.slot(mesh)).is_some_and(|m| m.masked)
    }

    /// Meshes of `variant` (all of them for `None`).
    fn variant_meshes(&self, variant: Option<usize>) -> impl Iterator<Item = &ConvertedMesh> {
        self.meshes
            .iter()
            .filter(move |m| variant.is_none_or(|v| m.source_mesh == v))
    }
}

fn run(
    root: &Path,
    mut assets: Vec<(String, bool)>,
    vg_enabled: bool,
    config: &Config,
    tx: &SyncSender<Event>,
    stop: &AtomicBool,
) {
    let cache = root.join("assets/vg_stress/cache");
    // Runtime fallback of the build-script provisioning: download and convert
    // the scans if they are missing. Status shows in the window title.
    let assets_dir = root.join("assets/vg_stress");
    if !crate::vg_stress_prep::is_provisioned_with(&assets_dir, config) {
        let progress = |message: &str| {
            let _ = tx.send(Event::Status(format!("Preparing assets: {message}")));
        };
        if let Err(error) = crate::vg_stress_prep::provision_with(&assets_dir, config, &progress) {
            if !send(
                tx,
                Event::Error(format!("asset provisioning: {error}")),
                stop,
            ) {
                return;
            }
        }
    }
    // Keep architectural order (ground/shell first), then the smaller scans.
    // Variants of one file sort together so its import is shared.
    assets.sort_by_key(|(asset, vg)| {
        (
            *vg,
            if *vg {
                std::fs::metadata(resolve_path(root, asset))
                    .map(|m| m.len())
                    .unwrap_or(u64::MAX)
            } else {
                0
            },
            asset.clone(),
        )
    });
    let mut vg_jobs = Vec::new();
    for (asset, vg) in assets {
        if !send(
            tx,
            Event::Status(format!(
                "Loading {}",
                Path::new(&asset)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
            )),
            stop,
        ) {
            return;
        }
        match preview(root, &cache, &asset, vg) {
            Ok(parts) => {
                let contents = parts
                    .iter()
                    .filter_map(|p| p.meta.content.map(content_id))
                    .collect::<Vec<_>>();
                for part in parts {
                    if !send(tx, Event::Part(part), stop) {
                        return;
                    }
                }
                if vg && vg_enabled {
                    vg_jobs.push((asset, contents));
                }
            }
            Err(error) => {
                if !send(tx, Event::Error(format!("{asset}: {error}")), stop) {
                    return;
                }
            }
        }
    }

    // The previews are in; the big tree imports are not needed until the cooks.
    clear_scene_cache();

    // Poly Haven marble and tree textures: decoded, mipped and capped to the
    // quality budget off the window thread; one bounded upload per event.
    let mats = Mat::all(config);
    let jobs = textures::jobs(&assets_dir, &mats, config);
    let texture_count = jobs.len();
    let _ = tx.send(Event::Status(format!("Decoding {} textures", jobs.len())));
    let texture_started = std::time::Instant::now();
    let texture_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("texture decode pool");
    texture_pool.install(|| {
        jobs.into_par_iter().for_each(|job| {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            match textures::decode_with_opacity(&job.path, job.opacity.as_deref(), job.role, job.cap) {
                Ok(d) => {
                    let _ = send(
                        tx,
                        Event::Texture {
                            bindings: job.bindings,
                            srgb: job.role == textures::Role::Srgb,
                            width: d.width,
                            height: d.height,
                            mips: d.mips,
                        },
                        stop,
                    );
                }
                // The material keeps its flat colour.
                Err(error) => {
                    let _ = send(tx, Event::Error(format!("texture {error}")), stop);
                }
            }
        });
    });
    eprintln!(
        "[Museum] decoded {texture_count} textures in {:.2}s (4 workers)",
        texture_started.elapsed().as_secs_f64()
    );

    // Consume every warm VG cache before starting any cold, expensive cook.
    let mut misses = Vec::new();
    for (asset, contents) in vg_jobs {
        let mut missing = HashSet::new();
        for content in contents {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            if let Some(source) = read_vgeo(&cache, content) {
                if !send(tx, Event::Vgeo(content, source), stop) {
                    return;
                }
            } else {
                missing.insert(content);
            }
        }
        if !missing.is_empty() {
            misses.push((asset, missing));
        }
    }

    // Serial source imports bound peak scratch memory to one scan. A dedicated
    // Rayon pool leaves CPU time for rendering while the cluster builder runs.
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .expect("VG cook pool");
    for (asset, mut missing) in misses {
        if !send(
            tx,
            Event::Status(format!(
                "Preparing VG: {}",
                Path::new(&asset)
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
            )),
            stop,
        ) {
            return;
        }
        let (_, variant) = split_key(&asset);
        let path = resolve_path(root, &asset);
        let scene = match load_scene(&path) {
            Ok(scene) => scene,
            Err(error) => {
                if !send(tx, Event::Error(format!("VG {asset}: {error}")), stop) {
                    return;
                }
                continue;
            }
        };
        for mesh in scene.variant_meshes(variant) {
            if stop.load(Ordering::Relaxed) {
                return;
            }
            if scene.masked(mesh) && !MASKED_PARTS_USE_VG {
                continue;
            }
            let content = stable_content_id(&mesh.vertices, &mesh.indices);
            if !missing.remove(&content) {
                continue;
            }
            let result = pool.install(|| {
                load_or_build_vgeo_cache(
                    content,
                    &mesh.vertices,
                    &mesh.indices,
                    cache.join(format!("{content:032x}.vgeo")),
                )
            });
            let event = match result {
                Ok((source, _)) => Event::Vgeo(content, source),
                Err(error) => Event::Error(format!("VG {asset}: {error}; keeping preview")),
            };
            if !send(tx, event, stop) {
                return;
            }
        }
        if !missing.is_empty()
            && !send(
                tx,
                Event::Error(format!(
                    "{asset} changed during loading; restart to refresh previews"
                )),
                stop,
            )
        {
            return;
        }
    }
}

// (run() ends here; the cached import is dropped by its caller)
// Version the preview cache independently of VG: it also contains importer
// transforms, source fingerprints, material slots and the preview size policy.
const PREVIEW_MAGIC: &[u8; 8] = b"VGPREV05";

fn preview_path(cache: &Path, path: &Path, variant: Option<usize>) -> Result<PathBuf, String> {
    let meta = std::fs::metadata(path).map_err(|e| e.to_string())?;
    let modified = meta
        .modified()
        .map_err(|e| e.to_string())?
        .duration_since(std::time::UNIX_EPOCH)
        .map_err(|e| e.to_string())?
        .as_nanos();
    let key = format!(
        "{}:{}:{}:{}:{}:{}:{}",
        path.display(),
        meta.len(),
        modified,
        super::MAX_CLASSIC_PROXY_VERTICES,
        super::SCAN_SCENE_HEIGHT_METERS,
        MASKED_PROXY_VERTICES,
        variant.map(|v| v as i64).unwrap_or(-1)
    );
    let hash = key.bytes().fold(0xcbf29ce484222325u64, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
    });
    Ok(cache.join(format!("preview-v4-{hash:016x}.bin")))
}

fn preview(root: &Path, cache: &Path, asset: &str, vg: bool) -> Result<Vec<PreparedPart>, String> {
    let (_, variant) = split_key(asset);
    let path = resolve_path(root, asset);
    let cached = preview_path(cache, &path, variant)?;
    if let Ok(parts) = read_preview(&cached, asset) {
        return Ok(parts);
    }
    let scene = load_scene(&path)?;
    let mut min = glam::Vec3::splat(f32::INFINITY);
    let mut max = glam::Vec3::splat(f32::NEG_INFINITY);
    for mesh in scene.variant_meshes(variant) {
        for vertex in &mesh.vertices {
            let point = mesh
                .node_transform
                .transform_point3(glam::Vec3::from_array(vertex.position));
            min = min.min(point);
            max = max.max(point);
        }
    }
    let height = max.y - min.y;
    let expected_height = if vg {
        super::is_scan_asset(&path).then_some(super::SCAN_SCENE_HEIGHT_METERS)
    } else {
        let manifest: serde_json::Value = serde_json::from_slice(
            &std::fs::read(root.join("assets/vg_stress/kit/meshes/kit_manifest.json"))
                .map_err(|e| e.to_string())?,
        )
        .map_err(|e| e.to_string())?;
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let bounds = &manifest[stem.as_ref()]["bounds_blender"];
        bounds[5]
            .as_f64()
            .zip(bounds[2].as_f64())
            .map(|(hi, lo)| (hi - lo) as f32)
    };
    let scale = expected_height
        .filter(|h| *h > 1e-5 && height > 1e-5)
        .map(|expected| expected / height)
        .unwrap_or(1.0);
    log::info!("{asset}: imported height {height:.3}, scene scale {scale:.6}");
    let correction = glam::Mat4::from_scale(glam::Vec3::splat(scale));
    let mut parts = Vec::new();
    for mesh in scene.variant_meshes(variant) {
        if mesh.indices.len() < 3 || mesh.vertices.is_empty() {
            continue;
        }
        let slot = scene.slot(mesh);
        let masked = scene.masked(mesh);
        // Masked parts get no VG content id: they render classically.
        let use_vg = vg && (MASKED_PARTS_USE_VG || !masked);
        let content =
            use_vg.then(|| content_words(stable_content_id(&mesh.vertices, &mesh.indices)));
        let radius = mesh
            .vertices
            .iter()
            .map(|v| glam::Vec3::from_array(v.position).length())
            .fold(0.0f32, f32::max);
        let source_triangles = (mesh.indices.len() / 3) as u64;
        let upload = if vg {
            let limit = if masked && !MASKED_PARTS_USE_VG {
                MASKED_PROXY_VERTICES
            } else {
                super::MAX_CLASSIC_PROXY_VERTICES
            };
            classic_proxy(&mesh.vertices, &mesh.indices, limit)
        } else {
            MeshUpload {
                vertices: mesh.vertices.clone(),
                indices: mesh.indices.clone(),
            }
        };
        let mut node = mesh.node_transform;
        if variant.is_some() {
            // The package lays its variants side by side for display; each
            // placed tree stands on its own origin.
            node.w_axis = glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
        }
        let meta = PartMeta {
            slot,
            transform: (correction * node).to_cols_array(),
            radius,
            content,
            source_triangles,
            masked,
            vertices: upload.vertices.len(),
            indices: upload.indices.len(),
        };
        parts.push(PreparedPart {
            asset: asset.to_string(),
            meta,
            mesh: upload,
        });
    }
    if parts.is_empty() {
        return Err("asset contains no triangles".into());
    }
    if let Err(error) = write_preview(&cached, &parts) {
        log::warn!("Preview cache {}: {error}", cached.display());
    }
    Ok(parts)
}

fn write_preview(path: &Path, parts: &[PreparedPart]) -> Result<(), String> {
    fn write(path: &Path, parts: &[PreparedPart]) -> std::io::Result<()> {
        std::fs::create_dir_all(path.parent().unwrap())?;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        let metadata = serde_json::to_vec(&parts.iter().map(|p| &p.meta).collect::<Vec<_>>())?;
        file.write_all(PREVIEW_MAGIC)?;
        file.write_all(&(metadata.len() as u32).to_le_bytes())?;
        file.write_all(&metadata)?;
        for part in parts {
            file.write_all(bytemuck::cast_slice(&part.mesh.vertices))?;
            file.write_all(bytemuck::cast_slice(&part.mesh.indices))?;
        }
        file.flush()
    }
    let temp = path.with_extension("tmp");
    write(&temp, parts)
        .and_then(|_| super::replace_cache_file(&temp, path).map_err(std::io::Error::other))
        .map_err(|e| {
            let _ = std::fs::remove_file(&temp);
            e.to_string()
        })
}

fn read_preview(path: &Path, asset: &str) -> Result<Vec<PreparedPart>, String> {
    let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
    if file.metadata().map_err(|e| e.to_string())?.len() > 256 * 1024 * 1024 {
        return Err("oversize preview".into());
    }
    let mut bytes = Vec::new();
    std::io::BufReader::new(file)
        .read_to_end(&mut bytes)
        .map_err(|e| e.to_string())?;
    decode_preview(&bytes, asset)
}

fn decode_preview(bytes: &[u8], asset: &str) -> Result<Vec<PreparedPart>, String> {
    if bytes.get(..8) != Some(PREVIEW_MAGIC.as_slice()) {
        return Err("preview header".into());
    }
    let n = u32::from_le_bytes(
        bytes
            .get(8..12)
            .ok_or("preview metadata length")?
            .try_into()
            .unwrap(),
    ) as usize;
    let mut cursor = 12usize.checked_add(n).ok_or("preview overflow")?;
    let metadata: Vec<PartMeta> =
        serde_json::from_slice(bytes.get(12..cursor).ok_or("preview metadata")?)
            .map_err(|e| e.to_string())?;
    let mut parts = Vec::new();
    for meta in metadata {
        let mut take = |n: usize, stride: usize| -> Result<&[u8], String> {
            let end = cursor
                .checked_add(n.checked_mul(stride).ok_or("preview overflow")?)
                .ok_or("preview overflow")?;
            let result = bytes.get(cursor..end).ok_or("preview truncated")?;
            cursor = end;
            Ok(result)
        };
        let vertices: Vec<PackedVertex> =
            bytemuck::pod_collect_to_vec(take(meta.vertices, std::mem::size_of::<PackedVertex>())?);
        let indices: Vec<u32> = bytemuck::pod_collect_to_vec(take(meta.indices, 4)?);
        if indices.len() % 3 != 0
            || indices.iter().any(|&i| i as usize >= vertices.len())
            || !meta.radius.is_finite()
            || meta.transform.iter().any(|f| !f.is_finite())
        {
            return Err("invalid preview geometry".into());
        }
        parts.push(PreparedPart {
            asset: asset.to_string(),
            meta,
            mesh: MeshUpload { vertices, indices },
        });
    }
    if cursor != bytes.len() || parts.is_empty() {
        return Err("preview trailing data or empty".into());
    }
    Ok(parts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corrupt_preview_is_a_cache_miss() {
        assert!(decode_preview(b"VGPREV03", "mesh").is_err());
        let mut bytes = PREVIEW_MAGIC.to_vec();
        bytes.extend(u32::MAX.to_le_bytes());
        assert!(decode_preview(&bytes, "mesh").is_err());
    }

    #[test]
    fn imported_column_matches_blender_bounds_and_preview_round_trips() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let asset = "assets/vg_stress/kit/meshes/doric_column.fbx";
        let cache = std::env::temp_dir().join(format!(
            "vg-preview-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let parts = preview(root, &cache, asset, false).unwrap();
        let path = preview_path(&cache, &root.join(asset), None).unwrap();
        let loaded = read_preview(&path, asset).unwrap();
        let mut lo = glam::Vec3::splat(f32::INFINITY);
        let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
        for (a, b) in parts.iter().zip(&loaded) {
            assert_eq!(a.mesh.indices, b.mesh.indices);
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&a.mesh.vertices),
                bytemuck::cast_slice::<_, u8>(&b.mesh.vertices)
            );
            assert_eq!(a.meta.transform, b.meta.transform);
            assert_eq!(a.meta.slot, "marble");
            let transform = glam::Mat4::from_cols_array(&a.meta.transform);
            for v in &a.mesh.vertices {
                let p = transform.transform_point3(glam::Vec3::from_array(v.position));
                lo = lo.min(p);
                hi = hi.max(p);
            }
        }
        let dimensions = hi - lo;
        assert!(
            (dimensions - glam::Vec3::new(1.34, 6.2, 1.34))
                .abs()
                .max_element()
                < 0.005,
            "{dimensions:?}"
        );
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.pop();
        assert!(decode_preview(&bytes, asset).is_err());
        std::fs::write(&path, &bytes).unwrap();
        let repaired = preview(root, &cache, asset, false).unwrap();
        assert_eq!(repaired.len(), parts.len());
        assert_eq!(read_preview(&path, asset).unwrap().len(), parts.len());
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(cache).unwrap();
    }

    #[test]
    fn museum_references_existing_assets_and_material_slots() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let layout = super::super::vg_stress_data::layout::museum_layout(
            &super::super::vg_stress_data::dims::Dims::load(),
        );
        assert!(layout.placements.len() > 400);
        assert!(layout.statues.len() >= 18);
        for placement in layout.placements {
            if placement.kit_piece().is_some() {
                assert!(root.join(&placement.asset).is_file(), "{}", placement.asset);
            } else if placement.asset.contains('#') {
                // Poly Haven trees are provisioned on demand.
                let (path, variant) = split_key(&placement.asset);
                assert!(
                    path.starts_with("assets/vg_stress/trees/") && path.ends_with(".gltf"),
                    "{path}"
                );
                assert!(variant.is_some());
            } else {
                // Scans are prepared on demand (vg_stress_prep), not committed.
                assert!(
                    crate::is_scan_asset(&root.join(&placement.asset))
                        && placement.asset.ends_with(".glb"),
                    "{}",
                    placement.asset
                );
            }
            if let Some(piece) = placement.kit_piece() {
                assert!(
                    !super::super::vg_stress_data::materials::kit_slots(piece).is_empty(),
                    "{piece}"
                );
            }
        }
    }

    /// Local scan fixtures are intentionally not downloaded or cooked by a
    /// test. Run explicitly after the demo has populated its caches.
    #[test]
    #[ignore = "requires the six local museum scan preview and VG caches"]
    fn cached_museum_scans_reduce_triangles_with_distance() {
        use helio_pass_vg_cull::reference::{
            RefAsset, RefInstance, RefView, select_clusters, select_clusters_ex,
        };
        use helio_vg_build::VgeoReader;
        use helio_vg_format::{
            GpuCluster, GpuVgInstance, INST_CONE_CULL, INST_ENABLED, cluster_slot_local,
            cluster_slot_page,
        };
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let cache = root.join("assets/vg_stress/cache");
        let vgeo_cache = std::env::var_os("HELIO_VG_CANDIDATE_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| cache.clone());
        for name in [
            "dragon",
            "happy_buddha",
            "manuscript",
            "xyzrgb_dragon",
            "thai_statue",
            "lucy",
        ] {
            if std::env::var("HELIO_VG_SCAN").is_ok_and(|scan| scan != name) {
                continue;
            }
            let asset = format!("assets/vg_stress/meshes/{name}.glb");
            // Rebuild stale/missing preview caches from the local prepared
            // scan; validation must not depend on which exhibits the
            // interactive demo happened to load this run.
            let parts = preview(root, &cache, &asset, true)
                .unwrap_or_else(|error| panic!("{name}: could not prepare preview: {error}"));
            for part in parts {
                let content = content_id(part.meta.content.unwrap());
                let bytes = std::fs::read(vgeo_cache.join(format!("{content:032x}.vgeo")))
                    .expect("missing VG cache");
                let reader = VgeoReader::parse(&bytes).unwrap();
                assert!(
                    helio_vg_build::is_current_cook(&reader.header),
                    "{name}: stale cook, rebuild cache"
                );
                // Decode headers once: selection must not reparse a whole page
                // for each of the millions of cluster lookups in this sweep.
                let pages: Vec<Vec<GpuCluster>> = (0..reader.header.asset.page_count)
                    .map(|p| reader.page(p).unwrap().clusters().unwrap().to_vec())
                    .collect();
                let slots: Vec<_> = (0..reader.header.asset.page_count).map(Some).collect();
                let lookup = |p: u32, l: u32| pages[p as usize][l as usize];
                let asset = RefAsset {
                    root_node: reader.header.asset.root_node,
                    nodes: &reader.nodes,
                    group_refs: &reader.group_refs,
                    page_slot: &slots,
                    cluster: &lookup,
                };
                let transform = glam::Mat4::from_cols_array(&part.meta.transform);
                let mut lo = glam::Vec3::splat(f32::INFINITY);
                let mut hi = glam::Vec3::splat(f32::NEG_INFINITY);
                for c in pages
                    .iter()
                    .flatten()
                    .filter(|c| c.flags & GpuCluster::FLAG_LEAF != 0)
                {
                    let center = glam::Vec3::new(c.sphere[0], c.sphere[1], c.sphere[2]);
                    lo = lo.min(center - glam::Vec3::splat(c.sphere[3]));
                    hi = hi.max(center + glam::Vec3::splat(c.sphere[3]));
                }
                let center = transform.transform_point3((lo + hi) * 0.5);
                let inst = RefInstance {
                    matrix: part.meta.transform,
                    inst: GpuVgInstance {
                        asset_slot: 0,
                        // Every part admitted to VG is opaque. Masked tree
                        // cards have no VG content id, so backface cone culling
                        // is safe for this instance set.
                        flags: INST_ENABLED | INST_CONE_CULL,
                        lod_error_scale: 1.0,
                        object_row: 0,
                    },
                };
                let roots: Vec<_> = pages
                    .iter()
                    .flatten()
                    .filter(|c| c.flags & GpuCluster::FLAG_ROOT != 0)
                    .collect();
                let root_tris: u64 = roots
                    .iter()
                    .map(|c| u64::from(c.counts_material & 0xff))
                    .sum();
                let max_radius = roots
                    .iter()
                    .map(|c| c.sphere[3] * inst.scale())
                    .fold(0.0f32, f32::max);
                let max_error = roots
                    .iter()
                    .map(|c| c.self_error * inst.scale())
                    .fold(0.0f32, f32::max);
                eprintln!(
                    "{name}: coarsest={root_tris} tris in {} roots, max root radius={max_radius:.3}m, error={max_error:.6}m",
                    roots.len()
                );
                let mut counts = Vec::new();
                let mut cone_counts = Vec::new();
                for distance in [3.0, 6.0, 12.0, 24.0, 48.0, 96.0] {
                    let view = RefView::looking_down_neg_z(
                        (center + glam::Vec3::Z * distance).to_array(),
                        std::f32::consts::FRAC_PI_4,
                        0.1,
                        720.0,
                    );
                    let selected = select_clusters(&asset, &view, 1.0, &inst);
                    let triangles: u64 = selected
                        .drawn
                        .iter()
                        .map(|&s| {
                            u64::from(
                                lookup(cluster_slot_page(s), cluster_slot_local(s)).counts_material
                                    & 0xff,
                            )
                        })
                        .sum();
                    assert!(triangles > 0, "{name}: disappeared at {distance}m");
                    assert!(selected.requested_pages.is_empty());
                    counts.push(triangles);
                    let cone_selected = select_clusters_ex(&asset, &view, 1.0, &inst, true);
                    let cone_triangles: u64 = cone_selected
                        .drawn
                        .iter()
                        .map(|&s| {
                            u64::from(
                                lookup(cluster_slot_page(s), cluster_slot_local(s)).counts_material
                                    & 0xff,
                            )
                        })
                        .sum();
                    assert!(
                        cone_triangles > 0,
                        "{name}: cone cull removed the whole asset at {distance}m"
                    );
                    assert!(
                        cone_triangles <= triangles,
                        "{name}: cone cull added triangles at {distance}m"
                    );
                    cone_counts.push(cone_triangles);
                }
                eprintln!(
                    "{name}: source={} triangles; 3/6/12/24/48/96m at 720p, 1px: no-cone={counts:?}, cone={cone_counts:?}",
                    part.meta.source_triangles
                );
                let fov_y = std::f32::consts::FRAC_PI_4;
                let focal = 720.0 / (2.0 * (fov_y * 0.5).tan());
                let mut pixel_height_counts = Vec::new();
                for pixel_height in [18.0, 100.0, 1_000.0] {
                    let distance = focal * 2.0 / pixel_height;
                    let view = RefView::looking_down_neg_z(
                        (center + glam::Vec3::Z * distance).to_array(),
                        std::f32::consts::FRAC_PI_4,
                        0.1,
                        720.0,
                    );
                    let mut by_tau = Vec::new();
                    for tau in [1.0, 2.0, 4.0, 8.0] {
                        let selected = select_clusters(&asset, &view, tau, &inst);
                        let triangles: u64 = selected
                            .drawn
                            .iter()
                            .map(|&s| {
                                u64::from(
                                    lookup(cluster_slot_page(s), cluster_slot_local(s))
                                        .counts_material
                                        & 0xff,
                                )
                            })
                            .sum();
                        by_tau.push(triangles);
                    }
                    assert!(
                        by_tau.windows(2).all(|w| w[1] <= w[0]),
                        "{name}: looser pixel error must not select more geometry: {by_tau:?}"
                    );
                    pixel_height_counts.push((pixel_height as u32, by_tau));
                }
                eprintln!(
                    "{name}: projected object height -> (tau 1/2/4/8 px) triangle counts: {pixel_height_counts:?}"
                );
                assert!(
                    counts.windows(2).all(|w| w[1] <= w[0]),
                    "{name}: {counts:?}"
                );
                assert!(
                    counts[5] * 4 < counts[0],
                    "{name}: insufficient distance reduction: {counts:?}"
                );
                assert!(
                    counts[5] <= 25_000,
                    "{name}: distant triangle budget regressed: {counts:?}"
                );
                assert!(
                    root_tris <= 7_000,
                    "{name}: coarsest geometry is too dense: {root_tris}"
                );
                // Establish the actual geometry floor separately from the
                // ordinary distance sweep. Dense distant cuts may still be
                // intermediate levels because their cached errors are large.
                let far = RefView::looking_down_neg_z(
                    (center + glam::Vec3::Z * 1.0e8).to_array(),
                    std::f32::consts::FRAC_PI_4,
                    0.1,
                    720.0,
                );
                let coarsest = select_clusters(&asset, &far, 1.0, &inst);
                let coarsest_tris: u64 = coarsest
                    .drawn
                    .iter()
                    .map(|&s| {
                        u64::from(
                            lookup(cluster_slot_page(s), cluster_slot_local(s)).counts_material
                                & 0xff,
                        )
                    })
                    .sum();
                assert_eq!(
                    coarsest_tris, root_tris,
                    "{name}: finite error must eventually select roots"
                );
            }
        }
    }

    #[test]
    #[ignore = "explicit offline cache cook of local scan sources"]
    fn cook_museum_scan_caches() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"));
        let cache = std::env::var_os("HELIO_VG_CANDIDATE_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|| root.join("assets/vg_stress/cache"));
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        for name in [
            "dragon",
            "happy_buddha",
            "manuscript",
            "xyzrgb_dragon",
            "thai_statue",
            "lucy",
        ] {
            if std::env::var("HELIO_VG_SCAN").is_ok_and(|scan| scan != name) {
                continue;
            }
            eprintln!("Cooking {name}");
            let now = std::time::Instant::now();
            let scene = load_scene_file_with_config(
                &root.join(format!("assets/vg_stress/meshes/{name}.glb")),
                LoadConfig::default(),
            )
            .unwrap();
            for mesh in scene.meshes {
                let content = stable_content_id(&mesh.vertices, &mesh.indices);
                pool.install(|| {
                    load_or_build_vgeo_cache(
                        content,
                        &mesh.vertices,
                        &mesh.indices,
                        cache.join(format!("{content:032x}.vgeo")),
                    )
                })
                .unwrap();
            }
            eprintln!("Cooked {name} in {:.2}s", now.elapsed().as_secs_f64());
        }
    }
}
