use std::path::{Path, PathBuf};

use helio::MeshUpload;

/// SceneDB writes a component's `sync_component` wants to make, queued
/// instead of applied inline.
///
/// `ComponentRuntimeBehavior::sync_component` runs under the component-sync
/// pass's read lock on `WorldSceneStore` (see `engine_backend`'s
/// `sync_scene` doc for exactly why: a write lock held across the whole
/// pass previously caused a real, shipped gizmo drag-release freeze). It
/// therefore never has `&mut World`, only `&pulsar_scenedb::World`
/// (indirectly, via `Subsystems`) plus this queue -- push a closure here
/// instead of authoring immediately, and `engine_backend` applies every
/// queued write during the pass's own short, pre-existing Phase 2 write
/// lock (the same one `refresh_world_component_gpu_mirror_for_class` already
/// runs under, for the identical reason).
///
/// Registered once per `sync_snapshot_components` call via `register_ref`,
/// shared (by the caller) across the whole sync pass so every entity's
/// queued writes land in one `Vec`, applied together.
#[derive(Default)]
pub struct PendingWorldWrites {
    writes: Vec<Box<dyn FnOnce(&mut pulsar_scenedb::World) + Send>>,
}

impl PendingWorldWrites {
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue a write. `entity` is typically captured by the closure (e.g.
    /// via `move |world| { world.insert(entity, component); }`) -- this
    /// method doesn't thread it through itself since some writes (a
    /// `World::remove`, a multi-entity edit) don't fit a single-entity shape.
    pub fn push(&mut self, write: impl FnOnce(&mut pulsar_scenedb::World) + Send + 'static) {
        self.writes.push(Box::new(write));
    }

    /// Apply and clear every queued write. Called by `engine_backend` inside
    /// the sync pass's Phase 2 write lock.
    pub fn drain_and_apply(&mut self, world: &mut pulsar_scenedb::World) {
        for write in self.writes.drain(..) {
            write(world);
        }
    }
}

/// The engine's built-in assets — resolved at compile time so embedded
/// primitives (SM_Cube, SM_Sphere, etc.) are always available.
///
/// Five levels up, not three: this crate lives at
/// `crates/renderer/helio/crates/helio-component` (moved here from
/// `crates/subsystems/pulsar_rendering`, renamed `helio_component` --
/// Pulsar-Native#561), two levels deeper than its old home.
const ENGINE_ASSETS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../../../assets");

macro_rules! prim_bytes {
    ($name:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../../../assets/meshes/primitives/",
            $name,
            ".fbx"
        ))
    };
}

fn embedded_primitive(stem: &str) -> Option<&'static [u8]> {
    match stem {
        "SM_Cube" => Some(prim_bytes!("SM_Cube")),
        "SM_Sphere" => Some(prim_bytes!("SM_Sphere")),
        "SM_Cylinder" => Some(prim_bytes!("SM_Cylinder")),
        "SM_Plane" => Some(prim_bytes!("SM_Plane")),
        "SM_Torus" => Some(prim_bytes!("SM_Torus")),
        _ => None,
    }
}

/// Resolve an asset path to an existing file on disk.
///
/// Checks (in order):
///  1. absolute path
///  2. project-root-relative
///  3. working-directory-relative
///  4. `cwd/assets/` (editor convention)
///  5. engine built-in assets (embedded primitives dir)
pub fn resolve_asset_path(project_root: &Path, asset: &str) -> PathBuf {
    let norm = asset.replace('\\', "/");
    let p = Path::new(&norm);

    if p.is_absolute() && p.exists() {
        return p.to_path_buf();
    }

    let proj = project_root.join(&norm);
    if proj.exists() {
        return proj;
    }

    if let Ok(cwd) = std::env::current_dir() {
        let cwd_path = cwd.join(&norm);
        if cwd_path.exists() {
            return cwd_path;
        }
        let cwd_assets = cwd.join("assets").join(&norm);
        if cwd_assets.exists() {
            return cwd_assets;
        }
    }

    let engine = Path::new(ENGINE_ASSETS_DIR).join(&norm);
    if engine.exists() {
        return engine;
    }

    proj
}

/// Load a mesh file from disk (or from embedded primitive bytes) into a
/// [`MeshUpload`] payload.
///
/// Components call this when they need to load geometry that hasn't been
/// cached yet.  The `path` should already be resolved to an absolute path
/// (use [`resolve_asset_path`] first if needed).
pub fn load_mesh_upload(path: &Path) -> Option<MeshUpload> {
    // Engine-native baked mesh asset (`.mesh`), produced at import time from the
    // source model. Load it directly — no conversion or options (issues #391/#409).
    if path.extension().and_then(|e| e.to_str()) == Some("mesh") {
        let bytes = std::fs::read(path).ok()?;
        let (mesh, id) = crate::mesh_cache::decode(&bytes)?;

        // Content-id provenance (Pulsar-Native#658): prime the memoization
        // cache now, from the id `decode` just produced (read directly for
        // v2, computed on the fly for v1) — so `MeshAssetPath::content_id`'s
        // later resolve, driven by SceneDB's GPU-mirror write dispatch, is
        // a warm hit instead of a second cold hash of this same file.
        crate::mesh_cache::prime_content_id_cache(path, id);

        // v1 backfill: a file with no header id gets upgraded to v2 in
        // place, atomically (write to a sibling temp file, then rename —
        // never a partial/torn write visible to a concurrent reader) so
        // the NEXT load reads the id straight from the header instead of
        // re-hashing. Best-effort: any failure here (read-only project
        // dir, concurrent access, etc.) is silently ignored -- the load
        // itself already succeeded with a correct in-memory id, and the
        // upgrade is a pure optimization, never load-bearing.
        if bytes.len() < 8 || u32::from_le_bytes(bytes[4..8].try_into().unwrap_or_default()) != 2 {
            let upgraded = crate::mesh_cache::encode(&mesh, id);
            let tmp = path.with_extension("mesh.tmp");
            if std::fs::write(&tmp, &upgraded).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }

        return Some(mesh);
    }

    let cfg = helio_asset_compat::LoadConfig {
        flip_uv_y: true,
        merge_meshes: false,
        import_scale: glam::Vec3::ONE,
    };

    // Try disk first.
    if path.exists() {
        return helio_asset_compat::load_scene_file_with_config(path, cfg)
            .ok()?
            .meshes
            .into_iter()
            .next()
            .map(|m| MeshUpload {
                vertices: m.vertices,
                indices: m.indices,
            });
    }

    // Fallback: check embedded primitives.
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("");
    if let Some(bytes) = embedded_primitive(stem) {
        return helio_asset_compat::load_scene_bytes_with_config(bytes, "fbx", None, cfg)
            .ok()?
            .meshes
            .into_iter()
            .next()
            .map(|m| MeshUpload {
                vertices: m.vertices,
                indices: m.indices,
            });
    }

    None
}
