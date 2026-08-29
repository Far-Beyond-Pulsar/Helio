//! Engine-native baked mesh assets (issues #391 / #409).
//!
//! Model import happens **at copy time**: a dropped source model
//! (fbx/obj/gltf/usd) is converted with the chosen import options into an
//! engine-native `.mesh` asset written into the project. **The source file
//! itself is not brought into the project** — only the native asset. Components
//! reference the `.mesh` asset and [`crate::subsystems::load_mesh_upload`] loads
//! it directly, with no per-load conversion or import options.
//!
//! Format (`PMSH`): a small header (magic + version + vertex/index counts)
//! followed by the bytemuck-packed [`PackedVertex`] and `u32` index arrays.
//!
//! NOTE: only mesh geometry is baked today. Materials/textures from the source
//! scene are not yet written as native assets — that's a follow-up once the
//! engine's native material-asset format is wired in here.
//!
//! # v2: content-id provenance (Pulsar-Native#632/#658)
//!
//! v2 appends a 16-byte `content_id: u128` to the header — the xxh3-128
//! hash of the decoded vertex+index bytes, computed once at import time
//! (see [`content_id_for_bytes`]) and stored so every later load reads it
//! directly instead of re-hashing. This is what
//! [`crate::components::static_mesh_component::MeshAssetPath::content_id`]
//! feeds into SceneDB's `ContentAddressed`/interned-var-len mechanism —
//! `helio-component` and `pulsar_scenedb` both stay opaque to what the id
//! MEANS; it's just a stable identity two components can compare.
//!
//! v1 files (no id in the header) decode fine — [`decode`] computes the
//! same hash on the fly and [`crate::subsystems::load_mesh_upload`]
//! opportunistically rewrites the file as v2 so the NEXT load is a direct
//! header read, not a repeated hash. A failed rewrite (read-only project
//! dir, etc.) never fails the load itself; the in-memory id from that one
//! call is still correct, just not persisted.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use bytemuck::Zeroable;
use helio::{MeshUpload, PackedVertex};
use pulsar_reflection::{RuntimeTypeInfo, TypeStructure, RUNTIME_TYPE_REGISTRY};

const MAGIC: &[u8; 4] = b"PMSH";
/// Current WRITE version — every fresh `encode` call produces this.
const VERSION: u32 = 2;
const HEADER: usize = 4 + 4 + 8 + 8; // magic + version + vertex_count + index_count
/// v2 only: `HEADER` bytes plus a trailing 16-byte `content_id: u128`,
/// BEFORE the vertex/index payload (so a v1 reader that somehow ignored the
/// version check would still fail the length check rather than misread
/// content-id bytes as geometry).
const HEADER_V2: usize = HEADER + 16;

// ---------------------------------------------------------------------------
// Engine-native import-schema types (bridge from solid_rs::configurator).
// ---------------------------------------------------------------------------

/// UI constraints for an import field.
#[derive(Debug, Clone, Default)]
pub struct FieldConstraints {
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub step: Option<f64>,
}

/// A single configurable import option — the engine-native equivalent of
/// [`solid_rs::configurator::OptionField`] but expressed in terms of the
/// reflection system so the gpui property editors can render it generically.
pub struct ImportField {
    pub key: String,
    pub label: String,
    pub doc: String,
    pub type_info: &'static RuntimeTypeInfo,
    pub default: Box<dyn Any + Send>,
    pub constraints: FieldConstraints,
}

impl fmt::Debug for ImportField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ImportField")
            .field("key", &self.key)
            .field("label", &self.label)
            .field("doc", &self.doc)
            .field("type_info", &self.type_info)
            .field("default", &"<dyn Any>")
            .field("constraints", &self.constraints)
            .finish()
    }
}

/// An ordered set of [`ImportField`]s describing a loader's import options.
#[derive(Debug)]
pub struct OptionsSchema {
    pub fields: Vec<ImportField>,
}

// ---------------------------------------------------------------------------
// Conversion from solid_rs schema types
// ---------------------------------------------------------------------------

/// Leak a value into a `&'static` reference (tiny allocation, modal lifetime).
#[allow(dead_code)]
fn leak_static<T: 'static>(val: T) -> &'static T {
    Box::leak(Box::new(val))
}

fn build_enum_type_info(label: &str, choices: &[String]) -> &'static RuntimeTypeInfo {
    let variants: Vec<&'static str> = choices
        .iter()
        .map(|c| Box::leak(c.clone().into_boxed_str()) as &'static str)
        .collect();
    let variants: &'static [&'static str] = Box::leak(variants.into_boxed_slice());
    Box::leak(Box::new(RuntimeTypeInfo {
        type_id: TypeId::of::<u64>(),
        type_name: Box::leak(format!("enum:{label}").into_boxed_str()),
        size: 8,
        align: 8,
        structure: TypeStructure::Enum { variants },
        color: None,
    }))
}

fn convert_default(_kind: &helio_asset_compat::OptionKind, dv: &helio_asset_compat::OptionValue) -> Box<dyn Any + Send> {
    use helio_asset_compat::OptionValue as OV;
    match dv {
        OV::Bool(b) => Box::new(*b),
        OV::Int(i) => Box::new(*i),
        OV::Float(f) => Box::new(*f),
        OV::Text(s) => Box::new(s.clone()),
        OV::Choice(s) => Box::new(s.clone()),
    }
}

fn convert_field(field: &helio_asset_compat::OptionField) -> ImportField {
    use helio_asset_compat::OptionKind as OK;

    let (type_info, constraints) = match &field.kind {
        OK::Bool => (
            RUNTIME_TYPE_REGISTRY.get::<bool>().expect("bool registered"),
            FieldConstraints::default(),
        ),
        OK::Int { min, max, step } => (
            RUNTIME_TYPE_REGISTRY.get::<i64>().expect("i64 registered"),
            FieldConstraints {
                min: min.map(|v| v as f64),
                max: max.map(|v| v as f64),
                step: step.map(|v| v as f64),
            },
        ),
        OK::Float { min, max, step } => (
            RUNTIME_TYPE_REGISTRY.get::<f64>().expect("f64 registered"),
            FieldConstraints {
                min: *min,
                max: *max,
                step: *step,
            },
        ),
        OK::Enum { choices } => (
            build_enum_type_info(&field.label, choices),
            FieldConstraints::default(),
        ),
        OK::Text => (
            RUNTIME_TYPE_REGISTRY.get::<String>().expect("String registered"),
            FieldConstraints::default(),
        ),
    };

    ImportField {
        key: field.key.clone(),
        label: field.label.clone(),
        doc: field.doc.clone(),
        type_info,
        default: convert_default(&field.kind, &field.default),
        constraints,
    }
}

/// Convert solid_rs configurator values to the engine's dynamic map.
pub fn hashmap_from_option_values(values: &helio_asset_compat::OptionValues) -> HashMap<String, Box<dyn Any + Send>> {
    use helio_asset_compat::OptionValue as OV;
    let mut map = HashMap::new();
    for (k, v) in values.0.iter() {
        let val: Box<dyn Any + Send> = match v {
            OV::Bool(b) => Box::new(*b),
            OV::Int(i) => Box::new(*i),
            OV::Float(f) => Box::new(*f),
            OV::Text(s) => Box::new(s.clone()),
            OV::Choice(s) => Box::new(s.clone()),
        };
        map.insert(k.clone(), val);
    }
    map
}

/// Convert the engine's dynamic value map back to solid_rs configurator values.
pub fn option_values_from_hashmap(map: &HashMap<String, Box<dyn Any + Send>>) -> helio_asset_compat::OptionValues {
    use helio_asset_compat::OptionValue as OV;
    let mut vals = helio_asset_compat::OptionValues::new();
    for (k, v) in map {
        let ov = if let Some(b) = v.downcast_ref::<bool>() {
            OV::Bool(*b)
        } else if let Some(i) = v.downcast_ref::<i64>() {
            OV::Int(*i)
        } else if let Some(f) = v.downcast_ref::<f64>() {
            OV::Float(*f)
        } else if let Some(s) = v.downcast_ref::<String>() {
            OV::Text(s.clone())
        } else if let Some(u) = v.downcast_ref::<u64>() {
            OV::Int(*u as i64)
        } else if let Some(i) = v.downcast_ref::<i32>() {
            OV::Int(*i as i64)
        } else if let Some(f) = v.downcast_ref::<f32>() {
            OV::Float(*f as f64)
        } else {
            continue;
        };
        vals.set(k, ov);
    }
    vals
}

/// Native asset path for an imported source model: `<dest_dir>/<stem>.mesh`
/// (e.g. dropping `foo.fbx` into `dir` → `dir/foo.mesh`).
pub fn native_mesh_path(dest_dir: &Path, source: &Path) -> PathBuf {
    let stem = source.file_stem().and_then(|s| s.to_str()).unwrap_or("mesh");
    dest_dir.join(format!("{stem}.mesh"))
}

/// The import-options schema advertised for source extension `ext` (no leading
/// dot), for driving a configurator UI. `None` if the format isn't importable.
pub fn options_schema(ext: &str) -> Option<OptionsSchema> {
    let helio_schema = helio_asset_compat::options_schema_for_extension(ext)?;
    Some(OptionsSchema {
        fields: helio_schema.fields.iter().map(convert_field).collect(),
    })
}

/// Whether `ext` (without leading dot) is a source model format we import.
pub fn is_importable_model(ext: &str) -> bool {
    matches!(
        ext.to_ascii_lowercase().as_str(),
        "fbx" | "obj" | "gltf" | "glb" | "usd" | "usda" | "usdc" | "usdz"
    )
}

/// Path → content id memoization, canonical-path-keyed so two different
/// spellings of the SAME file (`a.png` vs `A/../a.png`, a hardlink/second
/// copy resolving through a symlink, etc.) converge on one cache entry and
/// therefore one id (Pulsar-Native#661's path-aliasing test). Value is
/// `(mtime, len, id)`: a stale entry (file's current mtime/len disagree
/// with what's cached) is treated as a miss, so an edited file mints a new
/// id on its next resolve rather than serving a stale one forever.
static CONTENT_ID_CACHE: std::sync::OnceLock<std::sync::Mutex<HashMap<PathBuf, (std::time::SystemTime, u64, u128)>>> =
    std::sync::OnceLock::new();

fn content_id_cache() -> &'static std::sync::Mutex<HashMap<PathBuf, (std::time::SystemTime, u64, u128)>> {
    CONTENT_ID_CACHE.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

/// Resolve a stable content id for the mesh asset at `abs_path` (an
/// already-resolved absolute path — see `subsystems::resolve_asset_path`).
///
/// - A native `.mesh` file: reads the id straight out of its v2 header (a
///   `HEADER_V2`-byte read, not a full geometry decode) when present; a v1
///   file with no stored id falls through to the memoization path below
///   (it'll be upgraded to v2 the next time `load_mesh_upload` loads it,
///   see that fn's doc, but this call itself doesn't write anything).
/// - Anything else (a v1 file with nothing to read yet, or a non-native
///   source path pre-import): canonicalizes `abs_path` (this is what makes
///   two spellings of the same file converge — symlinks and `..` segments
///   both resolve here) and checks the mtime/len-validated memoization
///   cache; on a miss/staleness, hashes the file's raw bytes once
///   (xxh3-128) and caches the result.
///
/// `None` only if `abs_path` can't be read/canonicalized at all (a
/// dangling reference) — mirrors `MeshAssetPath::content_id`'s own
/// `HandleId::ZERO`-on-failure convention at its call site.
pub fn content_id_for_path(abs_path: &Path) -> Option<u128> {
    if abs_path.extension().and_then(|e| e.to_str()) == Some("mesh") {
        if let Ok(bytes) = std::fs::read(abs_path) {
            if bytes.len() >= HEADER_V2 && &bytes[0..4] == MAGIC {
                if let Ok(version) = bytes[4..8].try_into().map(u32::from_le_bytes) {
                    if version == VERSION {
                        if let Ok(id_bytes) = bytes[HEADER..HEADER_V2].try_into() {
                            return Some(u128::from_le_bytes(id_bytes));
                        }
                    }
                }
            }
        }
    }

    let canonical = std::fs::canonicalize(abs_path).ok()?;
    let meta = std::fs::metadata(&canonical).ok()?;
    let mtime = meta.modified().ok()?;
    let len = meta.len();

    {
        let cache = content_id_cache().lock().expect("content id cache mutex poisoned");
        if let Some(&(cached_mtime, cached_len, id)) = cache.get(&canonical) {
            if cached_mtime == mtime && cached_len == len {
                return Some(id);
            }
        }
    }

    let bytes = std::fs::read(&canonical).ok()?;
    let id = twox_hash::XxHash3_128::oneshot(&bytes);
    content_id_cache().lock().expect("content id cache mutex poisoned").insert(canonical, (mtime, len, id));
    Some(id)
}

/// Primes the memoization cache directly from an id [`decode`] already
/// computed (or read) during a hydrate-time load — so the write-path's
/// later [`content_id_for_path`] call (from
/// `MeshAssetPath::content_id`, driven by SceneDB's derive-generated GPU
/// mirror dispatch) is a warm cache hit, not a second cold hash of the same
/// file. A no-op if `abs_path` can't be canonicalized or stat'd (matches
/// [`content_id_for_path`]'s own silent-`None` tolerance — priming is an
/// optimization, never load-bearing for correctness).
pub fn prime_content_id_cache(abs_path: &Path, id: u128) {
    let Ok(canonical) = std::fs::canonicalize(abs_path) else { return };
    let Ok(meta) = std::fs::metadata(&canonical) else { return };
    let Ok(mtime) = meta.modified() else { return };
    content_id_cache().lock().expect("content id cache mutex poisoned").insert(canonical, (mtime, meta.len(), id));
}

/// xxh3-128 over the SAME bytes [`encode`] writes for the geometry payload
/// (vertices then indices, both in their native `Pod` byte representation)
/// — the content identity for a mesh whose header doesn't already carry one
/// (v1 backfill) or that was never a native `.mesh` file at all (a source
/// format, hashed post-conversion). Two calls with byte-identical geometry
/// always produce the same id; this crate never claims anything stronger
/// (e.g. semantic equivalence of differently-tessellated meshes).
pub fn content_id_for_bytes(mesh: &MeshUpload) -> u128 {
    let mut buf = Vec::with_capacity(
        mesh.vertices.len() * std::mem::size_of::<PackedVertex>() + mesh.indices.len() * 4,
    );
    buf.extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
    buf.extend_from_slice(bytemuck::cast_slice(&mesh.indices));
    twox_hash::XxHash3_128::oneshot(&buf)
}

/// Serialise a [`MeshUpload`] into the native `.mesh` byte format (always
/// v2 — see the module doc). `content_id` is the identity to store in the
/// header; callers that don't already have one on hand (a fresh import)
/// should compute it via [`content_id_for_bytes`] first.
pub fn encode(mesh: &MeshUpload, content_id: u128) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        HEADER_V2
            + mesh.vertices.len() * std::mem::size_of::<PackedVertex>()
            + mesh.indices.len() * 4,
    );
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(mesh.vertices.len() as u64).to_le_bytes());
    out.extend_from_slice(&(mesh.indices.len() as u64).to_le_bytes());
    out.extend_from_slice(&content_id.to_le_bytes());
    out.extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
    out.extend_from_slice(bytemuck::cast_slice(&mesh.indices));
    out
}

/// Parse a [`MeshUpload`] plus its content id from native `.mesh` bytes, or
/// `None` if invalid / an unsupported version / a size mismatch (callers
/// may fall back to converting a source). v1 files (no stored id) get one
/// computed on the fly via [`content_id_for_bytes`] — identical to what a
/// fresh v2 write of the same geometry would store, so a later backfill
/// write produces a byte-stable upgrade, not a new identity.
pub fn decode(bytes: &[u8]) -> Option<(MeshUpload, u128)> {
    if bytes.len() < HEADER || &bytes[0..4] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    if version != 1 && version != VERSION {
        return None;
    }
    let vcount = u64::from_le_bytes(bytes[8..16].try_into().ok()?) as usize;
    let icount = u64::from_le_bytes(bytes[16..24].try_into().ok()?) as usize;
    let vbytes = vcount.checked_mul(std::mem::size_of::<PackedVertex>())?;
    let ibytes = icount.checked_mul(4)?;
    let (vstart, stored_id) = if version == 1 {
        (HEADER, None)
    } else {
        (HEADER_V2, Some(u128::from_le_bytes(bytes[HEADER..HEADER_V2].try_into().ok()?)))
    };
    let istart = vstart.checked_add(vbytes)?;
    let end = istart.checked_add(ibytes)?;
    if bytes.len() < end {
        return None;
    }

    // Copy into properly-aligned Vecs — the source byte slice alignment is not
    // guaranteed to match `PackedVertex`, so casting it directly could panic.
    let mut vertices = vec![PackedVertex::zeroed(); vcount];
    bytemuck::cast_slice_mut(&mut vertices).copy_from_slice(&bytes[vstart..istart]);
    let mut indices = vec![0u32; icount];
    bytemuck::cast_slice_mut(&mut indices).copy_from_slice(&bytes[istart..end]);

    let mesh = MeshUpload { vertices, indices };
    let id = match stored_id {
        Some(id) => id,
        None => content_id_for_bytes(&mesh),
    };
    Some((mesh, id))
}

/// Resolve import options for a native asset — options stored from a previous
/// import (keyed by the native path) if present, otherwise the source format's
/// schema defaults.
pub fn resolve_options(native: &Path, ext: &str) -> HashMap<String, Box<dyn Any + Send>> {
    if let Some(root) = engine_state::get_project_path() {
        let root = Path::new(&root);
        let key = engine_fs::import_options::asset_key(root, native);
        if let Some(json) = engine_fs::import_options::get(root, &key) {
            if let Ok(values) = serde_json::from_value::<helio_asset_compat::OptionValues>(json) {
                return hashmap_from_option_values(&values);
            }
        }
    }
    helio_asset_compat::options_schema_for_extension(ext)
        .map(|s| hashmap_from_option_values(&s.default_values()))
        .unwrap_or_default()
}

/// Import `source` into an engine-native `.mesh` asset at `native`, converting
/// with `values`. The source file is **not** copied into the project. Persists
/// the chosen options (keyed by the native path) for reimport. Returns the
/// written native path.
pub fn import_model_to_native(
    source: &Path,
    native: &Path,
    values: &HashMap<String, Box<dyn Any + Send>>,
) -> Result<PathBuf, String> {
    let ov = option_values_from_hashmap(values);
    let scene = helio_asset_compat::load_scene_file_with_values(source, &ov)
        .map_err(|e| format!("import conversion failed: {e}"))?;

    let mesh = scene
        .meshes
        .into_iter()
        .next()
        .ok_or_else(|| "model contained no meshes".to_string())?;
    let upload = MeshUpload {
        vertices: mesh.vertices,
        indices: mesh.indices,
    };

    let content_id = content_id_for_bytes(&upload);
    std::fs::write(native, encode(&upload, content_id))
        .map_err(|e| format!("failed to write native mesh {}: {e}", native.display()))?;

    // Persist chosen options for reimport / configurator pre-fill (#409).
    if let Some(root) = engine_state::get_project_path() {
        let root = Path::new(&root);
        let key = engine_fs::import_options::asset_key(root, native);
        let ext = source.extension().and_then(|e| e.to_str()).unwrap_or("");
        if let Some(schema) = helio_asset_compat::options_schema_for_extension(ext) {
            let mut json_map = serde_json::Map::new();
            for field in &schema.fields {
                if let Some(val) = values.get(&field.key) {
                    if let Ok(json) = RUNTIME_TYPE_REGISTRY.serialize_json_for_any(val.as_ref()) {
                        json_map.insert(field.key.clone(), json);
                    }
                }
            }
            let _ = engine_fs::import_options::set(root, &key, serde_json::Value::Object(json_map));
        }
    }

    Ok(native.to_path_buf())
}

/// Import `source` into `dest_dir` as a native `.mesh`, resolving options from
/// storage (reimport) or schema defaults. Convenience for the drop flow when no
/// configurator modal supplied explicit options. Returns the native path.
pub fn import_model_to_native_default(source: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
    let native = native_mesh_path(dest_dir, source);
    let ext = source.extension().and_then(|e| e.to_str()).unwrap_or("");
    let values = resolve_options(&native, ext);
    import_model_to_native(source, &native, &values)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_path_uses_stem_and_dot_mesh() {
        assert_eq!(
            native_mesh_path(Path::new("/proj/models"), Path::new("/downloads/foo.fbx")),
            PathBuf::from("/proj/models/foo.mesh")
        );
    }

    #[test]
    fn encode_decode_roundtrip() {
        let mesh = MeshUpload {
            vertices: vec![PackedVertex::zeroed(); 3],
            indices: vec![0u32, 1, 2],
        };
        let id = content_id_for_bytes(&mesh);
        let bytes = encode(&mesh, id);
        let (back, decoded_id) = decode(&bytes).expect("decode");
        assert_eq!(back.vertices.len(), 3);
        assert_eq!(back.indices, vec![0, 1, 2]);
        assert_eq!(decoded_id, id, "v2 must round-trip the exact stored id, not recompute it");
        // Truncated / garbage input is rejected, not panicked on.
        assert!(decode(&bytes[..10]).is_none());
        assert!(decode(b"nope").is_none());
    }

    #[test]
    fn v1_files_decode_with_a_computed_id_matching_a_fresh_v2_encode() {
        // Hand-roll a v1 header (no content_id field) around the same
        // geometry `encode` would write, to prove the backfill path
        // produces the IDENTICAL id a fresh v2 write of the same bytes
        // would -- the whole point of `content_id_for_bytes` being the
        // single source both `encode` (fresh import) and `decode`'s v1 arm
        // (backfill) call.
        let mesh = MeshUpload {
            vertices: vec![PackedVertex::zeroed(); 2],
            indices: vec![0u32, 1],
        };
        let mut v1 = Vec::new();
        v1.extend_from_slice(MAGIC);
        v1.extend_from_slice(&1u32.to_le_bytes());
        v1.extend_from_slice(&(mesh.vertices.len() as u64).to_le_bytes());
        v1.extend_from_slice(&(mesh.indices.len() as u64).to_le_bytes());
        v1.extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
        v1.extend_from_slice(bytemuck::cast_slice(&mesh.indices));

        let (decoded, id) = decode(&v1).expect("v1 must still decode");
        assert_eq!(decoded.indices, mesh.indices);
        assert_eq!(id, content_id_for_bytes(&mesh));

        let v2 = encode(&mesh, id);
        let (_, id2) = decode(&v2).expect("v2 decode");
        assert_eq!(id2, id, "backfilled id matches a fresh v2 write of the same geometry");
    }

    #[test]
    fn different_geometry_never_collides_in_practice() {
        let a = MeshUpload { vertices: vec![PackedVertex::zeroed(); 3], indices: vec![0, 1, 2] };
        let b = MeshUpload { vertices: vec![PackedVertex::zeroed(); 3], indices: vec![0, 2, 1] };
        assert_ne!(content_id_for_bytes(&a), content_id_for_bytes(&b));
    }
}
