//! Engine-native baked mesh assets (issues #391 / #409).
//!
//! Model import happens **at copy time**: a dropped source model
//! (fbx/obj/gltf/usd) is converted with the chosen import options into an
//! engine-native `.mesh` asset written into the project. **The source file
//! itself is not brought into the project** — only the native asset. Components
//! reference the `.mesh` asset and [`crate::subsystems::load_mesh_upload`] loads
//! it directly, with no per-load conversion or import options.
//!
//! Format (`PMSH`): a small header followed by the bytemuck-packed
//! [`PackedVertex`] and `u32` index arrays.
//!
//! **v2** (current) header layout, little-endian throughout:
//!
//! ```text
//! 0..4    magic "PMSH"
//! 4..8    version (= 2)
//! 8..16   vertex_count
//! 16..24  index_count
//! 24..40  content_id (u128 LE) -- the asset's [`ContentId`]
//! 40..    payload: PackedVertex[vertices] ++ u32[indices]
//! ```
//!
//! **v1** is the same file minus the `content_id` field (header ends at 24)
//! and remains DECODEABLE forever -- backward compatibility is a hard rule
//! here, version-gated exactly like the original decoder gated v1-only.
//! On decoding a v1 file the id is computed from the decoded payload
//! ([`content_id_for_payload`] -- byte-for-byte what a v2 writer would have
//! stored, since v1's payload region layout is identical), and the load
//! path best-effort REWRITES the file as v2 so identities converge on disk
//! too (`subsystems.rs` drives that rewrite; failure is logged, never
//! fatal -- a read-only project still loads fine, it just stays v1).
//!
//! Identity definition: an asset's `ContentId` is ALWAYS the XXH3-128
//! digest of its canonical payload bytes (`PackedVertex` array bytes ++
//! index array bytes), never of the container. That makes ids identical
//! across v1/v2 containers AND across import routes (a `.mesh` file and an
//! FBX that decode to identical geometry dedup to one identity).
//!
//! NOTE: only mesh geometry is baked today. Materials/textures from the source
//! scene are not yet written as native assets — that's a follow-up once the
//! engine's native material-asset format is wired in here.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};

use bytemuck::Zeroable;
use helio::{MeshUpload, PackedVertex};
use pulsar_reflection::{RuntimeTypeInfo, TypeStructure, RUNTIME_TYPE_REGISTRY};

use crate::content_id::{ContentId, ContentHasher};

const MAGIC: &[u8; 4] = b"PMSH";
const VERSION: u32 = 2;
const HEADER_V1: usize = 4 + 4 + 8 + 8; // magic + version + vertex_count + index_count
const HEADER_V2: usize = HEADER_V1 + 16; // ... + content_id

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

/// The canonical content identity of decoded mesh payload bytes. This is
/// THE definition every producer and consumer agrees on: v2 headers store
/// exactly this, v1 files get it computed at decode, and non-PMSH sources
/// (FBX etc.) get it computed from their decoded arrays -- so all routes to
/// identical geometry converge on one id. Two-part streaming update ==
/// hashing the concatenated region (pinned by ContentHasher's chunking
/// tests).
pub fn content_id_for_payload(vertices: &[PackedVertex], indices: &[u32]) -> ContentId {
    let mut hasher = ContentHasher::new();
    hasher.update(bytemuck::cast_slice(vertices));
    hasher.update(bytemuck::cast_slice(indices));
    hasher.finish()
}

/// Serialise a [`MeshUpload`] into the native `.mesh` byte format -- always
/// the CURRENT version (v2), with the content id computed from the payload.
/// Callers that already hold the id (dedup cache hits) should prefer
/// [`encode_with_id`] to skip the redundant hash.
pub fn encode(mesh: &MeshUpload) -> Vec<u8> {
    encode_with_id(mesh, content_id_for_payload(&mesh.vertices, &mesh.indices))
}

/// Same as [`encode`], with a caller-supplied identity. MUST equal
/// [`content_id_for_payload`] of this same mesh for the file to be honest;
/// enforced by a debug assertion (a mismatched id would silently split one
/// content into two identities downstream).
pub fn encode_with_id(mesh: &MeshUpload, content_id: ContentId) -> Vec<u8> {
    debug_assert_eq!(
        content_id,
        content_id_for_payload(&mesh.vertices, &mesh.indices),
        "caller-supplied content id does not match the payload"
    );
    let mut out = Vec::with_capacity(
        HEADER_V2
            + mesh.vertices.len() * std::mem::size_of::<PackedVertex>()
            + mesh.indices.len() * 4,
    );
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&(mesh.vertices.len() as u64).to_le_bytes());
    out.extend_from_slice(&(mesh.indices.len() as u64).to_le_bytes());
    out.extend_from_slice(&content_id.0.to_le_bytes());
    out.extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
    out.extend_from_slice(bytemuck::cast_slice(&mesh.indices));
    out
}

/// What [`decode_detailed`] produced: the mesh plus its authoritative
/// identity and which container version it came from.
#[derive(Debug, Clone)]
pub struct DecodedMesh {
    pub upload: MeshUpload,
    pub content_id: ContentId,
    /// Container version the bytes actually had (1 or 2). Callers use this
    /// to drive the best-effort v1→v2 rewrite; decode itself never rewrites
    /// (it has no path, only bytes).
    pub source_version: u32,
}

/// Parse a [`DecodedMesh`] from native `.mesh` bytes, or `None` if invalid /
/// a version or size mismatch (callers may fall back to converting a
/// source). Accepts BOTH container versions strictly:
///
/// - v2: the declared header id is AUTHORITATIVE (a corrupted payload must
///   not silently fork an existing identity mid-session); a mismatch with
///   the recomputed digest is logged loudly but does not reject the load.
/// - v1: no declared id exists; it is computed from the decoded payload.
///
/// Truncated files are REJECTED, never partially accepted: the length check
/// below covers the full declared vertex+index extent past the actual
/// header size, so a file cut short anywhere fails closed.
pub fn decode_detailed(bytes: &[u8]) -> Option<DecodedMesh> {
    if bytes.len() < HEADER_V1 || &bytes[0..4] != MAGIC {
        return None;
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().ok()?);
    let (header_len, declared_id) = match version {
        1 => (HEADER_V1, None),
        2 => {
            if bytes.len() < HEADER_V2 {
                return None;
            }
            (
                HEADER_V2,
                Some(ContentId(u128::from_le_bytes(bytes[24..40].try_into().ok()?))),
            )
        }
        // Strictly future-rejecting: an unknown NEWER version must not be
        // half-interpreted by old code.
        _ => return None,
    };

    let counts_start = 8;
    let vcount = u64::from_le_bytes(bytes[counts_start..counts_start + 8].try_into().ok()?) as usize;
    let icount =
        u64::from_le_bytes(bytes[counts_start + 8..counts_start + 16].try_into().ok()?) as usize;
    let vbytes = vcount.checked_mul(std::mem::size_of::<PackedVertex>())?;
    let ibytes = icount.checked_mul(4)?;
    let vstart = header_len;
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

    let recomputed = content_id_for_payload(&vertices, &indices);
    let content_id = match declared_id {
        Some(declared) => {
            if declared != recomputed {
                tracing::warn!(
                    "PMSH v2 file's declared content id {} does not match its payload ({}) -- \
                     trusting the DECLARED id (corrupted payload must not fork identity)",
                    declared,
                    recomputed
                );
            }
            declared
        }
        None => recomputed,
    };

    Some(DecodedMesh { upload: MeshUpload { vertices, indices }, content_id, source_version: version })
}

/// Backward-compatible entry point: parse just the mesh (v1 or v2), or
/// `None`. Identical acceptance rules to [`decode_detailed`].
pub fn decode(bytes: &[u8]) -> Option<MeshUpload> {
    decode_detailed(bytes).map(|d| d.upload)
}

/// Best-effort in-place upgrade of a v1 `.mesh` FILE to v2: decodes,
/// re-encodes with the computed id, writes through a temp file + rename
/// (atomic on the same volume -- a crash mid-upgrade leaves either the
/// intact v1 or the intact v2, never a torn file). Returns `false` when
/// anything went wrong (not v1, unreadable, read-only project, ...) --
/// callers treat that as purely informational; the asset still loads.
pub fn upgrade_v1_file(path: &Path) -> bool {
    let Ok(bytes) = std::fs::read(path) else {
        return false;
    };
    let Some(decoded) = decode_detailed(&bytes) else {
        return false;
    };
    if decoded.source_version != 1 {
        return false; // already v2 (or unknown): nothing to do
    }
    let upgraded = encode_with_id(&decoded.upload, decoded.content_id);

    let tmp = path.with_extension("mesh.upgrade-tmp");
    if std::fs::write(&tmp, &upgraded).is_err() {
        return false;
    }
    match std::fs::rename(&tmp, path) {
        Ok(()) => true,
        Err(_) => {
            let _ = std::fs::remove_file(&tmp);
            false
        }
    }
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

    std::fs::write(native, encode(&upload))
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

    fn sample_mesh() -> MeshUpload {
        let mut v = vec![PackedVertex::zeroed(); 5];
        for (i, vert) in v.iter_mut().enumerate() {
            // Arbitrary, deterministic non-zero bytes so hashing exercises
            // real content.
            vert.position = [i as f32, 2.0 * i as f32, 3.0 * i as f32];
        }
        MeshUpload { vertices: v, indices: vec![0, 1, 2, 3, 4] }
    }

    /// Hand-builds a legacy v1 container around `payload` -- the exact
    /// byte layout the old (pre-v2) writer produced.
    fn encode_v1(mesh: &MeshUpload) -> Vec<u8> {
        let mut out = Vec::with_capacity(HEADER_V1 + mesh.vertices.len() * std::mem::size_of::<PackedVertex>() + mesh.indices.len() * 4);
        out.extend_from_slice(MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&(mesh.vertices.len() as u64).to_le_bytes());
        out.extend_from_slice(&(mesh.indices.len() as u64).to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice(&mesh.vertices));
        out.extend_from_slice(bytemuck::cast_slice(&mesh.indices));
        out
    }

    #[test]
    fn native_path_uses_stem_and_dot_mesh() {
        assert_eq!(
            native_mesh_path(Path::new("/proj/models"), Path::new("/downloads/foo.fbx")),
            PathBuf::from("/proj/models/foo.mesh")
        );
    }

    #[test]
    fn encode_decode_roundtrip_v2() {
        let mesh = sample_mesh();
        let bytes = encode(&mesh);
        let decoded = decode_detailed(&bytes).expect("decode");
        assert_eq!(decoded.source_version, 2);
        assert_eq!(decoded.upload.vertices.len(), 5);
        assert_eq!(decoded.upload.indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(decoded.content_id, content_id_for_payload(&mesh.vertices, &mesh.indices));
    }

    #[test]
    fn truncated_and_garbage_input_is_rejected_never_panics() {
        let bytes = encode(&sample_mesh());
        // Truncations at every interesting boundary: inside magic, inside
        // header, mid-payload, one byte short of the end.
        for cut in [3usize, 10, HEADER_V1 + 8, bytes.len() - 1] {
            assert!(decode_detailed(&bytes[..cut]).is_none(), "truncation at {cut} was accepted");
        }
        assert!(decode(b"nope").is_none());
        assert!(decode_detailed(b"").is_none());
    }

    #[test]
    fn unknown_future_version_is_rejected_strictly() {
        let mut bytes = encode(&sample_mesh());
        bytes[4..8].copy_from_slice(&99u32.to_le_bytes());
        assert!(decode_detailed(&bytes).is_none(), "newer-than-known must fail closed");
    }

    #[test]
    fn v1_decodes_with_computed_id_matching_v2_of_same_content() {
        // THE backward-compat invariant: the same geometry in a v1
        // container and a v2 container must resolve to ONE identity.
        let mesh = sample_mesh();
        let v1_bytes = encode_v1(&mesh);
        let from_v1 = decode_detailed(&v1_bytes).expect("v1 decodes");
        assert_eq!(from_v1.source_version, 1);

        let expected = content_id_for_payload(&mesh.vertices, &mesh.indices);
        assert_eq!(from_v1.content_id, expected);

        let from_v2 = decode_detailed(&encode_with_id(&mesh, expected)).unwrap();
        assert_eq!(from_v1.content_id, from_v2.content_id, "v1 and v2 of identical content share identity");
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&from_v1.upload.vertices),
            bytemuck::cast_slice::<_, u8>(&from_v2.upload.vertices)
        );
    }

    #[test]
    fn declared_v2_id_is_trusted_over_recomputed_payload() {
        let mesh = sample_mesh();
        let fake = ContentId(0xDEAD_BEEF_CAFE_F00D_1234_5678_9ABC_DEF0);
        let mut bytes = encode(&mesh);
        // Overwrite the declared id with a lie.
        bytes[24..40].copy_from_slice(&fake.0.to_le_bytes());

        let decoded = decode_detailed(&bytes).unwrap();
        assert_eq!(decoded.content_id, fake, "declared id is authoritative even when it lies");
    }

    #[test]
    fn upgrade_v1_file_rewrites_atomically_to_v2() {
        let dir = std::env::temp_dir().join(format!("helio-pmsh-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("upgrade.mesh");
        let mesh = sample_mesh();

        std::fs::write(&path, encode_v1(&mesh)).unwrap();
        assert!(upgrade_v1_file(&Path::new(&path)), "upgrade succeeds");

        let upgraded_bytes = std::fs::read(&path).unwrap();
        let decoded = decode_detailed(&upgraded_bytes).unwrap();
        assert_eq!(decoded.source_version, 2);
        assert_eq!(decoded.content_id, content_id_for_payload(&mesh.vertices, &mesh.indices));

        // No temp litter left behind.
        assert!(!path.with_extension("mesh.upgrade-tmp").exists());

        // Upgrading an already-v2 file is a no-op reporting false.
        assert!(!upgrade_v1_file(Path::new(&path)));

        // Upgrading garbage reports false without touching the file.
        let junk = dir.join("junk.mesh");
        std::fs::write(&junk, b"garbage").unwrap();
        assert!(!upgrade_v1_file(Path::new(&junk)));
        assert_eq!(std::fs::read(&junk).unwrap(), b"garbage");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn empty_mesh_round_trips() {
        let empty = MeshUpload { vertices: Vec::new(), indices: Vec::new() };
        let bytes = encode(&empty);
        let decoded = decode_detailed(&bytes).unwrap();
        assert!(decoded.upload.vertices.is_empty() && decoded.upload.indices.is_empty());
        assert_eq!(decoded.content_id, content_id_for_payload(&[], &[]));
        assert_eq!(bytes.len(), HEADER_V2, "empty payload is header-only");
    }
}
