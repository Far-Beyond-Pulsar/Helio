//! What a static mesh instance uploads for drawing (Pulsar-Native#1035,
//! Phase 2).
//!
//! [`StaticMeshComponent`] mirrors its geometry itself (the content-interned
//! `builtin_mesh_vertex`/`builtin_mesh_index` pools). Everything else a
//! renderer needs to draw the instance is derived here from the authored
//! value, as a second GPU registration on the component: the local bounds,
//! the movable flag, and one [`MeshSectionDraw`] per material section with
//! the section's resolved material. SceneDB runs it on every insert,
//! guarded write, removal, despawn and mirror replay of the component, so the
//! rows always describe the current value; nothing calls it.
//!
//! The rows describe the instance alone. Placement (the owner object's
//! transform and visibility) and the instance's enabled state are separate
//! rows on other entities; the renderer's scene join combines them on the
//! GPU each frame they change.
//!
//! Material resolution lives here, with the component that names the
//! material assets: an explicit surface override, a scalar surface asset, a
//! compiled shader-graph folder (whose textures are registered in the
//! mirror's texture store), or the mesh's imported surface.

use pulsar_scenedb::gpu::GpuMirrorHandle;
use pulsar_scenedb_derive::SceneStore;

use super::{StaticMeshComponent, StaticMeshMaterialSlot, SurfaceMaterialAsset};
use crate::mesh_cache::{ImportedSurfaceMaterial, MeshSection};

/// Pool of every instance's [`MeshSectionDraw`]s; its handle table is
/// `static_mesh_draw_sections::handles`, keyed by the instance entity.
pub const MESH_SECTIONS_BUFFER: &str = "static_mesh_draw_sections";
/// Per instance: local bounding sphere, `[center.xyz, radius]`.
pub const MESH_BOUNDS_BUFFER: &str = "static_mesh_draw_bounds";
/// Per instance: object-row flags (`helio::INSTANCE_FLAG_MOVABLE`).
pub const MESH_FLAGS_BUFFER: &str = "static_mesh_draw_flags";

/// One drawable section of a mesh instance: an index range of the mesh's
/// own indices and the material it draws with.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshSectionDraw {
    /// First index of the section, relative to the mesh's index range.
    pub first_index: u32,
    pub index_count: u32,
    /// Shading class of the material (`helio_mats::MATERIAL_CLASS_*`).
    pub material_class: u32,
    pub graph_hash_lo: u32,
    pub graph_hash_hi: u32,
    pub _pad: [u32; 3],
    pub material: helio::GpuMaterial,
}

// SAFETY: `#[repr(C)]`, every field `Pod`, no padding beyond the explicit
// `_pad` (5 u32s + 3 u32s = 32 bytes ahead of a 16-byte-aligned material).
unsafe impl pulsar_scenedb::page::Pod for MeshSectionDraw {}

/// The derived per-instance draw data; see the module doc.
#[derive(SceneStore, Clone, Debug)]
pub struct StaticMeshDraw {
    #[gpu(buffer = "static_mesh_draw_bounds")]
    pub bounds_local: [f32; 4],
    #[gpu(buffer = "static_mesh_draw_flags")]
    pub flags: u32,
    #[gpu(buffer = "static_mesh_draw_sections")]
    pub sections: Vec<MeshSectionDraw>,
}

impl StaticMeshDraw {
    /// The draw data for `mesh`, resolving each section's material.
    /// `mirror` provides the texture store shader-graph materials register
    /// their textures in; without one those materials fall back to the
    /// imported surface.
    pub fn of(mesh: &StaticMeshComponent, mirror: Option<&GpuMirrorHandle>) -> Self {
        let sections: Vec<MeshSection> = if !mesh.mesh_sections.is_empty() {
            mesh.mesh_sections.clone()
        } else if mesh.indices.is_empty() {
            Vec::new()
        } else {
            vec![MeshSection {
                first_index: 0,
                index_count: mesh.indices.len() as u32,
                material_slot: 0,
            }]
        };
        let sections = sections
            .iter()
            .map(|section| {
                let material = resolve_slot_material(
                    mesh.material_slots.slots.get(section.material_slot as usize),
                    mirror,
                );
                MeshSectionDraw {
                    first_index: section.first_index,
                    index_count: section.index_count,
                    material_class: material.material_class,
                    graph_hash_lo: material.graph_hash as u32,
                    graph_hash_hi: (material.graph_hash >> 32) as u32,
                    _pad: [0; 3],
                    material: surface_material(&material.surface),
                }
            })
            .collect();
        Self {
            bounds_local: mesh.bounds_local,
            flags: if helio::Movability::from(mesh.movability).can_move() {
                helio::INSTANCE_FLAG_MOVABLE
            } else {
                0
            },
            sections,
        }
    }
}

fn static_mesh_draw_dispatch(
    mirror: &GpuMirrorHandle,
    row: u32,
    data: *const (),
    is_new_insert: bool,
) {
    // SAFETY: SceneDB reaches this only through `StaticMeshComponent`'s own
    // `ComponentId`, with a pointer to a live `StaticMeshComponent`.
    let mesh = unsafe { &*(data as *const StaticMeshComponent) };
    let draw = StaticMeshDraw::of(mesh, Some(mirror));
    pulsar_scenedb::gpu::write_derived_row(mirror, row, &draw, is_new_insert);
}

fn static_mesh_draw_clear(mirror: &GpuMirrorHandle, row: u32) {
    pulsar_scenedb::gpu::clear_derived_row::<StaticMeshDraw>(mirror, row);
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuMirrorRegistration {
        component_id: pulsar_scenedb::component_id::<StaticMeshComponent>,
        dispatch: static_mesh_draw_dispatch,
    }
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuClearRegistration {
        component_id: pulsar_scenedb::component_id::<StaticMeshComponent>,
        clear: static_mesh_draw_clear,
    }
}

/// A surface as a material row: the scalar PBR values, no textures.
/// `alpha < 1` selects the blended, transparent-only path (glass).
fn surface_material(surface: &ImportedSurfaceMaterial) -> helio::GpuMaterial {
    let alpha = surface.alpha.clamp(0.0, 1.0);
    let missing = helio::GpuMaterial::NO_TEXTURE;
    let mut flags = 0;
    if alpha < 1.0 {
        flags |= helio_mats::FLAG_ALPHA_BLEND | helio_mats::FLAG_TRANSPARENT_ONLY;
    }
    helio::GpuMaterial {
        base_color: [
            surface.base_color[0],
            surface.base_color[1],
            surface.base_color[2],
            alpha,
        ],
        emissive: [
            surface.emissive[0],
            surface.emissive[1],
            surface.emissive[2],
            surface.emissive_intensity,
        ],
        roughness_metallic: [surface.roughness, surface.metallic, 1.5, 0.5],
        tex_base_color: missing,
        tex_normal: missing,
        tex_roughness: missing,
        tex_emissive: missing,
        tex_occlusion: missing,
        workflow: 0,
        flags,
        material_class: 0,
        class_params: [0.0; 4],
    }
}

/// A slot's material as drawn: the surface values, and for a shader-graph
/// material its class and registered graph.
#[derive(Clone)]
pub struct ResolvedSlotMaterial {
    pub surface: ImportedSurfaceMaterial,
    pub material_class: u32,
    pub graph_hash: u64,
}

/// Resolve one slot's material. An empty slot or one whose assignment
/// cannot be loaded draws with the mesh's imported surface (with a warning
/// for the latter).
pub fn resolve_slot_material(
    slot: Option<&StaticMeshMaterialSlot>,
    mirror: Option<&GpuMirrorHandle>,
) -> ResolvedSlotMaterial {
    let imported = || ResolvedSlotMaterial {
        surface: slot.map_or_else(Default::default, |slot| {
            slot.surface_override.unwrap_or(slot.imported_surface)
        }),
        material_class: helio_mats::MATERIAL_CLASS_DEFAULT,
        graph_hash: 0,
    };
    let Some(slot) = slot else {
        return imported();
    };
    if let Some(override_surface) = slot.surface_override {
        return ResolvedSlotMaterial {
            surface: override_surface,
            material_class: helio_mats::MATERIAL_CLASS_DEFAULT,
            graph_hash: 0,
        };
    }
    if slot.material_asset.trim().is_empty() {
        return imported();
    }
    let Some(project_root) = engine_state::get_project_path() else {
        return imported();
    };
    let path = crate::subsystems::resolve_asset_path(
        std::path::Path::new(&project_root),
        &slot.material_asset,
    );
    let graph_file = if path.is_dir() {
        Some(path.join("shader_graph_save.json"))
    } else if path
        .file_name()
        .is_some_and(|name| name == "shader_graph_save.json")
    {
        Some(path.clone())
    } else {
        None
    };
    if let Some(graph_file) = graph_file.filter(|file| file.is_file()) {
        let compiled = match mirror {
            Some(mirror) => {
                graph_material_source(&graph_file, std::path::Path::new(&project_root), mirror)
            }
            None => Err("no GPU mirror to register the graph's textures with".to_string()),
        };
        match compiled {
            Ok((hash, source)) => {
                helio_mats::register_graph_source(hash, source);
                return ResolvedSlotMaterial {
                    surface: slot.imported_surface,
                    material_class: helio_mats::MATERIAL_CLASS_CUSTOM,
                    graph_hash: hash,
                };
            }
            Err(error) => {
                tracing::warn!(path = %graph_file.display(), %error, "could not compile Blueprint material graph; using imported FBX material")
            }
        }
    }
    let loaded = std::fs::read(&path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<SurfaceMaterialAsset>(&bytes).ok());
    match loaded {
        Some(material) if material.version == 1 => ResolvedSlotMaterial {
            surface: ImportedSurfaceMaterial {
                base_color: material.base_color,
                roughness: material.roughness,
                metallic: material.metallic,
                emissive: material.emissive_color,
                emissive_intensity: material.emissive_intensity,
                alpha: material.alpha,
            },
            material_class: helio_mats::MATERIAL_CLASS_DEFAULT,
            graph_hash: 0,
        },
        _ => {
            tracing::warn!(
                path = %path.display(),
                "static mesh material asset could not be loaded; using the imported FBX material"
            );
            imported()
        }
    }
}

fn graph_material_cache() -> &'static std::sync::Mutex<
    std::collections::HashMap<std::path::PathBuf, (u64, Result<(u64, String), String>)>,
> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<
            std::collections::HashMap<std::path::PathBuf, (u64, Result<(u64, String), String>)>,
        >,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn graph_texture_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, (u64, u32)>> {
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, (u64, u32)>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

fn register_graph_texture(path: &std::path::Path, mirror: &GpuMirrorHandle) -> Result<u32, String> {
    use std::hash::{Hash, Hasher};
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut hasher);
    let fingerprint = hasher.finish();
    let mut cache = graph_texture_cache()
        .lock()
        .map_err(|_| "graph texture cache lock poisoned".to_string())?;
    if let Some((cached_fingerprint, slot)) = cache.get(path) {
        if *cached_fingerprint == fingerprint {
            return Ok(*slot);
        }
    }
    let texture_store = mirror
        .texture_store()
        .ok_or_else(|| "SceneDB has no material texture store".to_string())?;
    let mut store = texture_store
        .write()
        .map_err(|_| "SceneDB texture store lock poisoned".to_string())?;
    if let Some((_, old_slot)) = cache.get(path) {
        let _ = store.unregister(*old_slot);
    }
    let decoded = image::load_from_memory(&bytes)
        .map_err(|error| format!("could not decode texture image: {error}"))?
        .to_rgba8();
    let (width, height) = decoded.dimensions();
    if width == 0 || height == 0 {
        return Err("texture image has empty dimensions".to_string());
    }
    let device = mirror.store().device_arc();
    let descriptor = wgpu::TextureDescriptor {
        label: Some("Blueprint Material Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    };
    let slot = store
        .register(&device, mirror.queue(), &descriptor, decoded.as_raw())
        .map_err(|error| format!("could not register texture in SceneDB: {error:?}"))?;
    cache.insert(path.to_path_buf(), (fingerprint, slot));
    Ok(slot)
}

fn graph_material_source(
    path: &std::path::Path,
    project_root: &std::path::Path,
    mirror: &GpuMirrorHandle,
) -> Result<(u64, String), String> {
    let bytes = std::fs::read(path).map_err(|error| error.to_string())?;
    use std::hash::{Hash, Hasher};
    let mut fingerprint_hasher = std::collections::hash_map::DefaultHasher::new();
    bytes.hash(&mut fingerprint_hasher);
    let fingerprint = fingerprint_hasher.finish();
    let result = (|| {
        let text = std::str::from_utf8(&bytes).map_err(|error| error.to_string())?;
        // Shader graph saves may carry a line comment before their JSON body.
        let json_start = text
            .find('{')
            .ok_or_else(|| "shader graph JSON object is missing".to_string())?;
        let document: serde_json::Value = serde_json::from_str(&text[json_start..])
            .map_err(|error| format!("invalid shader graph JSON: {error}"))?;
        let graph_value = document
            .get("main_graph")
            .ok_or_else(|| "shader graph asset has no main_graph".to_string())?;
        let mut texture_bindings = std::collections::HashMap::new();
        if let Some(nodes) = graph_value
            .get("nodes")
            .and_then(serde_json::Value::as_object)
        {
            for node in nodes.values().filter(|node| {
                matches!(
                    node.get("node_type").and_then(serde_json::Value::as_str),
                    Some("sample_texture" | "sample_texture_level" | "sample_texture_grad")
                )
            }) {
                let Some(asset) = node
                    .get("properties")
                    .and_then(|props| props.get("texture"))
                    .and_then(serde_json::Value::as_str)
                else {
                    continue;
                };
                let texture_path = if std::path::Path::new(asset).is_absolute() {
                    std::path::PathBuf::from(asset)
                } else {
                    project_root.join(asset)
                };
                let slot = register_graph_texture(&texture_path, mirror)
                    .map_err(|error| format!("texture '{}': {error}", texture_path.display()))?;
                texture_bindings.insert(asset.to_string(), slot);
            }
        }
        if let Ok(cache) = graph_material_cache().lock() {
            if let Some((cached_fingerprint, cached_result)) = cache.get(path) {
                if *cached_fingerprint == fingerprint {
                    return cached_result.clone();
                }
            }
        }
        let graph: psgc::GraphDescription = serde_json::from_value(graph_value.clone())
            .map_err(|error| format!("invalid main_graph: {error}"))?;
        let generated = psgc::compile_shader(&graph)
            .map_err(|error| format!("shader graph compile failed: {error}"))?;
        let snippet = adapt_graph_wgsl(&generated, &texture_bindings)?;
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        snippet.hash(&mut hasher);
        let hash = hasher.finish().max(1);
        Ok((hash, snippet))
    })();
    if let Ok(mut cache) = graph_material_cache().lock() {
        cache.insert(path.to_path_buf(), (fingerprint, result.clone()));
    }
    result
}

fn adapt_graph_wgsl(
    generated: &str,
    texture_bindings: &std::collections::HashMap<String, u32>,
) -> Result<String, String> {
    let mut source = generated.to_string();
    for (asset, slot) in texture_bindings {
        source = source.replace(asset, &format!("scene_textures[{slot}u]"));
    }
    bind_graph_texture_samplers(&mut source);
    if let Some(start) = source.find("struct Uniforms {") {
        let end = source[start..]
            .find("};")
            .map(|offset| start + offset + 2)
            .ok_or_else(|| "malformed PSGC Uniforms declaration".to_string())?;
        source.replace_range(start..end, "");
    }
    source = source.replace("@group(0) @binding(0) var<uniform> uniforms: Uniforms;", "");
    source = source.replace("uniforms.time", "0.0");
    source = source.replace("FragmentOutput", "PulsarGraphOutput");
    for location in 0..8 {
        source = source.replace(&format!("@location({location}) "), "");
    }

    let entry = source
        .find("@fragment\nfn fragment_main(")
        .or_else(|| source.find("@fragment\r\nfn fragment_main("))
        .ok_or_else(|| "PSGC output is not a fragment shader".to_string())?;
    let open = source[entry..]
        .find('{')
        .map(|offset| entry + offset)
        .ok_or_else(|| "malformed PSGC fragment entry point".to_string())?;
    let replacement = "fn pulsar_material_graph(input: VertexOutput) -> PulsarGraphOutput {\n    let frag_coord = input.clip_position;\n    let uv = input.tex_coords;\n    let normal = input.world_normal;\n    let world_pos = input.world_position;";
    source.replace_range(entry..=open, replacement);

    let body = r#"let graph_surface = pulsar_material_graph(input);
albedo = graph_surface.base_color;
// Preserve the connected RGBA color's alpha. The separate opacity input
// modulates coverage and defaults to 1 when it is not connected.
alpha = clamp(graph_surface.base_color.a, 0.0, 1.0) * clamp(graph_surface.opacity, 0.0, 1.0);
albedo.a = alpha;
roughness = clamp(graph_surface.roughness, 0.045, 1.0);
metallic = clamp(graph_surface.metallic, 0.0, 1.0);
ao = clamp(graph_surface.ambient_occlusion, 0.0, 1.0);
emissive = graph_surface.emissive_color.rgb * graph_surface.emissive_color.a;
let graph_normal_length = length(graph_surface.normal);
if graph_normal_length > 0.0001 { N = graph_surface.normal / graph_normal_length; }
specular_f0 = clamp(mix(vec3<f32>(0.04), albedo.rgb, metallic), vec3<f32>(0.0), vec3<f32>(0.999));"#;
    Ok(format!(
        "/*RADIANT_GRAPH_DECLARATIONS*/\n{source}\n/*RADIANT_GRAPH_BODY*/\n{body}"
    ))
}

fn bind_graph_texture_samplers(source: &mut String) {
    for function in [
        "textureSample(",
        "textureSampleLevel(",
        "textureSampleGrad(",
    ] {
        let mut search_from = 0;
        while let Some(relative) = source[search_from..].find(function) {
            let call_start = search_from + relative;
            let args_start = call_start + function.len();
            let Some(first_comma_rel) = source[args_start..].find(',') else {
                break;
            };
            let first_comma = args_start + first_comma_rel;
            let first_arg = source[args_start..first_comma]
                .trim()
                .trim_matches(['(', ')'])
                .trim();
            let Some(slot_text) = first_arg
                .strip_prefix("scene_textures[")
                .and_then(|value| value.strip_suffix(']'))
            else {
                search_from = first_comma + 1;
                continue;
            };
            let Some(second_comma_rel) = source[first_comma + 1..].find(',') else {
                break;
            };
            let second_comma = first_comma + 1 + second_comma_rel;
            source.replace_range(
                first_comma + 1..second_comma,
                &format!(" scene_samplers[{slot_text}]"),
            );
            search_from = second_comma + 1;
        }
    }
}
