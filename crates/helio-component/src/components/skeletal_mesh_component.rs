//! Runtime skeletal mesh instances and SceneDB asset bindings.
//!
//! The asset files are addressed by typed stable ids. SceneDB owns the loaded
//! mesh, skeleton and clip payloads; the component mirrors the vertex skin
//! stream and the current joint palette into the renderer's shared GPU scene.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use engine_class_derive::{
    engine_class, register_component_runtime, register_scene_props_applier,
    register_world_component,
};
use helio::PackedVertex;
use pulsar_reflection::{
    AnimationSequenceAssetKind, AssetKind, AssetRef, ScenePropsProjector, SkeletalMeshAssetKind,
    SkeletonAssetKind,
};
use pulsar_scenedb::{AssetId, AssetPayload, AssetView};
use skeletal_assets::diagnostics::Diagnostic;
use serde_json::Value;
use std::collections::HashMap;

use crate::components::{MeshGeometryKey, MeshSectionDraw, StaticMeshDraw};
use crate::mesh_cache::MeshSection;

#[path = "skeletal_notifies.rs"]
mod notifies;
pub use notifies::{
    AnimationNotifyCatalog, AnimationNotifyDiagnostic, AnimationNotifyEvent,
    SkeletalAnimationEvents, SkeletalAnimationEventsEventWriterExt, animation_notify_descriptor,
    animation_source_for_file, flush_animation_notifies, project_animation_notify_events,
    register_animation_notifies, register_loaded_animation_notifies,
};

pub const SKELETAL_MESH_ASSET_KIND: &str = "skmesh";
pub const SKELETON_ASSET_KIND: &str = "skeleton";
pub const ANIMATION_SEQUENCE_ASSET_KIND: &str = "animseq";
pub const SKELETAL_SKIN_VERTICES_BUFFER: &str = "skeletal_skin_vertices";
pub const SKELETAL_BONE_PALETTE_BUFFER: &str = "skeletal_bone_palette";
pub const SKELETAL_PREVIOUS_BONE_PALETTE_BUFFER: &str = "skeletal_previous_bone_palette";
pub const SKELETAL_SKIN_HANDLES_BUFFER: &str = "skeletal_skin_vertices::handles";
pub const SKELETAL_PALETTE_HANDLES_BUFFER: &str = "skeletal_bone_palette::handles";

/// Per-vertex four-influence skin data. Kept in a storage buffer parallel to
/// `builtin_mesh_vertex`, so static geometry keeps its established vertex ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinVertex {
    pub joints: [u32; 4],
    pub weights: [f32; 4],
}
unsafe impl pulsar_scenedb::page::Pod for SkinVertex {}

#[derive(Clone)]
pub struct LoadedSkeletalMesh(pub Arc<skeletal_assets::skmesh::SkeletalMeshAsset>);
#[derive(Clone)]
pub struct LoadedSkeleton(pub Arc<skeletal_assets::assets::SkeletonAsset>);
#[derive(Clone)]
pub struct LoadedAnimationSequence(pub Arc<skeletal_assets::assets::AnimSeqAsset>);

/// A skinned mesh placed in the scene. The authored links are typed UUIDs;
/// loaded data and GPU-ready geometry are derived and never serialized.
#[engine_class(
    category = "Rendering",
    default,
    clone,
    debug,
    serialize,
    deserialize,
    scene_store
)]
#[scene_store(collect_handles = crate::components::collect_skeletal_assets)]
pub struct SkeletalMeshComponent {
    pub mesh: Option<AssetRef<SkeletalMeshAssetKind>>,
    pub animation: Option<AssetRef<AnimationSequenceAssetKind>>,
    #[serde(default)]
    pub playing: bool,
    #[serde(default)]
    pub looping: bool,
    #[serde(default = "default_playback_rate")]
    pub playback_rate: f32,

    #[serde(skip)]
    pub mesh_asset_id: AssetId,
    /// The skeleton the mesh is bound to, read from the mesh header. The mesh
    /// is the only authority; the component never stores a skeleton ref.
    #[serde(skip)]
    pub mesh_skeleton: Option<AssetRef<SkeletonAssetKind>>,
    #[serde(skip)]
    pub skeleton_asset_id: AssetId,
    /// Repairable problems with the component's asset links, recomputed whenever
    /// they change. Never serialized.
    #[serde(skip)]
    pub diagnostics: Vec<Diagnostic>,
    /// A skeleton ref found in an old saved component; kept only to report a
    /// disagreement with the mesh.
    #[serde(skip)]
    legacy_skeleton: Option<AssetRef<SkeletonAssetKind>>,
    #[serde(skip)]
    pub animation_asset_id: AssetId,
    #[serde(skip)]
    pub loaded_mesh: Option<Arc<skeletal_assets::skmesh::SkeletalMeshAsset>>,
    #[serde(skip)]
    pub loaded_skeleton: Option<Arc<skeletal_assets::assets::SkeletonAsset>>,
    #[serde(skip)]
    pub loaded_animation: Option<Arc<skeletal_assets::assets::AnimSeqAsset>>,
    #[serde(skip)]
    pub elapsed_seconds: f32,
    #[serde(skip)]
    notify_playback: notifies::NotifyPlayback,
    /// Evaluated parent-composed transforms in skeleton-local space, ordered
    /// exactly like `loaded_skeleton.bones`. These are pre-inverse-bind and
    /// intended for runtime consumers such as articulated physics bodies.
    #[serde(skip)]
    pub bone_world_pose: Vec<[[f32; 4]; 4]>,
    #[serde(skip)]
    physics_pose_active: bool,
    #[serde(skip)]
    pub bounds_local: [f32; 4],
    #[serde(skip)]
    pub mesh_sections: Vec<MeshSection>,
    #[serde(skip)]
    pub geometry_key: MeshGeometryKey,

    /// The regular mesh vertex/index payloads are still consumed by Helio's
    /// GPU-driven scene join. Skin influences and the bone palette are read by
    /// the skinned G-buffer vertex path using these per-instance ranges.
    #[gpu(buffer = "builtin_mesh_vertex", mirror = Once, content_id = "geometry_key")]
    #[serde(skip)]
    pub vertices: Vec<PackedVertex>,
    #[gpu(buffer = "builtin_mesh_index", mirror = Once, content_id = "geometry_key")]
    #[serde(skip)]
    pub indices: Vec<u32>,
    #[gpu(buffer = "skeletal_skin_vertices", mirror = Once, content_id = "geometry_key")]
    #[serde(skip)]
    pub skin_vertices: Vec<SkinVertex>,
    #[serde(skip)]
    pub bone_palette: Vec<[[f32; 4]; 4]>,
    /// Current palette followed by the previous-frame palette in the same
    /// variable-length GPU range; SceneJoin exposes the half-size count.
    #[gpu(buffer = "skeletal_bone_palette")]
    #[serde(skip)]
    pub gpu_bone_palettes: Vec<[[f32; 4]; 4]>,
}

fn default_playback_rate() -> f32 {
    1.0
}

#[register_scene_props_applier]
impl ScenePropsProjector for SkeletalMeshComponent {
    const CLASS_NAME: &'static str = "SkeletalMeshComponent";
    fn apply_scene_props(props: &mut HashMap<String, Value>, _component_data: Option<&Value>) {
        props.remove("mesh");
        props.remove("skeleton");
        props.remove("animation");
    }
}

#[register_world_component(
    decode = decode_skeletal_mesh_component,
    property_written = skeletal_mesh_property_written
)]
impl SkeletalMeshComponent {}

impl SkeletalMeshComponent {
    /// Rebuild all resolved SceneDB references after a decode or authored edit.
    pub fn attach_assets(&mut self) {
        register_skeletal_mesh_assets();
        self.mesh_asset_id = asset_id_for_ref(self.mesh.as_ref());
        self.mesh_skeleton = mesh_skeleton_ref(self.mesh.as_ref());
        self.skeleton_asset_id = asset_id_for_ref(self.mesh_skeleton.as_ref());
        self.animation_asset_id = asset_id_for_ref(self.animation.as_ref());
        if self.mesh_asset_id.is_zero() {
            self.clear_mesh();
        }
        self.refresh_pose();
        self.refresh_diagnostics();
    }

    /// Advance playback and refresh the GPU joint palette. Called by the
    /// runtime component tick once the scene lifecycle wires this component.
    pub fn advance(&mut self, delta_seconds: f32) {
        if let Err(error) = self.advance_notifies(delta_seconds) {
            self.playing = false;
            tracing::warn!(%error, "could not advance skeletal animation");
        }
    }

    fn clear_mesh(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.skin_vertices.clear();
        self.bone_palette.clear();
        self.gpu_bone_palettes.clear();
        self.bone_world_pose.clear();
        self.mesh_sections.clear();
        self.bounds_local = [0.0, 0.0, 0.0, 0.5];
        self.geometry_key = MeshGeometryKey::NONE;
    }

    fn load_mesh(&mut self, asset: &skeletal_assets::skmesh::SkeletalMeshAsset) {
        self.clear_mesh();
        let Some(lod) = asset.lod0() else {
            return;
        };
        self.vertices = lod
            .vertices
            .iter()
            .map(|v| {
                let normal = glam::Vec3::from_array(v.normal).normalize_or_zero();
                PackedVertex::from_components(
                    v.position,
                    normal.to_array(),
                    v.uv,
                    [1.0, 0.0, 0.0],
                    1.0,
                )
            })
            .collect();
        self.indices = lod.indices.clone();
        self.skin_vertices = lod
            .vertices
            .iter()
            .map(|v| SkinVertex {
                joints: v.joints.map(u32::from),
                weights: v.weights,
            })
            .collect();
        self.mesh_sections = lod
            .sections
            .iter()
            .map(|s| MeshSection {
                first_index: s.first_index,
                index_count: s.index_count,
                material_slot: s.material_slot,
            })
            .collect();
        self.bounds_local = bounding_sphere(lod.vertices.iter().map(|v| v.position));
        // SceneDB's keyed mesh pools use this identity to make the geometry
        // resident exactly once per source mesh while poses change separately.
        let key = asset
            .id
            .to_string()
            .bytes()
            .fold(0xcbf29ce484222325u128, |h, b| {
                (h ^ u128::from(b)).wrapping_mul(0x100000001b3)
            })
            .max(1);
        self.geometry_key = MeshGeometryKey(key);
    }

    fn refresh_pose(&mut self) {
        let (Some(mesh), Some(skeleton)) = (&self.loaded_mesh, &self.loaded_skeleton) else {
            self.bone_palette.clear();
            self.gpu_bone_palettes.clear();
            self.bone_world_pose.clear();
            return;
        };
        if mesh
            .skeleton
            .as_ref()
            .is_some_and(|id| id.as_str() != skeleton.id.as_str())
        {
            tracing::warn!(mesh = %mesh.id, skeleton = %skeleton.id, "skeletal mesh and skeleton references do not match");
            self.bone_palette.clear();
            self.gpu_bone_palettes.clear();
            self.bone_world_pose.clear();
            return;
        }
        let skeleton = skeleton.skeleton();
        let clip = self
            .loaded_animation
            .as_ref()
            .filter(|animation| {
                animation.skeleton.as_ref().is_none_or(|id| {
                    id.as_str() == skeleton_id(self.mesh_skeleton.as_ref()).as_deref().unwrap_or("")
                })
            })
            .map(|animation| &animation.clip);
        let bind_pose = skeletal_assets::animation::AnimationClip::bind_pose();
        let clip = clip.unwrap_or(&bind_pose);
        let world = skeletal_assets::animation::evaluate_world_transforms(
            &skeleton,
            clip,
            self.elapsed_seconds,
        );
        self.bone_world_pose = skeleton
            .bones
            .iter()
            .map(|bone| {
                let matrix = world
                    .get(&bone.id)
                    .copied()
                    .unwrap_or(skeletal_assets::math::Mat4::IDENTITY);
                glam::Mat4::from_cols_array(&matrix.0).to_cols_array_2d()
            })
            .collect();
        let palette = skeletal_assets::skinning::joint_matrices(mesh, &world);
        self.set_palette(
            palette
                .iter()
                .map(|m| glam::Mat4::from_cols_array(&m.0).to_cols_array_2d())
                .collect(),
        );
    }

    /// Recompute [`Self::diagnostics`] from the resolved links.
    pub fn refresh_diagnostics(&mut self) {
        self.diagnostics = link_diagnostics(
            self.mesh.as_ref(),
            self.mesh_asset_id.is_zero(),
            self.mesh_skeleton.as_ref(),
            self.legacy_skeleton.as_ref(),
            self.loaded_animation.as_deref().and_then(|a| a.skeleton.as_ref()).map(|id| id.as_str().to_string()),
            self.animation.is_some(),
        );
        for d in &self.diagnostics {
            tracing::warn!(code = d.code, "{}", d.message);
        }
    }

    /// Replace the evaluated pose with articulated-body transforms in
    /// skeleton-local space, ordered like `loaded_skeleton.bones`. This also
    /// refreshes the GPU skin palette. Returns false if assets are not ready,
    /// the pose has the wrong number of bones, or any matrix is non-finite.
    /// While active, animation playback will not overwrite the physics pose.
    pub fn apply_physics_pose(&mut self, pose: Vec<[[f32; 4]; 4]>) -> bool {
        let (Some(mesh), Some(skeleton)) = (&self.loaded_mesh, &self.loaded_skeleton) else {
            return false;
        };
        if pose.len() != skeleton.bones.len()
            || pose
                .iter()
                .flatten()
                .flatten()
                .any(|value| !value.is_finite())
        {
            return false;
        }
        let world: HashMap<_, _> = skeleton
            .bones
            .iter()
            .zip(&pose)
            .map(|(bone, matrix)| {
                let flat = std::array::from_fn(|index| matrix[index / 4][index % 4]);
                (bone.id.clone(), skeletal_assets::math::Mat4(flat))
            })
            .collect();
        let palette = skeletal_assets::skinning::joint_matrices(mesh, &world);
        self.bone_world_pose = pose;
        self.set_palette(
            palette
                .iter()
                .map(|m| glam::Mat4::from_cols_array(&m.0).to_cols_array_2d())
                .collect(),
        );
        self.physics_pose_active = true;
        true
    }

    fn set_palette(&mut self, current: Vec<[[f32; 4]; 4]>) {
        let previous = if self.bone_palette.len() == current.len() {
            self.bone_palette.clone()
        } else {
            // New asset or first evaluation: no motion from an uninitialized pose.
            current.clone()
        };
        self.bone_palette = current;
        self.gpu_bone_palettes.clear();
        self.gpu_bone_palettes.extend_from_slice(&self.bone_palette);
        self.gpu_bone_palettes.extend_from_slice(&previous);
    }

    /// Return pose ownership to the animation player and refresh its current
    /// frame immediately.
    pub fn clear_physics_pose(&mut self) {
        self.physics_pose_active = false;
        self.refresh_pose();
    }
}

fn bounding_sphere(positions: impl Iterator<Item = [f32; 3]>) -> [f32; 4] {
    let points: Vec<_> = positions.map(glam::Vec3::from_array).collect();
    if points.is_empty() {
        return [0.0, 0.0, 0.0, 0.5];
    }
    let min = points
        .iter()
        .copied()
        .reduce(glam::Vec3::min)
        .unwrap_or(glam::Vec3::ZERO);
    let max = points
        .iter()
        .copied()
        .reduce(glam::Vec3::max)
        .unwrap_or(glam::Vec3::ZERO);
    let center = (min + max) * 0.5;
    let radius = points
        .iter()
        .map(|p| p.distance(center))
        .fold(0.0, f32::max);
    [center.x, center.y, center.z, radius]
}

/// Problems with a component's asset links, as repairable diagnostics.
///
/// `animation_skeleton` is the skeleton id of the loaded animation, if loaded.
pub(crate) fn link_diagnostics(
    mesh: Option<&AssetRef<SkeletalMeshAssetKind>>,
    mesh_unresolved: bool,
    mesh_skeleton: Option<&AssetRef<SkeletonAssetKind>>,
    legacy_skeleton: Option<&AssetRef<SkeletonAssetKind>>,
    animation_skeleton: Option<String>,
    has_animation: bool,
) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    if mesh.is_some() && mesh_unresolved {
        out.push(
            Diagnostic::error("component.mesh_unresolved", "The skeletal mesh is not in this project's asset index")
                .at("mesh")
                .repair("Pick the mesh again or restore the missing file"),
        );
    } else if mesh.is_some() && mesh_skeleton.is_none() {
        out.push(
            Diagnostic::error("component.mesh_without_skeleton", "The skeletal mesh names no skeleton, so it cannot be posed")
                .at("mesh")
                .repair("Bind the mesh to a skeleton in the Skeletal Mesh editor"),
        );
    }
    if let Some(legacy) = legacy_skeleton {
        if mesh_skeleton.is_some_and(|m| m != legacy) {
            out.push(
                Diagnostic::warning(
                    "component.legacy_skeleton_disagrees",
                    "This component was saved with a skeleton that differs from its mesh's; the mesh's skeleton is used and the old value is dropped on the next save",
                )
                .at("skeleton")
                .repair("Check that the mesh is the one you meant, then save the scene"),
            );
        }
    }
    if has_animation {
        if let (Some(animation), Some(mesh)) = (animation_skeleton, mesh_skeleton) {
            if animation != mesh.as_str() {
                out.push(
                    Diagnostic::error(
                        "component.animation_skeleton_mismatch",
                        "The animation is for a different skeleton than the mesh, so it is not played",
                    )
                    .at("animation")
                    .repair("Pick an animation made for the mesh's skeleton"),
                );
            }
        }
    }
    out
}

/// The skeleton the mesh at `mesh` is bound to, read from its header only.
fn mesh_skeleton_ref(mesh: Option<&AssetRef<SkeletalMeshAssetKind>>) -> Option<AssetRef<SkeletonAssetKind>> {
    let path = path_for_ref(mesh?)?;
    match skeletal_assets::skmesh::SkeletalMeshAsset::peek_header_at(&path) {
        Ok((_, skeleton)) => skeleton.map(|s| s.to_ref()),
        Err(error) => {
            tracing::warn!(%error, path = %path.display(), "could not read the skeletal mesh header");
            None
        }
    }
}

fn skeleton_id(reference: Option<&AssetRef<SkeletonAssetKind>>) -> Option<String> {
    reference.map(|reference| reference.as_str().to_owned())
}

/// The project file a typed reference resolves to, through the file index.
fn path_for_ref<K: AssetKind>(reference: &AssetRef<K>) -> Option<PathBuf> {
    let root = PathBuf::from(engine_state::get_project_path()?);
    let path = engine_fs::open_project_file_index(&root)
        .ok()
        .flatten()
        .and_then(|index| index.path_for_asset_id_kind(K::KIND, reference.as_str()).ok().flatten());
    let Some(path) = path else {
        tracing::warn!(kind = K::KIND, id = reference.as_str(), "skeletal asset id is not indexed in this project");
        return None;
    };
    let full_path = root.join(&path);
    Some(std::fs::canonicalize(&full_path).unwrap_or(full_path))
}

fn asset_id_for_ref<K: AssetKind>(reference: Option<&AssetRef<K>>) -> AssetId {
    let Some(path) = reference.and_then(path_for_ref) else {
        return AssetId::ZERO;
    };
    AssetId::declare(K::KIND, &path.to_string_lossy())
}

pub fn register_skeletal_mesh_assets() {
    static REGISTER: std::sync::Once = std::sync::Once::new();
    REGISTER.call_once(|| {
        pulsar_scenedb::register_loader(SKELETAL_MESH_ASSET_KIND, load_skeletal_mesh);
        pulsar_scenedb::register_loader(SKELETON_ASSET_KIND, load_skeleton);
        pulsar_scenedb::register_loader(ANIMATION_SEQUENCE_ASSET_KIND, load_animation_sequence);
        pulsar_scenedb::register_binding::<SkeletalMeshComponent>(
            SKELETAL_MESH_ASSET_KIND,
            bind_skeletal_mesh,
        );
        pulsar_scenedb::register_binding::<SkeletalMeshComponent>(
            SKELETON_ASSET_KIND,
            bind_skeleton,
        );
        pulsar_scenedb::register_binding::<SkeletalMeshComponent>(
            ANIMATION_SEQUENCE_ASSET_KIND,
            bind_animation_sequence,
        );
    });
}

fn load_skeletal_mesh(source: &str) -> Result<AssetPayload, String> {
    let bytes = engine_fs::virtual_fs::read_file(Path::new(source))
        .map_err(|e| format!("could not read skeletal mesh '{source}': {e}"))?;
    let mesh = skeletal_assets::skmesh::SkeletalMeshAsset::decode(&bytes)
        .map_err(|e| format!("could not decode skeletal mesh '{source}': {e}"))?;
    Ok(Arc::new(LoadedSkeletalMesh(Arc::new(mesh))))
}

fn load_skeleton(source: &str) -> Result<AssetPayload, String> {
    let skeleton = skeletal_assets::assets::load_skeleton_asset(Path::new(source))
        .map_err(|e| e.to_string())?;
    Ok(Arc::new(LoadedSkeleton(Arc::new(skeleton))))
}

fn load_animation_sequence(source: &str) -> Result<AssetPayload, String> {
    let animation = notifies::load_sequence_for_events(Path::new(source))?;
    Ok(Arc::new(LoadedAnimationSequence(Arc::new(animation))))
}

fn bind_skeletal_mesh(component: &mut SkeletalMeshComponent, view: &AssetView) {
    if view.id != component.mesh_asset_id {
        return;
    }
    if let Some(loaded) = view.payload::<LoadedSkeletalMesh>() {
        component.load_mesh(&loaded.0);
        component.loaded_mesh = Some(loaded.0.clone());
    } else {
        component.loaded_mesh = None;
        component.clear_mesh();
    }
    component.refresh_pose();
}

fn bind_skeleton(component: &mut SkeletalMeshComponent, view: &AssetView) {
    if view.id != component.skeleton_asset_id {
        return;
    }
    component.loaded_skeleton = view
        .payload::<LoadedSkeleton>()
        .map(|loaded| loaded.0.clone());
    component.refresh_pose();
}

fn bind_animation_sequence(component: &mut SkeletalMeshComponent, view: &AssetView) {
    if view.id != component.animation_asset_id {
        return;
    }
    component.loaded_animation = view
        .payload::<LoadedAnimationSequence>()
        .map(|loaded| loaded.0.clone());
    let _ = component.seek_animation(0.0);
}

/// Handle-ledger collection for the three stable references.
pub(crate) fn collect_skeletal_assets(
    component: &SkeletalMeshComponent,
    out: &mut Vec<pulsar_scenedb::handle_ledger::HandleId>,
) {
    register_skeletal_mesh_assets();
    for id in [
        asset_id_for_ref(component.mesh.as_ref()),
        asset_id_for_ref(mesh_skeleton_ref(component.mesh.as_ref()).as_ref()),
        asset_id_for_ref(component.animation.as_ref()),
    ] {
        out.push(id.handle());
    }
}

fn decode_skeletal_mesh_component(data: &Value) -> Result<SkeletalMeshComponent, String> {
    let mut component: SkeletalMeshComponent = pulsar_world_registry::decode_json(data)?;
    // Scenes saved before the mesh became the only skeleton authority carry a
    // `skeleton` ref. It is not part of the component any more; if it disagrees
    // with the mesh the user is told, and the next save drops it.
    component.legacy_skeleton = data
        .get("skeleton")
        .and_then(Value::as_str)
        .and_then(AssetRef::parse);
    component.attach_assets();
    Ok(component)
}

fn skeletal_mesh_property_written(component: &mut SkeletalMeshComponent, property: Option<&str>) {
    if matches!(
        property,
        None | Some("mesh") | Some("animation")
    ) {
        component.attach_assets();
    }
    component.refresh_diagnostics();
}

#[register_component_runtime]
impl SkeletalMeshComponent {
    fn begin_play(&mut self, _context: &mut pulsar_world_registry::ComponentContext) {
        self.begin_notify_playback();
    }

    fn tick(&mut self, context: &mut pulsar_world_registry::ComponentContext, delta_seconds: f32) {
        self.reconcile_notify_playback();
        if let Err(error) = self.register_notify_events(context.events.hub()) {
            tracing::warn!(%error, "could not register animation notifies");
            return;
        }
        // a failed suffix is retried in the next AfterScripts publication phase.
        if self.notify_playback.has_pending() {
            return;
        }
        self.advance(delta_seconds);
    }
}

fn skeletal_mesh_draw_dispatch(
    mirror: &pulsar_scenedb::gpu::GpuMirrorHandle,
    row: u32,
    data: *const (),
    is_new_insert: bool,
) {
    // SAFETY: SceneDB invokes the registration only for this component id.
    let component = unsafe { &*(data as *const SkeletalMeshComponent) };
    let sections = component
        .mesh_sections
        .iter()
        .map(|section| {
            let surface = crate::mesh_cache::ImportedSurfaceMaterial::default();
            MeshSectionDraw {
                first_index: section.first_index,
                index_count: section.index_count,
                material_class: 0,
                graph_hash_lo: 0,
                graph_hash_hi: 0,
                _pad: [0; 3],
                material: crate::components::surface_material(&surface, [u32::MAX; 5]),
            }
        })
        .collect::<Vec<_>>();
    let draw = StaticMeshDraw {
        bounds_local: component.bounds_local,
        flags: helio::INSTANCE_FLAG_MOVABLE,
        sections,
    };
    pulsar_scenedb::gpu::write_derived_row(mirror, row, &draw, is_new_insert);
}

fn skeletal_mesh_draw_clear(mirror: &pulsar_scenedb::gpu::GpuMirrorHandle, row: u32) {
    pulsar_scenedb::gpu::clear_derived_row::<StaticMeshDraw>(mirror, row);
}

pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuMirrorRegistration {
        component_id: pulsar_scenedb::component_id::<SkeletalMeshComponent>,
        dispatch: skeletal_mesh_draw_dispatch,
    }
}
pulsar_scenedb::pulsar_reflection::inventory::submit! {
    pulsar_scenedb::gpu::GpuClearRegistration {
        component_id: pulsar_scenedb::component_id::<SkeletalMeshComponent>,
        clear: skeletal_mesh_draw_clear,
    }
}

#[cfg(test)]
mod authority_tests {
    use super::*;

    const MESH: &str = "11111111-1111-4111-8111-111111111111";
    const SKELETON_A: &str = "aaaaaaaa-0000-4000-8000-00000000000a";
    const SKELETON_B: &str = "bbbbbbbb-0000-4000-8000-00000000000b";

    fn skeleton(id: &str) -> Option<AssetRef<SkeletonAssetKind>> {
        AssetRef::parse(id)
    }

    fn codes(d: &[Diagnostic]) -> Vec<&'static str> {
        d.iter().map(|d| d.code).collect()
    }

    #[test]
    fn the_component_has_no_authored_skeleton_and_old_values_do_not_come_back() {
        let old = serde_json::json!({
            "mesh": MESH,
            "skeleton": SKELETON_B,
            "animation": null,
            "playing": true,
            "looping": false,
            "playback_rate": 1.0
        });
        let component = decode_skeletal_mesh_component(&old).expect("old scenes still load");
        assert_eq!(component.mesh.as_ref().map(|m| m.as_str()), Some(MESH));
        let saved = serde_json::to_value(&component).unwrap();
        assert!(saved.get("skeleton").is_none(), "the skeleton is no longer written: {saved}");
        assert_eq!(saved["mesh"], MESH);
        // and a fresh component never writes one either
        assert!(serde_json::to_value(SkeletalMeshComponent::default()).unwrap().get("skeleton").is_none());
    }

    #[test]
    fn an_old_skeleton_that_disagrees_with_the_mesh_is_a_repairable_diagnostic() {
        let mesh: AssetRef<SkeletalMeshAssetKind> = AssetRef::parse(MESH).unwrap();
        let agree = link_diagnostics(Some(&mesh), false, skeleton(SKELETON_A).as_ref(), skeleton(SKELETON_A).as_ref(), None, false);
        assert!(agree.is_empty(), "{agree:?}");
        let disagree = link_diagnostics(Some(&mesh), false, skeleton(SKELETON_A).as_ref(), skeleton(SKELETON_B).as_ref(), None, false);
        assert_eq!(codes(&disagree), ["component.legacy_skeleton_disagrees"]);
        assert!(disagree[0].repair.is_some());
        assert_eq!(disagree[0].severity, skeletal_assets::issue::Severity::Warning);
        // no legacy value, no diagnostic
        assert!(link_diagnostics(Some(&mesh), false, skeleton(SKELETON_A).as_ref(), None, None, false).is_empty());
    }

    #[test]
    fn unresolved_meshes_meshes_without_skeletons_and_foreign_animations_are_reported() {
        let mesh: AssetRef<SkeletalMeshAssetKind> = AssetRef::parse(MESH).unwrap();
        assert_eq!(codes(&link_diagnostics(Some(&mesh), true, None, None, None, false)), ["component.mesh_unresolved"]);
        assert_eq!(codes(&link_diagnostics(Some(&mesh), false, None, None, None, false)), ["component.mesh_without_skeleton"]);
        let wrong = link_diagnostics(Some(&mesh), false, skeleton(SKELETON_A).as_ref(), None, Some(SKELETON_B.into()), true);
        assert_eq!(codes(&wrong), ["component.animation_skeleton_mismatch"]);
        let right = link_diagnostics(Some(&mesh), false, skeleton(SKELETON_A).as_ref(), None, Some(SKELETON_A.into()), true);
        assert!(right.is_empty());
        // no mesh at all is not an error: the component is simply empty
        assert!(link_diagnostics(None, true, None, None, None, false).is_empty());
    }
}
