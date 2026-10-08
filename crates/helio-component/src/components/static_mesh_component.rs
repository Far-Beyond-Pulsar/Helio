//! Static mesh component for mesh asset assignment.

use engine_class_derive::{
    engine_class, register_scene_props_applier, register_world_component,
};
use helio::PackedVertex;
use pulsar_reflection::{ReflectError, ScenePropsProjector};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::asset_component::AssetComponentRegistration;
use crate::mesh_cache::{MeshMaterialSlot, MeshSection};
use crate::subsystems::{load_mesh_asset_upload, resolve_asset_path};

pulsar_reflection::inventory::submit! {
    AssetComponentRegistration {
        asset_kind: plugin_editor_api::AssetKind::Mesh,
        class_name: "StaticMeshComponent",
        value_for: |path| Box::new(StaticMeshComponent::for_mesh_asset(path)),
    }
}
// Mat4/Quat/Vec3 used to build the transform passed to sync_mesh_object.

// ── MeshAssetPath ─────────────────────────────────────────────────────────────

/// Strongly-typed wrapper for mesh asset paths.
///
/// Using this as a field type causes the reflection property inspector to render
/// a mesh-asset search browser (via `MeshAssetPicker`) instead of a plain text box.
///
/// Serialises transparently as a JSON string so existing scene files require no
/// migration.
///
/// # Example
///
/// ```ignore
/// #[property]
/// pub mesh_asset: MeshAssetPath,
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MeshAssetPath(pub String);

impl MeshAssetPath {
    /// Create a new `MeshAssetPath` from any string-like value.
    pub fn new(path: impl Into<String>) -> Self {
        Self(path.into())
    }

    /// Borrow the inner path string.
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Returns `true` if the path is empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// SceneDB's content-identity seam (Pulsar-Native#632/#659): lets
/// `vertices`/`indices` intern their GPU-resident geometry by content
/// instead of allocating one private copy per entity — see those fields'
/// `#[gpu(content_id = "mesh_asset")]` attribute and
/// `pulsar_scenedb::handle_ledger::ContentAddressed`'s own doc for the
/// mechanism this drives. Resolution: an empty path is
/// `HandleId::ZERO` (no asset, opts out of interning, matches every other
/// zero-value convention in this codebase); otherwise resolves the
/// project-relative path exactly like `decode_static_mesh_component`
/// already does and defers to `mesh_cache::content_id_for_path` (native
/// `.mesh` v2: a header read; anything else: a canonical-path + mtime/size
/// memoized hash — see that fn's own doc for why this converges path
/// aliases onto one id and mints a new one on a real edit). No project path
/// available, or the file can't be read at all, ALSO falls back to
/// `HandleId::ZERO` — a dangling/unresolvable reference behaves like "no
/// asset" for interning purposes rather than panicking or erroring; the
/// existing hydrate-time tolerance for a missing mesh already covers the
/// user-visible side of this (empty `vertices`/`indices`).
impl pulsar_scenedb::handle_ledger::ContentAddressed for MeshAssetPath {
    fn content_id(&self) -> pulsar_scenedb::handle_ledger::HandleId {
        use pulsar_scenedb::handle_ledger::HandleId;

        let path = self.0.trim();
        if path.is_empty() {
            return HandleId::ZERO;
        }
        let Some(project_root) = engine_state::get_project_path() else {
            return HandleId::ZERO;
        };
        let abs_path =
            crate::subsystems::resolve_asset_path(std::path::Path::new(&project_root), path);
        crate::mesh_cache::content_id_for_path(&abs_path)
            .map(HandleId)
            .unwrap_or(HandleId::ZERO)
    }
}

impl std::fmt::Display for MeshAssetPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for MeshAssetPath {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for MeshAssetPath {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// One component-level material choice corresponding to a mesh asset slot.
/// `source_material` is a stable matching key for reimport reconciliation;
/// it is not a GPU material index.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StaticMeshMaterialSlot {
    pub source_material: Option<u32>,
    pub name: String,
    #[serde(default)]
    pub material_asset: String,
    #[serde(default)]
    pub imported_surface: crate::mesh_cache::ImportedSurfaceMaterial,
    #[serde(default)]
    pub surface_override: Option<crate::mesh_cache::ImportedSurfaceMaterial>,
}

/// Temporary persistence adapter for levels authored with the deprecated
/// `MaterialOverrideComponent`. Scene loading migrates this into each mesh
/// slot and clears it before the component is saved again.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct LegacyMaterialOverrideData {
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive_color: [f32; 3],
    pub emissive_intensity: f32,
    pub alpha: f32,
}

impl LegacyMaterialOverrideData {
    pub fn into_surface(self) -> crate::mesh_cache::ImportedSurfaceMaterial {
        crate::mesh_cache::ImportedSurfaceMaterial {
            base_color: self.base_color,
            roughness: self.roughness,
            metallic: self.metallic,
            emissive: self.emissive_color,
            emissive_intensity: self.emissive_intensity,
            alpha: self.alpha,
        }
    }
}

/// Editor-facing dynamic list of per-slot material assignments. Kept as one
/// reflected value so a custom property editor can render a named picker for
/// every imported slot instead of exposing an unhelpful raw vector editor.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct StaticMeshMaterialSlots {
    pub slots: Vec<StaticMeshMaterialSlot>,
}

/// Small scalar-PBR surface document supported by mesh slots alongside
/// compiled shader-graph material folders.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SurfaceMaterialAsset {
    pub version: u32,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub emissive_color: [f32; 3],
    pub emissive_intensity: f32,
    pub alpha: f32,
}

impl Default for SurfaceMaterialAsset {
    fn default() -> Self {
        Self {
            version: 1,
            base_color: [0.22, 0.15, 0.08, 1.0],
            metallic: 0.0,
            roughness: 0.7,
            emissive_color: [0.0; 3],
            emissive_intensity: 0.0,
            alpha: 1.0,
        }
    }
}

impl From<crate::mesh_cache::ImportedSurfaceMaterial> for SurfaceMaterialAsset {
    fn from(value: crate::mesh_cache::ImportedSurfaceMaterial) -> Self {
        Self {
            base_color: value.base_color,
            metallic: value.metallic,
            roughness: value.roughness,
            emissive_color: value.emissive,
            emissive_intensity: value.emissive_intensity,
            alpha: value.alpha,
            ..Self::default()
        }
    }
}

// ── Reflection registration ───────────────────────────────────────────────────

fn serialize_mesh_asset_path_json(
    value: &MeshAssetPath,
) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    Ok(serde_json::json!(value.0))
}

fn deserialize_mesh_asset_path_json(
    value: serde_json::Value,
) -> pulsar_reflection::ReflectResult<MeshAssetPath> {
    value
        .as_str()
        .map(|s| MeshAssetPath(s.to_string()))
        .ok_or_else(|| ReflectError::TypeMismatch {
            expected: "MeshAssetPath",
            found: format!("{:?}", value),
        })
}

// ── MeshAssetPath property editor ─────────────────────────────────────────────

/// Engine primitives that are always offered, even in an empty project.
const BUILTIN_MESHES: &[&str] = &[
    "meshes/primitives/SM_Cube.fbx",
    "meshes/primitives/SM_Sphere.fbx",
    "meshes/primitives/SM_Cylinder.fbx",
    "meshes/primitives/SM_Plane.fbx",
    "meshes/primitives/SM_Torus.fbx",
];

/// Property editor for [`MeshAssetPath`] — a searchable mesh-asset browser.
///
/// Owns its [`MeshAssetPicker`](ui_common::asset_picker::MeshAssetPicker) child
/// entity and the subscription that turns a pick into a write-back.
pub struct MeshAssetEditor {
    label: String,
    id_prefix: String,
    prop_name: String,
    picker: gpui::Entity<ui_common::asset_picker::MeshAssetPicker>,
    path: String,
    write_back: pulsar_reflection::PropertyWriteBack,
    _subs: Vec<gpui::Subscription>,
}

impl MeshAssetEditor {
    fn new(
        args: &pulsar_reflection::PropertyEditorArgs<'_>,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> Self {
        use gpui::AppContext as _;
        use ui_common::asset_picker::{AssetPickedEvent, AssetQuery, MeshAssetPicker};

        let path = args
            .current_value
            .downcast_ref::<MeshAssetPath>()
            .map(|p| p.0.clone())
            .unwrap_or_default();

        let project_root = engine_state::get_project_path().map(std::path::PathBuf::from);
        let queries = vec![
            AssetQuery::extension("mesh"),
            AssetQuery::extension("fbx"),
            AssetQuery::extension("gltf"),
            AssetQuery::extension("glb"),
            AssetQuery::extension("obj"),
        ];

        let picker = cx.new(|cx| {
            MeshAssetPicker::new(
                path.clone(),
                BUILTIN_MESHES.iter().map(|s| s.to_string()).collect(),
                project_root,
                queries,
                window,
                cx,
            )
        });

        let subs = vec![cx.subscribe_in(
            &picker,
            window,
            |this: &mut Self, picker, _event: &AssetPickedEvent, window, cx| {
                let selected = picker.read(cx).selected_path().to_string();
                if this.path == selected {
                    return;
                }
                this.path = selected.clone();
                (this.write_back)(Box::new(MeshAssetPath(selected)), window, cx);
                cx.notify();
            },
        )];

        Self {
            label: args.display_name.to_string(),
            id_prefix: args.id_prefix.to_string(),
            prop_name: args.prop_name.to_string(),
            picker,
            path,
            write_back: args.write_back.clone(),
            _subs: subs,
        }
    }

    /// Accept a mesh assigned elsewhere — e.g. dropped straight onto the
    /// viewport, which writes `mesh_asset` without going through this row.
    fn set_value(&mut self, path: &MeshAssetPath, cx: &mut gpui::Context<Self>) {
        if self.path == path.0 {
            return;
        }
        self.path = path.0.clone();
        self.picker.update(cx, |picker, _| {
            picker.set_selected_path(path.0.clone());
        });
        cx.notify();
    }
}

impl gpui::Render for MeshAssetEditor {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use gpui::prelude::*;
        use ui::button::{Button, ButtonVariants as _};
        use ui::{ActiveTheme, Sizable, h_flex, popover::Popover};

        let display = if self.path.is_empty() {
            "No mesh selected".to_string()
        } else {
            std::path::Path::new(&self.path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(&self.path)
                .to_string()
        };

        let picker = self.picker.clone();

        h_flex()
            .w_full()
            .justify_between()
            .items_center()
            .gap_2()
            .py_1()
            .child(
                gpui::div()
                    .text_sm()
                    .text_color(cx.theme().muted_foreground)
                    .child(self.label.clone()),
            )
            .child(
                Popover::<ui_common::asset_picker::MeshAssetPicker>::new(format!(
                    "mesh-asset-picker-{}-{}",
                    self.id_prefix, self.prop_name
                ))
                .anchor(gpui::Corner::BottomRight)
                .trigger(
                    Button::new(format!(
                        "mesh-asset-picker-btn-{}-{}",
                        self.id_prefix, self.prop_name
                    ))
                    .label(display)
                    .small()
                    .ghost()
                    .dropdown_caret(true),
                )
                .content(move |_window, _cx| picker.clone()),
            )
    }
}

fn mesh_asset_editor(
    args: &pulsar_reflection::PropertyEditorArgs<'_>,
    window: &mut gpui::Window,
    cx: &mut gpui::App,
) -> pulsar_reflection::BoundPropertyEditor {
    use gpui::AppContext as _;

    let entity = cx.new(|cx| MeshAssetEditor::new(args, window, cx));
    pulsar_reflection::BoundPropertyEditor::new(
        entity,
        |editor: &mut MeshAssetEditor, value: &MeshAssetPath, _window, cx| {
            editor.set_value(value, cx)
        },
    )
}

/// Register `MeshAssetPath` with the reflection system.
///
/// `structure = String` makes `type_info.is_string()` return `true`, so the
/// type round-trips through the JSON codec as a plain string; the mesh-browser
/// UI comes from the `editor` registration above.
#[pulsar_reflection::pulsar_type(
    serialize_json_with = serialize_mesh_asset_path_json,
    deserialize_json_with = deserialize_mesh_asset_path_json,
    editor = mesh_asset_editor
)]
#[allow(dead_code)]
type RegisteredMeshAssetPath = MeshAssetPath;

fn serialize_material_slots_json(
    value: &StaticMeshMaterialSlots,
) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    serde_json::to_value(value)
        .map_err(|error| ReflectError::SerializationFailed(error.to_string()))
}

fn deserialize_material_slots_json(
    value: serde_json::Value,
) -> pulsar_reflection::ReflectResult<StaticMeshMaterialSlots> {
    serde_json::from_value(value)
        .map_err(|error| ReflectError::DeserializationFailed(error.to_string()))
}

struct MaterialSlotPickerRow {
    key: (Option<u32>, usize),
    name: String,
    path: String,
    picker: gpui::Entity<ui_common::asset_picker::MeshAssetPicker>,
}

struct StaticMeshMaterialSlotsEditor {
    label: String,
    id_prefix: String,
    prop_name: String,
    value: StaticMeshMaterialSlots,
    rows: Vec<MaterialSlotPickerRow>,
    write_back: pulsar_reflection::PropertyWriteBack,
    subscriptions: Vec<gpui::Subscription>,
}

impl StaticMeshMaterialSlotsEditor {
    fn new(
        args: &pulsar_reflection::PropertyEditorArgs<'_>,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> Self {
        let mut editor = Self {
            label: args.display_name.to_owned(),
            id_prefix: args.id_prefix.to_owned(),
            prop_name: args.prop_name.to_owned(),
            value: args
                .current_value
                .downcast_ref::<StaticMeshMaterialSlots>()
                .cloned()
                .unwrap_or_default(),
            rows: Vec::new(),
            write_back: args.write_back.clone(),
            subscriptions: Vec::new(),
        };
        let initial = editor.value.clone();
        editor.rebuild_rows(&initial, window, cx);
        editor
    }

    fn rebuild_rows(
        &mut self,
        value: &StaticMeshMaterialSlots,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) {
        use gpui::AppContext as _;
        use ui_common::asset_picker::{AssetPickedEvent, AssetQuery, MeshAssetPicker};

        self.rows.clear();
        self.subscriptions.clear();
        self.value = value.clone();
        let project_root = engine_state::get_project_path().map(std::path::PathBuf::from);
        let queries = vec![
            AssetQuery::extension("mat"),
            AssetQuery::folder_marker("shader_graph_save.json"),
        ];
        for (index, slot) in value.slots.iter().enumerate() {
            let key = (slot.source_material, index);
            let picker = cx.new(|cx| {
                MeshAssetPicker::new(
                    slot.material_asset.clone(),
                    Vec::new(),
                    project_root.clone(),
                    queries.clone(),
                    window,
                    cx,
                )
            });
            self.subscriptions.push(cx.subscribe_in(
                &picker,
                window,
                move |this: &mut Self, picker, _event: &AssetPickedEvent, window, cx| {
                    let selected = picker.read(cx).selected_path().to_owned();
                    this.set_slot_material(key, selected, window, cx);
                },
            ));
            self.rows.push(MaterialSlotPickerRow {
                key,
                name: if slot.name.is_empty() {
                    format!("Material Slot {}", index + 1)
                } else {
                    slot.name.clone()
                },
                path: slot.material_asset.clone(),
                picker,
            });
        }
        cx.notify();
    }

    fn set_slot_material(
        &mut self,
        key: (Option<u32>, usize),
        path: String,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) {
        if let Some(slot) = self
            .value
            .slots
            .get_mut(key.1)
            .filter(|slot| slot.source_material == key.0)
        {
            if slot.material_asset == path {
                return;
            }
            slot.material_asset = path.clone();
            if let Some(row) = self.rows.get_mut(key.1) {
                row.path = path;
            }
            (self.write_back)(Box::new(self.value.clone()), window, cx);
            cx.notify();
        }
    }

    fn create_material_asset(
        &mut self,
        key: (Option<u32>, usize),
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) {
        let Some(project_root) = engine_state::get_project_path().map(std::path::PathBuf::from)
        else {
            return;
        };
        let Some(slot) = self
            .value
            .slots
            .get(key.1)
            .filter(|slot| slot.source_material == key.0)
            .cloned()
        else {
            return;
        };
        let safe_name: String = slot
            .name
            .chars()
            .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
            .collect();
        let directory = project_root.join("materials");
        if let Err(error) = std::fs::create_dir_all(&directory) {
            tracing::warn!(%error, path = %directory.display(), "could not create materials directory");
            return;
        }
        let stem = if safe_name.trim_matches('_').is_empty() {
            format!("Material_{}", key.1 + 1)
        } else {
            safe_name
        };
        let (path, relative) = (0..)
            .map(|suffix| {
                let file_name = if suffix == 0 {
                    format!("{stem}.mat")
                } else {
                    format!("{stem}_{suffix}.mat")
                };
                (directory.join(&file_name), format!("materials/{file_name}"))
            })
            .find(|(path, _)| !path.exists())
            .expect("unbounded unique material filename search");
        let material = SurfaceMaterialAsset::from(slot.imported_surface);
        let Ok(bytes) = serde_json::to_vec_pretty(&material) else {
            return;
        };
        if let Err(error) = std::fs::write(&path, bytes) {
            tracing::warn!(%error, path = %path.display(), "could not write surface material asset");
            return;
        }
        self.set_slot_material(key, relative.clone(), window, cx);
        if let Some(row) = self.rows.get(key.1) {
            row.picker.update(cx, |picker, _| picker.set_selected_path(relative));
        }
    }

    fn set_value(
        &mut self,
        value: &StaticMeshMaterialSlots,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) {
        let same_slots = self.value.slots.len() == value.slots.len()
            && self
                .value
                .slots
                .iter()
                .zip(&value.slots)
                .all(|(a, b)| a.source_material == b.source_material && a.name == b.name);
        if !same_slots {
            self.rebuild_rows(value, window, cx);
            return;
        }
        self.value = value.clone();
        for (row, slot) in self.rows.iter_mut().zip(&value.slots) {
            if row.path != slot.material_asset {
                row.path = slot.material_asset.clone();
                row.picker.update(cx, |picker, _| {
                    picker.set_selected_path(slot.material_asset.clone())
                });
            }
        }
        cx.notify();
    }
}

impl gpui::Render for StaticMeshMaterialSlotsEditor {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use gpui::prelude::*;
        use ui::button::{Button, ButtonVariants as _};
        use ui::{ActiveTheme, Sizable, h_flex, popover::Popover, v_flex};

        let mut rows = v_flex().w_full().gap_1().child(
            gpui::div()
                .text_sm()
                .text_color(cx.theme().muted_foreground)
                .child(self.label.clone()),
        );
        if self.rows.is_empty() {
            rows = rows.child(
                gpui::div()
                    .text_xs()
                    .text_color(cx.theme().muted_foreground)
                    .child("Select a mesh to load its material slots"),
            );
        }
        for (index, row) in self.rows.iter().enumerate() {
            let picker = row.picker.clone();
            let row_key = row.key;
            let display = if row.path.is_empty() {
                "Use imported material".to_owned()
            } else {
                std::path::Path::new(&row.path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(&row.path)
                    .to_owned()
            };
            rows = rows.child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .items_center()
                    .gap_2()
                    .child(gpui::div().text_xs().child(row.name.clone()))
                    .child(
                        Popover::<ui_common::asset_picker::MeshAssetPicker>::new(format!(
                            "material-slot-picker-{}-{}-{index}",
                            self.id_prefix, self.prop_name
                        ))
                        .anchor(gpui::Corner::BottomRight)
                        .trigger(
                            Button::new(format!(
                                "material-slot-picker-btn-{}-{index}",
                                self.id_prefix
                            ))
                            .label(display)
                            .small()
                            .ghost()
                            .dropdown_caret(true),
                        )
                        .content(move |_window, _cx| picker.clone()),
                    )
                    .child(Button::new(format!(
                            "create-material-{}-{}-{index}",
                            self.id_prefix, self.prop_name
                        ))
                        .label("Create")
                        .on_click(cx.listener(move |this, _, window, cx| {
                            this.create_material_asset(row_key, window, cx);
                        }))),
            );
        }
        rows
    }
}

fn material_slots_editor(
    args: &pulsar_reflection::PropertyEditorArgs<'_>,
    window: &mut gpui::Window,
    cx: &mut gpui::App,
) -> pulsar_reflection::BoundPropertyEditor {
    use gpui::AppContext as _;
    let entity = cx.new(|cx| StaticMeshMaterialSlotsEditor::new(args, window, cx));
    pulsar_reflection::BoundPropertyEditor::new(
        entity,
        |editor: &mut StaticMeshMaterialSlotsEditor,
         value: &StaticMeshMaterialSlots,
         window,
         cx| { editor.set_value(value, window, cx) },
    )
}

#[pulsar_reflection::pulsar_type(
    serialize_json_with = serialize_material_slots_json,
    deserialize_json_with = deserialize_material_slots_json,
    editor = material_slots_editor
)]
#[allow(dead_code)]
type RegisteredStaticMeshMaterialSlots = StaticMeshMaterialSlots;

// ── StaticMeshComponent ───────────────────────────────────────────────────────

/// Attaches a mesh asset to a scene object.
///
/// `scene_store` (Pulsar-Native#561 Phase D): opts this struct into
/// `#[gpu]`-mirrored fields via `#[engine_class]`'s delegation to
/// `pulsar_scenedb::SceneStore` -- see `vertices`/`indices` below, and
/// `decode_static_mesh_component`'s doc for how they get populated. A
/// `#[gpu] Vec<T>` field routes through SceneDB's variable-length codegen
/// path, which implies no `Copy`/`Pod` requirement on this struct (see
/// `engine_class_derive`'s `struct_has_gpu_vec_field` check) -- unlike
/// every OTHER `scene_store` struct so far, which are all plain fixed-size
/// `Pod` rows.
#[engine_class(
    category = "Rendering",
    default,
    clone,
    debug,
    serialize,
    deserialize,
    scene_store
)]
pub struct StaticMeshComponent {
    /// Relative asset path to the mesh file (e.g. "meshes/primitives/SM_Cube.fbx").
    ///
    /// Typed as [`MeshAssetPath`] so the property inspector renders a mesh-asset
    /// search browser instead of a plain text input.
    #[property]
    pub mesh_asset: MeshAssetPath,

    /// Material assets assigned to the imported mesh's named material slots.
    /// Empty assignments use the mesh's imported/default material.
    #[property]
    #[serde(default)]
    pub material_slots: StaticMeshMaterialSlots,

    /// One-load migration field; old levels store these surface values in a
    /// separate component record. Hydration fans them out over the imported
    /// slots, then clears this field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub legacy_material_override: Option<LegacyMaterialOverrideData>,

    /// What may change about this mesh at runtime (Pulsar-Native#837). It
    /// sets the movable flag of the mesh's draw rows ([`super::StaticMeshDraw`])
    /// and the object's [`super::object_movability`]; levels saved before it
    /// load as Static, which is how their rows were already flagged.
    #[property]
    #[serde(default)]
    pub movability: super::ObjectMovability,

    /// The mesh's actual vertex/index data -- not an indirect handle into a
    /// separate asset registry, the payload itself (per the governing rule:
    /// "it doesn't hold an int32 that points to the mesh, it holds the
    /// mesh"). Populated once, at hydrate time, by
    /// `decode_static_mesh_component` (or `property_written` when
    /// `mesh_asset` changes) -- never authored directly. Never
    /// round-tripped through JSON: mesh geometry lives in
    /// the asset file `mesh_asset` already names, re-derived at hydrate
    /// time, not duplicated into every saved scene.
    ///
    /// `content_id = "mesh_asset"` (Pulsar-Native#632/#659): SceneDB routes
    /// this field's GPU allocation through its content-id-interned pool
    /// instead of one private allocation per entity -- ten components
    /// naming the same `mesh_asset` upload and store the geometry ONCE,
    /// freed automatically when the last reference despawns/removes. This
    /// is the ENTIRE consumer-side change the dedup feature asks for: the
    /// attribute plus `MeshAssetPath`'s `ContentAddressed` impl above.
    /// Nothing else about this struct, its hydrate, or any renderer call
    /// site changes -- `..._gpu_handle` accessors keep their exact
    /// signature and now transparently resolve to a range shared with every
    /// other entity referencing the same asset.
    #[gpu(buffer = "builtin_mesh_vertex", mirror = Once, content_id = "mesh_asset")]
    #[serde(skip)]
    pub vertices: Vec<PackedVertex>,
    /// See [`Self::vertices`] -- same rules, the index half of the same
    /// upload, interned under the SAME `mesh_asset` content id (a separate
    /// pool from `vertices`' own -- see `gpu::interned_pool`'s module doc
    /// on why sharing an id across two pools needs no coordination between
    /// them).
    #[gpu(buffer = "builtin_mesh_index", mirror = Once, content_id = "mesh_asset")]
    #[serde(skip)]
    pub indices: Vec<u32>,

    /// Names and source material indices discovered while loading the mesh.
    /// These are asset metadata, not authored per-instance overrides.
    #[serde(skip)]
    pub material_slot_metadata: Vec<MeshMaterialSlot>,

    /// Index ranges that route imported geometry to material slots.
    #[serde(skip)]
    pub mesh_sections: Vec<MeshSection>,

    /// Local-space bounding sphere (xyz = center, w = radius) computed once
    /// from `vertices`' actual positions at hydrate time -- see
    /// `decode_static_mesh_component`. Uploaded with the derived draw rows
    /// ([`super::StaticMeshDraw`]); the renderer's scene join transforms it
    /// by the owner's transform for culling. Not derived from
    /// `transform.scale` -- a thin mesh at scale 1.0 and a cube at scale 1.0
    /// have different real extents and must not collapse to the same bound.
    #[serde(skip)]
    pub bounds_local: [f32; 4],
}

#[register_scene_props_applier]
impl ScenePropsProjector for StaticMeshComponent {
    const CLASS_NAME: &'static str = "StaticMeshComponent";

    fn apply_scene_props(props: &mut HashMap<String, Value>, component_data: Option<&Value>) {
        props.remove("mesh_asset");
        let Some(data) = component_data else { return };
        if let Some(path) = data
            .as_object()
            .and_then(|o| o.get("mesh_asset"))
            .and_then(|v| v.as_str())
            .filter(|s| !s.trim().is_empty())
        {
            props.insert("mesh_asset".to_string(), Value::from(path));
        }
    }
}

/// Load `mesh_asset`'s vertex/index data and local bounding sphere from
/// disk, or the "no mesh" defaults ([`local_bounding_sphere`]'s own empty
/// fallback) if the path is empty, unresolvable, or fails to load.
///
/// Shared by [`decode_static_mesh_component`] (first attach) and
/// [`static_mesh_property_written`] (a live `mesh_asset` edit made
/// *after* attach, via the properties panel) -- both need the exact same
/// disk-load behavior, just triggered at different times. Resolves the
/// project-relative path via `engine_state::get_project_path()` -- a
/// global, context-free accessor, since neither caller's fixed signature
/// (`&Value` / `&mut Self, Option<&str>`) carries a project root.
fn load_mesh_geometry(mesh_asset: &str) -> ([f32; 4], crate::mesh_cache::MeshAssetUpload) {
    let mesh_asset = mesh_asset.trim();
    // Baseline fallback for "no mesh assigned" / "failed to load" -- matches
    // the safety-net minimum the old transform-scale heuristic used
    // (`scale.length().max(0.2) * 0.5`), overwritten below on a real load.
    let mut bounds_local = [0.0, 0.0, 0.0, 0.5];
    let mut upload = crate::mesh_cache::MeshAssetUpload::default();

    if !mesh_asset.is_empty() {
        match engine_state::get_project_path() {
            Some(project_root) => {
                let abs_path = resolve_asset_path(std::path::Path::new(&project_root), mesh_asset);
                match load_mesh_asset_upload(&abs_path) {
                    Some(loaded) => {
                        tracing::info!(
                            "StaticMeshComponent: loaded '{}' ({} vertices, {} indices, {} material sections)",
                            abs_path.display(),
                            loaded.geometry.vertices.len(),
                            loaded.geometry.indices.len(),
                            loaded.sections.len()
                        );
                        bounds_local = local_bounding_sphere(&loaded.geometry.vertices);
                        upload = loaded;
                    }
                    None => {
                        tracing::warn!(
                            "StaticMeshComponent: failed to load mesh '{}' ({})",
                            mesh_asset,
                            abs_path.display()
                        );
                    }
                }
            }
            None => {
                tracing::warn!(
                    "StaticMeshComponent: no project path available, cannot resolve mesh_asset '{}'",
                    mesh_asset
                );
            }
        }
    }

    (bounds_local, upload)
}

fn reconcile_material_slots(
    existing: &StaticMeshMaterialSlots,
    asset_slots: &[MeshMaterialSlot],
) -> StaticMeshMaterialSlots {
    let slots = asset_slots
        .iter()
        .enumerate()
        .map(|(index, source)| {
            let previous = existing
                .slots
                .iter()
                .find(|slot| {
                    source.source_material.is_some()
                        && slot.source_material == source.source_material
                })
                .or_else(|| {
                    existing
                        .slots
                        .iter()
                        .find(|slot| slot.name == source.name && !source.name.is_empty())
                });
            StaticMeshMaterialSlot {
                source_material: source.source_material,
                name: if source.name.is_empty() {
                    format!("Material Slot {}", index + 1)
                } else {
                    source.name.clone()
                },
                material_asset: previous
                    .map(|slot| slot.material_asset.clone())
                    .unwrap_or_default(),
                imported_surface: source.surface,
                surface_override: previous.and_then(|slot| slot.surface_override),
            }
        })
        .collect();
    StaticMeshMaterialSlots { slots }
}

fn apply_legacy_material_override(component: &mut StaticMeshComponent) {
    let Some(legacy) = component.legacy_material_override.take() else {
        return;
    };
    let surface = legacy.into_surface();
    for slot in &mut component.material_slots.slots {
        slot.surface_override = Some(surface);
    }
}

/// Load `component.mesh_asset`'s vertex/index data, sections and material
/// slots into the component's own fields. Shared by [`decode_static_mesh_component`]
/// (a value entering from a file or tool) and [`static_mesh_property_written`]
/// (a live edit that changed `mesh_asset`).
///
/// A missing or unloadable `mesh_asset` is not a failure: the component keeps
/// empty `vertices`/`indices` (a real, if invisible, mesh), the same as an
/// object with no mesh assigned.
fn load_mesh_asset_into(component: &mut StaticMeshComponent) {
    let (bounds_local, upload) = load_mesh_geometry(component.mesh_asset.as_str());
    component.bounds_local = bounds_local;
    component.vertices = upload.geometry.vertices;
    component.indices = upload.geometry.indices;
    component.material_slots =
        reconcile_material_slots(&component.material_slots, &upload.material_slots);
    apply_legacy_material_override(component);
    component.mesh_sections = upload.sections;
    component.material_slot_metadata = upload.material_slots;
}

impl StaticMeshComponent {
    /// A component showing the mesh asset at the project-relative `path`,
    /// its geometry loaded, as decoding `{"mesh_asset": path}` would give.
    pub fn for_mesh_asset(path: &str) -> Self {
        let mut component = Self {
            mesh_asset: MeshAssetPath::new(path),
            ..Default::default()
        };
        load_mesh_asset_into(&mut component);
        component
    }
}

/// `StaticMeshComponent`'s JSON boundary decoder (Pulsar-Native#561 Phase
/// D). The serialized form references its mesh by `mesh_asset`; decoding
/// loads that asset's data once, here, so the value that enters the world is
/// complete. Its `#[gpu]` pools are then written by SceneDB's normal insert.
fn decode_static_mesh_component(data: &serde_json::Value) -> Result<StaticMeshComponent, String> {
    let mut component: StaticMeshComponent =
        serde_json::from_value(data.clone()).map_err(|error| error.to_string())?;
    load_mesh_asset_into(&mut component);
    Ok(component)
}

/// `property_written` hook: a reflected write that changed `mesh_asset`
/// (the properties panel's mesh picker, a script) loads the newly named
/// asset under the same write guard, so SceneDB commits the new path and its
/// geometry together. Writes to any other property load nothing.
fn static_mesh_property_written(component: &mut StaticMeshComponent, property: Option<&str>) {
    if matches!(property, None | Some("mesh_asset")) {
        load_mesh_asset_into(component);
    }
}

/// Local-space bounding sphere (xyz = center, w = radius) from a mesh's
/// actual vertex positions: center = AABB midpoint, radius = distance from
/// that center to the farthest AABB corner (conservative, cheap -- no need
/// for a tighter Ritter-style fit here). Falls back to the same 0.5 default
/// as "no mesh loaded" if `vertices` is empty.
fn local_bounding_sphere(vertices: &[PackedVertex]) -> [f32; 4] {
    let Some(first) = vertices.first() else {
        return [0.0, 0.0, 0.0, 0.5];
    };
    let mut min = first.position;
    let mut max = first.position;
    for v in &vertices[1..] {
        for axis in 0..3 {
            min[axis] = min[axis].min(v.position[axis]);
            max[axis] = max[axis].max(v.position[axis]);
        }
    }
    let center = [
        (min[0] + max[0]) * 0.5,
        (min[1] + max[1]) * 0.5,
        (min[2] + max[2]) * 0.5,
    ];
    let extent = [max[0] - center[0], max[1] - center[1], max[2] - center[2]];
    let radius = (extent[0] * extent[0] + extent[1] * extent[1] + extent[2] * extent[2]).sqrt();
    [center[0], center[1], center[2], radius.max(0.001)]
}

// Phase B4 (Pulsar-Native#555): the first component migrated onto
// pulsar_world_registry's World bridge.
#[register_world_component(
    decode = decode_static_mesh_component,
    property_written = static_mesh_property_written
)]
impl StaticMeshComponent {}
