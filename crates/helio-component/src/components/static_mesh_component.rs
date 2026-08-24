//! Static mesh component for mesh asset assignment.

use engine_class_derive::{
    engine_class, register_runtime_behavior, register_scene_props_applier, register_world_component,
};
use helio::PackedVertex;
use pulsar_reflection::{
    ComponentRuntimeBehavior, ComponentRuntimeContext, ReflectError, RuntimeComponentOwner,
    ScenePropsProjector,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use crate::asset_component::AssetComponentRegistration;
use crate::subsystems::{
    ensure_content_ledger_attached, load_mesh_upload_shared, resolve_asset_path,
};

pulsar_reflection::inventory::submit! {
    AssetComponentRegistration {
        asset_kind: plugin_editor_api::AssetKind::Mesh,
        class_name: "StaticMeshComponent",
        data_field: "mesh_asset",
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

// ── StaticMeshComponent ───────────────────────────────────────────────────────

/// Attaches a mesh asset to a scene object.
///
/// `scene_store` (Pulsar-Native#561 Phase D): opts this struct into
/// `#[gpu]`-mirrored fields via `#[engine_class]`'s delegation to
/// `pulsar_scenedb::SceneStore` -- see `vertices`/`indices` below, and
/// `hydrate_static_mesh_component`'s doc for how they get populated. A
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

    /// Content-addressed identity of `mesh_asset`'s decoded geometry
    /// (Pulsar-Native content-dedup work). NOT user-visible and never
    /// serialized (`skip`): derived at hydrate time from the payload bytes
    /// themselves, exactly like `vertices`/`indices` -- a scene file that
    /// predates this field loads unchanged, and property editors never see
    /// it (it is not a `#[property]`, and `MeshAssetPath` JSON stays a
    /// plain string).
    ///
    /// Its TYPE is what matters mechanically:
    /// `pulsar_scenedb::handle_ledger::HandleId` -- SceneDB's derive
    /// recognizes handle fields by type, so `World::insert` acquires a
    /// reference into the attached ledger on first hydrate, swaps counts
    /// when `mesh_asset` changes in place, releases on component removal,
    /// and batch-releases through the despawn fast path. That is the whole
    /// "last reference out frees everything" story: two entities pointing
    /// at one file resolve to one identity with count 2; when both die,
    /// the count hits zero exactly once and the drop callback evicts the
    /// shared CPU parse (see `crate::content_ledger`). Zero means "no
    /// asset" (load failed / not yet hydrated) and produces no events.
    #[serde(skip)]
    pub content_id: pulsar_scenedb::handle_ledger::HandleId,

    /// The mesh's actual vertex/index data -- not an indirect handle into a
    /// separate asset registry, the payload itself (per the governing rule:
    /// "it doesn't hold an int32 that points to the mesh, it holds the
    /// mesh"). Populated once, at hydrate time, by
    /// `hydrate_static_mesh_component` -- never authored directly, never
    /// touched by `sync_component` (which only ever sees `&World`, never
    /// disk I/O). Never round-tripped through JSON: mesh geometry lives in
    /// the asset file `mesh_asset` already names, re-derived at hydrate
    /// time, not duplicated into every saved scene.
    ///
    /// Since the shared-parse layer landed, these are cloned out of ONE
    /// deduplicated parse per distinct asset (`subsystems::
    /// load_mesh_upload_shared`) -- N components hydrating the same file
    /// decode once; each still materializes its own `Vec`s here (per-field
    /// GPU mirroring is per-entity by design; pool-level single-instancing
    /// of identical geometry is a later milestone).
    #[gpu]
    #[serde(skip)]
    pub vertices: Vec<PackedVertex>,
    /// See [`Self::vertices`] -- same rules, the index half of the same
    /// upload.
    #[gpu]
    #[serde(skip)]
    pub indices: Vec<u32>,
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

/// Custom hydrate for `#[register_world_component(hydrate = ...)]`
/// (Pulsar-Native#561 Phase D). Loads `mesh_asset`'s actual vertex/index
/// data, once, right here at hydrate time -- not per render frame, and not
/// through any Helio-specific code (`sync_component`'s dispatch only ever
/// gets `&World`, deliberately, so it structurally can't do disk I/O; this
/// is the one call site that already has `&mut World`). Resolves the
/// project-relative path via `engine_state::get_project_path()` -- a
/// global, context-free accessor, since the fixed hydrate signature
/// (`&mut World, Entity, &Value`) has no `ComponentRuntimeContext` to pull
/// a project root from the way `sync_component` does.
///
/// A missing or unloadable `mesh_asset` is not a hydrate failure -- mirrors
/// `sync_component`'s existing "no mesh_asset" tolerance -- the component
/// still hydrates, just with empty `vertices`/`indices` (a real, if
/// invisible, entity, same as today's `insert_actor`-based path leaves an
/// object with no mesh assigned).
fn hydrate_static_mesh_component(
    world: &mut pulsar_scenedb::World,
    entity: pulsar_scenedb::Entity,
    data: &serde_json::Value,
) -> Result<(), String> {
    let mut parsed: StaticMeshComponent =
        serde_json::from_value(data.clone()).map_err(|error| error.to_string())?;

    // Ledger wiring must precede the insert below, or this component's
    // acquisition arrives pre-ledger (un-counted acquire that its despawn
    // WILL release against). Idempotent; also registers the CPU-eviction
    // drop callback once per process. See `ensure_content_ledger_attached`.
    ensure_content_ledger_attached(world);

    let mesh_asset = parsed.mesh_asset.as_str().trim();
    if !mesh_asset.is_empty() {
        match engine_state::get_project_path() {
            Some(project_root) => {
                let abs_path = resolve_asset_path(std::path::Path::new(&project_root), mesh_asset);
                // Shared-parse dedup: N components hydrating the same asset
                // cost one read + one hash + one decode total (freshness
                // validated by a single stat per call). Identity comes from
                // the same pass -- no second hashing round-trip.
                match load_mesh_upload_shared(&abs_path) {
                    Some(shared) => {
                        tracing::info!(
                            "StaticMeshComponent hydrate: loaded '{}' ({} vertices, {} indices, content {})",
                            abs_path.display(),
                            shared.upload.vertices.len(),
                            shared.upload.indices.len(),
                            shared.content_id
                        );
                        parsed.vertices = shared.upload.vertices.clone();
                        parsed.indices = shared.upload.indices.clone();
                        // The ledger event fires inside `world.insert`
                        // below, driven by THIS field's type (`HandleId`):
                        // first insert acquires, later re-hydrates with an
                        // unchanged id are exact no-ops, a changed asset
                        // swaps counts, and despawn releases through the
                        // SceneDB capability end-to-end.
                        parsed.content_id =
                            pulsar_scenedb::handle_ledger::HandleId(shared.content_id.0);
                    }
                    None => {
                        tracing::warn!(
                            "StaticMeshComponent hydrate: failed to load mesh '{}' ({})",
                            mesh_asset,
                            abs_path.display()
                        );
                        // content_id stays ZERO ("no asset"): zero produces
                        // no ledger events, so a failed load never poisons
                        // counts.
                    }
                }
            }
            None => {
                tracing::warn!(
                    "StaticMeshComponent hydrate: no project path available, cannot resolve mesh_asset '{}'",
                    mesh_asset
                );
            }
        }
    }

    world.insert(entity, parsed);
    Ok(())
}

// Phase B4 (Pulsar-Native#555): the first component migrated onto
// pulsar_world_registry's World bridge -- proves the pattern before B5
// rolls it out to the rest. `#[register_world_component]` must be written
// above `#[register_runtime_behavior]` (see that macro's own doc for why:
// only the bottom attribute in the stack re-emits the impl block).
#[register_world_component(hydrate = hydrate_static_mesh_component)]
#[register_runtime_behavior]
impl ComponentRuntimeBehavior for StaticMeshComponent {
    const CLASS_NAME: &'static str = "StaticMeshComponent";

    fn sync_component(
        _owner: &RuntimeComponentOwner,
        _component_index: usize,
        _component: &Self,
        _context: &mut dyn ComponentRuntimeContext,
    ) {
        // Deliberately empty (Pulsar-Native#561 Phase E cutover). This used
        // to load `mesh_asset` itself and call `Renderer::scene_mut()
        // .insert_actor(SceneActor::mesh(upload))` -- a second, independent
        // copy of the mesh data in Helio's own mesh pool, loaded from disk a
        // second time every dirty pass, on top of what `hydrate_static_mesh_component`
        // already does (loads the file once, populates this component's own
        // `#[gpu] vertices`/`indices` fields, which SceneDB mirrors straight
        // into the SAME pool `helio::Scene`'s `MeshPool` reads from -- see
        // `mesh.rs`'s `rebind_static_pools`/`adopt_static_slice`). Resolving
        // a `MeshId`/`ObjectDescriptor` for that already-GPU-resident data
        // needs the entity's row (`entity.index()`) and the SceneDB-side
        // `..._gpu_handle` accessors this trait's `&Self`-only signature has
        // no way to reach -- that's `engine_backend`'s
        // `HelioRenderer::sync_snapshot_components`, which already has
        // `Entity`/`World` in scope for exactly this reason.
    }
}
