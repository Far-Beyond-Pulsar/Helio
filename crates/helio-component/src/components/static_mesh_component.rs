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
use crate::subsystems::{load_mesh_upload, resolve_asset_path};
use crate::texture_cache::{TexturePayload, TextureSemantic};

pulsar_reflection::inventory::submit! {
    AssetComponentRegistration {
        asset_kind: plugin_editor_api::AssetKind::Mesh,
        class_name: "StaticMeshComponent",
        data_field: "mesh_asset",
    }
}
// Helio#237: a dropped texture lands on the primary color slot — the same
// drop-to-assign UX meshes have, pointed at this class's first texture chain.
pulsar_reflection::inventory::submit! {
    AssetComponentRegistration {
        asset_kind: plugin_editor_api::AssetKind::Texture,
        class_name: "StaticMeshComponent",
        data_field: "base_color_asset",
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
/// project-relative path exactly like `hydrate_static_mesh_component`
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
        let abs_path = crate::subsystems::resolve_asset_path(std::path::Path::new(&project_root), path);
        crate::mesh_cache::content_id_for_path(&abs_path).map(HandleId).unwrap_or(HandleId::ZERO)
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

// ── TextureAssetPath (Helio#237) ─────────────────────────────────────────────

/// Strongly-typed wrapper for texture asset paths — the seven-slot twin of
/// [`MeshAssetPath`] (Helio#237 texel-streaming S1). Using this as a field
/// type causes the property inspector to render a texture-asset search
/// browser (`.ptex` plus raw source images) instead of a plain text box.
///
/// Serialises transparently as a JSON string, so scene files keep storing
/// plain paths — the user-facing API is unchanged; content identity and
/// SceneDB tiering are derived entirely from the path at hydrate time.
///
/// # Example
///
/// ```ignore
/// #[property]
/// pub base_color_asset: TextureAssetPath,
/// ```
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct TextureAssetPath(pub String);

impl TextureAssetPath {
    /// Create a new `TextureAssetPath` from any string-like value.
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

/// SceneDB's content-identity seam for textures — byte-for-byte the same
/// contract as `MeshAssetPath`'s impl above: an empty path is
/// [`pulsar_scenedb::handle_ledger::HandleId::ZERO`]; otherwise resolve the
/// project-relative path like `hydrate_static_mesh_component` does and defer
/// to `texture_cache::content_id_for_path` (native `.ptex`: a header read;
/// anything else: the shared canonical-path + mtime/len memoized hash). Path
/// aliases converge onto one id; a real edit mints a new one; dangling
/// references behave like "no asset", never an error.
impl pulsar_scenedb::handle_ledger::ContentAddressed for TextureAssetPath {
    fn content_id(&self) -> pulsar_scenedb::handle_ledger::HandleId {
        use pulsar_scenedb::handle_ledger::HandleId;

        let path = self.0.trim();
        if path.is_empty() {
            return HandleId::ZERO;
        }
        let Some(project_root) = engine_state::get_project_path() else {
            return HandleId::ZERO;
        };
        let abs_path = crate::subsystems::resolve_asset_path(std::path::Path::new(&project_root), path);
        crate::texture_cache::content_id_for_path(&abs_path).map(HandleId).unwrap_or(HandleId::ZERO)
    }
}

impl std::fmt::Display for TextureAssetPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<String> for TextureAssetPath {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TextureAssetPath {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ── TextureAssetPath reflection registration ────────────────────────────────

fn serialize_texture_asset_path_json(
    value: &TextureAssetPath,
) -> pulsar_reflection::ReflectResult<serde_json::Value> {
    Ok(serde_json::json!(value.0))
}

fn deserialize_texture_asset_path_json(
    value: serde_json::Value,
) -> pulsar_reflection::ReflectResult<TextureAssetPath> {
    value
        .as_str()
        .map(|s| TextureAssetPath(s.to_string()))
        .ok_or_else(|| ReflectError::TypeMismatch {
            expected: "TextureAssetPath",
            found: format!("{:?}", value),
        })
}

// ── TextureAssetPath property editor ─────────────────────────────────────────

/// Property editor for [`TextureAssetPath`] — a searchable texture-asset
/// browser, cloned from [`MeshAssetEditor`]'s shape but filtered to texture
/// sources (native `.ptex` plus the importable raster formats).
pub struct TextureAssetEditor {
    label: String,
    id_prefix: String,
    prop_name: String,
    picker: gpui::Entity<ui_common::asset_picker::MeshAssetPicker>,
    path: String,
    write_back: pulsar_reflection::PropertyWriteBack,
    _subs: Vec<gpui::Subscription>,
}

/// Extensions the texture picker offers, in browse order.
const TEXTURE_PICKER_EXTENSIONS: &[&str] =
    &["ptex", "png", "jpg", "jpeg", "bmp", "tga", "webp"];

impl TextureAssetEditor {
    fn new(
        args: &pulsar_reflection::PropertyEditorArgs<'_>,
        window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> Self {
        use gpui::AppContext as _;
        use ui_common::asset_picker::{AssetPickedEvent, AssetQuery};

        let path = args
            .current_value
            .downcast_ref::<TextureAssetPath>()
            .map(|p| p.0.clone())
            .unwrap_or_default();

        let project_root = engine_state::get_project_path().map(std::path::PathBuf::from);
        // No builtin textures exist (unlike BUILTIN_MESHES): every entry comes
        // from the project's own assets.
        let queries = TEXTURE_PICKER_EXTENSIONS
            .iter()
            .map(|&e| AssetQuery::extension(e))
            .collect();

        let picker = cx.new(|cx| {
            ui_common::asset_picker::MeshAssetPicker::new(path.clone(), Vec::new(), project_root, queries, window, cx)
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
                (this.write_back)(Box::new(TextureAssetPath(selected)), window, cx);
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

    /// Accept a texture assigned elsewhere — e.g. dropped straight onto the
    /// viewport or another row's editor.
    fn set_value(&mut self, path: &TextureAssetPath, cx: &mut gpui::Context<Self>) {
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

impl gpui::Render for TextureAssetEditor {
    fn render(
        &mut self,
        _window: &mut gpui::Window,
        cx: &mut gpui::Context<Self>,
    ) -> impl gpui::IntoElement {
        use gpui::prelude::*;
        use ui::button::{Button, ButtonVariants as _};
        use ui::{ActiveTheme, Sizable, h_flex, popover::Popover};

        let display = if self.path.is_empty() {
            "No texture selected".to_string()
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
                    "texture-asset-picker-{}-{}",
                    self.id_prefix, self.prop_name
                ))
                .anchor(gpui::Corner::BottomRight)
                .trigger(
                    Button::new(format!(
                        "texture-asset-picker-btn-{}-{}",
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

fn texture_asset_editor(
    args: &pulsar_reflection::PropertyEditorArgs<'_>,
    window: &mut gpui::Window,
    cx: &mut gpui::App,
) -> pulsar_reflection::BoundPropertyEditor {
    use gpui::AppContext as _;

    let entity = cx.new(|cx| TextureAssetEditor::new(args, window, cx));
    pulsar_reflection::BoundPropertyEditor::new(
        entity,
        |editor: &mut TextureAssetEditor, value: &TextureAssetPath, _window, cx| {
            editor.set_value(value, cx)
        },
    )
}

/// Register `TextureAssetPath` with the reflection system (same mechanics as
/// `RegisteredMeshAssetPath`: JSON-transparent string + browser editor).
#[pulsar_reflection::pulsar_type(
    serialize_json_with = serialize_texture_asset_path_json,
    deserialize_json_with = deserialize_texture_asset_path_json,
    editor = texture_asset_editor
)]
#[allow(dead_code)]
type RegisteredTextureAssetPath = TextureAssetPath;

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
#[engine_class(category = "Rendering", default, clone, debug, serialize, deserialize, scene_store)]
pub struct StaticMeshComponent {
    /// Relative asset path to the mesh file (e.g. "meshes/primitives/SM_Cube.fbx").
    ///
    /// Typed as [`MeshAssetPath`] so the property inspector renders a mesh-asset
    /// search browser instead of a plain text input.
    #[property]
    pub mesh_asset: MeshAssetPath,

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
    #[gpu(mirror = Once, content_id = "mesh_asset")]
    #[serde(skip)]
    pub vertices: Vec<PackedVertex>,
    /// See [`Self::vertices`] -- same rules, the index half of the same
    /// upload, interned under the SAME `mesh_asset` content id (a separate
    /// pool from `vertices`' own -- see `gpu::interned_pool`'s module doc
    /// on why sharing an id across two pools needs no coordination between
    /// them).
    #[gpu(mirror = Once, content_id = "mesh_asset")]
    #[serde(skip)]
    pub indices: Vec<u32>,

    // ── Texture slots (Helio#237 texel-streaming S1) ─────────────────────
    //
    // Seven authored slots mirroring `libhelio::MaterialTextures`, each a
    // (path wrapper, texel payload) pair following the exact mesh_asset
    // pattern: the wrapper is the light, JSON-visible authored field; the
    // `Vec<TexturePayload>` is the heavy GPU payload interned by its
    // wrapper's content id (one chain per slot — sharing a texture across
    // entities stores it once; two meshes sharing ONLY base_color share
    // exactly that one chain). Missing/unloadable = empty payload = ZERO
    // semantics: absent slots leave defaults and the component still
    // hydrates (see `hydrate_static_mesh_component`). Scene JSON still
    // stores plain paths; nothing renderer-side changes (no shader/sampling
    // work here — that's S2/Helio#238).
    /// Base color / albedo (`MaterialTextures::base_color`).
    #[property]
    pub base_color_asset: TextureAssetPath,
    /// Normal map (`MaterialTextures::normal`); BC5 X/Y — Z is reconstructed
    /// at sample time (S2), never stored.
    #[property]
    pub normal_asset: TextureAssetPath,
    /// Roughness (R) + metallic (G) packed pair
    /// (`MaterialTextures::roughness_metallic`).
    #[property]
    pub roughness_metallic_asset: TextureAssetPath,
    /// Emissive color (`MaterialTextures::emissive`).
    #[property]
    pub emissive_asset: TextureAssetPath,
    /// Ambient occlusion mask (`MaterialTextures::occlusion`).
    #[property]
    pub occlusion_asset: TextureAssetPath,
    /// Specular color (`MaterialTextures::specular_color`).
    #[property]
    pub specular_color_asset: TextureAssetPath,
    /// Specular weight scalar (`MaterialTextures::specular_weight`).
    #[property]
    pub specular_weight_asset: TextureAssetPath,

    /// Canonical coarse-first `.ptex` body bytes for [`Self::base_color_asset`]
    /// — see `texture_cache`'s module doc for the container/segment contract.
    #[gpu(mirror = Once, content_id = "base_color_asset")]
    #[serde(skip)]
    pub base_color_data: Vec<TexturePayload>,
    /// See [`Self::base_color_data`] — the normal chain.
    #[gpu(mirror = Once, content_id = "normal_asset")]
    #[serde(skip)]
    pub normal_data: Vec<TexturePayload>,
    /// See [`Self::base_color_data`] — the roughness/metallic chain.
    #[gpu(mirror = Once, content_id = "roughness_metallic_asset")]
    #[serde(skip)]
    pub roughness_metallic_data: Vec<TexturePayload>,
    /// See [`Self::base_color_data`] — the emissive chain.
    #[gpu(mirror = Once, content_id = "emissive_asset")]
    #[serde(skip)]
    pub emissive_data: Vec<TexturePayload>,
    /// See [`Self::base_color_data`] — the occlusion chain.
    #[gpu(mirror = Once, content_id = "occlusion_asset")]
    #[serde(skip)]
    pub occlusion_data: Vec<TexturePayload>,
    /// See [`Self::base_color_data`] — the specular-color chain.
    #[gpu(mirror = Once, content_id = "specular_color_asset")]
    #[serde(skip)]
    pub specular_color_data: Vec<TexturePayload>,
    /// See [`Self::base_color_data`] — the specular-weight chain.
    #[gpu(mirror = Once, content_id = "specular_weight_asset")]
    #[serde(skip)]
    pub specular_weight_data: Vec<TexturePayload>,
}

#[register_scene_props_applier]
impl ScenePropsProjector for StaticMeshComponent {
    const CLASS_NAME: &'static str = "StaticMeshComponent";

    fn apply_scene_props(props: &mut HashMap<String, Value>, component_data: Option<&Value>) {
        props.remove("mesh_asset");
        // Helio#237: the seven texture slots project exactly like mesh_asset —
        // stripped from the generic props, re-inserted only when authored
        // non-empty (an empty slot means "no texture", not "clear to path ''").
        props.remove("base_color_asset");
        props.remove("normal_asset");
        props.remove("roughness_metallic_asset");
        props.remove("emissive_asset");
        props.remove("occlusion_asset");
        props.remove("specular_color_asset");
        props.remove("specular_weight_asset");
        let Some(data) = component_data else { return };
        let Some(object) = data.as_object() else { return };
        for key in [
            "mesh_asset",
            "base_color_asset",
            "normal_asset",
            "roughness_metallic_asset",
            "emissive_asset",
            "occlusion_asset",
            "specular_color_asset",
            "specular_weight_asset",
        ] {
            if let Some(path) = object.get(key).and_then(|v| v.as_str()).filter(|s| !s.trim().is_empty()) {
                props.insert(key.to_string(), Value::from(path));
            }
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

    let mesh_asset = parsed.mesh_asset.as_str().trim();
    if !mesh_asset.is_empty() {
        match engine_state::get_project_path() {
            Some(project_root) => {
                let abs_path = resolve_asset_path(std::path::Path::new(&project_root), mesh_asset);
                match load_mesh_upload(&abs_path) {
                    Some(upload) => {
                        tracing::info!(
                            "StaticMeshComponent hydrate: loaded '{}' ({} vertices, {} indices)",
                            abs_path.display(),
                            upload.vertices.len(),
                            upload.indices.len()
                        );
                        parsed.vertices = upload.vertices;
                        parsed.indices = upload.indices;
                    }
                    None => {
                        tracing::warn!(
                            "StaticMeshComponent hydrate: failed to load mesh '{}' ({})",
                            mesh_asset,
                            abs_path.display()
                        );
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

    // Helio#237: the seven texture slots resolve+decode exactly like
    // mesh_asset above, once, here at hydrate time. A native `.ptex` parses
    // its own container; a raw source image runs the import pipeline in
    // memory (no writes). The slot's semantic drives the BCn mapping only for
    // that raw-source path. A missing/unloadable slot is NOT a hydrate
    // failure — identical tolerance to the mesh arm: the component still
    // hydrates, just with that chain empty (ZERO semantics).
    let project = engine_state::get_project_path();
    let texture_slots: [(&TextureAssetPath, TextureSemantic, &mut Vec<TexturePayload>); 7] = [
        (&parsed.base_color_asset, TextureSemantic::BaseColor, &mut parsed.base_color_data),
        (&parsed.normal_asset, TextureSemantic::Normal, &mut parsed.normal_data),
        (
            &parsed.roughness_metallic_asset,
            TextureSemantic::MetallicRoughness,
            &mut parsed.roughness_metallic_data,
        ),
        (&parsed.emissive_asset, TextureSemantic::Emissive, &mut parsed.emissive_data),
        (&parsed.occlusion_asset, TextureSemantic::Occlusion, &mut parsed.occlusion_data),
        (
            &parsed.specular_color_asset,
            TextureSemantic::SpecularColor,
            &mut parsed.specular_color_data,
        ),
        (
            &parsed.specular_weight_asset,
            TextureSemantic::SpecularWeight,
            &mut parsed.specular_weight_data,
        ),
    ];
    for (slot, semantic, data) in texture_slots {
        let path = slot.as_str().trim();
        if path.is_empty() {
            continue;
        }
        let Some(project_root) = project.as_deref() else {
            tracing::warn!(
                "StaticMeshComponent hydrate: no project path available, cannot resolve texture '{}' ({})",
                path,
                semantic.suffix()
            );
            continue;
        };
        let abs_path = resolve_asset_path(std::path::Path::new(project_root), path);
        match crate::texture_cache::decoded_body_for_path(&abs_path, semantic) {
            Some(body) => {
                tracing::info!(
                    "StaticMeshComponent hydrate: loaded texture '{}' ({}, {} bytes)",
                    abs_path.display(),
                    semantic.suffix(),
                    body.len()
                );
                *data = body.iter().copied().map(TexturePayload).collect();
            }
            None => {
                tracing::warn!(
                    "StaticMeshComponent hydrate: failed to load texture '{}' ({})",
                    path,
                    abs_path.display()
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
