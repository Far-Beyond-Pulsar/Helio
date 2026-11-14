//! Node types: what a node looks like (pins, parameters) and how it adds its
//! passes to a graph.
//!
//! This crate knows no pass and no renderer. Features register
//! [`NodeDescriptor`]s for what they own (see [`crate::Feature`]); editors list
//! whatever is registered, and `helio-graph-asset` compiles graphs of them.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use helio_core::RenderGraph;
use serde_json::Value;

use crate::pins::{PinDesc, PinType};

/// A node instance's parameters, by key.
pub type Params = BTreeMap<String, Value>;

/// How an editor edits a parameter.
#[derive(Clone, Debug, PartialEq)]
pub enum ParamType {
    Text,
    Bool,
    Int { min: i64, max: i64 },
    Float { min: f64, max: f64 },
    Choice(Vec<String>),
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParamDesc {
    pub key: String,
    pub label: String,
    pub ty: ParamType,
    pub default: Value,
}

impl ParamDesc {
    pub fn new(key: &str, label: &str, ty: ParamType, default: Value) -> Self {
        Self {
            key: key.into(),
            label: label.into(),
            ty,
            default,
        }
    }

    pub fn text(key: &str, label: &str, default: &str) -> Self {
        Self::new(key, label, ParamType::Text, Value::from(default))
    }

    pub fn boolean(key: &str, label: &str, default: bool) -> Self {
        Self::new(key, label, ParamType::Bool, Value::from(default))
    }

    pub fn int(key: &str, label: &str, min: i64, max: i64, default: i64) -> Self {
        Self::new(
            key,
            label,
            ParamType::Int { min, max },
            Value::from(default),
        )
    }

    pub fn choice(key: &str, label: &str, options: &[&str], default: &str) -> Self {
        Self::new(
            key,
            label,
            ParamType::Choice(options.iter().map(|o| o.to_string()).collect()),
            Value::from(default),
        )
    }
}

/// The pins a node instance has, which can depend on its parameters.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NodePins {
    #[serde(default)]
    pub inputs: Vec<PinDesc>,
    #[serde(default)]
    pub outputs: Vec<PinDesc>,
}

/// The frame a graph is built for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameDesc {
    /// Output size.
    pub width: u32,
    pub height: u32,
    /// Size passes render at (render scale applied).
    pub internal_width: u32,
    pub internal_height: u32,
    pub surface_format: wgpu::TextureFormat,
    /// Rendering stereo views for a headset.
    pub xr: bool,
}

impl FrameDesc {
    pub fn render_size(&self) -> helio_core::RenderSize {
        helio_core::RenderSize::new(
            [self.internal_width, self.internal_height],
            [self.width, self.height],
        )
    }
}

/// What a node build gets from whoever builds the graph, plus a place for
/// nodes to share state with each other.
///
/// Renderer-wide facts are fields. Everything else a node needs (its
/// feature's settings, host objects such as a debug-draw state or the scene
/// handle) is a **setting**: a value of its own type the builder of the graph
/// supplies ([`Self::insert_setting`]) and the node asks for by type
/// ([`Self::setting`], [`Self::require`]). The renderer knows no feature's
/// settings type, and a feature knows no other feature's.
pub struct BuildEnv<'e> {
    pub device: &'e Arc<wgpu::Device>,
    pub queue: &'e Arc<wgpu::Queue>,
    pub frame: FrameDesc,
    pub camera_buffer: &'e wgpu::Buffer,
    pub debug_camera_buffer: &'e wgpu::Buffer,
    /// The renderer's named frame statistics (`helio_core::stats`): a node
    /// takes the slots of the stats its passes count into here, at build
    /// time, creating any stat no one has contributed to yet.
    pub stats: &'e helio_core::FrameStats,
    pub owns_device: bool,
    settings: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
    shared: HashMap<TypeId, Box<dyn Any>>,
}

impl<'e> BuildEnv<'e> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &'e Arc<wgpu::Device>,
        queue: &'e Arc<wgpu::Queue>,
        frame: FrameDesc,
        camera_buffer: &'e wgpu::Buffer,
        debug_camera_buffer: &'e wgpu::Buffer,
        stats: &'e helio_core::FrameStats,
        owns_device: bool,
    ) -> Self {
        Self {
            device,
            queue,
            frame,
            camera_buffer,
            debug_camera_buffer,
            stats,
            owns_device,
            settings: HashMap::new(),
            shared: HashMap::new(),
        }
    }

    /// Supplies a setting of type `T`, replacing one already there.
    pub fn insert_setting<T: Any + Send + Sync>(&mut self, value: T) {
        self.settings.insert(TypeId::of::<T>(), Arc::new(value));
    }

    /// [`Self::insert_setting`], by value.
    pub fn with_setting<T: Any + Send + Sync>(mut self, value: T) -> Self {
        self.insert_setting(value);
        self
    }

    /// Supplies every feature's settings (a renderer config carries them).
    pub fn with_feature_settings(mut self, settings: &crate::FeatureSettings) -> Self {
        self.settings.extend(settings.entries());
        self
    }

    /// The setting of type `T`, if the graph's builder supplied one.
    pub fn setting<T: Any>(&self) -> Option<&T> {
        self.settings
            .get(&TypeId::of::<T>())
            .and_then(|value| value.downcast_ref::<T>())
    }

    /// The setting of type `T`, or an error naming it: a node that cannot
    /// build without it fails by name instead of guessing.
    pub fn require<T: Any>(&self) -> Result<&T, String> {
        self.setting::<T>().ok_or_else(|| {
            format!(
                "this graph's builder supplies no `{}` setting",
                std::any::type_name::<T>()
            )
        })
    }

    /// Render size passes are prepared at (`render_scale` applied).
    /// The format the scene-linear colour is lit, fogged and anti-aliased in
    /// up to the tone map: the [`SceneColorFormat`] setting, or
    /// [`DEFAULT_SCENE_FORMAT`].
    pub fn scene_format(&self) -> wgpu::TextureFormat {
        self.setting::<SceneColorFormat>()
            .map_or(DEFAULT_SCENE_FORMAT, |format| format.0)
    }

    pub fn internal_size(&self) -> (u32, u32) {
        (self.frame.internal_width, self.frame.internal_height)
    }

    /// The shared value of type `T`, if an earlier node created it.
    pub fn shared_existing<T: Any>(&mut self) -> Option<&mut T> {
        self.shared
            .get_mut(&TypeId::of::<T>())
            .and_then(|value| value.downcast_mut::<T>())
    }

    /// State shared between the nodes of one build, created on first use by
    /// whichever node needs it. Keyed by type, so a feature's node types
    /// can share a value no one else knows about.
    pub fn shared<T: Any>(&mut self, init: impl FnOnce() -> T) -> &mut T {
        self.shared
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(init()))
            .downcast_mut::<T>()
            .expect("shared state is stored under its own type id")
    }
}

/// What a node's build function works with.
pub struct NodeBuild<'a, 'e> {
    pub graph: &'a mut RenderGraph,
    pub env: &'a mut BuildEnv<'e>,
    /// The instance's parameters, defaults filled in.
    pub params: &'a Params,
    pub node_id: &'a str,
}

impl NodeBuild<'_, '_> {
    pub fn text(&self, key: &str) -> String {
        self.params
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    }

    pub fn int(&self, key: &str) -> i64 {
        self.params
            .get(key)
            .and_then(Value::as_i64)
            .unwrap_or_default()
    }

    pub fn boolean(&self, key: &str) -> bool {
        self.params
            .get(key)
            .and_then(Value::as_bool)
            .unwrap_or_default()
    }
}

pub type PinsFn = Arc<dyn Fn(&Params) -> NodePins + Send + Sync>;
pub type BuildFn = Arc<dyn Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync>;

/// The scene-linear colour's format when a graph's builder sets none: half
/// floats, so emitters above 1.0 reach bloom and flare unclamped.
pub const DEFAULT_SCENE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The format the scene-linear colour is lit, fogged and anti-aliased in up
/// to the tone map, as a setting (a lighting model sets the format their lighting
/// prefers). Read it with [`BuildEnv::scene_format`].
#[derive(Clone, Copy, Debug)]
pub struct SceneColorFormat(pub wgpu::TextureFormat);

/// The kind of the node that builds pass `name`: `helio.pass.<name>`.
pub fn pass_kind(name: &str) -> String {
    format!("helio.pass.{name}")
}

/// A pass node: kind `helio.pass.<name>`, one constructor, an `Enabled`
/// toggle, pins generated from the passes it builds.
pub fn pass_node(
    name: &str,
    title: &str,
    category: &str,
    description: &str,
    build: impl Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync + 'static,
) -> NodeDescriptor {
    NodeDescriptor::new(&pass_kind(name), title, category, description, build)
        .with_param(ParamDesc::boolean("enabled", "Enabled", true))
        .with_generated_pins()
}

/// The error a node gives when a node it needs was not built before it.
pub fn missing(node: &str, needs: &str) -> String {
    format!("{node} needs the {needs} node to be built before it")
}

/// A named place in a preset graph where features' nodes join it (an
/// extension point: a slot or a draw phase's stage, which `helio-frame-api`
/// defines). A preset places the nodes registered at a point there, by order,
/// then kind, so a feature joins the pipeline without the preset naming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExtensionPoint(pub &'static str);

/// Checks the passes a node added (their indices in the graph) once it has
/// built: where a node type enforces a contract on what its passes declare
/// (a draw phase's attachments, say).
pub type CheckFn = Arc<
    dyn Fn(&BuildEnv<'_>, &RenderGraph, std::ops::Range<usize>) -> Result<(), String> + Send + Sync,
>;

/// A node type.
#[derive(Clone)]
pub struct NodeDescriptor {
    /// Stable id stored in assets, e.g. `helio.stage.geometry`.
    pub kind: String,
    pub title: String,
    /// Palette group, e.g. `Resources`.
    pub category: String,
    pub description: String,
    pub params: Vec<ParamDesc>,
    /// Pins come from what the node's passes declare, found by a probe and kept in
    /// the registry's pin table (see [`NodeRegistry::set_pins`]).
    pub generated: bool,
    /// A performance marker follows this node (see [`NodeRegistry::set_marker`]).
    pub marker_after: bool,
    /// A node that only supplies data: it has no flow pins, and is never part of
    /// the execution chain.
    pub data_only: bool,
    /// Inputs the node can also take, beyond what its passes declare by default:
    /// the pass author lists them where a parameter chooses between resources.
    pub alternative_inputs: Vec<(String, PinType)>,
    /// Why the node builds nothing where a capability it needs is absent (a
    /// platform, say): validation reports it so the graph that claims the pass
    /// is detectably short of it. `None` when the node always builds.
    pub unsupported: Option<&'static str>,
    /// The extension point the node joins a preset graph at, and its order
    /// there (lower first). `None`: the preset places it by name.
    pub placement: Option<(ExtensionPoint, i32)>,
    pins: PinsFn,
    build: BuildFn,
    checks: Vec<CheckFn>,
}

impl NodeDescriptor {
    pub fn new(
        kind: &str,
        title: &str,
        category: &str,
        description: &str,
        build: impl Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            kind: kind.into(),
            title: title.into(),
            category: category.into(),
            description: description.into(),
            params: Vec::new(),
            generated: false,
            marker_after: false,
            data_only: false,
            alternative_inputs: Vec::new(),
            unsupported: None,
            placement: None,
            pins: Arc::new(|_| NodePins::default()),
            build: Arc::new(build),
            checks: Vec::new(),
        }
    }

    /// Joins preset graphs at `point`, `order` among the point's nodes. A
    /// preset chains its nodes by flow pins, so a placed node needs them
    /// (`with_generated_pins`).
    pub fn at(mut self, point: ExtensionPoint, order: i32) -> Self {
        self.placement = Some((point, order));
        self
    }

    /// Adds a check run on the passes the node added, once it has built.
    pub fn with_check(mut self, check: CheckFn) -> Self {
        self.checks.push(check);
        self
    }

    pub fn with_param(mut self, param: ParamDesc) -> Self {
        self.params.push(param);
        self
    }

    pub fn with_pins(mut self, inputs: Vec<PinDesc>, outputs: Vec<PinDesc>) -> Self {
        self.pins = Arc::new(move |_| NodePins {
            inputs: inputs.clone(),
            outputs: outputs.clone(),
        });
        self
    }

    /// The pins are generated from what the node's passes declare.
    pub fn with_generated_pins(mut self) -> Self {
        self.generated = true;
        self
    }

    /// The node only supplies data (host inputs, SceneDB buffers): no flow pins.
    pub fn with_data_only(mut self) -> Self {
        self.data_only = true;
        self
    }

    /// A performance marker is inserted after this node by the registry's marker.
    pub fn with_marker_after(mut self) -> Self {
        self.marker_after = true;
        self
    }

    /// Registers why this node builds nothing where the capability it needs
    /// is absent; validation reports it as a warning.
    pub fn with_unsupported(mut self, reason: &'static str) -> Self {
        self.unsupported = Some(reason);
        self
    }

    /// [`with_unsupported`](Self::with_unsupported) where `supported` is false.
    pub fn with_unsupported_when(mut self, supported: bool, reason: &'static str) -> Self {
        if !supported {
            self.unsupported = Some(reason);
        }
        self
    }

    /// Resources this node can also read, as optional inputs of type `ty`.
    pub fn with_alternative_inputs(mut self, names: &[&str], ty: PinType) -> Self {
        self.alternative_inputs
            .extend(names.iter().map(|name| (name.to_string(), ty)));
        self
    }

    /// Pins computed from the instance's parameters.
    pub fn with_dynamic_pins(
        mut self,
        pins: impl Fn(&Params) -> NodePins + Send + Sync + 'static,
    ) -> Self {
        self.pins = Arc::new(pins);
        self
    }

    /// `params` with every declared parameter present.
    pub fn effective_params(&self, params: &Params) -> Params {
        let mut out = params.clone();
        for param in &self.params {
            out.entry(param.key.clone())
                .or_insert_with(|| param.default.clone());
        }
        out
    }

    pub fn pins(&self, params: &Params) -> NodePins {
        (self.pins)(&self.effective_params(params))
    }

    /// Adds the node's passes to `build.graph`, then runs its checks on them.
    pub fn run(&self, build: &mut NodeBuild<'_, '_>) -> Result<(), String> {
        let first = build.graph.pass_count();
        (self.build)(build)?;
        let added = first..build.graph.pass_count();
        for check in &self.checks {
            check(build.env, build.graph, added.clone())?;
        }
        Ok(())
    }
}

/// The node types available to editors and the compiler.
#[derive(Clone, Default)]
pub struct NodeRegistry {
    nodes: BTreeMap<String, Arc<NodeDescriptor>>,
    /// The feature that registered each node kind ([`crate::Feature::id`]),
    /// for kinds a feature added through its [`crate::FeatureRegistrar`].
    /// A kind registered directly (the core nodes every graph has) has no
    /// entry: no feature owns it.
    owners: BTreeMap<String, &'static str>,
    /// Kinds a host knows exist but did not register, with why (a plugin that
    /// owns them is disabled). Validation names the reason instead of
    /// reporting an unknown kind.
    absent: BTreeMap<String, String>,
    pin_table: PinTable,
    marker: Option<MarkerFn>,
    preparers: Vec<PrepareFn>,
    finishers: Vec<FinishFn>,
    scene_derivations: Vec<helio_core::SceneDerivationFactory>,
    swap_policy: helio_core::SwapPolicy,
    untraced_geometry: Vec<&'static str>,
    provisions: HashMap<TypeId, Vec<Arc<dyn Any + Send + Sync>>>,
}

/// Pins of generated nodes by kind, shared by a registry and its clones so a
/// probe result reaches every descriptor without re-registering them.
pub type PinTable = Arc<std::sync::RwLock<HashMap<String, NodePins>>>;

/// The pins of a generated node: its kind's entry in `table` (flow pins until
/// a probe filled it, none for a data-only node), plus the alternative inputs
/// it can take.
fn generated_pins(table: &PinTable, descriptor: &NodeDescriptor) -> PinsFn {
    let table = Arc::clone(table);
    let kind = descriptor.kind.clone();
    let alternatives = descriptor.alternative_inputs.clone();
    let data_only = descriptor.data_only;
    Arc::new(move |_| {
        let mut pins = table
            .read()
            .ok()
            .and_then(|table| table.get(&kind).cloned())
            .unwrap_or_else(|| {
                if data_only {
                    NodePins::default()
                } else {
                    NodePins {
                        inputs: vec![PinDesc::flow("in")],
                        outputs: vec![PinDesc::flow("out")],
                    }
                }
            });
        for (name, ty) in &alternatives {
            if !pins.inputs.iter().any(|pin| pin.resource == *name) {
                pins.inputs
                    .push(PinDesc::resource(*ty, name.clone()).optional());
            }
        }
        pins
    })
}

/// Adds a performance marker after a node that asked for one; the second
/// argument is the node's kind.
pub type MarkerFn = Arc<dyn Fn(&mut NodeBuild<'_, '_>, &str) + Send + Sync>;

/// Runs once before any node of a graph is built: where the graph's builder
/// turns what it knows (a renderer's config, say) into the settings features
/// read, so a feature never reads the builder's own types.
pub type PrepareFn = Arc<dyn Fn(&mut BuildEnv<'_>) + Send + Sync>;

/// Runs once every node of a graph is built: where features that meet are
/// joined (a terrain's ground handed to foliage placement, say), since each
/// node only sees the graph built before it.
pub type FinishFn =
    Arc<dyn Fn(&mut BuildEnv<'_>, &mut RenderGraph) -> Result<(), String> + Send + Sync>;

impl NodeRegistry {
    /// An empty registry (`helio_graph_asset::core_registry` adds the
    /// resource, scene-input and output nodes every graph needs).
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds or replaces a node type that no feature owns (a core node every
    /// graph has). A feature adds its nodes through
    /// [`crate::FeatureRegistrar::node`], which records it as their owner.
    pub fn register(&mut self, descriptor: NodeDescriptor) {
        self.owners.remove(&descriptor.kind);
        self.insert(descriptor);
    }

    /// Adds or replaces a node type registered by feature `feature`.
    pub(crate) fn register_owned(&mut self, descriptor: NodeDescriptor, feature: &'static str) {
        self.owners.insert(descriptor.kind.clone(), feature);
        self.insert(descriptor);
    }

    /// The feature that registered the node of `kind` (`None`: a core node,
    /// or no such node).
    pub fn feature_of(&self, kind: &str) -> Option<&'static str> {
        self.owners.get(kind).copied()
    }

    /// Every feature's node kinds, by feature id, each list in kind order:
    /// which feature contributed which nodes (a later feature that replaces
    /// a kind owns it).
    pub fn feature_nodes(&self) -> BTreeMap<&'static str, Vec<&str>> {
        let mut by_feature: BTreeMap<&'static str, Vec<&str>> = BTreeMap::new();
        for (kind, feature) in &self.owners {
            by_feature.entry(feature).or_default().push(kind.as_str());
        }
        by_feature
    }

    /// Records that node `kind` is not registered because of `reason` (a plugin
    /// that owns it is disabled): a graph naming it reports the reason (see
    /// [`Self::absence`]). A kind that is registered is not absent.
    pub fn mark_absent(&mut self, kind: impl Into<String>, reason: impl Into<String>) {
        let kind = kind.into();
        if !self.nodes.contains_key(&kind) {
            self.absent.insert(kind, reason.into());
        }
    }

    /// Why node `kind` is not registered, when a host said ([`Self::mark_absent`]).
    pub fn absence(&self, kind: &str) -> Option<&str> {
        self.absent.get(kind).map(String::as_str)
    }

    fn insert(&mut self, mut descriptor: NodeDescriptor) {
        self.absent.remove(&descriptor.kind);
        if descriptor.generated {
            descriptor.pins = generated_pins(&self.pin_table, &descriptor);
        }
        self.nodes
            .insert(descriptor.kind.clone(), Arc::new(descriptor));
    }

    /// Whether a probe has given the generated node of `kind` its pins.
    pub fn has_pins(&self, kind: &str) -> bool {
        self.pin_table
            .read()
            .is_ok_and(|table| table.contains_key(kind))
    }

    /// This registry, except that a generated node whose kind no probe has
    /// given pins takes them from `cached` (a graph asset's record of the pins
    /// its nodes had when it was written), so a wired graph validates and
    /// builds where nothing has probed, as in a game. Probed pins win, so a
    /// wire the passes no longer declare is still reported. A plain clone when
    /// `cached` adds nothing.
    pub fn with_pin_fallback(&self, cached: &BTreeMap<String, NodePins>) -> Self {
        let missing: Vec<(&String, &NodePins)> = cached
            .iter()
            .filter(|(kind, _)| {
                self.nodes.get(*kind).is_some_and(|node| node.generated) && !self.has_pins(kind)
            })
            .collect();
        if missing.is_empty() {
            return self.clone();
        }
        let mut table = self
            .pin_table
            .read()
            .map(|table| table.clone())
            .unwrap_or_default();
        for (kind, pins) in missing {
            table.insert(kind.clone(), pins.clone());
        }
        let mut view = self.clone();
        view.pin_table = Arc::new(std::sync::RwLock::new(table));
        for node in view.nodes.values_mut() {
            if node.generated {
                let mut descriptor = NodeDescriptor::clone(node);
                descriptor.pins = generated_pins(&view.pin_table, &descriptor);
                *node = Arc::new(descriptor);
            }
        }
        view
    }

    /// The probed pins of each generated kind of `kinds`: what a graph asset
    /// records of its nodes (`GraphAsset::pins`). Kinds no probe has given
    /// pins are left out.
    pub fn probed_pins<'k>(
        &self,
        kinds: impl IntoIterator<Item = &'k str>,
    ) -> BTreeMap<String, NodePins> {
        let Ok(table) = self.pin_table.read() else {
            return BTreeMap::new();
        };
        kinds
            .into_iter()
            .filter(|kind| self.nodes.get(*kind).is_some_and(|node| node.generated))
            .filter_map(|kind| Some((kind.to_owned(), table.get(kind)?.clone())))
            .collect()
    }

    /// Sets the pins of the generated node of `kind`.
    pub fn set_pins(&self, kind: &str, pins: NodePins) {
        if let Ok(mut table) = self.pin_table.write() {
            table.insert(kind.to_string(), pins);
        }
    }

    /// Every generated node's pins, in kind order.
    pub fn generated_pins(&self) -> Vec<NodePins> {
        self.pin_table.read().map_or_else(
            |_| Vec::new(),
            |table| {
                let mut kinds: Vec<_> = table.iter().collect();
                kinds.sort_by(|a, b| a.0.cmp(b.0));
                kinds.into_iter().map(|(_, pins)| pins.clone()).collect()
            },
        )
    }

    /// Whether any generated pins have been set (a probe has run).
    pub fn has_generated_pins(&self) -> bool {
        self.pin_table
            .read()
            .map_or(false, |table| !table.is_empty())
    }

    /// Sets what adds a performance marker after nodes built `with_marker_after`.
    pub fn set_marker(&mut self, marker: MarkerFn) {
        self.marker = Some(marker);
    }

    pub fn marker(&self) -> Option<&MarkerFn> {
        self.marker.as_ref()
    }

    /// Adds a step run after every node of a graph is built.
    pub fn add_finisher(&mut self, finish: FinishFn) {
        self.finishers.push(finish);
    }

    /// Adds a step run before any node of a graph is built.
    pub fn add_preparer(&mut self, prepare: PrepareFn) {
        self.preparers.push(prepare);
    }

    /// Adds a scene derivation the registry's graphs bring (see
    /// [`helio_core::SceneDerivationFactory`]); a later one with the same
    /// name replaces the earlier.
    pub fn add_scene_derivation(&mut self, derivation: helio_core::SceneDerivationFactory) {
        self.scene_derivations
            .retain(|existing| existing.name != derivation.name);
        self.scene_derivations.push(derivation);
    }

    /// The scene derivations the features registered, in registration order.
    pub fn scene_derivations(&self) -> &[helio_core::SceneDerivationFactory] {
        &self.scene_derivations
    }

    /// Declares a scene buffer of drawn geometry that the scene's ray tracing
    /// does not see (see [`crate::FeatureRegistrar::untraced_geometry`]).
    pub fn add_untraced_geometry(&mut self, buffer: &'static str) {
        if !self.untraced_geometry.contains(&buffer) {
            self.untraced_geometry.push(buffer);
        }
    }

    /// The scene buffers of drawn geometry the scene's ray tracing does not
    /// see, as the features declared them.
    pub fn untraced_geometry(&self) -> &[&'static str] {
        &self.untraced_geometry
    }

    /// Adds a `value` of a type a frame contract defines (a lighting model,
    /// say); whoever reads the contract finds it with [`Self::provided`].
    /// The registry holds it by type and knows nothing of it.
    pub fn provide<T: Any + Send + Sync>(&mut self, value: T) {
        self.provisions
            .entry(TypeId::of::<T>())
            .or_default()
            .push(Arc::new(value));
    }

    /// The values of type `T` the features provided, in registration order.
    pub fn provided<T: Any>(&self) -> impl Iterator<Item = &T> {
        self.provisions
            .get(&TypeId::of::<T>())
            .into_iter()
            .flatten()
            .filter_map(|value| value.downcast_ref::<T>())
    }

    /// Declares what of the features' passes shader hot reload may swap
    /// without rebuilding the graph (see [`helio_core::SwapPolicy`]).
    pub fn add_swap_policy(&mut self, policy: &helio_core::SwapPolicy) {
        self.swap_policy = std::mem::take(&mut self.swap_policy).merged(policy);
    }

    /// What the features declared swappable.
    pub fn swap_policy(&self) -> &helio_core::SwapPolicy {
        &self.swap_policy
    }

    /// The steps run before any node is built, in registration order.
    pub fn preparers(&self) -> &[PrepareFn] {
        &self.preparers
    }

    /// The steps run after every node is built, in registration order.
    pub fn finishers(&self) -> &[FinishFn] {
        &self.finishers
    }

    /// The node types placed at `point`, by order then kind.
    pub fn placed(&self, point: ExtensionPoint) -> Vec<Arc<NodeDescriptor>> {
        let mut nodes: Vec<_> = self
            .nodes
            .values()
            .filter(|d| d.placement.is_some_and(|(p, _)| p == point))
            .cloned()
            .collect();
        nodes.sort_by(|a, b| {
            (a.placement.map(|(_, o)| o), &a.kind).cmp(&(b.placement.map(|(_, o)| o), &b.kind))
        });
        nodes
    }

    pub fn get(&self, kind: &str) -> Option<&Arc<NodeDescriptor>> {
        self.nodes.get(kind)
    }

    /// Every node type, ordered by category then title.
    pub fn list(&self) -> Vec<Arc<NodeDescriptor>> {
        let mut nodes: Vec<_> = self.nodes.values().cloned().collect();
        nodes.sort_by(|a, b| (&a.category, &a.title).cmp(&(&b.category, &b.title)));
        nodes
    }
}
