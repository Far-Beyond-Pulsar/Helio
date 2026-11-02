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
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NodePins {
    pub inputs: Vec<PinDesc>,
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
    pub cull_stats_buffer: &'e wgpu::Buffer,
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
        cull_stats_buffer: &'e wgpu::Buffer,
        owns_device: bool,
    ) -> Self {
        Self {
            device,
            queue,
            frame,
            camera_buffer,
            debug_camera_buffer,
            cull_stats_buffer,
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

/// Where in the frame a feature's node runs: the spine's extension points.
/// A preset graph places the nodes registered for each slot there (by order,
/// then kind), so a feature joins the pipeline without the preset naming it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Slot {
    /// After the spine's G-buffer is complete, before decals: G-buffer
    /// contributors that need its depth (voxel terrain) and the tracers that
    /// read it (GI probe updates and their tracers).
    Geometry,
    /// After decals, before deferred lighting: what lighting reads that is
    /// not the G-buffer (GI irradiance, the clouds' sun visibility).
    LightingInputs,
    /// After the scene is lit, before the overlays, fog and transparency:
    /// resolves that write the lit colour and keep the pixels they do not
    /// touch.
    Lighting,
    /// After everything drawn over the final image: consumers of the
    /// finished frame (captures, readbacks).
    Output,
}

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
    pub slot: Option<(Slot, i32)>,
    pins: PinsFn,
    build: BuildFn,
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
            slot: None,
            pins: Arc::new(|_| NodePins::default()),
            build: Arc::new(build),
        }
    }

    /// Joins preset graphs at `slot`, `order` among the slot's nodes.
    pub fn in_slot(mut self, slot: Slot, order: i32) -> Self {
        self.slot = Some((slot, order));
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

    /// Adds the node's passes to `build.graph`.
    pub fn run(&self, build: &mut NodeBuild<'_, '_>) -> Result<(), String> {
        (self.build)(build)
    }
}

/// The node types available to editors and the compiler.
#[derive(Clone, Default)]
pub struct NodeRegistry {
    nodes: BTreeMap<String, Arc<NodeDescriptor>>,
    pin_table: PinTable,
    marker: Option<MarkerFn>,
    finishers: Vec<FinishFn>,
}

/// Pins of generated nodes by kind, shared by a registry and its clones so a
/// probe result reaches every descriptor without re-registering them.
pub type PinTable = Arc<std::sync::RwLock<HashMap<String, NodePins>>>;

/// Adds a performance marker after a node that asked for one; the second
/// argument is the node's kind.
pub type MarkerFn = Arc<dyn Fn(&mut NodeBuild<'_, '_>, &str) + Send + Sync>;

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

    /// Adds or replaces a node type.
    pub fn register(&mut self, mut descriptor: NodeDescriptor) {
        if descriptor.generated {
            let table = Arc::clone(&self.pin_table);
            let kind = descriptor.kind.clone();
            let alternatives = descriptor.alternative_inputs.clone();
            let data_only = descriptor.data_only;
            descriptor.pins = Arc::new(move |_| {
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
            });
        }
        self.nodes
            .insert(descriptor.kind.clone(), Arc::new(descriptor));
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

    /// The steps run after every node is built, in registration order.
    pub fn finishers(&self) -> &[FinishFn] {
        &self.finishers
    }

    /// The node types registered for `slot`, by order then kind.
    pub fn slotted(&self, slot: Slot) -> Vec<Arc<NodeDescriptor>> {
        let mut nodes: Vec<_> = self
            .nodes
            .values()
            .filter(|d| d.slot.is_some_and(|(s, _)| s == slot))
            .cloned()
            .collect();
        nodes.sort_by(|a, b| {
            (a.slot.map(|(_, o)| o), &a.kind).cmp(&(b.slot.map(|(_, o)| o), &b.kind))
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
