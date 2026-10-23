//! Node types: what a node looks like (pins, parameters) and how it adds its
//! passes to a graph.
//!
//! This crate knows no pass. Pass crates, and crates that compose passes
//! (`helio-default-passes`), register [`NodeDescriptor`]s for what they own;
//! the editor lists whatever is registered.

use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use helio::{DebugDrawState, RendererConfig, SceneDbHandle};
use helio_core::RenderGraph;
use serde_json::Value;

use crate::asset::Params;
use crate::pins::{PinDesc, PinType};

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
        Self { key: key.into(), label: label.into(), ty, default }
    }

    pub fn text(key: &str, label: &str, default: &str) -> Self {
        Self::new(key, label, ParamType::Text, Value::from(default))
    }

    pub fn boolean(key: &str, label: &str, default: bool) -> Self {
        Self::new(key, label, ParamType::Bool, Value::from(default))
    }

    pub fn int(key: &str, label: &str, min: i64, max: i64, default: i64) -> Self {
        Self::new(key, label, ParamType::Int { min, max }, Value::from(default))
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

/// Things a node build needs from the renderer, plus a place for nodes to
/// share state with each other.
pub struct BuildEnv<'e> {
    pub device: &'e Arc<wgpu::Device>,
    pub queue: &'e Arc<wgpu::Queue>,
    pub config: RendererConfig,
    pub debug_state: Arc<Mutex<DebugDrawState>>,
    pub camera_buffer: &'e wgpu::Buffer,
    pub debug_camera_buffer: &'e wgpu::Buffer,
    pub cull_stats_buffer: &'e wgpu::Buffer,
    pub owns_device: bool,
    pub scene_db: SceneDbHandle,
    shared: HashMap<TypeId, Box<dyn Any>>,
}

impl<'e> BuildEnv<'e> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &'e Arc<wgpu::Device>,
        queue: &'e Arc<wgpu::Queue>,
        config: RendererConfig,
        debug_state: Arc<Mutex<DebugDrawState>>,
        camera_buffer: &'e wgpu::Buffer,
        debug_camera_buffer: &'e wgpu::Buffer,
        cull_stats_buffer: &'e wgpu::Buffer,
        owns_device: bool,
        scene_db: SceneDbHandle,
    ) -> Self {
        Self {
            device,
            queue,
            config,
            debug_state,
            camera_buffer,
            debug_camera_buffer,
            cull_stats_buffer,
            owns_device,
            scene_db,
            shared: HashMap::new(),
        }
    }

    /// Render size passes are prepared at (`render_scale` applied).
    pub fn internal_size(&self) -> (u32, u32) {
        (self.config.internal_width(), self.config.internal_height())
    }

    /// The shared value of type `T`, if an earlier node created it.
    pub fn shared_existing<T: Any>(&mut self) -> Option<&mut T> {
        self.shared.get_mut(&TypeId::of::<T>()).and_then(|value| value.downcast_mut::<T>())
    }

    /// State shared between the nodes of one build, created on first use by
    /// whichever node needs it. Keyed by type, so a pass crate's node types
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
        self.params.get(key).and_then(Value::as_str).unwrap_or_default().to_string()
    }

    pub fn int(&self, key: &str) -> i64 {
        self.params.get(key).and_then(Value::as_i64).unwrap_or_default()
    }

    pub fn boolean(&self, key: &str) -> bool {
        self.params.get(key).and_then(Value::as_bool).unwrap_or_default()
    }
}

pub type PinsFn = Arc<dyn Fn(&Params) -> NodePins + Send + Sync>;
pub type BuildFn = Arc<dyn Fn(&mut NodeBuild<'_, '_>) -> Result<(), String> + Send + Sync>;

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
            pins: Arc::new(|_| NodePins::default()),
            build: Arc::new(build),
        }
    }

    pub fn with_param(mut self, param: ParamDesc) -> Self {
        self.params.push(param);
        self
    }

    pub fn with_pins(mut self, inputs: Vec<PinDesc>, outputs: Vec<PinDesc>) -> Self {
        self.pins = Arc::new(move |_| NodePins { inputs: inputs.clone(), outputs: outputs.clone() });
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

    /// Resources this node can also read, as optional inputs of type `ty`.
    pub fn with_alternative_inputs(mut self, names: &[&str], ty: PinType) -> Self {
        self.alternative_inputs.extend(names.iter().map(|name| (name.to_string(), ty)));
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
            out.entry(param.key.clone()).or_insert_with(|| param.default.clone());
        }
        out
    }

    pub fn pins(&self, params: &Params) -> NodePins {
        (self.pins)(&self.effective_params(params))
    }

    pub(crate) fn run(&self, build: &mut NodeBuild<'_, '_>) -> Result<(), String> {
        (self.build)(build)
    }
}

/// The node types available to editors and the compiler.
#[derive(Clone, Default)]
pub struct NodeRegistry {
    nodes: BTreeMap<String, Arc<NodeDescriptor>>,
    pin_table: PinTable,
    marker: Option<MarkerFn>,
}

/// Pins of generated nodes by kind, shared by a registry and its clones so a
/// probe result reaches every descriptor without re-registering them.
pub type PinTable = Arc<std::sync::RwLock<HashMap<String, NodePins>>>;

/// Adds a performance marker after a node that asked for one; the second
/// argument is the node's kind.
pub type MarkerFn = Arc<dyn Fn(&mut NodeBuild<'_, '_>, &str) + Send + Sync>;

impl NodeRegistry {
    /// An empty registry. Use [`NodeRegistry::with_core`] for the resource,
    /// scene-input and output nodes every graph needs.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry holding the core nodes ([`crate::core_nodes`]).
    pub fn with_core() -> Self {
        let mut registry = Self::new();
        crate::core_nodes::register(&mut registry);
        registry
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
                        pins.inputs.push(PinDesc::resource(*ty, name.clone()).optional());
                    }
                }
                pins
            });
        }
        self.nodes.insert(descriptor.kind.clone(), Arc::new(descriptor));
    }

    /// Sets the pins of the generated node of `kind`.
    pub fn set_pins(&self, kind: &str, pins: NodePins) {
        if let Ok(mut table) = self.pin_table.write() {
            table.insert(kind.to_string(), pins);
        }
    }

    /// Every generated node's pins, in kind order.
    pub fn generated_pins(&self) -> Vec<NodePins> {
        self.pin_table.read().map_or_else(|_| Vec::new(), |table| {
            let mut kinds: Vec<_> = table.iter().collect();
            kinds.sort_by(|a, b| a.0.cmp(b.0));
            kinds.into_iter().map(|(_, pins)| pins.clone()).collect()
        })
    }

    /// Whether any generated pins have been set (a probe has run).
    pub fn has_generated_pins(&self) -> bool {
        self.pin_table.read().map_or(false, |table| !table.is_empty())
    }

    /// Sets what adds a performance marker after nodes built `with_marker_after`.
    pub fn set_marker(&mut self, marker: MarkerFn) {
        self.marker = Some(marker);
    }

    pub fn marker(&self) -> Option<&MarkerFn> {
        self.marker.as_ref()
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
