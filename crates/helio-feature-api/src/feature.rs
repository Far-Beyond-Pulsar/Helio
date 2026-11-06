//! A rendering feature: the unit an engine plugin adds to Helio.
//!
//! A feature owns passes, the node types that build them, the buffer
//! contracts its passes read and the settings they take. It registers all of
//! that through a [`FeatureRegistrar`]; the renderer and the graph compiler
//! only ever see what was registered. Features never import each other's
//! implementation: they meet at buffer contracts and shared frame resources
//! (docs/architecture/layered-rewrite.md §3).

use std::sync::Arc;

use helio_core::BufferContract;

use crate::registry::{MarkerFn, NodeDescriptor, NodeRegistry};

/// A rendering feature (`helio.sky`, `helio.gi`, `helio.voxel` …).
pub trait Feature: Send + Sync + 'static {
    /// Stable id, used in diagnostics and by engines to name the feature.
    fn id(&self) -> &'static str;

    /// Registers what the feature contributes. Called once per registry.
    fn register(&self, registrar: &mut FeatureRegistrar<'_>);
}

/// A buffer contract a feature's passes read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ContractUse {
    pub feature: &'static str,
    pub key: &'static str,
    pub row_bytes: u64,
}

/// What a feature registers into.
pub struct FeatureRegistrar<'r> {
    feature: &'static str,
    registry: &'r mut NodeRegistry,
    contracts: &'r mut Vec<ContractUse>,
}

impl FeatureRegistrar<'_> {
    /// The id of the feature registering.
    pub fn feature(&self) -> &'static str {
        self.feature
    }

    /// Adds (or replaces) a node type.
    pub fn node(&mut self, descriptor: NodeDescriptor) {
        self.registry.register(descriptor);
    }

    /// Adds a step run before any node of a graph is built (see
    /// [`crate::PrepareFn`]).
    pub fn prepare(&mut self, prepare: crate::PrepareFn) {
        self.registry.add_preparer(prepare);
    }

    /// Adds a step run once every node of a graph is built (see
    /// [`crate::FinishFn`]).
    pub fn finish(&mut self, finish: crate::FinishFn) {
        self.registry.add_finisher(finish);
    }

    /// Adds a scene derivation the feature's graphs bring: made once per
    /// renderer and run after the host's derivations (see
    /// [`helio_core::SceneDerivationFactory`]).
    pub fn scene_derivation(&mut self, derivation: helio_core::SceneDerivationFactory) {
        self.registry.add_scene_derivation(derivation);
    }

    /// Declares `buffer`, a scene buffer of geometry the feature draws, as
    /// geometry the scene's ray tracing does not see (a kind with no
    /// acceleration-structure projection: foliage, voxel volumes). A ray
    /// tracing projection refuses a scene that holds such rows rather than
    /// silently tracing a scene without them (`Renderer::untraced_geometry`).
    pub fn untraced_geometry(&mut self, buffer: &'static str) {
        self.registry.add_untraced_geometry(buffer);
    }

    /// Declares a pass type of the feature that shares no builder-created
    /// resource with another pass: shader hot reload swaps it alone instead
    /// of rebuilding the graph (see [`helio_core::SwapPolicy`]).
    pub fn swappable<T: ?Sized>(&mut self) {
        self.registry
            .add_swap_policy(&helio_core::SwapPolicy::new().independent::<T>());
    }

    /// Declares pass types (`helio_core::graph::type_name_of`) that share
    /// builder-created resources: hot reload swaps them together.
    pub fn swap_group(&mut self, members: &[&'static str]) {
        self.registry
            .add_swap_policy(&helio_core::SwapPolicy::new().group_types(members));
    }

    /// Sets what adds a performance marker after nodes built
    /// `with_marker_after`.
    pub fn marker(&mut self, marker: MarkerFn) {
        self.registry.set_marker(marker);
    }

    /// Records that the feature's passes read `contract`, so an engine can
    /// check it produces every buffer its features consume.
    pub fn consumes<R: 'static>(&mut self, contract: BufferContract<R>) {
        self.contracts.push(ContractUse {
            feature: self.feature,
            key: contract.name(),
            row_bytes: BufferContract::<R>::ROW_BYTES,
        });
    }
}

/// An ordered list of features: what an engine (or a preset) renders with.
#[derive(Clone, Default)]
pub struct FeatureSet {
    features: Vec<Arc<dyn Feature>>,
}

impl FeatureSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds `feature`; a later feature with the same id replaces the earlier.
    pub fn with(mut self, feature: impl Feature) -> Self {
        self.add(Arc::new(feature));
        self
    }

    pub fn add(&mut self, feature: Arc<dyn Feature>) {
        self.features.retain(|f| f.id() != feature.id());
        self.features.push(feature);
    }

    /// This set with `other`'s features after its own (an engine's features
    /// added to a preset's list).
    pub fn extended(mut self, other: &FeatureSet) -> Self {
        for feature in &other.features {
            self.add(feature.clone());
        }
        self
    }

    pub fn ids(&self) -> Vec<&'static str> {
        self.features.iter().map(|f| f.id()).collect()
    }

    pub fn contains(&self, id: &str) -> bool {
        self.features.iter().any(|f| f.id() == id)
    }

    /// Registers every feature into `registry`, in order, and returns the
    /// buffer contracts they read.
    pub fn register_into(&self, registry: &mut NodeRegistry) -> Vec<ContractUse> {
        let mut contracts = Vec::new();
        for feature in &self.features {
            feature.register(&mut FeatureRegistrar {
                feature: feature.id(),
                registry,
                contracts: &mut contracts,
            });
        }
        contracts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Probe(&'static str, &'static str);

    impl Feature for Probe {
        fn id(&self) -> &'static str {
            self.0
        }

        fn register(&self, registrar: &mut FeatureRegistrar<'_>) {
            registrar.node(NodeDescriptor::new(self.1, "Probe", "Test", "", |_| Ok(())));
            registrar.consumes(BufferContract::<[f32; 4]>::new("probe_rows"));
        }
    }

    #[test]
    fn features_register_nodes_and_record_contracts() {
        let set = FeatureSet::new()
            .with(Probe("a", "a.node"))
            .with(Probe("b", "b.node"));
        let mut registry = NodeRegistry::new();
        let contracts = set.register_into(&mut registry);
        assert!(registry.get("a.node").is_some() && registry.get("b.node").is_some());
        assert_eq!(
            contracts,
            vec![
                ContractUse {
                    feature: "a",
                    key: "probe_rows",
                    row_bytes: 16
                },
                ContractUse {
                    feature: "b",
                    key: "probe_rows",
                    row_bytes: 16
                },
            ]
        );
    }

    #[test]
    fn a_feature_with_the_same_id_replaces_the_earlier() {
        let set = FeatureSet::new()
            .with(Probe("a", "first"))
            .with(Probe("a", "second"));
        assert_eq!(set.ids(), vec!["a"]);
        let mut registry = NodeRegistry::new();
        set.register_into(&mut registry);
        assert!(registry.get("first").is_none() && registry.get("second").is_some());
    }

    #[test]
    fn placed_nodes_list_by_order_then_kind_and_finishers_keep_their_order() {
        use crate::ExtensionPoint;
        const GEOMETRY: ExtensionPoint = ExtensionPoint("geometry");
        const LIGHTING: ExtensionPoint = ExtensionPoint("lighting");
        let node = |kind: &str| NodeDescriptor::new(kind, kind, "Test", "", |_| Ok(()));
        let mut registry = NodeRegistry::new();
        registry.register(node("late").at(GEOMETRY, 20));
        registry.register(node("b").at(GEOMETRY, 0));
        registry.register(node("a").at(GEOMETRY, 0));
        registry.register(node("lighting").at(LIGHTING, 0));
        registry.register(node("named"));
        let kinds = |point| {
            registry
                .placed(point)
                .iter()
                .map(|d| d.kind.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(kinds(GEOMETRY), ["a", "b", "late"]);
        assert_eq!(kinds(LIGHTING), ["lighting"]);

        registry.add_finisher(std::sync::Arc::new(|_, _| Ok(())));
        registry.add_finisher(std::sync::Arc::new(|_, _| Err("second".into())));
        assert_eq!(registry.finishers().len(), 2);
    }
}
