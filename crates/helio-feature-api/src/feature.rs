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
    fn slotted_nodes_list_by_order_then_kind_and_finishers_keep_their_order() {
        let node = |kind: &str| NodeDescriptor::new(kind, kind, "Test", "", |_| Ok(()));
        let mut registry = NodeRegistry::new();
        registry.register(node("late").in_slot(crate::Slot::Geometry, 20));
        registry.register(node("b").in_slot(crate::Slot::Geometry, 0));
        registry.register(node("a").in_slot(crate::Slot::Geometry, 0));
        registry.register(node("lighting").in_slot(crate::Slot::LightingInputs, 0));
        registry.register(node("named"));
        let kinds = |slot| {
            registry
                .slotted(slot)
                .iter()
                .map(|d| d.kind.clone())
                .collect::<Vec<_>>()
        };
        assert_eq!(kinds(crate::Slot::Geometry), ["a", "b", "late"]);
        assert_eq!(kinds(crate::Slot::LightingInputs), ["lighting"]);

        registry.add_finisher(std::sync::Arc::new(|_, _| Ok(())));
        registry.add_finisher(std::sync::Arc::new(|_, _| Err("second".into())));
        assert_eq!(registry.finishers().len(), 2);
    }
}
