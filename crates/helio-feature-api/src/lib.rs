//! The contract between Helio and its rendering features.
//!
//! A [`Feature`] registers node types ([`NodeDescriptor`]) that build its
//! passes into a render graph, and declares the buffer contracts those passes
//! read. Node builds get a [`BuildEnv`]: frame facts plus typed settings the
//! graph's builder supplies. This crate depends on `helio-core` only, so a
//! feature (or an engine plugin carrying one) does not link the renderer.

mod feature;
mod pins;
mod registry;
mod settings;

pub use feature::{ContractUse, Feature, FeatureRegistrar, FeatureSet};
pub use pins::{PinDesc, PinKind, PinShape, PinType};
pub use registry::{
    BuildEnv, BuildFn, FrameDesc, MarkerFn, NodeBuild, NodeDescriptor, NodePins, NodeRegistry,
    ParamDesc, ParamType, Params, PinTable, PinsFn,
};
pub use settings::FeatureSettings;
