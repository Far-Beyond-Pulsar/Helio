//! Render graphs as data.
//!
//! A [`GraphAsset`] is nodes and wires. Each node is a [`NodeDescriptor`] from
//! a [`NodeRegistry`] that adds passes to a Helio `RenderGraph`; a wire joins an
//! output pin to an input pin of the same [`PinType`] and resource name.
//! [`GraphAssetPipeline`] compiles an asset on demand and implements
//! [`helio::IntoGraph`], so a renderer can be built from one like from any
//! other pipeline.
//!
//! This crate knows no pass. Registries are filled by the crates that own the
//! passes (`helio-default-graphs` registers Helio's), plus the core resource
//! nodes in [`core_nodes`].

mod asset;
mod compile;
pub mod core_nodes;
mod pins;
mod registry;
pub mod resources;
mod validate;

pub use asset::{
    Connection, GraphAsset, NodeInstance, Params, PinRef, EXTENSION, FORMAT_VERSION,
};
pub use compile::{compile, Fallback, GraphAssetPipeline};
pub use pins::{PinDesc, PinKind, PinShape, PinType};
pub use registry::{
    BuildEnv, BuildFn, NodeBuild, NodeDescriptor, NodePins, NodeRegistry, ParamDesc, ParamType,
    PinsFn,
};
pub use validate::{validate, Diagnostic, Plan, Severity, Validation};
