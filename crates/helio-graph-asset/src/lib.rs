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
//! passes (`helio-default-passes` registers Helio's), plus the core resource
//! nodes in [`core_nodes`].

mod asset;
pub mod buffers;
mod compile;
pub mod core_nodes;
pub mod layout;
mod pins;
mod probe;
mod registry;
pub mod resources;
pub mod sections;
mod validate;

pub use asset::{
    Connection, GraphAsset, GraphComment, GraphSection, NodeInstance, Params, PinRef, SectionPort,
    EXTENSION, FORMAT_VERSION, SECTION_ENTRY, SECTION_EXIT, SECTION_PREFIX,
};
pub use compile::{compile, Fallback, GraphAssetPipeline};
pub use layout::{auto_layout, LANE_PREFIX};
pub use pins::{PinDesc, PinKind, PinShape, PinType};
pub use probe::{output_pin, probe, Probe, SCENEDB_PREFIX};
pub use registry::{
    BuildEnv, BuildFn, MarkerFn, NodeBuild, NodeDescriptor, NodePins, NodeRegistry, ParamDesc,
    ParamType, PinTable, PinsFn,
};
pub use validate::{validate, Diagnostic, Plan, Severity, Validation};
