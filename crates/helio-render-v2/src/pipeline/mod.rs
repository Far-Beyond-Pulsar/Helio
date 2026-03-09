//! Pipeline management system

mod cache;
mod spec;
mod descriptor;

pub use cache::PipelineCache;
pub use spec::{PipelineVariant, MaterialDomain, ShadingModel, BlendMode, VertexFactory, MaterialKey};
pub use descriptor::PipelineDescriptor;
