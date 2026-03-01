//! Helio Render V2 - Production-grade renderer using idiomatic wgpu
//!
//! This is a complete rewrite of the Helio renderer that solves the problems
//! with the previous string-injection based system:
//!
//! - Uses specialization constants instead of shader string injection
//! - Pre-compiled pipeline variants with instant hot-swapping
//! - Shared bind groups for proper resource sharing between features
//! - Render graph with automatic dependency resolution
//! - Resource pooling with aliasing for efficient memory usage
//! - Direct wgpu API (no custom abstractions)

pub mod resources;
pub mod features;
pub mod pipeline;
pub mod graph;
pub mod passes;
pub mod shaders;
pub mod mesh;
pub mod scene;

mod renderer;
mod camera;

pub use renderer::{Renderer, RendererConfig};
pub use camera::Camera;
pub use mesh::{GpuMesh, PackedVertex, DrawCall};
pub use scene::{Scene, SceneObject, SceneLight};

/// Result type for renderer operations
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during rendering
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Feature error: {0}")]
    Feature(String),

    #[error("Pipeline error: {0}")]
    Pipeline(String),

    #[error("Graph error: {0}")]
    Graph(String),

    #[error("Resource error: {0}")]
    Resource(String),

    #[error("Shader error: {0}")]
    Shader(String),

    #[error("WGPU error: {0}")]
    Wgpu(String),
}

impl From<wgpu::Error> for Error {
    fn from(err: wgpu::Error) -> Self {
        Error::Wgpu(err.to_string())
    }
}
