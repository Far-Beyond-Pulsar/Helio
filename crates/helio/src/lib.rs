//! High-level facade over `helio-v3`.
//!
//! This crate restores a stable handle-based scene API on top of the lower-level
//! GPU-native core. Scene mutations stay O(1) with respect to scene size on the
//! CPU side by using:
//!
//! - generational handles for public resources,
//! - sparse slots for stable-index resources like materials,
//! - dense swap-remove arenas for objects and lights,
//! - partial dirty-range uploads to `helio-v3` managers.

mod arena;
mod forward;
mod handles;
mod mesh;
mod renderer;
mod scene;

pub use handles::{LightId, MaterialId, MeshId, ObjectId};
pub use mesh::{MeshBuffers, MeshSlice, MeshUpload, PackedVertex};
pub use renderer::{Renderer, RendererConfig};
pub use scene::{Camera, ObjectDescriptor, Result as SceneResult, Scene, SceneError};

pub use helio_v3::{
    DrawIndexedIndirectArgs, Error, GpuCameraUniforms, GpuDrawCall, GpuInstanceAabb,
    GpuInstanceData, GpuLight, GpuMaterial, GpuScene, RenderGraph, RenderPass, Result,
};
