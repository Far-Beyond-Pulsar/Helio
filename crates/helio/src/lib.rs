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
mod groups;
mod handles;
mod material;
mod mesh;
mod quark_commands;
mod renderer;
mod scene;
mod vg;

pub use groups::{GroupId, GroupMask};
pub use handles::{LightId, MaterialId, MeshId, ObjectId, TextureId, VirtualObjectId, WaterHitboxId, WaterVolumeId};
pub use helio_pass_billboard::BillboardInstance;
pub use material::{
    MaterialAsset, MaterialTextureRef, MaterialTextures, TextureSamplerDesc, TextureTransform,
    TextureUpload, MAX_TEXTURES,
};
pub use mesh::{MeshBuffers, MeshSlice, MeshUpload, PackedVertex};
pub use quark_commands::{HelioAction, HelioCommandBridge, register_helio_commands};
pub use renderer::{
    build_simple_graph, build_hlfs_graph, required_wgpu_features, required_wgpu_limits, GiConfig, Renderer,
    RendererConfig,
};
pub use scene::{
    Camera, ObjectDescriptor, Result as SceneResult, Scene, SceneError,
    SceneActor, SceneActorId, SceneActorTrait,
    WaterHitboxActor, WaterHitboxDescriptor,
    WaterVolumeActor, WaterVolumeDescriptor,
};
pub use vg::{VirtualMeshId, VirtualMeshUpload, VirtualObjectDescriptor};

pub use helio_v3::{
    DrawIndexedIndirectArgs, Error, GpuCameraUniforms, GpuDrawCall, GpuInstanceAabb,
    GpuInstanceData, GpuLight, GpuMaterial, GpuScene, RenderGraph, RenderPass, Result,
};
pub use libhelio::{LightType, ShadowQuality, VolumetricClouds, SkyActor};