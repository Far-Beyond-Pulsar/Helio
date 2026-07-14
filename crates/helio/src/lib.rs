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
mod editor;
mod groups;
mod handles;
mod material;
mod mesh;
mod picking;
mod renderer;
mod scene;
mod vg;

pub use editor::{EditorState, GizmoAxis, GizmoMode};
pub use groups::{GroupId, GroupMask};
pub use handles::{
    LightId, MaterialId, MeshId, MultiMeshId, ObjectId, SectionedInstanceId, TextureId,
    VirtualObjectId, WaterHitboxId, WaterVolumeId,
};
pub use helio_pass_billboard::BillboardInstance;
pub use helio_pass_debug_overlay::{DebugOverlayPass, DebugOverlayState};
pub use helio_pass_perf_overlay::{PerfOverlayMode, PerfOverlayPass};
pub use material::{
    MaterialAsset, MaterialTextureRef, MaterialTextures, TextureSamplerDesc, TextureTransform,
    TextureUpload, MAX_TEXTURES,
};
pub use mesh::{MeshBuffers, MeshSlice, MeshUpload, PackedVertex, SectionedMeshUpload};
pub use picking::{PickHit, ScenePicker};
pub use renderer::{
    build_default_graph_external, build_fxaa_graph, build_fxaa_graph_external,
    build_fxaa_hlfs_graph, build_fxaa_hlfs_graph_external, build_hlfs_graph, build_simple_graph,
    required_wgpu_features, required_wgpu_limits, GiConfig, Renderer, RendererConfig,
};
pub use scene::{
    Camera, ObjectDescriptor, PickableObject, Result as SceneResult, Scene, SceneActor,
    SceneActorId, SceneActorTrait, SceneError, WaterHitboxActor, WaterHitboxDescriptor,
    WaterVolumeActor, WaterVolumeDescriptor,
};
pub use vg::{VirtualMeshId, VirtualMeshLodUpload, VirtualMeshUpload, VirtualObjectDescriptor};

pub use helio_v3::{
    DebugViewDescriptor, DrawIndexedIndirectArgs, Error, GpuCameraUniforms, GpuDrawCall,
    GpuInstanceAabb, GpuInstanceData, GpuLight, GpuMaterial, GpuScene, RenderGraph, RenderPass,
    Result,
};
pub use libhelio::{LightType, Movability, ShadowQuality, SkyActor, VolumetricClouds};
