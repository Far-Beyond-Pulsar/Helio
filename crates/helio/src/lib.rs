//! High-level renderer facade over `helio-core`.
//!
//! Persistent render state is owned by the frontend SceneDB and mirrored into
//! generic GPU component buffers. This crate owns backend GPU machinery and
//! transient render products; it does not own a typed scene container.

mod camera;
#[cfg(not(target_arch = "wasm32"))]
mod graphics_context;
mod quark_commands;
mod renderer;
// SceneDB is the sole scene authority; the legacy scene container was removed.

pub use camera::{follow_origin, Camera, WORLD_ORIGIN_CELL};
/// Unified window, WGPU, and optional OpenXR startup (native targets only).
#[cfg(not(target_arch = "wasm32"))]
pub use graphics_context::{
    HelioGraphicsConfig, HelioGraphicsContext, HelioGraphicsMode, HelioXrInput,
};
pub use helio_core::wgpu_instance_descriptor;
pub use helio_core::{
    Actor, DebugViewDescriptor, DebugViewLegend, DebugViewLegendStop, Entity, Error,
    GpuCameraUniforms, GpuTimingAvailability, Movability, RenderGraph, RenderPass,
    RenderPassTiming, RenderTimingSnapshot, Result, RoutingIssue, RoutingSeverity,
};
pub use helio_frame_api::lighting::AmbientLight;
pub use helio_pass_postprocess::{HdrOutputMode, TonemapOperator};
pub use helio_pass_tsr::TsrQuality;
pub use helio_settings;
pub use quark_commands::{register_helio_commands, HelioAction, HelioCommandBridge};
pub use renderer::{
    ray_queries_usable, recommended_instance_flags, required_experimental_features,
    required_wgpu_features, required_wgpu_limits, usable_adapter_features, DebugBatch,
    DebugCameraUniform, DebugDrawPass, DebugDrawState, DebugVertex, GraphRebuilder, IntoGraph,
    NamedPipeline, PassBuildContext, PassGraphBuilderFn, PerfOverlayMode, PipelineFactory,
    PipelineFailure, PipelineInfo, PipelineStatus, RenderMode,
    RenderPipelineRegistry, Renderer, RendererBuilder, RendererConfig, SceneDbHandle, StereoMirror,
    TsrSetting, WorldPoint, WorldVertex, XrMirrorStats, UNNAMED_PIPELINE,
};
#[cfg(all(feature = "shader-hot-reload", not(target_arch = "wasm32")))]
pub use renderer::{ReloadMode, ShaderReloadStatus};
