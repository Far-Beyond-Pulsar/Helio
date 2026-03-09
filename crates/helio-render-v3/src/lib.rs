//! helio-render-v3 — ground-up renderer rewrite.
//!
//! Architecture highlights:
//! - **HISM** (Hierarchical Instanced Static Mesh) batching via stable `HismHandle(u32)`.
//! - **Persistent scene state** — the renderer owns CPU+GPU scene state.  Call
//!   `add_instance`, `remove_instance`, `set_instance_transform`, `add_light`,
//!   `remove_light`, `update_light`, `set_sky`, `set_skylight` to mutate it;
//!   only changed handles are re-uploaded.
//! - **Delta-only GPU uploads** — at steady state: zero alloc, zero buffer
//!   allocation, just a 144-byte camera write + render graph execute.
//! - **Incremental draw-list maintenance** — `flush_dirty_handles` touches only
//!   the instance buffers whose contents actually changed.
//! - **`override` shader constants** baked per pipeline — zero runtime branching.

pub mod camera;
pub mod mesh;
pub mod material;
pub mod hism;
pub mod scene;
pub mod debug_draw;
pub mod profiler;
pub mod graph;
pub mod pipeline;
pub mod resources;
pub mod passes;
pub mod renderer;

// ── Public surface ─────────────────────────────────────────────────────────────

pub use camera::{Camera, GlobalsUniform};
pub use mesh::{PackedVertex, GpuMesh, DrawCall, INSTANCE_STRIDE};
pub use material::{Material, GpuMaterial, BlendMode, MaterialUniform};
pub use hism::{HismHandle, HismRegistry};
pub use scene::{InstanceId, LightId, SceneLight, LightType, SkyAtmosphere, Skylight, GpuLight};
pub use debug_draw::{DebugShape, DebugDrawBatch, DebugVertex};
pub use profiler::{GpuProfiler, CpuFrameStats};
pub use resources::{FrameTextures, StubTextures};
pub use renderer::{Renderer, RendererConfig, AntiAliasingMode, BloomConfig, SsaoConfig, RendererError};
pub use passes::shadow::ShadowConfig;
pub use passes::radiance_cascades::RcConfig;
pub use passes::billboard::BillboardConfig;

// ── Crate-level error type ─────────────────────────────────────────────────────

use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("surface error: {0}")]
    Surface(#[from] wgpu::SurfaceError),
    #[error("wgpu device request error: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("resize with zero dimension")]
    ZeroSize,
    #[error("render error: {0}")]
    Renderer(#[from] RendererError),
}

pub type Result<T> = std::result::Result<T, Error>;
