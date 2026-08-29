//! OpenXR (Vulkan) integration for the Helio engine.
//!
//! This crate adapts a wgpu-backed (Vulkan) renderer to an OpenXR session:
//!
//! - [`instance`] / [`XrInstance`]: runtime + system + graphics-requirement
//!   negotiation.
//! - [`session`] / [`XrSession`]: session lifecycle, spaces, event pumping and
//!   the frame begin/end flow.
//! - [`swapchain`] / [`XrSwapchain`]: swapchain creation and wrapping the
//!   runtime's `VkImage`s as wgpu textures.
//! - [`graphics`]: the [`openxr::Graphics`] trait implementation that ties the
//!   Vulkan handles managed by wgpu into OpenXR.
//! - [`camera`]: conversion of per-eye view poses into [`libhelio::GpuCameraUniforms`].
//!
//! The `openxr` crate (0.21) has no built-in wgpu module, so [`graphics`]
//! implements [`openxr::Graphics`] against the raw Vulkan handles extracted
//! through wgpu's `as_hal()` escape hatch and wgpu-hal's `texture_from_raw`.
//! Both are inherently `unsafe`, backend-specific (Vulkan), and wgpu-version
//! specific; the exact wgpu APIs are pinned in `Cargo.lock` via the workspace.
//!
//! The crate only compiles for native targets; OpenXR requires a local runtime
//! and a VR headset, so everything is `#[cfg(not(target_arch = "wasm32"))]`.

#![cfg(not(target_arch = "wasm32"))]

pub mod camera;
pub mod context;
pub mod error;
pub mod graphics;
pub mod input;
pub mod instance;
pub mod session;
pub mod swapchain;

pub use camera::{view_to_world_matrix, xr_view_to_camera, ViewPose};
pub use context::{create_wgpu_device, create_wgpu_instance};
pub use error::XrError;
pub use graphics::WgpuGraphics;
pub use input::{ControllerState, XrInput};
pub use instance::XrInstance;
pub use openxr::Time;
pub use session::{LocatedViews, SessionEvent, XrSession};
pub use swapchain::XrSwapchain;

/// Result alias for helio-xr operations.
pub type Result<T> = std::result::Result<T, XrError>;

/// The concrete OpenXR session handle Helio's graphics binding produces.
///
/// Re-exported so applications can hold a session handle (for
/// [`XrInput::sync`], say) without taking a direct dependency on the `openxr`
/// crate and having to keep its version in lockstep with this one.
/// `openxr::Session` is reference-counted, so cloning one is cheap and the
/// clone stays valid alongside the renderer's copy.
pub type XrSessionHandle = openxr::Session<WgpuGraphics>;
