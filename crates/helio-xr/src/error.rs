//! Error type for helio-xr.

use thiserror::Error;

/// Errors produced by the helio-xr crate.
#[derive(Debug, Error)]
pub enum XrError {
    /// The OpenXR runtime returned an error result.
    #[error("OpenXR error: {0}")]
    OpenXr(#[from] openxr::sys::Result),

    /// The OpenXR loader could not be found or was invalid.
    #[error("failed to load the OpenXR runtime: {0}")]
    Load(String),

    /// The platform/runtime does not support what was requested.
    #[error("{0}")]
    Platform(String),

    /// The wgpu device has no usable Vulkan backend, so raw handles cannot be
    /// extracted for OpenXR.
    #[error("no usable Vulkan backend for OpenXR: {0}")]
    GraphicsUnavailable(String),

    /// The requested texture format is not offered by the runtime for an
    /// XR swapchain.
    #[error("texture format {0:?} is not usable for an OpenXR swapchain")]
    UnsupportedFormat(wgpu::TextureFormat),

    /// The swapchain could not be created (including the single-layer retry).
    #[error("could not create the OpenXR swapchain: {0}")]
    Swapchain(String),

    /// A `VkImage` from the runtime could not be wrapped as a wgpu texture.
    #[error("could not wrap an OpenXR swapchain image as a wgpu texture: {0}")]
    TextureWrap(String),
}
