//! Error types for scene operations.

use thiserror::Error;

/// Error type for scene operations.
///
/// Returned by scene resource management methods when invalid handles are used,
/// resources are still in use, or capacity limits are exceeded.
#[derive(Debug, Error)]
pub enum SceneError {
    /// An invalid handle was used (the resource no longer exists or never existed).
    #[error("invalid {resource} handle")]
    InvalidHandle {
        /// The type of resource that was invalid (e.g., "object", "material", "light").
        resource: &'static str,
    },

    /// A resource cannot be removed because it is still referenced by other resources.
    #[error("{resource} is still in use")]
    ResourceInUse {
        /// The type of resource that is still in use.
        resource: &'static str,
    },

    /// The scene's texture capacity has been exceeded.
    ///
    /// The scene can hold a maximum of [`crate::MAX_TEXTURES`] textures.
    #[error("scene texture capacity exceeded")]
    TextureCapacityExceeded,
}

/// Result type for scene operations.
///
/// Alias for `std::result::Result<T, SceneError>`.
pub type Result<T> = std::result::Result<T, SceneError>;

/// Helper to construct an [`SceneError::InvalidHandle`] error.
///
/// # Example
/// ```ignore
/// return Err(invalid("object"));
/// ```
pub(super) fn invalid(resource: &'static str) -> SceneError {
    SceneError::InvalidHandle { resource }
}

