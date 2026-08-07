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
    /// The capacity is selected from the device's complete material binding tier.
    #[error("scene texture capacity exceeded for the active material binding tier")]
    TextureCapacityExceeded,

    /// An operation was rejected because of an incompatible resource state.
    #[error("invalid operation: {reason}")]
    InvalidOperation {
        /// Human-readable description of why the operation was rejected.
        reason: &'static str,
    },

    /// All GPU coordinate-space slots are in use.
    ///
    /// Sublevels and portals share one small fixed-size GPU buffer
    /// (`libhelio::MAX_COORDINATE_SPACES` slots, slot 0 reserved for world
    /// space). Remove an existing sublevel/portal before adding another once
    /// this is hit — this is not expected to occur in normal use.
    #[error("coordinate space capacity exceeded (all sublevel/portal slots in use)")]
    CoordinateSpaceCapacityExceeded,
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
