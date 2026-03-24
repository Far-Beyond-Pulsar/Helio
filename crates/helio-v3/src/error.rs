//! Error types for helio-v3.
//!
//! This module defines the error types used throughout helio-v3. All functions that can fail
//! return `Result<T>`, where `Result<T> = std::result::Result<T, Error>`.
//!
//! # Design
//!
//! Errors are defined using `thiserror` for ergonomic error handling:
//! - Automatic `Display` implementation
//! - Automatic `Error` trait implementation
//! - Clear error messages with context
//!
//! # Error Categories
//!
//! - **GPU errors**: Device lost, out of memory, etc.
//! - **Shader errors**: Compilation failures, missing entry points
//! - **Resource errors**: Missing buffers, textures, bind groups
//! - **Pass errors**: Invalid configuration, missing dependencies
//! - **Profiling errors**: Query set overflow, readback failures
//!
//! # Example
//!
//! ```rust,no_run
//! use helio_v3::{Result, Error};
//!
//! fn my_function() -> Result<()> {
//!     // ... code that might fail ...
//!     Err(Error::Gpu("Device lost".to_string()))
//! }
//!
//! match my_function() {
//!     Ok(_) => println!("Success"),
//!     Err(e) => eprintln!("Error: {}", e),
//! }
//! ```

use thiserror::Error;

/// Error type for helio-v3.
///
/// All fallible operations return `Result<T>`, where `Result<T> = std::result::Result<T, Error>`.
///
/// # Variants
///
/// - [`Error::Gpu`] - GPU device errors (device lost, out of memory)
/// - [`Error::ShaderCompilation`] - Shader compilation failures
/// - [`Error::ResourceNotFound`] - Missing buffers, textures, bind groups
/// - [`Error::InvalidPassConfig`] - Invalid pass configuration
/// - [`Error::Profiling`] - Profiling system errors
///
/// # Example
///
/// ```rust,no_run
/// use helio_v3::{Result, Error};
///
/// fn create_pipeline() -> Result<wgpu::RenderPipeline> {
///     // ... shader compilation ...
///     Err(Error::ShaderCompilation("Missing vertex shader entry point".to_string()))
/// }
/// ```
#[derive(Debug, Error)]
pub enum Error {
    /// GPU device error (device lost, out of memory, etc.).
    ///
    /// This error occurs when the GPU device becomes unavailable or runs out of resources.
    /// It is rare and typically indicates a driver crash or system-level issue.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_v3::Error;
    ///
    /// let error = Error::Gpu("Device lost".to_string());
    /// eprintln!("GPU error: {}", error);
    /// ```
    #[error("GPU error: {0}")]
    Gpu(String),

    /// Shader compilation error.
    ///
    /// This error occurs when a shader fails to compile. Common causes:
    /// - Syntax errors in WGSL code
    /// - Missing entry points (`@vertex`, `@fragment`, `@compute`)
    /// - Type mismatches in shader code
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_v3::Error;
    ///
    /// let error = Error::ShaderCompilation("Missing @vertex entry point".to_string());
    /// eprintln!("Shader error: {}", error);
    /// ```
    #[error("Shader compilation error: {0}")]
    ShaderCompilation(String),

    /// Resource not found error.
    ///
    /// This error occurs when a required resource (buffer, texture, bind group) is missing.
    /// Common causes:
    /// - Accessing a resource before it is created
    /// - Typo in resource name
    /// - Resource not declared in `declare_resources()`
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_v3::Error;
    ///
    /// let error = Error::ResourceNotFound("gbuffer_albedo".to_string());
    /// eprintln!("Resource error: {}", error);
    /// ```
    #[error("Resource not found: {0}")]
    ResourceNotFound(String),

    /// Invalid pass configuration error.
    ///
    /// This error occurs when a pass is configured incorrectly. Common causes:
    /// - Missing pipeline or bind group
    /// - Invalid render target format
    /// - Incompatible depth/stencil configuration
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_v3::Error;
    ///
    /// let error = Error::InvalidPassConfig("Missing pipeline".to_string());
    /// eprintln!("Pass error: {}", error);
    /// ```
    #[error("Invalid pass configuration: {0}")]
    InvalidPassConfig(String),

    /// Profiling system error.
    ///
    /// This error occurs when the profiling system fails. Common causes:
    /// - Query set overflow (too many passes)
    /// - GPU timestamp query not supported by device
    /// - Async readback failure
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_v3::Error;
    ///
    /// let error = Error::Profiling("Query set overflow".to_string());
    /// eprintln!("Profiling error: {}", error);
    /// ```
    #[error("Profiling error: {0}")]
    Profiling(String),
}

/// Result type alias for helio-v3.
///
/// Shorthand for `std::result::Result<T, Error>`. Used throughout helio-v3 for fallible operations.
///
/// # Example
///
/// ```rust,no_run
/// use helio_v3::Result;
///
/// fn my_function() -> Result<()> {
///     // ... code that might fail ...
///     Ok(())
/// }
/// ```
pub type Result<T> = std::result::Result<T, Error>;

