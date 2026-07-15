//! # nebula-core
//!
//! Foundation layer of the Nebula baking framework.
//!
//! Everything in this crate is `no_std`-compatible in terms of logic; the only
//! platform requirement is a working wgpu device (which implies async).
//!
//! ## Module layout
//! - [`context`]  — [`BakeContext`]: holds the wgpu device + queue; entry point
//! - [`scene`]    — [`SceneGeometry`], [`LightSource`], [`AudioEmitter`], etc.
//! - [`traits`]   — [`BakePass`], [`BakeInput`], [`BakeOutput`], [`BakeSerializer`]
//! - [`progress`] — [`ProgressReporter`] — optional async progress callbacks
//! - [`error`]    — [`NebulaError`]

pub mod context;
pub mod error;
pub mod progress;
pub mod scene;
pub mod traits;

pub use context::BakeContext;
pub use error::NebulaError;
pub use progress::{NullReporter, ProgressReporter};
pub use scene::{
    AudioEmitter, BakeMesh, LightSource, LightSourceKind, MaterialDesc, SceneGeometry,
    SurfacePoint, Transform,
};
pub use traits::{BakeInput, BakeOutput, BakePass, BakeSerializer};

pub type Result<T> = std::result::Result<T, NebulaError>;
