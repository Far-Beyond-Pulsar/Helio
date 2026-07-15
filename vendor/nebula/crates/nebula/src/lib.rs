//! # Nebula
//!
//! Modular GPU-accelerated offline data baking framework.
//!
//! This façade crate re-exports every Nebula sub-crate behind a single
//! dependency entry.  Individual baker modules are controlled via Cargo
//! features (all enabled by default).
//!
//! ## Quick start
//!
//! ```toml
//! [dependencies]
//! nebula = "0.1"
//! ```
//!
//! ```rust,ignore
//! use nebula::prelude::*;
//!
//! #[pollster::main]
//! async fn main() {
//!     let ctx = BakeContext::new().await.unwrap();
//!     let scene = SceneGeometry::default();
//!
//!     #[cfg(feature = "light")]
//!     {
//!         use nebula::light::{LightmapBaker, LightmapConfig};
//!         let output = LightmapBaker.execute(&scene, &LightmapConfig::fast(), &ctx, &NullReporter).await.unwrap();
//!         println!("Lightmap baked: {}×{}", output.width, output.height);
//!     }
//! }
//! ```

// ── Core / GPU / Serialization ────────────────────────────────────────────────

pub use nebula_core   as core;
pub use nebula_gpu    as gpu;
pub use nebula_serialize as serialize;

// ── Baker modules (feature-gated) ────────────────────────────────────────────

#[cfg(feature = "light")]
pub use nebula_light as light;

#[cfg(feature = "ao")]
pub use nebula_ao as ao;

#[cfg(feature = "probe")]
pub use nebula_probe as probe;

#[cfg(feature = "audio")]
pub use nebula_audio as audio;

#[cfg(feature = "visibility")]
pub use nebula_visibility as visibility;

#[cfg(feature = "nav")]
pub use nebula_nav as nav;

// ── Prelude ───────────────────────────────────────────────────────────────────

/// Commonly used types re-exported for convenience.
pub mod prelude {
    pub use nebula_core::{
        context::BakeContext,
        error::NebulaError,
        progress::{NullReporter, ProgressReporter},
        scene::SceneGeometry,
        traits::{BakeInput, BakeOutput, BakePass},
    };
    pub use nebula_serialize::chunk::ChunkTag;
}
