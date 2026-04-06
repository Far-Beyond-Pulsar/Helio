//! # helio-bake
//!
//! Offline baking integration for Helio, powered by Nebula.
//!
//! ## What is baked
//!
//! | Pass               | What it produces                          | Replaces at runtime      |
//! |--------------------|-------------------------------------------|--------------------------|
//! | Ambient Occlusion  | R32F full-res AO texture                  | Screen-space SSAO        |
//! | Lightmaps          | RGBA lightmap atlas (RGBA32F or RGBA16F)  | Real-time direct lights  |
//! | Reflection + SH    | Pre-filtered cubemap + L2 SH coefficients | Runtime IBL probes       |
//! | PVS                | Compressed bitfield grid                  | GPU occlusion culling    |
//!
//! ## Usage
//!
//! ```rust,ignore
//! use helio_bake::{BakeConfig, BakeRequest};
//! use nebula::prelude::{SceneGeometry, BakeMesh, LightSource, LightSourceKind, Transform};
//!
//! // 1. Describe your scene geometry for baking
//! let mut scene = SceneGeometry::new();
//! scene.add_mesh(BakeMesh { /* ... */ });
//! scene.add_light(LightSource { kind: LightSourceKind::Directional { direction: [0., -1., 0.] },
//!     color: [1., 1., 1.], intensity: 100_000., bake_enabled: true, casts_shadows: true });
//!
//! // 2. Configure what to bake and where to cache it
//! let config = BakeConfig::fast("outdoor_scene");
//!
//! // 3. Hand off to the renderer — baking blocks until complete before frame 1
//! renderer.configure_bake(BakeRequest { scene, config });
//! ```

mod bake;
mod cache;
mod config;
mod data;
mod inject;

pub use bake::{run_bake_blocking, BakeError};
pub use config::{BakeConfig, ProbeSpec};
pub use data::BakedData;
pub use inject::BakeInjectPass;

// Re-export the Nebula scene types users need to build a BakeRequest.
pub use nebula::prelude::{BakeContext, NullReporter, SceneGeometry};
pub use nebula::core::scene::{BakeMesh, LightSource, LightSourceKind, MaterialDesc as BakeMaterialDesc};
pub use nebula::ao::AoConfig;
pub use nebula::light::LightmapConfig;
pub use nebula::probe::ProbeConfig;
pub use nebula::visibility::PvsConfig;

/// Everything the renderer needs to perform pre-frame-1 baking.
pub struct BakeRequest {
    /// Scene geometry description (meshes, normals, UVs, lights, sky panorama).
    ///
    /// Build this from your loaded scene data before calling
    /// [`Renderer::configure_bake`](helio::Renderer::configure_bake).
    pub scene: SceneGeometry,

    /// Which passes to bake and quality/cache settings.
    pub config: BakeConfig,
}
