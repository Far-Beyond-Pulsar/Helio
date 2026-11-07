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
mod cpu_lightmap;
mod data;
mod inject;

pub use bake::{run_bake_blocking, BakeError};
pub use cache::CachedAtlasRegion;
pub use config::{BakeConfig, CpuLightmapConfig, ProbeSpec};
pub use data::BakedData;
pub use inject::BakeInjectPass;

// Re-export the Nebula scene types users need to build a BakeRequest.
pub use nebula::ao::AoConfig;
pub use nebula::core::scene::{
    BakeMesh, LightSource, LightSourceKind, MaterialDesc as BakeMaterialDesc,
};
pub use nebula::light::LightmapConfig;
pub use nebula::prelude::{BakeContext, NullReporter, SceneGeometry};
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

/// The id a baked mesh is recovered by: `slot` (the host's index for the
/// mesh) in bytes 0..8, little-endian, zeros after; nil without one. Bake
/// recovery reads `mesh_id[0] == slot as u64`, `mesh_id[1] == 0`.
pub fn bake_mesh_id(slot: Option<u32>) -> uuid::Uuid {
    let Some(slot) = slot else {
        return uuid::Uuid::nil();
    };
    let mut id_bytes = [0u8; 16];
    id_bytes[0..8].copy_from_slice(&(slot as u64).to_le_bytes());
    uuid::Uuid::from_bytes(id_bytes)
}
