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

/// Convert a [`MeshUpload`] with a world-space transform into a [`BakeMesh`] for use
/// in a [`BakeRequest`].
///
/// Positions are pre-multiplied by `transform` so the baker receives world-space
/// geometry.  Normals are rotated by the inverse-transpose to handle non-uniform
/// scaling.  Use [`SceneGeometry::add_mesh`] to add the returned mesh to your scene.
///
/// # Example
/// ```rust,ignore
/// let mut scene = SceneGeometry::new();
/// scene.add_mesh(mesh_upload_to_bake(&box_mesh([0.0,0.0,0.0], [5.0,0.1,5.0]),
///                                    glam::Mat4::IDENTITY, None));
/// helio_components::bake(&mut renderer, scene, BakeConfig::fast("my_scene"));
/// ```
pub fn mesh_upload_to_bake(
    upload: &helio_core::MeshUpload,
    transform: glam::Mat4,
    mesh_slot: Option<u32>,
) -> BakeMesh {
    fn unpack_snorm8(b: u8) -> f32 {
        (b as i8) as f32 / 127.0
    }
    let normal_mat = glam::Mat3::from_mat4(transform).inverse().transpose();

    // Generate deterministic ID from mesh slot (if provided).
    // Encode as a UUID with (slot as u64) in bytes 0..8 (little-endian) and zeros in bytes 8..16.
    // Bake recovery: mesh_id[0] == slot as u64, mesh_id[1] == 0.
    let id = if let Some(slot) = mesh_slot {
        let mut id_bytes = [0u8; 16];
        id_bytes[0..8].copy_from_slice(&(slot as u64).to_le_bytes());
        uuid::Uuid::from_bytes(id_bytes)
    } else {
        uuid::Uuid::nil()
    };

    // Select which UV channel to pass to Nebula for baking.
    //
    // If the mesh has a dedicated lightmap UV channel (UV1, non-overlapping [0,1]),
    // pass it explicitly so the bake uses the same coordinates the runtime shader
    // will use.  Detection: at least one vertex must have a clearly non-zero UV1
    // value (|u| or |v| > 1e-4).
    //
    // If UV1 is absent (all-zero — the common case when a mesh ships with only one
    // UV channel), fall back to UV0.  The runtime shader also falls back to UV0
    // (clamped to [0,1]) in this case, so both sides agree.
    let has_uv1 = upload
        .vertices
        .iter()
        .any(|v| v.tex_coords1[0].abs() > 1e-4 || v.tex_coords1[1].abs() > 1e-4);
    let lightmap_uvs = if has_uv1 {
        Some(
            upload
                .vertices
                .iter()
                .map(|v| v.tex_coords1)
                .collect::<Vec<_>>(),
        )
    } else {
        // Explicitly pass UV0 so Nebula bakes to it.  Passing None might
        // cause some Nebula versions to auto-generate UVs that the runtime
        // cannot recover, leading to a UV mismatch and zero lightmap effect.
        Some(
            upload
                .vertices
                .iter()
                .map(|v| v.tex_coords0)
                .collect::<Vec<_>>(),
        )
    };

    BakeMesh {
        id,
        positions: upload
            .vertices
            .iter()
            .map(|v| {
                transform
                    .transform_point3(glam::Vec3::from_array(v.position))
                    .to_array()
            })
            .collect(),
        normals: upload
            .vertices
            .iter()
            .map(|v| {
                let p = v.normal;
                let n = glam::Vec3::new(
                    unpack_snorm8(p as u8),
                    unpack_snorm8((p >> 8) as u8),
                    unpack_snorm8((p >> 16) as u8),
                );
                (normal_mat * n).normalize_or_zero().to_array()
            })
            .collect(),
        uvs: upload.vertices.iter().map(|v| v.tex_coords0).collect(),
        lightmap_uvs,
        indices: upload.indices.clone(),
        material_ids: vec![0u32; upload.indices.len() / 3],
        world_transform: Default::default(),
    }
}
