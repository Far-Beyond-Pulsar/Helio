//! High-level facade over `helio-core`.
//!
//! This crate restores a stable handle-based scene API on top of the lower-level
//! GPU-native core. Scene mutations stay O(1) with respect to scene size on the
//! CPU side by using:
//!
//! - generational handles for public resources,
//! - sparse slots for stable-index resources like materials,
//! - dense swap-remove arenas for objects and lights,
//! - partial dirty-range uploads to `helio-core` managers.

mod arena;
mod editor;
mod groups;
mod handles;
mod material;
mod mesh;
mod picking;
mod quark_commands;
pub mod radiant;
mod renderer;
mod scene;
mod terrain;
mod vg;

#[cfg(target_arch = "wasm32")]
mod wasm_cpp_alloc;

pub use editor::{EditorState, GizmoAxis, GizmoMode};
pub use groups::{GroupId, GroupMask};
pub use handles::{
    DecalId, LightId, MaterialId, MeshId, MultiMeshId, ObjectId, SectionedInstanceId, TextureId,
    VirtualObjectId, VoxelVolumeId, WaterHitboxId, WaterVolumeId,
};
pub use material::{
    MaterialAsset, MaterialTextureRef, MaterialTextures, TextureSamplerDesc, TextureTransform,
    TextureUpload, MAX_TEXTURES,
};
pub use mesh::{MeshBuffers, MeshSlice, MeshUpload, PackedVertex, SectionedMeshUpload};
pub use picking::{PickHit, ScenePicker};
pub use quark_commands::{register_helio_commands, HelioAction, HelioCommandBridge};
pub use renderer::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, DebugCameraUniform, DebugDrawPass,
    DebugDrawState, GiConfig, GraphRebuilder, PerfOverlayMode, Renderer, RendererConfig,
};
pub use scene::{
    Camera, DecalActor, ObjectDescriptor, PickableObject, ReflectionCaptureActor,
    ReflectionCaptureDescriptor, Result as SceneResult, Scene, SceneActor,
    SceneActorId, SceneActorTrait, SceneError, VoxelMode, VoxelVolumeDescriptor,
    WaterHitboxActor, WaterHitboxDescriptor,
    WaterVolumeActor, WaterVolumeDescriptor,
};
pub use terrain::{VoxelTerrain, VOXEL_TERRAIN_GRID_DIM};
pub use vg::{VirtualMeshId, VirtualMeshUpload, VirtualObjectDescriptor};

#[cfg(feature = "bake")]
pub use helio_bake::{
    AoConfig, BakeConfig, BakeMesh, BakeRequest, BakedData, LightSource, LightSourceKind,
    LightmapConfig, ProbeConfig, ProbeSpec, SceneGeometry,
};
pub use helio_core::{
    Actor, Component, ComponentRegistry, ComponentSlot, ComponentVec, DebugViewDescriptor,
    DrawIndexedIndirectArgs, Entity, Error, GpuCameraUniforms, GpuDrawCall, GpuInstanceAabb,
    GpuInstanceData, GpuLight, GpuMaterial, GpuScene, RenderGraph, RenderPass, Result,
};
pub use libhelio::{LightType, Movability, ShadowQuality, SkyActor, VolumetricClouds};

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
/// renderer.configure_bake(BakeRequest { scene, config: BakeConfig::fast("my_scene") });
/// ```
#[cfg(feature = "bake")]
pub fn mesh_upload_to_bake(
    upload: &MeshUpload,
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
