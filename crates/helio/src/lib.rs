//! High-level facade over `helio-v3`.
//!
//! This crate restores a stable handle-based scene API on top of the lower-level
//! GPU-native core. Scene mutations stay O(1) with respect to scene size on the
//! CPU side by using:
//!
//! - generational handles for public resources,
//! - sparse slots for stable-index resources like materials,
//! - dense swap-remove arenas for objects and lights,
//! - partial dirty-range uploads to `helio-v3` managers.

mod arena;
mod groups;
mod handles;
mod material;
mod mesh;
mod quark_commands;
mod renderer;
mod scene;
mod vg;

pub use groups::{GroupId, GroupMask};
pub use handles::{LightId, MaterialId, MeshId, ObjectId, TextureId, VirtualObjectId, WaterHitboxId, WaterVolumeId};
pub use helio_pass_billboard::BillboardInstance;
pub use helio_pass_perf_overlay::{PerfOverlayMode, PerfOverlayPass};
pub use material::{
    MaterialAsset, MaterialTextureRef, MaterialTextures, TextureSamplerDesc, TextureTransform,
    TextureUpload, MAX_TEXTURES,
};
pub use mesh::{MeshBuffers, MeshSlice, MeshUpload, PackedVertex};
pub use quark_commands::{HelioAction, HelioCommandBridge, register_helio_commands};
pub use renderer::{
    build_simple_graph, build_hlfs_graph, required_wgpu_features, required_wgpu_limits, GiConfig, Renderer,
    RendererConfig,
};
pub use scene::{
    Camera, ObjectDescriptor, Result as SceneResult, Scene, SceneError,
    SceneActor, SceneActorId, SceneActorTrait,
    WaterHitboxActor, WaterHitboxDescriptor,
    WaterVolumeActor, WaterVolumeDescriptor,
};
pub use vg::{VirtualMeshId, VirtualMeshUpload, VirtualObjectDescriptor};

pub use helio_v3::{
    DrawIndexedIndirectArgs, Error, GpuCameraUniforms, GpuDrawCall, GpuInstanceAabb,
    GpuInstanceData, GpuLight, GpuMaterial, GpuScene, RenderGraph, RenderPass, Result,
};
pub use libhelio::{LightType, Movability, ShadowQuality, VolumetricClouds, SkyActor};
pub use helio_bake::{
    BakeConfig, BakeRequest, BakedData, ProbeSpec,
    SceneGeometry, BakeMesh, LightSource, LightSourceKind,
    AoConfig, LightmapConfig, ProbeConfig,
};

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
///                                    glam::Mat4::IDENTITY));
/// renderer.configure_bake(BakeRequest { scene, config: BakeConfig::fast("my_scene") });
/// ```
pub fn mesh_upload_to_bake(upload: &MeshUpload, transform: glam::Mat4) -> BakeMesh {
    fn unpack_snorm8(b: u8) -> f32 { (b as i8) as f32 / 127.0 }
    let normal_mat = glam::Mat3::from_mat4(transform).inverse().transpose();
    BakeMesh {
        id: Default::default(),
        positions: upload.vertices.iter().map(|v| {
            transform.transform_point3(glam::Vec3::from_array(v.position)).to_array()
        }).collect(),
        normals: upload.vertices.iter().map(|v| {
            let p = v.normal;
            let n = glam::Vec3::new(
                unpack_snorm8(p as u8),
                unpack_snorm8((p >> 8) as u8),
                unpack_snorm8((p >> 16) as u8),
            );
            (normal_mat * n).normalize_or_zero().to_array()
        }).collect(),
        uvs: upload.vertices.iter().map(|v| v.tex_coords0).collect(),
        lightmap_uvs: None,
        indices: upload.indices.clone(),
        material_ids: vec![0u32; upload.indices.len() / 3],
        world_transform: Default::default(),
    }
}