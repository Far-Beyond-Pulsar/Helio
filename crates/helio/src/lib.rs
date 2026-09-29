//! High-level renderer facade over `helio-core`.
//!
//! Persistent render state is owned by the frontend SceneDB and mirrored into
//! generic GPU component buffers. This crate owns backend GPU machinery and
//! transient render products; it does not own a typed scene container.

mod camera;
mod material;
mod quark_commands;
mod renderer;
// SceneDB is the sole scene authority; the legacy scene container was removed.

#[cfg(target_arch = "wasm32")]
mod wasm_cpp_alloc;

/// Portal pair math and SceneDB GPU contracts owned by the portal passes.
pub use helio_pass_portal_cull::{
    crossing_detected, plane_signed_distance, portal_pose_facing, GpuPortalView, PortalPair,
    PortalPose, PortalProjectionBridge, PortalProjectionConfig, PortalProjectionFrame,
    PortalProjectionKey, ProjectionError, RuntimePortalKey, MAX_PORTAL_CHAINS,
};
pub use helio_pass_sky::{CloudPipelineConfig, CloudQuality, CloudRenderMode, CloudResolution};
pub use helio_pass_tsr::TsrQuality;
pub use helio_mats::{
    MaterialBindingConfig, MaterialBindingMode, BINDLESS_MATERIAL_FEATURES,
    EXPANDED_MATERIAL_TEXTURE_RESERVE, MAX_MATERIAL_TEXTURES,
};
pub use helio_core::{MeshUpload, PackedVertex, SectionedMeshUpload};
pub use material::{TextureSamplerDesc, TextureTransform, TextureUpload, MAX_TEXTURES};
pub use quark_commands::{register_helio_commands, HelioAction, HelioCommandBridge};
pub use renderer::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits,
    BillboardInstance, DebugCameraUniform, DebugDrawPass, DebugDrawState, GiConfig, GraphRebuilder,
    PassBuildContext, PassGraphBuilderFn, PerfOverlayMode, RenderMode, Renderer,
    RendererBuilder, RendererConfig, SceneDbHandle,
};
pub use camera::Camera;
#[cfg(feature = "bake")]
pub use helio_bake::{
    AoConfig, BakeConfig, BakeMesh, BakeRequest, BakedData, LightSource, LightSourceKind,
    LightmapConfig, ProbeConfig, ProbeSpec, SceneGeometry,
};
pub use helio_core::{
    Actor, DebugViewDescriptor, Entity, Error, GpuCameraUniforms, GpuTimingAvailability, Movability,
    RenderGraph, RenderPass, RenderPassTiming, RenderTimingSnapshot, Result,
};
pub use helio_pass_forward_lit::{GpuLight, LightType};
pub use helio_pass_object_batch::{
    DrawIndexedIndirectArgs, GpuDrawCall, GpuInstanceAabb, GpuInstanceData, INSTANCE_FLAG_MOVABLE,
};
pub use helio_mats::GpuMaterial;
pub use helio_pass_postprocess::{HdrOutputMode, TonemapOperator};
pub use helio_pass_shadow_matrix::ShadowQuality;
pub use helio_pass_sky::{SkyActor, VolumetricClouds};

/// Project a SceneDB world into the [`SceneGeometry`] a bake reads
/// (Helio#256): every drawn object (`StaticObjectComponent` and the
/// `MeshComponent` it references, in world space) and every live light
/// (`LightComponent`). This is the explicit CPU projection
/// [`Renderer::set_bake_scene`] takes; the renderer itself never traverses
/// scene entities.
///
/// Call it after the scene is populated and before
/// [`Renderer::auto_bake`]. Objects whose mesh row is gone (a stale
/// generation) are skipped.
#[cfg(feature = "bake")]
pub fn bake_scene_from_world(world: &pulsar_scenedb::World) -> SceneGeometry {
    use helio_pass_forward_lit::LightComponent;
    use helio_pass_gbuffer::{MeshComponent, StaticObjectComponent};

    let meshes: std::collections::HashMap<u32, (u32, &MeshComponent)> = world
        .query::<&MeshComponent>()
        .map(|(entity, mesh)| (entity.index(), (entity.generation(), mesh)))
        .collect();
    let mut scene = SceneGeometry::new();
    for (_, object) in world.query::<&StaticObjectComponent>() {
        let Some(&(generation, mesh)) = meshes.get(&object.mesh_slot) else {
            continue;
        };
        // Object rows store `generation + 1` so zero means "never written".
        if generation.wrapping_add(1) != object.mesh_generation {
            continue;
        }
        let upload = MeshUpload {
            vertices: mesh.vertices.clone(),
            indices: mesh.indices.clone(),
        };
        let transform = glam::Mat4::from_cols_array_2d(&object.transform);
        scene.add_mesh(mesh_upload_to_bake(&upload, transform, Some(object.mesh_slot)));
    }
    for (_, light) in world.query::<&LightComponent>() {
        let light = GpuLight::from(*light);
        if light.color_intensity[3] <= 0.0 {
            continue;
        }
        let [x, y, z, range] = light.position_range;
        let direction = [
            light.direction_outer[0],
            light.direction_outer[1],
            light.direction_outer[2],
        ];
        let kind = match light.light_type {
            t if t == LightType::Directional as u32 => LightSourceKind::Directional { direction },
            // GpuLight stores cosines; Nebula takes radians.
            t if t == LightType::Spot as u32 => LightSourceKind::Spot {
                position: [x, y, z],
                direction,
                range,
                inner_angle: light.inner_angle.clamp(-1.0, 1.0).acos(),
                outer_angle: light.direction_outer[3].clamp(-1.0, 1.0).acos(),
            },
            _ => LightSourceKind::Point { position: [x, y, z], range },
        };
        scene.add_light(LightSource {
            kind,
            color: [
                light.color_intensity[0],
                light.color_intensity[1],
                light.color_intensity[2],
            ],
            intensity: light.color_intensity[3],
            bake_enabled: true,
            casts_shadows: light.shadow_index != u32::MAX,
        });
    }
    scene
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

#[cfg(all(test, feature = "bake"))]
mod bake_scene_tests {
    use super::*;
    use helio_pass_forward_lit::LightComponent;
    use helio_pass_gbuffer::{MeshComponent, StaticObjectComponent};

    fn vertex(position: [f32; 3]) -> PackedVertex {
        PackedVertex { position, ..Default::default() }
    }

    /// Helio#256: the bake input comes from SceneDB rows, in world space,
    /// and skips what is not really there.
    #[test]
    fn projects_objects_and_live_lights_from_the_world() {
        let mut world = pulsar_scenedb::World::new();
        let mesh = world.spawn();
        world.insert(
            mesh,
            MeshComponent {
                vertices: vec![vertex([0.0, 0.0, 0.0]), vertex([1.0, 0.0, 0.0]), vertex([0.0, 1.0, 0.0])],
                indices: vec![0, 1, 2],
            },
        );
        let object = |mesh_generation: u32, offset: f32| {
            StaticObjectComponent::new(
                mesh.index(),
                mesh_generation,
                0,
                1,
                glam::Mat4::from_translation(glam::Vec3::new(offset, 0.0, 0.0)),
                [offset, 0.0, 0.0, 1.0],
                3,
                0,
                0,
                0,
                0,
                0,
            )
        };
        let live = world.spawn();
        world.insert(live, object(mesh.generation().wrapping_add(1), 10.0));
        // A row pointing at a mesh generation that no longer exists.
        let stale = world.spawn();
        world.insert(stale, object(mesh.generation().wrapping_add(2), 20.0));

        let mut sun = GpuLight::default();
        sun.light_type = LightType::Directional as u32;
        sun.direction_outer = [0.0, -1.0, 0.0, 0.0];
        sun.color_intensity = [1.0, 0.9, 0.8, 3.0];
        let sun_entity = world.spawn();
        world.insert(sun_entity, LightComponent::from(sun));
        let vacant = world.spawn();
        world.insert(vacant, LightComponent::from(GpuLight { color_intensity: [0.0; 4], ..GpuLight::default() }));

        let scene = bake_scene_from_world(&world);
        assert_eq!(scene.meshes.len(), 1, "stale object rows are skipped");
        assert_eq!(scene.meshes[0].positions[1], [11.0, 0.0, 0.0], "positions are world space");
        assert_eq!(scene.meshes[0].indices, vec![0, 1, 2]);
        assert_eq!(scene.lights.len(), 1, "dark rows are not lights");
        assert!(matches!(scene.lights[0].kind, LightSourceKind::Directional { direction } if direction == [0.0, -1.0, 0.0]));
        assert_eq!(scene.lights[0].intensity, 3.0);
        assert!(!scene.lights[0].casts_shadows, "GpuLight::default() requests no shadow");
    }
}
