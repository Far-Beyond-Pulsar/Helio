//! Core scene structure and lifecycle methods.
//!
//! This module contains the main [`Scene`] struct definition, constructor,
//! and core lifecycle methods like [`flush`](Scene::flush) and [`advance_frame`](Scene::advance_frame).

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Mat4;
use helio_v3::{scene::GrowableBuffer, GpuCameraUniforms, GpuScene};
use libhelio::GpuShadowMatrix;
use wgpu::util::DeviceExt;

use crate::arena::{DenseArena, SparsePool};
use crate::groups::GroupMask;
use crate::handles::{LightId, MaterialId, ObjectId, TextureId, VirtualObjectId};
use crate::material::MAX_TEXTURES;
use crate::mesh::{MeshPool, MeshUpload};
use crate::vg::{VirtualMeshId, VirtualMeshUpload};

use super::actor::{LightActor, MeshActor, ObjectActor, SceneActor, SceneActorTrait, VirtualMeshActor};
use super::camera::Camera;
use super::types::{
    LightRecord, MaterialRecord, ObjectRecord, TextureRecord, VirtualMeshRecord,
    VirtualObjectRecord,
};
use libhelio::sky::SkyContext;

/// High-level scene management with persistent GPU-driven state.
///
/// See the [module-level documentation](crate::scene) for architecture details and usage examples.
pub struct Scene {
    /// GPU scene resources (buffers, bind groups, etc.)
    pub(in crate::scene) gpu_scene: GpuScene,

    /// Mesh pool (shared vertex/index buffers)
    pub(in crate::scene) mesh_pool: MeshPool,

    /// Texture pool (sparse array with reference counting)
    pub(in crate::scene) textures: SparsePool<TextureRecord, TextureId>,

    /// Texture binding version (increments on add/remove)
    pub(in crate::scene) texture_binding_version: u64,

    /// Material texture storage buffer (GPU-side texture descriptors)
    pub(in crate::scene) material_textures: GrowableBuffer<crate::material::GpuMaterialTextures>,

    /// Placeholder texture (1x1 white)
    pub(in crate::scene) _placeholder_texture: wgpu::Texture,

    /// Placeholder texture view
    pub(in crate::scene) placeholder_view: wgpu::TextureView,

    /// Placeholder sampler
    pub(in crate::scene) placeholder_sampler: wgpu::Sampler,

    /// Material pool (sparse array with reference counting)
    pub(in crate::scene) materials: SparsePool<MaterialRecord, MaterialId>,

    /// Light pool (dense array)
    pub(in crate::scene) lights: DenseArena<LightRecord, LightId>,

    /// Object pool (dense array)
    pub(in crate::scene) objects: DenseArena<ObjectRecord, ObjectId>,

    /// True when the objects list has changed and the GPU instance/draw_call/indirect
    /// buffers need to be rebuilt from scratch (sorted by mesh+material for instancing).
    pub(in crate::scene) objects_dirty: bool,

    /// True when the scene layout has been optimized (sorted by mesh+material for instancing).
    /// When false, objects use persistent slots (1 draw per object, O(1) add/remove).
    /// When true, objects are sorted for cache coherency (instanced batching).
    pub(in crate::scene) objects_layout_optimized: bool,

    /// Previous frame's view-projection matrix (for temporal effects)
    pub(in crate::scene) prev_view_proj: Mat4,

    /// Bitmask of currently hidden groups — bit N = GroupId(N) is hidden.
    /// An object is invisible if any of its groups intersects this mask.
    pub(in crate::scene) group_hidden: GroupMask,

    /// Per-frame custom trait-based scene actors.
    pub(in crate::scene) custom_actors: Vec<Box<dyn SceneActorTrait>>,

    // ── Virtual geometry ──────────────────────────────────────────────────────
    /// All uploaded virtual meshes keyed by their handle.
    pub(in crate::scene) vg_meshes: HashMap<VirtualMeshId, VirtualMeshRecord>,

    /// Next free VirtualMeshId slot counter (monotonically increasing).
    pub(in crate::scene) vg_next_mesh_id: u32,

    /// Dense array of virtual objects (one entry per `insert_virtual_object` call).
    pub(in crate::scene) vg_objects: DenseArena<VirtualObjectRecord, VirtualObjectId>,

    /// Set when VG topology or transforms change; triggers `rebuild_vg_buffers()`.
    pub(in crate::scene) vg_objects_dirty: bool,

    /// Monotonically increasing counter forwarded to `VgFrameData::buffer_version`.
    /// The VG pass re-uploads GPU buffers only when this advances.
    pub(in crate::scene) vg_buffer_version: u64,

    /// Flattened meshlet entries for the current VG layout (rebuilt when dirty).
    pub(in crate::scene) vg_cpu_meshlets: Vec<libhelio::GpuMeshletEntry>,

    /// Instance data for all VG objects (one entry per VG object, in order).
    pub(in crate::scene) vg_cpu_instances: Vec<helio_v3::GpuInstanceData>,
}

impl Scene {
    /// Create a new empty scene.
    ///
    /// Initializes all resource pools, creates placeholder textures, and sets up
    /// GPU buffers with default capacities.
    ///
    /// # Parameters
    /// - `device`: GPU device for buffer/texture creation
    /// - `queue`: GPU queue for initial uploads
    ///
    /// # Returns
    /// A new [`Scene`] ready for resource insertion.
    ///
    /// # Initial State
    /// - All resource pools are empty
    /// - Scene is in persistent mode (`objects_layout_optimized = false`)
    /// - First `flush()` will rebuild GPU buffers
    ///
    /// # Performance
    /// - CPU cost: O(1) struct initialization
    /// - GPU cost: Creates placeholder texture, allocates initial buffer capacity
    /// - Memory: Allocates arena/pool structures with default capacity
    ///
    /// # Example
    /// ```ignore
    /// use std::sync::Arc;
    /// use helio::Scene;
    ///
    /// let device = Arc::new(gpu_device);
    /// let queue = Arc::new(gpu_queue);
    /// let scene = Scene::new(device, queue);
    /// ```
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        helio_v3::upload::record_upload_bytes(4);
        let placeholder_texture = device.create_texture_with_data(
            &queue,
            &wgpu::TextureDescriptor {
                label: Some("Helio Placeholder Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &[255, 255, 255, 255],
        );
        let placeholder_view =
            placeholder_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let placeholder_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Helio Placeholder Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Self {
            mesh_pool: MeshPool::new(device.clone()),
            gpu_scene: GpuScene::new(device.clone(), queue.clone()),
            textures: SparsePool::new(),
            texture_binding_version: 0,
            material_textures: GrowableBuffer::new(
                device,
                256,
                wgpu::BufferUsages::STORAGE,
                "Helio Material Texture Buffer",
            ),
            _placeholder_texture: placeholder_texture,
            placeholder_view,
            placeholder_sampler,
            materials: SparsePool::new(),
            lights: DenseArena::new(),
            objects: DenseArena::new(),
            objects_dirty: true,             // rebuild on first flush
            objects_layout_optimized: false, // start in persistent mode
            prev_view_proj: Mat4::IDENTITY,
            group_hidden: GroupMask::NONE,
            custom_actors: Vec::new(),
            vg_meshes: HashMap::new(),
            vg_next_mesh_id: 0,
            vg_objects: DenseArena::new(),
            vg_objects_dirty: false,
            vg_buffer_version: 0,
            vg_cpu_meshlets: Vec::new(),
            vg_cpu_instances: Vec::new(),
        }
    }

    /// Get read-only access to the GPU scene resources.
    ///
    /// Returns a reference to the internal [`GpuScene`] containing all GPU buffers,
    /// bind groups, and render state. Used by the renderer to access GPU resources.
    ///
    /// # Returns
    /// A reference to the [`GpuScene`].
    pub fn gpu_scene(&self) -> &GpuScene {
        &self.gpu_scene
    }

    /// Insert a custom trait-based scene actor.
    ///
    /// This can be e.g. `SceneActor::Sky`, `MeshActor`, `LightActor`, or other custom actors.
    pub fn insert_actor<A: SceneActorTrait + 'static>(&mut self, mut actor: A) -> crate::scene::actor::SceneActorId {
        actor.on_attach(self);
        let id = actor.inserted_id();
        self.custom_actors.push(Box::new(actor));
        id
    }

    /// Returns effective sky context for the current frame.
    pub fn sky_context(&self) -> SkyContext {
        // First preference: explicit sky actor.
        for actor in self.custom_actors.iter() {
            if let Some(sky) = actor.sky_context() {
                return sky;
            }
        }

        SkyContext::default()
    }

    /// Set the render target size for camera calculations.
    ///
    /// Updates the internal width/height used for aspect ratio calculations
    /// and viewport-dependent effects.
    ///
    /// # Parameters
    /// - `width`: Render target width in pixels
    /// - `height`: Render target height in pixels
    ///
    /// # Example
    /// ```ignore
    /// scene.set_render_size(1920, 1080);
    /// ```
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.gpu_scene.width = width;
        self.gpu_scene.height = height;
    }

    /// Update the scene's camera for the current frame.
    ///
    /// Computes camera uniforms and uploads them to the GPU. Also stores the
    /// previous frame's view-projection matrix for temporal effects (TAA, motion blur).
    ///
    /// # Parameters
    /// - `camera`: Camera parameters (view, projection, position, near, far, jitter)
    ///
    /// # Performance
    /// - CPU cost: O(1) - matrix multiplication and uniform construction
    /// - GPU cost: O(1) - writes to camera uniform buffer
    ///
    /// # Temporal Effects
    ///
    /// The previous frame's view-projection matrix is stored for:
    /// - Temporal anti-aliasing (TAA) - reprojection
    /// - Motion blur - velocity calculation
    /// - Temporal upsampling - history sampling
    ///
    /// # Example
    /// ```ignore
    /// use helio::Camera;
    /// use glam::{Mat4, Vec3};
    ///
    /// let camera = Camera::perspective_look_at(
    ///     Vec3::new(0.0, 5.0, 10.0), // position
    ///     Vec3::ZERO,                // look_at
    ///     Vec3::Y,                   // up
    ///     60.0_f32.to_radians(),     // fov_y
    ///     16.0 / 9.0,                // aspect
    ///     0.1,                       // near
    ///     1000.0,                    // far
    /// );
    /// scene.update_camera(camera);
    /// ```
    pub fn update_camera(&mut self, camera: Camera) {
        let uniforms = GpuCameraUniforms::new(
            camera.view,
            camera.proj,
            camera.position,
            camera.near,
            camera.far,
            self.gpu_scene.frame_count as u32,
            camera.jitter,
            self.prev_view_proj,
        );
        // Store the UNJITTERED view_proj so next frame's motion-vector
        // reprojection (prev_view_proj) is not contaminated by this frame's jitter.
        let inv_jitter = Mat4::from_translation(glam::Vec3::new(
            -camera.jitter[0], -camera.jitter[1], 0.0,
        ));
        let unjittered_proj = inv_jitter * camera.proj;
        self.prev_view_proj = unjittered_proj * camera.view;
        self.gpu_scene.camera.update(uniforms);
    }

    /// Flush pending changes to GPU buffers.
    ///
    /// This method:
    /// 1. Assigns shadow atlas base layers to shadow-casting lights
    /// 2. Flushes mesh pool uploads (vertex/index data)
    /// 3. Flushes material texture buffer uploads
    /// 4. Rebuilds object instance buffers if dirty (persistent or optimized mode)
    /// 5. Rebuilds virtual geometry buffers if dirty
    /// 6. Flushes all GPU scene buffers (instances, draws, indirect, visibility, etc.)
    ///
    /// # Performance
    ///
    /// **Clean state (no topology changes):**
    /// - CPU cost: O(1) - only shadow index assignment
    /// - GPU cost: O(lights) shadow index updates
    ///
    /// **Dirty state (topology changed):**
    /// - CPU cost: O(N) for persistent rebuild, O(N log N) for optimized rebuild
    /// - GPU cost: O(N) buffer uploads for all object data
    ///
    /// # Shadow Management
    ///
    /// Automatically assigns shadow atlas layers to shadow-casting lights:
    /// - Maximum 42 shadow casters (42 × 6 = 252 atlas layers)
    /// - 6 slots per light (point = 6 faces, directional = 4 cascades + 2 padding, spot = 1 + 5 padding)
    /// - Lights beyond the cap have shadows disabled automatically
    ///
    /// # When to Call
    ///
    /// Call `flush()` after all scene modifications for the frame, before rendering:
    /// ```ignore
    /// // Modify scene
    /// scene.insert_object(desc)?;
    /// scene.update_object_transform(id, transform)?;
    /// scene.hide_group(group_id);
    ///
    /// // Flush changes
    /// scene.flush();
    ///
    /// // Render
    /// renderer.render(&scene, target)?;
    /// ```
    pub fn flush(&mut self) {
        // Assign sequential shadow atlas base layers to each shadow-casting light.
        // Convention: shadow_index == u32::MAX  → no shadow.
        // Always 6 slots per light (matches FACES_PER_LIGHT in shadow_matrices.wgsl):
        //   Point:       6 cube-face matrices
        //   Directional: 4 CSM cascades + 2 identity padding slots
        //   Spot:        1 perspective matrix + 5 unused (zeroed) slots
        // Cap at 42 shadow casters (42 × 6 = 252 ≤ 256 atlas layers).
        {
            const MAX_SHADOW_CASTERS: usize = 42;
            const FACES_PER_LIGHT: u32 = 6;
            let light_count = self.gpu_scene.lights.len();
            let mut next_layer: u32 = 0;
            let mut shadow_caster_count = 0usize;
            for i in 0..light_count {
                let light = self.gpu_scene.lights.0.as_slice()[i];
                if light.shadow_index == u32::MAX {
                    // Explicitly disabled — leave as-is.
                    continue;
                }
                if shadow_caster_count >= MAX_SHADOW_CASTERS {
                    // Over cap: disable shadow for this light.
                    let mut disabled = light;
                    disabled.shadow_index = u32::MAX;
                    self.gpu_scene.lights.update(i, disabled);
                    continue;
                }
                let mut assigned = light;
                assigned.shadow_index = next_layer;
                self.gpu_scene.lights.update(i, assigned);
                next_layer += FACES_PER_LIGHT;
                shadow_caster_count += 1;
            }
            let needed = (next_layer as usize).max(1);
            if self.gpu_scene.shadow_matrices.len() != needed {
                self.gpu_scene
                    .shadow_matrices
                    .set_data(vec![GpuShadowMatrix::zeroed(); needed]);
            }
        }
        let queue = self.gpu_scene.queue.clone();
        self.mesh_pool.flush(&queue);
        self.material_textures.flush(&queue);
        // Rebuild instanced draw lists when the object set has changed.
        if self.objects_dirty {
            if self.objects_layout_optimized {
                self.rebuild_instance_buffers_optimized();
            } else {
                self.rebuild_instance_buffers_persistent();
            }
            self.objects_dirty = false;
        }
        // Rebuild virtual geometry CPU buffers when VG topology or transforms changed.
        if self.vg_objects_dirty {
            self.rebuild_vg_buffers();
            self.vg_objects_dirty = false;
        }
        self.gpu_scene.flush();
    }

    /// Advance the frame counter.
    ///
    /// Increments the internal frame counter used for temporal effects and shader logic.
    /// Call this once per frame after rendering.
    ///
    /// # Frame Counter Uses
    /// - Temporal anti-aliasing (TAA) - jitter pattern sequencing
    /// - Temporal dithering - noise pattern variation
    /// - Shader debugging - frame-dependent visualization
    ///
    /// # Example
    /// ```ignore
    /// loop {
    ///     scene.update_camera(camera);
    ///     scene.flush();
    ///     renderer.render(&scene, target)?;
    ///     scene.advance_frame();
    /// }
    /// ```
    pub fn advance_frame(&mut self) {
        // Tick custom trait-based actors.
        let scene_ptr: *mut Scene = self;
        for actor in self.custom_actors.iter_mut() {
            if actor.is_active() {
                unsafe { actor.on_tick(&mut *scene_ptr) };
            }
        }

        self.gpu_scene.frame_count = self.gpu_scene.frame_count.wrapping_add(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use libhelio::{SkyActor, VolumetricClouds};

    fn create_test_device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No adapter found");

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_defaults(),
                ..Default::default()
            },
        ))
        .expect("Failed to create device");

        (Arc::new(device), Arc::new(queue))
    }

    #[test]
    fn test_sky_actor_detection_default() {
        let (device, queue) = create_test_device();
        let scene = Scene::new(device, queue);

        let sky_ctx = scene.sky_context();
        assert!(!sky_ctx.has_sky, "Default scene should have no sky");
        assert!(sky_ctx.clouds.is_none(), "Default scene should have no clouds");
    }

    #[test]
    fn test_sky_actor_detection_with_clouds() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        // Insert sky actor with clouds
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new()
                .with_sky_color([0.5, 0.7, 1.0])
                .with_clouds(VolumetricClouds {
                    coverage: 0.6,
                    density: 0.8,
                    ..Default::default()
                })
        ));

        let sky_ctx = scene.sky_context();
        assert!(sky_ctx.has_sky, "Sky actor should be detected");
        assert!(sky_ctx.clouds.is_some(), "Cloud settings should be detected");

        if let Some(clouds) = sky_ctx.clouds {
            assert!((clouds.coverage - 0.6).abs() < 0.01, "Coverage should match");
            assert!((clouds.density - 0.8).abs() < 0.01, "Density should match");
        }
    }

    #[test]
    fn test_multiple_sky_actors_first_wins() {
        let (device, queue) = create_test_device();
        let mut scene = Scene::new(device, queue);

        // Insert first sky actor
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new().with_sky_color([1.0, 0.0, 0.0])
        ));

        // Insert second sky actor (should be ignored)
        scene.insert_actor(SceneActor::Sky(
            SkyActor::new().with_sky_color([0.0, 1.0, 0.0])
        ));

        let sky_ctx = scene.sky_context();
        // First actor wins
        assert!((sky_ctx.sky_color[0] - 1.0).abs() < 0.01, "Should use first actor's color");
    }
}

