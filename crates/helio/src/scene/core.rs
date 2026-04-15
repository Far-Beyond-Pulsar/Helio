//! Core scene structure and lifecycle methods.
//!
//! This module contains the main [`Scene`] struct definition, constructor,
//! and core lifecycle methods like [`flush`](Scene::flush) and [`advance_frame`](Scene::advance_frame).

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::Zeroable;
use glam::Mat4;
use helio_v3::{scene::GrowableBuffer, GpuCameraUniforms, GpuScene};
use libhelio::{GpuLight, GpuShadowMatrix};
use wgpu::util::DeviceExt;

use crate::arena::{DenseArena, SparsePool};
use crate::groups::GroupMask;
use crate::handles::{LightId, MaterialId, MultiMeshId, ObjectId, SectionedInstanceId, TextureId, VirtualObjectId, WaterHitboxId, WaterVolumeId};
use crate::mesh::{MeshPool, MultiMeshRecord};
use crate::scene::multi_mesh::SectionedInstanceRecord;
use crate::scene::SceneActorTrait;
use crate::vg::VirtualMeshId;

use super::camera::Camera;
use super::types::{
    LightRecord, MaterialRecord, ObjectRecord, TextureRecord, VirtualMeshRecord,
    VirtualObjectRecord, WaterHitboxRecord, WaterVolumeRecord,
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

    /// True when a Static or Stationary object has been added or removed since the last
    /// shadow atlas render. Triggers a re-render of the static shadow atlas.
    pub(in crate::scene) static_objects_dirty: bool,

    /// True when static/stationary geometry or lights have been added since the last bake.
    /// When this is true and a bake was previously configured, the user must explicitly
    /// call auto_bake() again to rebake the scene with the new static content.
    pub(in crate::scene) bake_invalidated: bool,

    /// True when objects have been added or removed via persistent-mode delta operations.
    /// In persistent mode, insert/remove bypass the full rebuild, so shadow partition
    /// indirect buffers must be explicitly rebuilt on the next flush.
    pub(in crate::scene) shadow_partition_dirty: bool,

    /// Previous frame's view-projection matrix (for temporal effects)
    pub(in crate::scene) prev_view_proj: Mat4,

    /// Bitmask of currently hidden groups — bit N = GroupId(N) is hidden.
    /// An object is invisible if any of its groups intersects this mask.
    pub(in crate::scene) group_hidden: GroupMask,

    /// Generation counter for movable objects - increments when any Movable object's transform changes.
    /// Used by shadow caching to detect when Movable objects move.
    pub(in crate::scene) movable_objects_generation: u64,

    /// Generation counter for movable lights - increments when any Movable light's position/direction changes.
    /// Used by shadow caching to detect when Movable lights move.
    pub(in crate::scene) movable_lights_generation: u64,

    /// Per-caster content hash from the previous flush(). Compared against the newly-computed
    /// hash each frame to detect which shadow caster regions have changed. When a caster's hash
    /// differs, gpu_scene.per_caster_dirty_gen[slot] is incremented so ShadowPass re-renders
    /// only that caster's 6 atlas faces — not the entire atlas.
    pub(in crate::scene) per_caster_last_hash: [u64; 42],

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

    // ── Water volumes ─────────────────────────────────────────────────────────
    /// Water volumes (dense array)
    pub(in crate::scene) water_volumes: DenseArena<WaterVolumeRecord, WaterVolumeId>,

    /// Set when water volumes are added/removed/updated
    pub(in crate::scene) water_volumes_dirty: bool,

    /// Dirty range of water volumes that need GPU upload.
    pub(in crate::scene) water_volumes_dirty_range: Option<(usize, usize)>,

    // ── Water hitboxes ────────────────────────────────────────────────────────
    /// AABB hitboxes that displace the water heightfield simulation
    pub(in crate::scene) water_hitboxes: DenseArena<WaterHitboxRecord, WaterHitboxId>,

    /// Set when hitboxes are added/removed/updated
    pub(in crate::scene) water_hitboxes_dirty: bool,

    /// Dirty range of water hitboxes that need GPU upload.
    pub(in crate::scene) water_hitboxes_dirty_range: Option<(usize, usize)>,
    // ── Multi-material (sectioned) meshes ─────────────────────────────────────
    /// Sectioned mesh assets: one record per `insert_sectioned_mesh` call.
    /// Each record stores N `MeshId`s (one per section) all sharing the same vertex buffer.
    pub(in crate::scene) multi_meshes: SparsePool<MultiMeshRecord, MultiMeshId>,

    /// Placed sectioned mesh instances.  Each entry owns N `ObjectId`s (one per section)
    /// and back-references the `MultiMeshId` asset it was created from.
    pub(in crate::scene) sectioned_instances: SparsePool<SectionedInstanceRecord, SectionedInstanceId>,

    /// Reverse lookup: given any section's `ObjectId`, find the owning `SectionedInstanceId`.
    /// Populated by `insert_sectioned_object` and cleaned up by `remove_sectioned_object`.
    pub(in crate::scene) section_to_instance: HashMap<ObjectId, SectionedInstanceId>,
}

/// FNV-1a hash over f32 bit patterns. Used for per-caster shadow dirty tracking.
/// Hashing bit patterns (not float values) ensures NaN and -0.0 are handled consistently.
#[inline]
fn fnv1a_f32s(vals: &[f32]) -> u64 {
    const OFFSET: u64 = 14695981039346656037;
    const PRIME: u64 = 1099511628211;
    let mut h = OFFSET;
    for &v in vals {
        h ^= v.to_bits() as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
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
            static_objects_dirty: true,      // rebuild static shadow atlas on first flush
            bake_invalidated: false,         // no bake configured yet
            shadow_partition_dirty: false,   // full rebuild on first flush handles this
            prev_view_proj: Mat4::IDENTITY,
            group_hidden: GroupMask::NONE,
            movable_objects_generation: 0,
            movable_lights_generation: 0,
            custom_actors: Vec::new(),
            vg_meshes: HashMap::new(),
            vg_next_mesh_id: 0,
            vg_objects: DenseArena::new(),
            vg_objects_dirty: false,
            vg_buffer_version: 0,
            vg_cpu_meshlets: Vec::new(),
            vg_cpu_instances: Vec::new(),
            water_volumes: DenseArena::new(),
            water_volumes_dirty: false,
            water_volumes_dirty_range: None,
            water_hitboxes: DenseArena::new(),
            water_hitboxes_dirty: false,
            water_hitboxes_dirty_range: None,
            multi_meshes: SparsePool::new(),
            sectioned_instances: SparsePool::new(),
            section_to_instance: HashMap::new(),
            per_caster_last_hash: [u64::MAX; 42],
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

    /// Iterate over all live lights, yielding the handle and GPU light data.
    pub(crate) fn iter_lights(&self) -> impl Iterator<Item = (LightId, &GpuLight)> + '_ {
        self.lights.iter_with_handles().map(|(id, record)| (id, &record.gpu))
    }

    /// Get the GPU light data for a single light by its handle.
    pub(crate) fn get_light(&self, id: LightId) -> Option<GpuLight> {
        self.lights.get_with_index(id).map(|(_, record)| record.gpu)
    }

    /// Returns true if static geometry or lights have been added since the last bake.
    ///
    /// When this returns true after a bake has been configured, the baked lighting
    /// is out of date and `auto_bake()` should be called again to rebake with the
    /// new static content.
    pub fn is_bake_invalidated(&self) -> bool {
        self.bake_invalidated
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
        self.gpu_scene.camera_generation = self.gpu_scene.camera_generation.wrapping_add(1);
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
        // ── Rebuild lights buffer to only contain movable lights ─────────────
        // Static/stationary lights are baked and should not contribute to real-time lighting.
        // This dramatically improves performance when scenes have many baked lights.
        {
            let light_rec_count = self.lights.dense_len();
            let mut movable_lights: Vec<GpuLight> = Vec::with_capacity(light_rec_count);
            
            for i in 0..light_rec_count {
                if let Some(record) = self.lights.get_dense(i) {
                    if record.movability.can_move() {
                        movable_lights.push(record.gpu);
                    }
                }
            }
            
            // Replace the lights buffer with only movable lights
            self.gpu_scene.lights.set_data(movable_lights.clone());
            self.gpu_scene.movable_light_count = movable_lights.len() as u32;
            
            if movable_lights.len() < light_rec_count {
                log::trace!(
                    "[helio] Filtered lights for runtime: {} movable, {} static/stationary (baked)",
                    movable_lights.len(),
                    light_rec_count - movable_lights.len()
                );
            }
        }
        
        // Assign shadow atlas slots to the highest-importance shadow-casting lights.
        //
        // Problem with sequential assignment: the first N lights inserted always win the
        // 42-caster budget, regardless of how far away or how dim they are. A bright
        // close light inserted after slot 42 is full gets no shadow.
        //
        // Solution — two-phase importance selection:
        //   Phase 1: Score every shadow-requesting light by VIEW-INDEPENDENT importance:
        //              intensity × range²
        //            Directional lights always score ∞ (global, never culled).
        //            Sort descending → top 42 are the frame's active casters.
        //   Phase 2: Re-sort the WINNERS by their GPU buffer index (stable secondary key).
        //            Same lights that were in budget last frame keep the same atlas slots,
        //            preventing slot churn from minor score fluctuations. Only new entrants
        //            and exits cause slot reassignment (and thus dirty-gen bumps).
        //
        // IMPORTANT: Camera distance is intentionally NOT used in scoring. Using camera
        // distance causes the budget to reshuffle every frame the camera moves, which
        // triggers shadow atlas re-renders (expensive with many draw calls). The budget
        // should only change when lights are added/removed or their properties change.
        {
            const MAX_SHADOW_CASTERS: usize = 42;
            const FACES_PER_LIGHT: u32 = 6;
            let light_count = self.gpu_scene.lights.len();

            // Phase 1: score and select the top MAX_SHADOW_CASTERS.
            let mut scored: Vec<(f32, usize)> = Vec::with_capacity(light_count);
            for i in 0..light_count {
                let light = self.gpu_scene.lights.0.as_slice()[i];
                if light.shadow_index == u32::MAX {
                    continue; // user explicitly disabled shadows on this light
                }
                let score = if light.light_type == 0 {
                    // Directional: infinite range, always highest priority.
                    f32::MAX
                } else {
                    let range = light.position_range[3].max(0.001);
                    // intensity × range² — view-independent, stable across camera moves.
                    // Larger/brighter lights win the budget regardless of camera position.
                    light.color_intensity[3] * (range * range)
                };
                scored.push((score, i));
            }

            // Sort descending by importance to determine which lights win the budget.
            scored.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });

            let winner_count = scored.len().min(MAX_SHADOW_CASTERS);

            // Phase 2: re-sort winners by their buffer index (stable secondary key).
            // Lights that stay in budget from frame to frame retain the same atlas slot,
            // keeping per-caster dirty gens stable and avoiding spurious re-renders.
            scored[..winner_count].sort_unstable_by_key(|&(_, i)| i);

            // Assign atlas slots to winners; disable everything else.
            let mut next_layer: u32 = 0;
            for (rank, &(_, i)) in scored.iter().enumerate() {
                let light = self.gpu_scene.lights.0.as_slice()[i];
                if rank < MAX_SHADOW_CASTERS {
                    let mut assigned = light;
                    assigned.shadow_index = next_layer;
                    self.gpu_scene.lights.update(i, assigned);
                    next_layer += FACES_PER_LIGHT;
                } else {
                    let mut disabled = light;
                    disabled.shadow_index = u32::MAX;
                    self.gpu_scene.lights.update(i, disabled);
                }
            }
            let needed = (next_layer as usize).max(1);
            if self.gpu_scene.shadow_matrices.len() != needed {
                self.gpu_scene
                    .shadow_matrices
                    .set_data(vec![GpuShadowMatrix::zeroed(); needed]);
            }
        }

        // ── Per-caster shadow dirty tracking ─────────────────────────────────
        // Compute a content hash per shadow caster. Each hash covers:
        //   • The caster light's own geometry (position, range, direction).
        //   • All movable objects whose bounding sphere overlaps the light's range.
        // Directional lights always include every movable object (infinite range).
        // Casters whose hash differs from last frame bump their dirty gen counter;
        // ShadowPass then re-renders only those casters' atlas faces.
        {
            let light_count = self.gpu_scene.lights.len();
            let mut new_hashes = [0u64; 42];

            // Pass 1: hash each shadow-casting light's geometry into its slot.
            for i in 0..light_count {
                let light = self.gpu_scene.lights.0.as_slice()[i];
                if light.shadow_index == u32::MAX {
                    continue;
                }
                let slot = (light.shadow_index / 6) as usize;
                if slot >= 42 {
                    continue;
                }
                let base_hash = fnv1a_f32s(&light.position_range)
                    ^ fnv1a_f32s(&light.direction_outer)
                    ^ (light.light_type as u64).wrapping_mul(2654435761);
                // Directional lights use CSM cascades, which are derived from the camera
                // frustum. When the camera moves, the cascade matrices change and the
                // shadow atlas must be re-rendered. Include the camera position in the
                // hash so any camera movement bumps the dirty gen for directional casters.
                new_hashes[slot] = if light.light_type == 0 {
                    let cam_pos = self.gpu_scene.camera.position();
                    base_hash ^ fnv1a_f32s(&cam_pos)
                } else {
                    base_hash
                };
            }

            // Pass 2: fold movable object bounds into each caster they intersect.
            //
            // IMPORTANT: instance.bounds[0..3] (world-space center) is only set at
            // insert time and is NOT updated by update_object_transform(). We therefore:
            //   - Hash the model matrix (always current, changed by every transform update).
            //   - Extract the world-space position from the model's translation column
            //     (column 3 = W axis) for the sphere-sphere overlap test.
            //   - Keep instance.bounds[3] (radius) which is stable unless the caller
            //     explicitly calls update_object_bounds().
            let n_objects = self.objects.dense_len();
            for oi in 0..n_objects {
                let Some(obj) = self.objects.get_dense(oi) else {
                    continue;
                };
                if !obj.movability.can_move() {
                    continue;
                }
                // Translation column of the model matrix is always the current world position.
                let model = Mat4::from_cols_array(&obj.instance.model);
                let w = model.w_axis;
                let bx = w.x;
                let by = w.y;
                let bz = w.z;
                let br = obj.instance.bounds[3]; // radius unchanged by transform updates
                // Full model matrix hash: any movement/rotation/scale changes the hash.
                let obj_hash = fnv1a_f32s(&obj.instance.model);
                for i in 0..light_count {
                    let light = self.gpu_scene.lights.0.as_slice()[i];
                    if light.shadow_index == u32::MAX {
                        continue;
                    }
                    let slot = (light.shadow_index / 6) as usize;
                    if slot >= 42 {
                        continue;
                    }
                    if light.light_type == 0 {
                        // Directional: always overlaps all movable objects.
                        new_hashes[slot] ^= obj_hash;
                    } else {
                        let dx = light.position_range[0] - bx;
                        let dy = light.position_range[1] - by;
                        let dz = light.position_range[2] - bz;
                        let dist_sq = dx * dx + dy * dy + dz * dz;
                        let combined_r = light.position_range[3] + br;
                        if dist_sq <= combined_r * combined_r {
                            new_hashes[slot] ^= obj_hash;
                        }
                    }
                }
            }

            // Bump dirty gen for any caster whose content hash changed.
            for slot in 0..42usize {
                if new_hashes[slot] != self.per_caster_last_hash[slot] {
                    self.gpu_scene.per_caster_dirty_gen[slot] =
                        self.gpu_scene.per_caster_dirty_gen[slot].wrapping_add(1);
                    self.per_caster_last_hash[slot] = new_hashes[slot];
                }
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
            // Full rebuild already called rebuild_shadow_partition_buffers().
            self.shadow_partition_dirty = false;
        }
        // Persistent-mode delta inserts/removes bypass the full rebuild, so shadow
        // partition indirect buffers need an explicit rebuild here.
        if self.shadow_partition_dirty {
            self.rebuild_shadow_partition_buffers();
            self.shadow_partition_dirty = false;
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

    /// Build a [`SceneGeometry`](helio_bake::SceneGeometry) from all static objects and lights.
    ///
    /// Automatically extracts all objects and lights marked as Static or Stationary
    /// (i.e., not Movable) and converts them to bake-ready geometry. This eliminates
    /// the need to manually duplicate scene information for baking.
    ///
    /// # Returns
    /// A `SceneGeometry` containing:
    /// - All static object meshes with their world transforms applied
    /// - All static lights configured for baking
    ///
    /// # Example
    /// ```ignore
    /// // After building your scene normally...
    /// let bake_scene = scene.build_static_bake_scene();
    /// renderer.configure_bake(BakeRequest {
    ///     scene: bake_scene,
    ///     config: BakeConfig::fast("my_scene"),
    /// });
    /// ```
    #[cfg(feature = "bake")]
    pub fn build_static_bake_scene(&mut self) -> helio_bake::SceneGeometry {
        use helio_bake::{LightSource, LightSourceKind, SceneGeometry};
        use libhelio::{LightType, Movability};
        
        let mut bake_scene = SceneGeometry::new();
        let mut static_object_count = 0;
        let mut static_light_count = 0;
        
        // Extract all static objects
        for i in 0..self.objects.dense_len() {
            let Some(object_record) = self.objects.get_dense(i) else {
                continue;
            };
            
            // Skip movable objects - only bake static and stationary geometry
            if object_record.movability == Movability::Movable {
                continue;
            }
            
            // Extract mesh data from the pool
            let Some(mesh_upload) = self.mesh_pool.extract_mesh_data(object_record.mesh) else {
                continue;
            };
            
            // Convert to bake mesh with world transform applied
            // Pass mesh slot to generate deterministic UUID for lightmap region mapping
            let transform = Mat4::from_cols_array(&object_record.instance.model);
            let mesh_slot = object_record.mesh.slot();
            let bake_mesh = crate::mesh_upload_to_bake(&mesh_upload, transform, Some(mesh_slot));
            bake_scene.add_mesh(bake_mesh);
            static_object_count += 1;
        }
        
        // Extract all static lights
        for i in 0..self.lights.dense_len() {
            let Some(light_record) = self.lights.get_dense(i) else {
                continue;
            };
            
            // Include ALL lights in the bake regardless of movability.
            // Lights default to Movable even for static scenes; filtering them out
            // would result in a zero-light bake and an all-black lightmap.
            // If a user wants a light to be purely dynamic (never baked), they
            // should set bake_enabled = false on the BakeMesh's LightSource.
            let gpu_light = &light_record.gpu;
            let light_type = gpu_light.light_type;
            
            // Determine light kind from type
            let kind = if light_type == LightType::Directional as u32 {
                LightSourceKind::Directional {
                    direction: [
                        gpu_light.direction_outer[0],
                        gpu_light.direction_outer[1],
                        gpu_light.direction_outer[2],
                    ],
                }
            } else if light_type == LightType::Point as u32 {
                LightSourceKind::Point {
                    position: [
                        gpu_light.position_range[0],
                        gpu_light.position_range[1],
                        gpu_light.position_range[2],
                    ],
                    range: gpu_light.position_range[3],
                }
            } else if light_type == LightType::Spot as u32 {
                LightSourceKind::Spot {
                    position: [
                        gpu_light.position_range[0],
                        gpu_light.position_range[1],
                        gpu_light.position_range[2],
                    ],
                    direction: [
                        gpu_light.direction_outer[0],
                        gpu_light.direction_outer[1],
                        gpu_light.direction_outer[2],
                    ],
                    range: gpu_light.position_range[3],
                    inner_angle: gpu_light.inner_angle.acos(),
                    outer_angle: gpu_light.direction_outer[3].acos(),
                }
            } else {
                continue; // Unknown light type
            };
            
            bake_scene.add_light(LightSource {
                kind,
                color: [
                    gpu_light.color_intensity[0],
                    gpu_light.color_intensity[1],
                    gpu_light.color_intensity[2],
                ],
                intensity: gpu_light.color_intensity[3],
                bake_enabled: true,
                casts_shadows: gpu_light.shadow_index != u32::MAX,
            });
            static_light_count += 1;
        }
        
        // ── Transform lightmap UVs into atlas space ────────────────────────────
        //
        // Nebula's `build_atlas_regions` assigns each mesh an equal-area cell in
        // the atlas using a ceil(sqrt(N)) × ceil(sqrt(N)) grid.  The bake WGSL
        // shader at each texel searches ALL mesh triangles to find which triangle
        // contains that atlas-space `lm_uv`.  For correctness, vertex `lm_uv`
        // values must therefore be in ATLAS UV space, NOT in per-mesh [0,1]² UV
        // space.
        //
        // Without this transform every mesh's UV0 covers [0,1]², so for every
        // texel all N meshes' triangles match — mesh 0 always wins (listed first),
        // its lighting bleeds into every other mesh's atlas cell, and meshes 1…N-1
        // all show mesh 0's lighting at runtime.  Three-way correctness chain:
        //   bake:    `lm_uv_atlas = uv_offset + UV0 * uv_scale`  → unique range per mesh
        //   runtime: `atlas_uv   = uv_offset + UV0 * uv_scale`   → same atlas address
        //   result:  runtime UV  == bake UV                        → correct texel lookup
        let n = bake_scene.meshes.len();
        if n > 1 {
            let cols = (n as f64).sqrt().ceil() as u32;
            let rows = (n as u32).div_ceil(cols);
            let cell_w = 1.0_f32 / cols as f32;
            let cell_h = 1.0_f32 / rows as f32;
            for (i, mesh) in bake_scene.meshes.iter_mut().enumerate() {
                let col = (i as u32) % cols;
                let row = (i as u32) / cols;
                let uo = col as f32 * cell_w;
                let vo = row as f32 * cell_h;
                if let Some(uvs) = mesh.lightmap_uvs.as_mut() {
                    for uv in uvs.iter_mut() {
                        uv[0] = uo + uv[0] * cell_w;
                        uv[1] = vo + uv[1] * cell_h;
                    }
                }
            }
            log::debug!(
                "[helio-bake] Transformed lightmap UVs to atlas space: {} meshes → {}×{} grid ({:.4}×{:.4} cells)",
                n, cols, rows, cell_w, cell_h
            );
        }

        log::info!(
            "[helio-bake] Auto-extracted {} static/stationary objects and {} lights for baking",
            static_object_count,
            static_light_count
        );
        
        // Clear the invalidation flag - scene is now synced with bake data
        self.bake_invalidated = false;
        
        bake_scene
    }
}

#[cfg(test)]
mod tests {
    use crate::SceneActor;

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

