//! Core scene structure and constructor.
//!
//! This module contains the main [`Scene`] struct definition, constructor,
//! and trivial getters. Lifecycle methods, flush, camera, water, and stats
//! each live in their own sub-modules.

use std::collections::HashMap;
use std::sync::Arc;

use helio_core::scene::GrowableBuffer;
use helio_core::GpuScene;
use helio_voxel_core::VoxelEdit;
use wgpu::util::DeviceExt;

use crate::arena::{DenseArena, SparsePool};
use crate::groups::GroupMask;
use crate::handles::{
    LightId, MaterialId, MultiMeshId, ObjectId, PostProcessVolumeId, SectionedInstanceId,
    TextureId, VirtualObjectId, WaterHitboxId, WaterVolumeId, VoxelVolumeId,
};
use super::voxel::VoxelVolumeRecord;
use super::types::VoxelVolumeDescriptor;
use crate::mesh::{MeshPool, MultiMeshRecord};
use crate::scene::multi_mesh::SectionedInstanceRecord;
use crate::scene::SceneActorTrait;
use crate::vg::VirtualMeshId;

use super::errors::{invalid, Result};
use super::types::{
    LightRecord, MaterialRecord, ObjectRecord, PostProcessVolumeRecord, TextureRecord,
    VirtualMeshRecord, VirtualObjectRecord, WaterHitboxRecord, WaterVolumeRecord,
};

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
    pub(in crate::scene) prev_view_proj: glam::Mat4,

    /// Bitmask of currently hidden groups — bit N = GroupId(N) is hidden.
    /// An object is invisible if any of its groups intersects this mask.
    pub(in crate::scene) group_hidden: GroupMask,

    /// Generation counter for movable objects - increments when any Movable object's transform changes.
    /// Used by shadow caching to detect when Movable objects move.
    pub(in crate::scene) movable_objects_generation: u64,

    /// Generation counter for movable lights - increments when any Movable light's position/direction changes.
    /// Used by shadow caching to detect when Movable lights move.
    pub(in crate::scene) movable_lights_generation: u64,

    /// Per-frame custom trait-based scene actors.
    pub(in crate::scene) custom_actors: Vec<Box<dyn SceneActorTrait>>,

    // ── Virtual geometry ──────────────────────────────────────────────────────
    /// All uploaded virtual meshes keyed by their handle.
    pub(in crate::scene) vg_meshes: HashMap<VirtualMeshId, VirtualMeshRecord>,

    /// Next free VirtualMeshId slot counter (monotonically increasing).
    pub(in crate::scene) vg_next_mesh_id: u32,

    /// Dense array of virtual objects (one entry per `insert_virtual_object` call).
    pub(in crate::scene) vg_objects: DenseArena<VirtualObjectRecord, VirtualObjectId>,

    /// Set when VG topology changes; triggers `rebuild_vg_buffers()`.
    pub(in crate::scene) vg_objects_dirty: bool,

    /// Monotonically increasing counter forwarded to `VgFrameData::buffer_version`.
    /// The VG pass re-uploads GPU buffers only when this advances.
    pub(in crate::scene) vg_buffer_version: u64,

    /// Transform-only changes accumulated until the next scene flush (end exclusive).
    pub(in crate::scene) vg_instance_dirty_range: Option<(usize, usize)>,

    /// Last transform-only range published to the render pass (end exclusive).
    pub(in crate::scene) vg_published_instance_dirty_range: Option<(usize, usize)>,

    /// Monotonic version for published transform-only changes.
    pub(in crate::scene) vg_instance_version: u64,

    /// Unique meshlet entries for virtual meshes referenced by the current VG layout.
    pub(in crate::scene) vg_cpu_meshlets: Vec<libhelio::GpuMeshletEntry>,

    /// Object-level LOD ranges and bounds (one entry per VG object).
    pub(in crate::scene) vg_cpu_objects: Vec<libhelio::GpuVgObject>,

    /// Instance data for all VG objects (one entry per VG object, in order).
    pub(in crate::scene) vg_cpu_instances: Vec<helio_core::GpuInstanceData>,

    /// Immutable 64-meshlet expansion spans for the second GPU cull stage.
    pub(in crate::scene) vg_cpu_work_items: Vec<libhelio::GpuVgWorkItem>,

    /// Exact worst-case number of draws after choosing one LOD per object.
    pub(in crate::scene) vg_max_draw_count: u32,

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

    // ── Post-process volumes ─────────────────────────────────────────────────────
    /// Post-process volumes (dense array)
    pub(in crate::scene) pp_volumes: DenseArena<PostProcessVolumeRecord, PostProcessVolumeId>,

    pub(in crate::scene) pp_volumes_dirty: bool,

    pub(in crate::scene) pp_volumes_dirty_range: Option<(usize, usize)>,

    // ── Multi-material (sectioned) meshes ─────────────────────────────────────
    /// Sectioned mesh assets: one record per `insert_sectioned_mesh` call.
    /// Each record stores N `MeshId`s (one per section) all sharing the same vertex buffer.
    pub(in crate::scene) multi_meshes: SparsePool<MultiMeshRecord, MultiMeshId>,

    /// Placed sectioned mesh instances.  Each entry owns N `ObjectId`s (one per section)
    /// and back-references the `MultiMeshId` asset it was created from.
    pub(in crate::scene) sectioned_instances:
        SparsePool<SectionedInstanceRecord, SectionedInstanceId>,

    /// Reverse lookup: given any section's `ObjectId`, find the owning `SectionedInstanceId`.
    /// Populated by `insert_sectioned_object` and cleaned up by `remove_sectioned_object`.
    pub(in crate::scene) section_to_instance: HashMap<ObjectId, SectionedInstanceId>,

    // ── Voxel volumes ──────────────────────────────────────────────────────────
    /// Voxel volumes (dense array)
    pub(in crate::scene) voxel_volumes: DenseArena<VoxelVolumeRecord, VoxelVolumeId>,
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
        helio_core::upload::record_upload_bytes(4);
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
            prev_view_proj: glam::Mat4::IDENTITY,
            group_hidden: GroupMask::NONE,
            movable_objects_generation: 0,
            movable_lights_generation: 0,
            custom_actors: Vec::new(),
            vg_meshes: HashMap::new(),
            vg_next_mesh_id: 0,
            vg_objects: DenseArena::new(),
            vg_objects_dirty: false,
            vg_buffer_version: 0,
            vg_instance_dirty_range: None,
            vg_published_instance_dirty_range: None,
            vg_instance_version: 0,
            vg_cpu_meshlets: Vec::new(),
            vg_cpu_objects: Vec::new(),
            vg_cpu_instances: Vec::new(),
            vg_cpu_work_items: Vec::new(),
            vg_max_draw_count: 0,
            water_volumes: DenseArena::new(),
            water_volumes_dirty: false,
            water_volumes_dirty_range: None,
            water_hitboxes: DenseArena::new(),
            water_hitboxes_dirty: false,
            water_hitboxes_dirty_range: None,
            pp_volumes: DenseArena::new(),
            pp_volumes_dirty: false,
            pp_volumes_dirty_range: None,
            multi_meshes: SparsePool::new(),
            sectioned_instances: SparsePool::new(),
            section_to_instance: HashMap::new(),
            voxel_volumes: DenseArena::new(),
        }
    }

    pub fn insert_voxel_volume(&mut self, descriptor: VoxelVolumeDescriptor) -> Result<VoxelVolumeId> {
        let gpu_slot = self.voxel_volumes.len() as u32;
        let id = self.voxel_volumes.insert_with(|id| {
            VoxelVolumeRecord::new(id, gpu_slot, &descriptor)
        });

        if let Some(record) = self.voxel_volumes.get(id) {
            record.upload_to_gpu(&mut self.gpu_scene, gpu_slot);
        }

        self.gpu_scene.voxel_volume_count = self.voxel_volumes.len() as u32;
        Ok(id)
    }

    pub fn remove_voxel_volume(&mut self, id: VoxelVolumeId) -> Result<()> {
        self.voxel_volumes.remove(id);
        self.gpu_scene.voxel_volume_count = self.voxel_volumes.len() as u32;
        Ok(())
    }

    pub fn edit_voxel_volume(&mut self, id: VoxelVolumeId, edit: VoxelEdit) -> Result<()> {
        if let Some(record) = self.voxel_volumes.get_mut(id) {
            record.edit(&edit);
            record.push_edits_to_gpu(&mut self.gpu_scene, &[edit]);
            Ok(())
        } else {
            Err(invalid("voxel volume"))
        }
    }

    pub fn voxel_volume(&self, id: VoxelVolumeId) -> Option<&VoxelVolumeRecord> {
        self.voxel_volumes.get(id)
    }
}
