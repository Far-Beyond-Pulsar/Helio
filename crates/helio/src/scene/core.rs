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

use super::types::VoxelVolumeDescriptor;
use super::voxel::VoxelVolumeRecord;
use crate::arena::{DenseArena, SparsePool};
use crate::groups::GroupMask;
use crate::handles::{
    DecalId, LightId, MaterialId, MultiMeshId, ObjectId, PortalId, PostProcessVolumeId,
    ReflectionCaptureId, SectionedInstanceId, SublevelId, TextureId, VirtualObjectId,
    VoxelVolumeId, WaterHitboxId, WaterVolumeId,
};
use crate::mesh::{MeshPool, MultiMeshRecord};
use crate::radiant::RadiantGraphRegistry;
use crate::scene::multi_mesh::SectionedInstanceRecord;
use crate::scene::SceneActorTrait;
use crate::vg::VirtualMeshId;

use pulsar_scenedb::gpu::{
    EngineGpuContext, FrameDriver, GpuMirrorHandle, RegionClassConfig, SceneGpuConfig,
    SceneGpuStore,
};
use pulsar_scenedb::{SubsystemRegistry, World};

use super::errors::{invalid, Result};
use super::portals::PortalRecord;
use super::scenedb::SceneGpuMirrorSubsystem;
use super::scenedb_components::register_gpu_columns;
use super::sublevels::SublevelRecord;
use super::types::{
    DecalRecord, LightRecord, MaterialRecord, ObjectRecord, PostProcessVolumeRecord,
    ReflectionCaptureRecord, TextureRecord, VirtualMeshRecord, VirtualObjectRecord,
    WaterHitboxRecord, WaterVolumeRecord,
};

/// High-level scene management with persistent GPU-driven state.
///
/// See the [module-level documentation](crate::scene) for architecture details and usage examples.
pub struct Scene {
    /// GPU scene resources (buffers, bind groups, etc.). Scene owns a copy
    /// for internal operations (camera, voxels, decals); the Renderer owns a
    /// SEPARATE copy synced from Scene's pipeline buffers for graph execution.
    pub(in crate::scene) gpu_scene: GpuScene,

    /// Mesh pool (shared vertex/index buffers)
    pub(in crate::scene) mesh_pool: MeshPool,

    /// CPU archetype ECS World (pulsar_scenedb) — shared substrate for
    /// object/light/material storage (draft PR #209 migration target) and
    /// the GPU world-mirror bridge (Helio #210 foundation). Mirrored GPU
    /// columns are flushed each frame via [`SceneGpuMirrorSubsystem`].
    pub(in crate::scene) world: World,

    /// Frame-boundary phase machine driver (SceneDB§C3). Calls `begin()`
    /// each frame; the returned witnesses are consumed through simulate →
    /// harvest → boundary → compact → sync.
    pub(in crate::scene) scenedb_driver: FrameDriver,

    /// Registered engine subsystems hooked into SceneDB's phase machine.
    /// At minimum contains [`SceneGpuMirrorSubsystem`] (world-mirror flush);
    /// future stages add lights/materials/instance bridge subsystems.
    pub(in crate::scene) scenedb_subsystems: SubsystemRegistry,

    /// Arc-wrapped SceneGpuStore shared with [`World`]'s attached
    /// [`GpuMirrorHandle`] and the mirror subsystem's clone. World-mirror
    /// columns (growable / dirty-tracked GPU buffers) are registered here.
    /// Flushed via `&self` in the subsystem's boundary hook.
    pub(in crate::scene) mirror_store: std::sync::Arc<SceneGpuStore>,

    /// By-value SceneGpuStore that drives the genuine `&mut` witness chain
    /// (`retire`, `compact`, `sync`) with empty cells each frame. The
    /// [`RetiredPhase`] witness produced here is what gates the subsystem
    /// boundary hook. See the two-store design in `scenedb.rs`.
    pub(in crate::scene) phase_store: SceneGpuStore,

    /// Texture pool (sparse array with reference counting)
    pub(in crate::scene) textures: SparsePool<TextureRecord, TextureId>,

    /// Texture binding version (increments on add/remove)
    pub(in crate::scene) texture_binding_version: u64,

    /// Material texture representation and capacity selected for this device.
    pub(in crate::scene) material_binding: libhelio::MaterialBindingConfig,

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
    /// Decal pool (dense array)
    pub(in crate::scene) decals: DenseArena<DecalRecord, DecalId>,
    pub(in crate::scene) decals_dirty: bool,
    pub(in crate::scene) decals_dirty_range: Option<(usize, usize)>,

    /// Reverse index: application-defined `user_tag` → handle.
    ///
    /// Lets an application find the actor it created for one of its own
    /// entities without keeping a parallel id map of its own. Tag `0` is
    /// the "untagged" default and is never indexed. Maintained by the
    /// insert/remove paths for objects and lights.
    pub(in crate::scene) objects_by_tag: HashMap<u64, ObjectId>,
    pub(in crate::scene) lights_by_tag: HashMap<u64, LightId>,

    /// GPU lights buffer (movable lights only — static/stationary are baked).
    /// Populated by `flush()` from the SceneDB world-mirror. This is a render
    /// executor-owned intermediate buffer (not SceneDB-owned) following the
    /// architecture boundary: SceneDB holds per-entity HelioGpuLight, the render
    /// executor owns the compacted movable-only buffer.
    pub(crate) lights_gpu: helio_core::scene::GrowableBuffer<libhelio::GpuLight>,

    /// Number of movable lights in `lights_gpu` (static/stationary excluded from runtime).
    pub(crate) movable_light_count: u32,

    /// Sorted instance data buffer (mesh+material sorted, render executor-owned).
    pub(crate) instances_gpu: helio_core::scene::GrowableBuffer<libhelio::GpuInstanceData>,

    /// Draw call templates (one per unique mesh+material pair).
    pub(crate) draw_calls_gpu: helio_core::scene::GrowableBuffer<libhelio::GpuDrawCall>,

    /// Indirect draw command buffer.
    pub(crate) indirect_gpu: helio_core::scene::GrowableBuffer<libhelio::DrawIndexedIndirectArgs>,

    /// Per-instance AABB buffer (for GPU culling).
    pub(crate) aabbs_gpu: helio_core::scene::GrowableBuffer<libhelio::GpuInstanceAabb>,

    /// Number of instances in the sorted buffer.
    pub(crate) instance_count: u32,

    /// Number of draw calls in the draw call buffer.
    pub(crate) draw_count: u32,

    /// True when the objects list has changed and the GPU instance/draw_call/indirect
    /// buffers need to be rebuilt from scratch (sorted by mesh+material for instancing).
    pub(in crate::scene) objects_dirty: bool,

    /// True when a Static or Stationary object has been added or removed since the last
    /// shadow atlas render. Triggers a re-render of the static shadow atlas.
    pub(in crate::scene) static_objects_dirty: bool,

    /// True when static/stationary geometry or lights have been added since the last bake.
    /// When this is true and a bake was previously configured, the user must explicitly
    /// call auto_bake() again to rebake the scene with the new static content.
    pub(in crate::scene) bake_invalidated: bool,

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
    pub(crate) movable_lights_generation: u64,

    /// Number of shadow-map array layers available in the active render graph.
    /// Six consecutive layers are reserved per realtime shadow caster.
    pub(in crate::scene) shadow_face_capacity: u32,

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

    // ── Radiant material graph system ──────────────────────────────────────────
    /// Registry of compiled graph WGSL snippets, keyed by content hash.
    pub radiant_graphs: RadiantGraphRegistry,

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

    // ── Foliage ───────────────────────────────────────────────────────────────
    /// Registered foliage types (grass, bushes, trees).
    pub(in crate::scene) foliage_types:
        DenseArena<super::foliage::FoliageTypeRecord, crate::handles::FoliageTypeId>,

    /// Set when foliage type or layer *topology* changes. Never set by a wind change —
    /// see `scene::foliage`'s module header for why that distinction is load-bearing.
    pub(in crate::scene) foliage_types_dirty: bool,

    /// CPU mirror of the foliage type table, published as raw bytes each frame.
    pub(in crate::scene) foliage_cpu_types: Vec<helio_foliage_core::GpuFoliageType>,

    /// Version of the foliage type table. Gates re-upload; must not advance for wind.
    pub(in crate::scene) foliage_generation: u64,

    /// Foliage layers (where each type grows).
    pub(in crate::scene) foliage_layers:
        DenseArena<super::foliage::FoliageLayerRecord, crate::handles::FoliageLayerId>,

    /// Set when the foliage layer table changes (bounds or infinite extent).
    pub(in crate::scene) foliage_layers_dirty: bool,

    /// CPU mirror of the foliage layer table, published as raw bytes each frame.
    pub(in crate::scene) foliage_cpu_layers: Vec<helio_foliage_core::GpuFoliageLayer>,

    /// Bodies that displace foliage.
    pub(in crate::scene) foliage_interactors:
        DenseArena<super::foliage::FoliageInteractorRecord, crate::handles::FoliageInteractorId>,

    pub(in crate::scene) foliage_interactors_dirty: bool,

    pub(in crate::scene) foliage_interactors_dirty_range: Option<(usize, usize)>,

    /// CPU mirror of the interactor buffer.
    pub(in crate::scene) foliage_cpu_interactors: Vec<super::foliage::GpuFoliageInteractor>,

    /// Global wind state. Advanced once per frame via `Scene::advance_wind`.
    pub(in crate::scene) wind: libhelio::Wind,

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

    // ── Reflection captures ─────────────────────────────────────────────────────
    pub(in crate::scene) reflection_captures:
        DenseArena<ReflectionCaptureRecord, ReflectionCaptureId>,

    // ── Coordinate spaces (sublevels + portals) ──────────────────────────────────
    // See `helio_core::CoordinateSpaceBuffer` for the GPU side. Sublevels and
    // portals are both just consumers of the same small fixed-size slot table
    // (slot 0 reserved, permanently identity/world-space); this free list is
    // shared between them so neither can claim a slot the other already owns.
    /// Free GPU coordinate-space slots available for reuse (LIFO).
    pub(in crate::scene) coordinate_space_free: Vec<u32>,
    /// Next never-yet-allocated coordinate-space slot. Starts at 1 (slot 0 is
    /// the permanent world-space identity, never handed out).
    pub(in crate::scene) coordinate_space_next: u32,

    /// Sublevels: a group of objects rendered through one shared, cheaply
    /// movable coordinate-space transform. See `scene::sublevels`.
    pub(in crate::scene) sublevels: SparsePool<SublevelRecord, SublevelId>,

    /// Portals: a pair of poses whose `pair_map_inverse` is one more
    /// coordinate space, used to draw a clipped duplicate of nearby geometry.
    /// See `scene::portals`.
    pub(in crate::scene) portals: SparsePool<PortalRecord, PortalId>,
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
    /// - First `flush()` will rebuild GPU buffers with automatic instancing
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
        let material_binding = libhelio::MaterialBindingConfig::for_device(&device);
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

        // ── SceneDB world-mirror bridge (Helio #210 foundation) ────
        let scenedb_ctx = EngineGpuContext::new(device.clone(), queue.clone());
        let scenedb_cfg = SceneGpuConfig {
            classes: vec![RegionClassConfig {
                capacity: 64,
                max_resident_cells: 1,
            }],
            tombstone_headroom: SceneGpuConfig::default_headroom(),
            max_cells_metadata: 16,
        };
        let mut mirror_store_raw = SceneGpuStore::new(&scenedb_ctx, scenedb_cfg);
        register_gpu_columns(&mut mirror_store_raw, &device);
        let mirror_store = Arc::new(mirror_store_raw);
        let mut world = World::new();
        world.attach_gpu_mirror(GpuMirrorHandle::new(mirror_store.clone(), queue.clone()));
        let mut scenedb_subsystems = SubsystemRegistry::new();
        scenedb_subsystems.register(SceneGpuMirrorSubsystem::new(mirror_store.clone(), queue.clone()));
        let phase_store = SceneGpuStore::new(
            &scenedb_ctx,
            SceneGpuConfig {
                classes: vec![],
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        );
        let scenedb_driver = FrameDriver::new();

        // ── Render executor-owned pipeline buffers (Phase 3b) ────────────────
        let instances_gpu = helio_core::scene::GrowableBuffer::new(
            device.clone(), 1024, wgpu::BufferUsages::STORAGE, "Helio Sorted Instances",
        );
        let draw_calls_gpu = helio_core::scene::GrowableBuffer::new(
            device.clone(), 512, wgpu::BufferUsages::STORAGE, "Helio Draw Calls",
        );
        let indirect_gpu = helio_core::scene::GrowableBuffer::new(
            device.clone(), 512,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            "Helio Indirect Args",
        );
        let aabbs_gpu = helio_core::scene::GrowableBuffer::new(
            device.clone(), 1024, wgpu::BufferUsages::STORAGE, "Helio Sorted AABBs",
        );

        Self {
            mesh_pool: MeshPool::new(device.clone()),
            gpu_scene: GpuScene::new(device.clone(), queue.clone()),
            world,
            scenedb_driver,
            scenedb_subsystems,
            mirror_store,
            phase_store,
            instances_gpu,
            draw_calls_gpu,
            indirect_gpu,
            aabbs_gpu,
            textures: SparsePool::new(),
            texture_binding_version: 0,
            material_binding,
            lights_gpu: helio_core::scene::GrowableBuffer::new(
                device.clone(),
                256,
                wgpu::BufferUsages::STORAGE,
                "Helio Scene Lights",
            ),
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
            decals: DenseArena::new(),
            decals_dirty: false,
            decals_dirty_range: None,
            objects_by_tag: HashMap::new(),
            lights_by_tag: HashMap::new(),
            movable_light_count: 0,
            instance_count: 0,
            draw_count: 0,
            objects_dirty: true,             // rebuild on first flush
            static_objects_dirty: true,      // rebuild static shadow atlas on first flush
            bake_invalidated: false,         // no bake configured yet
            prev_view_proj: glam::Mat4::IDENTITY,
            group_hidden: GroupMask::NONE,
            movable_objects_generation: 0,
            movable_lights_generation: 0,
            shadow_face_capacity: 32,
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
            radiant_graphs: RadiantGraphRegistry::new(),
            water_volumes: DenseArena::new(),
            water_volumes_dirty: false,
            water_volumes_dirty_range: None,
            foliage_types: DenseArena::new(),
            foliage_types_dirty: false,
            foliage_cpu_types: Vec::new(),
            foliage_generation: 0,
            foliage_layers: DenseArena::new(),
            foliage_layers_dirty: false,
            foliage_cpu_layers: Vec::new(),
            foliage_interactors: DenseArena::new(),
            foliage_interactors_dirty: false,
            foliage_interactors_dirty_range: None,
            foliage_cpu_interactors: Vec::new(),
            wind: libhelio::Wind::default(),
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
            reflection_captures: DenseArena::new(),
            coordinate_space_free: Vec::new(),
            coordinate_space_next: 1, // slot 0 reserved for world-space identity
            sublevels: SparsePool::new(),
            portals: SparsePool::new(),
        }
    }

    pub(crate) fn set_shadow_face_capacity(&mut self, capacity: u32) {
        self.shadow_face_capacity = capacity.clamp(1, 256);
    }

    /// Claims one GPU coordinate-space slot (see `helio_core::CoordinateSpaceBuffer`),
    /// reusing a freed slot before minting a new one. `None` when the fixed-size
    /// backing buffer (`libhelio::MAX_COORDINATE_SPACES` slots) is exhausted.
    pub(in crate::scene) fn alloc_coordinate_space(&mut self) -> Option<u32> {
        if let Some(slot) = self.coordinate_space_free.pop() {
            return Some(slot);
        }
        if self.coordinate_space_next >= libhelio::MAX_COORDINATE_SPACES {
            return None;
        }
        let slot = self.coordinate_space_next;
        self.coordinate_space_next += 1;
        Some(slot)
    }

    /// Releases a GPU coordinate-space slot for reuse and resets it to identity,
    /// so a stale transform can never leak into whatever claims the slot next.
    pub(in crate::scene) fn free_coordinate_space(&mut self, slot: u32) {
        self.gpu_scene
            .coordinate_spaces
            .update_slot(slot, glam::Mat4::IDENTITY.to_cols_array());
        self.coordinate_space_free.push(slot);
    }

    pub fn insert_voxel_volume(
        &mut self,
        descriptor: VoxelVolumeDescriptor,
    ) -> Result<VoxelVolumeId> {
        let gpu_slot = self.voxel_volumes.len() as u32;
        let id = self
            .voxel_volumes
            .insert_with(|id| VoxelVolumeRecord::new(id, gpu_slot, &descriptor));

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

    /// Set the material class and graph hash for an existing material.
    ///
    /// `material_class` selects the surface archetype template (0 = default PBR).
    /// `graph_hash` selects a WGSL snippet from the graph registry (0 = none).
    /// `feature_flags` overrides the material's feature flags (pass `None` to keep existing).
    pub fn set_material_class(
        &mut self,
        material_id: MaterialId,
        material_class: u32,
        graph_hash: u64,
        feature_flags: Option<u32>,
    ) -> Result<()> {
        let Some((slot, record)) = self.materials.get_mut_with_slot(material_id) else {
            return Err(invalid("material"));
        };
        record.gpu.material_class = material_class;
        record.graph_hash = graph_hash;
        if let Some(flags) = feature_flags {
            record.gpu.flags = flags;
        }
        let updated = self.gpu_scene.materials.update(slot, record.gpu);
        debug_assert!(updated);
        Ok(())
    }

    /// Update only the class_params of a material (no texture revalidation).
    pub fn update_material_class_params(
        &mut self,
        material_id: MaterialId,
        params: [f32; 4],
    ) -> Result<()> {
        let Some((slot, record)) = self.materials.get_mut_with_slot(material_id) else {
            return Err(invalid("material"));
        };
        record.gpu.class_params = params;
        self.gpu_scene.materials.update(slot, record.gpu);
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

    pub fn insert_decal(&mut self, decal: libhelio::GpuDecal) -> DecalId {
        let (id, slot) = self.decals.insert(DecalRecord {
            gpu: decal,
            movability: libhelio::Movability::Movable,
            user_tag: 0,
        });
        self.gpu_scene.decals.push(decal);
        self.decals_dirty = true;
        self.decals_dirty_range = match self.decals_dirty_range {
            Some((start, end)) => Some((start.min(slot), end.max(slot + 1))),
            None => Some((slot, slot + 1)),
        };
        id
    }

    pub fn insert_decal_with_tag(
        &mut self,
        decal: libhelio::GpuDecal,
        user_tag: u64,
        movability: Option<libhelio::Movability>,
    ) -> DecalId {
        let (id, slot) = self.decals.insert(DecalRecord {
            gpu: decal,
            movability: movability.unwrap_or(libhelio::Movability::Movable),
            user_tag,
        });
        self.gpu_scene.decals.push(decal);
        self.decals_dirty = true;
        self.decals_dirty_range = match self.decals_dirty_range {
            Some((start, end)) => Some((start.min(slot), end.max(slot + 1))),
            None => Some((slot, slot + 1)),
        };
        id
    }

    pub fn remove_decal(&mut self, id: DecalId) -> bool {
        let removed = self.decals.remove(id);
        if removed.is_some() {
            self.decals_dirty = true;
            self.decals_dirty_range = None; // full rebuild on remove for simplicity
            true
        } else {
            false
        }
    }

    pub fn update_decal(&mut self, id: DecalId, decal: libhelio::GpuDecal) -> bool {
        if let Some((slot, record)) = self.decals.get_mut_with_index(id) {
            record.gpu = decal;
            self.gpu_scene.decals.update(slot, decal);
            return true;
        }
        false
    }

    pub fn decal_count(&self) -> usize {
        self.decals.dense_len()
    }

    /// Returns a reference to the TLAS (Top-Level Acceleration Structure) for
    /// hardware ray tracing, if available. Returns `None` on non-RT hardware
    /// or when the TLAS has not been built yet.
    pub fn tlas(&self) -> Option<&wgpu::Tlas> {
        self.gpu_scene.tlas_manager.tlas()
    }

    /// Store a type-erased template registry on the GpuScene so the GBufferPass
    /// can find it across graph rebuilds (window resize).
    pub fn set_template_registry(&mut self, reg: Box<dyn std::any::Any + Send + Sync>) {
        self.gpu_scene.template_registry = Some(reg);
    }

    /// Store a type-erased TRANSPARENT template registry on the GpuScene so the
    /// TransparentPass can find it across graph rebuilds.
    pub fn set_transparent_template_registry(&mut self, reg: Box<dyn std::any::Any + Send + Sync>) {
        self.gpu_scene.transparent_template_registry = Some(reg);
    }
}
