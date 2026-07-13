//! Public types and internal record structures for scene management.

use glam::Mat4;
use helio_core::{GpuDrawCall, GpuInstanceAabb, GpuInstanceData, GpuLight, GpuMaterial};
use libhelio::{GpuMeshletEntry, GpuPostProcessVolume, GpuWaterHitbox, GpuWaterVolume};
use bytemuck::{Pod, Zeroable};

use crate::groups::GroupMask;
use crate::handles::{MaterialId, MeshId, ObjectId};
use crate::material::MaterialTextures;
use crate::vg::VirtualMeshId;

/// Descriptor for creating a voxel volume in the scene
#[derive(Debug, Clone)]
pub struct VoxelVolumeDescriptor {
    pub voxel_size: f32,
    pub root_extent: f32,
    pub local_to_world: glam::Mat4,
    pub movability: Option<libhelio::Movability>,
    pub mode: Option<super::voxel::VoxelMode>,
    pub material_palette: Vec<helio_voxel_core::GpuVoxelMaterial>,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Public Types
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Descriptor for creating a renderable object in the scene.
///
/// Objects are the primary renderable entities in Helio. Each object references
/// a mesh and material, has a world-space transform, and can be assigned to
/// visibility groups.
///
/// # Performance
/// - `insert_object()` is O(1) in persistent mode (default)
/// - Call `optimize_scene_layout()` after bulk loading for optimal GPU batching
///
/// # Example
/// ```ignore
/// let obj_id = scene.insert_object(ObjectDescriptor {
///     mesh: mesh_id,
///     material: material_id,
///     transform: Mat4::from_translation(Vec3::new(0.0, 0.0, 0.0)),
///     bounds: [0.0, 1.5, 0.0, 1.6],  // sphere: center (xyz) + radius (w)
///     flags: 0,                      // bit 0 = casts shadow, bit 1 = receives shadow
///     groups: GroupMask::NONE,       // always visible
/// })?;
/// ```
#[derive(Debug, Clone, Copy)]
pub struct ObjectDescriptor {
    /// Mesh handle returned by [`crate::Scene::insert_mesh`].
    pub mesh: MeshId,

    /// Material handle returned by [`crate::Scene::insert_material`].
    pub material: MaterialId,

    /// Object's model matrix (world transform, column-major).
    ///
    /// Transforms vertices from object-local space to world space.
    pub transform: Mat4,

    /// Bounding sphere in world space: `[center.x, center.y, center.z, radius]`.
    ///
    /// Used for GPU frustum culling. Must accurately enclose the mesh or the object
    /// will be incorrectly culled.
    ///
    /// # Important
    /// This sphere is stored alongside the model matrix and transformed through it
    /// at cull time. The radius scales by the maximum scale component of the transform.
    pub bounds: [f32; 4],

    /// Render flags: bit 0 = casts shadow, bit 1 = receives shadow.
    pub flags: u32,

    /// Group membership bitmask for batch visibility control.
    ///
    /// An object is hidden if **any** of its groups are currently hidden.
    /// Use [`GroupMask::NONE`] for objects that are always visible.
    pub groups: GroupMask,

    /// Movability mode. Defaults to Static when None.
    /// Set to Some(Movability::Movable) for objects that will update their transforms at runtime.
    pub movability: Option<libhelio::Movability>,

    /// Application-defined tag stored alongside the object on the CPU side.
    ///
    /// Helio does not interpret this value. Engines use it to associate a scene
    /// object with an external identifier (e.g. a hashed scene-database ID)
    /// so that [`crate::ScenePicker`] hits can be resolved back to the owning
    /// entity without maintaining a separate reverse-lookup map.
    ///
    /// Defaults to `0` (no tag).
    pub user_tag: u64,
}

/// A scene object exposed for CPU-side picking queries.
///
/// Returned by [`crate::Scene::iter_pickable_objects`].  The caller builds a
/// [`crate::ScenePicker`] by registering meshes and then syncing instances.
#[derive(Debug, Clone, Copy)]
pub struct PickableObject {
    /// Stable handle to the object.
    pub id: ObjectId,

    /// Mesh handle — used to look up the per-mesh BVH in [`crate::ScenePicker`].
    pub mesh_id: MeshId,

    /// Current world-space model matrix (updated by `update_object_transform`).
    pub transform: Mat4,

    /// Application-defined tag — see [`ObjectDescriptor::user_tag`].
    pub user_tag: u64,
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// Internal Record Types (pub(crate) - not part of public API)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

/// Internal record for a material slot.
///
/// Stores GPU data, texture references, and reference count for automatic cleanup.
#[derive(Debug, Clone)]
pub(crate) struct MaterialRecord {
    /// GPU-side material parameters (base color, roughness, metallic, etc.).
    pub gpu: GpuMaterial,

    /// CPU-side texture references (for ref counting).
    pub textures: MaterialTextures,

    /// Number of objects currently using this material.
    pub ref_count: u32,
}

/// Internal record for a light.
///
/// Simple wrapper around GPU light data.
#[derive(Debug, Clone)]
pub(crate) struct LightRecord {
    /// GPU-side light parameters (position, color, intensity, type, etc.).
    pub gpu: GpuLight,
    /// Mobility mode (Static, Stationary, Movable).
    pub movability: libhelio::Movability,
    /// Application-defined tag — see [`ObjectDescriptor::user_tag`].
    pub user_tag: u64,
}

/// Internal record for a scene object.
///
/// Stores all data needed to render an object: mesh/material references,
/// GPU instance data, bounding volume, and cached GPU slot.
#[derive(Debug, Clone)]
pub(crate) struct ObjectRecord {
    /// Mesh handle (for ref counting).
    pub mesh: MeshId,

    /// Material handle (for ref counting).
    pub material: MaterialId,

    /// Group membership bitmask.
    pub groups: GroupMask,

    /// Movability mode (Static, Stationary, Movable).
    pub movability: libhelio::Movability,

    /// Application-defined tag — see [`ObjectDescriptor::user_tag`].
    pub user_tag: u64,

    /// GPU instance data (model matrix, normal matrix, bounds, mesh/material indices).
    pub instance: GpuInstanceData,

    /// Axis-aligned bounding box (converted from sphere bounds).
    pub aabb: GpuInstanceAabb,

    /// Draw call template (index count, first index, etc.).
    pub draw: GpuDrawCall,

    /// Cached GPU buffer slot for O(1) transform updates.
    ///
    /// - In persistent mode: equals dense array index
    /// - In optimized mode: set by `rebuild_instance_buffers_optimized()`
    pub gpu_slot: u32,
}

/// Internal record for a texture.
///
/// Stores GPU texture, view, sampler, and reference count.
#[derive(Debug)]
pub(crate) struct TextureRecord {
    /// GPU texture resource (owned, not accessed directly).
    pub _texture: wgpu::Texture,

    /// Texture view for shader binding.
    pub view: wgpu::TextureView,

    /// Sampler for texture filtering.
    pub sampler: wgpu::Sampler,

    /// Number of materials currently using this texture.
    pub ref_count: u32,
}

/// Internal record for a virtual mesh (meshlet-based LOD mesh).
///
/// Stores mesh handles for each LOD level and precomputed meshlet descriptors.
#[derive(Debug, Clone)]
pub(crate) struct VirtualMeshRecord {
    /// Mesh pool handles for each uploaded LOD level.
    pub mesh_ids: Vec<MeshId>,

    /// Precomputed meshlet descriptors for all LODs combined.
    pub meshlets: Vec<GpuMeshletEntry>,

    /// Conservative mesh-local sphere used for object culling and LOD distance.
    pub local_bounds: [f32; 4],

    /// Number of valid LOD ranges.
    pub lod_count: u32,

    /// Measured accumulated object-space simplification errors.
    pub lod_errors: [f32; libhelio::VG_LOD_LEVELS],

    /// Per-LOD offsets into `meshlets`, before the shared frame-buffer base is applied.
    pub lod_first_meshlets: [u32; libhelio::VG_LOD_LEVELS],

    /// Per-LOD meshlet counts.
    pub lod_meshlet_counts: [u32; libhelio::VG_LOD_LEVELS],

    /// Largest per-LOD meshlet count.
    pub max_meshlet_count: u32,

    /// Number of virtual objects currently using this mesh.
    pub ref_count: u32,
}

/// Internal record for a virtual object instance.
///
/// References a virtual mesh and stores instance data for GPU-driven rendering.
#[derive(Debug, Clone)]
pub(crate) struct VirtualObjectRecord {
    /// Virtual mesh handle.
    pub virtual_mesh: VirtualMeshId,

    /// Group membership bitmask.
    pub groups: GroupMask,

    /// Movability mode (Static, Stationary, Movable).
    pub movability: libhelio::Movability,

    /// GPU instance data (model matrix, normal matrix, bounds).
    pub instance: GpuInstanceData,
}

/// Internal record for a water volume.
///
/// Stores GPU-side water volume parameters for water rendering passes.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct WaterVolumeRecord {
    /// GPU water volume descriptor with all rendering parameters.
    pub gpu: GpuWaterVolume,
}

/// Internal record for a water hitbox.
///
/// Stores the previous and current AABB extents used by the heightfield
/// simulation to produce realistic wave displacement on object entry/exit.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct WaterHitboxRecord {
    /// GPU hitbox data (old bounds, new bounds, displacement params).
    pub gpu: GpuWaterHitbox,
}

/// Internal record for a post-process volume.
///
/// Stores GPU-side volume parameters for post-processing volume evaluation.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct PostProcessVolumeRecord {
    /// GPU post-process volume descriptor with bounds, priority, and settings.
    pub gpu: GpuPostProcessVolume,
}
