//! Zero-copy scene resource references.
//!
//! `SceneResources` provides borrowed references to GPU scene buffers. This struct is passed
//! to render passes via `PassContext::scene`, enabling zero-copy access to scene data.
//!
//! # Design Pattern: Zero-Copy Access
//!
//! Instead of cloning buffers or using `Arc<Mutex<_>>`, helio-core passes borrowed references:
//!
//! ```text
//! Traditional (bad):
//! ├── Arc<Mutex<GpuScene>> (locks, overhead)
//! └── scene.lock().unwrap() (runtime cost)
//!
//! Helio v3 (good):
//! ├── SceneResources<'a> (zero-copy references)
//! └── ctx.scene.lights.buffer() (no locks, no clones)
//! ```
//!
//! # Lifetime
//!
//! The `'a` lifetime ensures that all borrowed references outlive the context. This prevents
//! dangling references and ensures safety without runtime overhead.
//!
//! # Performance
//!
//! - **O(1)**: Creating `SceneResources` is constant-time (no allocations)
//! - **Zero clones**: All fields are references (`&`)
//! - **Zero locks**: No `Arc<Mutex<_>>` or `RwLock<_>` (single-threaded per frame)
//!
//! # Example
//!
//! ```rust,no_run
//! use helio_core::{RenderPass, PassContext, Result};
//!
//! struct MyPass {
//!     pipeline: wgpu::RenderPipeline,
//! }
//!
//! impl RenderPass for MyPass {
//!     fn name(&self) -> &'static str {
//!         "MyPass"
//!     }
//!
//!     fn render_pass_descriptor<'a>(
//!         &'a self,
//!         _: &'a wgpu::TextureView,
//!         _: &'a wgpu::TextureView,
//!         _: &'a helio_core::FrameResources<'a>,
//!     ) -> Option<wgpu::RenderPassDescriptor<'a>> {
//!         None
//!     }
//!
//!     fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
//!         // Zero-copy access to scene resources
//!         // let light_buffer = ctx.scene.lights.buffer();   // &wgpu::Buffer
//!         // let mesh_buffer = ctx.scene.meshes.buffer();    // &wgpu::Buffer
//!         // let material_buffer = ctx.scene.materials.buffer(); // &wgpu::Buffer
//!
//!         // Use buffers in bind groups (no clones)
//!         // let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
//!         //     layout: &layout,
//!         //     entries: &[
//!         //         wgpu::BindGroupEntry {
//!         //             binding: 0,
//!         //             resource: light_buffer.as_entire_binding(),
//!         //         },
//!         //     ],
//!         //     label: Some("Scene Bind Group"),
//!         // });
//!
//!         Ok(())
//!     }
//! }
//! ```

use crate::component::ComponentRegistry;

/// Zero-copy references to GPU scene resources.
///
/// `SceneResources` provides borrowed references (`&`) to all scene buffers. This enables
/// passes to access scene data without clones or locks.
///
/// # Design
///
/// All fields are references to managers that implement `GpuSceneManager`. Passes access
/// GPU buffers via `resources.lights.buffer()`, `resources.meshes.buffer()`, etc.
///
/// # Lifetime
///
/// The `'a` lifetime ties this struct to the `GpuScene` it was created from. This ensures
/// that buffers are not freed while passes are using them.
///
/// # Performance
///
/// - **O(1)**: Creating this struct is constant-time (no allocations)
/// - **Zero clones**: All fields are references
/// - **Zero locks**: No `Arc<Mutex<_>>` (single-threaded per frame)
///
/// # Example
///
/// ```rust,no_run
/// # use helio_core::{GpuScene, RenderPass, PassContext, Result};
/// # use std::sync::Arc;
/// # fn example(device: wgpu::Device, queue: wgpu::Queue) {
/// # let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
/// // Get zero-copy references
/// let resources = scene.resources();
///
/// // Access buffers (future API)
/// // let light_buffer = resources.lights.buffer();   // &wgpu::Buffer
/// // let mesh_buffer = resources.meshes.buffer();    // &wgpu::Buffer
/// // let material_buffer = resources.materials.buffer(); // &wgpu::Buffer
/// # }
/// ```
///
/// # Future API
///
/// When managers are implemented, this struct will have fields like:
///
/// ```rust,ignore
/// pub struct SceneResources<'a> {
///     pub lights: &'a GpuLightBuffer,
///     pub meshes: &'a GpuMeshBuffer,
///     pub materials: &'a GpuMaterialBuffer,
///     pub camera: &'a GpuCameraBuffer,
/// }
/// ```
pub struct SceneResources<'a> {
    pub camera: &'a wgpu::Buffer,
    pub instances: &'a wgpu::Buffer,
    pub aabbs: &'a wgpu::Buffer,
    pub draw_calls: &'a wgpu::Buffer,
    pub lights: &'a wgpu::Buffer,
    pub decals: &'a wgpu::Buffer,
    pub decal_count: u32,
    pub materials: &'a wgpu::Buffer,
    pub shadow_matrices: &'a wgpu::Buffer,
    pub indirect: &'a wgpu::Buffer,
    pub visibility: &'a wgpu::Buffer,
    /// Per-draw-call-group compacted original instance slots surviving GPU
    /// frustum culling (see `IndirectDispatchPass`). Passes drawing through
    /// `indirect`/`draw_calls` should index `instances` through this buffer
    /// (`instances[compacted_indices[instance_index]]`) rather than directly.
    pub compacted_indices: &'a wgpu::Buffer,
    /// Final surviving instance slots after frustum + Hi-Z occlusion culling.
    /// Consumers drawing through `indirect`/`draw_calls` should use this one,
    /// not `compacted_indices` (which is frustum-only, an intermediate stage).
    pub compacted_indices_2: &'a wgpu::Buffer,
    /// Coordinate-space transforms (current frame). Slot 0 = identity. Shaders
    /// index this with the id packed into `GpuInstanceData.flags` bits 8-15
    /// (`libhelio::coordinate_space`) to place sublevel/portal content.
    pub coordinate_spaces: &'a wgpu::Buffer,
    /// Coordinate-space transforms as of the previous frame — same indexing as
    /// `coordinate_spaces`, used to compute correct per-space motion vectors.
    pub coordinate_spaces_prev: &'a wgpu::Buffer,
    pub instance_count: u32,
    pub draw_count: u32,
    pub light_count: u32,
    pub shadow_count: u32,
    /// Generation counter for movable objects (increments when any Movable object moves)
    pub movable_objects_generation: u64,
    /// Generation counter for movable lights (increments when any Movable light moves)
    pub movable_lights_generation: u64,
    /// Generation counter for camera (increments when camera view/projection changes)
    pub camera_generation: u64,

    // ── Shadow partition buffers (Unreal-style static/dynamic split) ──────────
    // Both passes use `instances` (main buffer) — only the indirect call lists differ.
    /// Indirect draw commands for Static/Stationary objects (first_instance into main `instances`).
    pub shadow_static_indirect: &'a wgpu::Buffer,
    /// Indirect draw commands for Movable objects (first_instance into main `instances`).
    pub shadow_movable_indirect: &'a wgpu::Buffer,
    /// Number of draw calls in shadow_static_indirect.
    pub shadow_static_draw_count: u32,
    /// Number of draw calls in shadow_movable_indirect.
    pub shadow_movable_draw_count: u32,
    /// Increments when static object topology changes; triggers static atlas re-render.
    pub static_objects_generation: u64,
    /// Number of movable lights in the lights buffer (static/stationary excluded from runtime).
    pub movable_light_count: u32,
    /// Per-caster dirty generation counters (one per shadow caster slot, 42 max).
    /// Copied from GpuScene::per_caster_dirty_gen each frame. ShadowPass compares against
    /// its own last-rendered gen to decide which caster faces need re-rendering.
    pub per_caster_dirty_gen: [u64; 42],

    /// Component registry for type-erased storage access.
    pub components: &'a ComponentRegistry,

    pub voxel_volumes: &'a wgpu::Buffer,
    pub voxel_edit_ring: &'a wgpu::Buffer,
    pub voxel_brick_pool: &'a wgpu::Buffer,
    pub voxel_data_pool: &'a wgpu::Buffer,
    pub voxel_volume_count: u32,
    pub voxel_volumes_generation: u64,

    /// Material class ranges for the GBuffer pass: [(class, graph_hash, start, count), ...]
    /// Each range is uniform in both material_class and graph_hash so a single
    /// PSO works for all indirect entries it covers.
    /// Built during scene flush.
    pub material_class_ranges: &'a [(u32, u64, u32, u32)],
    pub transparent_material_class_ranges: &'a [(u32, u64, u32, u32)],
    /// Forward-shaded material class ranges (excluded from GBuffer pass).
    pub forward_material_class_ranges: &'a [(u32, u64, u32, u32)],

    /// Graph hashes indexed by material slot. Populated during flush.
    pub material_graph_hashes: &'a [u64],

    /// Compiled graph WGSL snippets keyed by hash. Populated during flush.
    pub graph_wgsl_snippets: &'a std::collections::HashMap<u64, String>,

    /// Custom template registrations that survive graph rebuilds.
    /// GBufferPass downcasts to `RadiantTemplateRegistry` before each frame.
    pub template_registry: &'a Option<Box<dyn std::any::Any + Send + Sync>>,

    /// Separate template registry for transparent materials (water, glass, etc.).
    /// TransparentPass reads this instead of `template_registry` to avoid picking
    /// up gbuffer templates with incompatible bind group layouts.
    pub transparent_template_registry: &'a Option<Box<dyn std::any::Any + Send + Sync>>,

    /// Reflection capture storage buffer.
    pub reflection_captures: &'a wgpu::Buffer,
    /// Number of reflection captures in the buffer.
    pub reflection_capture_count: u32,

    /// Active portals' render data (`libhelio::GpuPortalView`). Consumed by
    /// `helio-pass-portal-cull` / `helio-pass-portal-instances`.
    pub portal_views: &'a wgpu::Buffer,
    /// Number of active portals in `portal_views`.
    pub portal_view_count: u32,

    /// Every valid portal *chain* (`libhelio::GpuPortalChain`) up to
    /// `libhelio::MAX_CHAIN_DEPTH` deep — every sequence of portal indices,
    /// including repeats, that represents "look through this portal, then
    /// through this one, then...". This is what makes portals recursively
    /// reflect each other automatically: content is mapped through the
    /// *composed* transform of a whole chain, not just one portal, and each
    /// stage is independently clip-tested against its own portal's opening.
    /// Rebuilt whenever the portal set changes (add/remove/pose update), not
    /// every frame — see `helio::Scene::add_portal` and neighbors.
    pub portal_chains: &'a wgpu::Buffer,
    /// Number of valid chains in `portal_chains`.
    pub portal_chain_count: u32,

    /// Whether hardware ray tracing (TLAS + ray queries) is available.
    pub rt_available: bool,
}
