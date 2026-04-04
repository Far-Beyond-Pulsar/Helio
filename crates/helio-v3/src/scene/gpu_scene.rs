//! GPU-native scene container with persistent state.
//!
//! `GpuScene` is the central container for all scene state (lights, meshes, materials, camera).
//! Unlike traditional scene graphs, all data lives on the GPU with dirty-tracked CPU mirrors.
//!
//! # Design Pattern: GPU-Native Scene
//!
//! Helio v3 follows the **AAA GPU-driven pattern**:
//!
//! 1. **All data on GPU**: Lights, meshes, materials stored in GPU buffers
//! 2. **Dirty tracking**: CPU mirrors track changes, upload only deltas
//! 3. **Zero-copy access**: Passes borrow `&wgpu::Buffer` references
//! 4. **Persistent state**: Buffers persist across frames (no per-frame allocations)
//!
//! # Architecture
//!
//! ```text
//! GpuScene
//! ├── device: Arc<wgpu::Device>
//! ├── queue: Arc<wgpu::Queue>
//! ├── lights: GpuLightBuffer (dirty-tracked)
//! ├── meshes: GpuMeshBuffer (dirty-tracked)
//! ├── materials: GpuMaterialBuffer (dirty-tracked)
//! ├── camera: GpuCameraBuffer (dirty-tracked)
//! └── frame_count: u64
//! ```
//!
//! # Lifecycle
//!
//! ```text
//! init:
//!     scene = GpuScene::new(device, queue)
//!
//! per-frame:
//!     scene.lights.add(light)        // Marks dirty
//!     scene.meshes.update(id, mesh)  // Marks dirty
//!     scene.flush()                  // Upload dirty data (O(changed))
//!     graph.execute(&scene, ...)     // Passes read GPU buffers
//! ```
//!
//! # Performance
//!
//! - **O(changed)**: `flush()` uploads only changed data, not entire scene
//! - **O(1) at steady state**: If no changes, `flush()` is a no-op (zero cost)
//! - **Zero allocations**: All buffers are pre-allocated and reused
//! - **Zero clones**: Passes receive `&wgpu::Buffer` references
//!
//! # Example
//!
//! ```rust,no_run
//! use helio_v3::GpuScene;
//! use std::sync::Arc;
//!
//! let mut scene = GpuScene::new(
//!     Arc::new(device),
//!     Arc::new(queue),
//! );
//!
//! // Add scene objects (future API)
//! // let light_id = scene.lights.add(PointLight { ... });
//! // scene.meshes.add(Mesh { ... });
//! // scene.materials.add(Material { ... });
//!
//! // Flush dirty data to GPU (zero-cost if nothing changed)
//! scene.flush();
//!
//! // Passes receive zero-copy references
//! let resources = scene.resources();
//! // let light_buffer = resources.lights.buffer(); // &wgpu::Buffer
//! // let mesh_buffer = resources.meshes.buffer();   // &wgpu::Buffer
//! ```

use crate::scene::managers::{
    GpuAabbBuffer, GpuCameraBuffer, GpuDrawCallBuffer, GpuIndirectBuffer, GpuInstanceBuffer,
    GpuLightBuffer, GpuMaterialBuffer, GpuShadowMatrixBuffer, GpuVisibilityBuffer,
};
use crate::scene::SceneResources;
use std::sync::Arc;

/// GPU-native scene container with dirty-tracked state.
///
/// `GpuScene` manages all scene data (lights, meshes, materials, camera) with:
/// - **GPU-native storage**: All data in GPU buffers (no CPU iteration)
/// - **Dirty tracking**: Upload only changed data (zero cost at steady state)
/// - **Zero-copy access**: Passes borrow `&wgpu::Buffer` references
/// - **Persistent buffers**: No per-frame allocations
///
/// # Design
///
/// Scene data is managed by **buffer managers** (e.g., `GpuLightBuffer`, `GpuMeshBuffer`)
/// that implement `GpuSceneManager`. Each manager:
/// 1. Maintains a CPU mirror of GPU data
/// 2. Tracks dirty state with a boolean flag
/// 3. Uploads only changed data in `flush()`
///
/// # Lifecycle
///
/// ```text
/// GpuScene::new(device, queue)
/// ├── Create managers (lights, meshes, materials)
/// └── Allocate initial GPU buffers
///
/// Per-frame:
/// ├── scene.lights.add(light)        // Marks dirty
/// ├── scene.meshes.update(id, mesh)  // Marks dirty
/// ├── scene.flush()                  // Upload dirty data
/// └── graph.execute(&scene, ...)     // Passes read buffers
/// ```
///
/// # Performance
///
/// - **O(changed)**: `flush()` uploads only changed data
/// - **O(1) at steady state**: If no changes, `flush()` is a no-op
/// - **Zero allocations**: All buffers are pre-allocated
/// - **Zero clones**: All access is by reference
///
/// # Example
///
/// ```rust,no_run
/// use helio_v3::GpuScene;
/// use std::sync::Arc;
///
/// let mut scene = GpuScene::new(
///     Arc::new(device),
///     Arc::new(queue),
/// );
///
/// // Frame 1: Add lights (dirty = true)
/// // scene.lights.add(PointLight { position: [0.0, 5.0, 0.0], color: [1.0, 1.0, 1.0] });
/// // scene.lights.add(SpotLight { position: [10.0, 5.0, 0.0], direction: [0.0, -1.0, 0.0] });
/// scene.flush(); // Uploads to GPU
///
/// // Frame 2: No changes (dirty = false)
/// scene.flush(); // No-op (zero cost)
///
/// // Frame 3: Update one light (dirty = true)
/// // scene.lights.update(light_id, PointLight { position: [1.0, 5.0, 0.0], color: [1.0, 0.0, 0.0] });
/// scene.flush(); // Uploads only changed data
/// ```
pub struct GpuScene {
    /// GPU device (shared across scene).
    ///
    /// Used by managers to create buffers when capacity grows.
    pub device: Arc<wgpu::Device>,

    /// GPU queue (shared across scene).
    ///
    /// Used by `flush()` to upload data to GPU.
    pub queue: Arc<wgpu::Queue>,

    /// Current frame number (starts at 0).
    ///
    /// Incremented each frame, useful for time-based effects.
    pub frame_count: u64,

    /// Render target width in pixels.
    pub width: u32,

    /// Render target height in pixels.
    pub height: u32,

    /// Generation counter for movable objects - increments when any Movable object moves.
    /// Used by shadow caching to detect movement.
    pub movable_objects_generation: u64,

    /// Generation counter for movable lights - increments when any Movable light moves.
    /// Used by shadow caching to detect movement.
    pub movable_lights_generation: u64,

    pub camera: GpuCameraBuffer,
    pub instances: GpuInstanceBuffer,
    pub aabbs: GpuAabbBuffer,
    pub draw_calls: GpuDrawCallBuffer,
    pub lights: GpuLightBuffer,
    pub materials: GpuMaterialBuffer,
    pub shadow_matrices: GpuShadowMatrixBuffer,
    pub indirect: GpuIndirectBuffer,
    pub visibility: GpuVisibilityBuffer,

    // ── Shadow partition buffers (Unreal-style static/dynamic split) ──────────
    // NOTE: Both pass kinds use `instances` (the main transforms buffer) at binding 1.
    // We only partition the INDIRECT DRAW CALL buffers so that each atlas can be
    // rendered with a single `multi_draw_indexed_indirect` call. This means
    // `first_instance` in each indirect entry is the object's dense_index into
    // `instances`, keeping transform data in a single place that stays in sync
    // when `update_object_transform` writes to it.
    //
    // Obsolete approach (DO NOT restore): splitting instance data into two copies
    // (shadow_static_instances / shadow_movable_instances) caused dynamic shadows to
    // freeze because the copies were never updated on `update_object_transform`.

    /// Indirect draw commands for Static/Stationary objects (indexes into `instances`).
    pub shadow_static_indirect: GpuIndirectBuffer,
    /// Indirect draw commands for Movable objects (indexes into `instances`).
    pub shadow_movable_indirect: GpuIndirectBuffer,
    /// Number of draw calls in shadow_static_indirect.
    pub shadow_static_draw_count: u32,
    /// Number of draw calls in shadow_movable_indirect.
    pub shadow_movable_draw_count: u32,
    /// Increments when the static object set changes (add/remove of Static/Stationary objects).
    /// Used by ShadowPass to know when to re-render the static shadow atlas.
    pub static_objects_generation: u64,
}

impl GpuScene {
    /// Creates a new GPU scene.
    ///
    /// Initializes managers with default capacities (e.g., 1024 lights, 4096 meshes).
    /// Buffers are pre-allocated to avoid reallocation during gameplay.
    ///
    /// # Parameters
    ///
    /// - `device`: GPU device (wrapped in `Arc` for sharing)
    /// - `queue`: GPU queue (wrapped in `Arc` for sharing)
    ///
    /// # Performance
    ///
    /// - **O(1)**: Allocates buffers once at startup
    /// - **Pre-allocation**: Managers allocate initial capacity (e.g., 1024 lights)
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use helio_v3::GpuScene;
    /// use std::sync::Arc;
    ///
    /// let scene = GpuScene::new(
    ///     Arc::new(device),
    ///     Arc::new(queue),
    /// );
    /// ```
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let camera = GpuCameraBuffer::new(&device);
        let instances = GpuInstanceBuffer::new(device.clone());
        let aabbs = GpuAabbBuffer::new(device.clone());
        let draw_calls = GpuDrawCallBuffer::new(device.clone());
        let lights = GpuLightBuffer::new(device.clone());
        let materials = GpuMaterialBuffer::new(device.clone());
        let shadow_matrices = GpuShadowMatrixBuffer::new(device.clone());
        let indirect = GpuIndirectBuffer::new(device.clone());
        let visibility = GpuVisibilityBuffer::new(device.clone());
        let shadow_static_indirect = GpuIndirectBuffer::new(device.clone());
        let shadow_movable_indirect = GpuIndirectBuffer::new(device.clone());
        Self {
            device,
            queue,
            frame_count: 0,
            width: 0,
            height: 0,
            movable_objects_generation: 0,
            movable_lights_generation: 0,
            static_objects_generation: 0,
            camera,
            instances,
            aabbs,
            draw_calls,
            lights,
            materials,
            shadow_matrices,
            indirect,
            visibility,
            shadow_static_indirect,
            shadow_movable_indirect,
            shadow_static_draw_count: 0,
            shadow_movable_draw_count: 0,
        }
    }

    /// Returns zero-copy references to GPU resources.
    ///
    /// Creates a `SceneResources` struct with borrowed references to all GPU buffers.
    /// Passes receive this struct via `PassContext::scene`.
    ///
    /// # Performance
    ///
    /// - **O(1)**: Returns borrowed references (no clones)
    /// - **Zero-copy**: All fields are `&wgpu::Buffer` references
    ///
    /// # Lifetime
    ///
    /// The returned `SceneResources<'_>` borrows `self`, ensuring buffers are not
    /// freed while passes are using them.
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use helio_v3::GpuScene;
    /// # use std::sync::Arc;
    /// # let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
    /// let resources = scene.resources();
    /// // let light_buffer = resources.lights.buffer(); // &wgpu::Buffer
    /// // let mesh_buffer = resources.meshes.buffer();   // &wgpu::Buffer
    /// ```
    pub fn resources(&self) -> SceneResources<'_> {
        SceneResources {
            camera: self.camera.buffer(),
            instances: self.instances.buffer(),
            aabbs: self.aabbs.buffer(),
            draw_calls: self.draw_calls.buffer(),
            lights: self.lights.buffer(),
            materials: self.materials.buffer(),
            shadow_matrices: self.shadow_matrices.buffer(),
            indirect: self.indirect.buffer(),
            visibility: self.visibility.buffer(),
            instance_count: self.instances.len() as u32,
            draw_count: self.draw_calls.len() as u32,
            light_count: self.lights.len() as u32,
            shadow_count: self.shadow_matrices.len() as u32,
            movable_objects_generation: self.movable_objects_generation,
            movable_lights_generation: self.movable_lights_generation,
            shadow_static_indirect: self.shadow_static_indirect.buffer(),
            shadow_movable_indirect: self.shadow_movable_indirect.buffer(),
            shadow_static_draw_count: self.shadow_static_draw_count,
            shadow_movable_draw_count: self.shadow_movable_draw_count,
            static_objects_generation: self.static_objects_generation,
        }
    }

    /// Flushes all dirty managers to GPU.
    ///
    /// Uploads changed data from CPU mirrors to GPU buffers. This is a **zero-cost operation**
    /// at steady state (if no changes occurred since last flush).
    ///
    /// # Performance
    ///
    /// - **O(changed)**: Uploads only changed data, not entire scene
    /// - **O(1) at steady state**: If all managers are clean, this is a no-op
    /// - **Zero allocations**: All buffers are pre-allocated
    ///
    /// # Usage
    ///
    /// Call `flush()` once per frame **before** `RenderGraph::execute()`:
    ///
    /// ```rust,no_run
    /// # use helio_v3::{GpuScene, RenderGraph};
    /// # use std::sync::Arc;
    /// # let mut scene = GpuScene::new(Arc::new(device), Arc::new(queue));
    /// # let mut graph = RenderGraph::new(&device, &queue);
    /// # let target = &view;
    /// # let depth = &depth_view;
    /// // Update scene objects
    /// // scene.lights.add(light);
    /// // scene.meshes.update(id, mesh);
    ///
    /// // Flush changes to GPU (zero-cost if nothing changed)
    /// scene.flush();
    ///
    /// // Execute render graph (passes read GPU buffers)
    /// // graph.execute(&scene, target, depth);
    /// ```
    ///
    /// # Example: Dirty Tracking
    ///
    /// ```text
    /// Frame 1:
    ///   scene.lights.add(light)  // lights.dirty = true
    ///   scene.flush()            // Uploads light buffer
    ///
    /// Frame 2:
    ///   scene.flush()            // No-op (lights.dirty = false)
    ///
    /// Frame 3:
    ///   scene.lights.update(id, light)  // lights.dirty = true
    ///   scene.flush()                   // Uploads light buffer
    /// ```
    pub fn flush(&mut self) {
        let queue: &wgpu::Queue = &self.queue;
        self.camera.flush(queue);
        self.instances.flush(queue);
        self.aabbs.flush(queue);
        self.draw_calls.flush(queue);
        self.lights.flush(queue);
        self.materials.flush(queue);
        self.shadow_matrices.flush(queue);
        self.indirect.flush(queue);
        self.visibility.flush(queue);
        self.shadow_static_indirect.flush(queue);
        self.shadow_movable_indirect.flush(queue);
    }
}

