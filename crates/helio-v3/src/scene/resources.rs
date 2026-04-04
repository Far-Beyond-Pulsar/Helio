//! Zero-copy scene resource references.
//!
//! `SceneResources` provides borrowed references to GPU scene buffers. This struct is passed
//! to render passes via `PassContext::scene`, enabling zero-copy access to scene data.
//!
//! # Design Pattern: Zero-Copy Access
//!
//! Instead of cloning buffers or using `Arc<Mutex<_>>`, helio-v3 passes borrowed references:
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
//! use helio_v3::{RenderPass, PassContext, Result};
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
/// # use helio_v3::{GpuScene, RenderPass, PassContext, Result};
/// # use std::sync::Arc;
/// # let scene = GpuScene::new(Arc::new(device), Arc::new(queue));
/// // Get zero-copy references
/// let resources = scene.resources();
///
/// // Access buffers (future API)
/// // let light_buffer = resources.lights.buffer();   // &wgpu::Buffer
/// // let mesh_buffer = resources.meshes.buffer();    // &wgpu::Buffer
/// // let material_buffer = resources.materials.buffer(); // &wgpu::Buffer
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
    pub materials: &'a wgpu::Buffer,
    pub shadow_matrices: &'a wgpu::Buffer,
    pub indirect: &'a wgpu::Buffer,
    pub visibility: &'a wgpu::Buffer,
    pub instance_count: u32,
    pub draw_count: u32,
    pub light_count: u32,
    pub shadow_count: u32,
    /// Generation counter for movable objects (increments when any Movable object moves)
    pub movable_objects_generation: u64,
    /// Generation counter for movable lights (increments when any Movable light moves)
    pub movable_lights_generation: u64,
}

