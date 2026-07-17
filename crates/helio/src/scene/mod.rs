//! High-level scene management with automatic instancing.
//!
//! # Architecture
//!
//! Objects in the scene are automatically sorted by `(mesh_id, material_id)` and
//! grouped into instanced draw calls on every GPU buffer rebuild. No explicit
//! optimization step is required — the renderer always batches objects sharing
//! the same mesh and material.
//!
//! ## Zero CPU Cost at Steady State
//!
//! - Transform updates are O(1) via cached GPU slot writes
//! - No per-frame iteration over scene objects
//! - GPU frustum culling via indirect dispatch
//!
//! # Usage Example
//!
//! ```ignore
//! use helio::{Scene, ObjectDescriptor};
//! use glam::{Mat4, Vec3};
//!
//! // Create scene
//! let mut scene = Scene::new(device, queue);
//!
//! // Load resources
//! let mesh_id = scene.insert_mesh(mesh_upload);
//! let material_id = scene.insert_material(material);
//!
//! // Add objects — instancing is automatic
//! for transform in level_transforms {
//!     scene.insert_object(ObjectDescriptor {
//!         mesh: mesh_id,
//!         material: material_id,
//!         transform,
//!         bounds: [0.0, 1.0, 0.0, 1.0],
//!         flags: 0,
//!         groups: GroupMask::NONE,
//!     })?;
//! }
//!
//! // Render loop - O(1) per frame
//! loop {
//!     scene.update_camera(camera);
//!     scene.flush();
//!     renderer.render(&scene, target)?;
//! }
//! ```
//!
//! # Performance Characteristics
//!
//! | Operation | Cost |
//! |-----------|------|
//! | `insert_object` / `remove_object` | O(1) CPU + deferred rebuild |
//! | `update_object_transform` | O(1) GPU write |
//! | GPU buffer rebuild | O(N log N) sort + O(N) upload |
//! | Render (CPU) | O(1) |
//! | Draw calls (GPU) | D (one per unique mesh+material pair) |
//!
//! See the [GPU-Driven Pipeline](https://docs.farbeyondpulsar.com/helio/gpu-driven-pipeline)
//! documentation for complete architectural details.

mod actor;
mod camera;
mod core;
mod editor_debug;
mod errors;
mod flush;
mod groups;
mod helpers;
mod lifecycle;
mod multi_mesh;
mod objects;
mod postprocess;
mod resources;
mod stats;
mod types;
mod virtual_geometry;
mod voxel;
mod water;

pub use actor::{
    DecalActor, PostProcessVolumeActor, ReflectionCaptureActor, ReflectionCaptureDescriptor,
    SceneActor, SceneActorId, SceneActorTrait, WaterHitboxDescriptor, WaterHitboxActor,
    WaterVolumeDescriptor, WaterVolumeActor,
};
pub use camera::Camera;
pub use core::Scene;
pub use errors::*;
pub use types::{ObjectDescriptor, PickableObject, VoxelVolumeDescriptor};
pub use voxel::VoxelMode;

