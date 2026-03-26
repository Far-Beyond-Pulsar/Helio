//! High-level scene management with persistent GPU-driven state.
//!
//! # Architecture
//!
//! Helio's scene uses a **hybrid slot architecture** that balances add/remove speed with
//! GPU rendering efficiency:
//!
//! - **Persistent mode (default):** O(1) add/remove with delta GPU uploads. Each object
//!   gets its own draw call. Ideal for dynamic scenes (streaming, particles, destruction).
//!
//! - **Optimized mode (explicit):** Call [`Scene::optimize_scene_layout`] to sort objects
//!   by (mesh, material) for optimal GPU cache coherency and automatic instancing. Ideal
//!   for static scenes after bulk loading.
//!
//! ## Zero CPU Cost at Steady State
//!
//! - Transform updates are O(1) via cached GPU slot writes
//! - No per-frame iteration over scene objects
//! - GPU frustum culling via indirect dispatch
//! - Delta uploads only for changed data
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
//! // Add objects (O(1) in persistent mode)
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
//! // Optimize for GPU performance (after bulk loading)
//! scene.optimize_scene_layout();
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
//! | Operation | Persistent Mode | Optimized Mode |
//! |-----------|----------------|----------------|
//! | `insert_object` | O(1) delta upload | O(1) + invalidate |
//! | `remove_object` | O(1) swap-remove | O(1) + invalidate |
//! | `update_object_transform` | O(1) GPU write | O(1) GPU write |
//! | `optimize_scene_layout` | — | O(N log N) sort |
//! | Render (CPU) | O(1) | O(1) |
//! | Draw calls (GPU) | N (one per object) | D (one per mesh+material) |
//!
//! See the [GPU-Driven Pipeline](https://docs.farbeyondpulsar.com/helio/gpu-driven-pipeline)
//! documentation for complete architectural details.

mod camera;
mod core;
mod errors;
mod groups;
mod helpers;
mod objects;
mod resources;
mod types;
mod virtual_geometry;

pub use actor::{MeshActor, SceneActor};
pub use camera::Camera;
pub use core::Scene;
pub use errors::*;
pub use types::ObjectDescriptor;

