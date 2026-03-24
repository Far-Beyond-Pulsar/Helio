//! GPU-native scene state with dirty tracking.
//!
//! This module provides the core scene management system for helio-v3. All scene data
//! (lights, meshes, materials, camera) lives on the GPU with dirty-tracked CPU mirrors.
//!
//! # Design Pattern: GPU-Native Scene
//!
//! Helio v3 follows the AAA GPU-driven pattern:
//!
//! 1. **All data on GPU**: Lights, meshes, materials stored in GPU buffers
//! 2. **Dirty tracking**: CPU mirrors track changes, upload only deltas
//! 3. **Zero-copy access**: Passes borrow `&wgpu::Buffer` references
//! 4. **Persistent state**: Buffers persist across frames (no per-frame allocations)
//!
//! # Components
//!
//! - [`GpuScene`] - Main scene container with dirty-tracked state
//! - [`SceneResources`] - Zero-copy resource references passed to passes
//!
//! # Performance
//!
//! - **O(changed)**: `flush()` uploads only changed data, not entire scene
//! - **O(1) at steady state**: If no changes, `flush()` is a no-op (zero cost)
//! - **Zero allocations**: All buffers are pre-allocated and reused
//! - **Zero clones**: All access is by reference
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
//! // scene.lights.add(PointLight { ... });
//! // scene.meshes.add(Mesh { ... });
//!
//! // Flush dirty data to GPU (zero-cost if nothing changed)
//! scene.flush();
//!
//! // Passes receive zero-copy references
//! let resources = scene.resources();
//! // let light_buffer = resources.lights.buffer(); // &wgpu::Buffer
//! ```

mod gpu_scene;
pub mod managers;
mod resources;

pub use gpu_scene::GpuScene;
pub use managers::*;
pub use resources::SceneResources;

