//! Virtual geometry (GPU-driven meshlet rendering) for large-scale geometry.
//!
//! Virtual geometry uses GPU-driven meshlet rendering with automatic LOD selection.
//! This system allows rendering millions of triangles efficiently by:
//! - Decomposing meshes into small clusters (meshlets) of ~64 triangles
//! - Generating multiple LOD levels via vertex clustering
//! - GPU-driven frustum culling and LOD selection per-meshlet
//!
//! # Architecture
//!
//! Unlike regular objects (which use indirect rendering with CPU-side scene management),
//! virtual geometry is **fully GPU-driven**:
//! - Meshlets are stored in a flat GPU buffer
//! - Instances reference ranges in the meshlet buffer
//! - GPU compute shader performs per-meshlet culling and LOD selection
//! - No CPU readback or per-frame iteration
//!
//! # Module Organization
//!
//! - [`meshes`]: Virtual mesh upload and removal (meshletization, LOD generation)
//! - [`objects`]: Virtual object instancing and transform updates
//! - [`rebuild`]: CPU-side buffer rebuild and frame data packaging
//!
//! # Example
//!
//! ```ignore
//! use helio::{VirtualMeshUpload, VirtualObjectDescriptor};
//!
//! // Upload high-res mesh (auto-generates LODs and meshlets)
//! let vg_mesh_id = scene.insert_virtual_mesh(VirtualMeshUpload {
//!     vertices: high_res_vertices,
//!     indices: high_res_indices,
//! });
//!
//! // Instance it multiple times
//! for transform in transforms {
//!     scene.insert_virtual_object(VirtualObjectDescriptor {
//!         virtual_mesh: vg_mesh_id,
//!         transform,
//!         bounds: [0.0, 0.0, 0.0, 10.0],
//!         material_id: 0,
//!         flags: 0,
//!         groups: GroupMask::NONE,
//!     })?;
//! }
//! ```

mod meshes;
mod objects;
mod rebuild;

