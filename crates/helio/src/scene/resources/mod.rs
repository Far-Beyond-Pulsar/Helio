//! Resource management for the scene (meshes, textures, materials, lights).
//!
//! Resources are shared, reference-counted assets that multiple objects can reference.
//! This module provides insert/update/remove operations for all resource types.
//!
//! # Resource Types
//!
//! - **Meshes** ([`meshes`]): Vertex and index data stored in shared GPU buffers
//! - **Textures** ([`textures`]): 2D images with samplers for material slots
//! - **Materials** ([`materials`]): Surface appearance (color, roughness, textures)
//! - **Lights** ([`lights`]): Scene lighting (point, directional, spot)
//!
//! # Reference Counting
//!
//! Meshes, textures, and materials are reference-counted. They cannot be removed
//! while objects are using them. Call [`Scene::remove_object`](crate::Scene::remove_object)
//! first to decrement reference counts.
//!
//! Lights are not reference-counted and can be removed at any time.

mod meshes;
mod textures;
mod materials;
mod lights;
