//! Object management for the scene (insert, update, remove, rebuild).
//!
//! Objects are the primary renderable entities in Helio. Each object references
//! a mesh and material, has a world-space transform, and can be assigned to
//! visibility groups.
//!
//! # Automatic Instancing
//!
//! Objects sharing the same mesh and material are automatically batched into
//! instanced draw calls on every flush. No explicit optimization step is needed —
//! the renderer always sorts and groups objects by `(mesh_id, material_id)` when
//! rebuilding GPU buffers after topology changes.
//!
//! # Module Organization
//!
//! - [`insert`]: Object insertion
//! - [`update`]: Transform and material updates
//! - [`remove`]: Object removal
//! - [`rebuild`]: GPU buffer rebuild with automatic instancing

mod insert;
mod rebuild;
mod remove;
mod update;

