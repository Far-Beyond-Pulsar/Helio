//! Object management for the scene (insert, update, remove, rebuild).
//!
//! Objects are the primary renderable entities in Helio. Each object references
//! a mesh and material, has a world-space transform, and can be assigned to
//! visibility groups.
//!
//! # Hybrid Slot Architecture
//!
//! The scene uses a hybrid approach to balance add/remove speed with GPU rendering
//! efficiency:
//!
//! - **Persistent mode (default):** O(1) add/remove with delta GPU uploads. Each object
//!   gets its own draw call. Ideal for dynamic scenes.
//!
//! - **Optimized mode (explicit):** Call [`Scene::optimize_scene_layout`](crate::Scene::optimize_scene_layout)
//!   to sort objects by (mesh, material) for optimal GPU cache coherency and automatic
//!   instancing. Ideal for static scenes after bulk loading.
//!
//! # Module Organization
//!
//! - [`insert`]: Object insertion (O(1) persistent mode)
//! - [`update`]: Transform and material updates (O(1) in both modes)
//! - [`remove`]: Object removal (O(1) persistent mode)
//! - [`rebuild`]: GPU buffer rebuild for both persistent and optimized modes

mod insert;
mod rebuild;
mod remove;
mod update;
