//! Shared GPU types for the Helio renderer.
//!
//! All types here are `#[repr(C)]` and implement `bytemuck::Pod + bytemuck::Zeroable`
//! so they can be uploaded directly to the GPU without any conversion.
//!
//! # Memory Layout
//!
//! All structs are designed with explicit alignment to match WGSL uniform/storage buffer rules:
//! - Scalars: natural alignment
//! - Vec3: 12 bytes but padded to 16 bytes in WGSL structs (use Vec4 in GPU types)
//! - Mat4: 64 bytes, 16-byte aligned
//! - All structs must be multiples of 16 bytes for uniform buffers

pub mod camera;
pub mod draw;
pub mod frame;
pub mod instance;
pub mod light;
pub mod material;
pub mod meshlet;
pub mod shadow;
pub mod sky;
pub mod water;

pub use camera::*;
pub use draw::*;
pub use frame::*;
pub use instance::*;
pub use light::*;
pub use material::*;
pub use meshlet::*;
pub use shadow::*;
pub use sky::{SkyActor, VolumetricClouds};
pub use water::*;
