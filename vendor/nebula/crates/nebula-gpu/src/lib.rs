//! # nebula-gpu
//!
//! Thin, strongly-typed wgpu helpers used internally by every baker crate.
//!
//! You can also use these utilities directly when writing custom bake passes.

pub mod buffer;
pub mod compute;
pub mod readback;
pub mod texture;

pub use buffer::{StorageBuffer, UniformBuffer};
pub use compute::{ComputePipeline, ComputePass};
pub use readback::GpuReadback;
pub use texture::{BakeTexture, BakeTextureArray, TextureFormat2D};

/// Maximum texture dimension supported — bakers clamp their requests to this.
pub const MAX_TEXTURE_DIM: u32 = 8192;

/// Default workgroup tile size used by all bake compute shaders.
pub const WORKGROUP_SIZE: u32 = 8;
