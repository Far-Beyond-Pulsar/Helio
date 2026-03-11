//! SDF (Signed Distance Field) game engine module
//!
//! Provides real-time constructive solid geometry rendering through:
//! - An ordered edit list (shape + transform + boolean operation)
//! - Sparse brick maps with u8-quantized atlas storage
//! - Geometry clip maps with multi-level LOD centered on camera
//! - Dynamic AABB BVH for O(log n) edit culling
//! - Fullscreen ray marching with trilinear interpolation

pub mod primitives;
pub mod edit_list;
pub mod edit_bvh;
pub mod uniforms;
pub mod feature;
pub mod brick;
pub mod clip_map;
pub mod terrain;
pub mod noise;
pub mod passes;

pub use primitives::{SdfShapeType, SdfShapeParams};
pub use edit_list::{SdfEdit, SdfEditList, BooleanOp, GpuSdfEdit};
pub use uniforms::SdfGridParams;
pub use feature::{SdfFeature, PickResult};
pub use terrain::{TerrainStyle, TerrainConfig};
