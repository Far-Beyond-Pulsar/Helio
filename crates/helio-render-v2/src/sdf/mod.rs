//! SDF (Signed Distance Field) game engine module
//!
//! Provides real-time constructive solid geometry rendering through:
//! - An ordered edit list (shape + transform + boolean operation)
//! - Dense 3D grid caching via compute shader
//! - Fullscreen ray marching with hardware trilinear interpolation

pub mod primitives;
pub mod edit_list;
pub mod uniforms;
pub mod feature;
pub mod passes;

pub use primitives::{SdfShapeType, SdfShapeParams};
pub use edit_list::{SdfEdit, SdfEditList, BooleanOp, GpuSdfEdit};
pub use uniforms::SdfGridParams;
pub use feature::SdfFeature;
