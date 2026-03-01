//! Built-in render passes

pub mod geometry;
pub mod shadow;
pub mod billboard;

pub use geometry::GeometryPass;
pub use shadow::ShadowPass;
pub use billboard::BillboardPass;
