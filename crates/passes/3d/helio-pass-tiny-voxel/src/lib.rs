//! Editable voxel world and the stored-terrain Helio pass.
pub mod chunks;
mod edits;
#[cfg(feature = "engine")]
pub mod engine;
pub mod journal;
pub mod landforms;
mod picking;
pub mod world;
use bytemuck::{Pod, Zeroable};
pub use world::World;
pub const SHADER: &str = include_str!(concat!(env!("OUT_DIR"), "/planet.wgsl"));
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Params {
    pub origin: [i32; 4],
    pub fraction: [f32; 4],
    pub radial: [f32; 4],
    pub right: [f32; 4],
    pub up: [f32; 4],
    pub forward: [f32; 4],
    pub screen: [f32; 4],
    pub lighting: [f32; 4],
    pub settings: [f32; 4],
}
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuEdit {
    cell: [i32; 3],
    material: u32,
    radius: f32,
    radius_units: u32,
    pad: [f32; 2],
}
