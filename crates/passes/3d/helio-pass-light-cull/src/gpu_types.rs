//! Cluster light grid bindings for forward rendering.
//!
//! Produced by `LightCullPass`, consumed by forward-lit passes and the
//! transparent pass.
#[derive(Clone, Copy)]
pub struct ClusterLightGrid<'a> {
    pub tile_light_lists: &'a wgpu::Buffer,
    pub tile_light_counts: &'a wgpu::Buffer,
    pub num_tiles_x: u32,
    pub num_tiles_y: u32,
}
