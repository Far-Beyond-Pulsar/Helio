use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HlfsGlobals {
    pub clip_origins: [[i32; 4]; 4],
    pub voxel_sizes: [f32; 4],
    pub frame_time: f32,
    pub temporal_decay: [f32; 4],
    pub near_field_size: f32,
    pub cascade_scale: f32,
    pub screen_size: [u32; 2],
    pub sample_count: u32,
    pub light_count: u32,
    pub dissipation_mode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct HlfsLevelState {
    pub origin: [i32; 3],
    pub origin_world: [f32; 3],
    pub half_extent: f32,
    pub voxel_size: f32,
    pub active_min: [i32; 3],
    pub active_max: [i32; 3],
}
