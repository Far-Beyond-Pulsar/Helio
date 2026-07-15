use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct GpuReflectionCapture {
    pub position_influence: [f32; 4],
    pub box_min: [f32; 4],
    pub box_max: [f32; 4],
    pub cubemap_index: i32,
    pub capture_type: u32,
    pub blend_weight: f32,
    pub _pad0: u32,
    pub _pad1: [f32; 4],
}
