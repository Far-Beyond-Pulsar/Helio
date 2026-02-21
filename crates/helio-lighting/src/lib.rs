use bytemuck::{Pod, Zeroable};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GIMode { None, Realtime, Baked }

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GIParams {
    pub num_rays: u32,
    pub max_bounces: u32,
    pub intensity: f32,
    pub _pad: u32,
}

impl Default for GIParams {
    fn default() -> Self { Self { num_rays: 1, max_bounces: 2, intensity: 1.0, _pad: 0 } }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct LightData {
    pub position: [f32; 3],
    pub light_type: u32,
    pub direction: [f32; 3],
    pub intensity: f32,
    pub color: [f32; 3],
    pub range: f32,
}
