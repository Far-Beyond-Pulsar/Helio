use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CascadeType { Volumetric3D = 0, HeightField2D = 1 }

pub struct CascadeData {
    pub index: u32,
    pub cascade_type: CascadeType,
    pub world_center: Vec3,
    pub world_extent: f32,
    pub probe_resolution: u32,
    pub radiance_texture: Option<wgpu::Texture>,
    pub radiance_view: Option<wgpu::TextureView>,
    pub radiance_history: Option<wgpu::Texture>,
    pub history_view: Option<wgpu::TextureView>,
    pub last_update_frame: u64,
}

impl CascadeData {
    pub fn new(index: u32, cascade_type: CascadeType, world_center: Vec3, world_extent: f32, probe_resolution: u32) -> Self {
        Self { index, cascade_type, world_center, world_extent, probe_resolution, radiance_texture: None, radiance_view: None, radiance_history: None, history_view: None, last_update_frame: 0 }
    }
    pub fn probe_spacing(&self) -> f32 { (self.world_extent * 2.0) / self.probe_resolution as f32 }
    pub fn needs_update(&self, current_frame: u64, cascade_count: u32) -> bool {
        (current_frame % cascade_count as u64) == self.index as u64
    }
}
