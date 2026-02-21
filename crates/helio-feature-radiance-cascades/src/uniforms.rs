use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuCascade {
    pub center_and_extent: [f32; 4],
    pub resolution_and_type: [f32; 4],
    pub texture_layer: u32,
    pub _pad0: u32, pub _pad1: u32, pub _pad2: u32,
}

impl Default for GpuCascade {
    fn default() -> Self {
        Self { center_and_extent: [0.0;4], resolution_and_type: [0.0;4], texture_layer: 0, _pad0: 0, _pad1: 0, _pad2: 0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RadianceCascadesUniforms {
    pub params: [f32; 4],
    pub cascades: [GpuCascade; 4],
}

impl RadianceCascadesUniforms {
    pub fn new(cascade_count: u32, gi_intensity: f32, integration_mode: u32, temporal_blend: f32) -> Self {
        Self { params: [cascade_count as f32, gi_intensity, integration_mode as f32, temporal_blend], cascades: [GpuCascade::default(); 4] }
    }
}

impl Default for RadianceCascadesUniforms {
    fn default() -> Self { Self { params: [0.0;4], cascades: [GpuCascade::default();4] } }
}
