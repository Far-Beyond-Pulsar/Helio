pub mod lights;
pub mod shadows;
pub mod gi;
pub mod probes;
pub mod volumetric;

pub use lights::*;
pub use shadows::*;
pub use gi::*;
pub use probes::*;
pub use volumetric::*;

use helio_core::gpu;
use glam::Vec3;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LightingGpuData {
    pub ambient_color: [f32; 3],
    pub num_directional_lights: u32,
    pub num_point_lights: u32,
    pub num_spot_lights: u32,
    pub num_area_lights: u32,
    pub gi_intensity: f32,
    pub shadow_bias: f32,
    pub shadow_normal_bias: f32,
    pub pcf_radius: f32,
    pub cascade_split_lambda: f32,
}

impl Default for LightingGpuData {
    fn default() -> Self {
        Self {
            ambient_color: [0.03, 0.03, 0.03],
            num_directional_lights: 0,
            num_point_lights: 0,
            num_spot_lights: 0,
            num_area_lights: 0,
            gi_intensity: 1.0,
            shadow_bias: 0.005,
            shadow_normal_bias: 0.01,
            pcf_radius: 1.0,
            cascade_split_lambda: 0.95,
        }
    }
}

pub struct LightingSystem {
    pub directional_lights: Vec<DirectionalLight>,
    pub point_lights: Vec<PointLight>,
    pub spot_lights: Vec<SpotLight>,
    pub area_lights: Vec<AreaLight>,
    pub shadow_system: ShadowSystem,
    pub gi_system: Option<GISystem>,
    pub probe_system: Option<ProbeSystem>,
    pub volumetric_system: Option<VolumetricSystem>,
    pub lighting_buffer: Option<gpu::Buffer>,
    pub light_data_buffer: Option<gpu::Buffer>,
}

impl LightingSystem {
    pub fn new(context: &gpu::Context) -> Self {
        Self {
            directional_lights: Vec::new(),
            point_lights: Vec::new(),
            spot_lights: Vec::new(),
            area_lights: Vec::new(),
            shadow_system: ShadowSystem::new(context),
            gi_system: None,
            probe_system: None,
            volumetric_system: None,
            lighting_buffer: None,
            light_data_buffer: None,
        }
    }

    pub fn add_directional_light(&mut self, light: DirectionalLight) {
        self.directional_lights.push(light);
    }

    pub fn add_point_light(&mut self, light: PointLight) {
        self.point_lights.push(light);
    }

    pub fn add_spot_light(&mut self, light: SpotLight) {
        self.spot_lights.push(light);
    }

    pub fn add_area_light(&mut self, light: AreaLight) {
        self.area_lights.push(light);
    }

    pub fn update_gpu_data(&mut self, context: &gpu::Context) {
        let _gpu_data = LightingGpuData {
            num_directional_lights: self.directional_lights.len() as u32,
            num_point_lights: self.point_lights.len() as u32,
            num_spot_lights: self.spot_lights.len() as u32,
            num_area_lights: self.area_lights.len() as u32,
            ..Default::default()
        };

        if self.lighting_buffer.is_none() {
            self.lighting_buffer = Some(context.create_buffer(gpu::BufferDesc {
                name: "lighting_data",
                size: std::mem::size_of::<LightingGpuData>() as u64,
                memory: gpu::Memory::Shared,
            }));
        }

        // Buffer mapping will be handled by implementation
    }

    pub fn cleanup(&mut self, context: &gpu::Context) {
        if let Some(buffer) = self.lighting_buffer.take() {
            context.destroy_buffer(buffer);
        }
        if let Some(buffer) = self.light_data_buffer.take() {
            context.destroy_buffer(buffer);
        }
        self.shadow_system.cleanup(context);
    }
}
