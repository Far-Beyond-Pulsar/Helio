//! Lighting feature - manages dynamic lights and uploads data to GPU

use super::{FeatureContext, PrepareContext};
use crate::features::{Feature, ShaderDefine};
use crate::Result;
use std::collections::HashMap;
use std::sync::Arc;

/// Light type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LightType {
    Directional,
    Point,
    Spot { inner_angle: f32, outer_angle: f32 },
}

/// Light configuration
#[derive(Debug, Clone, Copy)]
pub struct LightConfig {
    pub light_type: LightType,
    pub position: [f32; 3],
    /// Normalized direction vector (for directional and spot lights)
    pub direction: [f32; 3],
    pub color: [f32; 3],
    pub intensity: f32,
    /// Attenuation range (for point/spot lights)
    pub range: f32,
}

impl Default for LightConfig {
    fn default() -> Self {
        Self {
            light_type: LightType::Directional,
            position: [10.0, 15.0, 10.0],
            direction: [0.408, -0.816, 0.408],
            color: [1.0, 1.0, 1.0],
            intensity: 1.0,
            range: 100.0,
        }
    }
}

/// GPU-side light data (must match WGSL struct in geometry.wgsl)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct GpuLight {
    pub position: [f32; 3],
    pub light_type: f32,
    pub direction: [f32; 3],
    pub range: f32,
    pub color: [f32; 3],
    pub intensity: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
    pub _pad: [f32; 2],
}

/// Maximum number of lights supported per frame
pub(crate) const MAX_LIGHTS: u32 = 16;

/// Lighting feature
///
/// Creates a GPU storage buffer for lights. Scene lights are written to this
/// buffer by the renderer each frame via `render_scene()`.
pub struct LightingFeature {
    enabled: bool,
    light_buffer: Option<Arc<wgpu::Buffer>>,
}

impl LightingFeature {
    pub fn new() -> Self {
        Self {
            enabled: true,
            light_buffer: None,
        }
    }
}

impl Default for LightingFeature {
    fn default() -> Self {
        Self::new()
    }
}

impl Feature for LightingFeature {
    fn name(&self) -> &str {
        "lighting"
    }

    fn register(&mut self, ctx: &mut FeatureContext) -> Result<()> {
        let buffer_size = (std::mem::size_of::<GpuLight>() * MAX_LIGHTS as usize) as u64;
        let buf = Arc::new(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Light Storage Buffer"),
            size: buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        ctx.light_buffer = Some(buf.clone());
        self.light_buffer = Some(buf);
        log::info!("Lighting feature registered: {}B buffer", buffer_size);
        Ok(())
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> Result<()> {
        Ok(())
    }

    fn shader_defines(&self) -> HashMap<String, ShaderDefine> {
        let mut defines = HashMap::new();
        defines.insert("ENABLE_LIGHTING".into(), ShaderDefine::Bool(self.enabled));
        defines
    }

    fn is_enabled(&self) -> bool {
        self.enabled
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

