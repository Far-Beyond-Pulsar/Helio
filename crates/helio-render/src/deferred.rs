use helio_core::{gpu, Scene};
use helio_lighting::LightingSystem;
use crate::GBuffer;

pub struct DeferredRenderer {
    pub geometry_pipeline: Option<gpu::RenderPipeline>,
    pub lighting_pipeline: Option<gpu::ComputePipeline>,
}

impl DeferredRenderer {
    pub fn new(context: &gpu::Context) -> Self {
        Self {
            geometry_pipeline: None,
            lighting_pipeline: None,
        }
    }

    pub fn render(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        scene: &Scene,
        lighting: &LightingSystem,
        gbuffer: &GBuffer,
        target_view: gpu::TextureView,
        camera_buffer: gpu::Buffer,
    ) {
        
    }
}
