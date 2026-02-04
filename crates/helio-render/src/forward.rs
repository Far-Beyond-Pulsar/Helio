use helio_core::{gpu, Scene};
use helio_lighting::LightingSystem;
use crate::RendererConfig;

pub struct ForwardRenderer {
    pub pipeline: Option<gpu::RenderPipeline>,
}

impl ForwardRenderer {
    pub fn new(context: &gpu::Context, config: &RendererConfig) -> Self {
        Self {
            pipeline: None,
        }
    }

    pub fn render(
        &mut self,
        encoder: &mut gpu::CommandEncoder,
        scene: &Scene,
        lighting: &LightingSystem,
        target_view: gpu::TextureView,
        camera_buffer: gpu::Buffer,
    ) {
        
    }
}
