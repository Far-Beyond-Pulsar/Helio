use blade_graphics as gpu;
use crate::HelioError;
use std::sync::Arc;

pub type Result<T> = std::result::Result<T, HelioError>;

pub struct RenderContext {
    pub gpu_context: Arc<gpu::Context>,
}

impl RenderContext {
    pub fn new(gpu_context: Arc<gpu::Context>) -> Self {
        Self {
            gpu_context,
        }
    }
    
    pub fn begin_frame(&self) -> gpu::CommandEncoder {
        self.gpu_context.create_command_encoder(
            gpu::CommandEncoderDesc { name: "frame", buffer_count: 1 }
        )
    }
    
    pub fn submit(&self, mut encoder: gpu::CommandEncoder) {
        self.gpu_context.submit(&mut encoder);
    }
}
