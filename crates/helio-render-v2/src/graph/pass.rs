//! Render pass trait and execution context

use crate::Result;
use crate::resources::ResourceManager;
use super::PassResourceBuilder;

/// Render pass trait - implemented by all rendering passes
pub trait RenderPass: Send + Sync {
    /// Unique name for this pass
    fn name(&self) -> &str;

    /// Declare resource dependencies
    ///
    /// Called once during graph building to determine pass ordering.
    /// Passes should declare which resources they read, write, or create.
    fn declare_resources(&self, _builder: &mut PassResourceBuilder) {
        // Default: no resource dependencies
    }

    /// Execute the pass
    ///
    /// Called every frame during graph execution.
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()>;
}

/// Context for pass execution
pub struct PassContext<'a> {
    /// Command encoder for recording GPU commands
    pub encoder: &'a mut wgpu::CommandEncoder,

    /// Resource manager for accessing GPU resources
    pub resources: &'a ResourceManager,

    /// Main render target
    pub target: &'a wgpu::TextureView,

    /// Bind group 0 – camera + globals (shared by all passes)
    pub global_bind_group: &'a wgpu::BindGroup,

    /// Bind group 2 – lights, shadows, env (shared by all lit passes)
    pub lighting_bind_group: &'a wgpu::BindGroup,

    /// Sky / background clear color (linear RGB)
    pub sky_color: [f32; 3],
}

impl<'a> PassContext<'a> {
    /// Begin a render pass
    pub fn begin_render_pass(
        &mut self,
        label: &str,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment>],
        depth_stencil_attachment: Option<wgpu::RenderPassDepthStencilAttachment>,
    ) -> wgpu::RenderPass {
        self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
        })
    }

    /// Begin a compute pass
    pub fn begin_compute_pass(&mut self, label: &str) -> wgpu::ComputePass {
        self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        })
    }
}
