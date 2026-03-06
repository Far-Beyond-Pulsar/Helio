//! Render pass trait and execution context

use crate::Result;
use crate::resources::ResourceManager;
use crate::profiler::GpuProfiler;
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
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub resources: &'a ResourceManager,
    pub target: &'a wgpu::TextureView,
    pub depth_view: &'a wgpu::TextureView,
    pub global_bind_group: &'a wgpu::BindGroup,
    pub lighting_bind_group: &'a wgpu::BindGroup,
    /// Clear color used when `has_sky` is false
    pub sky_color: [f32; 3],
    /// True when SkyPass will render the background this frame
    pub has_sky: bool,
    /// Sky uniforms bind group (group 1 in sky pipeline)
    pub sky_bind_group: Option<&'a wgpu::BindGroup>,
    /// Raw pointer to the active `GpuProfiler`, or null when profiling is
    /// disabled.  Passes may call `scope_begin` / `scope_end` to record
    /// named GPU timing sub-scopes within their `execute()` impl.
    /// SAFETY: valid for the duration of the graph execute loop (single-threaded).
    pub(crate) profiler: *mut GpuProfiler,
    /// Camera world-space position for distance-based culling
    pub camera_position: glam::Vec3,
    /// Camera forward direction for view-depth sorting/culling.
    pub camera_forward: glam::Vec3,
}

impl<'a> PassContext<'a> {
    /// Begin a named GPU timing sub-scope inside this pass.
    ///
    /// Records a `write_timestamp` into the command encoder and returns an
    /// opaque end-slot handle.  Pass the handle to [`scope_end`] once the
    /// GPU commands for this scope have been recorded.
    /// Returns `None` when no profiler is active.
    pub fn scope_begin(&mut self, name: &str) -> Option<u32> {
        if self.profiler.is_null() { return None; }
        // SAFETY: profiler is valid and exclusively accessed on this thread.
        let p = unsafe { &mut *self.profiler };
        let (begin, end) = p.allocate_scope(name)?;
        self.encoder.write_timestamp(p.query_set(), begin);
        Some(end)
    }

    /// End a named GPU timing sub-scope started by [`scope_begin`].
    pub fn scope_end(&mut self, end_slot: Option<u32>) {
        if let Some(end) = end_slot {
            if !self.profiler.is_null() {
                let p = unsafe { &mut *self.profiler };
                self.encoder.write_timestamp(p.query_set(), end);
            }
        }
    }

    /// Begin a render pass
    pub fn begin_render_pass(
        &mut self,
        label: &str,
        color_attachments: &[Option<wgpu::RenderPassColorAttachment>],
        depth_stencil_attachment: Option<wgpu::RenderPassDepthStencilAttachment>,
    ) -> wgpu::RenderPass<'_> {
        self.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments,
            depth_stencil_attachment,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Begin a compute pass
    pub fn begin_compute_pass(&mut self, label: &str) -> wgpu::ComputePass<'_> {
        self.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        })
    }
}
