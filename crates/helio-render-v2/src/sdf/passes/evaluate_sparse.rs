//! SDF compute pass: evaluates the edit list on active sparse bricks

use crate::Result;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Compute pass that evaluates the SDF edit list for active bricks only.
/// Dispatches one workgroup per active brick.
pub struct SdfEvaluateSparsePass {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: Arc<wgpu::BindGroup>,
    active_count: Arc<AtomicU32>,
}

impl SdfEvaluateSparsePass {
    pub fn new(
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: Arc<wgpu::BindGroup>,
        active_count: Arc<AtomicU32>,
    ) -> Self {
        Self { pipeline, bind_group, active_count }
    }

    /// Replace the bind group (called when buffers are recreated).
    pub fn set_bind_group(&mut self, bind_group: Arc<wgpu::BindGroup>) {
        self.bind_group = bind_group;
    }
}

impl RenderPass for SdfEvaluateSparsePass {
    fn name(&self) -> &str { "sdf_evaluate_sparse" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.write(ResourceHandle::named("sdf_volume"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let count = self.active_count.load(Ordering::Relaxed);
        if count == 0 { return Ok(()); }

        let mut cpass = ctx.begin_compute_pass("SDF Evaluate Sparse");
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &*self.bind_group, &[]);
        cpass.dispatch_workgroups(count, 1, 1);
        Ok(())
    }
}
