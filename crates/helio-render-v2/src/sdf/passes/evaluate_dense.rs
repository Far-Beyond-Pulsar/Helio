//! SDF compute pass: evaluates the edit list on a dense 3D grid

use crate::Result;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use std::sync::Arc;

/// Compute pass that evaluates the SDF edit list at every voxel
/// and stores distances in a 3D storage texture.
pub struct SdfEvaluateDensePass {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: Arc<wgpu::BindGroup>,
    grid_dim: u32,
}

impl SdfEvaluateDensePass {
    pub fn new(
        pipeline: Arc<wgpu::ComputePipeline>,
        bind_group: Arc<wgpu::BindGroup>,
        grid_dim: u32,
    ) -> Self {
        Self { pipeline, bind_group, grid_dim }
    }

    /// Replace the bind group (called when the edit buffer is recreated)
    pub fn set_bind_group(&mut self, bind_group: Arc<wgpu::BindGroup>) {
        self.bind_group = bind_group;
    }
}

impl RenderPass for SdfEvaluateDensePass {
    fn name(&self) -> &str { "sdf_evaluate" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.write(ResourceHandle::named("sdf_volume"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        // eprintln!("[SDF] evaluate_dense execute() called, grid_dim={}", self.grid_dim);
        let groups = (self.grid_dim + 3) / 4;
        let mut cpass = ctx.begin_compute_pass("SDF Evaluate Dense");
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &*self.bind_group, &[]);
        cpass.dispatch_workgroups(groups, groups, groups);
        Ok(())
    }
}
