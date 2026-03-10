//! SDF clip map update compute pass
//!
//! Reuses the Phase 3 sparse evaluate compute pipeline. Dispatches it
//! once per level that has active bricks, switching bind groups between dispatches.

use crate::Result;
use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use std::sync::{Arc, Mutex};

/// Compute pass that evaluates SDF for all active bricks across clip map levels.
pub struct SdfClipUpdatePass {
    pipeline: Arc<wgpu::ComputePipeline>,
    level_bind_groups: Vec<Arc<wgpu::BindGroup>>,
    level_active_counts: Arc<Mutex<Vec<u32>>>,
}

impl SdfClipUpdatePass {
    pub fn new(
        pipeline: Arc<wgpu::ComputePipeline>,
        level_bind_groups: Vec<Arc<wgpu::BindGroup>>,
        level_active_counts: Arc<Mutex<Vec<u32>>>,
    ) -> Self {
        Self {
            pipeline,
            level_bind_groups,
            level_active_counts,
        }
    }
}

impl RenderPass for SdfClipUpdatePass {
    fn name(&self) -> &str { "sdf_clip_update" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.write(ResourceHandle::named("sdf_volume"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let counts = self.level_active_counts.lock().unwrap();

        let mut any_dispatched = false;
        for i in 0..self.level_bind_groups.len() {
            let count = counts.get(i).copied().unwrap_or(0);
            if count > 0 {
                any_dispatched = true;
                break;
            }
        }
        if !any_dispatched { return Ok(()); }

        let mut cpass = ctx.begin_compute_pass("SDF Clip Update");
        cpass.set_pipeline(&self.pipeline);

        for (i, bg) in self.level_bind_groups.iter().enumerate() {
            let count = counts.get(i).copied().unwrap_or(0);
            if count > 0 {
                cpass.set_bind_group(0, &**bg, &[]);
                cpass.dispatch_workgroups(count, 1, 1);
            }
        }

        Ok(())
    }
}
