use crate::rendering::PerfOverlayShared;
use crate::{ColorCompareParams, PerfOverlayMode};
use helio_core::{CachedBindGroup, PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::{Arc, Mutex};

// ── Analyzer Pass ───────────────────────────────────────────────────────────────

pub struct PerfOverlayAnalyzerPass {
    pub(crate) shared: Arc<Mutex<PerfOverlayShared>>,
    color_compare_bg: CachedBindGroup,
    blit_bg: CachedBindGroup,
}

impl PerfOverlayAnalyzerPass {
    pub fn new(shared: Arc<Mutex<PerfOverlayShared>>) -> Self {
        Self {
            shared,
            color_compare_bg: CachedBindGroup::new(),
            blit_bg: CachedBindGroup::new(),
        }
    }
}

impl RenderPass for PerfOverlayAnalyzerPass {
    fn name(&self) -> &'static str {
        "PerfOverlay Color Analyzer"
    }

    /// A member while the pass overdraw overlay is on.
    fn is_member(&mut self, _scene: &mut helio_core::SceneContent<'_>) -> bool {
        self.shared.lock().unwrap().get_mode() == PerfOverlayMode::PassOverdraw
    }

    fn on_leave(&mut self, device: &wgpu::Device) {
        self.shared.lock().unwrap().release_targets(device);
    }

    fn chain_transparent(&self) -> bool {
        true
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.shared.lock().unwrap().ensure_targets(ctx.device);
        let shared = self.shared.lock().unwrap();
        let color_compare_params = ColorCompareParams {
            screen_width: shared.internal_width,
            screen_height: shared.internal_height,
            _pad0: 0,
            _pad1: 0,
        };
        ctx.write_buffer(
            &shared.color_compare_params_buf,
            0,
            bytemuck::bytes_of(&color_compare_params),
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let shared = self.shared.lock().unwrap();
        if *shared.mode.lock().unwrap() != PerfOverlayMode::PassOverdraw {
            return Ok(());
        }

        let color_texture = if let Some(pre_aa) = ctx.registry.get(helio_core::ResourceKey::new(
            helio_frame_api::color::SCENE_COLOR,
        )) {
            pre_aa
        } else {
            ctx.target
        };

        if shared.runtime.lock().unwrap().frame_num != ctx.frame_num {
            ctx.compute_cmds()
                .clear_buffer(&shared.pass_overdraw_buf, 0, None);
            let mut runtime = shared.runtime.lock().unwrap();
            runtime.frame_num = ctx.frame_num;
            runtime.snapshot_valid = false;
        }

        let mut runtime = shared.runtime.lock().unwrap();
        if runtime.snapshot_valid {
            let color_compare_bg = self.color_compare_bg.get_or_create(
                ctx.device,
                &wgpu::BindGroupDescriptor {
                    label: Some("PerfOverlay Color Compare BG"),
                    layout: &shared.color_compare_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: shared.color_compare_params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &shared.color_snapshot_prev_view,
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(color_texture),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: shared.pass_overdraw_buf.as_entire_binding(),
                        },
                    ],
                },
            );

            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Color Compare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shared.color_compare_pipeline);
            pass.set_bind_group(0, color_compare_bg, &[]);
            let dispatch_x = shared.internal_width.div_ceil(16);
            let dispatch_y = shared.internal_height.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        } else {
            runtime.snapshot_valid = true;
        }
        drop(runtime);

        let blit_bg = self.blit_bg.get_or_create(
            ctx.device,
            &wgpu::BindGroupDescriptor {
                label: Some("PerfOverlay Blit BG"),
                layout: &shared.blit_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(color_texture),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(
                            &shared.color_snapshot_prev_view,
                        ),
                    },
                ],
            },
        );

        let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PerfOverlay Blit Color"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&shared.blit_pipeline);
        pass.set_bind_group(0, blit_bg, &[]);
        let dispatch_x = shared.internal_width.div_ceil(16);
        let dispatch_y = shared.internal_height.div_ceil(16);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, size: helio_core::RenderSize) {
        let [width, height] = size.internal;
        self.shared.lock().unwrap().on_resize(device, width, height);
    }
}
