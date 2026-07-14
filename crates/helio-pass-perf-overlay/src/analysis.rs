use crate::rendering::PerfOverlayShared;
use crate::{ColorCompareParams, ComputeCostParams, PerfOverlayMode};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::{Arc, Mutex};

// ── Analyzer Pass ───────────────────────────────────────────────────────────────

pub struct PerfOverlayAnalyzerPass {
    pub(crate) shared: Arc<Mutex<PerfOverlayShared>>,
}

pub struct PerfOverlayCostAnalyzerPass {
    pub(crate) shared: Arc<Mutex<PerfOverlayShared>>,
}

impl PerfOverlayAnalyzerPass {
    pub fn new(shared: Arc<Mutex<PerfOverlayShared>>) -> Self {
        Self { shared }
    }
}

impl RenderPass for PerfOverlayAnalyzerPass {
    fn name(&self) -> &'static str {
        "PerfOverlay Color Analyzer"
    }

    fn chain_transparent(&self) -> bool {
        true
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
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

        let color_texture = if let Some(pre_aa) = ctx.resources.pre_aa.get() {
            pre_aa
        } else {
            ctx.target
        };

        if shared.runtime.lock().unwrap().frame_num != ctx.frame_num {
            unsafe { &mut *ctx.compute_encoder_ptr }.clear_buffer(
                &shared.pass_overdraw_buf,
                0,
                None,
            );
            let mut runtime = shared.runtime.lock().unwrap();
            runtime.frame_num = ctx.frame_num;
            runtime.snapshot_valid = false;
        }

        let mut runtime = shared.runtime.lock().unwrap();
        if runtime.snapshot_valid {
            let color_compare_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            });

            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Color Compare"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shared.color_compare_pipeline);
            pass.set_bind_group(0, &color_compare_bg, &[]);
            let dispatch_x = shared.internal_width.div_ceil(16);
            let dispatch_y = shared.internal_height.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        } else {
            runtime.snapshot_valid = true;
        }
        drop(runtime);

        let blit_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PerfOverlay Blit BG"),
            layout: &shared.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(color_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&shared.color_snapshot_prev_view),
                },
            ],
        });

        let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PerfOverlay Blit Color"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&shared.blit_pipeline);
        pass.set_bind_group(0, &blit_bg, &[]);
        let dispatch_x = shared.internal_width.div_ceil(16);
        let dispatch_y = shared.internal_height.div_ceil(16);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.shared.lock().unwrap().on_resize(device, width, height);
    }
}

// ── Cost Analyzer Pass ─────────────────────────────────────────────────────────

impl PerfOverlayCostAnalyzerPass {
    pub fn new(shared: Arc<Mutex<PerfOverlayShared>>) -> Self {
        Self { shared }
    }
}

impl RenderPass for PerfOverlayCostAnalyzerPass {
    fn name(&self) -> &'static str {
        "PerfOverlay Cost Analyzer"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let mut shared = self.shared.lock().unwrap();

        if *shared.mode.lock().unwrap() == PerfOverlayMode::ShaderComplexity
            && shared.material_profiler.is_none()
        {
            shared.init_material_profiler(ctx.device, ctx.queue);
        }

        let timing_data_to_upload = if let Some(profiler) = &mut shared.material_profiler {
            if profiler.profiling_complete && !profiler.timings_uploaded {
                profiler.compute_final_timings();

                let mut timing_data: Vec<u8> = Vec::new();
                for entry in &profiler.timing_table {
                    timing_data.extend_from_slice(bytemuck::bytes_of(&entry.roughness));
                    timing_data.extend_from_slice(bytemuck::bytes_of(&entry.metallic));
                    timing_data.extend_from_slice(bytemuck::bytes_of(&entry.num_lights));
                    timing_data.extend_from_slice(bytemuck::bytes_of(&entry.gpu_time_ns));
                }

                profiler.timings_uploaded = true;
                Some(timing_data)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(timing_data) = timing_data_to_upload {
            ctx.queue
                .write_buffer(&shared.material_timing_buf, 0, &timing_data);
        }

        let num_timing_entries = if let Some(profiler) = &shared.material_profiler {
            if profiler.timings_uploaded {
                profiler.timing_table.len() as u32
            } else {
                0
            }
        } else {
            0
        };

        let cost_params = ComputeCostParams {
            screen_width: shared.internal_width,
            screen_height: shared.internal_height,
            num_tiles_x: shared.num_tiles_x,
            num_timing_entries,
        };
        ctx.write_buffer(
            &shared.cost_compute_params_buf,
            0,
            bytemuck::bytes_of(&cost_params),
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let mut shared = self.shared.lock().unwrap();
        if *shared.mode.lock().unwrap() != PerfOverlayMode::ShaderComplexity {
            return Ok(());
        }

        if let Some(profiler) = &mut shared.material_profiler {
            if !profiler.profiling_complete {
                profiler.profile_next(
                    ctx.device,
                    unsafe { &mut *ctx.encoder_ptr },
                    ctx.scene.lights,
                );

                profiler.read_current_sample_blocking(ctx.device, ctx.owns_device);
            }
        }

        if let (Some(gbuffer), Some(tile_light_counts)) = (
            ctx.resources.gbuffer.get(),
            ctx.resources.tile_light_counts.get(),
        ) {
            let cost_compute_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PerfOverlay Cost Compute BG"),
                layout: &shared.cost_compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shared.cost_compute_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(gbuffer.orm),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: tile_light_counts.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: shared.shader_cost_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: shared.material_timing_buf.as_entire_binding(),
                    },
                ],
            });

            if shared.runtime.lock().unwrap().frame_num != ctx.frame_num {
                unsafe { &mut *ctx.encoder_ptr }.clear_buffer(&shared.shader_cost_buf, 0, None);
                shared.runtime.lock().unwrap().frame_num = ctx.frame_num;
            }

            let mut pass = ctx.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PerfOverlay Cost Compute"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&shared.cost_compute_pipeline);
            pass.set_bind_group(0, &cost_compute_bg, &[]);
            let dispatch_x = shared.internal_width.div_ceil(16);
            let dispatch_y = shared.internal_height.div_ceil(16);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        Ok(())
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.shared.lock().unwrap().on_resize(device, width, height);
    }
}
