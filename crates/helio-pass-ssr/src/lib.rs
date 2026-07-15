use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsrTemporalUniforms {
    frame_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct SsrPass {
    trace_pipeline: wgpu::ComputePipeline,
    trace_bgl_0: wgpu::BindGroupLayout,
    trace_bgl_1: wgpu::BindGroupLayout,
    trace_bg_0: wgpu::BindGroup,
    trace_bg_1: Option<wgpu::BindGroup>,
    trace_bg_1_key: Option<(usize, usize, usize, usize, usize)>,

    denoise_pipeline: wgpu::ComputePipeline,
    denoise_bgl_1: wgpu::BindGroupLayout,
    denoise_bg_0: wgpu::BindGroup,
    denoise_bg_1: Option<wgpu::BindGroup>,
    denoise_bg_1_key: Option<(usize, usize, usize, usize, usize)>,

    temporal_pipeline: wgpu::ComputePipeline,
    temporal_bgl_0: wgpu::BindGroupLayout,
    temporal_bgl_1: wgpu::BindGroupLayout,
    temporal_bg_0: wgpu::BindGroup,
    temporal_bg_1: Option<wgpu::BindGroup>,
    temporal_bg_1_key: Option<(usize, usize, usize, usize)>,
    temporal_uniforms: wgpu::Buffer,

    linear_sampler: wgpu::Sampler,

    half_width: u32,
    half_height: u32,
}

impl SsrPass {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        camera_buf: &wgpu::Buffer,
        full_width: u32,
        full_height: u32,
    ) -> Self {
        let half_width = (full_width / 2).max(1);
        let half_height = (full_height / 2).max(1);

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SSR Linear Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        // ── Trace shader ────────────────────────────────────────────────────────
        let trace_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSR Trace Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/ssr_trace.wgsl").into(),
            ),
        });

        let trace_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Trace BGL0"),
            entries: &[buffer_uniform_entry(0)],
        });

        // No Hi-Z binding: the shared `hiz` pyramid is a max-reduction built for
        // occlusion culling. SSR needs min-depth, so the trace marches full-res
        // depth directly. See the header comment in ssr_trace.wgsl.
        let trace_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Trace BGL1"),
            entries: &[
                texture_float_entry(0),
                texture_float_entry(1),
                texture_depth_entry(2),
                texture_float_entry(3),
                sampler_entry(4),
                storage_entry(5),
            ],
        });

        let trace_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Trace PL"),
            bind_group_layouts: &[Some(&trace_bgl_0), Some(&trace_bgl_1)],
            immediate_size: 0,
        });

        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSR Trace Pipeline"),
            layout: Some(&trace_pl),
            module: &trace_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let trace_bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Trace BG0"),
            layout: &trace_bgl_0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        // ── Denoise shader ──────────────────────────────────────────────────────
        let denoise_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSR Denoise Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/ssr_denoise.wgsl").into(),
            ),
        });

        let denoise_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Denoise BGL0"),
            entries: &[buffer_uniform_entry(0)],
        });

        let denoise_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Denoise BGL1"),
            entries: &[
                texture_float_entry(0),
                texture_depth_entry(1),
                texture_float_entry(2),
                texture_float_entry(3),
                storage_entry(4),
            ],
        });

        let denoise_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Denoise PL"),
            bind_group_layouts: &[Some(&denoise_bgl_0), Some(&denoise_bgl_1)],
            immediate_size: 0,
        });

        let denoise_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSR Denoise Pipeline"),
            layout: Some(&denoise_pl),
            module: &denoise_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let denoise_bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Denoise BG0"),
            layout: &denoise_bgl_0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        // ── Temporal shader ─────────────────────────────────────────────────────
        let temporal_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SSR Temporal Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/ssr_temporal.wgsl").into(),
            ),
        });

        let temporal_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSR Temporal Uniforms"),
            size: std::mem::size_of::<SsrTemporalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let temporal_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Temporal BGL0"),
            entries: &[
                buffer_uniform_entry(0),
                buffer_uniform_entry(1),
            ],
        });

        let temporal_bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR Temporal BGL1"),
            entries: &[
                texture_depth_entry(0),
                texture_float_entry(1),
                texture_float_entry(2),
                sampler_entry(3),
                storage_entry(4),
            ],
        });

        let temporal_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR Temporal PL"),
            bind_group_layouts: &[Some(&temporal_bgl_0), Some(&temporal_bgl_1)],
            immediate_size: 0,
        });

        let temporal_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSR Temporal Pipeline"),
            layout: Some(&temporal_pl),
            module: &temporal_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let temporal_bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR Temporal BG0"),
            layout: &temporal_bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: temporal_uniforms.as_entire_binding(),
                },
            ],
        });

        Self {
            trace_pipeline,
            trace_bgl_0,
            trace_bgl_1,
            trace_bg_0,
            trace_bg_1: None,
            trace_bg_1_key: None,
            denoise_pipeline,
            denoise_bgl_1,
            denoise_bg_0,
            denoise_bg_1: None,
            denoise_bg_1_key: None,
            temporal_pipeline,
            temporal_bgl_0,
            temporal_bgl_1,
            temporal_bg_0,
            temporal_bg_1: None,
            temporal_bg_1_key: None,
            temporal_uniforms,
            linear_sampler,
            half_width,
            half_height,
        }
    }
}

impl RenderPass for SsrPass {
    fn name(&self) -> &'static str {
        "SsrPass"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "depth", "pre_aa"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["ssr_trace", "ssr_denoised", "ssr_accum", "ssr_history"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw("ssr_trace", wgpu::TextureFormat::Rgba16Float, ResourceSize::Scaled { divisor: 2 });
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
        builder.write_color_raw("ssr_denoised", wgpu::TextureFormat::Rgba16Float, ResourceSize::Scaled { divisor: 2 });
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
        builder.write_color_raw("ssr_accum", wgpu::TextureFormat::Rgba16Float, ResourceSize::Scaled { divisor: 2 });
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        builder.write_color_raw("ssr_history", wgpu::TextureFormat::Rgba16Float, ResourceSize::Scaled { divisor: 2 });
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST);
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn on_resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        self.half_width = (width / 2).max(1);
        self.half_height = (height / 2).max(1);
        self.trace_bg_1 = None;
        self.temporal_bg_1 = None;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let uniforms = SsrTemporalUniforms {
            frame_index: ctx.frame_num as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.temporal_uniforms, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let gbuffer = match ctx.resources.gbuffer.read("SsrPass") {
            Some(g) => g,
            None => return Ok(()),
        };

        let depth_view = ctx.depth;
        let pre_aa_view = match ctx.resources.pre_aa.get() {
            Some(v) => v,
            None => return Ok(()),
        };

        let ssr_trace = match ctx.resource_pool.get_view("ssr_trace") {
            Some(v) => v,
            None => return Ok(()),
        };
        let ssr_denoised = match ctx.resource_pool.get_view("ssr_denoised") {
            Some(v) => v,
            None => return Ok(()),
        };
        let ssr_accum = match ctx.resource_pool.get_view("ssr_accum") {
            Some(v) => v,
            None => return Ok(()),
        };
        let ssr_history = match ctx.resource_pool.get_view("ssr_history") {
            Some(v) => v,
            None => return Ok(()),
        };

        let dispatch_x = (self.half_width + 7) / 8;
        let dispatch_y = (self.half_height + 7) / 8;

        // ── Phase 1: SSR Trace ───────────────────────────────────────────────
        let trace_key = (
            gbuffer.normal as *const _ as usize,
            gbuffer.orm as *const _ as usize,
            depth_view as *const _ as usize,
            pre_aa_view as *const _ as usize,
            ssr_trace as *const _ as usize,
        );

        if self.trace_bg_1_key != Some(trace_key) {
            self.trace_bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSR Trace BG1"),
                layout: &self.trace_bgl_1,
                entries: &[
                    texture_view_entry(0, gbuffer.normal),
                    texture_view_entry(1, gbuffer.orm),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    texture_view_entry(3, pre_aa_view),
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(ssr_trace),
                    },
                ],
            }));
            self.trace_bg_1_key = Some(trace_key);
        }

        {
            let cpass = unsafe { &mut *ctx.compute_encoder_ptr };
            let mut pass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SSR Trace"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace_pipeline);
            pass.set_bind_group(0, &self.trace_bg_0, &[]);
            pass.set_bind_group(1, self.trace_bg_1.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // ── Phase 2: SSR Denoise (roughness-weighted bilateral blur) ─────────
        let denoise_key = (
            gbuffer.normal as *const _ as usize,
            depth_view as *const _ as usize,
            gbuffer.orm as *const _ as usize,
            ssr_trace as *const _ as usize,
            ssr_denoised as *const _ as usize,
        );

        if self.denoise_bg_1_key != Some(denoise_key) {
            self.denoise_bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSR Denoise BG1"),
                layout: &self.denoise_bgl_1,
                entries: &[
                    texture_view_entry(0, gbuffer.normal),
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    texture_view_entry(2, gbuffer.orm),
                    texture_view_entry(3, ssr_trace),
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(ssr_denoised),
                    },
                ],
            }));
            self.denoise_bg_1_key = Some(denoise_key);
        }

        {
            let cpass = unsafe { &mut *ctx.compute_encoder_ptr };
            let mut pass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SSR Denoise"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.denoise_pipeline);
            pass.set_bind_group(0, &self.denoise_bg_0, &[]);
            pass.set_bind_group(1, self.denoise_bg_1.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // ── Phase 3: SSR Temporal ────────────────────────────────────────────
        let temporal_key = (
            depth_view as *const _ as usize,
            ssr_denoised as *const _ as usize,
            ssr_history as *const _ as usize,
            ssr_accum as *const _ as usize,
        );

        if self.temporal_bg_1_key != Some(temporal_key) {
            self.temporal_bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSR Temporal BG1"),
                layout: &self.temporal_bgl_1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    texture_view_entry(1, ssr_denoised),
                    texture_view_entry(2, ssr_history),
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(ssr_accum),
                    },
                ],
            }));
            self.temporal_bg_1_key = Some(temporal_key);
        }

        {
            let cpass = unsafe { &mut *ctx.compute_encoder_ptr };
            let mut pass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SSR Temporal"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.temporal_pipeline);
            pass.set_bind_group(0, &self.temporal_bg_0, &[]);
            pass.set_bind_group(1, self.temporal_bg_1.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }

        // ── Phase 4: Copy ssr_accum → ssr_history for next frame ────────────
        let ssr_accum_tex = ctx.resource_pool.get_texture("ssr_accum").unwrap();
        let ssr_history_tex = ctx.resource_pool.get_texture("ssr_history").unwrap();
        let encoder = unsafe { &mut *ctx.encoder_ptr };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                texture: ssr_accum_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyTextureInfo {
                texture: ssr_history_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: self.half_width,
                height: self.half_height,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}
}

fn buffer_uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_float_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_depth_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn texture_view_entry<'a>(binding: u32, view: &'a wgpu::TextureView) -> wgpu::BindGroupEntry<'a> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}
