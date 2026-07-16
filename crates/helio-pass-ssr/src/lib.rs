//! Screen-space reflections.
//!
//! A single compute dispatch: one deterministic mirror ray per pixel, marched
//! against the `hiz_min` pyramid built by `HiZBuildPass`, writing colour +
//! confidence into `ssr_trace` for `DeferredLightPass` to composite over its
//! cubemap fallback.
//!
//! There is deliberately no temporal pass and no denoiser. Those belong to
//! *stochastic* SSR, where rough reflections are importance-sampled and the
//! Monte Carlo noise has to be resolved over time. This traces one exact
//! `reflect(-V, N)` per pixel, so its output is already stable frame to frame —
//! there is no noise to denoise and nothing to accumulate. Reference
//! implementations of non-stochastic SSR are likewise single-pass.

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, RenderPass, Result as HelioResult};

pub struct SsrPass {
    pipeline: wgpu::ComputePipeline,
    bgl_1: wgpu::BindGroupLayout,
    bg_0: wgpu::BindGroup,
    bg_1: Option<wgpu::BindGroup>,
    bg_1_key: Option<(usize, usize, usize, usize, usize, usize)>,

    linear_sampler: wgpu::Sampler,

    width: u32,
    height: u32,
}

impl SsrPass {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        camera_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Self {
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

        let shader = helio_core::shader::module(
            device,
            "SSR Trace Shader",
            include_str!("../shaders/ssr_trace.wgsl"),
        );

        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR BGL0"),
            entries: &[buffer_uniform_entry(0)],
        });

        // Binds `hiz_min`, not `hiz`: the shared pyramid is max-reduced for
        // occlusion culling, and a ray march needs min-depth. Both are built by
        // HiZBuildPass. See the header comment in ssr_trace.wgsl.
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR BGL1"),
            entries: &[
                texture_float_entry(0),
                texture_float_entry(1),
                texture_depth_entry(2),
                texture_float_entry(3),
                texture_unfiltered_entry(4), // hiz_min (R32Float, not filterable)
                sampler_entry(5),
                storage_entry(6),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSR Trace Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR BG0"),
            layout: &bgl_0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        Self {
            pipeline,
            bgl_1,
            bg_0,
            bg_1: None,
            bg_1_key: None,
            linear_sampler,
            width,
            height,
        }
    }
}

impl RenderPass for SsrPass {
    fn name(&self) -> &'static str {
        "SsrPass"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "depth", "hiz_min", "pre_aa"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["ssr_trace"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw(
            "ssr_trace",
            wgpu::TextureFormat::Rgba16Float,
            ResourceSize::MatchSurface,
        );
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
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
        self.width = width;
        self.height = height;
        self.bg_1 = None;
        self.bg_1_key = None;
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
        // Built by HiZBuildPass. A default (all-mip) view: the trace selects the
        // level itself via textureLoad, so it needs the whole chain, not one mip.
        let hiz_min_view = match ctx.resource_pool.get_view("hiz_min") {
            Some(v) => v,
            None => return Ok(()),
        };
        let ssr_trace = match ctx.resource_pool.get_view("ssr_trace") {
            Some(v) => v,
            None => return Ok(()),
        };

        let key = (
            gbuffer.normal as *const _ as usize,
            gbuffer.orm as *const _ as usize,
            depth_view as *const _ as usize,
            pre_aa_view as *const _ as usize,
            hiz_min_view as *const _ as usize,
            ssr_trace as *const _ as usize,
        );

        if self.bg_1_key != Some(key) {
            self.bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSR BG1"),
                layout: &self.bgl_1,
                entries: &[
                    texture_view_entry(0, gbuffer.normal),
                    texture_view_entry(1, gbuffer.orm),
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    texture_view_entry(3, pre_aa_view),
                    texture_view_entry(4, hiz_min_view),
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(ssr_trace),
                    },
                ],
            }));
            self.bg_1_key = Some(key);
        }

        let cpass = unsafe { &mut *ctx.compute_encoder_ptr };
        let mut pass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SSR Trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg_0, &[]);
        pass.set_bind_group(1, self.bg_1.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);

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

fn texture_unfiltered_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
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
