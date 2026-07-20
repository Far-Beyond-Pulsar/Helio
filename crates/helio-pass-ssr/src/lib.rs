//! Hybrid screen-space reflections.
//!
//! A single compute dispatch with two execution paths:
//!
//! **Default path** (no RT): Deterministic Hi-Z ray march against the `hiz_min`
//! pyramid. One mirror ray per pixel, stable frame to frame. This is the
//! original SSR behaviour.
//!
//! **RT path** (EXPERIMENTAL_RAY_QUERY available): The Hi-Z march is augmented
//! with hardware ray queries from the TLAS. When the Hi-Z march finds no hit or
//! returns low confidence, a ray query is issued against the scene TLAS. The two
//! results are blended by confidence. Additionally, when radiance cascade data
//! is available (`rc_cascades`), the shader can fall back to RC-based irradiance
//! for rough surfaces where SSR + RT lack enough samples.
//!
//! Writes Rgba16Float at full resolution: RGB = colour, A = hit confidence.

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, RenderPass, Result as HelioResult};

pub struct SsrPass {
    // Default (Hi-Z only) pipeline
    default_pipeline: wgpu::ComputePipeline,
    // RT pipeline (Hi-Z + ray query + RC fallback)
    rt_pipeline: Option<wgpu::ComputePipeline>,

    bgl_1: wgpu::BindGroupLayout,
    // Optional BGL2 for RT resources (TLAS + RC texture)
    bgl_2: Option<wgpu::BindGroupLayout>,

    bg_0: wgpu::BindGroup,
    bg_1: Option<wgpu::BindGroup>,
    bg_1_key: Option<(usize, usize, usize, usize, usize, usize)>,
    bg_2: Option<wgpu::BindGroup>,
    bg_2_key: Option<(usize, usize)>,

    linear_sampler: wgpu::Sampler,
    use_rt: bool,

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

        let use_rt = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);

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

        // Optional BGL2: TLAS + RC cascade texture (for RT path)
        let bgl_2 = use_rt.then(|| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SSR BGL2 (RT)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                    // RC cascade texture for reflection fallback (optional).
                    // Bound as unfiltered float to match Rgba16Float storage format.
                    texture_unfiltered_entry(1),
                ],
            })
        });

        // ── Default pipeline (Hi-Z only) ────────────────────────────────
        let default_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1)],
            immediate_size: 0,
        });

        let default_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSR Trace Pipeline"),
            layout: Some(&default_pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── RT pipeline (Hi-Z + ray query) ──────────────────────────────
        let rt_pipeline = use_rt.then(|| {
            // The RT shader is self-contained (declares Camera + helpers inline)
            // so `enable wgpu_ray_query;` can appear at line 1 as WGSL requires.
            let rt_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("SSR RT Trace Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    std::borrow::Cow::Owned(include_str!("../shaders/ssr_trace_rt.wgsl").to_string()),
                ),
            });

            let rt_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("SSR RT PL"),
                bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1), Some(bgl_2.as_ref().unwrap())],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SSR Hybrid Trace Pipeline"),
                layout: Some(&rt_pl),
                module: &rt_shader,
                entry_point: Some("cs_rt"),
                compilation_options: Default::default(),
                cache: None,
            })
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
            default_pipeline,
            rt_pipeline,
            bgl_1,
            bgl_2,
            bg_0,
            bg_1: None,
            bg_1_key: None,
            bg_2: None,
            bg_2_key: None,
            linear_sampler,
            use_rt,
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
        builder.with_extra_usage(
            wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        );
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
        self.bg_2 = None;
        self.bg_2_key = None;
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
        let hiz_min_view = match ctx.resource_pool.get_view("hiz_min") {
            Some(v) => v,
            None => return Ok(()),
        };
        let ssr_trace = match ctx.resource_pool.get_view("ssr_trace") {
            Some(v) => v,
            None => return Ok(()),
        };

        // ── BG1: always bound ───────────────────────────────────────────
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

        // ── Decide between default and RT path ──────────────────────────
        if self.use_rt {
            let main_scene = ctx.resources.main_scene.read("SsrPass");
            let tlas = main_scene.and_then(|ms| ms.tlas);

            if let Some(tlas_binding) = tlas {
                let rc_view = ctx.resources.rc_view.get();

                let rt_key = (
                    tlas_binding as *const _ as usize,
                    rc_view.map_or(0, |v| v as *const _ as usize),
                );

                if self.bg_2_key != Some(rt_key) {
                    // RC cascade texture (or scene_color as a dummy fallback — the
                    // shader checks texture dimensions before sampling RC data).
                    let rc_tex = rc_view.unwrap_or(pre_aa_view);

                    self.bg_2 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("SSR BG2 (RT)"),
                        layout: self.bgl_2.as_ref().unwrap(),
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: tlas_binding.as_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(rc_tex),
                            },
                        ],
                    }));
                    self.bg_2_key = Some(rt_key);
                }

                // RT path
                let cpass = unsafe { &mut *ctx.compute_encoder_ptr };
                let mut pass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SSR Hybrid Trace"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(self.rt_pipeline.as_ref().unwrap());
                pass.set_bind_group(0, &self.bg_0, &[]);
                pass.set_bind_group(1, self.bg_1.as_ref().unwrap(), &[]);
                pass.set_bind_group(2, self.bg_2.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);

                return Ok(());
            }
        }

        // ── Default: Hi-Z only ──────────────────────────────────────────
        let cpass = unsafe { &mut *ctx.compute_encoder_ptr };
        let mut pass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SSR Trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.default_pipeline);
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
