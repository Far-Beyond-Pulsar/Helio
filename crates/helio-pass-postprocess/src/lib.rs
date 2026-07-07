//! GPU-native post-processing pipeline for Helio.
//!
//! Sub-stages:
//!   1. `cs_exposure`           — luminance histogram → avg luminance (compute)
//!   2. `cs_bloom_down_extract` — extract brights from HDR → bloom mip 0 (compute)
//!   3. `cs_bloom_down`         — 2x downsample mip chain, 4 passes (compute)
//!   4. `fs_uber`               — tonemap, color grade, vignette, CA, grain (render)
//!
//! Three bind group layouts are used to avoid storage-write / sampled-read conflicts
//! on the same bloom texture within the same dispatch scope:
//!
//!   compute_main_bgl  (@group 0, compute only)  — uniforms, samplers, hdr, depth, avg_lum
//!                                                  NO bloom sampled textures
//!   render_main_bgl   (@group 0, render only)   — uniforms, samplers, hdr, depth,
//!                                                  bloom_0..4 sampled, avg_lum
//!   bloom_compute_bgl (@group 1, compute only)  — per-dispatch bloom src + dst

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const BLOOM_MIPS: u32 = 5;
const WG_BLOOM: u32 = 8;
const WG_EXPOSURE_X: u32 = 16;
const WG_EXPOSURE_Y: u32 = 16;

pub struct PostProcessPass {
    avg_luminance_buf: wgpu::Buffer,

    exposure_pipeline: wgpu::ComputePipeline,
    bloom_extract_pipeline: wgpu::ComputePipeline,
    bloom_down_pipeline: wgpu::ComputePipeline,
    uber_pipeline: wgpu::RenderPipeline,

    // Separate BGLs for compute vs render to prevent bloom texture usage conflicts
    compute_main_bgl: wgpu::BindGroupLayout,
    render_main_bgl: wgpu::BindGroupLayout,
    bloom_compute_bgl: wgpu::BindGroupLayout,

    // Compute main bind group (no bloom sampled — rebuilt when hdr/depth/cam/uniforms change)
    compute_main_bg: Option<wgpu::BindGroup>,
    // Render main bind group (includes bloom sampled — rebuilt on same key)
    render_main_bg: Option<wgpu::BindGroup>,
    main_bg_key: Option<(usize, usize, usize, usize)>,

    // Bloom extract BG (group 1): b0=mip1 sampled (dummy), b1=mip0 write storage.
    bloom_extract_bg: Option<(usize, wgpu::BindGroup)>,
    // Bloom downsample BGs (group 1) for mips 1-4: src=mip_N-1 sampled, dst=mip_N write.
    bloom_down_bgs: Vec<wgpu::BindGroup>,

    bloom_textures: Vec<wgpu::Texture>,
    bloom_sampled_views: Vec<wgpu::TextureView>,
    bloom_storage_views: Vec<wgpu::TextureView>,

    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,

    width: u32,
    height: u32,
    #[allow(dead_code)]
    format: wgpu::TextureFormat,

    first_frame: bool,
}

impl PostProcessPass {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/postprocess.wgsl").into()),
        });

        let avg_luminance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Avg Luminance"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Point Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let (bloom_textures, bloom_sampled_views, bloom_storage_views) =
            Self::create_bloom_mips(device, width, height);

        // ── Shared BGL entries (b0-b5, reused in both compute and render layouts) ─

        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sampled_tex_entry = |binding: u32, vis: wgpu::ShaderStages, depth: bool| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Texture {
                    sample_type: if depth {
                        wgpu::TextureSampleType::Depth
                    } else {
                        wgpu::TextureSampleType::Float { filterable: true }
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }
        };
        let sampler_entry = |binding: u32, vis: wgpu::ShaderStages, filtering: bool| {
            wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Sampler(if filtering {
                    wgpu::SamplerBindingType::Filtering
                } else {
                    wgpu::SamplerBindingType::NonFiltering
                }),
                count: None,
            }
        };
        let storage_buf_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let cv = wgpu::ShaderStages::COMPUTE;
        let fv = wgpu::ShaderStages::FRAGMENT;
        let cfv = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT;

        // ── compute_main_bgl: b0-b5 + b11, NO bloom sampled ──────────────────────
        // Used by all compute pipelines. Bloom mips are absent so they can be
        // simultaneously bound as STORAGE_WRITE in group 1 without conflicts.
        let compute_main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Compute Main BGL"),
            entries: &[
                uniform_entry(0, cfv),           // GpuPostProcessUniforms
                uniform_entry(1, cfv),           // CameraUniforms
                sampled_tex_entry(2, cfv, false), // hdr_input
                sampled_tex_entry(3, fv, true),  // depth_input (unused in compute, but keep for layout compat)
                sampler_entry(4, cfv, true),     // linear_samp
                sampler_entry(5, fv, false),     // point_samp
                storage_buf_entry(11, cfv),      // avg_luminance
            ],
        });

        // ── render_main_bgl: b0-b11, WITH bloom sampled ──────────────────────────
        // Used only by the render pipeline (fs_uber). No storage write textures in
        // this scope so no conflicts with the bloom sampled views at b6-b10.
        let render_main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Render Main BGL"),
            entries: &[
                uniform_entry(0, fv),
                uniform_entry(1, fv),
                sampled_tex_entry(2, fv, false),
                sampled_tex_entry(3, fv, true),
                sampler_entry(4, fv, true),
                sampler_entry(5, fv, false),
                sampled_tex_entry(6, fv, false),  // bloom_0
                sampled_tex_entry(7, fv, false),  // bloom_1
                sampled_tex_entry(8, fv, false),  // bloom_2
                sampled_tex_entry(9, fv, false),  // bloom_3
                sampled_tex_entry(10, fv, false), // bloom_4
                storage_buf_entry(11, fv),
            ],
        });

        // ── bloom_compute_bgl: per-dispatch src (sampled) + dst (write storage) ──
        let bloom_compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Bloom Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: cv,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: cv,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Precompute bloom_down BGs for mips 1-4 (src=mip_N-1 sampled, dst=mip_N write).
        let bloom_down_bgs = Self::make_bloom_down_bgs(device, &bloom_compute_bgl, &bloom_sampled_views, &bloom_storage_views);

        // ── Pipeline layouts ──────────────────────────────────────────────────────
        let exposure_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Exposure PL"),
            bind_group_layouts: &[Some(&compute_main_bgl)],
            immediate_size: 0,
        });
        let bloom_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Bloom PL"),
            bind_group_layouts: &[Some(&compute_main_bgl), Some(&bloom_compute_bgl)],
            immediate_size: 0,
        });
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Render PL"),
            bind_group_layouts: &[Some(&render_main_bgl)],
            immediate_size: 0,
        });

        let mk_compute = |label: &str, entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let exposure_pipeline = mk_compute("PostProcess Exposure", "cs_exposure", &exposure_pl);
        let bloom_extract_pipeline =
            mk_compute("PostProcess Bloom Extract", "cs_bloom_down_extract", &bloom_pl);
        let bloom_down_pipeline =
            mk_compute("PostProcess Bloom Down", "cs_bloom_down", &bloom_pl);

        let uber_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PostProcess Uber Pipeline"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_uber"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            avg_luminance_buf,
            exposure_pipeline,
            bloom_extract_pipeline,
            bloom_down_pipeline,
            uber_pipeline,
            compute_main_bgl,
            render_main_bgl,
            bloom_compute_bgl,
            compute_main_bg: None,
            render_main_bg: None,
            main_bg_key: None,
            bloom_extract_bg: None,
            bloom_down_bgs,
            bloom_textures,
            bloom_sampled_views,
            bloom_storage_views,
            linear_sampler,
            point_sampler,
            width,
            height,
            format,
            first_frame: true,
        }
    }

    fn create_bloom_mips(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>, Vec<wgpu::TextureView>) {
        let mut textures = Vec::with_capacity(BLOOM_MIPS as usize);
        let mut sampled_views = Vec::with_capacity(BLOOM_MIPS as usize);
        let mut storage_views = Vec::with_capacity(BLOOM_MIPS as usize);

        for i in 0..BLOOM_MIPS {
            let mw = (width >> (i + 1)).max(1);
            let mh = (height >> (i + 1)).max(1);
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Mip {}", i)),
                size: wgpu::Extent3d { width: mw, height: mh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            sampled_views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Bloom Mip {} Sampled", i)),
                ..Default::default()
            }));
            storage_views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Bloom Mip {} Storage", i)),
                ..Default::default()
            }));
            textures.push(tex);
        }

        (textures, sampled_views, storage_views)
    }

    fn make_bloom_down_bgs(
        device: &wgpu::Device,
        bloom_compute_bgl: &wgpu::BindGroupLayout,
        bloom_sampled_views: &[wgpu::TextureView],
        bloom_storage_views: &[wgpu::TextureView],
    ) -> Vec<wgpu::BindGroup> {
        (1..BLOOM_MIPS as usize)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("PostProcess Bloom Down BG mip{}", i)),
                    layout: bloom_compute_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&bloom_sampled_views[i - 1]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&bloom_storage_views[i]),
                        },
                    ],
                })
            })
            .collect()
    }

    fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        postprocess_buf: &wgpu::Buffer,
        pre_aa_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
    ) {
        // Compute main BG — no bloom sampled textures (prevents storage-write conflict)
        self.compute_main_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Compute Main BG"),
            layout: &self.compute_main_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: postprocess_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(pre_aa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.avg_luminance_buf.as_entire_binding(),
                },
            ],
        }));

        // Render main BG — includes all 5 bloom sampled views for fs_uber
        self.render_main_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Render Main BG"),
            layout: &self.render_main_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: postprocess_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(pre_aa_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[0]),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[1]),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[2]),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[3]),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[4]),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: self.avg_luminance_buf.as_entire_binding(),
                },
            ],
        }));
    }

    fn mip_dims(&self, mip: u32) -> (u32, u32) {
        ((self.width >> (mip + 1)).max(1), (self.height >> (mip + 1)).max(1))
    }
}

impl RenderPass for PostProcessPass {
    fn name(&self) -> &'static str { "PostProcess" }

    fn reads(&self) -> &'static [&'static str] { &["pre_aa"] }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let (textures, sampled_views, storage_views) =
            Self::create_bloom_mips(device, width, height);
        self.bloom_textures = textures;
        self.bloom_sampled_views = sampled_views;
        self.bloom_storage_views = storage_views;
        self.bloom_down_bgs = Self::make_bloom_down_bgs(
            device,
            &self.bloom_compute_bgl,
            &self.bloom_sampled_views,
            &self.bloom_storage_views,
        );
        self.compute_main_bg = None;
        self.render_main_bg = None;
        self.main_bg_key = None;
        self.bloom_extract_bg = None;
        self.first_frame = true;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if self.first_frame {
            self.first_frame = false;
            let initial: f32 = 0.18;
            ctx.queue.write_buffer(&self.avg_luminance_buf, 0, bytemuck::bytes_of(&initial));
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let pre_aa_view = ctx.resources.pre_aa.read("PostProcess").ok_or_else(|| {
            helio_core::Error::InvalidPassConfig(
                "PostProcess requires frame.pre_aa (from DeferredLightPass)".to_string(),
            )
        })?;

        let postprocess_buf =
            ctx.resources.postprocess_uniforms.read("PostProcess").ok_or_else(|| {
                helio_core::Error::InvalidPassConfig(
                    "PostProcess requires frame.postprocess_uniforms (from Renderer)".to_string(),
                )
            })?;

        let camera_buf = ctx.scene.camera;

        let bg_key = (
            pre_aa_view as *const _ as usize,
            ctx.depth as *const _ as usize,
            camera_buf as *const _ as usize,
            postprocess_buf as *const _ as usize,
        );
        if self.main_bg_key != Some(bg_key) {
            self.rebuild_bind_groups(ctx.device, postprocess_buf, pre_aa_view, ctx.depth, camera_buf);
            self.main_bg_key = Some(bg_key);
        }

        // Bloom extract BG: dst=mip0 write storage (group1 b1).
        // b0 (bloom_src) is declared unused by cs_bloom_down_extract, but wgpu still
        // validates that no two bindings in scope reference the same texture with
        // conflicting usages. Use mip1's sampled view as the dummy — a different
        // texture than mip0 — so RESOURCE and STORAGE_WRITE_ONLY never collide.
        let hdr_ptr = pre_aa_view as *const _ as usize;
        if self.bloom_extract_bg.as_ref().map(|(k, _)| *k) != Some(hdr_ptr) {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PostProcess Bloom Extract BG"),
                layout: &self.bloom_compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        // Dummy — different texture from mip0 to avoid usage conflict.
                        resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[1]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.bloom_storage_views[0]),
                    },
                ],
            });
            self.bloom_extract_bg = Some((hdr_ptr, bg));
        }

        let compute_bg = self.compute_main_bg.as_ref().unwrap();
        let render_bg = self.render_main_bg.as_ref().unwrap();
        let extract_bg = &self.bloom_extract_bg.as_ref().unwrap().1;
        let ce = ctx.compute_encoder_ptr;

        // 1. Auto-exposure histogram
        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Exposure"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.exposure_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
            let gx = (self.width / (4 * WG_EXPOSURE_X)).max(1);
            let gy = (self.height / (4 * WG_EXPOSURE_Y)).max(1);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // 2a. Bloom extract: HDR → mip 0
        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Bloom Extract"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.bloom_extract_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
            cpass.set_bind_group(1, extract_bg, &[]);
            let (mw, mh) = self.mip_dims(0);
            cpass.dispatch_workgroups(
                (mw + WG_BLOOM - 1) / WG_BLOOM,
                (mh + WG_BLOOM - 1) / WG_BLOOM,
                1,
            );
        }

        // 2b. Bloom downsample: mip N-1 → mip N, for N in 1..BLOOM_MIPS.
        // Separate passes ensure writes complete before the next mip reads them.
        for i in 0..(BLOOM_MIPS as usize - 1) {
            let (mw, mh) = self.mip_dims(i as u32 + 1);
            let mut cpass =
                unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("PostProcess Bloom Down mip{}", i + 1)),
                    timestamp_writes: None,
                });
            cpass.set_pipeline(&self.bloom_down_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
            cpass.set_bind_group(1, &self.bloom_down_bgs[i], &[]);
            cpass.dispatch_workgroups(
                (mw + WG_BLOOM - 1) / WG_BLOOM,
                (mh + WG_BLOOM - 1) / WG_BLOOM,
                1,
            );
        }

        // 3. Uber render pass — uses render_bg which includes bloom sampled views.
        //    All compute passes are complete; no storage writes are in scope here.
        {
            let color = [Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass =
                unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("PostProcess Uber"),
                    color_attachments: &color,
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            pass.set_pipeline(&self.uber_pipeline);
            pass.set_bind_group(0, render_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
