//! Decal apply compute pass.
//!
//! Runs after the GBuffer pass, before deferred lighting. Dispatches one
//! thread group per 16×16 tile. Each thread reads depth, reconstructs world
//! position, transforms into each active decal's local space, and blends
//! decal contributions into the GBuffer via storage texture writes.
//!
//! # Resources
//!
//! - reads:  `"gbuffer"`, `"depth"`
//! - writes: `"gbuffer"` (modified in-place)

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Per-frame globals.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DecalGlobals {
    decal_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct DecalPass {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bgl_0: wgpu::BindGroupLayout,
    bgl_1: wgpu::BindGroupLayout,
    bgl_2: wgpu::BindGroupLayout,
    bg_0: Option<wgpu::BindGroup>,
    bg_1: Option<wgpu::BindGroup>,
    bg_1_key: Option<(usize, usize, usize, usize, usize)>,
    bg_2: Option<wgpu::BindGroup>,
    globals_buf: wgpu::Buffer,
}

impl DecalPass {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        decal_buf: &wgpu::Buffer,
        camera_buf: &wgpu::Buffer,
        _w: u32,
        _h: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Decal Apply"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/decal_apply.wgsl").into(),
            ),
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DecalGlobals"),
            size: std::mem::size_of::<DecalGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── BGL 0: camera + globals + decal buffer ────────────────────────────
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Decal BGL 0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── BGL 1: depth + GBuffer read + GBuffer storage write ─────────────
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Decal BGL 1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // storage texture writes (binding 5-8)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // ── BGL 2: bindless decal textures + samplers ───────────────────────
        let bgl_2 = {
            #[cfg(not(any(target_arch = "wasm32", target_os = "macos", target_os = "ios")))]
            const MAX_DECAL_TEX: u32 = 16;
            #[cfg(any(target_arch = "wasm32", target_os = "macos", target_os = "ios"))]
            const MAX_DECAL_TEX: u32 = 4;

            let mut entries = vec![
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: Some(std::num::NonZeroU32::new(MAX_DECAL_TEX).unwrap()),
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: Some(std::num::NonZeroU32::new(MAX_DECAL_TEX).unwrap()),
                },
            ];
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Decal BGL 2"),
                entries: &entries,
            })
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Decal PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1), Some(&bgl_2)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Decal Apply"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Immutable BG 0: camera, globals, decals
        let bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Decal BG 0"),
            layout: &bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: decal_buf.as_entire_binding(),
                },
            ],
        });

        Self {
            pipeline,
            bgl_0,
            bgl_1,
            bgl_2,
            bg_0: Some(bg_0),
            bg_1: None,
            bg_1_key: None,
            bg_2: None,
            globals_buf,
        }
    }
}

impl RenderPass for DecalPass {
    fn name(&self) -> &'static str {
        "DecalApply"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("gbuffer");
        builder.read("depth");
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let decal_count = ctx.scene.decals.len() as u32;
        let globals = DecalGlobals {
            decal_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if ctx.scene.decal_count == 0 {
            return Ok(());
        }

        let gbuffer = match ctx.resources.gbuffer.read(self.name()) {
            Some(g) => g,
            None => return Ok(()),
        };

        let depth_view = ctx.depth;

        // Rebuild BG 1 when texture views change
        let key = (
            depth_view as *const _ as usize,
            gbuffer.albedo as *const _ as usize,
            gbuffer.normal as *const _ as usize,
            gbuffer.orm as *const _ as usize,
            gbuffer.emissive as *const _ as usize,
        );

        if self.bg_1_key != Some(key) || self.bg_1.is_none() {
            self.bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Decal BG 1"),
                layout: &self.bgl_1,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(gbuffer.albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(gbuffer.normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(gbuffer.orm),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(gbuffer.emissive),
                    },
                    // storage writes — same views (texture has STORAGE_BINDING usage)
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(gbuffer.albedo),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(gbuffer.normal),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(gbuffer.orm),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(gbuffer.emissive),
                    },
                ],
            }));
            self.bg_1_key = Some(key);
        }

        // Build BG 2 once (bindless decal textures, reuse scene texture arrays)
        if self.bg_2.is_none() {
            if let Some(main_scene) = ctx.resources.main_scene.read(self.name()) {
                let tex_views = main_scene.material_textures.texture_views;
                let samplers = main_scene.material_textures.samplers;
                self.bg_2 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Decal BG 2"),
                    layout: &self.bgl_2,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureViewArray(tex_views),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::SamplerArray(samplers),
                        },
                    ],
                }));
            }
        }

        let mut cpass = unsafe { &mut *ctx.encoder_ptr }
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DecalApply"),
                timestamp_writes: None,
            });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, self.bg_0.as_ref().unwrap(), &[]);
        if let Some(bg) = &self.bg_1 {
            cpass.set_bind_group(1, bg, &[]);
        }
        if let Some(bg) = &self.bg_2 {
            cpass.set_bind_group(2, bg, &[]);
        }

        let w = (ctx.width + 15) / 16;
        let h = (ctx.height + 15) / 16;
        cpass.dispatch_workgroups(w, h, 1);

        Ok(())
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "depth"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["gbuffer"]
    }
}
