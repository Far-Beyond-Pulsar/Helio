//! Cinematic Depth of Field (Bokeh DOF) Pass
//!
//! Implements a half-resolution split-pass DOF with physically based bokeh
//! aperture shapes. Supports both the classic Gaussian fallback (circular,
//! cheap) and the new polygonal bokeh mode.
//!
//! Pipeline:
//!   1. `cs_coc_prepass`  — compute CoC at full-res → half-res R16F buffer
//!   2. `cs_gather`       — half-res gather with Poisson disc + bokeh shape
//!   3. `fs_composite`    — full-res composite with CoC-aware upsample
//!
//! Reads:   "pre_dof" (scene colour), "depth"
//! Writes:  "post_dof" (final colour after DOF)
//! Internal: coc_tex (R32Float half-res), near_blur (RGBA16F half-res),
//!           far_blur (RGBA16F half-res)

mod bokeh_shape;

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const COC_SHADER_SRC: &str = include_str!("../shaders/dof_coc.wgsl");
const GATHER_SHADER_SRC: &str = include_str!("../shaders/dof_gather.wgsl");
const COMPOSITE_SHADER_SRC: &str = include_str!("../shaders/dof_composite.wgsl");

const WG_COC: u32 = 16;
const WG_GATHER: u32 = 8;

/// Byte offset of the DOF block within GpuPostProcessUniforms.
const DOF_BLOCK_OFFSET: u64 = 224;
/// Size of the DOF block (8 f32 fields).
const DOF_BLOCK_SIZE: u64 = 32;

pub struct DofPass {
    // Pipelines
    coc_pipeline: wgpu::ComputePipeline,
    gather_pipeline: wgpu::ComputePipeline,
    composite_pipeline: wgpu::RenderPipeline,

    // Internal textures (half-resolution)
    coc_tex: wgpu::Texture,
    coc_view: wgpu::TextureView,
    near_blur_tex: wgpu::Texture,
    near_blur_view: wgpu::TextureView,
    far_blur_tex: wgpu::Texture,
    far_blur_view: wgpu::TextureView,

    // Bokeh shape texture
    bokeh_tex: wgpu::Texture,
    bokeh_view: wgpu::TextureView,

    // Samplers
    linear_sampler: wgpu::Sampler,

    // Bind group layouts
    coc_bgl: wgpu::BindGroupLayout,
    gather_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,

    // Pipeline layouts
    gather_pl: wgpu::PipelineLayout,
    composite_pl: wgpu::PipelineLayout,

    // Bind groups (rebuilt when resources change)
    coc_bg: Option<wgpu::BindGroup>,
    gather_bg: Option<wgpu::BindGroup>,
    composite_bg: Option<wgpu::BindGroup>,

    // Cached keys for lazy rebuild
    bg_key_coc: Option<(usize, usize)>,
    bg_key_gather: Option<(usize, usize, usize, usize, usize, usize)>,
    bg_key_composite: Option<(usize, usize, usize, usize, usize)>,

    // Tiny uniform buffer holding a copy of the DOF block from the shared
    // postprocess_uniforms buffer. Contents are refreshed via GPU copy in execute().
    dof_block_buf: wgpu::Buffer,

    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
}

impl DofPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let half_w = (width + 1) / 2;
        let half_h = (height + 1) / 2;

        // ── Internal textures ───────────────────────────────────────────
        let coc_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF CoC"),
            size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let coc_view = coc_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let near_blur_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF Near Blur"),
            size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let near_blur_view = near_blur_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let far_blur_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF Far Blur"),
            size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let far_blur_view = far_blur_tex.create_view(&wgpu::TextureViewDescriptor::default());

        // ── Bokeh shape texture ─────────────────────────────────────────
        let bokeh_tex = bokeh_shape::create_bokeh_shape_texture(device, queue);
        let bokeh_view = bokeh_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // ── DOF block uniform buffer (populated via GPU copy each frame) ──
        let dof_block_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DOF Block Uniforms"),
            size: DOF_BLOCK_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Samplers ────────────────────────────────────────────────────
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("DOF Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ── Shaders ─────────────────────────────────────────────────────
        let coc_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DOF CoC Shader"),
            source: wgpu::ShaderSource::Wgsl(
                helio_core::shader::resolve(COC_SHADER_SRC).into_owned().into(),
            ),
        });
        let gather_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DOF Gather Shader"),
            source: wgpu::ShaderSource::Wgsl(
                helio_core::shader::resolve(GATHER_SHADER_SRC).into_owned().into(),
            ),
        });
        let composite_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DOF Composite Shader"),
            source: wgpu::ShaderSource::Wgsl(
                helio_core::shader::resolve(COMPOSITE_SHADER_SRC).into_owned().into(),
            ),
        });

        // ── Bind group layouts ──────────────────────────────────────────
        /// Uniform buffer binding for the DOF block of GpuPostProcessUniforms.
        /// The buffer is bound with offset DOF_BLOCK_OFFSET and size DOF_BLOCK_SIZE.
        let uniform_entry = |binding: u32, min_size: Option<wgpu::BufferSize>| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: min_size,
            },
            count: None,
        };

        // The engine-wide camera buffer (`GpuCameraBuffer`, label "Camera Storage")
        // is a storage buffer sized for 2 cameras (mono/stereo), matching every
        // other pass's `var<storage, read> cameras: array<CameraUniforms, 2>`.
        // Bindings that reference it must use this, not `uniform_entry`.
        let camera_storage_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let coc_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DOF CoC BGL"),
            entries: &[
                uniform_entry(0, wgpu::BufferSize::new(DOF_BLOCK_SIZE)),
                camera_storage_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Note: R32Float storage textures are used for the CoC buffer because
        // R16Float does not support STORAGE_BINDING on all wgpu backends.

        let gather_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DOF Gather BGL"),
            entries: &[
                uniform_entry(0, wgpu::BufferSize::new(DOF_BLOCK_SIZE)),
                camera_storage_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        // R32Float CoC buffer — only accessed via textureLoad, no sampler needed
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
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
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
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DOF Composite BGL"),
            entries: &[
                uniform_entry(0, wgpu::BufferSize::new(DOF_BLOCK_SIZE)),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        // R32Float CoC buffer — only accessed via textureLoad
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // ── Pipeline layouts ────────────────────────────────────────────
        let coc_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DOF CoC PL"),
            bind_group_layouts: &[Some(&coc_bgl)],
            immediate_size: 0,
        });

        let gather_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DOF Gather PL"),
            bind_group_layouts: &[Some(&gather_bgl)],
            immediate_size: 0,
        });

        let composite_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DOF Composite PL"),
            bind_group_layouts: &[Some(&composite_bgl)],
            immediate_size: 0,
        });

        // ── Pipelines ───────────────────────────────────────────────────
        let coc_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DOF CoC Pipeline"),
            layout: Some(&coc_pl),
            module: &coc_shader,
            entry_point: Some("cs_coc_prepass"),
            compilation_options: Default::default(),
            cache: None,
        });

        let gather_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DOF Gather Pipeline"),
            layout: Some(&gather_pl),
            module: &gather_shader,
            entry_point: Some("cs_gather"),
            compilation_options: Default::default(),
            cache: None,
        });

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DOF Composite Pipeline"),
            layout: Some(&composite_pl),
            vertex: wgpu::VertexState {
                module: &composite_shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &composite_shader,
                entry_point: Some("fs_composite"),
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
            coc_pipeline,
            gather_pipeline,
            composite_pipeline,
            coc_tex,
            coc_view,
            near_blur_tex,
            near_blur_view,
            far_blur_tex,
            far_blur_view,
            bokeh_tex,
            bokeh_view,
            linear_sampler,
            coc_bgl,
            gather_bgl,
            composite_bgl,
            gather_pl,
            composite_pl,
            coc_bg: None,
            gather_bg: None,
            composite_bg: None,
            bg_key_coc: None,
            bg_key_gather: None,
            bg_key_composite: None,
            dof_block_buf,
            width,
            height,
            format,
        }
    }

    fn rebuild_coc_bg(
        &mut self,
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DOF CoC BG"),
            layout: &self.coc_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.dof_block_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.coc_view) },
            ],
        });
        self.coc_bg = Some(bg);
    }

    fn rebuild_gather_bg(
        &mut self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DOF Gather BG"),
            layout: &self.gather_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.dof_block_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(src_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.coc_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.bokeh_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.near_blur_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.far_blur_view) },
            ],
        });
        self.gather_bg = Some(bg);
    }

    fn rebuild_composite_bg(
        &mut self,
        device: &wgpu::Device,
        src_view: &wgpu::TextureView,
    ) {
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DOF Composite BG"),
            layout: &self.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.dof_block_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(src_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.near_blur_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.far_blur_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&self.coc_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
            ],
        });
        self.composite_bg = Some(bg);
    }
}

impl RenderPass for DofPass {
    fn name(&self) -> &'static str {
        "DofPass"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["pre_dof", "depth"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_dof");
        builder.read("depth");
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;

        let half_w = (width + 1) / 2;
        let half_h = (height + 1) / 2;

        let new_coc = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF CoC"),
            size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.coc_tex = new_coc;
        self.coc_view = self.coc_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let new_near = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF Near Blur"),
            size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.near_blur_tex = new_near;
        self.near_blur_view = self.near_blur_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let new_far = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DOF Far Blur"),
            size: wgpu::Extent3d { width: half_w, height: half_h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.far_blur_tex = new_far;
        self.far_blur_view = self.far_blur_tex.create_view(&wgpu::TextureViewDescriptor::default());

        self.coc_bg = None;
        self.gather_bg = None;
        self.composite_bg = None;
        self.bg_key_coc = None;
        self.bg_key_gather = None;
        self.bg_key_composite = None;
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let src_view = match ctx.resources.pre_dof.get() {
            Some(v) => v,
            None => match ctx.resources.pre_aa.get() {
                Some(v) => v,
                None => return Ok(()),
            },
        };
        let Some(pp_buf) = ctx.resources.postprocess_uniforms.get() else {
            return Ok(());
        };
        let depth_view = ctx.depth;
        let camera_buf = ctx.scene.camera;

        let half_w = (self.width + 1) / 2;
        let half_h = (self.height + 1) / 2;

        // ── Lazy rebuild bind groups ────────────────────────────────────
        let coc_key = (
            depth_view as *const _ as usize,
            camera_buf as *const _ as usize,
        );
        if self.bg_key_coc != Some(coc_key) {
            self.rebuild_coc_bg(ctx.device, depth_view, camera_buf);
            self.bg_key_coc = Some(coc_key);
        }

        let gather_key = (
            src_view as *const _ as usize,
            camera_buf as *const _ as usize,
            0, 0, 0, 0,
        );
        if self.bg_key_gather != Some(gather_key) {
            self.rebuild_gather_bg(ctx.device, src_view, camera_buf);
            self.bg_key_gather = Some(gather_key);
        }

        let composite_key = (
            src_view as *const _ as usize,
            0, 0, 0, 0,
        );
        if self.bg_key_composite != Some(composite_key) {
            self.rebuild_composite_bg(ctx.device, src_view);
            self.bg_key_composite = Some(composite_key);
        }

        // ── Copy DOF block → compute encoder before dispatches ─────────
        // The postprocess uniform buffer lives on the render-encoder timeline.
        // Compute dispatches run on a separate encoder that submits first,
        // so we must copy here (on the compute encoder) to avoid a race.
        {
            let ce = ctx.compute_encoder_ptr;
            unsafe { &mut *ce }.copy_buffer_to_buffer(
                pp_buf, DOF_BLOCK_OFFSET,
                &self.dof_block_buf, 0,
                DOF_BLOCK_SIZE,
            );
        }

        // ── Pass 1: CoC pre-pass ────────────────────────────────────────
        {
            let ce = ctx.compute_encoder_ptr;
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DOF CoC"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.coc_pipeline);
            cpass.set_bind_group(0, self.coc_bg.as_ref().unwrap(), &[]);
            let gx = (half_w + WG_COC - 1) / WG_COC;
            let gy = (half_h + WG_COC - 1) / WG_COC;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // ── Pass 2: Gather ─────────────────────────────────────────────
        {
            let ce = ctx.compute_encoder_ptr;
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DOF Gather"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.gather_pipeline);
            cpass.set_bind_group(0, self.gather_bg.as_ref().unwrap(), &[]);
            let gx = (half_w + WG_GATHER - 1) / WG_GATHER;
            let gy = (half_h + WG_GATHER - 1) / WG_GATHER;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // ── Pass 3: Composite ──────────────────────────────────────────
        {
            let attachments = [Some(wgpu::RenderPassColorAttachment {
                view: ctx.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DOF Composite"),
                color_attachments: &attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.composite_pipeline);
            pass.set_bind_group(0, self.composite_bg.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
