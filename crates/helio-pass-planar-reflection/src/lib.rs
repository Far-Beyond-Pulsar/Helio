use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlanarGlobals {
    plane_pos: [f32; 4],
    plane_normal: [f32; 4],
    half_extents: [f32; 4],
    cos_angle_threshold: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

pub struct PlanarReflectionPass {
    pipeline: wgpu::ComputePipeline,
    bgl_1: wgpu::BindGroupLayout,
    bg_0: wgpu::BindGroup,
    bg_1: Option<wgpu::BindGroup>,
    bg_1_key: Option<(usize, usize, usize, usize)>,
    linear_sampler: wgpu::Sampler,
    globals_buf: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl PlanarReflectionPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        _surface_format: wgpu::TextureFormat,
    ) -> Self {
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Planar Linear Sampler"),
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
            "Planar Trace Shader",
            include_str!("../shaders/planar_trace.wgsl"),
        );

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planar Globals"),
            size: std::mem::size_of::<PlanarGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planar BGL0"),
            entries: &[
                buffer_uniform_entry(0),
                buffer_uniform_entry(1),
            ],
        });

        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Planar BGL1"),
            entries: &[
                texture_float_entry(0),
                texture_depth_entry(1),
                texture_float_entry(2),
                sampler_entry(3),
                storage_entry(4),
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planar PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Planar Trace Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bg_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planar BG0"),
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
            ],
        });

        Self {
            pipeline,
            bgl_1,
            bg_0,
            bg_1: None,
            bg_1_key: None,
            linear_sampler,
            globals_buf,
            width: 0,
            height: 0,
        }
    }
}

impl RenderPass for PlanarReflectionPass {
    fn name(&self) -> &'static str {
        "PlanarReflection"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "depth", "pre_aa"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["planar_reflection"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw(
            "planar_reflection",
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
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let globals = PlanarGlobals {
            // Default: XZ plane at origin, 50-unit half-extents, normal tolerance cos(15 deg)
            plane_pos: [0.0, 0.0, 0.0, 0.0],
            plane_normal: [0.0, 1.0, 0.0, 0.0],
            half_extents: [50.0, 50.0, 1.0, 0.0],
            cos_angle_threshold: 0.9659, // cos(15 deg)
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let gbuffer = match ctx.resources.gbuffer.read("PlanarReflection") {
            Some(g) => g,
            None => return Ok(()),
        };
        let depth_view = ctx.depth;
        let pre_aa_view = match ctx.resources.pre_aa.get() {
            Some(v) => v,
            None => return Ok(()),
        };
        let planar_tex = match ctx.resource_pool.get_view("planar_reflection") {
            Some(v) => v,
            None => return Ok(()),
        };

        let key = (
            gbuffer.normal as *const _ as usize,
            depth_view as *const _ as usize,
            pre_aa_view as *const _ as usize,
            planar_tex as *const _ as usize,
        );

        if self.bg_1_key != Some(key) {
            self.bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Planar BG1"),
                layout: &self.bgl_1,
                entries: &[
                    texture_view_entry(0, gbuffer.normal),
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    texture_view_entry(2, pre_aa_view),
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Sampler(&self.linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(planar_tex),
                    },
                ],
            }));
            self.bg_1_key = Some(key);
        }

        let cpass = unsafe { &mut *ctx.compute_encoder_ptr };
        let mut pass = cpass.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Planar Reflection Trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bg_0, &[]);
        pass.set_bind_group(1, self.bg_1.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);

        Ok(())
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        // Published by the graph automatically via the resource pool name.
    }
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
