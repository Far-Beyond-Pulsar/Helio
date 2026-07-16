const _RC_TRACE_WGSL: &str = include_str!("../shaders/rc_trace.wgsl");

use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const PROBE_DIM: u32 = 8;
const DIR_DIM: u32 = 4;
const ATLAS_W: u32 = PROBE_DIM * DIR_DIM;
const ATLAS_H: u32 = PROBE_DIM * PROBE_DIM * DIR_DIM;

const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RCDynamic {
    world_min: [f32; 4],
    world_max: [f32; 4],
    frame: u32,
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    sky_color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct RCStatic {
    cascade_index: u32,
    probe_dim: u32,
    dir_dim: u32,
    t_max_bits: u32,
    parent_probe_dim: u32,
    parent_dir_dim: u32,
    _pad0: u32,
    _pad1: u32,
}

pub struct RadianceCascadesPass {
    /// Fallback pipeline (no RT).
    fb_pipeline: wgpu::ComputePipeline,
    /// RT pipeline (real rc_trace.wgsl).
    rt_pipeline: Option<wgpu::ComputePipeline>,
    fb_bgl: wgpu::BindGroupLayout,
    rt_bgl: Option<wgpu::BindGroupLayout>,
    fb_bind_group: Option<wgpu::BindGroup>,
    uniform_buf: wgpu::Buffer,
    static_buf: Option<wgpu::Buffer>,
    use_rt: bool,
}

const FALLBACK_WGSL: &str = r#"
struct RCDynamic {
    world_min:   vec4<f32>,
    world_max:   vec4<f32>,
    frame:       u32,
    light_count: u32,
    _pad0:       u32,
    _pad1:       u32,
    sky_color:   vec4<f32>,
}
@group(0) @binding(0) var cascade_out: texture_storage_2d<rgba16float, write>;
@group(0) @binding(1) var<uniform>  rc_dyn: RCDynamic;

@compute @workgroup_size(8, 8)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(cascade_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }
    textureStore(cascade_out, vec2<i32>(i32(gid.x), i32(gid.y)),
        vec4<f32>(rc_dyn.sky_color.rgb * 0.05, 1.0));
}
"#;

impl RadianceCascadesPass {
    pub fn new(device: &wgpu::Device, lights_buf: &wgpu::Buffer) -> Self {
        let _ = lights_buf;

        let use_rt = device
            .features()
            .contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);

        // ── Uniform buffers ────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RC Dynamic Uniform"),
            size: std::mem::size_of::<RCDynamic>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let static_buf = use_rt.then(|| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RC Static Uniform"),
                size: std::mem::size_of::<RCStatic>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });

        // ── Fallback BGL & pipeline ────────────────────────────────────
        let fb_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RC Fallback BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
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
            ],
        });

        let fb_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RC Fallback Shader"),
            source: wgpu::ShaderSource::Wgsl(FALLBACK_WGSL.into()),
        });

        let fb_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RC Fallback PL"),
            bind_group_layouts: &[Some(&fb_bgl)],
            immediate_size: 0,
        });

        let fb_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RC Fallback Pipeline"),
            layout: Some(&fb_pl),
            module: &fb_shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── RT BGL & pipeline (if supported) ───────────────────────────
        let (rt_bgl, rt_pipeline) = if use_rt {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("RC Trace BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba16Float,
                            view_dimension: wgpu::TextureViewDimension::D2,
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
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::AccelerationStructure {
                            vertex_return: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
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

            let rt_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RC Trace Shader"),
                source: wgpu::ShaderSource::Wgsl(_RC_TRACE_WGSL.into()),
            });

            let rt_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RC Trace PL"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RC Trace Pipeline"),
                layout: Some(&rt_pl),
                module: &rt_shader,
                entry_point: Some("cs_trace"),
                compilation_options: Default::default(),
                cache: None,
            });

            (Some(bgl), Some(pipeline))
        } else {
            (None, None)
        };

        Self {
            fb_pipeline,
            rt_pipeline,
            fb_bgl,
            rt_bgl,
            fb_bind_group: None,
            uniform_buf,
            static_buf,
            use_rt,
        }
    }
}

impl RenderPass for RadianceCascadesPass {
    fn name(&self) -> &'static str {
        "RadianceCascades"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw(
            "rc_cascades",
            wgpu::TextureFormat::Rgba16Float,
            ResourceSize::Absolute {
                width: ATLAS_W,
                height: ATLAS_H,
            },
        );
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);

        if self.use_rt {
            builder.write_color_raw(
                "rc_history",
                wgpu::TextureFormat::Rgba16Float,
                ResourceSize::Absolute {
                    width: ATLAS_W,
                    height: ATLAS_H,
                },
            );
            builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
        }
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
        let light_count = ctx.scene.lights.len() as u32;
        let sky = ctx.frame_resources.sky.sky_color;
        let dyn_data = RCDynamic {
            world_min: [-10.0, -1.0, -10.0, 0.0],
            world_max: [10.0, 10.0, 10.0, 0.0],
            frame: ctx.frame_num as u32,
            light_count,
            _pad0: 0,
            _pad1: 0,
            sky_color: [sky[0], sky[1], sky[2], 0.0],
        };
        ctx.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&dyn_data));

        if let Some(ref static_buf) = self.static_buf {
            let static_data = RCStatic {
                cascade_index: 0,
                probe_dim: PROBE_DIM,
                dir_dim: DIR_DIM,
                t_max_bits: f32::MAX.to_bits(),
                parent_probe_dim: 0,
                parent_dir_dim: 0,
                _pad0: 0,
                _pad1: 0,
            };
            ctx.write_buffer(static_buf, 0, bytemuck::bytes_of(&static_data));
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.use_rt {
            self.execute_rt(ctx)
        } else {
            self.execute_fallback(ctx)
        }
    }
}

impl RadianceCascadesPass {
    fn execute_fallback(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.fb_bind_group.is_none() {
            let tex = ctx
                .resource_pool
                .get_texture("rc_cascades")
                .ok_or_else(|| {
                    helio_core::Error::InvalidPassConfig(
                        "RadianceCascades: missing rc_cascades texture".into(),
                    )
                })?;
            let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
            self.fb_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RC Fallback BG"),
                    layout: &self.fb_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.uniform_buf.as_entire_binding(),
                        },
                    ],
                }));
        }

        let wg_x = ATLAS_W.div_ceil(WORKGROUP_SIZE_X);
        let wg_y = ATLAS_H.div_ceil(WORKGROUP_SIZE_Y);

        let desc = wgpu::ComputePassDescriptor {
            label: Some("RadianceCascades"),
            timestamp_writes: None,
        };
        let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&desc);
        pass.set_pipeline(&self.fb_pipeline);
        pass.set_bind_group(0, self.fb_bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        Ok(())
    }

    fn execute_rt(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let rt_bgl = self.rt_bgl.as_ref().unwrap();
        let rt_pipeline = self.rt_pipeline.as_ref().unwrap();

        let cascade_out = ctx
            .resource_pool
            .get_texture("rc_cascades")
            .ok_or_else(|| {
                helio_core::Error::InvalidPassConfig(
                    "RadianceCascades: missing rc_cascades texture".into(),
                )
            })?;
        let cascade_out_view =
            cascade_out.create_view(&wgpu::TextureViewDescriptor::default());

        let history = ctx
            .resource_pool
            .get_texture("rc_history")
            .ok_or_else(|| {
                helio_core::Error::InvalidPassConfig(
                    "RadianceCascades: missing rc_history texture".into(),
                )
            })?;
        let history_view = history.create_view(&wgpu::TextureViewDescriptor::default());

        let lights_buf = ctx.scene.lights;

        // Get TLAS from frame resources (set by the renderer from GpuScene)
        let main_scene = ctx.resources.main_scene.read("RadianceCascades");
        let tlas = main_scene.and_then(|ms| ms.tlas);

        let Some(tlas) = tlas else {
            // No TLAS — fall back to the ambient-only fallback shader.
            return self.execute_fallback(ctx);
        };

        // NB: entries must be in binding order to match BGL.
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&cascade_out_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&cascade_out_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.uniform_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self.static_buf.as_ref().unwrap().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: tlas.as_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: lights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&history_view),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: wgpu::BindingResource::TextureView(&history_view),
            },
        ];

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RC Trace BG"),
            layout: rt_bgl,
            entries: &entries,
        });

        let wg_x = ATLAS_W.div_ceil(WORKGROUP_SIZE_X);
        let wg_y = ATLAS_H.div_ceil(WORKGROUP_SIZE_Y);

        let desc = wgpu::ComputePassDescriptor {
            label: Some("RadianceCascades (RT)"),
            timestamp_writes: None,
        };
        let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&desc);
        pass.set_pipeline(rt_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        Ok(())
    }
}
