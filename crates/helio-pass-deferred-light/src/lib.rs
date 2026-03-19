use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DeferredGlobals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],
    rc_world_min: [f32; 4],
    rc_world_max: [f32; 4],
    csm_splits: [f32; 4],
    debug_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct DeferredLightPass {
    pipeline: wgpu::RenderPipeline,
    globals_buf: wgpu::Buffer,
    bgl_0: wgpu::BindGroupLayout,
    bgl_1: wgpu::BindGroupLayout,
    bgl_2: wgpu::BindGroupLayout,
    bind_group_0: wgpu::BindGroup,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_2: Option<wgpu::BindGroup>,
    bind_group_1_key: Option<(usize, usize, usize, usize, usize)>,
    bind_group_2_key: Option<(usize, usize, usize, usize, usize, usize)>,
    pre_aa_texture: wgpu::Texture,
    pre_aa_view: wgpu::TextureView,
    pre_aa_format: wgpu::TextureFormat,
    fallback_shadow_texture: wgpu::Texture,
    fallback_shadow_view: wgpu::TextureView,
    fallback_shadow_sampler: wgpu::Sampler,
    fallback_env_texture: wgpu::Texture,
    fallback_env_view: wgpu::TextureView,
    fallback_env_sampler: wgpu::Sampler,
    fallback_rc_texture: wgpu::Texture,
    fallback_rc_view: wgpu::TextureView,
}

impl DeferredLightPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
        pre_aa_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Deferred Lighting Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/deferred_lighting.wgsl").into(),
            ),
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Deferred Globals"),
            size: std::mem::size_of::<DeferredGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DeferredLight BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DeferredLight BGL1"),
            entries: &[
                texture_entry(0, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(2, wgpu::TextureSampleType::Float { filterable: false }),
                texture_entry(3, wgpu::TextureSampleType::Float { filterable: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let bgl_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DeferredLight BGL2"),
            entries: &[
                storage_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::Cube,
                        multisampled: false,
                    },
                    count: None,
                },
                storage_entry(4),
                texture_entry(5, wgpu::TextureSampleType::Float { filterable: false }),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DeferredLight BG0"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("DeferredLight PL"),
            bind_group_layouts: &[&bgl_0, &bgl_1, &bgl_2],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DeferredLight Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: pre_aa_format,
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
            multiview: None,
            cache: None,
        });

        let (pre_aa_texture, pre_aa_view) =
            color_texture(device, width, height, pre_aa_format, "Deferred PreAA");
        let (fallback_shadow_texture, fallback_shadow_view) = fallback_shadow_texture(device);
        let fallback_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Deferred Fallback Shadow Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let (fallback_env_texture, fallback_env_view) = black_cube_texture(device, queue);
        let fallback_env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Deferred Env Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let (fallback_rc_texture, fallback_rc_view) = black_2d_texture(device, queue, "Deferred Fallback RC");

        Self {
            pipeline,
            globals_buf,
            bgl_0,
            bgl_1,
            bgl_2,
            bind_group_0,
            bind_group_1: None,
            bind_group_2: None,
            bind_group_1_key: None,
            bind_group_2_key: None,
            pre_aa_texture,
            pre_aa_view,
            pre_aa_format,
            fallback_shadow_texture,
            fallback_shadow_view,
            fallback_shadow_sampler,
            fallback_env_texture,
            fallback_env_view,
            fallback_env_sampler,
            fallback_rc_texture,
            fallback_rc_view,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (texture, view) =
            color_texture(device, width, height, self.pre_aa_format, "Deferred PreAA");
        self.pre_aa_texture = texture;
        self.pre_aa_view = view;
    }
}

impl RenderPass for DeferredLightPass {
    fn name(&self) -> &'static str {
        "DeferredLight"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.pre_aa = Some(&self.pre_aa_view);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let main_scene = ctx.frame_resources.main_scene.as_ref();
        let (ambient_color, ambient_intensity) = if let Some(main_scene) = main_scene {
            (main_scene.ambient_color, main_scene.ambient_intensity)
        } else {
            ([0.1, 0.1, 0.15], 0.1)
        };
        let globals = DeferredGlobals {
            frame: ctx.frame as u32,
            delta_time: 0.0,
            light_count: ctx.scene.lights.len() as u32,
            ambient_intensity,
            ambient_color: [ambient_color[0], ambient_color[1], ambient_color[2], 1.0],
            rc_world_min: [0.0; 4],
            rc_world_max: [0.0; 4],
            csm_splits: [5.0, 20.0, 60.0, 200.0],
            debug_mode: 0, // 0 = full PBR lighting
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let gbuffer = ctx
            .frame
            .gbuffer
            .as_ref()
            .ok_or_else(|| helio_v3::Error::InvalidPassConfig(
                "DeferredLight requires published gbuffer resources".to_string(),
            ))?;

        let gbuffer_key = (
            gbuffer.albedo as *const _ as usize,
            gbuffer.normal as *const _ as usize,
            gbuffer.orm as *const _ as usize,
            gbuffer.emissive as *const _ as usize,
            ctx.depth as *const _ as usize,
        );
        if self.bind_group_1_key != Some(gbuffer_key) {
            self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DeferredLight BG1"),
                layout: &self.bgl_1,
                entries: &[
                    texture_view_entry(0, gbuffer.albedo),
                    texture_view_entry(1, gbuffer.normal),
                    texture_view_entry(2, gbuffer.orm),
                    texture_view_entry(3, gbuffer.emissive),
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                ],
            }));
            self.bind_group_1_key = Some(gbuffer_key);
        }

        let shadow_view = ctx
            .frame
            .shadow_atlas
            .unwrap_or(&self.fallback_shadow_view);
        let shadow_sampler = ctx
            .frame
            .shadow_sampler
            .unwrap_or(&self.fallback_shadow_sampler);
        let rc_view = &self.fallback_rc_view;
        let env_view = &self.fallback_env_view;
        let scene_key = (
            ctx.scene.lights as *const _ as usize,
            shadow_view as *const _ as usize,
            shadow_sampler as *const _ as usize,
            env_view as *const _ as usize,
            ctx.scene.shadow_matrices as *const _ as usize,
            rc_view as *const _ as usize,
        );
        if self.bind_group_2_key != Some(scene_key) {
            self.bind_group_2 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("DeferredLight BG2"),
                layout: &self.bgl_2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.lights.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(shadow_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(shadow_sampler),
                    },
                    texture_view_entry(3, env_view),
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.scene.shadow_matrices.as_entire_binding(),
                    },
                    texture_view_entry(5, rc_view),
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::Sampler(&self.fallback_env_sampler),
                    },
                ],
            }));
            self.bind_group_2_key = Some(scene_key);
        }

        let color_target = ctx.frame.pre_aa.unwrap_or(ctx.target);
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: color_target,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("DeferredLight"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        };
        let mut pass = ctx.begin_render_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group_0, &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);
        pass.set_bind_group(2, self.bind_group_2.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
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

fn color_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn fallback_shadow_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Deferred Fallback Shadow"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    (texture, view)
}

fn black_2d_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let zero = [0u8; 8];
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &zero,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

fn black_cube_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Deferred Fallback Env Cube"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let zero = [0u8; 8 * 6];
    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &zero,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(8),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 6,
        },
    );
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::Cube),
        ..Default::default()
    });
    (texture, view)
}
