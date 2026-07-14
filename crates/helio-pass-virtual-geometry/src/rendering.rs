use crate::{CullUniforms, LodQuality, VgGlobals, INITIAL_INSTANCES, INITIAL_MESHLETS};
use helio_v3::graph::ResourceBuilder;
use helio_v3::{
    DebugViewDescriptor, GpuInstanceData, PassContext, PrepareContext, RenderPass,
    Result as HelioResult,
};

// ═══════════════════════════════════════════════════════════════════════════════
// VirtualGeometryPass
// ═══════════════════════════════════════════════════════════════════════════════

pub struct VirtualGeometryPass {
    pub(crate) cull_pipeline: wgpu::ComputePipeline,
    pub(crate) cull_bgl: wgpu::BindGroupLayout,
    pub(crate) cull_bind_group: Option<wgpu::BindGroup>,
    pub(crate) cull_bind_group_hiz_key: Option<(usize, usize)>,
    pub(crate) cull_buf: wgpu::Buffer,
    pub(crate) draw_pipeline: wgpu::RenderPipeline,
    pub(crate) debug_draw_pipeline: Option<wgpu::RenderPipeline>,
    pub(crate) lod_debug_pipeline: wgpu::RenderPipeline,
    pub(crate) draw_bgl_0: wgpu::BindGroupLayout,
    pub(crate) draw_bgl_1: wgpu::BindGroupLayout,
    pub(crate) draw_bg_0: Option<wgpu::BindGroup>,
    pub(crate) draw_bg_1: Option<wgpu::BindGroup>,
    pub(crate) bg1_version: Option<u64>,
    pub(crate) globals_buf: wgpu::Buffer,
    pub(crate) meshlet_buf: wgpu::Buffer,
    pub(crate) instance_buf: wgpu::Buffer,
    pub(crate) instance_scale_buf: wgpu::Buffer,
    pub(crate) indirect_buf: wgpu::Buffer,
    pub(crate) draw_count_buf: wgpu::Buffer,
    pub debug_mode: u32,
    pub lod_quality: LodQuality,
    pub(crate) last_version: u64,
    pub(crate) last_meshlet_count: u32,
}

impl VirtualGeometryPass {
    pub fn new(device: &wgpu::Device, camera_buf: &wgpu::Buffer) -> Self {
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vg_cull.wgsl").into()),
        });
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG GBuffer Shader"),
            source: wgpu::ShaderSource::Wgsl(
                libhelio::webgpu_material_shader(include_str!("../shaders/vg_gbuffer.wgsl")).into(),
            ),
        });

        let meshlet_buf = Self::make_meshlet_buf(device, INITIAL_MESHLETS);
        let instance_buf = Self::make_instance_buf(device, INITIAL_INSTANCES);
        let instance_scale_buf = Self::make_instance_scale_buf(device, INITIAL_INSTANCES);
        let indirect_buf = Self::make_indirect_buf(device, INITIAL_MESHLETS);
        let draw_count_buf = Self::make_draw_count_buf(device);

        let cull_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG CullUniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Globals"),
            size: std::mem::size_of::<VgGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VG Cull BGL"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
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

        let cull_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Cull PL"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VG Cull Pipeline"),
            layout: Some(&cull_pipeline_layout),
            module: &cull_shader,
            entry_point: Some("cs_cull"),
            compilation_options: Default::default(),
            cache: None,
        });

        let draw_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VG Draw BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let draw_bg_0 = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VG Draw BG0"),
            layout: &draw_bgl_0,
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
                    resource: instance_buf.as_entire_binding(),
                },
            ],
        }));

        let draw_bgl_1 = create_material_bgl(device);

        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Draw PL"),
            bind_group_layouts: &[Some(&draw_bgl_0), Some(&draw_bgl_1)],
            immediate_size: 0,
        });
        let vg_vertex_buffers = &[Some(wgpu::VertexBufferLayout {
            array_stride: 40,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 32,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 36,
                    shader_location: 4,
                },
            ],
        })];
        let gbuffer_targets = &[
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let draw_primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        };
        let draw_depth = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VG Draw Pipeline"),
            layout: Some(&draw_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: vg_vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: gbuffer_targets,
            }),
            primitive: draw_primitive,
            depth_stencil: draw_depth.clone(),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let debug_draw_pipeline = if device.features().contains(wgpu::Features::PRIMITIVE_INDEX) {
            Some(
                device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("VG Debug Pipeline"),
                    layout: Some(&draw_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &draw_shader,
                        entry_point: Some("vs_main"),
                        compilation_options: Default::default(),
                        buffers: vg_vertex_buffers,
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &draw_shader,
                        entry_point: Some("fs_debug"),
                        compilation_options: Default::default(),
                        targets: gbuffer_targets,
                    }),
                    primitive: draw_primitive,
                    depth_stencil: draw_depth.clone(),
                    multisample: wgpu::MultisampleState::default(),
                    multiview_mask: None,
                    cache: None,
                }),
            )
        } else {
            None
        };

        let lod_debug_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("VG LOD Debug Pipeline"),
            layout: Some(&draw_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &draw_shader,
                entry_point: Some("vs_debug_lod"),
                compilation_options: Default::default(),
                buffers: vg_vertex_buffers,
            },
            fragment: Some(wgpu::FragmentState {
                module: &draw_shader,
                entry_point: Some("fs_debug_lod"),
                compilation_options: Default::default(),
                targets: gbuffer_targets,
            }),
            primitive: draw_primitive,
            depth_stencil: draw_depth,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            cull_pipeline,
            cull_bgl,
            cull_bind_group: None,
            cull_bind_group_hiz_key: None,
            cull_buf,
            draw_pipeline,
            debug_draw_pipeline,
            lod_debug_pipeline,
            draw_bgl_0,
            draw_bgl_1,
            draw_bg_0,
            draw_bg_1: None,
            bg1_version: None,
            globals_buf,
            meshlet_buf,
            instance_buf,
            instance_scale_buf,
            indirect_buf,
            draw_count_buf,
            debug_mode: 0,
            lod_quality: LodQuality::default(),
            last_version: u64::MAX,
            last_meshlet_count: 0,
        }
    }

    fn make_meshlet_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Meshlet Buffer"),
            size: capacity * 64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_instance_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Instance Buffer"),
            size: capacity * 144,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_instance_scale_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Instance Scale Buffer"),
            size: capacity * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_indirect_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Indirect Buffer"),
            size: capacity * 20,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_draw_count_buf(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Draw Count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn rebuild_owned_bind_groups(&mut self, device: &wgpu::Device, camera_buf: &wgpu::Buffer) {
        self.cull_bind_group = None;
        self.draw_bg_0 = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VG Draw BG0"),
            layout: &self.draw_bgl_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.instance_buf.as_entire_binding(),
                },
            ],
        }));
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RenderPass impl
// ═══════════════════════════════════════════════════════════════════════════════

impl RenderPass for VirtualGeometryPass {
    fn name(&self) -> &'static str {
        "VirtualGeometry"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let Some(vg) = ctx.frame_resources.vg.get() else {
            return Ok(());
        };

        if vg.buffer_version != self.last_version {
            let camera_buf = ctx.scene.camera.buffer();
            let mut grew = false;

            let meshlet_capacity = self.meshlet_buf.size() / 64;
            if (vg.meshlet_count as u64) > meshlet_capacity {
                self.meshlet_buf = Self::make_meshlet_buf(ctx.device, vg.meshlet_count as u64 * 2);
                self.indirect_buf =
                    Self::make_indirect_buf(ctx.device, vg.meshlet_count as u64 * 2);
                grew = true;
            }
            let instance_capacity = self.instance_buf.size() / 144;
            if (vg.instance_count as u64) > instance_capacity {
                self.instance_buf =
                    Self::make_instance_buf(ctx.device, vg.instance_count as u64 * 2);
                self.instance_scale_buf =
                    Self::make_instance_scale_buf(ctx.device, vg.instance_count as u64 * 2);
                grew = true;
            }

            if grew {
                self.rebuild_owned_bind_groups(ctx.device, camera_buf);
            }

            ctx.write_buffer(&self.meshlet_buf, 0, vg.meshlets);
            ctx.write_buffer(&self.instance_buf, 0, vg.instances);

            let instances: &[GpuInstanceData] = bytemuck::cast_slice(vg.instances);
            let scales: Vec<f32> = instances
                .iter()
                .map(|inst| {
                    let scale_x = (inst.model[0] * inst.model[0]
                        + inst.model[1] * inst.model[1]
                        + inst.model[2] * inst.model[2])
                        .sqrt();
                    let scale_y = (inst.model[4] * inst.model[4]
                        + inst.model[5] * inst.model[5]
                        + inst.model[6] * inst.model[6])
                        .sqrt();
                    let scale_z = (inst.model[8] * inst.model[8]
                        + inst.model[9] * inst.model[9]
                        + inst.model[10] * inst.model[10])
                        .sqrt();
                    scale_x.max(scale_y).max(scale_z)
                })
                .collect();
            ctx.write_buffer(&self.instance_scale_buf, 0, bytemuck::cast_slice(&scales));

            self.last_version = vg.buffer_version;
            self.last_meshlet_count = vg.meshlet_count;
        }

        if self.last_meshlet_count == 0 {
            return Ok(());
        }

        let lod_thresholds = self.lod_quality.thresholds();
        let max_dim = ctx.width.max(ctx.height);
        let hiz_mip_count = (u32::BITS - max_dim.leading_zeros()).max(1);
        let cull_uni = CullUniforms {
            meshlet_count: self.last_meshlet_count,
            screen_width: ctx.width,
            screen_height: ctx.height,
            hiz_mip_count,
            lod_thresholds,
            _pad3: 0.0,
        };
        ctx.write_buffer(&self.cull_buf, 0, bytemuck::bytes_of(&cull_uni));

        let Some(main_scene) = ctx.frame_resources.main_scene.read("VirtualGeometry") else {
            return Ok(());
        };
        if self.draw_bg_1.is_none()
            || self.bg1_version != Some(main_scene.material_textures.version)
        {
            let mut entries = vec![
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: ctx.scene.materials.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: main_scene
                        .material_textures
                        .material_textures
                        .as_entire_binding(),
                },
            ];
            libhelio::push_webgpu_material_bindings(
                &mut entries,
                main_scene.material_textures.texture_views,
                main_scene.material_textures.samplers,
            );
            self.draw_bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VG Draw BG1"),
                layout: &self.draw_bgl_1,
                entries: &entries,
            }));
            self.bg1_version = Some(main_scene.material_textures.version);
        }

        let globals = VgGlobals {
            frame: ctx.frame_num as u32,
            delta_time: 0.016,
            light_count: ctx.scene.lights.len() as u32,
            ambient_intensity: main_scene.ambient_intensity,
            ambient_color: [
                main_scene.ambient_color[0],
                main_scene.ambient_color[1],
                main_scene.ambient_color[2],
                0.0,
            ],
            rc_world_min: [
                main_scene.rc_world_min[0],
                main_scene.rc_world_min[1],
                main_scene.rc_world_min[2],
                0.0,
            ],
            rc_world_max: [
                main_scene.rc_world_max[0],
                main_scene.rc_world_max[1],
                main_scene.rc_world_max[2],
                0.0,
            ],
            csm_splits: [5.0, 20.0, 60.0, 200.0],
            debug_mode: self.debug_mode,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));

        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let gbuffer = resources.gbuffer.read("VirtualGeometry")?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([
                Some(wgpu::RenderPassColorAttachment {
                    view: gbuffer.albedo,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: gbuffer.normal,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: gbuffer.orm,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: gbuffer.emissive,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("VG GBuffer"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.last_meshlet_count == 0 || ctx.resources.vg.is_none() {
            return Ok(());
        }

        let hiz_view = ctx
            .resources
            .hiz
            .as_ref()
            .expect("VirtualGeometry: 'hiz' view not routed by graph");
        let hiz_sampler = ctx
            .resources
            .hiz_sampler
            .as_ref()
            .expect("VirtualGeometry: 'hiz_sampler' not available");
        let hiz_key = (
            hiz_view as *const _ as usize,
            hiz_sampler as *const _ as usize,
        );
        if self.cull_bind_group.is_none() || self.cull_bind_group_hiz_key != Some(hiz_key) {
            self.cull_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VG Cull BG"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.cull_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: self.meshlet_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.instance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.indirect_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.draw_count_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: wgpu::BindingResource::TextureView(hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Sampler(hiz_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.instance_scale_buf.as_entire_binding(),
                    },
                ],
            }));
            self.cull_bind_group_hiz_key = Some(hiz_key);
        }

        let Some(cull_bg) = self.cull_bind_group.as_ref() else {
            return Ok(());
        };
        let Some(draw_bg0) = self.draw_bg_0.as_ref() else {
            return Ok(());
        };
        let Some(draw_bg1) = self.draw_bg_1.as_ref() else {
            return Ok(());
        };
        let Some(main_scene) = ctx.resources.main_scene.read("VirtualGeometry") else {
            return Ok(());
        };

        let meshlet_count = self.last_meshlet_count;

        unsafe { &mut *ctx.compute_encoder_ptr }.clear_buffer(&self.draw_count_buf, 0, None);
        unsafe { &mut *ctx.compute_encoder_ptr }.clear_buffer(&self.indirect_buf, 0, None);

        {
            let mut cpass = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("VG Cull"),
                    timestamp_writes: None,
                },
            );
            cpass.set_pipeline(&self.cull_pipeline);
            cpass.set_bind_group(0, cull_bg, &[]);
            cpass.dispatch_workgroups((meshlet_count + 63) / 64, 1, 1);
        }

        {
            let rpass = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };

            let active_pipeline = match self.debug_mode {
                20 => self
                    .debug_draw_pipeline
                    .as_ref()
                    .unwrap_or(&self.draw_pipeline),
                21 => &self.lod_debug_pipeline,
                _ => &self.draw_pipeline,
            };
            rpass.set_pipeline(active_pipeline);
            rpass.set_bind_group(0, draw_bg0, &[]);
            rpass.set_bind_group(1, draw_bg1, &[]);
            rpass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
            rpass.set_index_buffer(
                main_scene.mesh_buffers.indices.slice(..),
                wgpu::IndexFormat::Uint32,
            );
            for i in 0..meshlet_count {
                rpass.draw_indexed_indirect(&self.indirect_buf, i as u64 * 20);
            }
        }

        Ok(())
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "main_scene", "vg", "hiz"]
    }
    fn writes(&self) -> &'static [&'static str] {
        &["gbuffer"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("gbuffer");
        builder.read("vg");
        builder.read("hiz");
    }

    fn debug_views(&self) -> &'static [DebugViewDescriptor] {
        static VIEWS: &[DebugViewDescriptor] = &[
            DebugViewDescriptor {
                name: "VG Mesh Triangles",
                debug_mode: 20,
                description: "Per-meshlet solid colour — visualises meshlet boundaries",
            },
            DebugViewDescriptor {
                name: "VG LOD Heatmap",
                debug_mode: 21,
                description: "Colour by LOD level: green=LOD0 through red=LOD7",
            },
        ];
        VIEWS
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn create_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];
    entries.extend(libhelio::webgpu_material_layout_entries());
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("VG Material BGL"),
        entries: &entries,
    })
}
