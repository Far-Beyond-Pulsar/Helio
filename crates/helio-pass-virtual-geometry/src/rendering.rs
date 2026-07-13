use std::num::NonZeroU32;

use crate::{
    CullUniforms, InstanceCullData, LodQuality, VgGlobals, INITIAL_INSTANCES,
    INITIAL_MESHLETS, INITIAL_OBJECTS, MAX_TEXTURES,
};
use helio_core::graph::ResourceBuilder;
use helio_core::{
    DebugViewDescriptor, GpuInstanceData, PassContext, PrepareContext, RenderPass,
    Result as HelioResult,
};
use libhelio::VG_CULL_MESHLETS_PER_WORK_ITEM;

// ═══════════════════════════════════════════════════════════════════════════════
// VirtualGeometryPass
// ═══════════════════════════════════════════════════════════════════════════════

pub struct VirtualGeometryPass {
    pub(crate) select_pipeline: wgpu::ComputePipeline,
    pub(crate) cull_pipeline: wgpu::ComputePipeline,
    pub(crate) cull_bgl: wgpu::BindGroupLayout,
    pub(crate) cull_bind_group: Option<wgpu::BindGroup>,
    pub(crate) cull_bind_group_hiz_key: Option<(usize, usize)>,
    pub(crate) cull_buf: wgpu::Buffer,
    pub(crate) draw_pipeline: wgpu::RenderPipeline,
    pub(crate) debug_draw_pipeline: wgpu::RenderPipeline,
    pub(crate) lod_debug_pipeline: wgpu::RenderPipeline,
    pub(crate) draw_bgl_0: wgpu::BindGroupLayout,
    pub(crate) draw_bgl_1: wgpu::BindGroupLayout,
    pub(crate) draw_bg_0: Option<wgpu::BindGroup>,
    pub(crate) draw_bg_1: Option<wgpu::BindGroup>,
    pub(crate) bg1_version: Option<u64>,
    pub(crate) globals_buf: wgpu::Buffer,
    pub(crate) meshlet_buf: wgpu::Buffer,
    pub(crate) object_buf: wgpu::Buffer,
    pub(crate) instance_buf: wgpu::Buffer,
    pub(crate) instance_cull_buf: wgpu::Buffer,
    pub(crate) instance_cull_scratch: Vec<InstanceCullData>,
    pub(crate) work_item_buf: wgpu::Buffer,
    pub(crate) indirect_buf: wgpu::Buffer,
    pub(crate) draw_metadata_buf: wgpu::Buffer,
    pub(crate) draw_count_buf: wgpu::Buffer,
    pub(crate) use_count_indirect: bool,
    pub debug_mode: u32,
    pub lod_quality: LodQuality,
    pub(crate) last_version: u64,
    pub(crate) last_instance_version: u64,
    pub(crate) last_meshlet_count: u32,
    pub(crate) last_object_count: u32,
    pub(crate) last_work_item_count: u32,
    pub(crate) last_max_draw_count: u32,
    pub(crate) object_dispatch_width: u32,
    pub(crate) work_dispatch_width: u32,
}

impl VirtualGeometryPass {
    pub fn new(device: &wgpu::Device, camera_buf: &wgpu::Buffer) -> Self {
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vg_cull.wgsl").into()),
        });
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG GBuffer Shader"),
            source: wgpu::ShaderSource::Wgsl({
                let s = include_str!("../shaders/vg_gbuffer.wgsl")
                    .replace(
                        "binding_array<texture_2d<f32>, 256>",
                        &format!("binding_array<texture_2d<f32>, {MAX_TEXTURES}>"),
                    )
                    .replace(
                        "binding_array<sampler, 256>",
                        &format!("binding_array<sampler, {MAX_TEXTURES}>"),
                    );
                #[cfg(target_arch = "wasm32")]
                let s = s
                    .replace(&format!("binding_array<sampler, {MAX_TEXTURES}>"), "sampler")
                    .replace("scene_samplers[slot.texture_index]", "scene_samplers")
                    .replace(&format!("binding_array<texture_2d<f32>, {MAX_TEXTURES}>"), "texture_2d<f32>")
                    .replace("scene_textures[slot.texture_index]", "scene_textures")
                    .replace(
                        "return textureSample(scene_textures, scene_samplers, uv);",
                        "return textureSampleLevel(scene_textures, scene_samplers, uv, 0.0);",
                    );
                s.into()
            }),
        });

        let meshlet_buf = Self::make_meshlet_buf(device, INITIAL_MESHLETS);
        let object_buf = Self::make_object_buf(device, INITIAL_OBJECTS);
        let instance_buf = Self::make_instance_buf(device, INITIAL_INSTANCES);
        let instance_cull_buf = Self::make_instance_cull_buf(device, INITIAL_INSTANCES);
        let work_item_buf = Self::make_work_item_buf(device, INITIAL_OBJECTS);
        let indirect_buf = Self::make_indirect_buf(device, INITIAL_MESHLETS);
        let draw_metadata_buf = Self::make_draw_metadata_buf(device, INITIAL_MESHLETS);
        let draw_count_buf = Self::make_draw_count_buf(device);
        let use_count_indirect = device
            .features()
            .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT);

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
                // Camera and cull uniforms.
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
                // Immutable shared meshlet descriptors.
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
                // Per-object LOD ranges/bounds plus per-frame selected-LOD scratch.
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Instance transforms/materials.
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Compacted indirect commands, parallel draw metadata, and count.
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
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
        let select_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VG Object Select Pipeline"),
            layout: Some(&cull_pipeline_layout),
            module: &cull_shader,
            entry_point: Some("cs_select_objects"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VG Meshlet Cull Pipeline"),
            layout: Some(&cull_pipeline_layout),
            module: &cull_shader,
            entry_point: Some("cs_cull_meshlets"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: instance_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: draw_metadata_buf.as_entire_binding() },
            ],
        }));

        let draw_bgl_1 = create_material_bgl(device);

        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Draw PL"),
            bind_group_layouts: &[Some(&draw_bgl_0), Some(&draw_bgl_1)],
            immediate_size: 0,
        });
        let vg_vertex_buffers = &[wgpu::VertexBufferLayout {
            array_stride: 40,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0,  shader_location: 0 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32,   offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2,  offset: 16, shader_location: 2 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,     offset: 32, shader_location: 3 },
                wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32,     offset: 36, shader_location: 4 },
            ],
        }];
        let gbuffer_targets = &[
            Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm,   blend: None, write_mask: wgpu::ColorWrites::ALL }),
            Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float,  blend: None, write_mask: wgpu::ColorWrites::ALL }),
            Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm,   blend: None, write_mask: wgpu::ColorWrites::ALL }),
            Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float,  blend: None, write_mask: wgpu::ColorWrites::ALL }),
            Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rg16Float,    blend: None, write_mask: wgpu::ColorWrites::ALL }),
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

        let debug_draw_pipeline =
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
            });

        let lod_debug_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
            select_pipeline,
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
            object_buf,
            instance_buf,
            instance_cull_buf,
            instance_cull_scratch: Vec::with_capacity(INITIAL_INSTANCES as usize),
            work_item_buf,
            indirect_buf,
            draw_metadata_buf,
            draw_count_buf,
            use_count_indirect,
            debug_mode: 0,
            lod_quality: LodQuality::default(),
            last_version: u64::MAX,
            last_instance_version: u64::MAX,
            last_meshlet_count: 0,
            last_object_count: 0,
            last_work_item_count: 0,
            last_max_draw_count: 0,
            object_dispatch_width: 1,
            work_dispatch_width: 1,
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

    fn make_object_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Object Buffer"),
            size: capacity * 128,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_instance_cull_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Instance Cull Buffer"),
            size: capacity * std::mem::size_of::<InstanceCullData>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_work_item_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Work Item Buffer"),
            size: capacity * std::mem::size_of::<libhelio::GpuVgWorkItem>() as u64,
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

    fn make_draw_metadata_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Draw Metadata Buffer"),
            size: capacity * std::mem::size_of::<libhelio::GpuVgDraw>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }

    fn make_draw_count_buf(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Draw And Overflow Counters"),
            // draw_count at byte 0 remains directly consumable by
            // multi_draw_indexed_indirect_count; overflow_count is at byte 4.
            size: 8,
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
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.globals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.instance_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.draw_metadata_buf.as_entire_binding() },
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
                grew = true;
            }
            let object_capacity = self.object_buf.size() / 128;
            if (vg.object_count as u64) > object_capacity {
                self.object_buf = Self::make_object_buf(ctx.device, vg.object_count as u64 * 2);
                grew = true;
            }
            let instance_capacity = self.instance_buf.size() / 144;
            if (vg.object_count as u64) > instance_capacity {
                self.instance_buf = Self::make_instance_buf(ctx.device, vg.object_count as u64 * 2);
                self.instance_cull_buf = Self::make_instance_cull_buf(ctx.device, vg.object_count as u64 * 2);
                grew = true;
            }
            let work_item_capacity = self.work_item_buf.size()
                / std::mem::size_of::<libhelio::GpuVgWorkItem>() as u64;
            if (vg.work_item_count as u64) > work_item_capacity {
                self.work_item_buf =
                    Self::make_work_item_buf(ctx.device, vg.work_item_count as u64 * 2);
                grew = true;
            }
            let indirect_capacity = self.indirect_buf.size() / 20;
            if (vg.max_draw_count as u64) > indirect_capacity {
                let new_capacity = vg.max_draw_count as u64 * 2;
                self.indirect_buf = Self::make_indirect_buf(ctx.device, new_capacity);
                self.draw_metadata_buf =
                    Self::make_draw_metadata_buf(ctx.device, new_capacity);
                grew = true;
            }

            if grew {
                self.rebuild_owned_bind_groups(ctx.device, camera_buf);
            }

            ctx.write_buffer(&self.meshlet_buf, 0, vg.meshlets);
            ctx.write_buffer(&self.object_buf, 0, vg.objects);
            ctx.write_buffer(&self.instance_buf, 0, vg.instances);
            ctx.write_buffer(&self.work_item_buf, 0, vg.work_items);

            let instances: &[GpuInstanceData] = bytemuck::cast_slice(vg.instances);
            self.instance_cull_scratch.clear();
            self.instance_cull_scratch.extend(
                instances.iter().map(InstanceCullData::from_instance),
            );
            ctx.write_buffer(
                &self.instance_cull_buf,
                0,
                bytemuck::cast_slice(&self.instance_cull_scratch),
            );

            self.last_version = vg.buffer_version;
            self.last_instance_version = vg.instance_version;
            self.last_meshlet_count = vg.meshlet_count;
            self.last_object_count = vg.object_count;
            self.last_work_item_count = vg.work_item_count;
            self.last_max_draw_count = vg.max_draw_count;
        } else if vg.instance_version != self.last_instance_version {
            let start = vg.instance_dirty_start as usize;
            let count = vg.instance_dirty_count as usize;
            let end = start
                .checked_add(count)
                .expect("virtual geometry dirty instance range overflow");
            let instances: &[GpuInstanceData] = bytemuck::cast_slice(vg.instances);
            assert!(
                count > 0 && end <= instances.len(),
                "virtual geometry published an invalid dirty instance range"
            );

            let instance_offset = start as u64
                * std::mem::size_of::<GpuInstanceData>() as u64;
            ctx.write_buffer(
                &self.instance_buf,
                instance_offset,
                bytemuck::cast_slice(&instances[start..end]),
            );

            self.instance_cull_scratch.clear();
            self.instance_cull_scratch.extend(
                instances[start..end]
                    .iter()
                    .map(InstanceCullData::from_instance),
            );
            let cull_offset = start as u64
                * std::mem::size_of::<InstanceCullData>() as u64;
            ctx.write_buffer(
                &self.instance_cull_buf,
                cull_offset,
                bytemuck::cast_slice(&self.instance_cull_scratch),
            );
            self.last_instance_version = vg.instance_version;
        }

        if self.last_object_count == 0
            || self.last_work_item_count == 0
            || self.last_max_draw_count == 0
        {
            return Ok(());
        }

        let max_dim = ctx.width.max(ctx.height);
        let hiz_mip_count = (u32::BITS - max_dim.leading_zeros()).max(1);
        let max_dispatch = ctx.device.limits().max_compute_workgroups_per_dimension;
        let object_workgroups = self
            .last_object_count
            .div_ceil(VG_CULL_MESHLETS_PER_WORK_ITEM);
        self.object_dispatch_width = object_workgroups.min(max_dispatch).max(1);
        let object_dispatch_height = object_workgroups.div_ceil(self.object_dispatch_width);
        assert!(
            object_dispatch_height <= max_dispatch,
            "virtual geometry object dispatch exceeds the device's 2D workgroup grid"
        );
        self.work_dispatch_width = self.last_work_item_count.min(max_dispatch).max(1);
        let work_dispatch_height = self
            .last_work_item_count
            .div_ceil(self.work_dispatch_width);
        assert!(
            work_dispatch_height <= max_dispatch,
            "virtual geometry meshlet-work dispatch exceeds the device's 2D workgroup grid"
        );
        let cull_uni = CullUniforms {
            object_count: self.last_object_count,
            screen_width:  ctx.width,
            screen_height: ctx.height,
            hiz_mip_count,
            draw_capacity: self.last_max_draw_count,
            lod_error_threshold_px: self.lod_quality.max_error_pixels(),
            object_dispatch_width: self.object_dispatch_width,
            work_item_count: self.last_work_item_count,
            work_dispatch_width: self.work_dispatch_width,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.write_buffer(&self.cull_buf, 0, bytemuck::bytes_of(&cull_uni));

        let Some(main_scene) = ctx.frame_resources.main_scene.read("VirtualGeometry") else {
            return Ok(());
        };
        if self.draw_bg_1.is_none()
            || self.bg1_version != Some(main_scene.material_textures.version)
        {
            self.draw_bg_1 = Some(
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("VG Draw BG1"),
                    layout: &self.draw_bgl_1,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.materials.buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: main_scene.material_textures.material_textures.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            #[cfg(not(target_arch = "wasm32"))]
                            resource: wgpu::BindingResource::TextureViewArray(main_scene.material_textures.texture_views),
                            #[cfg(target_arch = "wasm32")]
                            resource: wgpu::BindingResource::TextureView(main_scene.material_textures.texture_views.first().copied().expect("scene must have at least one texture view")),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            #[cfg(not(target_arch = "wasm32"))]
                            resource: wgpu::BindingResource::SamplerArray(main_scene.material_textures.samplers),
                            #[cfg(target_arch = "wasm32")]
                            resource: wgpu::BindingResource::Sampler(main_scene.material_textures.samplers.first().copied().expect("scene must have at least one sampler")),
                        },
                    ],
                }),
            );
            self.bg1_version = Some(main_scene.material_textures.version);
        }

        let globals = VgGlobals {
            frame: ctx.frame_num as u32,
            delta_time: 0.016,
            light_count: ctx.scene.lights.len() as u32,
            ambient_intensity: main_scene.ambient_intensity,
            ambient_color: [main_scene.ambient_color[0], main_scene.ambient_color[1], main_scene.ambient_color[2], 0.0],
            rc_world_min: [main_scene.rc_world_min[0], main_scene.rc_world_min[1], main_scene.rc_world_min[2], 0.0],
            rc_world_max: [main_scene.rc_world_max[0], main_scene.rc_world_max[1], main_scene.rc_world_max[2], 0.0],
            csm_splits: [5.0, 20.0, 60.0, 200.0],
            debug_mode: self.debug_mode,
            _pad0: 0, _pad1: 0, _pad2: 0,
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
        let lightmap_uv = resources.gbuffer_lightmap_uv.read("VirtualGeometry")?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = Box::leak(Box::new([
            Some(wgpu::RenderPassColorAttachment { view: gbuffer.albedo,   resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: gbuffer.normal,   resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: gbuffer.orm,      resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: gbuffer.emissive, resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
            Some(wgpu::RenderPassColorAttachment { view: lightmap_uv,      resolve_target: None, depth_slice: None, ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store } }),
        ]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("VG GBuffer"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.last_object_count == 0
            || self.last_work_item_count == 0
            || self.last_max_draw_count == 0
            || ctx.resources.vg.is_none()
        {
            return Ok(());
        }

        let hiz_view = ctx.resources.hiz.as_ref()
            .expect("VirtualGeometry: 'hiz' view not routed by graph");
        let hiz_sampler = ctx.resources.hiz_sampler.as_ref()
            .expect("VirtualGeometry: 'hiz_sampler' not available");
        let hiz_key = (hiz_view as *const _ as usize, hiz_sampler as *const _ as usize);
        if self.cull_bind_group.is_none() || self.cull_bind_group_hiz_key != Some(hiz_key) {
            self.cull_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VG Cull BG"),
                layout: &self.cull_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.camera.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.cull_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: self.meshlet_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.object_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: self.instance_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.indirect_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: self.draw_metadata_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: self.draw_count_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(hiz_view) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(hiz_sampler) },
                    wgpu::BindGroupEntry { binding: 10, resource: self.instance_cull_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: self.work_item_buf.as_entire_binding() },
                ],
            }));
            self.cull_bind_group_hiz_key = Some(hiz_key);
        }

        let Some(cull_bg) = self.cull_bind_group.as_ref() else { return Ok(()); };
        let Some(draw_bg0) = self.draw_bg_0.as_ref() else { return Ok(()); };
        let Some(draw_bg1) = self.draw_bg_1.as_ref() else { return Ok(()); };
        let Some(main_scene) = ctx.resources.main_scene.read("VirtualGeometry") else { return Ok(()); };

        let max_draw_count = self.last_max_draw_count;

        unsafe { &mut *ctx.compute_encoder_ptr }.clear_buffer(&self.draw_count_buf, 0, None);
        if !self.use_count_indirect {
            unsafe { &mut *ctx.compute_encoder_ptr }.clear_buffer(&self.indirect_buf, 0, None);
        }

        {
            let mut cpass = unsafe { &mut *ctx.compute_encoder_ptr }
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("VG Object Select"),
                    timestamp_writes: None,
                });
            cpass.set_pipeline(&self.select_pipeline);
            cpass.set_bind_group(0, cull_bg, &[]);
            let object_workgroups = self
                .last_object_count
                .div_ceil(VG_CULL_MESHLETS_PER_WORK_ITEM);
            cpass.dispatch_workgroups(
                self.object_dispatch_width,
                object_workgroups.div_ceil(self.object_dispatch_width),
                1,
            );
        }

        {
            let mut cpass = unsafe { &mut *ctx.compute_encoder_ptr }
                .begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("VG Meshlet Cull"),
                    timestamp_writes: None,
                });
            cpass.set_pipeline(&self.cull_pipeline);
            cpass.set_bind_group(0, cull_bg, &[]);
            cpass.dispatch_workgroups(
                self.work_dispatch_width,
                self.last_work_item_count.div_ceil(self.work_dispatch_width),
                1,
            );
        }

        {
            let rpass = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };

            let active_pipeline = match self.debug_mode {
                20 => &self.debug_draw_pipeline,
                21 => &self.lod_debug_pipeline,
                _ => &self.draw_pipeline,
            };
            rpass.set_pipeline(active_pipeline);
            rpass.set_bind_group(0, draw_bg0, &[]);
            rpass.set_bind_group(1, draw_bg1, &[]);
            rpass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
            rpass.set_index_buffer(main_scene.mesh_buffers.indices.slice(..), wgpu::IndexFormat::Uint32);
            if self.use_count_indirect {
                rpass.multi_draw_indexed_indirect_count(&self.indirect_buf, 0, &self.draw_count_buf, 0, max_draw_count);
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                rpass.multi_draw_indexed_indirect(&self.indirect_buf, 0, max_draw_count);
                #[cfg(target_arch = "wasm32")]
                for i in 0..max_draw_count {
                    rpass.draw_indexed_indirect(&self.indirect_buf, i as u64 * 20);
                }
            }
        }

        Ok(())
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "main_scene", "vg", "hiz"]
    }
    fn writes(&self) -> &'static [&'static str] {
        &["gbuffer", "gbuffer_lightmap_uv"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("gbuffer");
        builder.read("vg");
        builder.read("hiz");
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
    }

    fn debug_views(&self) -> &'static [DebugViewDescriptor] {
        static VIEWS: &[DebugViewDescriptor] = &[
            DebugViewDescriptor { name: "VG Meshlets", debug_mode: 20, description: "One solid colour per meshlet — visualises cluster boundaries" },
            DebugViewDescriptor { name: "VG LOD Heatmap",    debug_mode: 21, description: "One flat colour per object LOD; green=LOD0 through magenta=LOD7" },
        ];
        VIEWS
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn create_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let count = NonZeroU32::new(MAX_TEXTURES as u32).expect("non-zero");
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("VG Material BGL"),
        entries: &[
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
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                #[cfg(not(target_arch = "wasm32"))]
                count: Some(count),
                #[cfg(target_arch = "wasm32")]
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                #[cfg(not(target_arch = "wasm32"))]
                count: Some(count),
                #[cfg(target_arch = "wasm32")]
                count: None,
            },
        ],
    })
}
