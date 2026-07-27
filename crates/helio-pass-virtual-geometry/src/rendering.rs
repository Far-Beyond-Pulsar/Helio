#[cfg(not(target_arch = "wasm32"))]
use std::num::NonZeroU32;

use crate::{
    CullUniforms, InstanceCullData, LodQuality, VgGlobals, VirtualGeometryBudget,
    VirtualGeometryDebugStats, DRAW_COUNTER_BYTES, INITIAL_INSTANCES, INITIAL_MESHLETS,
    INITIAL_OBJECTS, MAX_TEXTURES, MAX_MESHLETS_PER_TILE, TILE_SIZE_X, TILE_SIZE_Y,
};
use helio_core::graph::ResourceBuilder;
use helio_core::{
    DebugViewDescriptor, GpuInstanceData, PassContext, PrepareContext, RenderPass,
    Result as HelioResult,
};
use libhelio::{GpuVgObject, GpuVgWorkItem, VG_CULL_MESHLETS_PER_WORK_ITEM};

// ═══════════════════════════════════════════════════════════════════════════════
// VirtualGeometryPass
// ═══════════════════════════════════════════════════════════════════════════════

enum DebugReadbackState {
    Idle,
    CopySubmitted,
    Mapping(std::sync::Arc<std::sync::Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>),
}

pub struct VirtualGeometryPass {
    pub(crate) select_pipeline: wgpu::ComputePipeline,
    pub(crate) cull_pipeline: wgpu::ComputePipeline,
    pub(crate) cull_bgl: wgpu::BindGroupLayout,
    pub(crate) cull_bind_group: Option<wgpu::BindGroup>,
    pub(crate) cull_bind_group_hiz_key: Option<(usize, usize)>,
    pub(crate) cull_buf: wgpu::Buffer,
    // Visibility pass pipelines (depth-only, writes SV_Position, no color targets)
    pub(crate) visibility_opaque_pipeline: wgpu::RenderPipeline,
    pub(crate) visibility_alpha_pipeline: wgpu::RenderPipeline,
    // Shading pass pipelines (full GBuffer, depth test Equal, depth write disabled)
    pub(crate) shading_opaque_pipeline: wgpu::RenderPipeline,
    pub(crate) shading_alpha_pipeline: wgpu::RenderPipeline,
    pub(crate) debug_draw_pipeline: wgpu::RenderPipeline,
    pub(crate) lod_debug_pipeline: wgpu::RenderPipeline,
    pub(crate) draw_bgl_0: wgpu::BindGroupLayout,
    pub(crate) draw_bgl_1: wgpu::BindGroupLayout,
    pub(crate) draw_bg_0: Option<wgpu::BindGroup>,
    pub(crate) draw_bg_1: Option<wgpu::BindGroup>,
    pub(crate) bg1_version: Option<u64>,
    pub(crate) globals_buf: wgpu::Buffer,
    pub(crate) meshlet_buf: wgpu::Buffer,
    pub(crate) meshlet_vertex_buf: wgpu::Buffer,
    pub(crate) meshlet_index_buf: wgpu::Buffer,
    pub(crate) object_buf: wgpu::Buffer,
    pub(crate) instance_buf: wgpu::Buffer,
    pub(crate) instance_cull_buf: wgpu::Buffer,
    pub(crate) instance_cull_scratch: Vec<InstanceCullData>,
    pub(crate) work_item_buf: wgpu::Buffer,
    /// Per-meshlet atomic emit claim flags (cleared each frame). Ensures each
    /// meshlet is drawn at most once when multiple leaves fan in to one parent.
    pub(crate) meshlet_emit_flags_buf: wgpu::Buffer,
    pub(crate) indirect_buf: wgpu::Buffer,
    pub(crate) draw_metadata_buf: wgpu::Buffer,
    pub(crate) draw_count_buf: wgpu::Buffer,
    pub(crate) debug_readback_buf: wgpu::Buffer,
    debug_readback_state: DebugReadbackState,
    debug_stats: VirtualGeometryDebugStats,
    pub(crate) publication_limit: u32,
    pub(crate) use_count_indirect: bool,
    pub debug_mode: u32,
    pub lod_quality: LodQuality,
    pub(crate) last_version: u64,
    pub(crate) last_instance_version: u64,
    pub(crate) last_meshlet_count: u32,
    pub(crate) last_object_count: u32,
    pub(crate) last_work_item_count: u32,
    pub(crate) last_emit_flag_count: u32,
    pub(crate) object_dispatch_width: u32,
    pub(crate) work_dispatch_width: u32,
    // ── Software rasterizer (Phase 5) ─────────────────────────────────────
    pub use_sw_rasterizer: bool,
    last_screen_width: u32,
    last_screen_height: u32,
    // Binning pass
    binning_pipeline: wgpu::ComputePipeline,
    binning_bgl: wgpu::BindGroupLayout,
    binning_bg: Option<wgpu::BindGroup>,
    // Rasterize pass
    rasterize_pipeline: wgpu::ComputePipeline,
    rasterize_bgl: wgpu::BindGroupLayout,
    rasterize_bg: Option<wgpu::BindGroup>,
    // Shade pass
    shade_pipeline: wgpu::ComputePipeline,
    shade_bgl_0: wgpu::BindGroupLayout,
    shade_bgl_1: wgpu::BindGroupLayout,
    shade_bgl_2: wgpu::BindGroupLayout,
    shade_bg_0: Option<wgpu::BindGroup>,
    shade_bg_1: Option<wgpu::BindGroup>,
    shade_bg_2: Option<wgpu::BindGroup>,
    // SW rasterizer buffers
    visible_meshlet_ids_buf: wgpu::Buffer,
    visible_instance_ids_buf: wgpu::Buffer,
    visible_meshlet_count_buf: wgpu::Buffer,
    tile_counts_buf: wgpu::Buffer,
    tile_meshlet_ids_buf: wgpu::Buffer,
    tile_instance_ids_buf: wgpu::Buffer,
    visibility_depth_buf: wgpu::Buffer,
    visibility_data_buf: wgpu::Buffer,
    visibility_instance_buf: wgpu::Buffer,
}

impl VirtualGeometryPass {
    pub fn new(device: &wgpu::Device, camera_buf: &wgpu::Buffer) -> Self {
        Self::new_with_budget(device, camera_buf, VirtualGeometryBudget::default())
    }

    pub fn new_with_budget(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        budget: VirtualGeometryBudget,
    ) -> Self {
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vg_cull.wgsl").into()),
        });
        let draw_shader_source = {
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
            let s = libhelio::shader::apply_webgpu_material_bindings(&s, MAX_TEXTURES);
            s
        };
        let draw_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG GBuffer Shader"),
            source: wgpu::ShaderSource::Wgsl(draw_shader_source.clone().into()),
        });

        let meshlet_buf = Self::make_meshlet_buf(device, INITIAL_MESHLETS);
        let meshlet_vertex_buf = Self::make_meshlet_vertex_buf(device, INITIAL_MESHLETS * 64);
        let meshlet_index_buf = Self::make_meshlet_index_buf(device, INITIAL_MESHLETS * 64 * 3);
        let object_buf = Self::make_object_buf(device, INITIAL_OBJECTS);
        let instance_buf = Self::make_instance_buf(device, INITIAL_INSTANCES);
        let instance_cull_buf = Self::make_instance_cull_buf(device, INITIAL_INSTANCES);
        let work_item_buf = Self::make_work_item_buf(device, INITIAL_OBJECTS);
        let meshlet_emit_flags_buf = Self::make_meshlet_emit_flags_buf(device, INITIAL_MESHLETS);
        let initial_publication_capacity =
            INITIAL_MESHLETS.min(u64::from(budget.max_published_meshlets()));
        let indirect_buf = Self::make_indirect_buf(device, initial_publication_capacity);
        let draw_metadata_buf = Self::make_draw_metadata_buf(device, initial_publication_capacity);
        let draw_count_buf = Self::make_draw_count_buf(device);
        let debug_readback_buf = Self::make_debug_readback_buf(device);
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
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
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
                // Visible meshlet output (SW rasterizer)
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Per-meshlet global emit claim flags (atomic u32, cleared each frame).
                wgpu::BindGroupLayoutEntry {
                    binding: 15,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: draw_metadata_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: meshlet_vertex_buf.as_entire_binding(),
                },
            ],
        }));

        let draw_bgl_1 = create_material_bgl(device);

        let draw_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Draw PL"),
            bind_group_layouts: &[Some(&draw_bgl_0), Some(&draw_bgl_1)],
            immediate_size: 0,
        });
        // No vertex buffers — vertex data is read from the meshlet_vertices storage buffer.
        let vg_vertex_buffers = &[];
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
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba16Float,
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

        let visibility_depth = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });
        let shading_depth = Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Equal),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        });

        let make_pipeline = |label: &'static str,
                             entry: &'static str,
                             constants: &[(&str, f64)],
                             targets: &[Option<wgpu::ColorTargetState>],
                             depth: &Option<wgpu::DepthStencilState>| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&draw_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &draw_shader,
                    entry_point: Some("vs_main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: vg_vertex_buffers,
                },
                fragment: Some(wgpu::FragmentState {
                    module: &draw_shader,
                    entry_point: Some(entry),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants,
                        ..Default::default()
                    },
                    targets,
                }),
                primitive: draw_primitive,
                depth_stencil: depth.clone(),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // Visibility pipelines: depth-only, no color targets
        let visibility_opaque_pipeline = make_pipeline(
            "VG Visibility Opaque Pipeline",
            "fs_visibility",
            &[],
            &[],  // no color attachments
            &visibility_depth,
        );
        let visibility_alpha_pipeline = make_pipeline(
            "VG Visibility Alpha Pipeline",
            "fs_visibility",
            &[("has_alpha_test", 1.0)],
            &[],  // no color attachments
            &visibility_depth,
        );

        // Shading pipelines: full GBuffer, depth test Equal
        let shading_opaque_pipeline = make_pipeline(
            "VG Shading Opaque Pipeline",
            "fs_main",
            &[],
            gbuffer_targets,
            &shading_depth,
        );
        let shading_alpha_pipeline = make_pipeline(
            "VG Shading Alpha Pipeline",
            "fs_main",
            &[("has_alpha_test", 1.0)],
            gbuffer_targets,
            &shading_depth,
        );



        let debug_draw_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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

        // ── Software rasterizer compute pipelines (Phase 5) ─────────────
        let binning_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG Binning Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vg_binning.wgsl").into()),
        });
        let rasterize_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG Rasterize Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vg_rasterize.wgsl").into()),
        });
        let shade_shader_source = {
            let s = include_str!("../shaders/vg_shade.wgsl")
                .replace(
                    "binding_array<texture_2d<f32>, 256>",
                    &format!("binding_array<texture_2d<f32>, {MAX_TEXTURES}>"),
                )
                .replace(
                    "binding_array<sampler, 256>",
                    &format!("binding_array<sampler, {MAX_TEXTURES}>"),
                );
            #[cfg(target_arch = "wasm32")]
            let s = libhelio::shader::apply_webgpu_material_bindings(&s, MAX_TEXTURES);
            s
        };
        let shade_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("VG Shade Shader"),
            source: wgpu::ShaderSource::Wgsl(shade_shader_source.clone().into()),
        });

        // Binning BGL
        let binning_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VG Binning BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let binning_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Binning PL"),
            bind_group_layouts: &[Some(&binning_bgl)],
            immediate_size: 0,
        });
        let binning_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VG Binning Pipeline"),
            layout: Some(&binning_pipeline_layout),
            module: &binning_shader,
            entry_point: Some("cs_binning"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Rasterize BGL
        let rasterize_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VG Rasterize BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 9, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 10, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 11, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let rasterize_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Rasterize PL"),
            bind_group_layouts: &[Some(&rasterize_bgl)],
            immediate_size: 0,
        });
        let rasterize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VG Rasterize Pipeline"),
            layout: Some(&rasterize_pipeline_layout),
            module: &rasterize_shader,
            entry_point: Some("cs_rasterize"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Shade BGL 0 (visibility + meshlet + camera data)
        let shade_bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VG Shade BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 7, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 8, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });
        let shade_bgl_1 = create_material_bgl(device);
        // Shade BGL 2 (GBuffer storage textures for write)
        let shade_bgl_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VG Shade BGL2"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba8Unorm, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 6, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
            ],
        });
        let shade_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VG Shade PL"),
            bind_group_layouts: &[Some(&shade_bgl_0), Some(&shade_bgl_1), Some(&shade_bgl_2)],
            immediate_size: 0,
        });
        let shade_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VG Shade Pipeline"),
            layout: Some(&shade_pipeline_layout),
            module: &shade_shader,
            entry_point: Some("cs_shade"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── SW rasterizer buffer sizing ─────────────────────────────────
        let max_visible = (INITIAL_MESHLETS * 2) as u32;
        let tile_grid_x = (1920u32 + TILE_SIZE_X - 1) / TILE_SIZE_X;
        let tile_grid_y = (1080u32 + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
        let tile_count = tile_grid_x * tile_grid_y;

        let visible_meshlet_ids_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Visible Meshlet IDs"),
            size: max_visible as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visible_instance_ids_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Visible Instance IDs"),
            size: max_visible as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visible_meshlet_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Visible Meshlet Count"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let tile_counts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Tile Counts"),
            size: tile_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tile_data_count = tile_count * MAX_MESHLETS_PER_TILE;
        let tile_meshlet_ids_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Tile Meshlet IDs"),
            size: tile_data_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let tile_instance_ids_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Tile Instance IDs"),
            size: tile_data_count as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let initial_pixels = 1920u32 * 1080u32;
        let visibility_depth_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Visibility Depth"),
            size: initial_pixels as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visibility_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Visibility Data"),
            size: initial_pixels as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let visibility_instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Visibility Instance"),
            size: initial_pixels as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            select_pipeline,
            cull_pipeline,
            cull_bgl,
            cull_bind_group: None,
            cull_bind_group_hiz_key: None,
            cull_buf,
            visibility_opaque_pipeline,
            visibility_alpha_pipeline,
            shading_opaque_pipeline,
            shading_alpha_pipeline,
            debug_draw_pipeline,
            lod_debug_pipeline,
            draw_bgl_0,
            draw_bgl_1,
            draw_bg_0,
            draw_bg_1: None,
            bg1_version: None,
            globals_buf,
            meshlet_buf,
            meshlet_vertex_buf,
            meshlet_index_buf,
            object_buf,
            instance_buf,
            instance_cull_buf,
            instance_cull_scratch: Vec::with_capacity(INITIAL_INSTANCES as usize),
            work_item_buf,
            meshlet_emit_flags_buf,
            indirect_buf,
            draw_metadata_buf,
            draw_count_buf,
            debug_readback_buf,
            debug_readback_state: DebugReadbackState::Idle,
            debug_stats: VirtualGeometryDebugStats::default(),
            publication_limit: budget.max_published_meshlets(),
            use_count_indirect,
            debug_mode: 0,
            lod_quality: LodQuality::default(),
            last_version: u64::MAX,
            last_instance_version: u64::MAX,
            last_meshlet_count: 0,
            last_object_count: 0,
            last_work_item_count: 0,
            last_emit_flag_count: 0,
            object_dispatch_width: 1,
            work_dispatch_width: 1,
            // Software rasterizer fields
            use_sw_rasterizer: false,
            last_screen_width: 0,
            last_screen_height: 0,
            binning_pipeline,
            binning_bgl,
            binning_bg: None,
            rasterize_pipeline,
            rasterize_bgl,
            rasterize_bg: None,
            shade_pipeline,
            shade_bgl_0,
            shade_bgl_1,
            shade_bgl_2,
            shade_bg_0: None,
            shade_bg_1: None,
            shade_bg_2: None,
            visible_meshlet_ids_buf,
            visible_instance_ids_buf,
            visible_meshlet_count_buf,
            tile_counts_buf,
            tile_meshlet_ids_buf,
            tile_instance_ids_buf,
            visibility_depth_buf,
            visibility_data_buf,
            visibility_instance_buf,
        }
    }

    pub const fn publication_limit(&self) -> u32 {
        self.publication_limit
    }

    pub const fn debug_stats(&self) -> VirtualGeometryDebugStats {
        self.debug_stats
    }

    fn poll_debug_readback(&mut self, device: &wgpu::Device) {
        if matches!(self.debug_readback_state, DebugReadbackState::CopySubmitted) {
            let completion = std::sync::Arc::new(std::sync::Mutex::new(None));
            let callback_completion = std::sync::Arc::clone(&completion);
            self.debug_readback_buf
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    *callback_completion.lock().unwrap() = Some(result);
                });
            self.debug_readback_state = DebugReadbackState::Mapping(completion);
        }

        let DebugReadbackState::Mapping(completion) = &self.debug_readback_state else {
            return;
        };

        let _ = device.poll(wgpu::PollType::Poll);
        let result = completion.lock().unwrap().take();
        match result {
            Some(Ok(())) => {
                let mapped = self
                    .debug_readback_buf
                    .slice(..)
                    .get_mapped_range()
                    .expect("virtual geometry debug readback buffer should be mapped");
                let counters: &[u32] = bytemuck::cast_slice(&mapped);
                if let Some(stats) = VirtualGeometryDebugStats::from_counters(counters) {
                    self.debug_stats = stats;
                }
                drop(mapped);
                self.debug_readback_buf.unmap();
                self.debug_readback_state = DebugReadbackState::Idle;
            }
            Some(Err(error)) => {
                log::warn!("virtual geometry debug readback failed: {error}");
                self.debug_readback_state = DebugReadbackState::Idle;
            }
            None => {}
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

    fn make_meshlet_vertex_buf(device: &wgpu::Device, capacity_elements: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Meshlet Vertex Buffer"),
            size: capacity_elements * 48,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_meshlet_index_buf(device: &wgpu::Device, capacity_elements: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Meshlet Index Buffer"),
            size: capacity_elements * 2,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
            size: capacity * std::mem::size_of::<GpuVgObject>() as u64,
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

    fn make_meshlet_emit_flags_buf(device: &wgpu::Device, capacity: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Meshlet Emit Flags"),
            size: capacity.max(1) * 4,
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
            // multi_draw_indexed_indirect_count; the remaining counters are
            // optional debug telemetry and do not affect indirect execution.
            size: DRAW_COUNTER_BYTES,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn make_debug_readback_buf(device: &wgpu::Device) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Debug Counter Readback"),
            size: DRAW_COUNTER_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }

    fn make_sized_vis_ids_buf(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Visible Meshlet IDs"),
            size: (capacity as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    fn make_sized_vis_buf(device: &wgpu::Device, pixel_count: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VG Visibility Buffer"),
            size: pixel_count * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.draw_metadata_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.meshlet_vertex_buf.as_entire_binding(),
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
        self.poll_debug_readback(ctx.device);

        let Some(vg) = ctx.frame_resources.vg.get() else {
            return Ok(());
        };

        if vg.buffer_version != self.last_version {
            let camera_buf = ctx.scene.camera.buffer();
            let mut grew = false;
            let needed_draws = vg.meshlet_count as u64 * 2;

            if vg.meshlet_count * 2 > self.publication_limit {
                log::warn!(
                    "virtual geometry worst-case draw count {} exceeds the publication budget {}; excess visible meshlets will be counted and rejected",
                    vg.meshlet_count * 2,
                    self.publication_limit,
                );
            }

            let meshlet_capacity = self.meshlet_buf.size() / 64;
            if (vg.meshlet_count as u64) > meshlet_capacity {
                let new_cap = vg.meshlet_count as u64 * 2;
                self.meshlet_buf = Self::make_meshlet_buf(ctx.device, new_cap);
                grew = true;
            }
            let emit_flags_capacity = self.meshlet_emit_flags_buf.size() / 4;
            if (vg.emit_flag_count as u64) > emit_flags_capacity {
                self.meshlet_emit_flags_buf = Self::make_meshlet_emit_flags_buf(
                    ctx.device,
                    (vg.emit_flag_count as u64 * 2).max(1),
                );
                grew = true;
            }
            let meshlet_vertex_capacity = self.meshlet_vertex_buf.size() / 48;
            if (vg.meshlet_vertex_count as u64) > meshlet_vertex_capacity {
                self.meshlet_vertex_buf = Self::make_meshlet_vertex_buf(
                    ctx.device,
                    (vg.meshlet_vertex_count as u64 * 2).max(64),
                );
                grew = true;
            }
            let meshlet_index_capacity = self.meshlet_index_buf.size() / 2;
            if (vg.meshlet_index_count as u64) > meshlet_index_capacity {
                self.meshlet_index_buf = Self::make_meshlet_index_buf(
                    ctx.device,
                    (vg.meshlet_index_count as u64 * 2).max(64),
                );
                grew = true;
            }
            let object_stride = std::mem::size_of::<GpuVgObject>() as u64;
            let object_capacity = self.object_buf.size() / object_stride;
            if (vg.object_count as u64) > object_capacity {
                self.object_buf = Self::make_object_buf(ctx.device, vg.object_count as u64 * 2);
                grew = true;
            }
            let instance_capacity = self.instance_buf.size() / 144;
            if (vg.object_count as u64) > instance_capacity {
                self.instance_buf = Self::make_instance_buf(ctx.device, vg.object_count as u64 * 2);
                self.instance_cull_buf =
                    Self::make_instance_cull_buf(ctx.device, vg.object_count as u64 * 2);
                grew = true;
            }
            let work_item_capacity =
                self.work_item_buf.size() / std::mem::size_of::<libhelio::GpuVgWorkItem>() as u64;
            if (vg.work_item_count as u64) > work_item_capacity {
                self.work_item_buf =
                    Self::make_work_item_buf(ctx.device, vg.work_item_count as u64 * 2);
                grew = true;
            }
            let indirect_capacity = self.indirect_buf.size() / 20;
            if needed_draws > indirect_capacity {
                let new_capacity = (needed_draws * 2).max(65536);
                self.indirect_buf = Self::make_indirect_buf(ctx.device, new_capacity);
                self.draw_metadata_buf = Self::make_draw_metadata_buf(ctx.device, new_capacity);
                grew = true;
            }

            if grew {
                self.rebuild_owned_bind_groups(ctx.device, camera_buf);
            }

            ctx.write_buffer(&self.meshlet_buf, 0, vg.meshlets);
            ctx.write_buffer(&self.meshlet_vertex_buf, 0, vg.meshlet_vertices);
            ctx.write_buffer(&self.meshlet_index_buf, 0, vg.meshlet_indices);
            ctx.write_buffer(&self.object_buf, 0, vg.objects);
            ctx.write_buffer(&self.instance_buf, 0, vg.instances);

            let instances: &[GpuInstanceData] = bytemuck::cast_slice(vg.instances);
            let materials = ctx.scene.materials.as_slice();
            self.instance_cull_scratch.clear();
            self.instance_cull_scratch.extend(instances.iter().map(|inst| {
                let mat_flags = materials
                    .get(inst.material_id as usize)
                    .map(|m| m.flags)
                    .unwrap_or(0);
                InstanceCullData::from_instance(inst, mat_flags)
            }));
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
            self.last_emit_flag_count = vg.emit_flag_count;
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

            let instance_offset = start as u64 * std::mem::size_of::<GpuInstanceData>() as u64;
            ctx.write_buffer(
                &self.instance_buf,
                instance_offset,
                bytemuck::cast_slice(&instances[start..end]),
            );

            let materials = ctx.scene.materials.as_slice();
            self.instance_cull_scratch.clear();
            self.instance_cull_scratch.extend(
                instances[start..end]
                    .iter()
                    .map(|inst| {
                        let mat_flags = materials
                            .get(inst.material_id as usize)
                            .map(|m| m.flags)
                            .unwrap_or(0);
                        InstanceCullData::from_instance(inst, mat_flags)
                    }),
            );
            let cull_offset = start as u64 * std::mem::size_of::<InstanceCullData>() as u64;
            ctx.write_buffer(
                &self.instance_cull_buf,
                cull_offset,
                bytemuck::cast_slice(&self.instance_cull_scratch),
            );
            self.last_instance_version = vg.instance_version;
        }

        // Per-frame: sort work items by the nearest point on each object's
        // world-space bounding sphere so the cull shader processes near
        // meshlets first — draws are emitted in indirect buffer order, giving
        // approximate front-to-back execution and maximising early-Z kills.
        // Using the instance's stored world-space bounds (not the translation)
        // correctly handles large objects whose bounding sphere may be close
        // to the camera even when the instance centre is far away.
        {
            let cam_pos = ctx.scene.camera.position();
            let instances: &[GpuInstanceData] = bytemuck::cast_slice(vg.instances);
            let objects: &[GpuVgObject] = bytemuck::cast_slice(vg.objects);
            let work_items: &[GpuVgWorkItem] = bytemuck::cast_slice(vg.work_items);

            if !work_items.is_empty() && !objects.is_empty() && !instances.is_empty() {
                let mut sorted_work_items: Vec<GpuVgWorkItem> = work_items.to_vec();
                sorted_work_items.sort_by(|a, b| {
                    let obj_a = &objects[a.object_index as usize];
                    let inst_a = &instances[obj_a.instance_index as usize];
                    let obj_b = &objects[b.object_index as usize];
                    let inst_b = &instances[obj_b.instance_index as usize];

                    let dx_a = inst_a.bounds[0] - cam_pos[0];
                    let dy_a = inst_a.bounds[1] - cam_pos[1];
                    let dz_a = inst_a.bounds[2] - cam_pos[2];
                    let dist_a = (dx_a * dx_a + dy_a * dy_a + dz_a * dz_a).sqrt();
                    let near_a = (dist_a - inst_a.bounds[3]).max(0.0);

                    let dx_b = inst_b.bounds[0] - cam_pos[0];
                    let dy_b = inst_b.bounds[1] - cam_pos[1];
                    let dz_b = inst_b.bounds[2] - cam_pos[2];
                    let dist_b = (dx_b * dx_b + dy_b * dy_b + dz_b * dz_b).sqrt();
                    let near_b = (dist_b - inst_b.bounds[3]).max(0.0);

                    near_a
                        .partial_cmp(&near_b)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                ctx.write_buffer(
                    &self.work_item_buf,
                    0,
                    bytemuck::cast_slice(&sorted_work_items),
                );
            }
        }

        if self.last_object_count == 0 || self.last_work_item_count == 0 {
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
        let work_dispatch_height = self.last_work_item_count.div_ceil(self.work_dispatch_width);
        assert!(
            work_dispatch_height <= max_dispatch,
            "virtual geometry meshlet-work dispatch exceeds the device's 2D workgroup grid"
        );
        let cull_uni = CullUniforms {
            object_count: self.last_object_count,
            screen_width: ctx.width,
            screen_height: ctx.height,
            hiz_mip_count,
            draw_capacity: self.last_meshlet_count * 2,
            lod_error_threshold_px: self.lod_quality.max_error_pixels(),
            object_dispatch_width: self.object_dispatch_width,
            work_item_count: self.last_work_item_count,
            work_dispatch_width: self.work_dispatch_width,
            hiz_valid: (ctx.frame_num > 0) as u32,
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
            #[cfg(not(target_arch = "wasm32"))]
            {
                entries.push(wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureViewArray(
                        main_scene.material_textures.texture_views,
                    ),
                });
                entries.push(wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::SamplerArray(
                        main_scene.material_textures.samplers,
                    ),
                });
            }
            #[cfg(target_arch = "wasm32")]
            {
                for (index, view) in main_scene
                    .material_textures
                    .texture_views
                    .iter()
                    .enumerate()
                {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2 + index as u32,
                        resource: wgpu::BindingResource::TextureView(view),
                    });
                }
                for (index, sampler) in main_scene.material_textures.samplers.iter().enumerate() {
                    entries.push(wgpu::BindGroupEntry {
                        binding: 2 + MAX_TEXTURES as u32 + index as u32,
                        resource: wgpu::BindingResource::Sampler(sampler),
                    });
                }
            }
            self.draw_bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VG Draw BG1"),
                layout: &self.draw_bgl_1,
                entries: &entries,
            }));
            // Also update shade BG1 with the same materials
            self.shade_bg_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("VG Shade BG1"),
                layout: &self.shade_bgl_1,
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
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // The VG pass manages its own render passes (visibility + shading)
        // inside execute() via ctx.begin_render_pass().
        None
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.last_object_count == 0
            || self.last_work_item_count == 0
            || ctx.resources.vg.is_none()
        {
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
        let new_vis_count_capacity = (self.last_meshlet_count * 2).max(1);
        if self.visible_meshlet_ids_buf.size() < new_vis_count_capacity as u64 * 4
            || self.visible_meshlet_count_buf.size() < 4
        {
            self.visible_meshlet_ids_buf = Self::make_sized_vis_ids_buf(ctx.device, new_vis_count_capacity);
            self.visible_instance_ids_buf = Self::make_sized_vis_ids_buf(ctx.device, new_vis_count_capacity);
        }

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
                        resource: self.object_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.instance_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.indirect_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.draw_metadata_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.draw_count_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::TextureView(hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: wgpu::BindingResource::Sampler(hiz_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.instance_cull_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: self.work_item_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: self.visible_meshlet_ids_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 13,
                        resource: self.visible_instance_ids_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 14,
                        resource: self.visible_meshlet_count_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 15,
                        resource: self.meshlet_emit_flags_buf.as_entire_binding(),
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
        let max_draw_count = self.last_meshlet_count * 2;

        unsafe { &mut *ctx.compute_encoder_ptr }.clear_buffer(&self.draw_count_buf, 0, None);
        // Clear per-object emit claim flags so each (object, meshlet) can be claimed once.
        let emit_clear_bytes = (self.last_emit_flag_count as u64).saturating_mul(4);
        if emit_clear_bytes > 0 {
            unsafe { &mut *ctx.compute_encoder_ptr }.clear_buffer(
                &self.meshlet_emit_flags_buf,
                0,
                Some(emit_clear_bytes),
            );
        }
        if !self.use_count_indirect {
            unsafe { &mut *ctx.compute_encoder_ptr }.clear_buffer(&self.indirect_buf, 0, None);
        }

        {
            let mut cpass = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("VG Object Select"),
                    timestamp_writes: None,
                },
            );
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
            let mut cpass = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("VG Meshlet Cull"),
                    timestamp_writes: None,
                },
            );
            cpass.set_pipeline(&self.cull_pipeline);
            cpass.set_bind_group(0, cull_bg, &[]);
            cpass.dispatch_workgroups(
                self.work_dispatch_width,
                self.last_work_item_count.div_ceil(self.work_dispatch_width),
                1,
            );
        }

        // ── Software rasterizer path ─────────────────────────────────────
        if self.use_sw_rasterizer {
            let encoder = unsafe { &mut *ctx.compute_encoder_ptr };
            let sw_w = ctx.width;
            let sw_h = ctx.height;

            // Resize visibility buffers if screen dimensions changed
            if sw_w != self.last_screen_width || sw_h != self.last_screen_height {
                let pixels = sw_w as u64 * sw_h as u64;
                self.visibility_depth_buf = Self::make_sized_vis_buf(ctx.device, pixels);
                self.visibility_data_buf = Self::make_sized_vis_buf(ctx.device, pixels);
                self.visibility_instance_buf = Self::make_sized_vis_buf(ctx.device, pixels);

                let tile_grid_x = (sw_w + TILE_SIZE_X - 1) / TILE_SIZE_X;
                let tile_grid_y = (sw_h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
                let tile_count = tile_grid_x * tile_grid_y;
                self.tile_counts_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("VG Tile Counts"),
                    size: tile_count as u64 * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let tile_data_count = tile_count as u64 * MAX_MESHLETS_PER_TILE as u64;
                self.tile_meshlet_ids_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("VG Tile Meshlet IDs"),
                    size: tile_data_count * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                self.tile_instance_ids_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("VG Tile Instance IDs"),
                    size: tile_data_count * 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });

                // Rebuild bind groups that reference these buffers
                self.binning_bg = None;
                self.rasterize_bg = None;
                self.shade_bg_0 = None;

                self.last_screen_width = sw_w;
                self.last_screen_height = sw_h;
            }

            // Clear SW rasterizer buffers
            encoder.clear_buffer(&self.visible_meshlet_count_buf, 0, None);
            encoder.clear_buffer(&self.tile_counts_buf, 0, None);
            encoder.clear_buffer(&self.visibility_depth_buf, 0, None);
            encoder.clear_buffer(&self.visibility_data_buf, 0, None);
            encoder.clear_buffer(&self.visibility_instance_buf, 0, None);

            // Create bind groups if needed
            if self.binning_bg.is_none() {
                self.binning_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("VG Binning BG"),
                    layout: &self.binning_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.visible_meshlet_ids_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.visible_instance_ids_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.instance_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.meshlet_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: ctx.scene.camera.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: self.cull_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: self.tile_counts_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: self.tile_meshlet_ids_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: self.tile_instance_ids_buf.as_entire_binding() },
                    ],
                }));
            }
            if self.rasterize_bg.is_none() {
                self.rasterize_bg = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("VG Rasterize BG"),
                    layout: &self.rasterize_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.tile_counts_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.tile_meshlet_ids_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.tile_instance_ids_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.meshlet_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: self.meshlet_vertex_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: self.meshlet_index_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: self.visibility_depth_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: self.visibility_data_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: self.visibility_instance_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 9, resource: ctx.scene.camera.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 10, resource: self.cull_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 11, resource: self.instance_buf.as_entire_binding() },
                    ],
                }));
            }

            // ── Pass A: Tile Binning ─────────────────────────────────────
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("VG Tile Binning"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.binning_pipeline);
                if let Some(ref bg) = self.binning_bg {
                    cpass.set_bind_group(0, bg, &[]);
                }
                let vis_count = self.visible_meshlet_ids_buf.size() / 4;
                cpass.dispatch_workgroups((vis_count as u32 + 63) / 64, 1, 1);
            }

            // ── Pass B: Software Rasterize ────────────────────────────────
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("VG SW Rasterize"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.rasterize_pipeline);
                if let Some(ref bg) = self.rasterize_bg {
                    cpass.set_bind_group(0, bg, &[]);
                }
                let tile_grid_x = (sw_w + TILE_SIZE_X - 1) / TILE_SIZE_X;
                let tile_grid_y = (sw_h + TILE_SIZE_Y - 1) / TILE_SIZE_Y;
                cpass.dispatch_workgroups(tile_grid_x, tile_grid_y, 1);
            }

            // ── Pass C: Shade ─────────────────────────────────────────────
            {
                let Some(gbuffer) = ctx.resources.gbuffer.read("VirtualGeometry") else {
                    return Ok(());
                };
                let Some(lightmap_uv) = ctx.resources.gbuffer_lightmap_uv.read("VirtualGeometry") else {
                    return Ok(());
                };
                let Some(sss) = ctx.resources.gbuffer_sss.read("VirtualGeometry") else {
                    return Ok(());
                };
                let Some(extra) = ctx.resources.gbuffer_extra.read("VirtualGeometry") else {
                    return Ok(());
                };

                // Rebuild shade BG0 and BG2 each frame (visibility data and GBuffer views)
                self.shade_bg_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("VG Shade BG0"),
                    layout: &self.shade_bgl_0,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: self.visibility_depth_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: self.visibility_data_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.visibility_instance_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: self.meshlet_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: self.meshlet_vertex_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: self.meshlet_index_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 6, resource: ctx.scene.camera.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 7, resource: self.globals_buf.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 8, resource: self.instance_buf.as_entire_binding() },
                    ],
                }));
                self.shade_bg_2 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("VG Shade BG2"),
                    layout: &self.shade_bgl_2,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(gbuffer.albedo) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(gbuffer.normal) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(gbuffer.orm) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(gbuffer.emissive) },
                        wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(lightmap_uv) },
                        wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(sss) },
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(extra) },
                    ],
                }));

                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("VG Shade"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.shade_pipeline);
                if let Some(ref bg0) = self.shade_bg_0 {
                    cpass.set_bind_group(0, bg0, &[]);
                }
                if let Some(ref bg1) = self.shade_bg_1 {
                    cpass.set_bind_group(1, bg1, &[]);
                }
                if let Some(ref bg2) = self.shade_bg_2 {
                    cpass.set_bind_group(2, bg2, &[]);
                }
                let dispatch_x = (sw_w + 7) / 8;
                let dispatch_y = (sw_h + 7) / 8;
                cpass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
            }

            return Ok(());
        }

        if self.debug_mode == 21 && matches!(self.debug_readback_state, DebugReadbackState::Idle) {
            unsafe { &mut *ctx.compute_encoder_ptr }.copy_buffer_to_buffer(
                &self.draw_count_buf,
                0,
                &self.debug_readback_buf,
                0,
                DRAW_COUNTER_BYTES,
            );
            self.debug_readback_state = DebugReadbackState::CopySubmitted;
        }

        let opaque_capacity = max_draw_count / 2;
        let alpha_count = max_draw_count - opaque_capacity;

        let bind_and_draw = |rpass: &mut wgpu::RenderPass<'_>| {
            rpass.set_bind_group(0, draw_bg0, &[]);
            rpass.set_bind_group(1, draw_bg1, &[]);
            rpass.set_index_buffer(
                self.meshlet_index_buf.slice(..),
                wgpu::IndexFormat::Uint16,
            );
        };

        let draw_region = |rpass: &mut wgpu::RenderPass<'_>, pipeline: &wgpu::RenderPipeline, first_slot: u32, count: u32, counter_byte: u64| {
            rpass.set_pipeline(pipeline);
            if self.use_count_indirect {
                rpass.multi_draw_indexed_indirect_count(
                    &self.indirect_buf,
                    first_slot as u64 * 20,
                    &self.draw_count_buf,
                    counter_byte,
                    count,
                );
            } else {
                #[cfg(not(target_arch = "wasm32"))]
                rpass.multi_draw_indexed_indirect(
                    &self.indirect_buf,
                    first_slot as u64 * 20,
                    count,
                );
                #[cfg(target_arch = "wasm32")]
                for i in first_slot..first_slot + count {
                    rpass.draw_indexed_indirect(&self.indirect_buf, i as u64 * 20);
                }
            }
        };

        match self.debug_mode {
            20 | 21 => {
                // Debug modes: single render pass writing GBuffer (no two-pass)
                let Some(gbuffer) = ctx.resources.gbuffer.read("VirtualGeometry") else {
                    return Ok(());
                };
                let Some(lightmap_uv) = ctx.resources.gbuffer_lightmap_uv.read("VirtualGeometry") else {
                    return Ok(());
                };
                let Some(sss) = ctx.resources.gbuffer_sss.read("VirtualGeometry") else {
                    return Ok(());
                };
                let Some(extra) = ctx.resources.gbuffer_extra.read("VirtualGeometry") else {
                    return Ok(());
                };
                let dbg_atts = &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.albedo, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.normal, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.orm, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.emissive, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: lightmap_uv, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: sss, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: extra, resolve_target: None, depth_slice: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    }),
                ];
                let dbg_desc = wgpu::RenderPassDescriptor {
                    label: Some("VG Debug"),
                    color_attachments: dbg_atts,
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: ctx.depth,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                };
                let mut rpass = ctx.begin_render_pass(&dbg_desc);
                bind_and_draw(&mut rpass);
                let pipeline = if self.debug_mode == 20 {
                    &self.debug_draw_pipeline
                } else {
                    &self.lod_debug_pipeline
                };
                draw_region(&mut rpass, pipeline, 0, opaque_capacity, 0);
                draw_region(&mut rpass, pipeline, opaque_capacity, alpha_count, 4);
            }
            _ => {
                // ── PASS 1: Visibility (depth-only) ────────────────────────
                {
                    let vis_desc = wgpu::RenderPassDescriptor {
                        label: Some("VG Visibility"),
                        color_attachments: &[],  // no color targets
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: ctx.depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    };
                    let mut rpass = ctx.begin_render_pass(&vis_desc);
                    bind_and_draw(&mut rpass);
                    draw_region(&mut rpass, &self.visibility_opaque_pipeline, 0, opaque_capacity, 0);
                    draw_region(&mut rpass, &self.visibility_alpha_pipeline, opaque_capacity, alpha_count, 4);
                }

                // ── PASS 2: Shading (depth equal, full GBuffer) ───────────
                {
                    let Some(gbuffer) = ctx.resources.gbuffer.read("VirtualGeometry") else {
                        return Ok(());
                    };
                    let Some(lightmap_uv) = ctx.resources.gbuffer_lightmap_uv.read("VirtualGeometry") else {
                        return Ok(());
                    };
                    let Some(sss) = ctx.resources.gbuffer_sss.read("VirtualGeometry") else {
                        return Ok(());
                    };
                    let Some(extra) = ctx.resources.gbuffer_extra.read("VirtualGeometry") else {
                        return Ok(());
                    };
                    let shade_atts = &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: gbuffer.albedo, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: gbuffer.normal, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: gbuffer.orm, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: gbuffer.emissive, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: lightmap_uv, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: sss, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: extra, resolve_target: None, depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                        }),
                    ];
                    let shade_desc = wgpu::RenderPassDescriptor {
                        label: Some("VG Shading"),
                        color_attachments: shade_atts,
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: ctx.depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    };
                    let mut rpass = ctx.begin_render_pass(&shade_desc);
                    bind_and_draw(&mut rpass);
                    draw_region(&mut rpass, &self.shading_opaque_pipeline, 0, opaque_capacity, 0);
                    draw_region(&mut rpass, &self.shading_alpha_pipeline, opaque_capacity, alpha_count, 4);
                }
            }
        }

        Ok(())
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "main_scene", "vg", "hiz"]
    }
    fn writes(&self) -> &'static [&'static str] {
        &["gbuffer", "gbuffer_lightmap_uv", "gbuffer_sss", "gbuffer_extra"]
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
            DebugViewDescriptor {
                name: "VG Meshlets",
                debug_mode: 20,
                description: "One solid colour per meshlet — visualises cluster boundaries",
            },
            DebugViewDescriptor {
                name: "VG LOD Heatmap",
                debug_mode: 21,
                description: "One flat colour per object LOD; green=LOD0 through magenta=LOD7",
            },
        ];
        VIEWS
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

fn create_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let vis = wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE;
    #[cfg(not(target_arch = "wasm32"))]
    let count = NonZeroU32::new(MAX_TEXTURES as u32).expect("non-zero");
    let mut entries = vec![
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        },
    ];
    #[cfg(not(target_arch = "wasm32"))]
    {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: vis,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: Some(count),
        });
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 3,
            visibility: vis,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: Some(count),
        });
    }
    #[cfg(target_arch = "wasm32")]
    {
        for index in 0..MAX_TEXTURES {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + index as u32,
                visibility: vis,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        for index in 0..MAX_TEXTURES {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: 2 + MAX_TEXTURES as u32 + index as u32,
                visibility: vis,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
    }
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("VG Material BGL"),
        entries: &entries,
    })
}
