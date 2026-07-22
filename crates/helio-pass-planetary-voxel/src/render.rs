use crate::{
    FrameUpdateOutcome, GpuResidencyError, GpuUploadOutcome, PlanetaryVoxelGpuConfig,
    PlanetaryVoxelResidency, TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT, TransvoxelGpuError,
    TransvoxelGpuExtractor, TransvoxelGpuExtractorConfig, TransvoxelGpuTransitionExtractor,
    TransvoxelGpuTransitionExtractorConfig, TransvoxelTransitionGpuError,
};
use bytemuck::{Pod, Zeroable};
use helio_core::{
    PassContext, PrepareContext, RenderPass, Result as HelioResult,
    graph::{ResourceBuilder, ResourceSize},
};
use helio_planet_voxel_core::{
    CellWord, ContractError, EvictOutcome, GpuPageMeta, PageEvict, PageUpload, PlanetFrameUniform,
    PlanetId, PlanetPageKey, SourceGeneration, TRANSITION_FACE_MASK, UploadOutcome,
    VisibilityOutcome, VisiblePageSet,
};
use std::{
    collections::{BTreeSet, VecDeque},
    sync::{
        Mutex,
        mpsc::{self, Receiver, TryRecvError},
    },
};
use wgpu::util::DeviceExt;

const SURFACE_BANKS: u32 = 2;
const COPY_WORKGROUP_SIZE: u32 = 64;
const DRAW_ARGS_BYTES: u64 = 20;

pub const SURFACE_PUBLISH_WGSL: &str = include_str!("surface_publish.wgsl");
pub const SURFACE_DRAW_WGSL: &str = include_str!("surface_draw.wgsl");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetaryVoxelRenderConfig {
    pub residency: PlanetaryVoxelGpuConfig,
    pub max_pending_surfaces: u32,
    pub regular: TransvoxelGpuExtractorConfig,
    pub transition: TransvoxelGpuTransitionExtractorConfig,
    pub max_surface_bytes: u64,
}

impl PlanetaryVoxelRenderConfig {
    pub fn validation_demo() -> Self {
        Self {
            residency: PlanetaryVoxelGpuConfig::new(5, 16, 12, 5, 16, 4)
                .expect("validation residency configuration is valid"),
            max_pending_surfaces: 8,
            regular: TransvoxelGpuExtractorConfig::new(65_536, 131_072)
                .expect("validation regular extraction configuration is valid"),
            transition: TransvoxelGpuTransitionExtractorConfig::new(49_152, 147_456)
                .expect("validation transition extraction configuration is valid"),
            max_surface_bytes: 128 * 1024 * 1024,
        }
    }

    pub fn allocation_plan(self) -> Result<PlanetarySurfaceAllocationPlan, PlanetaryRenderError> {
        if self.max_pending_surfaces == 0 {
            return Err(PlanetaryRenderError::ZeroPendingSurfaces);
        }
        let pages = u64::from(self.residency.max_resident_pages);
        let banks = u64::from(SURFACE_BANKS);
        let regular_vertex_bytes = checked_product(&[
            pages,
            banks,
            u64::from(self.regular.max_vertices),
            core::mem::size_of::<crate::GpuTerrainVertex>() as u64,
        ])?;
        let regular_index_bytes = checked_product(&[
            pages,
            banks,
            u64::from(self.regular.max_indices),
            core::mem::size_of::<u32>() as u64,
        ])?;
        let transition_vertex_bytes = checked_product(&[
            pages,
            banks,
            u64::from(self.transition.max_vertices),
            core::mem::size_of::<crate::GpuTerrainVertex>() as u64,
        ])?;
        let transition_index_bytes = checked_product(&[
            pages,
            banks,
            u64::from(self.transition.max_indices),
            core::mem::size_of::<u32>() as u64,
        ])?;
        let state_bytes = pages
            .checked_mul(core::mem::size_of::<GpuSurfaceState>() as u64)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let draw_page_bytes = pages
            .checked_mul(core::mem::size_of::<GpuDrawPage>() as u64)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let feedback_bytes = core::mem::size_of::<GpuSurfaceFeedback>() as u64;
        let indirect_bytes = pages
            .checked_mul(DRAW_ARGS_BYTES)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let diagnostic_readback_bytes =
            [feedback_bytes, state_bytes, indirect_bytes, indirect_bytes]
                .into_iter()
                .try_fold(0_u64, |total, bytes| {
                    total
                        .checked_add(bytes)
                        .ok_or(PlanetaryRenderError::ArithmeticOverflow)
                })?;
        let total_bytes = [
            regular_vertex_bytes,
            regular_index_bytes,
            transition_vertex_bytes,
            transition_index_bytes,
            state_bytes,
            draw_page_bytes,
            feedback_bytes,
            indirect_bytes,
            indirect_bytes,
            diagnostic_readback_bytes,
        ]
        .into_iter()
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes)
                .ok_or(PlanetaryRenderError::ArithmeticOverflow)
        })?;
        if total_bytes > self.max_surface_bytes {
            return Err(PlanetaryRenderError::SurfaceBudget {
                requested: total_bytes,
                maximum: self.max_surface_bytes,
            });
        }
        Ok(PlanetarySurfaceAllocationPlan {
            regular_vertex_bytes,
            regular_index_bytes,
            transition_vertex_bytes,
            transition_index_bytes,
            state_bytes,
            draw_page_bytes,
            feedback_bytes,
            indirect_bytes,
            diagnostic_readback_bytes,
            total_bytes,
        })
    }

    fn validate_device(self, limits: &wgpu::Limits) -> Result<(), PlanetaryRenderError> {
        let plan = self.allocation_plan()?;
        for (name, bytes, storage) in [
            ("regular vertex arena", plan.regular_vertex_bytes, true),
            ("regular index arena", plan.regular_index_bytes, true),
            (
                "transition vertex arena",
                plan.transition_vertex_bytes,
                true,
            ),
            ("transition index arena", plan.transition_index_bytes, true),
            ("surface state", plan.state_bytes, true),
            ("draw pages", plan.draw_page_bytes, true),
            ("surface feedback", plan.feedback_bytes, true),
            ("regular indirect", plan.indirect_bytes, true),
            ("transition indirect", plan.indirect_bytes, true),
            ("diagnostic readback", plan.diagnostic_readback_bytes, false),
        ] {
            if bytes > limits.max_buffer_size
                || (storage && bytes > limits.max_storage_buffer_binding_size)
            {
                return Err(PlanetaryRenderError::DeviceBufferLimit {
                    name,
                    requested: bytes,
                    max_buffer_bytes: limits.max_buffer_size,
                    max_storage_bytes: limits.max_storage_buffer_binding_size,
                });
            }
        }
        if limits.max_storage_buffers_per_shader_stage < 7 {
            return Err(PlanetaryRenderError::StorageBindingLimit {
                required: 7,
                available: limits.max_storage_buffers_per_shader_stage,
            });
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetarySurfaceAllocationPlan {
    pub regular_vertex_bytes: u64,
    pub regular_index_bytes: u64,
    pub transition_vertex_bytes: u64,
    pub transition_index_bytes: u64,
    pub state_bytes: u64,
    pub draw_page_bytes: u64,
    pub feedback_bytes: u64,
    pub indirect_bytes: u64,
    pub diagnostic_readback_bytes: u64,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanetarySurfaceUpload {
    pub key: PlanetPageKey,
    pub generation: SourceGeneration,
    pub halo_samples: Box<[CellWord]>,
    pub transition_face_slabs: Box<[CellWord]>,
    pub transition_mask: u8,
    pub dirty_microbricks: u64,
}

impl PlanetarySurfaceUpload {
    pub fn validate(&self) -> Result<(), PlanetaryRenderError> {
        self.key.validate()?;
        if self.halo_samples.len() != crate::EXTRACTION_SAMPLE_COUNT {
            return Err(PlanetaryRenderError::RegularSampleCount {
                actual: self.halo_samples.len(),
                expected: crate::EXTRACTION_SAMPLE_COUNT,
            });
        }
        if self.transition_face_slabs.len() != TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT {
            return Err(PlanetaryRenderError::TransitionSampleCount {
                actual: self.transition_face_slabs.len(),
                expected: TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT,
            });
        }
        if self.transition_mask & !TRANSITION_FACE_MASK != 0 {
            return Err(PlanetaryRenderError::TransitionMask(self.transition_mask));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PlanetaryRenderCounters {
    pub queued_surfaces: usize,
    pub submitted_jobs: u64,
    pub stale_surface_rejections: u64,
    pub pending_backpressure: u64,
    pub cleared_slots: u64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanetaryRenderDiagnostics {
    pub gpu_submitted_jobs: u32,
    pub gpu_published_jobs: u32,
    pub gpu_stale_rejections: u32,
    pub gpu_overflow_rejections: u32,
    pub gpu_incomplete_rejections: u32,
    pub resident_lods: Vec<u8>,
    pub source_generation_min: Option<SourceGeneration>,
    pub source_generation_max: Option<SourceGeneration>,
    pub publication_generation_min: Option<u64>,
    pub publication_generation_max: Option<u64>,
    pub regular_vertices: u64,
    pub regular_indices: u64,
    pub transition_vertices: u64,
    pub transition_indices: u64,
    pub visible_regular_draws: u32,
    pub visible_transition_draws: u32,
    pub readback_failures: u64,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceJob {
    slot: u32,
    transition_mask: u32,
    generation_low: u32,
    generation_high: u32,
    regular_max_vertices: u32,
    regular_max_indices: u32,
    transition_max_vertices: u32,
    transition_max_indices: u32,
}

impl GpuSurfaceJob {
    fn new(
        slot: u32,
        transition_mask: u8,
        generation: u64,
        config: PlanetaryVoxelRenderConfig,
    ) -> Self {
        Self {
            slot,
            transition_mask: u32::from(transition_mask),
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            regular_max_vertices: config.regular.max_vertices,
            regular_max_indices: config.regular.max_indices,
            transition_max_vertices: config.transition.max_vertices,
            transition_max_indices: config.transition.max_indices,
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceState {
    generation_low: u32,
    generation_high: u32,
    active_bank: u32,
    valid: u32,
    regular_vertex_count: u32,
    regular_index_count: u32,
    transition_vertex_count: u32,
    transition_index_count: u32,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceFeedback {
    submitted_jobs: u32,
    published_jobs: u32,
    stale_rejections: u32,
    overflow_rejections: u32,
    incomplete_rejections: u32,
    _pad: [u32; 3],
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
struct GpuDrawPage {
    relative_lod0_cell_min: [i32; 3],
    lod: u32,
    camera_relative_m: [f32; 3],
    lod0_cell_size_m: f32,
    generation_low: u32,
    generation_high: u32,
    transition_mask: u32,
    visible: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct DrawIndexedIndirectArgs {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
}

struct PendingDiagnosticsReadback {
    buffer: wgpu::Buffer,
    receiver: Mutex<Receiver<Result<(), wgpu::BufferAsyncError>>>,
}

#[derive(Clone, Copy)]
struct DiagnosticsReadbackLayout {
    state_offset: u64,
    regular_draw_offset: u64,
    transition_draw_offset: u64,
    total_bytes: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttachmentMode {
    Standalone,
    Composited,
}

pub struct PlanetaryVoxelRenderPass {
    config: PlanetaryVoxelRenderConfig,
    residency: PlanetaryVoxelResidency,
    regular_extractor: TransvoxelGpuExtractor,
    transition_extractor: TransvoxelGpuTransitionExtractor,
    pending: VecDeque<PlanetarySurfaceUpload>,
    prepared: bool,
    visible: BTreeSet<PlanetPageKey>,
    counters: PlanetaryRenderCounters,
    job_buffer: wgpu::Buffer,
    state_buffer: wgpu::Buffer,
    draw_page_buffer: wgpu::Buffer,
    feedback_buffer: wgpu::Buffer,
    regular_vertex_arena: wgpu::Buffer,
    regular_index_arena: wgpu::Buffer,
    transition_vertex_arena: wgpu::Buffer,
    transition_index_arena: wgpu::Buffer,
    regular_indirect: wgpu::Buffer,
    transition_indirect: wgpu::Buffer,
    regular_copy_pipeline: wgpu::ComputePipeline,
    transition_copy_pipeline: wgpu::ComputePipeline,
    publish_pipeline: wgpu::ComputePipeline,
    visibility_pipeline: wgpu::ComputePipeline,
    regular_copy_bind_group: wgpu::BindGroup,
    transition_copy_bind_group: wgpu::BindGroup,
    publish_bind_group: wgpu::BindGroup,
    visibility_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group_layout: wgpu::BindGroupLayout,
    render_bind_group: Option<wgpu::BindGroup>,
    render_camera_key: Option<usize>,
    diagnostic_available: Option<wgpu::Buffer>,
    diagnostic_readback: Option<PendingDiagnosticsReadback>,
    diagnostics_cache: PlanetaryRenderDiagnostics,
    surface_format: wgpu::TextureFormat,
    attachment_mode: AttachmentMode,
}

impl PlanetaryVoxelRenderPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        config: PlanetaryVoxelRenderConfig,
    ) -> Result<Self, PlanetaryRenderError> {
        Self::new_with_attachment_mode(
            device,
            queue,
            surface_format,
            config,
            AttachmentMode::Standalone,
        )
    }

    pub fn new_composited(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        config: PlanetaryVoxelRenderConfig,
    ) -> Result<Self, PlanetaryRenderError> {
        Self::new_with_attachment_mode(
            device,
            queue,
            surface_format,
            config,
            AttachmentMode::Composited,
        )
    }

    fn new_with_attachment_mode(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        config: PlanetaryVoxelRenderConfig,
        attachment_mode: AttachmentMode,
    ) -> Result<Self, PlanetaryRenderError> {
        config.validate_device(&device.limits())?;
        let plan = config.allocation_plan()?;
        let residency = PlanetaryVoxelResidency::new(device, queue, config.residency)?;
        let regular_extractor = TransvoxelGpuExtractor::new(device, config.regular)?;
        let transition_extractor =
            TransvoxelGpuTransitionExtractor::new(device, config.transition)?;
        let job_buffer = create_buffer(
            device,
            "Planetary Surface Job",
            core::mem::size_of::<GpuSurfaceJob>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let state_buffer = create_zeroed_buffer::<GpuSurfaceState>(
            device,
            "Planetary Surface States",
            config.residency.max_resident_pages,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let draw_page_buffer = create_zeroed_buffer::<GpuDrawPage>(
            device,
            "Planetary Draw Pages",
            config.residency.max_resident_pages,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        );
        let feedback_buffer = create_zeroed_buffer::<GpuSurfaceFeedback>(
            device,
            "Planetary Surface Feedback",
            1,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );
        let regular_vertex_arena = create_buffer(
            device,
            "Planetary Regular Vertex Arena",
            plan.regular_vertex_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        );
        let regular_index_arena = create_buffer(
            device,
            "Planetary Regular Index Arena",
            plan.regular_index_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
        );
        let transition_vertex_arena = create_buffer(
            device,
            "Planetary Transition Vertex Arena",
            plan.transition_vertex_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
        );
        let transition_index_arena = create_buffer(
            device,
            "Planetary Transition Index Arena",
            plan.transition_index_bytes,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDEX,
        );
        let indirect_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let regular_indirect = create_buffer(
            device,
            "Planetary Regular Indirect Draws",
            plan.indirect_bytes,
            indirect_usage,
        );
        let transition_indirect = create_buffer(
            device,
            "Planetary Transition Indirect Draws",
            plan.indirect_bytes,
            indirect_usage,
        );
        let diagnostic_readback_buffer = create_buffer(
            device,
            "Planetary Surface Diagnostics Readback",
            plan.diagnostic_readback_bytes,
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        );

        let publish_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Surface Publication Shader"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_PUBLISH_WGSL.into()),
        });
        let regular_copy_pipeline =
            compute_pipeline(device, &publish_shader, "copy_regular_surface");
        let transition_copy_pipeline =
            compute_pipeline(device, &publish_shader, "copy_transition_surface");
        let publish_pipeline = compute_pipeline(device, &publish_shader, "publish_surface");
        let visibility_pipeline = compute_pipeline(device, &publish_shader, "refresh_visibility");
        let regular_copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Regular Surface Copy Bind Group"),
            layout: &regular_copy_pipeline.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &job_buffer),
                buffer_entry(1, residency.metadata_buffer()),
                buffer_entry(2, regular_extractor.counters_buffer()),
                buffer_entry(3, regular_extractor.vertices_buffer()),
                buffer_entry(4, regular_extractor.indices_buffer()),
                buffer_entry(5, &state_buffer),
                buffer_entry(6, &regular_vertex_arena),
                buffer_entry(7, &regular_index_arena),
            ],
        });
        let transition_copy_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Transition Surface Copy Bind Group"),
            layout: &transition_copy_pipeline.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &job_buffer),
                buffer_entry(1, residency.metadata_buffer()),
                buffer_entry(5, &state_buffer),
                buffer_entry(8, transition_extractor.counters_buffer()),
                buffer_entry(9, transition_extractor.vertices_buffer()),
                buffer_entry(10, transition_extractor.indices_buffer()),
                buffer_entry(11, &transition_vertex_arena),
                buffer_entry(12, &transition_index_arena),
            ],
        });
        let publish_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Surface Publish Bind Group"),
            layout: &publish_pipeline.get_bind_group_layout(0),
            entries: &[
                buffer_entry(0, &job_buffer),
                buffer_entry(1, residency.metadata_buffer()),
                buffer_entry(2, regular_extractor.counters_buffer()),
                buffer_entry(5, &state_buffer),
                buffer_entry(8, transition_extractor.counters_buffer()),
                buffer_entry(14, &regular_indirect),
                buffer_entry(15, &transition_indirect),
                buffer_entry(16, &feedback_buffer),
            ],
        });
        let visibility_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Planetary Surface Visibility Bind Group"),
            layout: &visibility_pipeline.get_bind_group_layout(0),
            entries: &[
                buffer_entry(5, &state_buffer),
                buffer_entry(13, &draw_page_buffer),
                buffer_entry(14, &regular_indirect),
                buffer_entry(15, &transition_indirect),
            ],
        });

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Surface Draw Shader"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_DRAW_WGSL.into()),
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Planetary Surface Draw Bind Group Layout"),
                entries: &[
                    uniform_layout_entry(0, wgpu::ShaderStages::VERTEX),
                    storage_layout_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT, true),
                ],
            });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planetary Surface Draw Pipeline Layout"),
            bind_group_layouts: &[Some(&render_bind_group_layout)],
            immediate_size: 0,
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Planetary Surface Draw Pipeline"),
            layout: Some(&render_layout),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: core::mem::size_of::<crate::GpuTerrainVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 16,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 28,
                            shader_location: 3,
                        },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        Ok(Self {
            config,
            residency,
            regular_extractor,
            transition_extractor,
            pending: VecDeque::new(),
            prepared: false,
            visible: BTreeSet::new(),
            counters: PlanetaryRenderCounters::default(),
            job_buffer,
            state_buffer,
            draw_page_buffer,
            feedback_buffer,
            regular_vertex_arena,
            regular_index_arena,
            transition_vertex_arena,
            transition_index_arena,
            regular_indirect,
            transition_indirect,
            regular_copy_pipeline,
            transition_copy_pipeline,
            publish_pipeline,
            visibility_pipeline,
            regular_copy_bind_group,
            transition_copy_bind_group,
            publish_bind_group,
            visibility_bind_group,
            render_pipeline,
            render_bind_group_layout,
            render_bind_group: None,
            render_camera_key: None,
            diagnostic_available: Some(diagnostic_readback_buffer),
            diagnostic_readback: None,
            diagnostics_cache: PlanetaryRenderDiagnostics::default(),
            surface_format,
            attachment_mode,
        })
    }

    pub const fn residency(&self) -> &PlanetaryVoxelResidency {
        &self.residency
    }

    pub fn residency_mut(&mut self) -> &mut PlanetaryVoxelResidency {
        &mut self.residency
    }

    pub fn counters(&self) -> PlanetaryRenderCounters {
        let mut counters = self.counters;
        counters.queued_surfaces = self.pending.len();
        counters
    }

    /// Polls a bounded asynchronous readback of publication state and starts
    /// the next snapshot when the previous one completes. This never waits for
    /// the GPU and is intended for diagnostics/tooling rather than rendering.
    pub fn poll_diagnostics(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> PlanetaryRenderDiagnostics {
        let _ = device.poll(wgpu::PollType::Poll);
        let mut completion = None;
        let mut disconnected = false;
        if let Some(pending) = self.diagnostic_readback.as_ref() {
            match pending
                .receiver
                .lock()
                .expect("planetary diagnostics receiver mutex is not poisoned")
                .try_recv()
            {
                Ok(result) => completion = Some(result),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => disconnected = true,
            }
        }
        if disconnected {
            self.diagnostic_readback = None;
            self.diagnostics_cache.readback_failures =
                self.diagnostics_cache.readback_failures.saturating_add(1);
        } else if let Some(result) = completion {
            let pending = self
                .diagnostic_readback
                .take()
                .expect("completed planetary diagnostics readback exists");
            match result {
                Ok(()) => {
                    if self.consume_diagnostics_readback(&pending.buffer) {
                        self.diagnostic_available = Some(pending.buffer);
                    } else {
                        self.diagnostics_cache.readback_failures =
                            self.diagnostics_cache.readback_failures.saturating_add(1);
                    }
                }
                Err(error) => {
                    log::warn!("planetary diagnostics readback failed: {error:?}");
                    self.diagnostics_cache.readback_failures =
                        self.diagnostics_cache.readback_failures.saturating_add(1);
                }
            }
        }
        self.refresh_cpu_diagnostics();
        if self.diagnostic_readback.is_none() {
            self.start_diagnostics_readback(device, queue);
        }
        self.diagnostics_cache.clone()
    }

    fn diagnostics_readback_layout(&self) -> DiagnosticsReadbackLayout {
        let pages = u64::from(self.config.residency.max_resident_pages);
        let state_offset = core::mem::size_of::<GpuSurfaceFeedback>() as u64;
        let regular_draw_offset =
            state_offset + pages * core::mem::size_of::<GpuSurfaceState>() as u64;
        let transition_draw_offset = regular_draw_offset + pages * DRAW_ARGS_BYTES;
        DiagnosticsReadbackLayout {
            state_offset,
            regular_draw_offset,
            transition_draw_offset,
            total_bytes: transition_draw_offset + pages * DRAW_ARGS_BYTES,
        }
    }

    fn start_diagnostics_readback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let layout = self.diagnostics_readback_layout();
        let readback = self.diagnostic_available.take().unwrap_or_else(|| {
            create_buffer(
                device,
                "Planetary Surface Diagnostics Readback",
                layout.total_bytes,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            )
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Planetary Surface Diagnostics Copy"),
        });
        encoder.copy_buffer_to_buffer(
            &self.feedback_buffer,
            0,
            &readback,
            0,
            core::mem::size_of::<GpuSurfaceFeedback>() as u64,
        );
        encoder.copy_buffer_to_buffer(
            &self.state_buffer,
            0,
            &readback,
            layout.state_offset,
            layout.regular_draw_offset - layout.state_offset,
        );
        encoder.copy_buffer_to_buffer(
            &self.regular_indirect,
            0,
            &readback,
            layout.regular_draw_offset,
            layout.transition_draw_offset - layout.regular_draw_offset,
        );
        encoder.copy_buffer_to_buffer(
            &self.transition_indirect,
            0,
            &readback,
            layout.transition_draw_offset,
            layout.total_bytes - layout.transition_draw_offset,
        );
        queue.submit([encoder.finish()]);
        let (sender, receiver) = mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        self.diagnostic_readback = Some(PendingDiagnosticsReadback {
            buffer: readback,
            receiver: Mutex::new(receiver),
        });
    }

    fn consume_diagnostics_readback(&mut self, buffer: &wgpu::Buffer) -> bool {
        let layout = self.diagnostics_readback_layout();
        let mapped = match buffer.slice(..).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(error) => {
                log::warn!("planetary diagnostics mapped range unavailable: {error:?}");
                return false;
            }
        };
        let feedback_bytes = &mapped[..layout.state_offset as usize];
        let feedback = *bytemuck::from_bytes::<GpuSurfaceFeedback>(feedback_bytes);
        let states = bytemuck::cast_slice::<u8, GpuSurfaceState>(
            &mapped[layout.state_offset as usize..layout.regular_draw_offset as usize],
        );
        let regular_draws = bytemuck::cast_slice::<u8, DrawIndexedIndirectArgs>(
            &mapped[layout.regular_draw_offset as usize..layout.transition_draw_offset as usize],
        );
        let transition_draws = bytemuck::cast_slice::<u8, DrawIndexedIndirectArgs>(
            &mapped[layout.transition_draw_offset as usize..layout.total_bytes as usize],
        );

        self.diagnostics_cache.gpu_submitted_jobs = feedback.submitted_jobs;
        self.diagnostics_cache.gpu_published_jobs = feedback.published_jobs;
        self.diagnostics_cache.gpu_stale_rejections = feedback.stale_rejections;
        self.diagnostics_cache.gpu_overflow_rejections = feedback.overflow_rejections;
        self.diagnostics_cache.gpu_incomplete_rejections = feedback.incomplete_rejections;
        self.diagnostics_cache.regular_vertices = states
            .iter()
            .filter(|state| state.valid != 0)
            .map(|state| u64::from(state.regular_vertex_count))
            .sum();
        self.diagnostics_cache.regular_indices = states
            .iter()
            .filter(|state| state.valid != 0)
            .map(|state| u64::from(state.regular_index_count))
            .sum();
        self.diagnostics_cache.transition_vertices = states
            .iter()
            .filter(|state| state.valid != 0)
            .map(|state| u64::from(state.transition_vertex_count))
            .sum();
        self.diagnostics_cache.transition_indices = states
            .iter()
            .filter(|state| state.valid != 0)
            .map(|state| u64::from(state.transition_index_count))
            .sum();
        self.diagnostics_cache.visible_regular_draws = regular_draws
            .iter()
            .filter(|draw| draw.instance_count != 0 && draw.index_count != 0)
            .count() as u32;
        self.diagnostics_cache.visible_transition_draws = transition_draws
            .iter()
            .filter(|draw| draw.instance_count != 0 && draw.index_count != 0)
            .count() as u32;
        drop(mapped);
        buffer.unmap();
        true
    }

    fn refresh_cpu_diagnostics(&mut self) {
        let mut lods = Vec::new();
        let mut source_min = None;
        let mut source_max = None;
        let mut publication_min = None;
        let mut publication_max = None;
        for (key, resident) in self.residency.cache().resident_pages() {
            lods.push(key.page.lod);
            source_min = Some(
                source_min.map_or(resident.generation, |value: SourceGeneration| {
                    value.min(resident.generation)
                }),
            );
            source_max = Some(
                source_max.map_or(resident.generation, |value: SourceGeneration| {
                    value.max(resident.generation)
                }),
            );
            publication_min = Some(
                publication_min.map_or(resident.publication_generation, |value: u64| {
                    value.min(resident.publication_generation)
                }),
            );
            publication_max = Some(
                publication_max.map_or(resident.publication_generation, |value: u64| {
                    value.max(resident.publication_generation)
                }),
            );
        }
        lods.sort_unstable();
        lods.dedup();
        self.diagnostics_cache.resident_lods = lods;
        self.diagnostics_cache.source_generation_min = source_min;
        self.diagnostics_cache.source_generation_max = source_max;
        self.diagnostics_cache.publication_generation_min = publication_min;
        self.diagnostics_cache.publication_generation_max = publication_max;
    }

    pub fn set_planet_frame(
        &mut self,
        queue: &wgpu::Queue,
        frame: PlanetFrameUniform,
    ) -> Result<FrameUpdateOutcome, PlanetaryRenderError> {
        let outcome = self.residency.set_planet_frame(queue, frame)?;
        if matches!(outcome, FrameUpdateOutcome::Applied { .. }) {
            self.publish_draw_pages(queue)?;
        }
        Ok(outcome)
    }

    pub fn apply_upload_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        uploads: Vec<PageUpload>,
    ) -> Result<Vec<GpuUploadOutcome>, PlanetaryRenderError> {
        let outcomes = self.residency.apply_upload_batch(device, queue, uploads)?;
        for outcome in &outcomes {
            if let GpuUploadOutcome::Residency(UploadOutcome::Inserted { evicted, .. }) = outcome {
                for page in evicted {
                    self.clear_slot(queue, page.slot);
                }
            }
        }
        self.publish_draw_pages(queue)?;
        Ok(outcomes)
    }

    pub fn apply_evict_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        evictions: Vec<PageEvict>,
    ) -> Result<Vec<EvictOutcome>, PlanetaryRenderError> {
        let outcomes = self.residency.apply_evict_batch(device, queue, evictions)?;
        for outcome in &outcomes {
            if let EvictOutcome::Recorded {
                removed: Some(page),
            } = outcome
            {
                self.clear_slot(queue, page.slot);
            }
        }
        self.publish_draw_pages(queue)?;
        Ok(outcomes)
    }

    pub fn apply_visible_set(
        &mut self,
        queue: &wgpu::Queue,
        set: VisiblePageSet,
    ) -> Result<VisibilityOutcome, PlanetaryRenderError> {
        let candidate: BTreeSet<_> = set.pages.iter().map(|page| page.key).collect();
        let outcome = self.residency.apply_visible_set(queue, set)?;
        if matches!(outcome, VisibilityOutcome::Applied { .. }) {
            self.visible = candidate;
            self.publish_draw_pages(queue)?;
        }
        Ok(outcome)
    }

    pub fn queue_surface(
        &mut self,
        upload: PlanetarySurfaceUpload,
    ) -> Result<(), PlanetaryRenderError> {
        upload.validate()?;
        let resident = self
            .residency
            .cache()
            .resident(upload.key)
            .ok_or(PlanetaryRenderError::SurfacePageNotResident(upload.key))?;
        if resident.generation != upload.generation {
            self.counters.stale_surface_rejections =
                self.counters.stale_surface_rejections.saturating_add(1);
            return Err(PlanetaryRenderError::SurfaceGeneration {
                key: upload.key,
                expected: resident.generation,
                actual: upload.generation,
            });
        }
        if let Some(existing) = self
            .pending
            .iter_mut()
            .find(|pending| pending.key == upload.key)
        {
            if upload.generation < existing.generation {
                self.counters.stale_surface_rejections =
                    self.counters.stale_surface_rejections.saturating_add(1);
                return Err(PlanetaryRenderError::PendingSurfaceStale(upload.key));
            }
            *existing = upload;
            return Ok(());
        }
        if self.pending.len() == self.config.max_pending_surfaces as usize {
            self.counters.pending_backpressure =
                self.counters.pending_backpressure.saturating_add(1);
            return Err(PlanetaryRenderError::PendingSurfaceCapacity {
                maximum: self.config.max_pending_surfaces,
            });
        }
        self.pending.push_back(upload);
        Ok(())
    }

    fn clear_slot(&mut self, queue: &wgpu::Queue, slot: u32) {
        let zero_state = GpuSurfaceState::default();
        queue.write_buffer(
            &self.state_buffer,
            u64::from(slot) * core::mem::size_of::<GpuSurfaceState>() as u64,
            bytemuck::bytes_of(&zero_state),
        );
        let zero_draw = [0_u8; DRAW_ARGS_BYTES as usize];
        let offset = u64::from(slot) * DRAW_ARGS_BYTES;
        queue.write_buffer(&self.regular_indirect, offset, &zero_draw);
        queue.write_buffer(&self.transition_indirect, offset, &zero_draw);
        self.pending.retain(|pending| {
            self.residency
                .cache()
                .resident(pending.key)
                .is_some_and(|page| page.slot != slot)
        });
        self.counters.cleared_slots = self.counters.cleared_slots.saturating_add(1);
    }

    fn publish_draw_pages(&self, queue: &wgpu::Queue) -> Result<(), PlanetaryRenderError> {
        let mut pages =
            vec![GpuDrawPage::default(); self.config.residency.max_resident_pages as usize];
        for (key, resident) in self.residency.cache().resident_pages() {
            let frame = self
                .residency
                .planet_frame(key.planet)
                .ok_or(PlanetaryRenderError::MissingPlanetFrame(key.planet))?;
            let meta = GpuPageMeta::new(
                key.page,
                frame.frame_origin_lod0_cell(),
                resident.slot,
                resident.publication_generation,
                0,
            )?;
            pages[resident.slot as usize] = GpuDrawPage {
                relative_lod0_cell_min: meta.relative_lod0_cell_min,
                lod: meta.lod,
                camera_relative_m: frame.camera_relative_m,
                lod0_cell_size_m: frame.lod0_cell_size_m,
                generation_low: resident.publication_generation as u32,
                generation_high: (resident.publication_generation >> 32) as u32,
                transition_mask: 0,
                visible: u32::from(self.visible.contains(&key)),
            };
        }
        queue.write_buffer(&self.draw_page_buffer, 0, bytemuck::cast_slice(&pages));
        Ok(())
    }
}

impl RenderPass for PlanetaryVoxelRenderPass {
    fn name(&self) -> &'static str {
        "PlanetaryVoxel"
    }

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw("pre_aa", self.surface_format, ResourceSize::MatchSurface);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.prepared = false;
        while let Some(front) = self.pending.front() {
            let Some(resident) = self.residency.cache().resident(front.key) else {
                self.pending.pop_front();
                self.counters.stale_surface_rejections =
                    self.counters.stale_surface_rejections.saturating_add(1);
                continue;
            };
            if resident.generation != front.generation {
                self.pending.pop_front();
                self.counters.stale_surface_rejections =
                    self.counters.stale_surface_rejections.saturating_add(1);
                continue;
            }
            let job = GpuSurfaceJob::new(
                resident.slot,
                front.transition_mask,
                resident.publication_generation,
                self.config,
            );
            ctx.write_buffer(&self.job_buffer, 0, bytemuck::bytes_of(&job));
            if let Err(error) = self.regular_extractor.prepare(
                ctx.queue,
                &front.halo_samples,
                resident.publication_generation,
                front.dirty_microbricks,
            ) {
                log::error!("planetary regular extraction prepare failed: {error}");
                self.pending.pop_front();
                continue;
            }
            if let Err(error) = self.transition_extractor.prepare(
                ctx.queue,
                &front.transition_face_slabs,
                front.transition_mask,
                resident.publication_generation,
            ) {
                log::error!("planetary transition extraction prepare failed: {error}");
                self.pending.pop_front();
                continue;
            }
            self.prepared = true;
            break;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let compute = unsafe { &mut *ctx.compute_encoder_ptr };
        if self.prepared {
            self.regular_extractor.encode(compute);
            self.transition_extractor.encode(compute);
            dispatch_compute(
                compute,
                &self.regular_copy_pipeline,
                &self.regular_copy_bind_group,
                self.config
                    .regular
                    .max_vertices
                    .max(self.config.regular.max_indices)
                    .div_ceil(COPY_WORKGROUP_SIZE),
                "Planetary Regular Surface Copy",
            );
            dispatch_compute(
                compute,
                &self.transition_copy_pipeline,
                &self.transition_copy_bind_group,
                self.config
                    .transition
                    .max_vertices
                    .max(self.config.transition.max_indices)
                    .div_ceil(COPY_WORKGROUP_SIZE),
                "Planetary Transition Surface Copy",
            );
            dispatch_compute(
                compute,
                &self.publish_pipeline,
                &self.publish_bind_group,
                1,
                "Planetary Surface Publish",
            );
            self.pending.pop_front();
            self.counters.submitted_jobs = self.counters.submitted_jobs.saturating_add(1);
            self.prepared = false;
        }
        dispatch_compute(
            compute,
            &self.visibility_pipeline,
            &self.visibility_bind_group,
            self.config
                .residency
                .max_resident_pages
                .div_ceil(COPY_WORKGROUP_SIZE),
            "Planetary Surface Visibility",
        );

        let camera_key = ctx.scene.camera as *const _ as usize;
        if self.render_camera_key != Some(camera_key) {
            self.render_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Planetary Surface Draw Bind Group"),
                    layout: &self.render_bind_group_layout,
                    entries: &[
                        buffer_entry(0, ctx.scene.camera),
                        buffer_entry(1, &self.draw_page_buffer),
                    ],
                }));
            self.render_camera_key = Some(camera_key);
        }
        let render = unsafe { &mut *ctx.active_render_pass_ptr().expect("render pass is active") };
        render.set_pipeline(&self.render_pipeline);
        render.set_bind_group(0, self.render_bind_group.as_ref().unwrap(), &[]);
        render.set_vertex_buffer(0, self.regular_vertex_arena.slice(..));
        render.set_index_buffer(
            self.regular_index_arena.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        #[cfg(not(target_arch = "wasm32"))]
        render.multi_draw_indexed_indirect(
            &self.regular_indirect,
            0,
            self.config.residency.max_resident_pages,
        );
        #[cfg(target_arch = "wasm32")]
        for index in 0..self.config.residency.max_resident_pages {
            render
                .draw_indexed_indirect(&self.regular_indirect, u64::from(index) * DRAW_ARGS_BYTES);
        }
        render.set_vertex_buffer(0, self.transition_vertex_arena.slice(..));
        render.set_index_buffer(
            self.transition_index_arena.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        #[cfg(not(target_arch = "wasm32"))]
        render.multi_draw_indexed_indirect(
            &self.transition_indirect,
            0,
            self.config.residency.max_resident_pages,
        );
        #[cfg(target_arch = "wasm32")]
        for index in 0..self.config.residency.max_resident_pages {
            render.draw_indexed_indirect(
                &self.transition_indirect,
                u64::from(index) * DRAW_ARGS_BYTES,
            );
        }
        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let pre_aa = resources.pre_aa.read("PlanetaryVoxel")?;
        let color_load = match self.attachment_mode {
            AttachmentMode::Standalone => wgpu::LoadOp::Clear(wgpu::Color {
                r: 0.004,
                g: 0.008,
                b: 0.018,
                a: 1.0,
            }),
            AttachmentMode::Composited => wgpu::LoadOp::Load,
        };
        let depth_load = match self.attachment_mode {
            AttachmentMode::Standalone => wgpu::LoadOp::Clear(1.0),
            AttachmentMode::Composited => wgpu::LoadOp::Load,
        };
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: pre_aa,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: color_load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("PlanetaryVoxel"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: depth_load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }
}

fn checked_product(values: &[u64]) -> Result<u64, PlanetaryRenderError> {
    values.iter().try_fold(1_u64, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)
    })
}

fn create_buffer(
    device: &wgpu::Device,
    label: &'static str,
    size: u64,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn create_zeroed_buffer<T: Pod + Zeroable>(
    device: &wgpu::Device,
    label: &'static str,
    count: u32,
    usage: wgpu::BufferUsages,
) -> wgpu::Buffer {
    let values = vec![T::zeroed(); count as usize];
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(&values),
        usage,
    })
}

fn compute_pipeline(
    device: &wgpu::Device,
    shader: &wgpu::ShaderModule,
    entry: &str,
) -> wgpu::ComputePipeline {
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: shader,
        entry_point: Some(entry),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_layout_entry(
    binding: u32,
    visibility: wgpu::ShaderStages,
    read_only: bool,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn dispatch_compute(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    workgroups: u32,
    label: &'static str,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some(label),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.dispatch_workgroups(workgroups.max(1), 1, 1);
}

#[derive(Debug, thiserror::Error)]
pub enum PlanetaryRenderError {
    #[error(transparent)]
    Residency(#[from] GpuResidencyError),
    #[error(transparent)]
    Contract(#[from] ContractError),
    #[error(transparent)]
    Address(#[from] helio_planet_voxel_core::AddressError),
    #[error(transparent)]
    Metadata(#[from] helio_planet_voxel_core::GpuPageMetaError),
    #[error(transparent)]
    RegularExtraction(#[from] TransvoxelGpuError),
    #[error(transparent)]
    TransitionExtraction(#[from] TransvoxelTransitionGpuError),
    #[error("planetary render queue must have at least one pending-surface slot")]
    ZeroPendingSurfaces,
    #[error("planetary render allocation arithmetic overflowed")]
    ArithmeticOverflow,
    #[error("planetary surface arenas request {requested} bytes; configured maximum is {maximum}")]
    SurfaceBudget { requested: u64, maximum: u64 },
    #[error(
        "{name} requests {requested} bytes (buffer {max_buffer_bytes}, storage {max_storage_bytes})"
    )]
    DeviceBufferLimit {
        name: &'static str,
        requested: u64,
        max_buffer_bytes: u64,
        max_storage_bytes: u64,
    },
    #[error("planetary publication needs {required} storage bindings; device exposes {available}")]
    StorageBindingLimit { required: u32, available: u32 },
    #[error("regular surface has {actual} samples; expected {expected}")]
    RegularSampleCount { actual: usize, expected: usize },
    #[error("transition surface has {actual} samples; expected {expected}")]
    TransitionSampleCount { actual: usize, expected: usize },
    #[error("transition mask {0:#010b} uses unsupported bits")]
    TransitionMask(u8),
    #[error("surface page {0:?} is not resident")]
    SurfacePageNotResident(PlanetPageKey),
    #[error("surface generation for {key:?} is {actual:?}; resident source is {expected:?}")]
    SurfaceGeneration {
        key: PlanetPageKey,
        expected: SourceGeneration,
        actual: SourceGeneration,
    },
    #[error("pending surface for {0:?} is newer")]
    PendingSurfaceStale(PlanetPageKey),
    #[error("planetary pending-surface queue reached its bounded capacity of {maximum}")]
    PendingSurfaceCapacity { maximum: u32 },
    #[error("planet {0:?} has no camera-local render frame")]
    MissingPlanetFrame(PlanetId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_config_is_bounded_and_below_its_declared_budget() {
        let config = PlanetaryVoxelRenderConfig::validation_demo();
        let plan = config.allocation_plan().unwrap();
        assert!(plan.total_bytes <= config.max_surface_bytes);
        assert_eq!(config.residency.max_resident_pages, 5);
        assert_eq!(plan.indirect_bytes, 5 * DRAW_ARGS_BYTES);
        assert_eq!(plan.feedback_bytes, 32);
        assert_eq!(
            plan.diagnostic_readback_bytes,
            plan.feedback_bytes + plan.state_bytes + plan.indirect_bytes * 2
        );
        assert_eq!(core::mem::size_of::<DrawIndexedIndirectArgs>(), 20);
    }

    #[test]
    fn zero_pending_capacity_is_rejected() {
        let mut config = PlanetaryVoxelRenderConfig::validation_demo();
        config.max_pending_surfaces = 0;
        assert!(matches!(
            config.allocation_plan(),
            Err(PlanetaryRenderError::ZeroPendingSurfaces)
        ));
    }

    #[test]
    fn surface_budget_is_enforced_before_gpu_allocation() {
        let mut config = PlanetaryVoxelRenderConfig::validation_demo();
        let required = config.allocation_plan().unwrap().total_bytes;
        config.max_surface_bytes = required - 1;
        assert!(matches!(
            config.allocation_plan(),
            Err(PlanetaryRenderError::SurfaceBudget {
                requested,
                maximum
            }) if requested == required && maximum == required - 1
        ));
    }

    #[test]
    fn surface_upload_rejects_wrong_extents_and_face_bits() {
        let key = PlanetPageKey::new(
            PlanetId([3; 16]),
            helio_planet_voxel_core::PageKey::new(0, [0, 0, 0]),
        );
        let mut upload = PlanetarySurfaceUpload {
            key,
            generation: SourceGeneration::new(1, 1),
            halo_samples: Box::new([]),
            transition_face_slabs: Box::new([]),
            transition_mask: 0,
            dirty_microbricks: 0,
        };
        assert!(matches!(
            upload.validate(),
            Err(PlanetaryRenderError::RegularSampleCount { actual: 0, .. })
        ));

        upload.halo_samples =
            vec![CellWord::AIR; crate::EXTRACTION_SAMPLE_COUNT].into_boxed_slice();
        assert!(matches!(
            upload.validate(),
            Err(PlanetaryRenderError::TransitionSampleCount { actual: 0, .. })
        ));

        upload.transition_face_slabs =
            vec![CellWord::AIR; TRANSITION_ALL_FACE_SLAB_SAMPLE_COUNT].into_boxed_slice();
        upload.transition_mask = 0x80;
        assert!(matches!(
            upload.validate(),
            Err(PlanetaryRenderError::TransitionMask(0x80))
        ));
    }
}
