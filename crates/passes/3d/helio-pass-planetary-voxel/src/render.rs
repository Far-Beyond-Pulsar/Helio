use crate::{
    BoundedExtractionPublisher, ExtractionError, ExtractionLimits, ExtractionReservation,
    FrameUpdateOutcome, GpuResidencyError, GpuSurfaceGatherCounters, GpuSurfaceGatherJob,
    GpuSurfaceSampler, GpuTerrainCullCounters, GpuTerrainCullUniforms, GpuTerrainDraw,
    GpuTerrainMeshlet, GpuTerrainMeshletBounds, GpuTransvoxelEmissionCounters,
    GpuTransvoxelTransitionCounters, GpuUploadOutcome, PlanetarySurfaceRequest,
    PlanetaryVoxelGpuConfig, PlanetaryVoxelResidency, PublicationOutcome,
    REGULAR_EXTRACTION_INDIRECT_OFFSETS, ReservationOutcome, SurfaceCounts, SurfaceSamplingError,
    TERRAIN_MESHLET_BUILD_WGSL, TERRAIN_MESHLET_CULL_WGSL, TRANSITION_EXTRACTION_INDIRECT_OFFSETS,
    TransvoxelGpuError, TransvoxelGpuExtractor, TransvoxelGpuExtractorConfig,
    TransvoxelGpuTransitionExtractor, TransvoxelGpuTransitionExtractorConfig,
    TransvoxelTransitionGpuError, max_meshlets_for_indices,
};
use bytemuck::{Pod, Zeroable};
use helio_core::{
    PassContext, PrepareContext, RenderPass, Result as HelioResult,
    graph::{ResourceBuilder, ResourceSize},
};
use helio_planet_voxel_core::{
    ContractError, EvictOutcome, GpuPageMeta, PageEvict, PageUpload, PlanetFrameUniform, PlanetId,
    PlanetPageKey, SourceGeneration, UploadOutcome, VisibilityOutcome, VisiblePage, VisiblePageSet,
    split_i64,
};
use std::{
    collections::{BTreeMap, BTreeSet, VecDeque},
    sync::{
        Mutex,
        mpsc::{self, Receiver, TryRecvError},
    },
};
use wgpu::util::DeviceExt;

const COPY_WORKGROUP_SIZE: u32 = 64;
const DRAW_ARGS_BYTES: u64 = 20;
const TERRAIN_DRAW_BYTES: u64 = core::mem::size_of::<GpuTerrainDraw>() as u64;

pub const SURFACE_PUBLISH_WGSL: &str = include_str!("surface_publish.wgsl");
pub const SURFACE_DRAW_WGSL: &str = include_str!("surface_draw.wgsl");

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum PlanetaryDrawPath {
    #[default]
    PageIndexed,
    Meshlets,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum PlanetaryDebugView {
    #[default]
    Material = 0,
    Meshlets = 1,
    ResidentPages = 2,
    Lod = 3,
    TransitionSeams = 4,
    Normals = 5,
}

impl PlanetaryDebugView {
    pub const ALL: [Self; 6] = [
        Self::Material,
        Self::Meshlets,
        Self::ResidentPages,
        Self::Lod,
        Self::TransitionSeams,
        Self::Normals,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::Material => "material",
            Self::Meshlets => "meshlets",
            Self::ResidentPages => "resident-pages",
            Self::Lod => "lod",
            Self::TransitionSeams => "transition-seams",
            Self::Normals => "normals",
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuTerrainDebugUniform {
    mode: u32,
    draw_path: u32,
    _pad: [u32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanetaryVoxelRenderConfig {
    pub residency: PlanetaryVoxelGpuConfig,
    /// Independent extracted-surface arenas. Residency slots remain free to
    /// hold either renderable pages or sampling-only support pages.
    pub max_surface_pages: u32,
    pub max_pending_surfaces: u32,
    /// Worst-case scratch output for one regular page extraction. Scratch is
    /// reused serially and is not multiplied by the number of resident pages.
    pub regular: TransvoxelGpuExtractorConfig,
    /// Worst-case scratch output for one transition extraction.
    pub transition: TransvoxelGpuTransitionExtractorConfig,
    /// Aggregate compact regular-surface arena shared by all current and
    /// transactional replacement pages.
    pub regular_arena: ExtractionLimits,
    /// Aggregate compact transition-surface arena shared by all current and
    /// transactional replacement pages.
    pub transition_arena: ExtractionLimits,
    pub max_surface_bytes: u64,
}

impl PlanetaryVoxelRenderConfig {
    pub fn validation_demo() -> Self {
        Self {
            residency: PlanetaryVoxelGpuConfig::new(32, 128, 16, 32, 64)
                .expect("validation residency configuration is valid"),
            max_surface_pages: 5,
            max_pending_surfaces: 5,
            regular: TransvoxelGpuExtractorConfig::default(),
            transition: TransvoxelGpuTransitionExtractorConfig::default(),
            regular_arena: ExtractionLimits::new(5, 5, 327_680, 655_360, 10_405)
                .expect("validation regular arena configuration is valid"),
            transition_arena: ExtractionLimits::new(5, 5, 245_760, 737_280, 11_705)
                .expect("validation transition arena configuration is valid"),
            max_surface_bytes: 128 * 1024 * 1024,
        }
    }

    /// Bounded culling-heavy fixture used by the unattended matched benchmark.
    ///
    /// The extraction capacities remain comfortably above the plane fixture's
    /// actual output while avoiding the production-sized per-page reservation
    /// that would make a 64-page benchmark needlessly expensive.
    pub fn benchmark_demo() -> Self {
        Self {
            residency: PlanetaryVoxelGpuConfig::new(320, 1_024, 64, 192, 320)
                .expect("benchmark residency configuration is valid"),
            max_surface_pages: 64,
            max_pending_surfaces: 64,
            regular: TransvoxelGpuExtractorConfig::default(),
            transition: TransvoxelGpuTransitionExtractorConfig::default(),
            regular_arena: ExtractionLimits::new(64, 64, 524_288, 1_048_576, 16_704)
                .expect("benchmark regular arena configuration is valid"),
            transition_arena: ExtractionLimits::new(64, 64, 131_072, 393_216, 6_272)
                .expect("benchmark transition arena configuration is valid"),
            max_surface_bytes: 128 * 1024 * 1024,
        }
    }

    /// Atomic active/pending residency for the live planetary SDF path.
    ///
    /// Four bounded 96-page active frontiers can remain visible while their
    /// complete replacements are uploaded and extracted. The larger residency
    /// tier is the canonical CPU/GPU page-cache bound: unlike the surface
    /// arenas, it also retains the 3x3x3 SDF sampling neighborhoods and
    /// transition slabs required by those frontiers. Allocation remains
    /// independent of logical planet size.
    pub fn production_planet() -> Self {
        Self {
            residency: PlanetaryVoxelGpuConfig::new(8_192, 32_768, 64, 192, 8_192)
                .expect("production planet residency configuration is valid"),
            max_surface_pages: 384,
            max_pending_surfaces: 384,
            regular: TransvoxelGpuExtractorConfig::default(),
            transition: TransvoxelGpuTransitionExtractorConfig::default(),
            regular_arena: ExtractionLimits::new(384, 384, 6_291_456, 12_582_912, 199_729)
                .expect("production planet regular arena configuration is valid"),
            transition_arena: ExtractionLimits::new(384, 384, 786_432, 2_359_296, 37_632)
                .expect("production planet transition arena configuration is valid"),
            max_surface_bytes: 512 * 1024 * 1024,
        }
    }

    pub fn allocation_plan(self) -> Result<PlanetarySurfaceAllocationPlan, PlanetaryRenderError> {
        if self.max_surface_pages == 0 {
            return Err(PlanetaryRenderError::ZeroSurfacePages);
        }
        if self.max_surface_pages > self.residency.max_resident_pages {
            return Err(PlanetaryRenderError::SurfacePageCapacity {
                surfaces: self.max_surface_pages,
                residents: self.residency.max_resident_pages,
            });
        }
        if self.max_pending_surfaces == 0 {
            return Err(PlanetaryRenderError::ZeroPendingSurfaces);
        }
        if self.regular_arena.max_page_slots != self.max_surface_pages
            || self.transition_arena.max_page_slots != self.max_surface_pages
            || self.regular_arena.max_pending_pages < self.max_pending_surfaces
            || self.transition_arena.max_pending_pages < self.max_pending_surfaces
        {
            return Err(PlanetaryRenderError::SurfaceArenaContract);
        }
        let surface_pages = u64::from(self.max_surface_pages);
        let regular_vertex_bytes = checked_product(&[
            u64::from(self.regular_arena.max_vertices),
            core::mem::size_of::<crate::GpuTerrainVertex>() as u64,
        ])?;
        let regular_index_bytes = checked_product(&[
            u64::from(self.regular_arena.max_indices),
            core::mem::size_of::<u32>() as u64,
        ])?;
        let transition_vertex_bytes = checked_product(&[
            u64::from(self.transition_arena.max_vertices),
            core::mem::size_of::<crate::GpuTerrainVertex>() as u64,
        ])?;
        let transition_index_bytes = checked_product(&[
            u64::from(self.transition_arena.max_indices),
            core::mem::size_of::<u32>() as u64,
        ])?;
        let regular_meshlets = self.regular_arena.max_meshlets;
        let transition_meshlets = self.transition_arena.max_meshlets;
        let regular_meshlet_bytes = checked_product(&[
            u64::from(regular_meshlets),
            core::mem::size_of::<GpuTerrainMeshlet>() as u64,
        ])?;
        let regular_meshlet_bounds_bytes = checked_product(&[
            u64::from(regular_meshlets),
            core::mem::size_of::<GpuTerrainMeshletBounds>() as u64,
        ])?;
        let transition_meshlet_bytes = checked_product(&[
            u64::from(transition_meshlets),
            core::mem::size_of::<GpuTerrainMeshlet>() as u64,
        ])?;
        let transition_meshlet_bounds_bytes = checked_product(&[
            u64::from(transition_meshlets),
            core::mem::size_of::<GpuTerrainMeshletBounds>() as u64,
        ])?;
        let state_bytes = surface_pages
            .checked_mul(core::mem::size_of::<GpuSurfaceState>() as u64)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let draw_page_bytes = surface_pages
            .checked_mul(core::mem::size_of::<GpuDrawPage>() as u64)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let feedback_bytes = core::mem::size_of::<GpuSurfaceFeedback>() as u64;
        let indirect_bytes = surface_pages
            .checked_mul(DRAW_ARGS_BYTES)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let regular_meshlet_draw_capacity = u64::from(regular_meshlets);
        let transition_meshlet_draw_capacity = u64::from(transition_meshlets);
        let regular_meshlet_indirect_bytes = regular_meshlet_draw_capacity
            .checked_mul(DRAW_ARGS_BYTES)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let transition_meshlet_indirect_bytes = transition_meshlet_draw_capacity
            .checked_mul(DRAW_ARGS_BYTES)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let regular_meshlet_draw_bytes = regular_meshlet_draw_capacity
            .checked_mul(TERRAIN_DRAW_BYTES)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let transition_meshlet_draw_bytes = transition_meshlet_draw_capacity
            .checked_mul(TERRAIN_DRAW_BYTES)
            .ok_or(PlanetaryRenderError::ArithmeticOverflow)?;
        let cull_counter_bytes = core::mem::size_of::<GpuTerrainCullCounters>() as u64;
        let diagnostic_readback_bytes = [
            feedback_bytes,
            core::mem::size_of::<GpuSurfaceGatherCounters>() as u64,
            cull_counter_bytes,
            state_bytes,
            indirect_bytes,
            indirect_bytes,
        ]
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
            regular_meshlet_bytes,
            regular_meshlet_bounds_bytes,
            transition_meshlet_bytes,
            transition_meshlet_bounds_bytes,
            state_bytes,
            draw_page_bytes,
            feedback_bytes,
            indirect_bytes,
            indirect_bytes,
            regular_meshlet_indirect_bytes,
            transition_meshlet_indirect_bytes,
            regular_meshlet_draw_bytes,
            transition_meshlet_draw_bytes,
            cull_counter_bytes,
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
            regular_meshlet_bytes,
            regular_meshlet_bounds_bytes,
            transition_meshlet_bytes,
            transition_meshlet_bounds_bytes,
            state_bytes,
            draw_page_bytes,
            feedback_bytes,
            indirect_bytes,
            regular_meshlet_draw_capacity: u32::try_from(regular_meshlet_draw_capacity)
                .map_err(|_| PlanetaryRenderError::ArithmeticOverflow)?,
            transition_meshlet_draw_capacity: u32::try_from(transition_meshlet_draw_capacity)
                .map_err(|_| PlanetaryRenderError::ArithmeticOverflow)?,
            regular_meshlet_indirect_bytes,
            transition_meshlet_indirect_bytes,
            regular_meshlet_draw_bytes,
            transition_meshlet_draw_bytes,
            cull_counter_bytes,
            diagnostic_readback_bytes,
            total_bytes,
        })
    }

    fn validate_device(self, limits: &wgpu::Limits) -> Result<(), PlanetaryRenderError> {
        self.regular_arena.validate_device(limits)?;
        self.transition_arena.validate_device(limits)?;
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
            ("regular meshlet arena", plan.regular_meshlet_bytes, true),
            (
                "regular meshlet bounds",
                plan.regular_meshlet_bounds_bytes,
                true,
            ),
            (
                "transition meshlet arena",
                plan.transition_meshlet_bytes,
                true,
            ),
            (
                "transition meshlet bounds",
                plan.transition_meshlet_bounds_bytes,
                true,
            ),
            ("surface state", plan.state_bytes, true),
            ("draw pages", plan.draw_page_bytes, true),
            ("surface feedback", plan.feedback_bytes, true),
            ("regular indirect", plan.indirect_bytes, true),
            ("transition indirect", plan.indirect_bytes, true),
            (
                "regular meshlet indirect",
                plan.regular_meshlet_indirect_bytes,
                true,
            ),
            (
                "transition meshlet indirect",
                plan.transition_meshlet_indirect_bytes,
                true,
            ),
            (
                "regular meshlet draw metadata",
                plan.regular_meshlet_draw_bytes,
                true,
            ),
            (
                "transition meshlet draw metadata",
                plan.transition_meshlet_draw_bytes,
                true,
            ),
            ("meshlet cull counters", plan.cull_counter_bytes, true),
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
    pub regular_meshlet_bytes: u64,
    pub regular_meshlet_bounds_bytes: u64,
    pub transition_meshlet_bytes: u64,
    pub transition_meshlet_bounds_bytes: u64,
    pub state_bytes: u64,
    pub draw_page_bytes: u64,
    pub feedback_bytes: u64,
    pub indirect_bytes: u64,
    pub regular_meshlet_draw_capacity: u32,
    pub transition_meshlet_draw_capacity: u32,
    pub regular_meshlet_indirect_bytes: u64,
    pub transition_meshlet_indirect_bytes: u64,
    pub regular_meshlet_draw_bytes: u64,
    pub transition_meshlet_draw_bytes: u64,
    pub cull_counter_bytes: u64,
    pub diagnostic_readback_bytes: u64,
    pub total_bytes: u64,
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
    pub active_surface_pages: u32,
    pub candidate_surface_pages: u32,
    pub ready_candidate_surface_pages: u32,
    pub resident_candidate_targets: u32,
    pub missing_candidate_dependencies: u32,
    pub cpu_surface_jobs_in_flight: u32,
    pub candidate_without_publication: u32,
    pub candidate_invalid_state: u32,
    pub candidate_generation_mismatches: u32,
    pub candidate_revision_mismatches: u32,
    pub candidate_transition_mismatches: u32,
    pub gpu_submitted_jobs: u32,
    pub gpu_published_jobs: u32,
    pub gpu_stale_rejections: u32,
    pub gpu_overflow_rejections: u32,
    pub gpu_incomplete_rejections: u32,
    pub gather_regular_samples: u32,
    pub gather_transition_samples: u32,
    pub gather_table_probes: u32,
    pub gather_page_misses: u32,
    pub gather_stale_targets: u32,
    pub gather_completed: u32,
    pub resident_lods: Vec<u8>,
    pub source_generation_min: Option<SourceGeneration>,
    pub source_generation_max: Option<SourceGeneration>,
    pub publication_generation_min: Option<u64>,
    pub publication_generation_max: Option<u64>,
    pub regular_vertices: u64,
    pub regular_indices: u64,
    pub transition_vertices: u64,
    pub transition_indices: u64,
    pub regular_meshlets: u64,
    pub transition_meshlets: u64,
    pub visible_regular_draws: u32,
    pub visible_transition_draws: u32,
    pub meshlet_draw_overflow: u32,
    pub meshlet_stale_rejections: u32,
    pub meshlet_frustum_rejections: u32,
    pub meshlet_cone_rejections: u32,
    pub meshlet_invalid_candidates: u32,
    pub readback_failures: u64,
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceJob {
    slot: u32,
    resident_slot: u32,
    transition_mask: u32,
    generation_low: u32,
    generation_high: u32,
    revision: u32,
    regular_first_vertex: u32,
    regular_first_index: u32,
    regular_first_meshlet: u32,
    transition_first_vertex: u32,
    transition_first_index: u32,
    transition_first_meshlet: u32,
    _pad: [u32; 4],
}

impl GpuSurfaceJob {
    fn new(
        slot: u32,
        resident_slot: u32,
        transition_mask: u8,
        generation: u64,
        revision: u32,
        regular: ExtractionReservation,
        transition: ExtractionReservation,
    ) -> Self {
        Self {
            slot,
            resident_slot,
            transition_mask: u32::from(transition_mask),
            generation_low: generation as u32,
            generation_high: (generation >> 32) as u32,
            revision,
            regular_first_vertex: regular.allocation.vertices.first,
            regular_first_index: regular.allocation.indices.first,
            regular_first_meshlet: regular.allocation.meshlets.first,
            transition_first_vertex: transition.allocation.vertices.first,
            transition_first_index: transition.allocation.indices.first,
            transition_first_meshlet: transition.allocation.meshlets.first,
            _pad: [0; 4],
        }
    }
}

#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
struct GpuSurfaceState {
    generation_low: u32,
    generation_high: u32,
    valid: u32,
    revision: u32,
    regular_first_vertex: u32,
    regular_vertex_count: u32,
    regular_first_index: u32,
    regular_index_count: u32,
    regular_first_meshlet: u32,
    regular_meshlet_count: u32,
    transition_first_vertex: u32,
    transition_vertex_count: u32,
    transition_first_index: u32,
    transition_index_count: u32,
    transition_first_meshlet: u32,
    transition_meshlet_count: u32,
    transition_mask: u32,
    _pad: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PublishedSurface {
    request: PlanetarySurfaceRequest,
    slot: u32,
    publication_generation: u64,
    revision: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CandidateSurface {
    request: PlanetarySurfaceRequest,
    slot: u32,
    revision: u32,
    publication_generation: Option<u64>,
    regular_reservation: Option<ExtractionReservation>,
    transition_reservation: Option<ExtractionReservation>,
    ready: bool,
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
    relative_lod0_cell_min_x: [u32; 2],
    relative_lod0_cell_min_y: [u32; 2],
    relative_lod0_cell_min_z: [u32; 2],
    lod: u32,
    _pad0: u32,
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

#[derive(Clone, Copy, Debug)]
struct PreparedExtraction {
    request: PlanetarySurfaceRequest,
    slot: u32,
    resident_slot: u32,
    publication_generation: u64,
    revision: u32,
}

struct PendingExtractionReadback {
    buffer: wgpu::Buffer,
    receiver: Option<Mutex<Receiver<Result<(), wgpu::BufferAsyncError>>>>,
    extraction: PreparedExtraction,
}

#[derive(Clone, Copy, Debug)]
struct PreparedPublication {
    extraction: PreparedExtraction,
    regular: ExtractionReservation,
    transition: ExtractionReservation,
}

#[derive(Clone, Copy)]
struct DiagnosticsReadbackLayout {
    gather_counter_offset: u64,
    cull_counter_offset: u64,
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
    surface_sampler: GpuSurfaceSampler,
    pending: VecDeque<PlanetarySurfaceRequest>,
    surface_requests: BTreeMap<PlanetPageKey, PlanetarySurfaceRequest>,
    surface_dependencies: BTreeMap<PlanetPageKey, BTreeSet<PlanetPageKey>>,
    surface_dependency_generations:
        BTreeMap<PlanetPageKey, BTreeMap<PlanetPageKey, SourceGeneration>>,
    dependency_targets: BTreeMap<PlanetPageKey, BTreeSet<PlanetPageKey>>,
    invalidated_surfaces: BTreeSet<PlanetPageKey>,
    /// Surface arenas are intentionally independent from residency slots.
    /// Sampling-only halo pages may occupy any residency slot without
    /// consuming one of these bounded renderable allocations.
    surface_slots: BTreeMap<PlanetPageKey, u32>,
    requested_visible: BTreeMap<PlanetPageKey, PlanetarySurfaceRequest>,
    active_surfaces: BTreeMap<PlanetPageKey, PublishedSurface>,
    candidate_surfaces: BTreeMap<PlanetPageKey, CandidateSurface>,
    handoff_planet: Option<PlanetId>,
    next_surface_revision: u32,
    prepared_extraction: Option<PreparedExtraction>,
    prepared_publication: Option<PreparedPublication>,
    extraction_readback: Option<PendingExtractionReadback>,
    extraction_available: Option<wgpu::Buffer>,
    regular_publisher: BoundedExtractionPublisher,
    transition_publisher: BoundedExtractionPublisher,
    visible: BTreeMap<PlanetPageKey, u8>,
    counters: PlanetaryRenderCounters,
    job_buffer: wgpu::Buffer,
    state_buffer: wgpu::Buffer,
    draw_page_buffer: wgpu::Buffer,
    feedback_buffer: wgpu::Buffer,
    regular_vertex_arena: wgpu::Buffer,
    regular_index_arena: wgpu::Buffer,
    transition_vertex_arena: wgpu::Buffer,
    transition_index_arena: wgpu::Buffer,
    regular_meshlet_arena: wgpu::Buffer,
    regular_meshlet_bounds: wgpu::Buffer,
    transition_meshlet_arena: wgpu::Buffer,
    transition_meshlet_bounds: wgpu::Buffer,
    regular_indirect: wgpu::Buffer,
    transition_indirect: wgpu::Buffer,
    regular_meshlet_indirect: wgpu::Buffer,
    transition_meshlet_indirect: wgpu::Buffer,
    regular_meshlet_draws: wgpu::Buffer,
    transition_meshlet_draws: wgpu::Buffer,
    meshlet_cull_counters: wgpu::Buffer,
    regular_cull_uniform: wgpu::Buffer,
    transition_cull_uniform: wgpu::Buffer,
    debug_uniform: wgpu::Buffer,
    regular_copy_pipeline: wgpu::ComputePipeline,
    transition_copy_pipeline: wgpu::ComputePipeline,
    regular_meshlet_build_pipeline: wgpu::ComputePipeline,
    transition_meshlet_build_pipeline: wgpu::ComputePipeline,
    publish_pipeline: wgpu::ComputePipeline,
    visibility_pipeline: wgpu::ComputePipeline,
    regular_meshlet_cull_pipeline: wgpu::ComputePipeline,
    transition_meshlet_cull_pipeline: wgpu::ComputePipeline,
    regular_copy_bind_group: wgpu::BindGroup,
    transition_copy_bind_group: wgpu::BindGroup,
    regular_meshlet_build_bind_group: wgpu::BindGroup,
    transition_meshlet_build_bind_group: wgpu::BindGroup,
    publish_bind_group: wgpu::BindGroup,
    visibility_bind_group: wgpu::BindGroup,
    regular_meshlet_cull_bind_group: Option<wgpu::BindGroup>,
    transition_meshlet_cull_bind_group: Option<wgpu::BindGroup>,
    page_render_pipeline: wgpu::RenderPipeline,
    page_transition_render_pipeline: wgpu::RenderPipeline,
    meshlet_render_pipeline: wgpu::RenderPipeline,
    render_bind_group_layout: wgpu::BindGroupLayout,
    regular_render_bind_group: Option<wgpu::BindGroup>,
    transition_render_bind_group: Option<wgpu::BindGroup>,
    render_camera_key: Option<usize>,
    draw_path: PlanetaryDrawPath,
    debug_view: PlanetaryDebugView,
    use_count_indirect: bool,
    regular_meshlet_draw_capacity: u32,
    transition_meshlet_draw_capacity: u32,
    diagnostic_available: Option<wgpu::Buffer>,
    diagnostic_readback: Option<PendingDiagnosticsReadback>,
    diagnostics_cache: PlanetaryRenderDiagnostics,
    surface_format: wgpu::TextureFormat,
    attachment_mode: AttachmentMode,
}

fn drain_invalidated_surface_requests(
    invalidated: &mut BTreeSet<PlanetPageKey>,
    pending: &mut VecDeque<PlanetarySurfaceRequest>,
    requests: &BTreeMap<PlanetPageKey, PlanetarySurfaceRequest>,
    maximum_pending: usize,
    mut resident_generation: impl FnMut(PlanetPageKey) -> Option<SourceGeneration>,
) {
    let candidates = invalidated.iter().copied().collect::<Vec<_>>();
    for target in candidates {
        if pending.iter().any(|queued| queued.key == target) {
            invalidated.remove(&target);
            continue;
        }
        if pending.len() >= maximum_pending {
            break;
        }
        let Some(request) = requests.get(&target).copied() else {
            invalidated.remove(&target);
            continue;
        };
        if resident_generation(target) != Some(request.generation) {
            invalidated.remove(&target);
            continue;
        }
        pending.push_back(request);
        invalidated.remove(&target);
    }
}

fn surface_request_is_current(
    current: Option<&PlanetarySurfaceRequest>,
    requested: PlanetarySurfaceRequest,
) -> bool {
    current.is_some_and(|current| {
        current.generation == requested.generation
            && current.transition_mask == requested.transition_mask
    })
}

fn observe_candidate_surface_state(candidate: &mut CandidateSurface, state: &GpuSurfaceState) {
    let Some(publication_generation) = candidate.publication_generation else {
        return;
    };
    let state_generation =
        u64::from(state.generation_low) | (u64::from(state.generation_high) << 32);
    // Readbacks are asynchronous snapshots. Once this exact immutable
    // candidate revision has been observed on the GPU, a later stale snapshot
    // must not make the transaction incomplete again.
    candidate.ready |= state.valid != 0
        && state_generation == publication_generation
        && state.revision == candidate.revision
        && state.transition_mask == u32::from(candidate.request.transition_mask);
}

const fn should_retarget_handoff(
    target_is_stale: bool,
    has_published_coverage: bool,
    has_progress_path: bool,
) -> bool {
    target_is_stale && (has_published_coverage || !has_progress_path)
}

impl PlanetaryVoxelRenderPass {
    pub const fn config(&self) -> PlanetaryVoxelRenderConfig {
        self.config
    }

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
        let surface_sampler = GpuSurfaceSampler::new(
            device,
            &residency,
            regular_extractor.sample_buffer(),
            transition_extractor.sample_buffer(),
        )?;
        let job_buffer = create_buffer(
            device,
            "Planetary Surface Job",
            core::mem::size_of::<GpuSurfaceJob>() as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let state_buffer = create_zeroed_buffer::<GpuSurfaceState>(
            device,
            "Planetary Surface States",
            config.max_surface_pages,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let draw_page_buffer = create_zeroed_buffer::<GpuDrawPage>(
            device,
            "Planetary Draw Pages",
            config.max_surface_pages,
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
        let regular_meshlet_arena = create_buffer(
            device,
            "Planetary Regular Meshlet Arena",
            plan.regular_meshlet_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let regular_meshlet_bounds = create_buffer(
            device,
            "Planetary Regular Meshlet Bounds",
            plan.regular_meshlet_bounds_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let transition_meshlet_arena = create_buffer(
            device,
            "Planetary Transition Meshlet Arena",
            plan.transition_meshlet_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let transition_meshlet_bounds = create_buffer(
            device,
            "Planetary Transition Meshlet Bounds",
            plan.transition_meshlet_bounds_bytes,
            wgpu::BufferUsages::STORAGE,
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
        let regular_meshlet_indirect = create_buffer(
            device,
            "Planetary Regular Meshlet Indirect Draws",
            plan.regular_meshlet_indirect_bytes,
            indirect_usage,
        );
        let transition_meshlet_indirect = create_buffer(
            device,
            "Planetary Transition Meshlet Indirect Draws",
            plan.transition_meshlet_indirect_bytes,
            indirect_usage,
        );
        let regular_meshlet_draws = create_buffer(
            device,
            "Planetary Regular Meshlet Draw Metadata",
            plan.regular_meshlet_draw_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let transition_meshlet_draws = create_buffer(
            device,
            "Planetary Transition Meshlet Draw Metadata",
            plan.transition_meshlet_draw_bytes,
            wgpu::BufferUsages::STORAGE,
        );
        let meshlet_cull_counters = create_zeroed_buffer::<GpuTerrainCullCounters>(
            device,
            "Planetary Meshlet Cull Counters",
            1,
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let regular_cull_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planetary Regular Meshlet Cull Uniform"),
            contents: bytemuck::bytes_of(&GpuTerrainCullUniforms {
                meshlet_capacity: config.regular_arena.max_meshlets,
                draw_capacity: plan.regular_meshlet_draw_capacity,
                surface_kind: 0,
                _pad: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let transition_cull_uniform =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Planetary Transition Meshlet Cull Uniform"),
                contents: bytemuck::bytes_of(&GpuTerrainCullUniforms {
                    meshlet_capacity: config.transition_arena.max_meshlets,
                    draw_capacity: plan.transition_meshlet_draw_capacity,
                    surface_kind: 1,
                    _pad: 0,
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let debug_uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Planetary Terrain Debug Uniform"),
            contents: bytemuck::bytes_of(&GpuTerrainDebugUniform {
                mode: PlanetaryDebugView::Material as u32,
                draw_path: PlanetaryDrawPath::PageIndexed as u32,
                _pad: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
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
        let meshlet_build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Terrain Meshlet Build Shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_MESHLET_BUILD_WGSL.into()),
        });
        let regular_meshlet_build_pipeline =
            compute_pipeline(device, &meshlet_build_shader, "build_regular");
        let transition_meshlet_build_pipeline =
            compute_pipeline(device, &meshlet_build_shader, "build_transition");
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
                buffer_entry(8, transition_extractor.counters_buffer()),
                buffer_entry(9, transition_extractor.vertices_buffer()),
                buffer_entry(10, transition_extractor.indices_buffer()),
                buffer_entry(11, &transition_vertex_arena),
                buffer_entry(12, &transition_index_arena),
            ],
        });
        let regular_meshlet_build_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Planetary Regular Meshlet Build Bind Group"),
                layout: &regular_meshlet_build_pipeline.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, &job_buffer),
                    buffer_entry(1, residency.metadata_buffer()),
                    buffer_entry(3, regular_extractor.counters_buffer()),
                    buffer_entry(4, &regular_vertex_arena),
                    buffer_entry(5, &regular_index_arena),
                    buffer_entry(6, &regular_meshlet_arena),
                    buffer_entry(7, &regular_meshlet_bounds),
                ],
            });
        let transition_meshlet_build_bind_group =
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Planetary Transition Meshlet Build Bind Group"),
                layout: &transition_meshlet_build_pipeline.get_bind_group_layout(0),
                entries: &[
                    buffer_entry(0, &job_buffer),
                    buffer_entry(1, residency.metadata_buffer()),
                    buffer_entry(8, transition_extractor.counters_buffer()),
                    buffer_entry(9, &transition_vertex_arena),
                    buffer_entry(10, &transition_index_arena),
                    buffer_entry(11, &transition_meshlet_arena),
                    buffer_entry(12, &transition_meshlet_bounds),
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

        let meshlet_cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Terrain Meshlet Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(TERRAIN_MESHLET_CULL_WGSL.into()),
        });
        let regular_meshlet_cull_pipeline =
            compute_pipeline(device, &meshlet_cull_shader, "cull_meshlets");
        let transition_meshlet_cull_pipeline =
            compute_pipeline(device, &meshlet_cull_shader, "cull_meshlets");

        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Planetary Surface Draw Shader"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_DRAW_WGSL.into()),
        });
        let render_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Planetary Surface Draw Bind Group Layout"),
                entries: &[
                    storage_layout_entry(0, wgpu::ShaderStages::VERTEX, true),
                    storage_layout_entry(1, wgpu::ShaderStages::VERTEX_FRAGMENT, true),
                    storage_layout_entry(2, wgpu::ShaderStages::VERTEX_FRAGMENT, true),
                    uniform_layout_entry(3, wgpu::ShaderStages::VERTEX_FRAGMENT),
                ],
            });
        let render_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Planetary Surface Draw Pipeline Layout"),
            bind_group_layouts: &[Some(&render_bind_group_layout)],
            immediate_size: 0,
        });
        let create_render_pipeline = |label: &'static str, vertex_entry: &'static str| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&render_layout),
                vertex: wgpu::VertexState {
                    module: &render_shader,
                    entry_point: Some(vertex_entry),
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
            })
        };
        let page_render_pipeline =
            create_render_pipeline("Planetary Page Surface Draw Pipeline", "vs_page");
        let page_transition_render_pipeline = create_render_pipeline(
            "Planetary Page Transition Surface Draw Pipeline",
            "vs_page_transition",
        );
        let meshlet_render_pipeline =
            create_render_pipeline("Planetary Meshlet Surface Draw Pipeline", "vs_meshlet");

        Ok(Self {
            config,
            residency,
            regular_extractor,
            transition_extractor,
            surface_sampler,
            pending: VecDeque::new(),
            surface_requests: BTreeMap::new(),
            surface_dependencies: BTreeMap::new(),
            surface_dependency_generations: BTreeMap::new(),
            dependency_targets: BTreeMap::new(),
            invalidated_surfaces: BTreeSet::new(),
            surface_slots: BTreeMap::new(),
            requested_visible: BTreeMap::new(),
            active_surfaces: BTreeMap::new(),
            candidate_surfaces: BTreeMap::new(),
            handoff_planet: None,
            next_surface_revision: 1,
            prepared_extraction: None,
            prepared_publication: None,
            extraction_readback: None,
            extraction_available: None,
            regular_publisher: BoundedExtractionPublisher::new(config.regular_arena),
            transition_publisher: BoundedExtractionPublisher::new(config.transition_arena),
            visible: BTreeMap::new(),
            counters: PlanetaryRenderCounters::default(),
            job_buffer,
            state_buffer,
            draw_page_buffer,
            feedback_buffer,
            regular_vertex_arena,
            regular_index_arena,
            transition_vertex_arena,
            transition_index_arena,
            regular_meshlet_arena,
            regular_meshlet_bounds,
            transition_meshlet_arena,
            transition_meshlet_bounds,
            regular_indirect,
            transition_indirect,
            regular_meshlet_indirect,
            transition_meshlet_indirect,
            regular_meshlet_draws,
            transition_meshlet_draws,
            meshlet_cull_counters,
            regular_cull_uniform,
            transition_cull_uniform,
            debug_uniform,
            regular_copy_pipeline,
            transition_copy_pipeline,
            regular_meshlet_build_pipeline,
            transition_meshlet_build_pipeline,
            publish_pipeline,
            visibility_pipeline,
            regular_meshlet_cull_pipeline,
            transition_meshlet_cull_pipeline,
            regular_copy_bind_group,
            transition_copy_bind_group,
            regular_meshlet_build_bind_group,
            transition_meshlet_build_bind_group,
            publish_bind_group,
            visibility_bind_group,
            regular_meshlet_cull_bind_group: None,
            transition_meshlet_cull_bind_group: None,
            page_render_pipeline,
            page_transition_render_pipeline,
            meshlet_render_pipeline,
            render_bind_group_layout,
            regular_render_bind_group: None,
            transition_render_bind_group: None,
            render_camera_key: None,
            draw_path: PlanetaryDrawPath::PageIndexed,
            debug_view: PlanetaryDebugView::Material,
            use_count_indirect: device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
            regular_meshlet_draw_capacity: plan.regular_meshlet_draw_capacity,
            transition_meshlet_draw_capacity: plan.transition_meshlet_draw_capacity,
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

    pub const fn draw_path(&self) -> PlanetaryDrawPath {
        self.draw_path
    }

    pub const fn debug_view(&self) -> PlanetaryDebugView {
        self.debug_view
    }

    pub fn set_draw_path(&mut self, queue: &wgpu::Queue, draw_path: PlanetaryDrawPath) {
        self.draw_path = draw_path;
        if draw_path == PlanetaryDrawPath::PageIndexed
            && self.debug_view == PlanetaryDebugView::Meshlets
        {
            // A page-sized baseline draw has no meshlet identity to visualize.
            // Never present page colors under the truthful meshlet label.
            self.debug_view = PlanetaryDebugView::Material;
        }
        self.write_debug_uniform(queue);
    }

    pub fn toggle_draw_path(&mut self, queue: &wgpu::Queue) -> PlanetaryDrawPath {
        let draw_path = match self.draw_path {
            PlanetaryDrawPath::PageIndexed => PlanetaryDrawPath::Meshlets,
            PlanetaryDrawPath::Meshlets => PlanetaryDrawPath::PageIndexed,
        };
        self.set_draw_path(queue, draw_path);
        self.draw_path
    }

    pub fn set_debug_view(&mut self, queue: &wgpu::Queue, debug_view: PlanetaryDebugView) {
        self.debug_view = debug_view;
        if debug_view == PlanetaryDebugView::Meshlets {
            self.draw_path = PlanetaryDrawPath::Meshlets;
        }
        self.write_debug_uniform(queue);
    }

    pub fn cycle_debug_view(&mut self, queue: &wgpu::Queue) -> PlanetaryDebugView {
        let index = PlanetaryDebugView::ALL
            .iter()
            .position(|view| *view == self.debug_view)
            .unwrap_or(0);
        self.set_debug_view(
            queue,
            PlanetaryDebugView::ALL[(index + 1) % PlanetaryDebugView::ALL.len()],
        );
        self.debug_view
    }

    fn write_debug_uniform(&self, queue: &wgpu::Queue) {
        queue.write_buffer(
            &self.debug_uniform,
            0,
            bytemuck::bytes_of(&GpuTerrainDebugUniform {
                mode: self.debug_view as u32,
                draw_path: self.draw_path as u32,
                _pad: [0; 2],
            }),
        );
    }

    /// Polls a bounded asynchronous readback of publication state and starts
    /// the next snapshot when the previous one completes. This never waits for
    /// the GPU and is intended for diagnostics/tooling rather than rendering.
    pub fn poll_diagnostics(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> PlanetaryRenderDiagnostics {
        self.advance_diagnostics_readback(device, queue);
        self.refresh_cpu_diagnostics();
        self.diagnostics_cache.clone()
    }

    fn advance_diagnostics_readback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
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
                    if self.consume_diagnostics_readback(queue, &pending.buffer) {
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
        if self.diagnostic_readback.is_none() {
            self.start_diagnostics_readback(device, queue);
        }
    }

    fn diagnostics_readback_layout(&self) -> DiagnosticsReadbackLayout {
        let pages = u64::from(self.config.max_surface_pages);
        let gather_counter_offset = core::mem::size_of::<GpuSurfaceFeedback>() as u64;
        let cull_counter_offset =
            gather_counter_offset + core::mem::size_of::<GpuSurfaceGatherCounters>() as u64;
        let state_offset =
            cull_counter_offset + core::mem::size_of::<GpuTerrainCullCounters>() as u64;
        let regular_draw_offset =
            state_offset + pages * core::mem::size_of::<GpuSurfaceState>() as u64;
        let transition_draw_offset = regular_draw_offset + pages * DRAW_ARGS_BYTES;
        DiagnosticsReadbackLayout {
            gather_counter_offset,
            cull_counter_offset,
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
            self.surface_sampler.counters_buffer(),
            0,
            &readback,
            layout.gather_counter_offset,
            layout.cull_counter_offset - layout.gather_counter_offset,
        );
        encoder.copy_buffer_to_buffer(
            &self.meshlet_cull_counters,
            0,
            &readback,
            layout.cull_counter_offset,
            layout.state_offset - layout.cull_counter_offset,
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

    fn consume_diagnostics_readback(&mut self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> bool {
        let layout = self.diagnostics_readback_layout();
        let mapped = match buffer.slice(..).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(error) => {
                log::warn!("planetary diagnostics mapped range unavailable: {error:?}");
                return false;
            }
        };
        let feedback_bytes = &mapped[..layout.gather_counter_offset as usize];
        let feedback = *bytemuck::from_bytes::<GpuSurfaceFeedback>(feedback_bytes);
        let gather_counters = *bytemuck::from_bytes::<GpuSurfaceGatherCounters>(
            &mapped[layout.gather_counter_offset as usize..layout.cull_counter_offset as usize],
        );
        let cull_counters = *bytemuck::from_bytes::<GpuTerrainCullCounters>(
            &mapped[layout.cull_counter_offset as usize..layout.state_offset as usize],
        );
        let states = bytemuck::cast_slice::<u8, GpuSurfaceState>(
            &mapped[layout.state_offset as usize..layout.regular_draw_offset as usize],
        )
        .to_vec();
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
        self.diagnostics_cache.gather_regular_samples = gather_counters.regular_samples;
        self.diagnostics_cache.gather_transition_samples = gather_counters.transition_samples;
        self.diagnostics_cache.gather_table_probes = gather_counters.table_probes;
        self.diagnostics_cache.gather_page_misses = gather_counters.page_misses;
        self.diagnostics_cache.gather_stale_targets = gather_counters.stale_targets;
        self.diagnostics_cache.gather_completed = gather_counters.completed;
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
        self.diagnostics_cache.regular_meshlets = states
            .iter()
            .filter(|state| state.valid != 0)
            .map(|state| u64::from(state.regular_meshlet_count))
            .sum();
        self.diagnostics_cache.transition_meshlets = states
            .iter()
            .filter(|state| state.valid != 0)
            .map(|state| u64::from(state.transition_meshlet_count))
            .sum();
        match self.draw_path {
            PlanetaryDrawPath::PageIndexed => {
                self.diagnostics_cache.visible_regular_draws = regular_draws
                    .iter()
                    .filter(|draw| draw.instance_count != 0 && draw.index_count != 0)
                    .count() as u32;
                self.diagnostics_cache.visible_transition_draws = transition_draws
                    .iter()
                    .filter(|draw| draw.instance_count != 0 && draw.index_count != 0)
                    .count()
                    as u32;
            }
            PlanetaryDrawPath::Meshlets => {
                self.diagnostics_cache.visible_regular_draws = cull_counters.regular_draws;
                self.diagnostics_cache.visible_transition_draws = cull_counters.transition_draws;
            }
        }
        self.diagnostics_cache.meshlet_draw_overflow = cull_counters.overflow;
        self.diagnostics_cache.meshlet_stale_rejections = cull_counters.stale;
        self.diagnostics_cache.meshlet_frustum_rejections = cull_counters.frustum_rejects;
        self.diagnostics_cache.meshlet_cone_rejections = cull_counters.cone_rejects;
        self.diagnostics_cache.meshlet_invalid_candidates = cull_counters.invalid_candidates;
        drop(mapped);
        buffer.unmap();
        let mut without_publication = 0_u32;
        let mut invalid_state = 0_u32;
        let mut generation_mismatches = 0_u32;
        let mut revision_mismatches = 0_u32;
        let mut transition_mismatches = 0_u32;
        for candidate in self.candidate_surfaces.values_mut() {
            let Some(publication_generation) = candidate.publication_generation else {
                without_publication = without_publication.saturating_add(1);
                continue;
            };
            let Some(state) = states.get(candidate.slot as usize) else {
                invalid_state = invalid_state.saturating_add(1);
                continue;
            };
            let state_generation =
                u64::from(state.generation_low) | (u64::from(state.generation_high) << 32);
            if state.valid == 0 {
                invalid_state = invalid_state.saturating_add(1);
            }
            if state_generation != publication_generation {
                generation_mismatches = generation_mismatches.saturating_add(1);
            }
            if state.revision != candidate.revision {
                revision_mismatches = revision_mismatches.saturating_add(1);
            }
            if state.transition_mask != u32::from(candidate.request.transition_mask) {
                transition_mismatches = transition_mismatches.saturating_add(1);
            }
            observe_candidate_surface_state(candidate, state);
        }
        self.diagnostics_cache.candidate_without_publication = without_publication;
        self.diagnostics_cache.candidate_invalid_state = invalid_state;
        self.diagnostics_cache.candidate_generation_mismatches = generation_mismatches;
        self.diagnostics_cache.candidate_revision_mismatches = revision_mismatches;
        self.diagnostics_cache.candidate_transition_mismatches = transition_mismatches;
        if let Err(error) = self.commit_ready_handoff(queue) {
            log::error!("planetary surface handoff commit failed: {error}");
            return false;
        }
        true
    }

    fn refresh_cpu_diagnostics(&mut self) {
        self.diagnostics_cache.active_surface_pages = self.active_surfaces.len() as u32;
        self.diagnostics_cache.candidate_surface_pages = self.candidate_surfaces.len() as u32;
        self.diagnostics_cache.ready_candidate_surface_pages = self
            .candidate_surfaces
            .values()
            .filter(|candidate| candidate.ready)
            .count() as u32;
        self.diagnostics_cache.resident_candidate_targets = self
            .candidate_surfaces
            .values()
            .filter(|candidate| {
                self.residency
                    .cache()
                    .resident(candidate.request.key)
                    .is_some_and(|resident| resident.generation == candidate.request.generation)
            })
            .count() as u32;
        let mut missing_candidate_dependencies = BTreeSet::new();
        for candidate in self.candidate_surfaces.values() {
            if let Ok(dependencies) = candidate.request.required_pages() {
                missing_candidate_dependencies.extend(
                    dependencies
                        .into_iter()
                        .filter(|key| self.residency.cache().resident(*key).is_none()),
                );
            }
        }
        self.diagnostics_cache.missing_candidate_dependencies =
            missing_candidate_dependencies.len() as u32;
        self.diagnostics_cache.cpu_surface_jobs_in_flight = u32::from(
            self.prepared_publication.is_some()
                || self.prepared_extraction.is_some()
                || self.extraction_readback.is_some(),
        );
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
        let upload_keys = uploads.iter().map(|upload| upload.key).collect::<Vec<_>>();
        let outcomes = self.residency.apply_upload_batch(device, queue, uploads)?;
        let mut changed_dependencies = BTreeSet::new();
        for (key, outcome) in upload_keys.into_iter().zip(&outcomes) {
            if matches!(
                outcome,
                GpuUploadOutcome::Residency(
                    UploadOutcome::Inserted { .. } | UploadOutcome::Replaced { .. }
                )
            ) {
                changed_dependencies.insert(key);
            }
        }
        self.requeue_changed_dependencies(queue, &changed_dependencies)?;
        self.publish_draw_pages(queue)?;
        Ok(outcomes)
    }

    pub fn apply_evict_batch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        evictions: Vec<PageEvict>,
    ) -> Result<Vec<EvictOutcome>, PlanetaryRenderError> {
        let eviction_keys = evictions
            .iter()
            .map(|eviction| eviction.key)
            .collect::<Vec<_>>();
        let outcomes = self.residency.apply_evict_batch(device, queue, evictions)?;
        let changed_dependencies = eviction_keys
            .into_iter()
            .zip(&outcomes)
            .filter_map(|(key, outcome)| {
                matches!(outcome, EvictOutcome::Recorded { removed: Some(_) }).then_some(key)
            })
            .collect::<BTreeSet<_>>();
        self.requeue_changed_dependencies(queue, &changed_dependencies)?;
        self.publish_draw_pages(queue)?;
        Ok(outcomes)
    }

    pub fn remove_planet_frame(
        &mut self,
        queue: &wgpu::Queue,
        planet: PlanetId,
    ) -> Result<bool, PlanetaryRenderError> {
        let removed = self.residency.remove_planet_frame(planet)?;
        self.retire_planet_surface_state(queue, planet)?;
        Ok(removed)
    }

    pub fn recreate_gpu_resources(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<(), PlanetaryRenderError> {
        self.residency.recreate_gpu_resources(device, queue)?;
        self.publish_draw_pages(queue)?;
        Ok(())
    }

    pub fn apply_visible_set(
        &mut self,
        queue: &wgpu::Queue,
        set: VisiblePageSet,
    ) -> Result<VisibilityOutcome, PlanetaryRenderError> {
        if set.pages.len() > self.config.max_surface_pages as usize {
            return Err(PlanetaryRenderError::SurfaceCapacity {
                requested: set.pages.len(),
                maximum: self.config.max_surface_pages,
            });
        }
        let surface_requests = set
            .pages
            .iter()
            .map(|page| PlanetarySurfaceRequest {
                key: page.key,
                generation: page.generation,
                transition_mask: page.transition_mask,
                dirty_microbricks: u64::MAX,
            })
            .collect::<Vec<_>>();
        for request in &surface_requests {
            request.validate()?;
            // Validate the complete sampling neighborhood before mutating the
            // visible frontier. Registration repeats this calculation, but it
            // must not be able to fail after residency has accepted the set.
            request.required_pages()?;
        }
        let requested_visible = surface_requests
            .iter()
            .map(|request| (request.key, *request))
            .collect::<BTreeMap<_, _>>();
        self.validate_handoff_capacity(&requested_visible)?;
        let protected = self.residency_protection_set(set.frame_index, &surface_requests)?;
        let outcome = self.residency.apply_visible_set(queue, protected)?;
        if matches!(outcome, VisibilityOutcome::Applied { .. }) {
            self.requested_visible = requested_visible;
            // Never let a stream of progressively finer targets cancel the
            // first complete coarse frontier. Without published coverage the
            // renderer has nothing to preserve, so refinement can otherwise
            // outrun extraction forever and leave a permanently black frame.
            if should_retarget_handoff(
                self.handoff_target_is_stale(),
                self.handoff_planet_has_published_coverage(),
                self.handoff_has_progress_path(),
            ) {
                self.retarget_handoff(queue)?;
            }
            self.start_next_handoff(queue)?;
        }
        Ok(outcome)
    }

    /// Residency protection is wider than the surface frontier. Extraction
    /// samples a symmetric halo and optional fine-side transition slabs, while
    /// an atomic handoff may deliberately keep an older complete candidate
    /// frontier alive until its last surface is published. Protect every
    /// currently resident sampling dependency for both frontiers; otherwise a
    /// newer CPU target can evict the pages needed by the frozen handoff and
    /// leave the renderer permanently black.
    fn residency_protection_set(
        &self,
        frame_index: u64,
        requested: &[PlanetarySurfaceRequest],
    ) -> Result<VisiblePageSet, PlanetaryRenderError> {
        let candidates = self
            .candidate_surfaces
            .values()
            .map(|candidate| candidate.request)
            .collect::<Vec<_>>();
        let mut pages = BTreeMap::new();

        for request in candidates.iter().chain(requested) {
            for key in request.required_pages()? {
                let Some(resident) = self.residency.cache().resident(key) else {
                    continue;
                };
                pages.entry(key).or_insert(VisiblePage {
                    key,
                    generation: resident.generation,
                    transition_mask: 0,
                });
            }
        }

        // Target identities carry an authoritative source generation even
        // before their upload arrives. Requested targets override incidental
        // dependency entries, while frozen candidates win same-key conflicts
        // because they are the transaction that can establish first coverage.
        for request in requested {
            pages.insert(
                request.key,
                VisiblePage {
                    key: request.key,
                    generation: request.generation,
                    transition_mask: request.transition_mask,
                },
            );
        }
        for request in candidates {
            pages.insert(
                request.key,
                VisiblePage {
                    key: request.key,
                    generation: request.generation,
                    transition_mask: request.transition_mask,
                },
            );
        }

        let protected = VisiblePageSet {
            frame_index,
            pages: pages.into_values().collect(),
        };
        protected.validate(self.residency.config().max_resident_pages as usize)?;
        Ok(protected)
    }

    fn validate_handoff_capacity(
        &self,
        requested: &BTreeMap<PlanetPageKey, PlanetarySurfaceRequest>,
    ) -> Result<(), PlanetaryRenderError> {
        let planets = self
            .active_surfaces
            .keys()
            .chain(requested.keys())
            .map(|key| key.planet)
            .collect::<BTreeSet<_>>();
        for planet in planets {
            let replacement_slots = requested
                .iter()
                .filter(|(key, request)| {
                    key.planet == planet
                        && (self.invalidated_surfaces.contains(key)
                            || !self.active_surfaces.get(key).is_some_and(|active| {
                                surface_request_is_current(Some(&active.request), **request)
                            }))
                })
                .count();
            let required = self.active_surfaces.len().saturating_add(replacement_slots);
            if required > self.config.max_surface_pages as usize {
                return Err(PlanetaryRenderError::SurfaceCapacity {
                    requested: required,
                    maximum: self.config.max_surface_pages,
                });
            }
            if replacement_slots > self.config.max_pending_surfaces as usize {
                return Err(PlanetaryRenderError::PendingSurfaceCapacity {
                    maximum: self.config.max_pending_surfaces,
                });
            }
        }
        Ok(())
    }

    fn handoff_target_is_stale(&self) -> bool {
        let Some(planet) = self.handoff_planet else {
            return false;
        };
        let requested = self
            .requested_visible
            .iter()
            .filter(|(key, _)| key.planet == planet)
            .map(|(key, request)| (*key, *request))
            .collect::<BTreeMap<_, _>>();
        let candidate = self
            .candidate_surfaces
            .iter()
            .map(|(key, surface)| (*key, surface.request))
            .collect::<BTreeMap<_, _>>();
        requested != candidate
    }

    fn handoff_planet_has_published_coverage(&self) -> bool {
        self.handoff_planet
            .is_some_and(|planet| self.active_surfaces.keys().any(|key| key.planet == planet))
    }

    fn handoff_has_progress_path(&self) -> bool {
        self.prepared_publication.is_some()
            || self.prepared_extraction.is_some()
            || self.extraction_readback.is_some()
            || !self.pending.is_empty()
    }

    fn retarget_handoff(&mut self, queue: &wgpu::Queue) -> Result<(), PlanetaryRenderError> {
        let Some(planet) = self.handoff_planet else {
            return self.start_next_handoff(queue);
        };
        let requested = self
            .requested_visible
            .iter()
            .filter(|(key, _)| key.planet == planet)
            .map(|(key, request)| (*key, *request))
            .collect::<Vec<_>>();
        let previous_candidates = core::mem::take(&mut self.candidate_surfaces);
        let active_slots = self
            .active_surfaces
            .values()
            .map(|surface| surface.slot)
            .collect::<BTreeSet<_>>();
        let mut used_slots = active_slots.clone();
        let mut candidates = BTreeMap::new();

        for (key, request) in &requested {
            if self.invalidated_surfaces.contains(key) {
                continue;
            }
            if let Some(candidate) = previous_candidates
                .get(key)
                .copied()
                .filter(|candidate| surface_request_is_current(Some(&candidate.request), *request))
            {
                used_slots.insert(candidate.slot);
                candidates.insert(*key, candidate);
                continue;
            }
            if let Some(active) = self
                .active_surfaces
                .get(key)
                .copied()
                .filter(|active| surface_request_is_current(Some(&active.request), *request))
            {
                candidates.insert(
                    *key,
                    CandidateSurface {
                        request: *request,
                        slot: active.slot,
                        revision: active.revision,
                        publication_generation: Some(active.publication_generation),
                        regular_reservation: None,
                        transition_reservation: None,
                        ready: true,
                    },
                );
            }
        }

        for (key, candidate) in &previous_candidates {
            if candidates.get(key).map(|kept| kept.slot) != Some(candidate.slot)
                && !active_slots.contains(&candidate.slot)
            {
                self.clear_slot(queue, candidate.slot);
                self.cancel_candidate_reservations(*candidate);
            }
        }

        for (key, request) in requested {
            if candidates.contains_key(&key) {
                continue;
            }
            let slot = (0..self.config.max_surface_pages)
                .find(|slot| !used_slots.contains(slot))
                .ok_or(PlanetaryRenderError::SurfaceCapacity {
                    requested: used_slots.len().saturating_add(1),
                    maximum: self.config.max_surface_pages,
                })?;
            used_slots.insert(slot);
            let revision = self.next_surface_revision;
            self.next_surface_revision = self.next_surface_revision.wrapping_add(1).max(1);
            candidates.insert(
                key,
                CandidateSurface {
                    request,
                    slot,
                    revision,
                    publication_generation: None,
                    regular_reservation: None,
                    transition_reservation: None,
                    ready: false,
                },
            );
        }

        self.pending.retain(|pending| {
            candidates.get(&pending.key).is_some_and(|candidate| {
                !candidate.ready && surface_request_is_current(Some(&candidate.request), *pending)
            })
        });
        let candidate_keys = candidates.keys().copied().collect::<BTreeSet<_>>();
        for key in previous_candidates.keys().copied() {
            if !candidate_keys.contains(&key) && !self.active_surfaces.contains_key(&key) {
                self.surface_slots.remove(&key);
                self.remove_surface_request(key);
            }
        }
        self.candidate_surfaces = candidates;
        let extraction = self
            .candidate_surfaces
            .values()
            .filter(|candidate| {
                !candidate.ready
                    && !self
                        .pending
                        .iter()
                        .any(|pending| pending.key == candidate.request.key)
            })
            .map(|candidate| candidate.request)
            .collect::<Vec<_>>();
        for request in extraction {
            self.surface_slots
                .insert(request.key, self.candidate_surfaces[&request.key].slot);
            self.register_surface_request(request)?;
            self.invalidated_surfaces.insert(request.key);
        }
        self.drain_invalidated_surfaces();
        self.commit_ready_handoff(queue)
    }

    fn start_next_handoff(&mut self, queue: &wgpu::Queue) -> Result<(), PlanetaryRenderError> {
        if self.handoff_planet.is_some() {
            return Ok(());
        }
        let planets = self
            .active_surfaces
            .keys()
            .chain(self.requested_visible.keys())
            .map(|key| key.planet)
            .collect::<BTreeSet<_>>();
        let Some(planet) = planets.into_iter().find(|planet| {
            let active = self
                .active_surfaces
                .iter()
                .filter(|(key, _)| key.planet == *planet)
                .map(|(key, surface)| (*key, surface.request))
                .collect::<BTreeMap<_, _>>();
            let requested = self
                .requested_visible
                .iter()
                .filter(|(key, _)| key.planet == *planet)
                .map(|(key, request)| (*key, *request))
                .collect::<BTreeMap<_, _>>();
            active != requested
                || self
                    .invalidated_surfaces
                    .iter()
                    .any(|key| key.planet == *planet && requested.contains_key(key))
        }) else {
            return Ok(());
        };

        let requested = self
            .requested_visible
            .iter()
            .filter(|(key, _)| key.planet == planet)
            .map(|(key, request)| (*key, *request))
            .collect::<Vec<_>>();
        if requested.is_empty() {
            self.retire_active_planet(queue, planet)?;
            return self.start_next_handoff(queue);
        }

        let mut used_slots = self
            .active_surfaces
            .values()
            .map(|surface| surface.slot)
            .collect::<BTreeSet<_>>();
        let mut candidates = BTreeMap::new();
        for (key, request) in requested {
            if let Some(active) = self.active_surfaces.get(&key).copied().filter(|active| {
                !self.invalidated_surfaces.contains(&key)
                    && surface_request_is_current(Some(&active.request), request)
            }) {
                candidates.insert(
                    key,
                    CandidateSurface {
                        request,
                        slot: active.slot,
                        revision: active.revision,
                        publication_generation: Some(active.publication_generation),
                        regular_reservation: None,
                        transition_reservation: None,
                        ready: true,
                    },
                );
                continue;
            }
            let slot = (0..self.config.max_surface_pages)
                .find(|slot| !used_slots.contains(slot))
                .ok_or(PlanetaryRenderError::SurfaceCapacity {
                    requested: used_slots.len().saturating_add(1),
                    maximum: self.config.max_surface_pages,
                })?;
            used_slots.insert(slot);
            let revision = self.next_surface_revision;
            self.next_surface_revision = self.next_surface_revision.wrapping_add(1).max(1);
            candidates.insert(
                key,
                CandidateSurface {
                    request,
                    slot,
                    revision,
                    publication_generation: None,
                    regular_reservation: None,
                    transition_reservation: None,
                    ready: false,
                },
            );
        }

        self.handoff_planet = Some(planet);
        self.candidate_surfaces = candidates;
        let extraction = self
            .candidate_surfaces
            .values()
            .filter(|candidate| !candidate.ready)
            .map(|candidate| candidate.request)
            .collect::<Vec<_>>();
        for request in extraction {
            self.surface_slots
                .insert(request.key, self.candidate_surfaces[&request.key].slot);
            self.register_surface_request(request)?;
            self.invalidated_surfaces.insert(request.key);
        }
        self.drain_invalidated_surfaces();
        self.commit_ready_handoff(queue)?;
        Ok(())
    }

    fn retire_planet_surface_state(
        &mut self,
        queue: &wgpu::Queue,
        planet: PlanetId,
    ) -> Result<(), PlanetaryRenderError> {
        self.requested_visible.retain(|key, _| key.planet != planet);
        if self.handoff_planet == Some(planet) {
            self.handoff_planet = None;
            for (key, candidate) in core::mem::take(&mut self.candidate_surfaces) {
                self.clear_slot(queue, candidate.slot);
                self.cancel_candidate_reservations(candidate);
                self.surface_slots.remove(&key);
                self.remove_surface_request(key);
            }
        }
        let requests = self
            .surface_requests
            .keys()
            .filter(|key| key.planet == planet)
            .copied()
            .collect::<Vec<_>>();
        for key in requests {
            self.remove_surface_request(key);
        }
        self.pending.retain(|request| request.key.planet != planet);
        self.invalidated_surfaces.retain(|key| key.planet != planet);
        self.retire_active_planet(queue, planet)?;
        self.start_next_handoff(queue)
    }

    fn retire_active_planet(
        &mut self,
        queue: &wgpu::Queue,
        planet: PlanetId,
    ) -> Result<(), PlanetaryRenderError> {
        let retired = self
            .active_surfaces
            .keys()
            .filter(|key| key.planet == planet)
            .copied()
            .collect::<Vec<_>>();
        for key in retired {
            if let Some(surface) = self.active_surfaces.remove(&key) {
                self.clear_slot(queue, surface.slot);
                let _ = self
                    .regular_publisher
                    .evict(key, u64::from(surface.revision));
                let _ = self
                    .transition_publisher
                    .evict(key, u64::from(surface.revision));
            }
            self.visible.remove(&key);
            self.surface_slots.remove(&key);
            self.remove_surface_request(key);
        }
        self.publish_draw_pages(queue)
    }

    fn commit_ready_handoff(&mut self, queue: &wgpu::Queue) -> Result<(), PlanetaryRenderError> {
        let Some(planet) = self.handoff_planet else {
            return Ok(());
        };
        if self
            .candidate_surfaces
            .values()
            .any(|surface| !surface.ready)
        {
            return Ok(());
        }
        let previous = self
            .active_surfaces
            .iter()
            .filter(|(key, _)| key.planet == planet)
            .map(|(key, surface)| (*key, *surface))
            .collect::<BTreeMap<_, _>>();
        let candidates = core::mem::take(&mut self.candidate_surfaces);
        for candidate in candidates.values() {
            match (
                candidate.regular_reservation,
                candidate.transition_reservation,
            ) {
                (Some(regular), Some(transition)) => {
                    match self.regular_publisher.publish(regular)? {
                        PublicationOutcome::Published { .. } => {}
                        PublicationOutcome::Stale { .. } => {
                            return Err(PlanetaryRenderError::StaleSurfaceAllocation(
                                candidate.request.key,
                            ));
                        }
                    }
                    match self.transition_publisher.publish(transition)? {
                        PublicationOutcome::Published { .. } => {}
                        PublicationOutcome::Stale { .. } => {
                            return Err(PlanetaryRenderError::StaleSurfaceAllocation(
                                candidate.request.key,
                            ));
                        }
                    }
                }
                (None, None) => {}
                _ => {
                    return Err(PlanetaryRenderError::IncompleteSurfaceAllocation(
                        candidate.request.key,
                    ));
                }
            }
        }
        let retained_slots = candidates
            .values()
            .map(|candidate| candidate.slot)
            .collect::<BTreeSet<_>>();
        for (key, surface) in previous {
            self.active_surfaces.remove(&key);
            if !retained_slots.contains(&surface.slot) {
                self.clear_slot(queue, surface.slot);
            }
            if !candidates.contains_key(&key) {
                let _ = self
                    .regular_publisher
                    .evict(key, u64::from(surface.revision));
                let _ = self
                    .transition_publisher
                    .evict(key, u64::from(surface.revision));
                self.surface_slots.remove(&key);
                self.remove_surface_request(key);
            }
            self.visible.remove(&key);
        }
        for (key, candidate) in candidates {
            let publication_generation = candidate
                .publication_generation
                .expect("ready candidate has a publication generation");
            self.surface_slots.insert(key, candidate.slot);
            self.visible.insert(key, candidate.request.transition_mask);
            self.active_surfaces.insert(
                key,
                PublishedSurface {
                    request: candidate.request,
                    slot: candidate.slot,
                    publication_generation,
                    revision: candidate.revision,
                },
            );
        }
        self.handoff_planet = None;
        self.publish_draw_pages(queue)?;
        self.start_next_handoff(queue)
    }

    fn cancel_candidate_reservations(&mut self, candidate: CandidateSurface) {
        let generation = u64::from(candidate.revision);
        if candidate.regular_reservation.is_some() {
            let _ = self
                .regular_publisher
                .cancel_pending(candidate.request.key, generation);
        }
        if candidate.transition_reservation.is_some() {
            let _ = self
                .transition_publisher
                .cancel_pending(candidate.request.key, generation);
        }
    }

    pub fn queue_surface(
        &mut self,
        request: PlanetarySurfaceRequest,
    ) -> Result<(), PlanetaryRenderError> {
        request.validate()?;
        request.required_pages()?;
        let resident = self
            .residency
            .cache()
            .resident(request.key)
            .ok_or(PlanetaryRenderError::SurfacePageNotResident(request.key))?;
        if resident.generation != request.generation {
            self.counters.stale_surface_rejections =
                self.counters.stale_surface_rejections.saturating_add(1);
            return Err(PlanetaryRenderError::SurfaceGeneration {
                key: request.key,
                expected: resident.generation,
                actual: request.generation,
            });
        }
        if let Some(index) = self
            .pending
            .iter()
            .position(|pending| pending.key == request.key)
        {
            if request.generation < self.pending[index].generation {
                self.counters.stale_surface_rejections =
                    self.counters.stale_surface_rejections.saturating_add(1);
                return Err(PlanetaryRenderError::PendingSurfaceStale(request.key));
            }
            self.register_surface_request(request)?;
            self.pending[index] = request;
            return Ok(());
        }
        if self.pending.len() == self.config.max_pending_surfaces as usize {
            self.counters.pending_backpressure =
                self.counters.pending_backpressure.saturating_add(1);
            return Err(PlanetaryRenderError::PendingSurfaceCapacity {
                maximum: self.config.max_pending_surfaces,
            });
        }
        self.ensure_surface_slot(request.key)?;
        self.register_surface_request(request)?;
        self.pending.push_back(request);
        Ok(())
    }

    fn ensure_surface_slot(&mut self, key: PlanetPageKey) -> Result<u32, PlanetaryRenderError> {
        if let Some(slot) = self.surface_slots.get(&key).copied() {
            return Ok(slot);
        }
        let used_slots = self
            .active_surfaces
            .values()
            .map(|surface| surface.slot)
            .chain(self.candidate_surfaces.values().map(|surface| surface.slot))
            .chain(self.surface_slots.values().copied())
            .collect::<BTreeSet<_>>();
        let slot = (0..self.config.max_surface_pages)
            .find(|slot| !used_slots.contains(slot))
            .ok_or(PlanetaryRenderError::SurfaceCapacity {
                requested: used_slots.len().saturating_add(1),
                maximum: self.config.max_surface_pages,
            })?;
        self.surface_slots.insert(key, slot);
        Ok(slot)
    }

    fn register_surface_request(
        &mut self,
        request: PlanetarySurfaceRequest,
    ) -> Result<(), PlanetaryRenderError> {
        let dependencies = request.required_pages()?;
        self.remove_surface_dependencies(request.key);
        for dependency in &dependencies {
            self.dependency_targets
                .entry(*dependency)
                .or_default()
                .insert(request.key);
        }
        let generations = dependencies
            .iter()
            .filter_map(|dependency| {
                self.residency
                    .cache()
                    .resident(*dependency)
                    .map(|resident| (*dependency, resident.generation))
            })
            .collect();
        self.surface_dependencies.insert(request.key, dependencies);
        self.surface_dependency_generations
            .insert(request.key, generations);
        self.surface_requests.insert(request.key, request);
        Ok(())
    }

    fn remove_surface_dependencies(&mut self, target: PlanetPageKey) {
        let Some(dependencies) = self.surface_dependencies.remove(&target) else {
            return;
        };
        for dependency in dependencies {
            let remove_entry =
                self.dependency_targets
                    .get_mut(&dependency)
                    .is_some_and(|targets| {
                        targets.remove(&target);
                        targets.is_empty()
                    });
            if remove_entry {
                self.dependency_targets.remove(&dependency);
            }
        }
    }

    fn remove_surface_request(&mut self, target: PlanetPageKey) {
        self.surface_requests.remove(&target);
        self.surface_dependency_generations.remove(&target);
        self.remove_surface_dependencies(target);
        self.invalidated_surfaces.remove(&target);
        self.pending.retain(|request| request.key != target);
    }

    fn requeue_changed_dependencies(
        &mut self,
        queue: &wgpu::Queue,
        changed: &BTreeSet<PlanetPageKey>,
    ) -> Result<(), PlanetaryRenderError> {
        let mut targets = BTreeSet::new();
        for dependency in changed {
            let current_generation = self
                .residency
                .cache()
                .resident(*dependency)
                .map(|resident| resident.generation);
            let Some(dependents) = self.dependency_targets.get(dependency) else {
                continue;
            };
            for target in dependents {
                let expected_generation = self
                    .surface_dependency_generations
                    .get(target)
                    .and_then(|generations| generations.get(dependency))
                    .copied();
                if current_generation != expected_generation {
                    targets.insert(*target);
                }
            }
        }
        self.invalidated_surfaces.extend(targets);
        self.start_next_handoff(queue)?;
        self.drain_invalidated_surfaces();
        Ok(())
    }

    fn drain_invalidated_surfaces(&mut self) {
        let candidate_requests = self
            .candidate_surfaces
            .iter()
            .filter(|(_, candidate)| !candidate.ready)
            .map(|(key, candidate)| (*key, candidate.request))
            .collect::<BTreeMap<_, _>>();
        drain_invalidated_surface_requests(
            &mut self.invalidated_surfaces,
            &mut self.pending,
            &candidate_requests,
            self.config.max_pending_surfaces as usize,
            |target| {
                self.residency
                    .cache()
                    .resident(target)
                    .map(|resident| resident.generation)
            },
        );
    }

    fn refresh_surface_dependency_generations(&mut self, target: PlanetPageKey) {
        let Some(dependencies) = self.surface_dependencies.get(&target) else {
            return;
        };
        let generations = dependencies
            .iter()
            .filter_map(|dependency| {
                self.residency
                    .cache()
                    .resident(*dependency)
                    .map(|resident| (*dependency, resident.generation))
            })
            .collect();
        self.surface_dependency_generations
            .insert(target, generations);
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
        self.counters.cleared_slots = self.counters.cleared_slots.saturating_add(1);
    }

    fn publish_draw_pages(&self, queue: &wgpu::Queue) -> Result<(), PlanetaryRenderError> {
        let mut pages = vec![GpuDrawPage::default(); self.config.max_surface_pages as usize];
        for (key, surface) in &self.active_surfaces {
            let frame = self
                .residency
                .planet_frame(key.planet)
                .ok_or(PlanetaryRenderError::MissingPlanetFrame(key.planet))?;
            let meta = GpuPageMeta::new(key.page, surface.slot, surface.publication_generation, 0)?;
            let relative_min = key
                .page
                .relative_lod0_cell_min(frame.frame_origin_lod0_cell())?;
            pages[surface.slot as usize] = GpuDrawPage {
                relative_lod0_cell_min_x: split_i64(relative_min[0]),
                relative_lod0_cell_min_y: split_i64(relative_min[1]),
                relative_lod0_cell_min_z: split_i64(relative_min[2]),
                lod: meta.lod,
                _pad0: 0,
                camera_relative_m: frame.camera_relative_m,
                lod0_cell_size_m: frame.lod0_cell_size_m,
                generation_low: surface.publication_generation as u32,
                generation_high: (surface.publication_generation >> 32) as u32,
                transition_mask: u32::from(surface.request.transition_mask),
                visible: 1,
            };
        }
        queue.write_buffer(&self.draw_page_buffer, 0, bytemuck::cast_slice(&pages));
        Ok(())
    }

    fn advance_extraction_readback(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if let Some(pending) = self
            .extraction_readback
            .as_mut()
            .filter(|pending| pending.receiver.is_none())
        {
            let (sender, receiver) = mpsc::channel();
            pending
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let _ = sender.send(result);
                });
            pending.receiver = Some(Mutex::new(receiver));
        }
        let _ = device.poll(wgpu::PollType::Poll);
        let completion = self.extraction_readback.as_ref().and_then(|pending| {
            let receiver = pending.receiver.as_ref()?;
            match receiver
                .lock()
                .expect("planetary extraction receiver mutex is not poisoned")
                .try_recv()
            {
                Ok(result) => Some(result),
                Err(TryRecvError::Empty) => None,
                Err(TryRecvError::Disconnected) => Some(Err(wgpu::BufferAsyncError)),
            }
        });
        let Some(completion) = completion else {
            return;
        };
        let pending = self
            .extraction_readback
            .take()
            .expect("completed planetary extraction readback exists");
        if let Err(error) = completion {
            log::error!("planetary extraction counter readback failed: {error:?}");
            self.invalidated_surfaces
                .insert(pending.extraction.request.key);
            self.drain_invalidated_surfaces();
            return;
        }
        let mapped = match pending.buffer.slice(..).get_mapped_range() {
            Ok(mapped) => mapped,
            Err(error) => {
                log::error!("planetary extraction counters unavailable: {error:?}");
                self.invalidated_surfaces
                    .insert(pending.extraction.request.key);
                self.drain_invalidated_surfaces();
                return;
            }
        };
        let regular_bytes = core::mem::size_of::<GpuTransvoxelEmissionCounters>();
        let regular =
            *bytemuck::from_bytes::<GpuTransvoxelEmissionCounters>(&mapped[..regular_bytes]);
        let transition =
            *bytemuck::from_bytes::<GpuTransvoxelTransitionCounters>(&mapped[regular_bytes..]);
        drop(mapped);
        pending.buffer.unmap();
        self.extraction_available = Some(pending.buffer);

        let extraction = pending.extraction;
        let candidate_is_current = self
            .candidate_surfaces
            .get(&extraction.request.key)
            .is_some_and(|candidate| {
                candidate.revision == extraction.revision
                    && surface_request_is_current(Some(&candidate.request), extraction.request)
            });
        let resident_is_current = self
            .residency
            .cache()
            .resident(extraction.request.key)
            .is_some_and(|resident| {
                resident.slot == extraction.resident_slot
                    && resident.publication_generation == extraction.publication_generation
            });
        if !candidate_is_current || !resident_is_current {
            return;
        }
        if regular.completed == 0
            || transition.completed == 0
            || regular.vertex_overflow != 0
            || regular.index_overflow != 0
            || transition.vertex_overflow != 0
            || transition.index_overflow != 0
        {
            log::error!(
                "planetary worst-case scratch extraction overflowed: regular {}/{} transition {}/{}",
                regular.required_vertices,
                regular.required_indices,
                transition.required_vertices,
                transition.required_indices,
            );
            self.invalidated_surfaces.insert(extraction.request.key);
            self.drain_invalidated_surfaces();
            return;
        }
        let regular_counts = SurfaceCounts {
            vertices: regular.emitted_vertices,
            indices: regular.emitted_indices,
            meshlets: max_meshlets_for_indices(regular.emitted_indices),
        };
        let transition_counts = SurfaceCounts {
            vertices: transition.emitted_vertices,
            indices: transition.emitted_indices,
            meshlets: max_meshlets_for_indices(transition.emitted_indices),
        };
        let allocation_generation = u64::from(extraction.revision);
        let regular_reservation = match self.regular_publisher.reserve(
            extraction.request.key,
            allocation_generation,
            regular_counts,
        ) {
            Ok(
                ReservationOutcome::Reserved(reservation)
                | ReservationOutcome::DuplicatePending(reservation),
            ) => reservation,
            Ok(ReservationOutcome::Current(_)) | Ok(ReservationOutcome::Stale { .. }) => return,
            Err(error) => {
                log::error!("planetary regular surface arena backpressure: {error}");
                self.invalidated_surfaces.insert(extraction.request.key);
                self.drain_invalidated_surfaces();
                return;
            }
        };
        let transition_reservation = match self.transition_publisher.reserve(
            extraction.request.key,
            allocation_generation,
            transition_counts,
        ) {
            Ok(
                ReservationOutcome::Reserved(reservation)
                | ReservationOutcome::DuplicatePending(reservation),
            ) => reservation,
            Ok(ReservationOutcome::Current(_)) | Ok(ReservationOutcome::Stale { .. }) => {
                let _ = self
                    .regular_publisher
                    .cancel_pending(extraction.request.key, allocation_generation);
                return;
            }
            Err(error) => {
                let _ = self
                    .regular_publisher
                    .cancel_pending(extraction.request.key, allocation_generation);
                log::error!("planetary transition surface arena backpressure: {error}");
                self.invalidated_surfaces.insert(extraction.request.key);
                self.drain_invalidated_surfaces();
                return;
            }
        };
        if let Some(candidate) = self.candidate_surfaces.get_mut(&extraction.request.key) {
            candidate.regular_reservation = Some(regular_reservation);
            candidate.transition_reservation = Some(transition_reservation);
        }
        let job = GpuSurfaceJob::new(
            extraction.slot,
            extraction.resident_slot,
            extraction.request.transition_mask,
            extraction.publication_generation,
            extraction.revision,
            regular_reservation,
            transition_reservation,
        );
        queue.write_buffer(&self.job_buffer, 0, bytemuck::bytes_of(&job));
        self.prepared_publication = Some(PreparedPublication {
            extraction,
            regular: regular_reservation,
            transition: transition_reservation,
        });
    }

    fn begin_extraction_readback(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        extraction: PreparedExtraction,
    ) {
        let regular_bytes = core::mem::size_of::<GpuTransvoxelEmissionCounters>() as u64;
        let transition_bytes = core::mem::size_of::<GpuTransvoxelTransitionCounters>() as u64;
        let readback = self.extraction_available.take().unwrap_or_else(|| {
            create_buffer(
                device,
                "Planetary Extraction Counter Readback",
                regular_bytes + transition_bytes,
                wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            )
        });
        encoder.copy_buffer_to_buffer(
            self.regular_extractor.counters_buffer(),
            0,
            &readback,
            0,
            regular_bytes,
        );
        encoder.copy_buffer_to_buffer(
            self.transition_extractor.counters_buffer(),
            0,
            &readback,
            regular_bytes,
            transition_bytes,
        );
        self.extraction_readback = Some(PendingExtractionReadback {
            buffer: readback,
            receiver: None,
            extraction,
        });
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
        self.advance_extraction_readback(ctx.device, ctx.queue);
        self.advance_diagnostics_readback(ctx.device, ctx.queue);
        self.drain_invalidated_surfaces();
        if self.prepared_publication.is_some()
            || self.prepared_extraction.is_some()
            || self.extraction_readback.is_some()
        {
            return Ok(());
        }
        let queued = self.pending.len();
        for _ in 0..queued {
            let front = self
                .pending
                .pop_front()
                .expect("queued count came from this bounded queue");
            let Some(resident) = self.residency.cache().resident(front.key) else {
                self.counters.stale_surface_rejections =
                    self.counters.stale_surface_rejections.saturating_add(1);
                continue;
            };
            if resident.generation != front.generation {
                self.counters.stale_surface_rejections =
                    self.counters.stale_surface_rejections.saturating_add(1);
                continue;
            }
            let Some(surface_slot) = self.surface_slots.get(&front.key).copied() else {
                self.counters.stale_surface_rejections =
                    self.counters.stale_surface_rejections.saturating_add(1);
                continue;
            };
            let revision = self
                .candidate_surfaces
                .get(&front.key)
                .filter(|candidate| surface_request_is_current(Some(&candidate.request), front))
                .map_or(0, |candidate| candidate.revision);
            let dependencies = match front.required_pages() {
                Ok(dependencies) => dependencies,
                Err(error) => {
                    log::error!("planetary surface dependency planning failed: {error}");
                    continue;
                }
            };
            if dependencies
                .iter()
                .any(|key| self.residency.cache().resident(*key).is_none())
            {
                self.pending.push_back(front);
                continue;
            }
            if self.residency.planet_frame(front.key.planet).is_none() {
                self.pending.push_back(front);
                continue;
            }
            let resident_slot = resident.slot;
            let publication_generation = resident.publication_generation;
            let metadata = match GpuPageMeta::new(
                front.key.page,
                resident_slot,
                publication_generation,
                front.transition_mask,
            ) {
                Ok(metadata) => metadata,
                Err(error) => {
                    log::error!("planetary surface metadata failed: {error}");
                    continue;
                }
            };
            self.surface_sampler.prepare(
                ctx.queue,
                GpuSurfaceGatherJob::new(front, metadata, self.residency.publication_epoch()),
            );
            self.regular_extractor.prepare_gpu_samples(
                ctx.queue,
                resident.publication_generation,
                front.dirty_microbricks,
                front.transition_mask,
            );
            if let Err(error) = self.transition_extractor.prepare_gpu_samples(
                ctx.queue,
                front.transition_mask,
                resident.publication_generation,
            ) {
                log::error!("planetary transition extraction prepare failed: {error}");
                continue;
            }
            self.refresh_surface_dependency_generations(front.key);
            self.prepared_extraction = Some(PreparedExtraction {
                request: front,
                slot: surface_slot,
                resident_slot,
                publication_generation,
                revision,
            });
            break;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let compute = unsafe { &mut *ctx.compute_encoder_ptr };
        if let Some(publication) = self.prepared_publication.take() {
            let regular_counts = publication.regular.allocation.counts();
            let transition_counts = publication.transition.allocation.counts();
            dispatch_compute(
                compute,
                &self.regular_copy_pipeline,
                &self.regular_copy_bind_group,
                regular_counts
                    .vertices
                    .max(regular_counts.indices)
                    .div_ceil(COPY_WORKGROUP_SIZE),
                "Planetary Regular Surface Copy",
            );
            dispatch_compute(
                compute,
                &self.transition_copy_pipeline,
                &self.transition_copy_bind_group,
                transition_counts
                    .vertices
                    .max(transition_counts.indices)
                    .div_ceil(COPY_WORKGROUP_SIZE),
                "Planetary Transition Surface Copy",
            );
            dispatch_compute(
                compute,
                &self.regular_meshlet_build_pipeline,
                &self.regular_meshlet_build_bind_group,
                regular_counts.meshlets.div_ceil(COPY_WORKGROUP_SIZE),
                "Planetary Regular Meshlet Build",
            );
            dispatch_compute(
                compute,
                &self.transition_meshlet_build_pipeline,
                &self.transition_meshlet_build_bind_group,
                transition_counts.meshlets.div_ceil(COPY_WORKGROUP_SIZE),
                "Planetary Transition Meshlet Build",
            );
            dispatch_compute(
                compute,
                &self.publish_pipeline,
                &self.publish_bind_group,
                1,
                "Planetary Surface Publish",
            );
            if let Some(candidate) = self
                .candidate_surfaces
                .get_mut(&publication.extraction.request.key)
                .filter(|candidate| candidate.revision == publication.extraction.revision)
            {
                candidate.publication_generation =
                    Some(publication.extraction.publication_generation);
                candidate.ready = false;
            }
            self.counters.submitted_jobs = self.counters.submitted_jobs.saturating_add(1);
        } else if let Some(extraction) = self.prepared_extraction.take() {
            self.surface_sampler.encode(compute);
            self.regular_extractor.encode_indirect(
                compute,
                self.surface_sampler.indirect_buffer(),
                REGULAR_EXTRACTION_INDIRECT_OFFSETS,
            );
            self.transition_extractor.encode_indirect(
                compute,
                self.surface_sampler.indirect_buffer(),
                TRANSITION_EXTRACTION_INDIRECT_OFFSETS,
            );
            self.begin_extraction_readback(ctx.device, compute, extraction);
        }
        dispatch_compute(
            compute,
            &self.visibility_pipeline,
            &self.visibility_bind_group,
            self.config.max_surface_pages.div_ceil(COPY_WORKGROUP_SIZE),
            "Planetary Surface Visibility",
        );

        let camera_key = ctx.scene.camera as *const _ as usize;
        if self.render_camera_key != Some(camera_key) {
            self.regular_meshlet_cull_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Planetary Regular Meshlet Cull Bind Group"),
                    layout: &self.regular_meshlet_cull_pipeline.get_bind_group_layout(0),
                    entries: &[
                        buffer_entry(0, ctx.scene.camera),
                        buffer_entry(1, &self.regular_cull_uniform),
                        buffer_entry(2, &self.state_buffer),
                        buffer_entry(3, &self.draw_page_buffer),
                        buffer_entry(4, &self.regular_meshlet_arena),
                        buffer_entry(5, &self.regular_meshlet_bounds),
                        buffer_entry(6, &self.regular_meshlet_indirect),
                        buffer_entry(7, &self.regular_meshlet_draws),
                        buffer_entry(8, &self.meshlet_cull_counters),
                    ],
                }));
            self.transition_meshlet_cull_bind_group = Some(
                ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Planetary Transition Meshlet Cull Bind Group"),
                    layout: &self
                        .transition_meshlet_cull_pipeline
                        .get_bind_group_layout(0),
                    entries: &[
                        buffer_entry(0, ctx.scene.camera),
                        buffer_entry(1, &self.transition_cull_uniform),
                        buffer_entry(2, &self.state_buffer),
                        buffer_entry(3, &self.draw_page_buffer),
                        buffer_entry(4, &self.transition_meshlet_arena),
                        buffer_entry(5, &self.transition_meshlet_bounds),
                        buffer_entry(6, &self.transition_meshlet_indirect),
                        buffer_entry(7, &self.transition_meshlet_draws),
                        buffer_entry(8, &self.meshlet_cull_counters),
                    ],
                }),
            );
            self.regular_render_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Planetary Regular Surface Draw Bind Group"),
                    layout: &self.render_bind_group_layout,
                    entries: &[
                        buffer_entry(0, ctx.scene.camera),
                        buffer_entry(1, &self.draw_page_buffer),
                        buffer_entry(2, &self.regular_meshlet_draws),
                        buffer_entry(3, &self.debug_uniform),
                    ],
                }));
            self.transition_render_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Planetary Transition Surface Draw Bind Group"),
                    layout: &self.render_bind_group_layout,
                    entries: &[
                        buffer_entry(0, ctx.scene.camera),
                        buffer_entry(1, &self.draw_page_buffer),
                        buffer_entry(2, &self.transition_meshlet_draws),
                        buffer_entry(3, &self.debug_uniform),
                    ],
                }));
            self.render_camera_key = Some(camera_key);
        }

        // Keep diagnostics truthful on the page baseline as well: no meshlet
        // work in that mode means zero current-frame meshlet counters.
        compute.clear_buffer(&self.meshlet_cull_counters, 0, None);
        if self.draw_path == PlanetaryDrawPath::Meshlets {
            if !self.use_count_indirect {
                compute.clear_buffer(&self.regular_meshlet_indirect, 0, None);
                compute.clear_buffer(&self.transition_meshlet_indirect, 0, None);
            }
            dispatch_compute(
                compute,
                &self.regular_meshlet_cull_pipeline,
                self.regular_meshlet_cull_bind_group
                    .as_ref()
                    .expect("regular meshlet cull bind group"),
                self.regular_meshlet_draw_capacity
                    .div_ceil(COPY_WORKGROUP_SIZE),
                "Planetary Regular Meshlet Cull",
            );
            dispatch_compute(
                compute,
                &self.transition_meshlet_cull_pipeline,
                self.transition_meshlet_cull_bind_group
                    .as_ref()
                    .expect("transition meshlet cull bind group"),
                self.transition_meshlet_draw_capacity
                    .div_ceil(COPY_WORKGROUP_SIZE),
                "Planetary Transition Meshlet Cull",
            );
        }

        let render = unsafe { &mut *ctx.active_render_pass_ptr().expect("render pass is active") };
        render.set_pipeline(match self.draw_path {
            PlanetaryDrawPath::PageIndexed => &self.page_render_pipeline,
            PlanetaryDrawPath::Meshlets => &self.meshlet_render_pipeline,
        });
        render.set_bind_group(
            0,
            self.regular_render_bind_group
                .as_ref()
                .expect("regular render bind group"),
            &[],
        );
        render.set_vertex_buffer(0, self.regular_vertex_arena.slice(..));
        render.set_index_buffer(
            self.regular_index_arena.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        match self.draw_path {
            PlanetaryDrawPath::PageIndexed => {
                draw_indirect_range(
                    render,
                    &self.regular_indirect,
                    self.config.max_surface_pages,
                );
            }
            PlanetaryDrawPath::Meshlets if self.use_count_indirect => {
                render.multi_draw_indexed_indirect_count(
                    &self.regular_meshlet_indirect,
                    0,
                    &self.meshlet_cull_counters,
                    0,
                    self.regular_meshlet_draw_capacity,
                );
            }
            PlanetaryDrawPath::Meshlets => {
                draw_indirect_range(
                    render,
                    &self.regular_meshlet_indirect,
                    self.regular_meshlet_draw_capacity,
                );
            }
        }
        render.set_bind_group(
            0,
            self.transition_render_bind_group
                .as_ref()
                .expect("transition render bind group"),
            &[],
        );
        render.set_vertex_buffer(0, self.transition_vertex_arena.slice(..));
        render.set_index_buffer(
            self.transition_index_arena.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        if self.draw_path == PlanetaryDrawPath::PageIndexed {
            render.set_pipeline(&self.page_transition_render_pipeline);
        }
        match self.draw_path {
            PlanetaryDrawPath::PageIndexed => {
                draw_indirect_range(
                    render,
                    &self.transition_indirect,
                    self.config.max_surface_pages,
                );
            }
            PlanetaryDrawPath::Meshlets if self.use_count_indirect => {
                render.multi_draw_indexed_indirect_count(
                    &self.transition_meshlet_indirect,
                    0,
                    &self.meshlet_cull_counters,
                    core::mem::size_of::<u32>() as u64,
                    self.transition_meshlet_draw_capacity,
                );
            }
            PlanetaryDrawPath::Meshlets => {
                draw_indirect_range(
                    render,
                    &self.transition_meshlet_indirect,
                    self.transition_meshlet_draw_capacity,
                );
            }
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

fn draw_indirect_range(render: &mut wgpu::RenderPass<'_>, buffer: &wgpu::Buffer, count: u32) {
    #[cfg(not(target_arch = "wasm32"))]
    render.multi_draw_indexed_indirect(buffer, 0, count);
    #[cfg(target_arch = "wasm32")]
    for index in 0..count {
        render.draw_indexed_indirect(buffer, u64::from(index) * DRAW_ARGS_BYTES);
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
    #[error(transparent)]
    SurfaceSampling(#[from] SurfaceSamplingError),
    #[error("planetary render queue must have at least one pending-surface slot")]
    ZeroPendingSurfaces,
    #[error("planetary surface arenas must match the page and pending publication contract")]
    SurfaceArenaContract,
    #[error("planetary surface allocation for {0:?} became stale before handoff commit")]
    StaleSurfaceAllocation(PlanetPageKey),
    #[error("planetary surface allocation for {0:?} is only partially reserved")]
    IncompleteSurfaceAllocation(PlanetPageKey),
    #[error(transparent)]
    Extraction(#[from] ExtractionError),
    #[error("planetary render allocation must reserve at least one surface page")]
    ZeroSurfacePages,
    #[error(
        "planetary render requests {surfaces} surface pages but residency has only {residents} slots"
    )]
    SurfacePageCapacity { surfaces: u32, residents: u32 },
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
    #[error("surface page {0:?} is not resident")]
    SurfacePageNotResident(PlanetPageKey),
    #[error("planetary visible set requests {requested} surfaces; capacity is {maximum}")]
    SurfaceCapacity { requested: usize, maximum: u32 },
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
    use helio_planet_voxel_core::{
        CellWord, PAGE_CELL_COUNT, PageKey, PlanetPosition, VisiblePage,
    };

    #[test]
    fn candidate_readiness_is_monotonic_across_asynchronous_snapshots() {
        let key = PlanetPageKey::new(PlanetId([0x4d; 16]), PageKey::new(3, [2, -1, 4]));
        let request = PlanetarySurfaceRequest {
            key,
            generation: SourceGeneration::new(7, 2),
            transition_mask: 0b00_1011,
            dirty_microbricks: u64::MAX,
        };
        let mut candidate = CandidateSurface {
            request,
            slot: 5,
            revision: 19,
            publication_generation: Some(43),
            regular_reservation: None,
            transition_reservation: None,
            ready: false,
        };
        let current = GpuSurfaceState {
            generation_low: 43,
            valid: 1,
            revision: 19,
            transition_mask: 0b00_1011,
            ..Default::default()
        };
        observe_candidate_surface_state(&mut candidate, &current);
        assert!(candidate.ready);

        let stale = GpuSurfaceState {
            generation_low: 42,
            valid: 1,
            revision: 18,
            transition_mask: 0,
            ..Default::default()
        };
        observe_candidate_surface_state(&mut candidate, &stale);
        assert!(candidate.ready);
    }

    #[test]
    fn initial_coarse_handoff_is_frozen_only_while_it_can_progress() {
        assert!(!should_retarget_handoff(true, false, true));
        assert!(should_retarget_handoff(true, true, true));
        assert!(should_retarget_handoff(true, false, false));
        assert!(!should_retarget_handoff(false, true, false));
    }

    #[test]
    fn validation_config_is_bounded_and_below_its_declared_budget() {
        let config = PlanetaryVoxelRenderConfig::validation_demo();
        let plan = config.allocation_plan().unwrap();
        assert!(plan.total_bytes <= config.max_surface_bytes);
        assert_eq!(config.residency.max_resident_pages, 32);
        assert_eq!(config.max_surface_pages, 5);
        assert_eq!(config.max_pending_surfaces, 5);
        assert_eq!(plan.indirect_bytes, 5 * DRAW_ARGS_BYTES);
        assert_eq!(plan.feedback_bytes, 32);
        assert_eq!(
            plan.diagnostic_readback_bytes,
            plan.feedback_bytes
                + core::mem::size_of::<GpuSurfaceGatherCounters>() as u64
                + plan.cull_counter_bytes
                + plan.state_bytes
                + plan.indirect_bytes * 2
        );
        assert_eq!(
            plan.regular_meshlet_indirect_bytes,
            u64::from(plan.regular_meshlet_draw_capacity) * DRAW_ARGS_BYTES
        );
        assert_eq!(
            plan.transition_meshlet_draw_bytes,
            u64::from(plan.transition_meshlet_draw_capacity) * TERRAIN_DRAW_BYTES
        );
        assert_eq!(core::mem::size_of::<DrawIndexedIndirectArgs>(), 20);
    }

    #[test]
    fn benchmark_config_is_bounded_and_below_its_declared_budget() {
        let config = PlanetaryVoxelRenderConfig::benchmark_demo();
        let plan = config.allocation_plan().unwrap();
        assert!(plan.total_bytes <= config.max_surface_bytes);
        assert_eq!(config.residency.max_resident_pages, 320);
        assert_eq!(config.max_surface_pages, 64);
        assert_eq!(config.max_pending_surfaces, 64);
        assert_eq!(plan.indirect_bytes, 64 * DRAW_ARGS_BYTES);
    }

    #[test]
    fn production_planet_config_holds_two_complete_bounded_frontiers() {
        let config = PlanetaryVoxelRenderConfig::production_planet();
        let plan = config.allocation_plan().unwrap();
        assert!(plan.total_bytes <= config.max_surface_bytes);
        assert_eq!(config.residency.max_resident_pages, 8_192);
        assert_eq!(config.max_surface_pages, 384);
        assert_eq!(config.residency.max_batch_pages, 192);
        assert_eq!(config.max_pending_surfaces, 384);
        assert_eq!(plan.indirect_bytes, 384 * DRAW_ARGS_BYTES);
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
    fn dependency_invalidations_wait_for_bounded_queue_capacity() {
        let planet = PlanetId([9; 16]);
        let generation = SourceGeneration::new(3, 7);
        let first =
            PlanetPageKey::new(planet, helio_planet_voxel_core::PageKey::new(0, [-1, 0, 0]));
        let second =
            PlanetPageKey::new(planet, helio_planet_voxel_core::PageKey::new(0, [0, 0, 0]));
        let request = |key| PlanetarySurfaceRequest {
            key,
            generation,
            transition_mask: 0,
            dirty_microbricks: u64::MAX,
        };
        let requests = BTreeMap::from([(first, request(first)), (second, request(second))]);
        let mut invalidated = BTreeSet::from([first, second]);
        let mut pending = VecDeque::new();

        drain_invalidated_surface_requests(&mut invalidated, &mut pending, &requests, 1, |_| {
            Some(generation)
        });
        assert_eq!(pending.len(), 1);
        assert_eq!(invalidated.len(), 1, "overflowed work must remain durable");

        pending.clear();
        drain_invalidated_surface_requests(&mut invalidated, &mut pending, &requests, 1, |_| {
            Some(generation)
        });
        assert_eq!(pending.len(), 1);
        assert!(invalidated.is_empty());
    }

    #[test]
    fn invalidation_does_not_duplicate_or_revive_stale_surface_requests() {
        let planet = PlanetId([4; 16]);
        let generation = SourceGeneration::new(1, 2);
        let key = PlanetPageKey::new(planet, helio_planet_voxel_core::PageKey::new(1, [-2, 3, 5]));
        let request = PlanetarySurfaceRequest {
            key,
            generation,
            transition_mask: 0,
            dirty_microbricks: u64::MAX,
        };
        let requests = BTreeMap::from([(key, request)]);
        let mut pending = VecDeque::from([request]);
        let mut invalidated = BTreeSet::from([key]);

        drain_invalidated_surface_requests(&mut invalidated, &mut pending, &requests, 8, |_| {
            Some(generation)
        });
        assert_eq!(pending.len(), 1);
        assert!(invalidated.is_empty());

        pending.clear();
        invalidated.insert(key);
        drain_invalidated_surface_requests(&mut invalidated, &mut pending, &requests, 8, |_| {
            Some(SourceGeneration::new(2, 0))
        });
        assert!(pending.is_empty());
        assert!(invalidated.is_empty());
    }

    #[test]
    fn visible_surface_requests_only_refresh_for_generation_or_topology_changes() {
        let key = PlanetPageKey::new(
            PlanetId([0x52; 16]),
            helio_planet_voxel_core::PageKey::new(4, [-7, 3, 11]),
        );
        let request = PlanetarySurfaceRequest {
            key,
            generation: SourceGeneration::new(5, 9),
            transition_mask: 0b00_0101,
            dirty_microbricks: u64::MAX,
        };

        assert!(!surface_request_is_current(None, request));
        assert!(surface_request_is_current(Some(&request), request));
        assert!(!surface_request_is_current(
            Some(&request),
            PlanetarySurfaceRequest {
                generation: SourceGeneration::new(5, 10),
                ..request
            }
        ));
        assert!(!surface_request_is_current(
            Some(&request),
            PlanetarySurfaceRequest {
                transition_mask: 0b00_0110,
                ..request
            }
        ));
    }

    #[test]
    fn accepted_visibility_registers_durable_surface_work_once() {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let mut adapter = None;
            for force_fallback_adapter in [false, true] {
                if let Ok(found) = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        force_fallback_adapter,
                        apply_limit_buckets: false,
                    })
                    .await
                {
                    adapter = Some(found);
                    break;
                }
            }
            let Some(adapter) = adapter else {
                eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER");
                return;
            };
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Planetary visible-surface scheduling test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .unwrap();
            let mut pass = PlanetaryVoxelRenderPass::new(
                &device,
                &queue,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                PlanetaryVoxelRenderConfig::validation_demo(),
            )
            .unwrap();
            let planet = PlanetId([0x31; 16]);
            let key = PlanetPageKey::new(planet, PageKey::new(1, [0, 0, 0]));
            let generation = SourceGeneration::new(1, 1);
            pass.set_planet_frame(
                &queue,
                PlanetFrameUniform::from_camera(planet, PlanetPosition::from_lod0_cell([0; 3]), 1),
            )
            .unwrap();
            pass.apply_upload_batch(
                &device,
                &queue,
                vec![
                    PageUpload::new(key, generation, vec![CellWord::AIR; PAGE_CELL_COUNT]).unwrap(),
                ],
            )
            .unwrap();

            let visible = |frame_index, transition_mask| VisiblePageSet {
                frame_index,
                pages: vec![VisiblePage {
                    key,
                    generation,
                    transition_mask,
                }],
            };
            pass.apply_visible_set(&queue, visible(1, 0)).unwrap();
            assert_eq!(pass.surface_requests.len(), 1);
            assert!(pass.invalidated_surfaces.is_empty());
            assert_eq!(
                pass.pending
                    .iter()
                    .map(|request| request.key)
                    .collect::<Vec<_>>(),
                vec![key]
            );
            let first_revision = pass.candidate_surfaces[&key].revision;

            let replacement_key = PlanetPageKey::new(planet, PageKey::new(1, [8, 0, 0]));
            let protected = pass
                .residency_protection_set(
                    2,
                    &[PlanetarySurfaceRequest {
                        key: replacement_key,
                        generation,
                        transition_mask: 0,
                        dirty_microbricks: u64::MAX,
                    }],
                )
                .unwrap();
            assert!(protected.pages.iter().any(|page| page.key == key));
            assert!(protected
                .pages
                .iter()
                .any(|page| page.key == replacement_key));

            pass.apply_visible_set(&queue, visible(2, 0)).unwrap();
            assert!(
                pass.invalidated_surfaces.is_empty(),
                "an unchanged frontier must not re-extract every frame"
            );
            assert_eq!(pass.pending.len(), 1);
            assert_eq!(pass.candidate_surfaces[&key].revision, first_revision);

            pass.apply_visible_set(&queue, visible(3, 1)).unwrap();
            assert!(pass.invalidated_surfaces.is_empty());
            assert_eq!(pass.pending.len(), 1);
            assert_eq!(
                pass.candidate_surfaces[&key].revision, first_revision,
                "the first complete coarse frontier must survive target churn"
            );
            assert_eq!(pass.pending.front().unwrap().transition_mask, 0);
            assert_eq!(pass.surface_requests[&key].transition_mask, 0);
            assert_eq!(pass.requested_visible[&key].transition_mask, 1);

            // Model completion of the first extraction. Committing it must
            // immediately schedule the remembered topology as a second atomic
            // handoff instead of losing that target or exposing a black frame.
            let candidate = pass.candidate_surfaces.get_mut(&key).unwrap();
            candidate.publication_generation = Some(1);
            candidate.ready = true;
            pass.pending.clear();
            pass.invalidated_surfaces.clear();
            pass.commit_ready_handoff(&queue).unwrap();

            assert_eq!(pass.active_surfaces[&key].request.transition_mask, 0);
            assert_ne!(pass.candidate_surfaces[&key].revision, first_revision);
            assert_eq!(pass.pending.front().unwrap().transition_mask, 1);
            assert_eq!(pass.surface_requests[&key].transition_mask, 1);
        });
    }

    #[test]
    fn replacement_frontier_keeps_parent_active_until_child_surface_is_ready() {
        pollster::block_on(async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let Ok(adapter) = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                })
                .await
            else {
                eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER");
                return;
            };
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Planetary atomic frontier handoff test device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .unwrap();
            let mut pass = PlanetaryVoxelRenderPass::new(
                &device,
                &queue,
                wgpu::TextureFormat::Rgba8UnormSrgb,
                PlanetaryVoxelRenderConfig::validation_demo(),
            )
            .unwrap();
            let planet = PlanetId([0x51; 16]);
            let parent = PlanetPageKey::new(planet, PageKey::new(2, [0, 0, 0]));
            let child = PlanetPageKey::new(planet, PageKey::new(1, [0, 0, 0]));
            let generation = SourceGeneration::new(7, 1);
            pass.set_planet_frame(
                &queue,
                PlanetFrameUniform::from_camera(planet, PlanetPosition::from_lod0_cell([0; 3]), 1),
            )
            .unwrap();
            pass.apply_upload_batch(
                &device,
                &queue,
                vec![
                    PageUpload::new(parent, generation, vec![CellWord::AIR; PAGE_CELL_COUNT])
                        .unwrap(),
                    PageUpload::new(child, generation, vec![CellWord::AIR; PAGE_CELL_COUNT])
                        .unwrap(),
                ],
            )
            .unwrap();
            let parent_publication = pass
                .residency
                .cache()
                .resident(parent)
                .unwrap()
                .publication_generation;
            let parent_request = PlanetarySurfaceRequest {
                key: parent,
                generation,
                transition_mask: 0,
                dirty_microbricks: u64::MAX,
            };
            pass.active_surfaces.insert(
                parent,
                PublishedSurface {
                    request: parent_request,
                    slot: 0,
                    publication_generation: parent_publication,
                    revision: 1,
                },
            );
            pass.surface_slots.insert(parent, 0);
            pass.visible.insert(parent, 0);
            pass.requested_visible.insert(parent, parent_request);

            pass.apply_visible_set(
                &queue,
                VisiblePageSet {
                    frame_index: 2,
                    pages: vec![VisiblePage {
                        key: child,
                        generation,
                        transition_mask: 0,
                    }],
                },
            )
            .unwrap();

            assert!(pass.active_surfaces.contains_key(&parent));
            assert!(!pass.active_surfaces.contains_key(&child));
            assert!(pass.visible.contains_key(&parent));
            assert_eq!(pass.handoff_planet, Some(planet));
            let child_publication = pass
                .residency
                .cache()
                .resident(child)
                .unwrap()
                .publication_generation;
            let candidate = pass.candidate_surfaces.get_mut(&child).unwrap();
            candidate.publication_generation = Some(child_publication);
            candidate.ready = true;
            pass.commit_ready_handoff(&queue).unwrap();

            assert!(!pass.active_surfaces.contains_key(&parent));
            assert!(pass.active_surfaces.contains_key(&child));
            assert!(!pass.visible.contains_key(&parent));
            assert!(pass.visible.contains_key(&child));
        });
    }
}
