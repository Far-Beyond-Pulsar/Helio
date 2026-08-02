//! Earth-radius camera-local validation for Helio's production planetary voxel path.
//!
//! Controls:
//!   Mouse click        - capture cursor / look
//!   W/A/S/D            - move through canonical planet space
//!   Space/Left Shift   - move up/down
//!   F2                 - toggle meshlet/page baseline draw path
//!   F3                 - cycle truthful terrain debug views
//!   F4                 - run a matched steady-state page/meshlet GPU benchmark
//!   F5                 - teleport between positive/negative planet coordinates
//!   F6                 - cycle ground, flight, and high-altitude validation
//!   Escape             - release cursor / exit

use glam::{EulerRot, Quat, Vec3};
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    DebugDrawState, RenderGraph, Renderer, RendererConfig, Scene,
};
use helio_pass_fxaa::FxaaPass;
use helio_pass_planetary_voxel::{
    ExtractionFixtureKind, HorizonLodFixturePlan, PlanetaryDebugView, PlanetaryDrawPath,
    PlanetaryRenderDiagnostics, PlanetarySurfaceRequest, PlanetaryVoxelRenderConfig,
    PlanetaryVoxelRenderPass, TerrainLodTopology,
};
use helio_planet_voxel_core::{
    EvictOutcome, PageEvict, PageKey, PageUpload, PlanetFrameUniform, PlanetId, PlanetPageKey,
    PlanetPosition, SourceGeneration, VisibilityOutcome, VisiblePage, VisiblePageSet,
    LOD0_CELL_SIZE_METERS, PAGE_CELL_BYTES, PAGE_EDGE, PAGE_EDGE_CELLS,
};
use std::{
    collections::{BTreeMap, HashSet, VecDeque},
    sync::Arc,
    time::Instant,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, DeviceEvents, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowId},
};

const EARTH_RADIUS_METERS: f64 = 6_371_000.0;
const LOOK_SENSITIVITY: f32 = 0.002;
const MOVE_SPEED_METERS_PER_SECOND: f64 = 1.5;
const INITIAL_YAW: f32 = -std::f32::consts::FRAC_PI_2;
const INITIAL_PITCH: f32 = -0.55;
const BENCHMARK_WARMUP_FRAMES: u32 = 60;
const BENCHMARK_SAMPLE_FRAMES: usize = 240;
const BENCHMARK_MAX_MISSING_TIMING_FRAMES: u32 = 600;
const BENCHMARK_GRID_EDGE: i64 = 8;
const BENCHMARK_DIAGNOSTIC_SETTLE_FRAMES: u32 = 16;
const HORIZON_ROOT_LOD: u8 = 11;
const HORIZON_MAX_PLAN_PAGES: usize = 192;
const HORIZON_HANDOFF_TIMEOUT_FRAMES: u64 = 1_200;
const HORIZON_ALTITUDES_METERS: [f64; 3] = [1.5, 100.0, 1_000.0];
const DEMO_SOURCE_GENERATION: SourceGeneration = SourceGeneration::new(1, 1);

#[derive(Clone, Copy, Debug)]
struct TimingSample {
    cpu_ms: f32,
    gpu_ms: f32,
}

#[derive(Clone, Copy, Debug)]
struct TimingSummary {
    cpu_p50_ms: f32,
    cpu_p95_ms: f32,
    gpu_p50_ms: f32,
    gpu_p95_ms: f32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum BenchmarkPhase {
    #[default]
    Idle,
    PageWarmup(u32),
    PageSamples,
    MeshletWarmup(u32),
    MeshletSamples,
    Complete,
}

#[derive(Default)]
struct PlanetBenchmark {
    phase: BenchmarkPhase,
    page_samples: Vec<TimingSample>,
    meshlet_samples: Vec<TimingSample>,
    page_summary: Option<TimingSummary>,
    meshlet_summary: Option<TimingSummary>,
    missing_timing_frames: u32,
}

impl PlanetBenchmark {
    fn start(&mut self) {
        self.phase = BenchmarkPhase::PageWarmup(0);
        self.page_samples.clear();
        self.meshlet_samples.clear();
        self.page_summary = None;
        self.meshlet_summary = None;
        self.missing_timing_frames = 0;
    }

    fn cancel(&mut self) {
        self.phase = BenchmarkPhase::Idle;
        self.page_samples.clear();
        self.meshlet_samples.clear();
        self.missing_timing_frames = 0;
    }

    const fn active(&self) -> bool {
        matches!(
            self.phase,
            BenchmarkPhase::PageWarmup(_)
                | BenchmarkPhase::PageSamples
                | BenchmarkPhase::MeshletWarmup(_)
                | BenchmarkPhase::MeshletSamples
        )
    }

    fn record(
        &mut self,
        draw_path: PlanetaryDrawPath,
        sample: Option<TimingSample>,
    ) -> Option<PlanetaryDrawPath> {
        match self.phase {
            BenchmarkPhase::Idle | BenchmarkPhase::Complete => {}
            BenchmarkPhase::PageWarmup(frames) => {
                debug_assert_eq!(draw_path, PlanetaryDrawPath::PageIndexed);
                let frames = frames + 1;
                self.phase = if frames >= BENCHMARK_WARMUP_FRAMES {
                    BenchmarkPhase::PageSamples
                } else {
                    BenchmarkPhase::PageWarmup(frames)
                };
            }
            BenchmarkPhase::PageSamples => {
                debug_assert_eq!(draw_path, PlanetaryDrawPath::PageIndexed);
                if let Some(sample) = sample {
                    self.page_samples.push(sample);
                    self.missing_timing_frames = 0;
                } else {
                    self.missing_timing_frames = self.missing_timing_frames.saturating_add(1);
                    if self.missing_timing_frames >= BENCHMARK_MAX_MISSING_TIMING_FRAMES {
                        self.phase = BenchmarkPhase::Complete;
                        return None;
                    }
                }
                if self.page_samples.len() == BENCHMARK_SAMPLE_FRAMES {
                    self.page_summary = summarize_timings(&self.page_samples);
                    self.phase = BenchmarkPhase::MeshletWarmup(0);
                    return Some(PlanetaryDrawPath::Meshlets);
                }
            }
            BenchmarkPhase::MeshletWarmup(frames) => {
                debug_assert_eq!(draw_path, PlanetaryDrawPath::Meshlets);
                let frames = frames + 1;
                self.phase = if frames >= BENCHMARK_WARMUP_FRAMES {
                    BenchmarkPhase::MeshletSamples
                } else {
                    BenchmarkPhase::MeshletWarmup(frames)
                };
            }
            BenchmarkPhase::MeshletSamples => {
                debug_assert_eq!(draw_path, PlanetaryDrawPath::Meshlets);
                if let Some(sample) = sample {
                    self.meshlet_samples.push(sample);
                    self.missing_timing_frames = 0;
                } else {
                    self.missing_timing_frames = self.missing_timing_frames.saturating_add(1);
                    if self.missing_timing_frames >= BENCHMARK_MAX_MISSING_TIMING_FRAMES {
                        self.phase = BenchmarkPhase::Complete;
                        return None;
                    }
                }
                if self.meshlet_samples.len() == BENCHMARK_SAMPLE_FRAMES {
                    self.meshlet_summary = summarize_timings(&self.meshlet_samples);
                    self.phase = BenchmarkPhase::Complete;
                    if let (Some(page), Some(meshlet)) = (self.page_summary, self.meshlet_summary) {
                        eprintln!(
                            "PLANET_MESHLET_BENCHMARK page_cpu_p50_ms={:.6} page_cpu_p95_ms={:.6} page_gpu_p50_ms={:.6} page_gpu_p95_ms={:.6} meshlet_cpu_p50_ms={:.6} meshlet_cpu_p95_ms={:.6} meshlet_gpu_p50_ms={:.6} meshlet_gpu_p95_ms={:.6}",
                            page.cpu_p50_ms,
                            page.cpu_p95_ms,
                            page.gpu_p50_ms,
                            page.gpu_p95_ms,
                            meshlet.cpu_p50_ms,
                            meshlet.cpu_p95_ms,
                            meshlet.gpu_p50_ms,
                            meshlet.gpu_p95_ms,
                        );
                    }
                }
            }
        }
        None
    }

    fn label(&self) -> String {
        match self.phase {
            BenchmarkPhase::Idle => "idle".into(),
            BenchmarkPhase::PageWarmup(frame) => {
                format!("page-warmup:{frame}/{BENCHMARK_WARMUP_FRAMES}")
            }
            BenchmarkPhase::PageSamples => {
                format!(
                    "page-samples:{}/{}",
                    self.page_samples.len(),
                    BENCHMARK_SAMPLE_FRAMES
                )
            }
            BenchmarkPhase::MeshletWarmup(frame) => {
                format!("meshlet-warmup:{frame}/{BENCHMARK_WARMUP_FRAMES}")
            }
            BenchmarkPhase::MeshletSamples => {
                format!(
                    "meshlet-samples:{}/{}",
                    self.meshlet_samples.len(),
                    BENCHMARK_SAMPLE_FRAMES
                )
            }
            BenchmarkPhase::Complete => match (self.page_summary, self.meshlet_summary) {
                (Some(page), Some(meshlet)) => format!(
                    "done gpu-p95 page{:.3} meshlet{:.3}ms",
                    page.gpu_p95_ms, meshlet.gpu_p95_ms
                ),
                _ => "complete-no-gpu-timings".into(),
            },
        }
    }
}

#[derive(Default)]
struct HorizonTrace {
    next_handoff_step: usize,
    awaiting_handoff: Option<u64>,
    cancellation_injected: bool,
    resize_stage: u8,
    resize_frames: u32,
    frames: u64,
    samples: Vec<TimingSample>,
    maximum_resident_pages: u32,
    maximum_queued_surfaces: usize,
    maximum_backpressure: u64,
}

fn summarize_timings(samples: &[TimingSample]) -> Option<TimingSummary> {
    if samples.is_empty() {
        return None;
    }
    let mut cpu = samples
        .iter()
        .map(|sample| sample.cpu_ms)
        .collect::<Vec<_>>();
    let mut gpu = samples
        .iter()
        .map(|sample| sample.gpu_ms)
        .collect::<Vec<_>>();
    cpu.sort_by(f32::total_cmp);
    gpu.sort_by(f32::total_cmp);
    Some(TimingSummary {
        cpu_p50_ms: percentile(&cpu, 0.50),
        cpu_p95_ms: percentile(&cpu, 0.95),
        gpu_p50_ms: percentile(&gpu, 0.50),
        gpu_p95_ms: percentile(&gpu, 0.95),
    })
}

fn percentile(sorted: &[f32], percentile: f32) -> f32 {
    let index = ((sorted.len() - 1) as f32 * percentile).round() as usize;
    sorted[index]
}

fn main() {
    env_logger::init();
    let auto_benchmark = std::env::args().any(|argument| argument == "--benchmark");
    let auto_horizon_trace = std::env::args().any(|argument| argument == "--horizon-trace");
    assert!(
        !(auto_benchmark && auto_horizon_trace),
        "--benchmark and --horizon-trace are separate matched workloads"
    );
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        state: None,
        auto_benchmark,
        auto_horizon_trace,
    };
    event_loop
        .run_app(&mut app)
        .expect("planet demo event loop");
}

struct App {
    state: Option<AppState>,
    auto_benchmark: bool,
    auto_horizon_trace: bool,
}

#[derive(Clone)]
struct HorizonResidentPlan {
    focus_page: PageKey,
    minimum_lod: u8,
    topology: TerrainLodTopology,
    generations: BTreeMap<PageKey, SourceGeneration>,
    /// Transition faces already present in each page's published surface.
    ///
    /// This can be a superset of the currently visible transition mask. The
    /// draw shader face-tags transition vertices and discards faces outside the
    /// visible set, allowing the old and replacement masks to share one
    /// generation safely during an atomic handoff.
    surface_masks: BTreeMap<PageKey, u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HorizonStreamWork {
    page: PageKey,
    generation: SourceGeneration,
    transition_mask: u8,
    upload_page: bool,
}

struct PendingHorizonHandoff {
    plan: HorizonResidentPlan,
    remaining_work: VecDeque<HorizonStreamWork>,
    cancelled: bool,
    target_published_jobs: u64,
    baseline_rejections: [u32; 3],
    started_frame: u64,
    upload_count: u64,
    upload_bytes: u64,
    surface_jobs: u64,
    transition_jobs: u64,
}

struct HorizonStreamingState {
    active: Option<HorizonResidentPlan>,
    pending: Option<PendingHorizonHandoff>,
    desired_focus_lod0_cell: [i64; 3],
    desired_minimum_lod: u8,
    next_visible_frame: u64,
    cancel_requested: bool,
    handoffs: u64,
    cancellations: u64,
    uploads: u64,
    upload_bytes: u64,
    surface_jobs: u64,
    transition_jobs: u64,
    evictions: u64,
    deferred_changes: u64,
    failure: Option<String>,
}

impl HorizonStreamingState {
    fn new(camera_m: [f64; 3], focus_m: [f64; 3]) -> Self {
        Self {
            active: None,
            pending: None,
            desired_focus_lod0_cell: horizon_focus_lod0_cell(focus_m),
            desired_minimum_lod: horizon_minimum_lod(camera_m[1]),
            next_visible_frame: 1,
            cancel_requested: false,
            handoffs: 0,
            cancellations: 0,
            uploads: 0,
            upload_bytes: 0,
            surface_jobs: 0,
            transition_jobs: 0,
            evictions: 0,
            deferred_changes: 0,
            failure: None,
        }
    }

    fn set_desired_camera(&mut self, camera_m: [f64; 3], focus_m: [f64; 3]) {
        let focus = horizon_focus_lod0_cell(focus_m);
        let minimum_lod = horizon_minimum_lod(camera_m[1]);
        let focus_page = PageKey::address_lod0_cell(0, focus)
            .ok()
            .map(|(page, _)| page);
        let pending_changed = self.pending.as_ref().is_some_and(|pending| {
            !pending.cancelled
                && (focus_page != Some(pending.plan.focus_page)
                    || minimum_lod != pending.plan.minimum_lod)
        });
        if pending_changed && !self.cancel_requested {
            self.cancel_requested = true;
            self.deferred_changes = self.deferred_changes.saturating_add(1);
        }
        self.desired_focus_lod0_cell = focus;
        self.desired_minimum_lod = minimum_lod;
    }

    fn update(
        &mut self,
        pass: &mut PlanetaryVoxelRenderPass,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planet: PlanetId,
        frame_index: u64,
        diagnostics: &PlanetaryRenderDiagnostics,
    ) {
        if self.failure.is_some() {
            return;
        }
        if self.cancel_requested {
            self.begin_pending_cancellation(pass);
        }
        if let Err(error) =
            self.finish_ready_handoff(pass, device, queue, planet, frame_index, diagnostics)
        {
            self.fail(error);
            return;
        }
        if self.pending.is_none() {
            if let Err(error) = self.schedule_if_changed(pass, frame_index, diagnostics) {
                self.fail(error);
                return;
            }
        }
        if let Err(error) = self.advance_pending_work(pass, device, queue, planet) {
            self.fail(error);
            return;
        }
        if let Err(error) =
            self.finish_ready_handoff(pass, device, queue, planet, frame_index, diagnostics)
        {
            self.fail(error);
        }
    }

    fn schedule_if_changed(
        &mut self,
        pass: &PlanetaryVoxelRenderPass,
        frame_index: u64,
        diagnostics: &PlanetaryRenderDiagnostics,
    ) -> Result<(), String> {
        let fixture = HorizonLodFixturePlan::build_with_minimum_lod(
            self.desired_focus_lod0_cell,
            HORIZON_ROOT_LOD,
            self.desired_minimum_lod,
            HORIZON_MAX_PLAN_PAGES,
        )
        .map_err(|error| format!("horizon topology planning failed: {error}"))?;
        let topology = fixture.topology().clone();
        let focus_page = PageKey::address_lod0_cell(0, self.desired_focus_lod0_cell)
            .map_err(|error| format!("horizon focus address failed: {error}"))?
            .0;
        if self.active.as_ref().is_some_and(|active| {
            active.focus_page == focus_page
                && active.minimum_lod == self.desired_minimum_lod
                && active.topology == topology
        }) {
            return Ok(());
        }

        let active_generations = self
            .active
            .as_ref()
            .map(|active| active.generations.clone())
            .unwrap_or_default();
        let active_surface_masks = self
            .active
            .as_ref()
            .map(|active| active.surface_masks.clone())
            .unwrap_or_default();
        let mut generations = BTreeMap::new();
        for page in topology.pages() {
            let generation = active_generations
                .get(&page)
                .copied()
                .unwrap_or_else(|| DEMO_SOURCE_GENERATION);
            generations.insert(page, generation);
        }

        let mut surface_masks = BTreeMap::new();
        let mut remaining_work = VecDeque::new();
        for page in topology.pages() {
            let generation = generations[&page];
            let transition_mask = topology
                .transition_mask(page)
                .expect("planned page has a transition mask");
            let is_new = !active_generations.contains_key(&page);
            let available_mask = active_surface_masks.get(&page).copied().unwrap_or(0);
            let extraction_mask = available_mask | transition_mask;
            surface_masks.insert(page, extraction_mask);
            if is_new || extraction_mask != available_mask {
                remaining_work.push_back(HorizonStreamWork {
                    page,
                    generation,
                    transition_mask: extraction_mask,
                    upload_page: is_new,
                });
            }
        }

        let surface_jobs = remaining_work.len() as u64;
        let target_published_jobs = pass
            .counters()
            .submitted_jobs
            .saturating_add(pass.counters().queued_surfaces as u64)
            .saturating_add(surface_jobs);
        let transition_jobs = remaining_work
            .iter()
            .filter(|work| work.transition_mask != 0)
            .count() as u64;
        self.pending = Some(PendingHorizonHandoff {
            plan: HorizonResidentPlan {
                focus_page,
                minimum_lod: self.desired_minimum_lod,
                topology,
                generations,
                surface_masks,
            },
            remaining_work,
            cancelled: false,
            target_published_jobs,
            baseline_rejections: rejection_counts(diagnostics),
            started_frame: frame_index,
            upload_count: 0,
            upload_bytes: 0,
            surface_jobs: 0,
            transition_jobs: 0,
        });
        debug_assert!(surface_jobs >= transition_jobs);
        Ok(())
    }

    fn begin_pending_cancellation(&mut self, pass: &PlanetaryVoxelRenderPass) {
        self.cancel_requested = false;
        let Some(pending) = self.pending.as_mut() else {
            return;
        };
        pending.cancelled = true;
        pending.remaining_work.clear();
        pending.target_published_jobs = pass
            .counters()
            .submitted_jobs
            .saturating_add(pass.counters().queued_surfaces as u64);
    }

    fn advance_pending_work(
        &mut self,
        pass: &mut PlanetaryVoxelRenderPass,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planet: PlanetId,
    ) -> Result<(), String> {
        let pending_upload = self
            .pending
            .as_ref()
            .filter(|pending| !pending.cancelled)
            .and_then(|pending| {
                pending
                    .remaining_work
                    .iter()
                    .position(|work| work.upload_page)
                    .map(|index| (index, pending.remaining_work[index]))
            });
        if let Some((index, work)) = pending_upload {
            let upload = build_page_upload(planet, work.page, work.generation);
            let outcomes = pass
                .apply_upload_batch(device, queue, vec![upload])
                .map_err(|error| format!("horizon target-page upload failed: {error}"))?;
            if !matches!(
                outcomes.as_slice(),
                [helio_pass_planetary_voxel::GpuUploadOutcome::Residency(
                    helio_planet_voxel_core::UploadOutcome::Inserted { .. }
                        | helio_planet_voxel_core::UploadOutcome::Duplicate { .. }
                )]
            ) {
                return Err(format!(
                    "horizon target-page residency rejected {:?}: {outcomes:?}",
                    work.page
                ));
            }
            let pending = self
                .pending
                .as_mut()
                .expect("target upload keeps its pending handoff");
            pending.remaining_work[index].upload_page = false;
            pending.upload_count = pending.upload_count.saturating_add(1);
            pending.upload_bytes = pending.upload_bytes.saturating_add(PAGE_CELL_BYTES as u64);
            return Ok(());
        }

        let Some(work) = self
            .pending
            .as_ref()
            .filter(|pending| !pending.cancelled)
            .and_then(|pending| pending.remaining_work.front().copied())
        else {
            return Ok(());
        };

        debug_assert!(!work.upload_page);
        let surface =
            build_surface_upload(planet, work.page, work.transition_mask, work.generation);
        self.prune_sampling_support_for(pass, device, queue, surface)?;
        let (support_uploads, support_bytes) =
            self.ensure_surface_dependencies(pass, device, queue, surface)?;
        let key = PlanetPageKey::new(planet, work.page);
        let resident = pass
            .residency()
            .cache()
            .resident(key)
            .ok_or_else(|| format!("horizon staging left {key:?} non-resident"))?;
        if resident.generation != work.generation {
            return Err(format!(
                "horizon staging generation mismatch for {key:?}: expected {:?}, got {:?}",
                work.generation, resident.generation
            ));
        }
        let surface_bytes = core::mem::size_of::<PlanetarySurfaceRequest>() as u64;
        pass.queue_surface(surface)
            .map_err(|error| format!("horizon incremental surface queue failed: {error}"))?;

        let pending = self
            .pending
            .as_mut()
            .expect("staged horizon work keeps its pending handoff");
        assert_eq!(pending.remaining_work.pop_front(), Some(work));
        pending.upload_count = pending.upload_count.saturating_add(support_uploads);
        pending.upload_bytes = pending
            .upload_bytes
            .saturating_add(surface_bytes)
            .saturating_add(support_bytes);
        pending.surface_jobs = pending.surface_jobs.saturating_add(1);
        pending.transition_jobs = pending
            .transition_jobs
            .saturating_add(u64::from(work.transition_mask != 0));
        Ok(())
    }

    fn prune_sampling_support_for(
        &mut self,
        pass: &mut PlanetaryVoxelRenderPass,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: PlanetarySurfaceRequest,
    ) -> Result<(), String> {
        let mut retained = surface
            .required_pages()
            .map_err(|error| format!("horizon support retention planning failed: {error}"))?;
        if let Some(active) = &self.active {
            retained.extend(
                active
                    .generations
                    .keys()
                    .map(|page| PlanetPageKey::new(surface.key.planet, *page)),
            );
        }
        if let Some(pending) = &self.pending {
            retained.extend(
                pending
                    .plan
                    .generations
                    .keys()
                    .map(|page| PlanetPageKey::new(surface.key.planet, *page)),
            );
        }
        let evictions = pass
            .residency()
            .cache()
            .resident_pages()
            .filter(|(key, _)| key.planet == surface.key.planet && !retained.contains(key))
            .map(|(key, resident)| PageEvict {
                key,
                generation: resident.generation,
            })
            .collect::<Vec<_>>();
        let eviction_count = apply_horizon_evictions(pass, device, queue, evictions)?;
        self.evictions = self.evictions.saturating_add(eviction_count);
        Ok(())
    }

    fn ensure_surface_dependencies(
        &mut self,
        pass: &mut PlanetaryVoxelRenderPass,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface: PlanetarySurfaceRequest,
    ) -> Result<(u64, u64), String> {
        let missing = surface
            .required_pages()
            .map_err(|error| format!("horizon surface dependency planning failed: {error}"))?
            .into_iter()
            .filter(|key| pass.residency().cache().resident(*key).is_none())
            .collect::<Vec<_>>();
        let upload_count = missing.len() as u64;
        if missing.is_empty() {
            return Ok((0, 0));
        }

        let mut uploads = Vec::with_capacity(missing.len());
        for key in missing {
            uploads.push(build_page_upload(
                key.planet,
                key.page,
                DEMO_SOURCE_GENERATION,
            ));
        }
        let batch_pages = pass.residency().config().max_batch_pages as usize;
        for chunk in uploads.chunks(batch_pages) {
            let outcomes = pass
                .apply_upload_batch(device, queue, chunk.to_vec())
                .map_err(|error| format!("horizon support-page upload failed: {error}"))?;
            if outcomes.iter().any(|outcome| {
                !matches!(
                    outcome,
                    helio_pass_planetary_voxel::GpuUploadOutcome::Residency(
                        helio_planet_voxel_core::UploadOutcome::Inserted { .. }
                            | helio_planet_voxel_core::UploadOutcome::Duplicate { .. }
                    )
                )
            }) {
                return Err(format!(
                    "horizon support-page residency rejected an exact dependency batch: {outcomes:?}"
                ));
            }
        }
        Ok((
            upload_count,
            upload_count.saturating_mul(PAGE_CELL_BYTES as u64),
        ))
    }

    fn finish_ready_handoff(
        &mut self,
        pass: &mut PlanetaryVoxelRenderPass,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        planet: PlanetId,
        frame_index: u64,
        diagnostics: &PlanetaryRenderDiagnostics,
    ) -> Result<(), String> {
        let Some(pending) = self.pending.as_ref() else {
            return Ok(());
        };
        let rejections = rejection_counts(diagnostics);
        if rejections
            .into_iter()
            .zip(pending.baseline_rejections)
            .any(|(current, baseline)| current > baseline)
        {
            return Err(format!(
                "horizon GPU rejected work: stale={} overflow={} incomplete={} gather[r={} t={} probes={} misses={} stale={} done={}]",
                diagnostics.gpu_stale_rejections,
                diagnostics.gpu_overflow_rejections,
                diagnostics.gpu_incomplete_rejections,
                diagnostics.gather_regular_samples,
                diagnostics.gather_transition_samples,
                diagnostics.gather_table_probes,
                diagnostics.gather_page_misses,
                diagnostics.gather_stale_targets,
                diagnostics.gather_completed,
            ));
        }
        if frame_index.saturating_sub(pending.started_frame) > HORIZON_HANDOFF_TIMEOUT_FRAMES {
            return Err(format!(
                "horizon handoff timed out after {} frames with {}/{} jobs published and {} queued",
                frame_index.saturating_sub(pending.started_frame),
                diagnostics.gpu_published_jobs,
                pending.target_published_jobs,
                pass.counters().queued_surfaces
            ));
        }
        let ready = pending.remaining_work.is_empty()
            && pass.counters().queued_surfaces == 0
            && u64::from(diagnostics.gpu_published_jobs) >= pending.target_published_jobs;
        if !ready {
            return Ok(());
        }

        let pending = self
            .pending
            .take()
            .expect("ready horizon handoff remains pending");
        if pending.cancelled {
            let active_generations = self.active.as_ref().map(|active| &active.generations);
            let evictions = pass
                .residency()
                .cache()
                .resident_pages()
                .filter(|(key, _)| {
                    key.planet == planet
                        && !active_generations.is_some_and(|active| active.contains_key(&key.page))
                })
                .map(|(key, resident)| PageEvict {
                    key,
                    generation: resident.generation,
                })
                .collect::<Vec<_>>();
            let eviction_count = apply_horizon_evictions(pass, device, queue, evictions)?;
            self.cancellations = self.cancellations.saturating_add(1);
            self.uploads = self.uploads.saturating_add(pending.upload_count);
            self.upload_bytes = self.upload_bytes.saturating_add(pending.upload_bytes);
            self.surface_jobs = self.surface_jobs.saturating_add(pending.surface_jobs);
            self.transition_jobs = self.transition_jobs.saturating_add(pending.transition_jobs);
            self.evictions = self.evictions.saturating_add(eviction_count);
            return Ok(());
        }

        for (page, generation) in &pending.plan.generations {
            let key = PlanetPageKey::new(planet, *page);
            let resident = pass
                .residency()
                .cache()
                .resident(key)
                .ok_or_else(|| format!("horizon handoff left {key:?} non-resident"))?;
            if resident.generation != *generation {
                return Err(format!(
                    "horizon handoff generation mismatch for {key:?}: expected {generation:?}, got {:?}",
                    resident.generation
                ));
            }
            let visible_mask = pending
                .plan
                .topology
                .transition_mask(*page)
                .expect("planned page has a transition mask");
            let surface_mask = pending.plan.surface_masks[page];
            if visible_mask & !surface_mask != 0 {
                return Err(format!(
                    "horizon handoff surface for {key:?} lacks visible transition faces {visible_mask:#08b} outside {surface_mask:#08b}"
                ));
            }
        }

        self.next_visible_frame = self.next_visible_frame.saturating_add(1);
        let visible = VisiblePageSet {
            frame_index: self.next_visible_frame,
            pages: pending
                .plan
                .topology
                .pages()
                .map(|page| VisiblePage {
                    key: PlanetPageKey::new(planet, page),
                    generation: pending.plan.generations[&page],
                    transition_mask: pending
                        .plan
                        .topology
                        .transition_mask(page)
                        .expect("planned page has a transition mask"),
                })
                .collect(),
        };
        let outcome = pass
            .apply_visible_set(queue, visible)
            .map_err(|error| format!("horizon visible-set handoff failed: {error}"))?;
        match outcome {
            VisibilityOutcome::Applied {
                resident,
                missing: 0,
                generation_mismatches: 0,
            } if resident == pending.plan.topology.stats().pages => {}
            other => {
                return Err(format!(
                    "horizon visible-set handoff was not complete: {other:?}"
                ));
            }
        }

        let evictions = pass
            .residency()
            .cache()
            .resident_pages()
            .filter(|(key, _)| {
                key.planet == planet && !pending.plan.generations.contains_key(&key.page)
            })
            .map(|(key, resident)| PageEvict {
                key,
                generation: resident.generation,
            })
            .collect::<Vec<_>>();
        let eviction_count = apply_horizon_evictions(pass, device, queue, evictions)?;
        self.handoffs = self.handoffs.saturating_add(1);
        self.uploads = self.uploads.saturating_add(pending.upload_count);
        self.upload_bytes = self.upload_bytes.saturating_add(pending.upload_bytes);
        self.surface_jobs = self.surface_jobs.saturating_add(pending.surface_jobs);
        self.transition_jobs = self.transition_jobs.saturating_add(pending.transition_jobs);
        self.evictions = self.evictions.saturating_add(eviction_count);
        self.active = Some(pending.plan);
        Ok(())
    }

    fn fail(&mut self, error: String) {
        log::error!("{error}");
        self.failure = Some(error);
    }

    fn label(&self) -> String {
        if let Some(failure) = &self.failure {
            return format!("FAIL:{failure}");
        }
        let active = self.active.as_ref().map(|plan| plan.topology.stats());
        let pending = self.pending.as_ref().map(|handoff| {
            (
                handoff.plan.topology.stats(),
                handoff.remaining_work.len(),
                handoff.cancelled,
            )
        });
        match (active, pending) {
            (Some(active), Some((pending, remaining, cancelled))) => format!(
                "handoff a{}:{}-{} p{}:{}-{} left{} cancel{} h{} c{} u{} e{} d{}",
                active.pages,
                active.minimum_lod,
                active.maximum_lod,
                pending.pages,
                pending.minimum_lod,
                pending.maximum_lod,
                remaining,
                u8::from(cancelled),
                self.handoffs,
                self.cancellations,
                self.uploads,
                self.evictions,
                self.deferred_changes,
            ),
            (Some(active), None) => format!(
                "active {}:{}-{} t{} h{} c{} u{}({}MiB) j{}/{} e{} d{}",
                active.pages,
                active.minimum_lod,
                active.maximum_lod,
                active.transition_faces,
                self.handoffs,
                self.cancellations,
                self.uploads,
                self.upload_bytes / (1024 * 1024),
                self.surface_jobs,
                self.transition_jobs,
                self.evictions,
                self.deferred_changes,
            ),
            (None, Some((pending, remaining, cancelled))) => format!(
                "loading {}:{}-{} t{} left{} cancel{}",
                pending.pages,
                pending.minimum_lod,
                pending.maximum_lod,
                pending.transition_faces,
                remaining,
                u8::from(cancelled),
            ),
            (None, None) => "idle".to_string(),
        }
    }
}

fn apply_horizon_evictions(
    pass: &mut PlanetaryVoxelRenderPass,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    evictions: Vec<PageEvict>,
) -> Result<u64, String> {
    let eviction_count = evictions.len() as u64;
    if evictions.is_empty() {
        return Ok(0);
    }
    let batch_pages = pass.residency().config().max_batch_pages as usize;
    for chunk in evictions.chunks(batch_pages) {
        let outcomes = pass
            .apply_evict_batch(device, queue, chunk.to_vec())
            .map_err(|error| format!("horizon eviction batch failed: {error}"))?;
        for (eviction, outcome) in chunk.iter().copied().zip(outcomes) {
            if !matches!(outcome, EvictOutcome::Recorded { .. }) {
                return Err(format!(
                    "horizon eviction was not recorded for {:?}: {outcome:?}",
                    eviction.key
                ));
            }
            if !pass
                .residency_mut()
                .retire_eviction_watermark(eviction.key, eviction.generation)
            {
                return Err(format!(
                    "horizon eviction watermark was not retired for {:?}",
                    eviction.key
                ));
            }
        }
    }
    Ok(eviction_count)
}

struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
    alpha_mode: wgpu::CompositeAlphaMode,
    renderer: Renderer,
    planet: PlanetId,
    canonical_camera_m: [f64; 3],
    spawn_camera_m: [f64; 3],
    frame_index: u64,
    yaw: f32,
    pitch: f32,
    keys: HashSet<KeyCode>,
    mouse_delta: (f32, f32),
    cursor_grabbed: bool,
    last_frame: Instant,
    last_title_update: Instant,
    benchmark: PlanetBenchmark,
    auto_benchmark_expected_jobs: Option<u64>,
    auto_fill_samples: Vec<TimingSample>,
    auto_completion_frames: u32,
    horizon_streaming: Option<HorizonStreamingState>,
    horizon_altitude_index: usize,
    planet_diagnostics: PlanetaryRenderDiagnostics,
    horizon_trace: Option<HorizonTrace>,
}

impl AppState {
    fn reset_transient_input(&mut self) {
        self.keys.clear();
        self.mouse_delta = (0.0, 0.0);
        self.cursor_grabbed = false;
        let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        self.window.set_cursor_visible(true);
        self.last_frame = Instant::now();
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.reset_transient_input();
        if width == 0 || height == 0 {
            return;
        }
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: self.surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: self.alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
        self.renderer.set_render_size(width, height);
        self.renderer
            .find_pass_mut::<PlanetaryVoxelRenderPass>()
            .expect("planetary pass")
            .residency_mut()
            .resize(width, height);
    }

    fn orientation(&self) -> Quat {
        Quat::from_euler(EulerRot::YXZ, self.yaw, self.pitch, 0.0)
    }

    fn update(&mut self, dt: f64) {
        if self.benchmark.active() || self.horizon_trace.is_some() {
            self.mouse_delta = (0.0, 0.0);
            return;
        }
        self.yaw -= self.mouse_delta.0 * LOOK_SENSITIVITY;
        self.pitch = (self.pitch - self.mouse_delta.1 * LOOK_SENSITIVITY).clamp(-1.5, 1.5);
        self.mouse_delta = (0.0, 0.0);
        let orientation = self.orientation();
        advance_camera(&mut self.canonical_camera_m, &self.keys, orientation, dt);
    }

    fn camera(&self, width: u32, height: u32) -> Camera {
        let orientation = self.orientation();
        Camera::perspective_look_at(
            Vec3::ZERO,
            orientation * -Vec3::Z,
            orientation * Vec3::Y,
            std::f32::consts::FRAC_PI_3,
            width as f32 / height.max(1) as f32,
            0.01,
            10_000.0,
        )
    }

    fn update_planet_frame(&mut self) {
        self.frame_index = self.frame_index.wrapping_add(1);
        let camera = PlanetPosition::from_meters(self.canonical_camera_m)
            .expect("bounded camera input remains canonical");
        let frame = PlanetFrameUniform::from_camera(self.planet, camera, self.frame_index);
        self.renderer
            .find_pass_mut::<PlanetaryVoxelRenderPass>()
            .expect("planetary pass")
            .set_planet_frame(&self.queue, frame)
            .expect("camera-local planet frame");
    }

    fn update_horizon_streaming(&mut self) {
        let focus_m = horizon_streaming_focus(self.canonical_camera_m, self.orientation());
        let Some(streaming) = self.horizon_streaming.as_mut() else {
            return;
        };
        streaming.set_desired_camera(self.canonical_camera_m, focus_m);
        let pass = self
            .renderer
            .find_pass_mut::<PlanetaryVoxelRenderPass>()
            .expect("planetary pass");
        self.planet_diagnostics = pass.poll_diagnostics(&self.device, &self.queue);
        streaming.update(
            pass,
            &self.device,
            &self.queue,
            self.planet,
            self.frame_index,
            &self.planet_diagnostics,
        );
    }

    fn update_title(&mut self) {
        if self.last_title_update.elapsed().as_millis() < 250 {
            return;
        }
        self.last_title_update = Instant::now();
        let device = self.device.clone();
        let queue = self.queue.clone();
        let uses_horizon_streaming = self.horizon_streaming.is_some();
        let (render, residency, diagnostics, draw_path, debug_view) = {
            let pass = self
                .renderer
                .find_pass_mut::<PlanetaryVoxelRenderPass>()
                .expect("planetary pass");
            let diagnostics = if uses_horizon_streaming {
                self.planet_diagnostics.clone()
            } else {
                pass.poll_diagnostics(&device, &queue)
            };
            (
                pass.counters(),
                pass.residency().counters(),
                diagnostics,
                pass.draw_path(),
                pass.debug_view(),
            )
        };
        let lods = diagnostics
            .resident_lods
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let source_generations = match (
            diagnostics.source_generation_min,
            diagnostics.source_generation_max,
        ) {
            (Some(minimum), Some(maximum)) => format!(
                "{}:{}-{}:{}",
                minimum.planet, minimum.page, maximum.planet, maximum.page
            ),
            _ => "-".into(),
        };
        let publication_generations = match (
            diagnostics.publication_generation_min,
            diagnostics.publication_generation_max,
        ) {
            (Some(minimum), Some(maximum)) => format!("{minimum}-{maximum}"),
            _ => "-".into(),
        };
        let camera_delta = [
            self.canonical_camera_m[0] - self.spawn_camera_m[0],
            self.canonical_camera_m[1] - self.spawn_camera_m[1],
            self.canonical_camera_m[2] - self.spawn_camera_m[2],
        ];
        let streaming = self
            .horizon_streaming
            .as_ref()
            .map(HorizonStreamingState::label)
            .unwrap_or_else(|| "static".to_string());
        self.window.set_title(&format!(
            "Helio Planet Voxels | {:?}/{} | bench {} | stream {streaming} | cam[{:+.2},{:+.2},{:+.2}]m look[{:+.2},{:+.2}] focus{} grab{} keys{} | R={EARTH_RADIUS_METERS:.0}m 10cm | pages {} lod[{lods}] | src {source_generations} pub {publication_generations} | gpu jobs {}/{} reject s{} o{} i{} | gather r{} t{} p{} miss{} stale{} done{} | regular V{} I{} M{} D{} | seam V{} I{} M{} D{} | cull o{} s{} f{} c{} x{} | queued {} bp{} rb{}",
            draw_path,
            debug_view.label(),
            self.benchmark.label(),
            camera_delta[0],
            camera_delta[1],
            camera_delta[2],
            self.yaw,
            self.pitch,
            u8::from(self.window.has_focus()),
            u8::from(self.cursor_grabbed),
            self.keys.len(),
            residency.resident_pages,
            diagnostics.gpu_published_jobs,
            diagnostics.gpu_submitted_jobs,
            diagnostics.gpu_stale_rejections,
            diagnostics.gpu_overflow_rejections,
            diagnostics.gpu_incomplete_rejections,
            diagnostics.gather_regular_samples,
            diagnostics.gather_transition_samples,
            diagnostics.gather_table_probes,
            diagnostics.gather_page_misses,
            diagnostics.gather_stale_targets,
            diagnostics.gather_completed,
            diagnostics.regular_vertices,
            diagnostics.regular_indices,
            diagnostics.regular_meshlets,
            diagnostics.visible_regular_draws,
            diagnostics.transition_vertices,
            diagnostics.transition_indices,
            diagnostics.transition_meshlets,
            diagnostics.visible_transition_draws,
            diagnostics.meshlet_draw_overflow,
            diagnostics.meshlet_stale_rejections,
            diagnostics.meshlet_frustum_rejections,
            diagnostics.meshlet_cone_rejections,
            diagnostics.meshlet_invalid_candidates,
            render.queued_surfaces,
            render.pending_backpressure + u64::from(residency.backpressure_events),
            diagnostics.readback_failures,
        ));
    }

    fn advance_horizon_trace(&mut self, timing: Option<TimingSample>) -> bool {
        let Some(trace) = self.horizon_trace.as_mut() else {
            return false;
        };
        trace.frames = trace.frames.saturating_add(1);
        if let Some(sample) = timing {
            trace.samples.push(sample);
        }
        let (render, residency) = {
            let pass = self
                .renderer
                .find_pass_mut::<PlanetaryVoxelRenderPass>()
                .expect("planetary pass");
            (pass.counters(), pass.residency().counters())
        };
        trace.maximum_resident_pages = trace.maximum_resident_pages.max(residency.resident_pages);
        trace.maximum_queued_surfaces = trace.maximum_queued_surfaces.max(render.queued_surfaces);
        trace.maximum_backpressure = trace
            .maximum_backpressure
            .max(render.pending_backpressure + u64::from(residency.backpressure_events));

        let (
            streaming_failure,
            settled,
            handoffs,
            uploads,
            upload_bytes,
            evictions,
            surface_jobs,
            transition_jobs,
            cancellations,
            pending_staged_jobs,
        ) = {
            let streaming = self
                .horizon_streaming
                .as_ref()
                .expect("horizon trace uses dynamic streaming");
            (
                streaming.failure.clone(),
                streaming.pending.is_none() && streaming.active.is_some(),
                streaming.handoffs,
                streaming.uploads,
                streaming.upload_bytes,
                streaming.evictions,
                streaming.surface_jobs,
                streaming.transition_jobs,
                streaming.cancellations,
                streaming
                    .pending
                    .as_ref()
                    .map_or(0, |pending| pending.surface_jobs),
            )
        };
        if let Some(failure) = streaming_failure {
            eprintln!(
                "PLANET_HORIZON_TRACE status=FAIL frames={} reason={failure:?}",
                trace.frames
            );
            return true;
        }
        if let Some(expected) = trace.awaiting_handoff {
            if settled && handoffs >= expected {
                trace.awaiting_handoff = None;
            } else {
                if trace.next_handoff_step == 1
                    && !trace.cancellation_injected
                    && pending_staged_jobs >= 1
                {
                    self.canonical_camera_m[0] += 6.4;
                    trace.cancellation_injected = true;
                }
                return false;
            }
        }

        if trace.next_handoff_step < 5 {
            if !settled {
                return false;
            }
            match trace.next_handoff_step {
                0 => self.canonical_camera_m[0] += 6.4,
                1 => {
                    self.canonical_camera_m[0] = -(EARTH_RADIUS_METERS + 0.6);
                    self.canonical_camera_m[2] = -3.3;
                }
                2 => self.canonical_camera_m[1] = HORIZON_ALTITUDES_METERS[1],
                3 => self.canonical_camera_m[1] = HORIZON_ALTITUDES_METERS[2],
                4 => self.canonical_camera_m[1] = HORIZON_ALTITUDES_METERS[0],
                _ => unreachable!(),
            }
            trace.next_handoff_step += 1;
            trace.awaiting_handoff = Some(handoffs.saturating_add(1));
            return false;
        }

        match trace.resize_stage {
            0 => {
                let _ = self
                    .window
                    .request_inner_size(winit::dpi::PhysicalSize::new(960, 540));
                trace.resize_stage = 1;
                trace.resize_frames = 0;
                false
            }
            1 if trace.resize_frames < 30 => {
                trace.resize_frames += 1;
                false
            }
            1 => {
                let _ = self
                    .window
                    .request_inner_size(winit::dpi::PhysicalSize::new(1280, 720));
                trace.resize_stage = 2;
                trace.resize_frames = 0;
                false
            }
            2 if trace.resize_frames < 60 => {
                trace.resize_frames += 1;
                false
            }
            _ => {
                let timing = summarize_timings(&trace.samples).unwrap_or(TimingSummary {
                    cpu_p50_ms: f32::NAN,
                    cpu_p95_ms: f32::NAN,
                    gpu_p50_ms: f32::NAN,
                    gpu_p95_ms: f32::NAN,
                });
                let dropped = u64::from(self.planet_diagnostics.gpu_stale_rejections)
                    + u64::from(self.planet_diagnostics.gpu_overflow_rejections)
                    + u64::from(self.planet_diagnostics.gpu_incomplete_rejections);
                let pass = u32::from(
                    dropped == 0
                        && trace.maximum_backpressure == 0
                        && self.planet_diagnostics.readback_failures == 0
                        && render.queued_surfaces == 0
                        && cancellations >= 1
                        && handoffs >= 6,
                );
                eprintln!(
                    "PLANET_HORIZON_TRACE status={} frames={} handoffs={} cancellations={} resident_max={} queued_max={} uploads={} upload_bytes={} evictions={} extraction_jobs={} transition_jobs={} dropped={} backpressure={} cpu_p50_ms={:.6} cpu_p95_ms={:.6} gpu_p50_ms={:.6} gpu_p95_ms={:.6}",
                    if pass == 1 { "PASS" } else { "FAIL" },
                    trace.frames,
                    handoffs,
                    cancellations,
                    trace.maximum_resident_pages,
                    trace.maximum_queued_surfaces,
                    uploads,
                    upload_bytes,
                    evictions,
                    surface_jobs,
                    transition_jobs,
                    dropped,
                    trace.maximum_backpressure,
                    timing.cpu_p50_ms,
                    timing.cpu_p95_ms,
                    timing.gpu_p50_ms,
                    timing.gpu_p95_ms,
                );
                true
            }
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        event_loop.listen_device_events(DeviceEvents::WhenFocused);
        let window = Arc::new(
            event_loop
                .create_window(
                    Window::default_attributes()
                        .with_title("Helio Planet Voxels")
                        .with_inner_size(winit::dpi::PhysicalSize::new(1280, 720)),
                )
                .expect("planet demo window"),
        );
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("planet demo surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("planet demo GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Helio Planet Voxel Demo Device"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("planet demo device");
        device.on_uncaptured_error(Arc::new(|error| {
            log::error!("planet demo uncaptured GPU error: {error}");
        }));
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let size = window.inner_size();
        let capabilities = surface.get_capabilities(&adapter);
        let surface_format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(capabilities.formats[0]);
        let alpha_mode = capabilities.alpha_modes[0];
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format: surface_format,
                color_space: wgpu::SurfaceColorSpace::Auto,
                width: size.width,
                height: size.height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        let renderer_config =
            RendererConfig::new(size.width, size.height, surface_format).with_render_scale(1.0);
        let scene = Scene::new(device.clone(), queue.clone());
        let debug_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Demo Debug Camera"),
            size: core::mem::size_of::<helio::DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planet Demo Cull Stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let debug_state = Arc::new(std::sync::Mutex::new(DebugDrawState::default()));
        let planet_config = if self.auto_benchmark {
            PlanetaryVoxelRenderConfig::benchmark_demo()
        } else {
            PlanetaryVoxelRenderConfig::horizon_demo()
        };
        let planet_pass =
            PlanetaryVoxelRenderPass::new(&device, &queue, surface_format, planet_config)
                .expect("bounded planetary render pass");
        let mut graph = RenderGraph::new(&device, &queue);
        graph.add_pass(Box::new(planet_pass));
        graph.add_pass(Box::new(FxaaPass::new(&device, surface_format)));
        graph.lock(size.width, size.height);
        let mut renderer = Renderer::new(
            device.clone(),
            queue.clone(),
            renderer_config.surface_format,
            renderer_config.width,
            renderer_config.height,
            renderer_config.render_scale,
            renderer_config,
            scene,
            graph,
            debug_state,
            debug_camera_buffer,
            cull_stats_buffer,
        );
        renderer.set_jitter_enabled(false);

        let planet = PlanetId(*b"HELIO-EARTH-DEMO");
        let (canonical_camera_m, pages) = if self.auto_benchmark {
            build_benchmark_patch(planet)
        } else {
            (horizon_spawn_camera(), Vec::new())
        };
        let expected_jobs = pages.len() as u64;
        if !pages.is_empty() {
            let pass = renderer
                .find_pass_mut::<PlanetaryVoxelRenderPass>()
                .expect("planetary pass");
            let camera = PlanetPosition::from_meters(canonical_camera_m).unwrap();
            pass.set_planet_frame(&queue, PlanetFrameUniform::from_camera(planet, camera, 1))
                .unwrap();
            pass.apply_upload_batch(
                &device,
                &queue,
                pages.iter().map(|page| page.page_upload.clone()).collect(),
            )
            .unwrap();
            let target_keys = pages
                .iter()
                .map(|page| page.page_upload.key)
                .collect::<std::collections::BTreeSet<_>>();
            let dependencies = pages
                .iter()
                .flat_map(|page| page.surface.required_pages().unwrap())
                .filter(|key| !target_keys.contains(key))
                .collect::<std::collections::BTreeSet<_>>();
            let support_uploads = dependencies
                .into_iter()
                .map(|key| build_page_upload(key.planet, key.page, DEMO_SOURCE_GENERATION))
                .collect::<Vec<_>>();
            for chunk in support_uploads.chunks(pass.residency().config().max_batch_pages as usize)
            {
                pass.apply_upload_batch(&device, &queue, chunk.to_vec())
                    .unwrap();
            }
            pass.apply_visible_set(
                &queue,
                VisiblePageSet {
                    frame_index: 1,
                    pages: pages
                        .iter()
                        .map(|page| VisiblePage {
                            key: page.page_upload.key,
                            generation: page.page_upload.generation,
                            transition_mask: page.surface.transition_mask,
                        })
                        .collect(),
                },
            )
            .unwrap();
            for page in pages {
                pass.queue_surface(page.surface).unwrap();
            }
        }
        let initial_orientation = Quat::from_euler(EulerRot::YXZ, INITIAL_YAW, INITIAL_PITCH, 0.0);
        let horizon_streaming = (!self.auto_benchmark).then(|| {
            HorizonStreamingState::new(
                canonical_camera_m,
                horizon_streaming_focus(canonical_camera_m, initial_orientation),
            )
        });

        self.state = Some(AppState {
            window,
            surface,
            device,
            queue,
            surface_format,
            alpha_mode,
            renderer,
            planet,
            canonical_camera_m,
            spawn_camera_m: canonical_camera_m,
            frame_index: 1,
            yaw: INITIAL_YAW,
            pitch: INITIAL_PITCH,
            keys: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            cursor_grabbed: false,
            last_frame: Instant::now(),
            last_title_update: Instant::now(),
            benchmark: PlanetBenchmark::default(),
            auto_benchmark_expected_jobs: self.auto_benchmark.then_some(expected_jobs),
            auto_fill_samples: Vec::new(),
            auto_completion_frames: 0,
            horizon_streaming,
            horizon_altitude_index: 0,
            planet_diagnostics: PlanetaryRenderDiagnostics::default(),
            horizon_trace: self.auto_horizon_trace.then(HorizonTrace::default),
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::Focused(false) => {
                state.reset_transient_input();
            }
            WindowEvent::Focused(true) => state.last_frame = Instant::now(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if state.cursor_grabbed {
                    state.reset_transient_input();
                } else {
                    event_loop.exit();
                }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: key_state,
                        physical_key: PhysicalKey::Code(key),
                        repeat,
                        ..
                    },
                ..
            } => match (key_state, key, repeat) {
                (ElementState::Pressed, KeyCode::F2, false) => {
                    state.benchmark.cancel();
                    state
                        .renderer
                        .find_pass_mut::<PlanetaryVoxelRenderPass>()
                        .expect("planetary pass")
                        .toggle_draw_path(&state.queue);
                    state.last_title_update = Instant::now()
                        .checked_sub(std::time::Duration::from_secs(1))
                        .unwrap_or_else(Instant::now);
                }
                (ElementState::Pressed, KeyCode::F3, false) => {
                    state.benchmark.cancel();
                    state
                        .renderer
                        .find_pass_mut::<PlanetaryVoxelRenderPass>()
                        .expect("planetary pass")
                        .cycle_debug_view(&state.queue);
                    state.last_title_update = Instant::now()
                        .checked_sub(std::time::Duration::from_secs(1))
                        .unwrap_or_else(Instant::now);
                }
                (ElementState::Pressed, KeyCode::F4, false) => {
                    state.benchmark.start();
                    let pass = state
                        .renderer
                        .find_pass_mut::<PlanetaryVoxelRenderPass>()
                        .expect("planetary pass");
                    pass.set_debug_view(&state.queue, PlanetaryDebugView::Material);
                    pass.set_draw_path(&state.queue, PlanetaryDrawPath::PageIndexed);
                    state.last_title_update = Instant::now()
                        .checked_sub(std::time::Duration::from_secs(1))
                        .unwrap_or_else(Instant::now);
                }
                (ElementState::Pressed, KeyCode::F5, false)
                    if state.horizon_streaming.is_some() =>
                {
                    let sign = if state.canonical_camera_m[0] >= 0.0 {
                        -1.0
                    } else {
                        1.0
                    };
                    state.canonical_camera_m[0] = sign * (EARTH_RADIUS_METERS + 0.6);
                    state.canonical_camera_m[2] = sign * 3.3;
                    state.last_title_update = Instant::now()
                        .checked_sub(std::time::Duration::from_secs(1))
                        .unwrap_or_else(Instant::now);
                }
                (ElementState::Pressed, KeyCode::F6, false)
                    if state.horizon_streaming.is_some() =>
                {
                    state.horizon_altitude_index =
                        (state.horizon_altitude_index + 1) % HORIZON_ALTITUDES_METERS.len();
                    state.canonical_camera_m[1] =
                        HORIZON_ALTITUDES_METERS[state.horizon_altitude_index];
                    state.last_title_update = Instant::now()
                        .checked_sub(std::time::Duration::from_secs(1))
                        .unwrap_or_else(Instant::now);
                }
                (ElementState::Pressed, _, _) => {
                    state.keys.insert(key);
                }
                (ElementState::Released, _, _) => {
                    state.keys.remove(&key);
                }
            },
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } if !state.cursor_grabbed => {
                state.window.focus_window();
                let grabbed = state
                    .window
                    .set_cursor_grab(CursorGrabMode::Confined)
                    .or_else(|_| state.window.set_cursor_grab(CursorGrabMode::Locked))
                    .is_ok();
                if grabbed {
                    state.cursor_grabbed = true;
                    state.window.set_cursor_visible(false);
                }
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                let dt = now.duration_since(state.last_frame).as_secs_f64().min(0.05);
                state.last_frame = now;
                state.update(dt);
                state.update_planet_frame();
                state.update_horizon_streaming();
                state.update_title();
                let size = state.window.inner_size();
                if size.width == 0 || size.height == 0 {
                    return;
                }
                let output = match state.surface.get_current_texture() {
                    wgpu::CurrentSurfaceTexture::Success(texture)
                    | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
                    _ => return,
                };
                let view = output
                    .texture
                    .create_view(&wgpu::TextureViewDescriptor::default());
                let render_succeeded = match state
                    .renderer
                    .render(&state.camera(size.width, size.height), &view)
                {
                    Ok(_) => true,
                    Err(error) => {
                        log::error!("planet render error: {error:?}");
                        false
                    }
                };
                let mut horizon_trace_complete = false;
                if render_succeeded {
                    let timing = state
                        .renderer
                        .timing_snapshot()
                        .passes
                        .iter()
                        .find(|timing| timing.name == "PlanetaryVoxel")
                        .and_then(|timing| {
                            Some(TimingSample {
                                cpu_ms: timing.cpu_ms?,
                                gpu_ms: timing.gpu_ms?,
                            })
                        });
                    let (draw_path, submitted_jobs) = {
                        let pass = state
                            .renderer
                            .find_pass_mut::<PlanetaryVoxelRenderPass>()
                            .expect("planetary pass");
                        (pass.draw_path(), pass.counters().submitted_jobs)
                    };
                    if let Some(expected_jobs) = state.auto_benchmark_expected_jobs {
                        if state.benchmark.phase == BenchmarkPhase::Idle {
                            if let Some(sample) = timing {
                                state.auto_fill_samples.push(sample);
                            }
                            if submitted_jobs >= expected_jobs {
                                if let Some(fill) = summarize_timings(&state.auto_fill_samples) {
                                    eprintln!(
                                        "PLANET_MESHLET_FILL_BENCHMARK jobs={} frames={} full_pass_cpu_p50_ms={:.6} full_pass_cpu_p95_ms={:.6} full_pass_gpu_p50_ms={:.6} full_pass_gpu_p95_ms={:.6}",
                                        expected_jobs,
                                        state.auto_fill_samples.len(),
                                        fill.cpu_p50_ms,
                                        fill.cpu_p95_ms,
                                        fill.gpu_p50_ms,
                                        fill.gpu_p95_ms,
                                    );
                                }
                                state.benchmark.start();
                                let pass = state
                                    .renderer
                                    .find_pass_mut::<PlanetaryVoxelRenderPass>()
                                    .expect("planetary pass");
                                pass.set_debug_view(&state.queue, PlanetaryDebugView::Material);
                                pass.set_draw_path(&state.queue, PlanetaryDrawPath::PageIndexed);
                            }
                        } else if let Some(next_path) = state.benchmark.record(draw_path, timing) {
                            state
                                .renderer
                                .find_pass_mut::<PlanetaryVoxelRenderPass>()
                                .expect("planetary pass")
                                .set_draw_path(&state.queue, next_path);
                        }
                    } else if let Some(next_path) = state.benchmark.record(draw_path, timing) {
                        state
                            .renderer
                            .find_pass_mut::<PlanetaryVoxelRenderPass>()
                            .expect("planetary pass")
                            .set_draw_path(&state.queue, next_path);
                    }
                    horizon_trace_complete = state.advance_horizon_trace(timing);
                }
                state.queue.present(output);
                if horizon_trace_complete {
                    event_loop.exit();
                    return;
                }
                if self.auto_benchmark && state.benchmark.phase == BenchmarkPhase::Complete {
                    let diagnostics = state
                        .renderer
                        .find_pass_mut::<PlanetaryVoxelRenderPass>()
                        .expect("planetary pass")
                        .poll_diagnostics(&state.device, &state.queue);
                    state.auto_completion_frames = state.auto_completion_frames.saturating_add(1);
                    if state.auto_completion_frames >= BENCHMARK_DIAGNOSTIC_SETTLE_FRAMES {
                        eprintln!(
                            "PLANET_MESHLET_CULL_COUNTS regular_meshlets={} transition_meshlets={} regular_draws={} transition_draws={} frustum_rejects={} cone_rejects={} stale={} overflow={} invalid={}",
                            diagnostics.regular_meshlets,
                            diagnostics.transition_meshlets,
                            diagnostics.visible_regular_draws,
                            diagnostics.visible_transition_draws,
                            diagnostics.meshlet_frustum_rejections,
                            diagnostics.meshlet_cone_rejections,
                            diagnostics.meshlet_stale_rejections,
                            diagnostics.meshlet_draw_overflow,
                            diagnostics.meshlet_invalid_candidates,
                        );
                        event_loop.exit();
                        return;
                    }
                }
                state.window.request_redraw();
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        let Some(state) = &mut self.state else {
            return;
        };
        if let DeviceEvent::MouseMotion { delta: (x, y) } = event {
            if state.cursor_grabbed {
                state.mouse_delta.0 += x as f32;
                state.mouse_delta.1 += y as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn advance_camera(position_m: &mut [f64; 3], keys: &HashSet<KeyCode>, orientation: Quat, dt: f64) {
    let look_forward = orientation * -Vec3::Z;
    let look_right = orientation * Vec3::X;
    let forward = Vec3::new(look_forward.x, 0.0, look_forward.z).normalize_or_zero();
    let right = Vec3::new(look_right.x, 0.0, look_right.z).normalize_or_zero();
    let mut direction = Vec3::ZERO;
    if keys.contains(&KeyCode::KeyW) {
        direction += forward;
    }
    if keys.contains(&KeyCode::KeyS) {
        direction -= forward;
    }
    if keys.contains(&KeyCode::KeyA) {
        direction -= right;
    }
    if keys.contains(&KeyCode::KeyD) {
        direction += right;
    }
    if keys.contains(&KeyCode::Space) {
        direction += Vec3::Y;
    }
    if keys.contains(&KeyCode::ShiftLeft) {
        direction -= Vec3::Y;
    }
    let step = direction.normalize_or_zero() * (MOVE_SPEED_METERS_PER_SECOND * dt) as f32;
    for axis in 0..3 {
        position_m[axis] += f64::from(step[axis]);
    }
}

fn horizon_spawn_camera() -> [f64; 3] {
    [EARTH_RADIUS_METERS + 0.6, HORIZON_ALTITUDES_METERS[0], 0.0]
}

fn horizon_streaming_focus(camera_m: [f64; 3], orientation: Quat) -> [f64; 3] {
    let forward = orientation * -Vec3::Z;
    let surface_y = -LOD0_CELL_SIZE_METERS;
    if forward.y < -1.0e-4 {
        let distance = ((surface_y - camera_m[1]) / f64::from(forward.y)).clamp(0.0, 3_000.0);
        [
            camera_m[0] + f64::from(forward.x) * distance,
            surface_y,
            camera_m[2] + f64::from(forward.z) * distance,
        ]
    } else {
        [camera_m[0], surface_y, camera_m[2]]
    }
}

fn horizon_focus_lod0_cell(camera_m: [f64; 3]) -> [i64; 3] {
    [
        (camera_m[0] / LOD0_CELL_SIZE_METERS).floor() as i64,
        -1,
        (camera_m[2] / LOD0_CELL_SIZE_METERS).floor() as i64,
    ]
}

fn horizon_minimum_lod(altitude_m: f64) -> u8 {
    if altitude_m < 32.0 {
        0
    } else if altitude_m < 256.0 {
        2
    } else {
        5
    }
}

fn rejection_counts(diagnostics: &PlanetaryRenderDiagnostics) -> [u32; 3] {
    [
        diagnostics.gpu_stale_rejections,
        diagnostics.gpu_overflow_rejections,
        diagnostics.gpu_incomplete_rejections,
    ]
}

struct DemoPage {
    page_upload: PageUpload,
    surface: PlanetarySurfaceRequest,
}

fn build_benchmark_patch(planet: PlanetId) -> ([f64; 3], Vec<DemoPage>) {
    let radius_cell = (EARTH_RADIUS_METERS / LOD0_CELL_SIZE_METERS) as i64;
    let first_page_x = radius_cell.div_euclid(PAGE_EDGE_CELLS);
    let first_page_z = -(BENCHMARK_GRID_EDGE / 2);
    let mut pages = Vec::with_capacity((BENCHMARK_GRID_EDGE * BENCHMARK_GRID_EDGE) as usize);
    for z in 0..BENCHMARK_GRID_EDGE {
        for x in 0..BENCHMARK_GRID_EDGE {
            let page = PageKey::new(0, [first_page_x + x, -1, first_page_z + z]);
            pages.push(build_demo_page(planet, page, 0));
        }
    }
    let first_min = pages[0].page_upload.key.page.lod0_cell_min().unwrap();
    let canonical_camera_m = [
        (first_min[0] - 8) as f64 * LOD0_CELL_SIZE_METERS,
        15.0 * LOD0_CELL_SIZE_METERS,
        0.0,
    ];
    (canonical_camera_m, pages)
}

fn build_demo_page(planet: PlanetId, page: PageKey, transition_mask: u8) -> DemoPage {
    let key = PlanetPageKey::new(planet, page);
    DemoPage {
        page_upload: build_page_upload(planet, page, DEMO_SOURCE_GENERATION),
        surface: PlanetarySurfaceRequest {
            key,
            generation: DEMO_SOURCE_GENERATION,
            transition_mask,
            dirty_microbricks: u64::MAX,
        },
    }
}

fn build_surface_upload(
    planet: PlanetId,
    page: PageKey,
    transition_mask: u8,
    generation: SourceGeneration,
) -> PlanetarySurfaceRequest {
    PlanetarySurfaceRequest {
        key: PlanetPageKey::new(planet, page),
        generation,
        transition_mask,
        dirty_microbricks: u64::MAX,
    }
}

fn build_page_upload(planet: PlanetId, page: PageKey, generation: SourceGeneration) -> PageUpload {
    let minimum = page.lod0_cell_min().unwrap();
    let scale = 1_i64 << page.lod;
    let mut cells = Vec::with_capacity(PAGE_EDGE * PAGE_EDGE * PAGE_EDGE);
    for z in 0..PAGE_EDGE as i64 {
        for y in 0..PAGE_EDGE as i64 {
            for x in 0..PAGE_EDGE as i64 {
                cells.push(ExtractionFixtureKind::Plane.sample_canonical([
                    minimum[0] + x * scale,
                    minimum[1] + y * scale,
                    minimum[2] + z * scale,
                ]));
            }
        }
    }
    PageUpload::new(PlanetPageKey::new(planet, page), generation, cells).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_camera_ray_intersects_a_resident_fine_page() {
        let planet = PlanetId(*b"HELIO-EARTH-DEMO");
        let camera = horizon_spawn_camera();
        let orientation = Quat::from_euler(EulerRot::YXZ, INITIAL_YAW, INITIAL_PITCH, 0.0);
        let focus_m = horizon_streaming_focus(camera, orientation);
        let plan = HorizonLodFixturePlan::build(
            horizon_focus_lod0_cell(focus_m),
            HORIZON_ROOT_LOD,
            HORIZON_MAX_PLAN_PAGES,
        )
        .unwrap();
        let forward = orientation * -Vec3::Z;
        assert!(forward.x > 0.0 && forward.y < 0.0);
        let surface_y = -LOD0_CELL_SIZE_METERS;
        let distance = (surface_y - camera[1]) / f64::from(forward.y);
        let intersection_x = camera[0] + f64::from(forward.x) * distance;
        let intersection_z = camera[2] + f64::from(forward.z) * distance;
        let intersection_cell =
            horizon_focus_lod0_cell([intersection_x, surface_y, intersection_z]);
        let fine = PageKey::address_lod0_cell(0, intersection_cell).unwrap().0;
        assert!(plan.topology().pages().any(|page| page == fine));

        let planet_camera = PlanetPosition::from_meters(camera).unwrap();
        let frame = PlanetFrameUniform::from_camera(planet, planet_camera, 1);
        let metadata = helio_planet_voxel_core::GpuPageMeta::new(
            fine,
            frame.frame_origin_lod0_cell(),
            0,
            1,
            0,
        )
        .unwrap();
        let local_surface = [16.0, 31.0, 16.0];
        let world = metadata.camera_local_position_m(frame, local_surface);
        let camera_uniform = Camera::perspective_look_at(
            Vec3::ZERO,
            orientation * -Vec3::Z,
            orientation * Vec3::Y,
            std::f32::consts::FRAC_PI_3,
            1280.0 / 720.0,
            0.01,
            2_000.0,
        );
        let clip = camera_uniform.proj * camera_uniform.view * Vec3::from_array(world).extend(1.0);
        let ndc = clip.truncate() / clip.w;
        assert!(
            clip.w > 0.0
                && ndc.x.abs() <= 1.0
                && ndc.y.abs() <= 1.0
                && (0.0..=1.0).contains(&ndc.z),
            "surface vertex world={world:?} projects to clip={clip:?}, ndc={ndc:?}"
        );
    }

    #[test]
    fn movement_is_horizontal_and_uses_explicit_vertical_keys() {
        let orientation = Quat::from_euler(EulerRot::YXZ, INITIAL_YAW, INITIAL_PITCH, 0.0);
        let mut position = [0.0; 3];
        let mut keys = HashSet::from([KeyCode::KeyW]);
        advance_camera(&mut position, &keys, orientation, 1.0);
        assert!((position[0] - MOVE_SPEED_METERS_PER_SECOND).abs() < 1.0e-5);
        assert_eq!(position[1], 0.0);
        assert!(position[2].abs() < 1.0e-5);

        position = [0.0; 3];
        keys = HashSet::from([KeyCode::Space]);
        advance_camera(&mut position, &keys, orientation, 1.0);
        assert_eq!(position, [0.0, MOVE_SPEED_METERS_PER_SECOND, 0.0]);
    }

    #[test]
    fn retained_transition_surfaces_cover_old_and_replacement_masks() {
        let first =
            HorizonLodFixturePlan::build([0, -1, 0], HORIZON_ROOT_LOD, HORIZON_MAX_PLAN_PAGES)
                .unwrap();
        let second =
            HorizonLodFixturePlan::build([64, -1, 0], HORIZON_ROOT_LOD, HORIZON_MAX_PLAN_PAGES)
                .unwrap();
        let replacement_pages = second.topology().pages().collect::<HashSet<_>>();
        let mut changed_shared_masks = 0;
        for page in first
            .topology()
            .pages()
            .filter(|page| replacement_pages.contains(page))
        {
            let old_mask = first.topology().transition_mask(page).unwrap();
            let replacement_mask = second.topology().transition_mask(page).unwrap();
            let available_mask = old_mask | replacement_mask;
            assert_eq!(old_mask & !available_mask, 0);
            assert_eq!(replacement_mask & !available_mask, 0);
            changed_shared_masks += usize::from(old_mask != replacement_mask);
        }
        assert!(
            changed_shared_masks > 0,
            "the boundary-crossing fixture must exercise retained-page mask changes"
        );
    }

    #[test]
    fn matched_benchmark_is_bounded_and_switches_paths_after_equal_samples() {
        let mut benchmark = PlanetBenchmark::default();
        benchmark.start();
        let sample = Some(TimingSample {
            cpu_ms: 0.25,
            gpu_ms: 0.5,
        });
        for _ in 0..BENCHMARK_WARMUP_FRAMES {
            assert_eq!(
                benchmark.record(PlanetaryDrawPath::PageIndexed, sample),
                None
            );
        }
        assert_eq!(benchmark.phase, BenchmarkPhase::PageSamples);
        for index in 0..BENCHMARK_SAMPLE_FRAMES {
            let next = benchmark.record(PlanetaryDrawPath::PageIndexed, sample);
            assert_eq!(
                next,
                (index + 1 == BENCHMARK_SAMPLE_FRAMES).then_some(PlanetaryDrawPath::Meshlets)
            );
        }
        for _ in 0..BENCHMARK_WARMUP_FRAMES {
            assert_eq!(benchmark.record(PlanetaryDrawPath::Meshlets, sample), None);
        }
        assert_eq!(benchmark.phase, BenchmarkPhase::MeshletSamples);
        for _ in 0..BENCHMARK_SAMPLE_FRAMES {
            assert_eq!(benchmark.record(PlanetaryDrawPath::Meshlets, sample), None);
        }
        assert_eq!(benchmark.phase, BenchmarkPhase::Complete);
        assert_eq!(benchmark.page_samples.len(), BENCHMARK_SAMPLE_FRAMES);
        assert_eq!(benchmark.meshlet_samples.len(), BENCHMARK_SAMPLE_FRAMES);
        assert_eq!(benchmark.page_summary.unwrap().gpu_p95_ms, 0.5);
        assert_eq!(benchmark.meshlet_summary.unwrap().cpu_p50_ms, 0.25);
    }

    #[test]
    fn benchmark_patch_is_bounded_unique_and_entirely_fine_lod() {
        let planet = PlanetId(*b"HELIO-EARTH-DEMO");
        let (_, pages) = build_benchmark_patch(planet);
        assert_eq!(
            pages.len(),
            (BENCHMARK_GRID_EDGE * BENCHMARK_GRID_EDGE) as usize
        );
        let keys: HashSet<_> = pages.iter().map(|page| page.page_upload.key).collect();
        assert_eq!(keys.len(), pages.len());
        assert!(keys.iter().all(|key| key.page.lod == 0));
    }
}
