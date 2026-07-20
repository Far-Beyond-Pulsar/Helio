//! M3-β T9: cull/draw pass GPU timing + the §14.2 readback-latency decision.
//!
//! This is the measurement task that closes the perf campaign's deferred
//! GPU-side claims (perf-validation report §2.7, contract #5/#43/#47's
//! timing sub-claim) now that a real cull pass (T5) and indirect draw
//! executor (T7) exist to time. Four deliverables, each a section below:
//!
//! 1. Cull dispatch GPU time at N ∈ {1k, 10k, 100k} tokens × visible
//!    fractions {100%, 50%, 10%}.
//! 2. Indirect draw GPU time for the visible set at the same 9 cells.
//! 3. THE §14.2 DECISION: readback-then-clamp (what T7/T8 do today) vs
//!    conservative max-count (dispatch `capacity` draws unconditionally,
//!    `instance_count=0` on the culled slots, no readback) — measured
//!    end-to-end (CPU wall time) at N=10,000/10% visible, the scale the
//!    plan brief itself names as "many culled draws for (b) to waste."
//! 4. Cull+draw total vs. a no-cull "draw everything" baseline at the same
//!    3 N-scales — first real evidence for perf-report DEFERRED claim #5
//!    ("CPU out of the GPU inner loop").
//!
//! ## Methodology inherited from T3 (perf-validation report §1.2/§1.3)
//!
//! **GPU-timestamp bracket.** [`GpuTimer`] below is copied wholesale from
//! `pulsar_scenedb/benches/gpu_timing.rs` (that file's own reuse note:
//! "later tasks can copy it wholesale into their own bench files") — same
//! two-submit shape (`submit([start])` → `work()` → `submit([end, resolve,
//! copy])`), same reasoning (`wgpu-core-30.0.0`'s `executions.insert(0, ..)`
//! always splices the queue's pending-writes belt at position 0 of a
//! submission, so a single-submit bracket built around `queue.write_buffer`
//! reads ~0 regardless of payload).
//!
//! **One documented deviation from T3's `work()` contract.** T3's doc
//! comment states `work` "must NOT itself call `queue.submit`" — true for
//! its own payload (`queue.write_buffer` calls only, which rely on the
//! pending-writes belt). This file's payload is different: real compute
//! dispatches / indexed-indirect draws, which cannot be expressed as
//! pending writes at all — they only exist once recorded into a command
//! buffer and submitted. This bench's `work()` closures therefore DO call
//! `queue.submit` on their own encoder. This is still bracket-correct: wgpu
//! guarantees same-queue submissions execute in submission order (T3's own
//! file doc cites this exact guarantee for why the two-submit form works),
//! and that guarantee does not depend on which submission call-site issued
//! which encoder — only on submission ORDER, which is: [start_ts] →
//! [work()'s own submit(s)] → [end_ts, resolve, copy]. No `write_buffer`
//! call happens inside any of this file's timed brackets, so the pending-
//! writes-splice hazard T3 discovered does not apply here at all; the
//! two-submit *shape* is kept purely for methodology continuity with T3/T4/
//! T6, not because this payload has the same failure mode.
//!
//! **Amplification.** Cull dispatch recording is O(1) CPU-side regardless
//! of N (one `dispatch_workgroups` call encodes the whole N-thread launch),
//! so cull timing uses a FIXED repeat count ([`REPEATS_CULL`]) at every
//! scale — GPU execution time already scales with N per dispatch, no
//! CPU-side blow-up risk. Draw timing is different: `DrawExecutor::record`
//! issues one CPU-side `draw_indexed_indirect` call PER COMMAND (draw.rs's
//! documented mechanism, not `multi_draw_indexed_indirect`), so repeating a
//! large command_count many times would blow the "sane wall time" budget —
//! draw repeat count is therefore ADAPTIVE (`draw_repeats`), targeting a
//! roughly constant total command count per bracket regardless of scale.
//!
//! **Fixed-slack honesty gates**, not noise-derived (T3's dead-bracket
//! lesson, perf-validation report §1.3(c)): every monotonicity assert below
//! uses a constant nanosecond slack, and the payload-visibility check
//! compares the largest-N cull mean against an INDEPENDENTLY measured noise
//! floor (an empty-`work()` bracket — literally zero GPU commands between
//! the two timestamp submits), not a floor derived from the same trend
//! being validated.
//!
//! ## Scene fixture shape
//!
//! `SpatialCell`/`CellStorage` hard-caps at 1024 rows/cell
//! (`pulsar_scenedb::MAX_PAGE_CAPACITY`), so N=100,000 tokens needs ~98
//! independently-registered cells (gpu_timing.rs's T3 precedent for "one
//! bigger cell is not an option"). Rows are placed at one of two fixed
//! positions: `x=0` (comfortably inside `support::view_proj_90deg`'s
//! fovy=90 frustum, matching `cull_pass.rs` row 0's geometry) or `x=1000`
//! (frustum-culled, matching `cull_pass.rs` row 1's geometry) — the first
//! `round(N * visible_fraction)` GLOBAL rows (in cell-registration order)
//! get the visible position, the rest get the culled one. This is verified,
//! not assumed: every fixture runs one untimed real cull dispatch + counter
//! readback before any timing loop starts, asserting `stale_drops==0`,
//! `oob_drops==0`, and `visible_count == round(N*frac)` exactly (house law:
//! self-verifying fixture guards).

#[path = "../tests/support/mod.rs"]
mod support;

use helio_scenedb::cull::{CullOutputBuffers, CullPass, CullRecord, CullUniforms};
use helio_scenedb::draw::{DrawExecutor, DrawUniforms};
use helio_scenedb::SceneDbBinding;
use pulsar_scenedb::gpu::{
    CellId, CellSlot, ClusterBuffer, EngineGpuContext, FrameDriver, GeometryArena, HarvestPipeline,
    HarvestStaging, MaterialRegistry, MeshClass, MeshRegistry, MeshletBuffer, RegionClassConfig,
    SceneGpuConfig, SceneGpuStore, View, ViewTokenBuffers,
};
use pulsar_scenedb::{Aabb, Handle, InstanceInfo, Scratchpad, SpatialCell};
use std::time::Instant;

use support::{frustum_planes_90deg, mesh, readback, translation, view_proj_90deg};

const MAX_CELL_ROWS: u32 = 1024;
const NS: [u32; 3] = [1_000, 10_000, 100_000];
const FRACS: [f32; 3] = [1.0, 0.5, 0.1];

const WARMUP_PASS: usize = 2;
const ITERS_PASS: usize = 8;
const REPEATS_CULL: u32 = 32;
/// Adaptive draw-bracket repeat count: target ~`DRAW_TARGET_TOTAL` recorded
/// `draw_indexed_indirect` calls per bracket sample regardless of scale
/// (module doc: CPU-side recording cost scales with command_count, unlike
/// cull's O(1)-per-dispatch recording).
const DRAW_TARGET_TOTAL: u32 = 1000;
const DRAW_REPEATS_CAP: u32 = 32;

const WARMUP_WALL: usize = 5;
const ITERS_WALL: usize = 30;

const QUAD_VERTICES: [[f32; 3]; 4] =
    [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]];
const QUAD_INDICES: [u32; 6] = [0, 1, 2, 0, 2, 3];

/// Mirrors `support::test_context_indirect_first_instance` PLUS
/// `TIMESTAMP_QUERY`/`TIMESTAMP_QUERY_INSIDE_ENCODERS` (T3's requirement for
/// `CommandEncoder::write_timestamp` outside a pass) — this bench is the
/// first `helio-scenedb` harness to need BOTH feature sets at once (T7's
/// indirect-first-instance requirement for nonzero `first_instance`, T3's
/// timestamp requirement for GPU-side pass timing).
fn test_context() -> EngineGpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .expect("no adapter -- GPU bench needs a local GPU");
    let required_features = wgpu::Features::INDIRECT_FIRST_INSTANCE
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    assert!(
        adapter.features().contains(required_features),
        "adapter must support INDIRECT_FIRST_INSTANCE + TIMESTAMP_QUERY(_INSIDE_ENCODERS) -- \
         verified present on this host's RTX 5080/Vulkan adapter (T3/T7 precedent)"
    );
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("helio-scenedb-pass-timing-bench"),
        required_features,
        required_limits: wgpu::Limits::default(),
        ..Default::default()
    }))
    .expect("device");
    EngineGpuContext::new(std::sync::Arc::new(device), std::sync::Arc::new(queue))
}

/// Two-submit GPU timestamp bracket -- copied wholesale from
/// `pulsar_scenedb/benches/gpu_timing.rs`'s `GpuTimer` (T3's own reuse
/// note). See this file's module doc for the one documented deviation
/// (`work()` here calls `queue.submit` itself, since its payload is real
/// dispatch/draw commands, not `queue.write_buffer` pending writes).
struct GpuTimer {
    query_set: wgpu::QuerySet,
    resolve_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
    period_ns: f32,
}

impl GpuTimer {
    fn new(ctx: &EngineGpuContext) -> Self {
        let query_set = ctx.device().create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("pass-timing-query-set"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
        let resolve_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass-timing-resolve"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = ctx.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("pass-timing-staging"),
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let period_ns = ctx.queue().get_timestamp_period();
        Self { query_set, resolve_buf, staging_buf, period_ns }
    }

    fn measure_ns(&self, ctx: &EngineGpuContext, work: impl FnOnce()) -> u64 {
        let mut enc_start = ctx.device().create_command_encoder(&Default::default());
        enc_start.write_timestamp(&self.query_set, 0);
        ctx.queue().submit([enc_start.finish()]);

        work();

        let mut enc_end = ctx.device().create_command_encoder(&Default::default());
        enc_end.write_timestamp(&self.query_set, 1);
        enc_end.resolve_query_set(&self.query_set, 0..2, &self.resolve_buf, 0);
        enc_end.copy_buffer_to_buffer(&self.resolve_buf, 0, &self.staging_buf, 0, 16);
        ctx.queue().submit([enc_end.finish()]);

        let slice = self.staging_buf.slice(..);
        slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
        ctx.device().poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let data = slice.get_mapped_range().expect("mapped range");
        let t0 = u64::from_le_bytes(data[0..8].try_into().unwrap());
        let t1 = u64::from_le_bytes(data[8..16].try_into().unwrap());
        drop(data);
        self.staging_buf.unmap();

        let ticks = t1.wrapping_sub(t0);
        (ticks as f64 * self.period_ns as f64).round() as u64
    }
}

/// mean / p95 (nearest-rank) over a sample set.
fn stats(mut samples: Vec<f64>) -> (f64, f64) {
    samples.sort_by(|a, b| a.total_cmp(b));
    let n = samples.len();
    let mean = samples.iter().copied().sum::<f64>() / n as f64;
    let p95_idx = (((n - 1) as f64) * 0.95).round() as usize;
    (mean, samples[p95_idx])
}

fn draw_repeats(command_count: u32) -> u32 {
    (DRAW_TARGET_TOTAL / command_count.max(1)).clamp(1, DRAW_REPEATS_CAP)
}

/// One fully-built, real (non-synthetic) scene fixture: `n` tokens spread
/// across `ceil(n/1024)` cells, `round(n*visible_fraction)` of them
/// (lowest global rows first) placed inside the frustum, the rest outside
/// it. `output` already holds a VERIFIED real cull result (see module doc)
/// -- `visible_count` is the measured, asserted-correct value, not the
/// requested `visible_fraction` rounded blind.
struct SceneFixture {
    #[allow(dead_code)]
    store: SceneGpuStore,
    view_tokens: ViewTokenBuffers,
    scene_binding: SceneDbBinding,
    cull_pass: CullPass,
    draw_executor: DrawExecutor,
    output: CullOutputBuffers,
    output_bind_group: wgpu::BindGroup,
    draw_output_bind_group: wgpu::BindGroup,
    n: u32,
    visible_count: u32,
}

#[allow(clippy::too_many_arguments)]
fn build_scene(
    ctx: &EngineGpuContext,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    meshes: &MeshRegistry,
    clusters: &ClusterBuffer,
    meshlets: &MeshletBuffer,
    materials: &MaterialRegistry,
    n: u32,
    visible_fraction: f32,
) -> SceneFixture {
    let visible_target = (n as f32 * visible_fraction).round() as u32;
    let num_cells = n.div_ceil(MAX_CELL_ROWS);

    let cfg = SceneGpuConfig {
        classes: vec![RegionClassConfig { capacity: MAX_CELL_ROWS, max_resident_cells: num_cells.max(1) }],
        tombstone_headroom: 8,
        max_cells_metadata: num_cells.max(1),
    };
    let mut store = SceneGpuStore::new(ctx, cfg);

    let mut caps: Vec<u32> = Vec::with_capacity(num_cells as usize);
    let mut remaining = n;
    for _ in 0..num_cells {
        let cap = remaining.min(MAX_CELL_ROWS);
        caps.push(cap);
        remaining -= cap;
    }
    let mut cells: Vec<SpatialCell> = caps.iter().map(|&c| SpatialCell::with_transform(c).unwrap()).collect();
    let ids: Vec<CellId> = cells.iter().map(|c| store.register_cell(c.storage(), 0).unwrap()).collect();
    let bases: Vec<u32> = ids.iter().map(|&id| store.row_region_base(id)).collect();

    // --- Allocate + write every row (untimed setup). ---
    let mut handles: Vec<Vec<Handle>> = Vec::with_capacity(num_cells as usize);
    for (&base, (cell, &cap)) in bases.iter().zip(cells.iter_mut().zip(caps.iter())) {
        let mut hs = Vec::with_capacity(cap as usize);
        for li in 0..cap {
            let global_row = base + li;
            let x = if global_row < visible_target { 0.0 } else { 1000.0 };
            let h = cell
                .alloc(Aabb { min: [x - 1.0, -1.0, -11.0], max: [x + 1.0, 1.0, -9.0] })
                .expect("cell capacity matches the fixed allocation count exactly");
            hs.push(h);
        }
        handles.push(hs);
    }

    let mut frames = FrameDriver::new();
    let sim = frames.begin();
    for (((cell, &id), &base), hs) in cells.iter_mut().zip(ids.iter()).zip(bases.iter()).zip(handles.iter()) {
        for (li, &h) in hs.iter().enumerate() {
            let global_row = base + li as u32;
            let x = if global_row < visible_target { 0.0 } else { 1000.0 };
            let xf = translation([x, 0.0, -10.0]);
            assert!(store.write_transform(id, cell.storage_mut(), h, &xf, &sim));
            assert!(store.write_instance_info(id, cell.storage_mut(), h, InstanceInfo { mesh_index: 0, flags: 0 }, &sim));
        }
    }

    let harvest_phase = sim.end().end();
    let pipeline = HarvestPipeline::new();
    let mut pad = Scratchpad::new();
    let mut staging = HarvestStaging::new();
    // Generous view AABB covering both the visible (x=0) and the culled
    // (x=1000) synthetic positions -- harvest is a broad-phase CPU query,
    // independent of the cull SHADER's own precise frustum test (module doc).
    let view = View::Aabb(Aabb { min: [-2000.0, -2000.0, -2000.0], max: [2000.0, 2000.0, 2000.0] });
    let mut total_harvested = 0u32;
    for (cell, &base) in cells.iter().zip(bases.iter()) {
        total_harvested += pipeline.harvest_cell(cell, base, MeshClass::Traditional, &view, &mut pad, &mut staging, &harvest_phase);
    }
    assert_eq!(total_harvested, n, "sanity: harvest must catch every synthetic row");

    let boundary = harvest_phase.end();
    {
        let mut slots: Vec<CellSlot> = cells.iter_mut().zip(ids.iter()).map(|(cell, &id)| CellSlot { id, cell: cell.storage_mut() }).collect();
        boundary.run(&mut store, &mut slots);
    }

    let mut view_tokens = ViewTokenBuffers::new(ctx, "pass-timing-view", 0);
    view_tokens.upload(ctx, &staging, MeshClass::Traditional);
    assert_eq!(view_tokens.count(), n, "sanity: every token uploaded");

    let scene_binding = SceneDbBinding::new(device, &store, meshes, clusters, meshlets, materials);
    let cull_pass = CullPass::new(device, &scene_binding.cull_layout);
    let draw_executor = DrawExecutor::new(device, &scene_binding.cull_layout);
    draw_executor.write_uniforms(queue, &DrawUniforms { view_proj: view_proj_90deg() });

    let output = CullOutputBuffers::new(device, "pass-timing-cull-output", n.max(1));
    output.clear_counters(queue);
    let output_bind_group = cull_pass.build_output_bind_group(device, &view_tokens, &output);
    let uniforms = CullUniforms {
        view_proj: view_proj_90deg(),
        planes: frustum_planes_90deg(),
        count: view_tokens.count(),
        mesh_count: meshes.len(),
        capacity: output.capacity(),
        reserved: 0,
    };
    cull_pass.write_uniforms(queue, &uniforms);

    // --- ONE untimed real cull dispatch: verifies the fixture's visible
    // fraction is exactly what was asked for (house law: self-verifying
    // fixture guard), and leaves `output` populated with a REAL cull result
    // every subsequent draw-timing / baseline measurement reads from. ---
    {
        let mut encoder = device.create_command_encoder(&Default::default());
        cull_pass.record(&mut encoder, &scene_binding.cull_bind_group, &output_bind_group, view_tokens.count());
        queue.submit([encoder.finish()]);
    }
    let hdr = readback(ctx, output.buffer(), CullOutputBuffers::HEADER_BYTES);
    let visible_count = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
    let stale_drops = u32::from_le_bytes(hdr[4..8].try_into().unwrap());
    let oob_drops = u32::from_le_bytes(hdr[8..12].try_into().unwrap());
    let frustum_drops = u32::from_le_bytes(hdr[12..16].try_into().unwrap());
    assert_eq!(stale_drops, 0, "fixture has no stale rows");
    assert_eq!(oob_drops, 0, "fixture has no OOB mesh_index rows");
    assert_eq!(visible_count, visible_target, "guard: the synthetic visible-fraction fixture must produce EXACTLY the intended visible count");
    assert_eq!(frustum_drops, n - visible_target, "guard: every non-visible row must be frustum-culled (no other drop category in this fixture)");

    let draw_output_bind_group = draw_executor.build_output_bind_group(device, &output);

    SceneFixture {
        store,
        view_tokens,
        scene_binding,
        cull_pass,
        draw_executor,
        output,
        output_bind_group,
        draw_output_bind_group,
        n,
        visible_count,
    }
}

struct CellResult {
    n: u32,
    frac: f32,
    visible: u32,
    cull_mean: f64,
    cull_p95: f64,
    draw_mean: f64,
    draw_p95: f64,
    baseline_draw_mean: Option<f64>,
    baseline_draw_p95: Option<f64>,
}

fn main() {
    let ctx = test_context();
    let device: &wgpu::Device = ctx.device().as_ref();
    let queue = ctx.queue();
    let timer = GpuTimer::new(&ctx);
    println!("timestamp_period_ns_per_tick = {:.4}", timer.period_ns);

    // --- Shared geometry + asset registries (mesh0 reused by every
    // fixture -- MeshDbBinding is built per-store, but the mesh/cluster/
    // meshlet/material registries themselves are store-independent, so one
    // shared set avoids rebuilding identical geometry 9 times). ---
    let mut arena = GeometryArena::new(&ctx, 4096, 4096);
    let vbytes: &[u8] = bytemuck::cast_slice(&QUAD_VERTICES);
    let ibytes: &[u8] = bytemuck::cast_slice(&QUAD_INDICES);
    let voff = arena.upload_vertices(queue, vbytes).unwrap();
    let ioff = arena.upload_indices(queue, ibytes).unwrap();
    let base_vertex = (voff / 12) as i32;
    let first_index = ioff / 4;
    let mut meshes = MeshRegistry::new(&ctx, 4);
    let mesh0 = meshes
        .register(queue, mesh([0.0, 0.0, 0.0], [0.5, 0.5, 0.5], QUAD_INDICES.len() as u32, first_index, base_vertex))
        .unwrap();
    assert_eq!(mesh0, 0);
    let clusters = ClusterBuffer::new(&ctx, 1);
    let meshlets = MeshletBuffer::new(&ctx, 1);
    let materials = MaterialRegistry::new(&ctx, 1);

    // --- Noise floor: an EMPTY bracket (zero GPU commands between the two
    // timestamp submits) -- the honest floor this file's payload-visibility
    // gate compares against (T3's dead-bracket lesson: derive the floor
    // independently of the trend being validated). ---
    let mut floor_samples = Vec::with_capacity(16);
    for _ in 0..16 {
        floor_samples.push(timer.measure_ns(&ctx, || {}) as f64);
    }
    let (floor_mean, floor_p95) = stats(floor_samples);
    println!("noise_floor_ns (empty two-submit bracket, 16 samples) = mean {floor_mean:.1} p95 {floor_p95:.1}");

    // ==================================================================
    // Deliverables 1, 2, 4: cull GPU time / draw GPU time / no-cull
    // baseline, at N x visible_fraction.
    // ==================================================================
    let mut results: Vec<CellResult> = Vec::new();
    let mut item3_fixture: Option<SceneFixture> = None;

    for &n in &NS {
        for &frac in &FRACS {
            let fixture = build_scene(&ctx, device, queue, &meshes, &clusters, &meshlets, &materials, n, frac);

            // --- (1) Cull dispatch GPU time: REPEATS_CULL dispatches of
            // the SAME N-thread cull, recorded into one encoder, one
            // bracket. Correctness of the output past the first dispatch
            // doesn't matter here (this measures dispatch cost, not a
            // second verification) -- overflow past `capacity` is silently
            // guarded by the shader itself (S14.2), never UB. ---
            let mut cull_samples = Vec::with_capacity(ITERS_PASS);
            for it in 0..(WARMUP_PASS + ITERS_PASS) {
                let ns = timer.measure_ns(&ctx, || {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    for _ in 0..REPEATS_CULL {
                        fixture.cull_pass.record(&mut encoder, &fixture.scene_binding.cull_bind_group, &fixture.output_bind_group, fixture.view_tokens.count());
                    }
                    queue.submit([encoder.finish()]);
                });
                if it >= WARMUP_PASS {
                    cull_samples.push(ns as f64 / REPEATS_CULL as f64);
                }
            }
            let (cull_mean, cull_p95) = stats(cull_samples);

            // --- (2) Indirect draw GPU time for the ALREADY-VERIFIED
            // visible set (fixture.output holds a real, checked cull
            // result from build_scene's setup dispatch). ---
            let command_count = fixture.visible_count;
            let draw_reps = draw_repeats(command_count.max(1));
            // ONE throwaway target, created OUTSIDE every timed bracket and
            // reused across all repeats/samples -- texture (re)allocation is
            // not part of "draw pass GPU time" and must not bleed into the
            // measurement (a per-repeat allocation would have, see the fix
            // note in this file's history / M3-b T9 review).
            let target = throwaway_target(device);
            let mut draw_samples = Vec::with_capacity(ITERS_PASS);
            for it in 0..(WARMUP_PASS + ITERS_PASS) {
                let ns = timer.measure_ns(&ctx, || {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    for _ in 0..draw_reps {
                        fixture.draw_executor.record(
                            &mut encoder,
                            &target,
                            &fixture.scene_binding.cull_bind_group,
                            &fixture.draw_output_bind_group,
                            &fixture.output,
                            arena.vertex_buffer(),
                            arena.index_buffer(),
                            command_count,
                        );
                    }
                    queue.submit([encoder.finish()]);
                });
                if it >= WARMUP_PASS {
                    draw_samples.push(ns as f64 / draw_reps as f64);
                }
            }
            let (draw_mean, draw_p95) = stats(draw_samples);

            // --- (4) No-cull baseline, only at frac==1.0 (visible_count ==
            // n there, i.e. fixture.output ALREADY holds a valid command
            // for every one of the n rows -- "draw everything" needs no
            // separate buffer). Skips the cull dispatch entirely: times
            // ONLY the draw pass issuing n commands. ---
            let (baseline_draw_mean, baseline_draw_p95) = if (frac - 1.0).abs() < 1e-6 {
                let reps = draw_repeats(n.max(1));
                let baseline_target = throwaway_target(device);
                let mut samples = Vec::with_capacity(ITERS_PASS);
                for it in 0..(WARMUP_PASS + ITERS_PASS) {
                    let ns = timer.measure_ns(&ctx, || {
                        let mut encoder = device.create_command_encoder(&Default::default());
                        for _ in 0..reps {
                            fixture.draw_executor.record(
                                &mut encoder,
                                &baseline_target,
                                &fixture.scene_binding.cull_bind_group,
                                &fixture.draw_output_bind_group,
                                &fixture.output,
                                arena.vertex_buffer(),
                                arena.index_buffer(),
                                n,
                            );
                        }
                        queue.submit([encoder.finish()]);
                    });
                    if it >= WARMUP_PASS {
                        samples.push(ns as f64 / reps as f64);
                    }
                }
                let (m, p) = stats(samples);
                (Some(m), Some(p))
            } else {
                (None, None)
            };

            println!(
                "N={n}\tfrac={frac}\tvisible={}\tcull_ns(mean/p95)={cull_mean:.1}/{cull_p95:.1}\tdraw_ns(mean/p95)={draw_mean:.1}/{draw_p95:.1}\tbaseline_draw_ns={baseline_draw_mean:?}/{baseline_draw_p95:?}",
                fixture.visible_count
            );

            results.push(CellResult {
                n,
                frac,
                visible: fixture.visible_count,
                cull_mean,
                cull_p95,
                draw_mean,
                draw_p95,
                baseline_draw_mean,
                baseline_draw_p95,
            });

            // Stash the exact scale the §14.2 comparison (item 3) below
            // wants (N=10,000 / 10% visible -- "many culled draws for (b)
            // to waste"), instead of rebuilding it a second time.
            if n == 10_000 && (frac - 0.1).abs() < 1e-6 {
                item3_fixture = Some(fixture);
            }
        }
    }

    // ==================================================================
    // Deliverable 3: THE §14.2 DECISION.
    // ==================================================================
    let fixture = item3_fixture.expect("N=10000/frac=0.1 cell must have run in the matrix above");
    let device2: &wgpu::Device = ctx.device().as_ref();

    // --- Standalone readback-latency measurement (the explicit §9(a)
    // ask): map_async + poll round trip on the cull pass's atomic counter
    // buffer, exactly as tests/draw_transform_sweep.rs's frame loop already
    // pays it every frame. Two variants: header-only (16 B -- the minimum a
    // real §14.2 clamp implementation needs) and full-buffer (what T7/T8's
    // existing `readback()` helper actually reads today, header+records). ---
    let mut header_only_ns = Vec::with_capacity(ITERS_WALL);
    let mut full_buffer_ns = Vec::with_capacity(ITERS_WALL);
    for it in 0..(WARMUP_WALL + ITERS_WALL) {
        {
            let mut encoder = device2.create_command_encoder(&Default::default());
            fixture.cull_pass.record(&mut encoder, &fixture.scene_binding.cull_bind_group, &fixture.output_bind_group, fixture.view_tokens.count());
            queue.submit([encoder.finish()]);
        }
        let t0 = Instant::now();
        let _hdr = readback(&ctx, fixture.output.buffer(), CullOutputBuffers::HEADER_BYTES);
        let t1 = Instant::now();
        if it >= WARMUP_WALL {
            header_only_ns.push((t1 - t0).as_nanos() as f64);
        }
    }
    for it in 0..(WARMUP_WALL + ITERS_WALL) {
        {
            let mut encoder = device2.create_command_encoder(&Default::default());
            fixture.cull_pass.record(&mut encoder, &fixture.scene_binding.cull_bind_group, &fixture.output_bind_group, fixture.view_tokens.count());
            queue.submit([encoder.finish()]);
        }
        let t0 = Instant::now();
        let _full = readback(&ctx, fixture.output.buffer(), fixture.output.byte_size());
        let t1 = Instant::now();
        if it >= WARMUP_WALL {
            full_buffer_ns.push((t1 - t0).as_nanos() as f64);
        }
    }
    let (header_mean, header_p95) = stats(header_only_ns);
    let (full_mean, full_p95) = stats(full_buffer_ns);
    println!();
    println!("readback_latency_ns header_only(mean/p95) = {header_mean:.1}/{header_p95:.1}");
    println!("readback_latency_ns full_buffer(mean/p95, matches T7/T8's existing readback()) = {full_mean:.1}/{full_p95:.1}");

    // --- Strategy (a): readback-then-clamp, exactly what T7/T8 do today.
    // Wall-clock per frame: cull submit -> map_async+poll STALL -> clamp ->
    // draw submit (not polled -- matches draw_transform_sweep.rs, which
    // never waits on the draw's own completion either). ---
    let a_target = throwaway_target(device2);
    let mut a_frame_ns = Vec::with_capacity(ITERS_WALL);
    let mut a_stall_ns = Vec::with_capacity(ITERS_WALL);
    for it in 0..(WARMUP_WALL + ITERS_WALL) {
        fixture.output.clear_counters(queue);
        let t_frame_start = Instant::now();
        {
            let mut encoder = device2.create_command_encoder(&Default::default());
            fixture.cull_pass.record(&mut encoder, &fixture.scene_binding.cull_bind_group, &fixture.output_bind_group, fixture.view_tokens.count());
            queue.submit([encoder.finish()]);
        }
        let t_stall_start = Instant::now();
        let hdr = readback(&ctx, fixture.output.buffer(), CullOutputBuffers::HEADER_BYTES);
        let t_stall_end = Instant::now();
        let visible = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        let command_count = visible.min(fixture.output.capacity());
        {
            let mut encoder = device2.create_command_encoder(&Default::default());
            fixture.draw_executor.record(
                &mut encoder,
                &a_target,
                &fixture.scene_binding.cull_bind_group,
                &fixture.draw_output_bind_group,
                &fixture.output,
                arena.vertex_buffer(),
                arena.index_buffer(),
                command_count,
            );
            queue.submit([encoder.finish()]);
        }
        let t_frame_end = Instant::now();
        if it >= WARMUP_WALL {
            a_frame_ns.push((t_frame_end - t_frame_start).as_nanos() as f64);
            a_stall_ns.push((t_stall_end - t_stall_start).as_nanos() as f64);
        }
    }
    let (a_frame_mean, a_frame_p95) = stats(a_frame_ns);
    let (a_stall_mean, a_stall_p95) = stats(a_stall_ns);

    // --- Strategy (b): conservative max-count. A SEPARATE output buffer,
    // capacity == n (10,000), whose tail is pre-filled ONCE (untimed) with
    // harmless `instance_count=0` records at every slot; the real per-frame
    // cull dispatch then overwrites slots [0, visible_count) with real
    // commands every frame -- overwriting the SAME compacted range each
    // time since this fixture's visible set never changes frame to frame,
    // so the pre-fill at slots >= visible_count is never touched by the
    // shader and never needs re-priming. NO readback of the counter at
    // all; the draw pass unconditionally issues `capacity` (== n) indirect
    // calls every frame. ---
    let mesh0_meta = (QUAD_INDICES.len() as u32, first_index, base_vertex);
    let output_b = CullOutputBuffers::new(device2, "pass-timing-conservative-output", fixture.n);
    fill_zero_instance_records(queue, &output_b, fixture.n, mesh0_meta);
    let output_bind_group_b = fixture.cull_pass.build_output_bind_group(device2, &fixture.view_tokens, &output_b);
    let draw_output_bind_group_b = fixture.draw_executor.build_output_bind_group(device2, &output_b);
    let uniforms_b = CullUniforms {
        view_proj: view_proj_90deg(),
        planes: frustum_planes_90deg(),
        count: fixture.view_tokens.count(),
        mesh_count: meshes.len(),
        capacity: output_b.capacity(),
        reserved: 0,
    };
    fixture.cull_pass.write_uniforms(queue, &uniforms_b);

    let b_target = throwaway_target(device2);
    let mut b_frame_ns = Vec::with_capacity(ITERS_WALL);
    for it in 0..(WARMUP_WALL + ITERS_WALL) {
        output_b.clear_counters(queue); // enqueue-only write, no stall
        let t_frame_start = Instant::now();
        {
            let mut encoder = device2.create_command_encoder(&Default::default());
            fixture.cull_pass.record(&mut encoder, &fixture.scene_binding.cull_bind_group, &output_bind_group_b, fixture.view_tokens.count());
            queue.submit([encoder.finish()]);
        }
        {
            let mut encoder = device2.create_command_encoder(&Default::default());
            fixture.draw_executor.record(
                &mut encoder,
                &b_target,
                &fixture.scene_binding.cull_bind_group,
                &draw_output_bind_group_b,
                &output_b,
                arena.vertex_buffer(),
                arena.index_buffer(),
                output_b.capacity(), // conservative: ALWAYS capacity draws
            );
            queue.submit([encoder.finish()]);
        }
        let t_frame_end = Instant::now();
        if it >= WARMUP_WALL {
            b_frame_ns.push((t_frame_end - t_frame_start).as_nanos() as f64);
        }
    }
    let (b_frame_mean, b_frame_p95) = stats(b_frame_ns);

    println!();
    println!("=== S14.2 DECISION (N=10000, 10% visible, visible_count={}) ===", fixture.visible_count);
    println!("(a) readback-then-clamp: frame_ns(mean/p95) = {a_frame_mean:.1}/{a_frame_p95:.1}  stall_ns(mean/p95) = {a_stall_mean:.1}/{a_stall_p95:.1}");
    println!("(b) conservative max-count: frame_ns(mean/p95) = {b_frame_mean:.1}/{b_frame_p95:.1}  wasted_draws = {}", fixture.n - fixture.visible_count);
    let winner = if a_frame_mean <= b_frame_mean { "(a) readback-then-clamp" } else { "(b) conservative max-count" };
    println!("winner at this scale: {winner} (delta {:.1} ns, {:.2}x)", (a_frame_mean - b_frame_mean).abs(), a_frame_mean.max(b_frame_mean) / a_frame_mean.min(b_frame_mean));

    // ==================================================================
    // Report: full matrix.
    // ==================================================================
    println!();
    println!("=== Full matrix (N x visible_fraction) ===");
    println!("N\tfrac\tvisible\tcull_ns(mean/p95)\tdraw_ns(mean/p95)\tbaseline_draw_ns(mean/p95)");
    for r in &results {
        println!(
            "{}\t{}\t{}\t{:.1}/{:.1}\t{:.1}/{:.1}\t{}",
            r.n,
            r.frac,
            r.visible,
            r.cull_mean,
            r.cull_p95,
            r.draw_mean,
            r.draw_p95,
            match (r.baseline_draw_mean, r.baseline_draw_p95) {
                (Some(m), Some(p)) => format!("{m:.1}/{p:.1}"),
                _ => "n/a".to_string(),
            }
        );
    }

    // ==================================================================
    // Honesty gates (house law): fixed slack, provably-failable, payload
    // clearly above the independently-measured floor.
    // ==================================================================
    // 5 microseconds, NOT derived from any measured noise-floor variable in
    // this file (T3's dead-bracket lesson: a floor-derived slack scales
    // with the very noise it should be catching) -- chosen from the
    // ABSOLUTE SCALE of what is being compared instead: amortized cull-
    // dispatch means on this host land in a 9.5-14 us band across all of
    // 1k/10k/100k tokens (measured finding, see below), so a slack an order
    // of magnitude tighter (500 ns) would fail on ordinary GPU-scheduling
    // jitter alone, and a slack an order of magnitude looser (50 us) would
    // stop being able to catch a real inversion. 5 us sits between those.
    const TREND_SLACK_NS: f64 = 5_000.0;

    // Payload visibility: the largest N's cull cost, un-amortized back to a
    // RAW per-bracket total (`cull_mean * REPEATS_CULL` -- the floor above
    // is a single un-amplified empty bracket, so comparing it against an
    // AMORTIZED per-dispatch mean would be an apples-to-oranges unit
    // mismatch; this is exactly why amplification is needed in the first
    // place -- a single un-amplified cull dispatch's raw bracket total,
    // ~floor + payload, would sit only modestly above the ~47-80us noise
    // floor measured on this host, indistinguishable from noise without
    // REPEATS_CULL's amortization) must be clearly above the empty-bracket
    // floor -- a dead/misplaced bracket reads ~flat, this catches it (T3's
    // exact gate, reused).
    let cull_100k_100 = results.iter().find(|r| r.n == 100_000 && (r.frac - 1.0).abs() < 1e-6).unwrap();
    let cull_100k_100_raw = cull_100k_100.cull_mean * REPEATS_CULL as f64;
    assert!(
        cull_100k_100_raw > floor_mean * 1.5,
        "payload invisible: N=100k cull RAW bracket total {cull_100k_100_raw:.1} ns (mean {:.1} ns x {REPEATS_CULL} reps) \
         not clearly above the empty-bracket floor {floor_mean:.1} ns -- dead or misplaced bracket",
        cull_100k_100.cull_mean
    );

    // Cull time rises with N (physics: more threads dispatched -> more GPU
    // work), holding frac fixed at 100% for an apples-to-apples N-only
    // comparison (dispatch_count == n regardless of frac -- see module doc).
    //
    // HONEST FINDING (not a harness defect): on this host, cull dispatch
    // cost across 1k-100k tokens is dominated by a roughly CONSTANT
    // per-dispatch-call floor (~9.5-14 us amortized, every N -- see the
    // printed matrix), not by thread count. Two independent runs both
    // showed N=1,000's amortized mean sitting AT OR ABOVE N=10,000's (never
    // the reverse of what "more work" would predict) -- i.e. the true
    // per-thread compute cost the cull term list pays (generation check +
    // bounds check + AABB transform + up to 8-corner near-clip test + 6
    // frustum planes) is small enough, at these token counts, that it does
    // not clear this host's own dispatch-to-dispatch scheduling jitter
    // until N reaches ~100k. This is the SAME shape as the perf-validation
    // report's T7 finding (§3.3: "compute/overhead-bound, not memory-bound,
    // at every scale tested") applied to a dispatch-call boundary instead
    // of a memory-bandwidth one. TREND_SLACK_NS is sized to this finding's
    // actual scale (see its own doc comment), not to paper over a broken
    // gate.
    let cull_1k_100 = results.iter().find(|r| r.n == 1_000 && (r.frac - 1.0).abs() < 1e-6).unwrap();
    let cull_10k_100 = results.iter().find(|r| r.n == 10_000 && (r.frac - 1.0).abs() < 1e-6).unwrap();
    assert!(
        cull_1k_100.cull_mean <= cull_10k_100.cull_mean + TREND_SLACK_NS,
        "monotonicity broken: cull(N=1k) {:.1} ns > cull(N=10k) {:.1} ns + {TREND_SLACK_NS} ns",
        cull_1k_100.cull_mean,
        cull_10k_100.cull_mean
    );
    assert!(
        cull_10k_100.cull_mean <= cull_100k_100.cull_mean + TREND_SLACK_NS,
        "monotonicity broken: cull(N=10k) {:.1} ns > cull(N=100k) {:.1} ns + {TREND_SLACK_NS} ns",
        cull_10k_100.cull_mean,
        cull_100k_100.cull_mean
    );

    // Draw time rises with visible (command) count (physics: more indirect
    // draws issued -> more GPU + recording work). Compare the smallest
    // visible count in the matrix (1k/10% -> V=100) against the largest
    // (100k/100% -> V=100000).
    let smallest_v = results.iter().min_by_key(|r| r.visible).unwrap();
    let largest_v = results.iter().max_by_key(|r| r.visible).unwrap();
    assert!(
        smallest_v.draw_mean <= largest_v.draw_mean + TREND_SLACK_NS,
        "monotonicity broken: draw(V={}) {:.1} ns > draw(V={}) {:.1} ns + {TREND_SLACK_NS} ns",
        smallest_v.visible,
        smallest_v.draw_mean,
        largest_v.visible,
        largest_v.draw_mean
    );

    // The §14.2 stall must be REAL (provably capable of failing: a dead/
    // no-op readback would read ~0, indistinguishable from a broken gate --
    // this asserts the measured stall clears the same submission noise
    // floor used above, i.e. it is doing real driver/kernel work, not
    // returning instantly).
    assert!(
        a_stall_mean > floor_mean * 0.5,
        "the S14.2 readback stall ({a_stall_mean:.1} ns) is suspiciously close to the empty-bracket \
         floor ({floor_mean:.1} ns) -- either this host's map_async is unrealistically fast, or the \
         readback call is a dead no-op; investigate before trusting the (a)-vs-(b) decision above"
    );

    println!();
    println!("self-check OK: floor visible, cull/draw monotonic in N/V (fixed {TREND_SLACK_NS} ns slack), S14.2 stall is real work");
}

/// A throwaway 64x64 Rgba8Unorm render target -- draw timing doesn't care
/// about pixel content, only GPU pass cost, so a fresh target per bracket
/// avoids any accidental cross-iteration state (load/clear cost is part of
/// what's being measured, matching `OffscreenTarget`'s own shape without
/// pulling in its `read_pixels` readback machinery this file never uses).
fn throwaway_target(device: &wgpu::Device) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("pass-timing-throwaway-target"),
        size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

/// Pre-fills EVERY record slot `0..count` with a valid-but-harmless
/// `instance_count=0` indirect-draw command (the §14.2 "conservative
/// max-count" design's steady-state shape for the slots a compacting cull
/// dispatch never touches -- see this file's module doc + the §14.2
/// section's own comment for why this is a fair, honest approximation and
/// not a correctness shortcut: the per-token visibility COMPUTE cost is
/// identical to strategy (a)'s, since both run the exact same shipped
/// `CULL_WGSL`; only the OUTPUT strategy and the CPU synchronization model
/// differ, which is what §14.2 asks to decide between).
fn fill_zero_instance_records(queue: &wgpu::Queue, output: &CullOutputBuffers, count: u32, mesh0: (u32, u32, i32)) {
    let (index_count, first_index, base_vertex) = mesh0;
    let records: Vec<CullRecord> = (0..count)
        .map(|slot| CullRecord {
            index_count,
            instance_count: 0,
            first_index,
            base_vertex,
            first_instance: slot,
            row: 0,
            flags: 0,
            reserved: 0,
        })
        .collect();
    queue.write_buffer(output.buffer(), CullOutputBuffers::HEADER_BYTES, bytemuck::cast_slice(&records));
}
