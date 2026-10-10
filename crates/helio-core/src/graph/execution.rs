use crate::graph::executor::{format_bpp, format_name};
use crate::graph::resource::GraphTexturePool;
use crate::graph::{PipelineFormatCache, PipelineFormatSet, PipelineRegistry};
use crate::{PassContext, PrepareContext, Profiler, RenderFrameStorage, RenderPass, Result, SceneInput};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex, OnceLock};

use super::recording_cache::{self, UnitBuffer, UnitCache};
use super::resource_lifetime::ResourceLifetime;
use crate::cmd_ir::{Cmd, RenderPassDesc, Stream};
use crate::context::RecordedStreams;
use super::scheduling::{compute_parallel_layers, CachedPass, PrePassAction};
use super::{DebugPassInfo, DebugResourceInfo, FrameDebugData};

/// One command buffer of a frame's submission, in submission order.
enum FrameBuffer {
    Once(wgpu::CommandBuffer),
    /// A unit's cached recording (see `recording_cache`): its first pass,
    /// the variant, and which of its two streams.
    Cached { pass: usize, variant: usize, compute: bool },
    /// A reusable command buffer encoded this frame but not cached, kept in
    /// the frame's `frame_reusable` list.
    Uncached(usize),
}

/// A recording-cache unit being recorded this frame (see `recording_cache`).
struct OpenUnit {
    /// Its first pass, which keys its cache.
    first: usize,
    /// One past its last pass.
    end: usize,
    compute: Stream,
    graphics: Stream,
    /// Whether its render pass is open on the graphics stream.
    render_pass_open: bool,
    /// The fused chain's GPU timing span, closed after its render pass.
    chain_span: Option<&'static str>,
}

/// `desc` as the graph opens it: with its store ops and the XR multiview mask.
fn captured_render_pass(
    desc: &wgpu::RenderPassDescriptor<'_>,
    store_ops: &[Option<wgpu::StoreOp>],
    xr_active: bool,
) -> Box<RenderPassDesc> {
    let mut captured = RenderPassDesc::capture(desc);
    for (attachment, store) in captured.color.iter_mut().zip(store_ops) {
        if let (Some(attachment), Some(store)) = (attachment.as_mut(), store) {
            attachment.ops.store = *store;
        }
    }
    if xr_active {
        captured.multiview_mask = Some(std::num::NonZeroU32::new(0b11).unwrap());
    }
    Box::new(captured)
}

/// How long `CommandEncoder::finish` took for one run of passes, recorded
/// when [`RenderGraph::set_finish_breakdown`] is on.
///
/// wgpu does its validation and encoding in `finish`, so its cost follows the
/// number of commands recorded; passes draw on raw `wgpu` passes the graph
/// cannot count, so this measures that cost per pass directly instead
/// (Pulsar-Native#813). Passes fused into one render-pass chain cannot be
/// split, so they share a segment.
#[derive(Clone, Debug)]
pub struct FinishSegment {
    /// The passes recorded into this segment, in graph order.
    pub passes: Vec<&'static str>,
    pub compute: std::time::Duration,
    pub graphics: std::time::Duration,
}

/// Environment variable that turns [`FinishSegment`] recording on for every
/// graph built, including ones rebuilt after a resize or settings change.
const FINISH_BREAKDOWN_ENV: &str = "HELIO_FINISH_BREAKDOWN";

/// Recording time (summed pass `execute` spans) after which the graphics
/// encoder is cut into a new segment. wgpu's `finish` costs several times
/// what recording the same commands did, so this yields segments of roughly
/// half a millisecond of finishing each.
const DEFAULT_FINISH_SEGMENT_BUDGET: std::time::Duration = std::time::Duration::from_micros(100);

/// Upper bound on resident encoder-finish threads.
#[cfg(not(target_arch = "wasm32"))]
const MAX_FINISH_WORKERS: usize = 4;

/// Reply index a finished compute encoder is tagged with; graphics segments
/// use their position in submission order.
#[cfg(not(target_arch = "wasm32"))]
const COMPUTE_SEGMENT: usize = usize::MAX;

#[cfg(not(target_arch = "wasm32"))]
type FinishReply = (usize, std::thread::Result<wgpu::CommandBuffer>);

#[cfg(not(target_arch = "wasm32"))]
struct FinishJob {
    index: usize,
    encoder: wgpu::CommandEncoder,
    /// Per-frame channel: a frame that returns early (a pass error) drops its
    /// receiver, so its late results can never reach the next frame.
    reply: mpsc::Sender<FinishReply>,
}

/// Resident threads that finish command-encoder segments while the render
/// thread keeps recording.
///
/// `CommandEncoder::finish` is where wgpu validates and encodes a recorded
/// stream, and it was most of Helio's CPU frame (Pulsar-Native#813). Nearly
/// all of it is the graphics encoder, so the graph cuts that encoder into
/// segments at pass boundaries and hands each one here as soon as it is cut:
/// segments finish in parallel with each other and with the recording of
/// later passes, and the render thread only finishes the last one. Persistent
/// threads rather than per-frame spawns, so a flamegraph shows a fixed set
/// of thread ids.
#[cfg(not(target_arch = "wasm32"))]
struct EncoderFinishPool {
    jobs: Option<mpsc::Sender<FinishJob>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

#[cfg(not(target_arch = "wasm32"))]
impl EncoderFinishPool {
    fn new() -> Self {
        // Leave a core for the render thread, which records meanwhile.
        let count = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1))
            .unwrap_or(2)
            .clamp(1, MAX_FINISH_WORKERS);
        let (jobs_tx, jobs_rx) = mpsc::channel::<FinishJob>();
        let jobs_rx = Arc::new(Mutex::new(jobs_rx));
        let workers = (0..count)
            .map(|i| {
                let jobs_rx = Arc::clone(&jobs_rx);
                std::thread::Builder::new()
                    .name(format!("helio-encoder-finish-{i}"))
                    .spawn(move || loop {
                        let job = match jobs_rx.lock() {
                            Ok(rx) => rx.recv(),
                            Err(_) => break,
                        };
                        let Ok(job) = job else { break };
                        // A validation error panics through wgpu's default
                        // error handler; carry it back so it surfaces on the
                        // render thread as it did when finish ran there.
                        let finished =
                            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                profiling::profile_scope!("RenderGraph: encoder.finish (segment)");
                                job.encoder.finish()
                            }));
                        let _ = job.reply.send((job.index, finished));
                    })
                    .expect("failed to spawn helio encoder-finish thread")
            })
            .collect();
        Self {
            jobs: Some(jobs_tx),
            workers,
        }
    }

    /// Queue `encoder` for finishing; gives it back if the pool is gone.
    fn submit(
        &self,
        index: usize,
        encoder: wgpu::CommandEncoder,
        reply: &mpsc::Sender<FinishReply>,
    ) -> std::result::Result<(), wgpu::CommandEncoder> {
        let job = FinishJob {
            index,
            encoder,
            reply: reply.clone(),
        };
        match self.jobs.as_ref() {
            Some(jobs) => jobs.send(job).map_err(|e| e.0.encoder),
            None => Err(job.encoder),
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for EncoderFinishPool {
    fn drop(&mut self) {
        self.jobs = None;
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

pub struct RenderGraph {
    pub(crate) passes: Vec<Box<dyn RenderPass>>,
    pass_index_map: HashMap<TypeId, usize>,
    profiler: Profiler,
    pub(crate) pool: GraphTexturePool,
    /// Dynamic-rendering pipeline cache shared by every pass's `PassContext`
    /// this frame, keyed by runtime attachment formats. See
    /// [`PipelineFormatCache`] for why a `RefCell` here doesn't violate the
    /// "zero locks in the render path" guarantee.
    pub(crate) pipeline_cache: PipelineFormatCache,
    /// Explicit host-owned formats reachable by this graph. An empty list
    /// means only the formats declared by each recipe are prepared.
    pipeline_formats: Vec<PipelineFormatSet>,
    pub(crate) pipeline_registries: Vec<PipelineRegistry>,
    pub(crate) reflected_pipelines: Vec<Option<crate::shader::ReflectedPipeline>>,
    pub(crate) resources: HashMap<String, ResourceLifetime>,
    /// `write_group` membership in declaration order: `(owning_pass_index,
    /// group_name, member_names_in_declared_order)`. A plain `Vec`, not a
    /// hash map, so member order is deterministic regardless of `resources`'
    /// iteration order. Rebuilt every `collect_declarations()`. See
    /// `docs/helio_3_0_spec.md` §5.
    pub(crate) resource_groups: Vec<(usize, &'static str, Vec<&'static str>)>,
    /// Owned storage for dynamically declared route names. The registry API
    /// currently uses static keys, so graph-owned names must live as long as
    /// the graph rather than being leaked on every rebuild.
    pub(crate) route_names: Vec<Box<str>>,
    pub(crate) pre_pass_actions: Vec<Vec<PrePassAction>>,
    pub(crate) device: std::sync::Arc<wgpu::Device>,
    pub(crate) internal_w: u32,
    pub(crate) internal_h: u32,
    pub(crate) output_w: u32,
    pub(crate) output_h: u32,
    delta_time: f32,
    /// The frame clock and its advance this frame (`PrepareContext::time`).
    frame_time: f32,
    frame_time_delta: f32,
    owns_device: bool,
    gpu_render_bundles: Vec<Option<wgpu::RenderBundle>>,
    resources_allocated: bool,
    pub(crate) subpass_chains: Vec<std::ops::Range<usize>>,
    pub(crate) parallel_layers: Vec<Vec<usize>>,
    /// Finishes graphics segments while recording continues, built on the
    /// first frame. See [`EncoderFinishPool`].
    #[cfg(not(target_arch = "wasm32"))]
    encoder_finish_pool: OnceLock<Arc<EncoderFinishPool>>,
    /// Recording time after which the graphics encoder is cut into a new
    /// segment for the finish pool.
    finish_segment_budget: std::time::Duration,
    /// Per pass, its reflected bind groups from the last frame and what they
    /// bind, so an unchanged group is reused instead of recreated. A group's
    /// key includes its layout, so a rebuilt pipeline invalidates it.
    reflected_group_cache: Vec<Vec<Option<crate::shader::CachedReflectedGroup>>>,
    /// Whether to split the encoders at every pass boundary and time each
    /// segment's finish on the render thread instead. Diagnostic only: many
    /// small command buffers cost more in total than a few large ones.
    finish_breakdown_enabled: bool,
    /// The last frame's segments while `finish_breakdown_enabled`.
    finish_breakdown: Vec<FinishSegment>,
    /// Per pass, its recording-cache state (see `recording_cache`). Reset
    /// whenever the graph is rebuilt.
    unit_caches: Vec<UnitCache>,
    recording_cache_enabled: bool,
    /// Set once the backend turned out unable to resubmit command buffers.
    recording_cache_unsupported: bool,
    recording_cache_active: bool,
    recording_cache_inactive_reason: Option<&'static str>,
    /// Passes the cache recorded directly last frame, and why.
    recording_cache_direct: Vec<(&'static str, &'static str)>,
    chain_membership: Vec<bool>,
    /// Previous frame's chain membership, used to detect which passes changed
    /// so only their bundles (and everything after) need rebuilding.
    prev_chain_membership: Vec<bool>,
    /// Generation counter incremented whenever chain membership changes.
    chain_generation: u64,
    last_bundle_chain_gen: Vec<u64>,
    locked: bool,
    pub(crate) xr_active: bool,
    pass_cache: Vec<Option<CachedPass>>,
    frame_count: u64,
    /// Optional resources passes declared for the current frame; see
    /// [`crate::FrameDemands`].
    frame_demands: crate::FrameDemands,
    /// Set by [`set_render_size`](Self::set_render_size) and consumed only
    /// after the first successful frame at the new size. Passes use this
    /// one-frame pulse through [`PrepareContext::resize`] to rebuild resources
    /// that depend on graph-owned texture dimensions.
    resize_pending: bool,
    /// Opaque storage for cross-crate data (e.g. a GraphRebuilder).
    /// Set by graph builders, consumed by the Renderer on construction.
    graph_data: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// Which pass types a graph builder allows to be swapped selectively.
    /// Kept apart from `graph_data` (which holds a single value, the
    /// rebuilder) so a builder can set both. See [`SwapPolicy`].
    swap_policy: Option<crate::graph::SwapPolicy>,
    /// Resource names registered via [`declare_external_input`](Self::declare_external_input) —
    /// resources supplied by the host rather than written by any pass in the
    /// graph. `validate_dependencies` treats every name in this set as
    /// available from pass index 0. See `docs/helio_3_0_spec.md` §6.
    external_inputs: std::collections::HashSet<&'static str>,
    /// Executor-owned transient descriptor storage. Passes that build dynamic
    /// attachment slices can retain them here for the duration of a frame.
    pub(crate) frame_storage: RenderFrameStorage,
}

/// Everything about a pass that the locked schedule was computed from. Two
/// passes with equal interfaces can trade places without re-locking the graph
/// (see [`RenderGraph::swap_passes_from`]).
#[derive(PartialEq)]
struct PassInterface {
    reads: &'static [&'static str],
    writes: &'static [&'static str],
    resources: Vec<crate::graph::ResourceDecl>,
    aliases: Vec<(&'static str, &'static str)>,
    chain_transparent: bool,
    requires_ray_tracing: bool,
    requires_camera_jitter: bool,
    initializes_target: bool,
    debug_views: Vec<(&'static str, u32)>,
}

impl PassInterface {
    fn of(pass: &dyn RenderPass) -> Self {
        let mut builder = crate::graph::ResourceBuilder::new();
        pass.declare_resources(&mut builder);
        Self {
            reads: pass.reads(),
            writes: pass.writes(),
            resources: builder.declarations().to_vec(),
            aliases: builder.published_aliases.clone(),
            chain_transparent: pass.chain_transparent(),
            requires_ray_tracing: pass.requires_ray_tracing(),
            requires_camera_jitter: pass.requires_camera_jitter(),
            initializes_target: pass.initializes_target(),
            debug_views: pass
                .debug_views()
                .iter()
                .map(|view| (view.name, view.debug_mode))
                .collect(),
        }
    }
}
impl RenderGraph {
    pub fn requires_ray_tracing(&self) -> bool {
        self.passes.iter().any(|pass| pass.requires_ray_tracing())
    }

    fn create_reflected_groups(
        &mut self,
        pass_index: usize,
        registry: &crate::ResourceRegistry<'_>,
    ) -> Result<Vec<wgpu::BindGroup>> {
        let Some(pipeline) = self
            .reflected_pipelines
            .get(pass_index)
            .and_then(Option::as_ref)
        else {
            return Ok(Vec::new());
        };
        if self.reflected_group_cache.len() < self.passes.len() {
            self.reflected_group_cache
                .resize_with(self.passes.len(), Vec::new);
        }
        crate::shader::reuse_or_create_reflected_bind_groups(
            self.passes[pass_index].name(),
            &pipeline.bindings,
            &pipeline.layouts,
            &pipeline.overrides,
            registry,
            &self.device,
            &mut self.reflected_group_cache[pass_index],
        )
        .map(|groups| groups)
        .map_err(|error| {
            crate::Error::ResourceNotFound(format!("{}: {error}", self.passes[pass_index].name()))
        })
    }

    pub fn new(device: &std::sync::Arc<wgpu::Device>, queue: &wgpu::Queue) -> Self {
        Self {
            passes: Vec::new(),
            pass_index_map: HashMap::new(),
            profiler: Profiler::new(device, queue),
            pool: GraphTexturePool::new(),
            pipeline_cache: PipelineFormatCache::with_device(device),
            pipeline_formats: Vec::new(),
            pipeline_registries: Vec::new(),
            reflected_pipelines: Vec::new(),
            resources: HashMap::new(),
            resource_groups: Vec::new(),
            route_names: Vec::new(),
            pre_pass_actions: Vec::new(),
            device: device.clone(),
            internal_w: 0,
            internal_h: 0,
            output_w: 0,
            output_h: 0,
            delta_time: 0.0,
            frame_time: 0.0,
            frame_time_delta: 0.0,
            owns_device: true,
            gpu_render_bundles: Vec::new(),
            resources_allocated: false,
            subpass_chains: Vec::new(),
            parallel_layers: Vec::new(),
            #[cfg(not(target_arch = "wasm32"))]
            encoder_finish_pool: OnceLock::new(),
            finish_segment_budget: DEFAULT_FINISH_SEGMENT_BUDGET,
            reflected_group_cache: Vec::new(),
            finish_breakdown_enabled: std::env::var_os(FINISH_BREAKDOWN_ENV)
                .is_some_and(|value| value != "0"),
            finish_breakdown: Vec::new(),
            unit_caches: Vec::new(),
            recording_cache_enabled: recording_cache::enabled_by_env(),
            recording_cache_unsupported: false,
            recording_cache_active: false,
            recording_cache_inactive_reason: None,
            recording_cache_direct: Vec::new(),
            chain_membership: Vec::new(),
            prev_chain_membership: Vec::new(),
            chain_generation: 0,
            last_bundle_chain_gen: Vec::new(),
            locked: false,
            xr_active: false,
            pass_cache: Vec::new(),
            frame_count: 0,
            frame_demands: crate::FrameDemands::default(),
            resize_pending: false,
            graph_data: None,
            swap_policy: None,
            external_inputs: std::collections::HashSet::new(),
            frame_storage: RenderFrameStorage::new(),
        }
    }

    pub fn new_with_external_device(
        device: &std::sync::Arc<wgpu::Device>,
        queue: &wgpu::Queue,
    ) -> Self {
        let mut graph = Self::new(device, queue);
        graph.owns_device = false;
        graph
    }

    pub fn set_delta_time(&mut self, dt: f32) {
        self.delta_time = dt;
    }

    /// Set the frame clock animation reads (`PrepareContext::time`) and how
    /// far it advanced since the previous frame (`time_delta`).
    pub fn set_frame_clock(&mut self, time: f32, delta: f32) {
        self.frame_time = time;
        self.frame_time_delta = delta;
    }

    /// Record how long encoder finishing takes per pass, readable afterwards
    /// from [`Self::finish_breakdown`] and shown in the flamegraph as
    /// `encoder.finish: <passes>` scopes. Also enabled by setting the
    /// `HELIO_FINISH_BREAKDOWN` environment variable. Adds overhead; leave it
    /// off outside profiling.
    pub fn set_finish_breakdown(&mut self, enabled: bool) {
        self.finish_breakdown_enabled = enabled;
        if !enabled {
            self.finish_breakdown.clear();
        }
    }

    /// How much pass recording goes into each graphics segment handed to the
    /// finish threads. Smaller budgets give more, shorter segments to spread
    /// across threads, at some per-command-buffer submit cost; zero cuts at
    /// every pass boundary outside a fused chain.
    pub fn set_finish_segment_budget(&mut self, budget: std::time::Duration) {
        self.finish_segment_budget = budget;
    }

    /// The last frame's per-pass finish cost; empty unless
    /// [`Self::set_finish_breakdown`] is on.
    pub fn finish_breakdown(&self) -> &[FinishSegment] {
        &self.finish_breakdown
    }

    /// Resubmit each cacheable unit's command buffers while it records the
    /// same commands (see `recording_cache`). On by default; the
    /// `HELIO_RECORDING_CACHE=0` environment variable switches it off.
    pub fn set_recording_cache(&mut self, enabled: bool) {
        self.recording_cache_enabled = enabled;
        if !enabled {
            self.reset_recording_cache();
        }
    }

    /// Which passes the recording cache hit or missed, and why.
    pub fn recording_cache_stats(&self) -> recording_cache::RecordingCacheStats {
        recording_cache::RecordingCacheStats {
            active: self.recording_cache_active,
            inactive_reason: self.recording_cache_inactive_reason,
            direct: self.recording_cache_direct.clone(),
            units: self
                .unit_caches
                .iter()
                .filter(|unit| {
                    unit.hits + unit.misses + unit.bypassed > 0 || unit.uncacheable.is_some()
                })
                .map(|unit| recording_cache::UnitCacheStats {
                    pass: unit.label,
                    hits: unit.hits,
                    misses: unit.misses,
                    bypassed: unit.bypassed,
                    cached_variants: unit.variants.len(),
                    uncacheable: unit.uncacheable,
                    last_miss: unit.last_miss.clone(),
                })
                .collect(),
        }
    }

    fn reset_recording_cache(&mut self) {
        self.unit_caches = (0..self.passes.len()).map(|_| UnitCache::default()).collect();
    }

    /// Where this frame's cacheable units start: the end (exclusive) of the
    /// unit starting at each pass. A unit is a pass outside a fused chain, or
    /// a whole chain, when every pass in it supports the cache.
    fn recording_units(&mut self) -> Vec<Option<usize>> {
        let count = self.passes.len();
        let mut ends = vec![None; count];
        self.recording_cache_direct.clear();
        let mut start = 0;
        while start < count {
            let chain = self
                .pass_cache
                .get(start)
                .and_then(|cached| cached.as_ref())
                .map(|cached| cached.chain_range.clone())
                .filter(|range| !range.is_empty());
            let range = match chain {
                Some(range) if range.start == start => range,
                // A chain member that does not start the chain, or a
                // chain-transparent pass bridged into one.
                Some(_) => {
                    start += 1;
                    continue;
                }
                None if self.chain_membership.get(start).copied().unwrap_or(false) => {
                    start += 1;
                    continue;
                }
                None => start..start + 1,
            };
            let chained = range.len() > 1;
            let frame = self.frame_count;
            let direct = if let Some(reason) = self.unit_caches[start].uncacheable {
                Some(reason)
            } else if self.unit_caches[start].backing_off(frame) {
                Some(recording_cache::BACKING_OFF)
            } else if !range.clone().all(|index| self.passes[index].supports_recording_cache()) {
                Some("a pass opts out")
            } else if chained && range.clone().any(|index| self.gpu_render_bundles[index].is_some()) {
                // A bundle opens a render pass of its own.
                Some("a fused chain with a render bundle")
            } else {
                None
            };
            match direct {
                None => ends[start] = Some(range.end),
                Some(reason) => self
                    .recording_cache_direct
                    .extend(range.clone().map(|index| (self.passes[index].name(), reason))),
            }
            start = range.end;
        }
        ends
    }

    /// Whether this frame can use the recording cache, or why not.
    fn recording_cache_decision(
        &self,
        finish_breakdown: bool,
    ) -> std::result::Result<(), &'static str> {
        if !self.recording_cache_enabled {
            Err("switched off")
        } else if self.recording_cache_unsupported {
            Err(recording_cache::BACKEND_UNSUPPORTED)
        } else if finish_breakdown {
            Err("the finish breakdown is on")
        } else {
            Ok(())
        }
    }

    /// Enables the driver-validated persistent pipeline cache. Must be called
    /// before the graph is locked so all recipe construction observes the
    /// same cache object.
    pub fn enable_pipeline_cache_persistence(&mut self, path: impl Into<std::path::PathBuf>) {
        assert!(
            !self.locked,
            "pipeline cache persistence must be configured before lock()"
        );
        self.pipeline_cache = PipelineFormatCache::with_persistent_path(&self.device, path);
    }

    /// Flushes the driver cache blob immediately. `PipelineFormatCache` also
    /// flushes it on graph teardown, but hosts can call this at a safe save
    /// point or during an orderly shutdown.
    pub fn persist_pipeline_cache(&self) -> std::io::Result<()> {
        self.pipeline_cache.persist()
    }

    /// Sets the host's complete, explicit enumeration of reachable attachment
    /// formats. If the graph is already locked, newly configured variants are
    /// scheduled immediately and become available without stalling recording.
    pub fn set_pipeline_formats(&mut self, formats: Vec<PipelineFormatSet>) {
        self.pipeline_formats = formats;
        if self.locked {
            self.prepare_pipeline_registries();
        }
    }

    /// Adds one host-reachable format combination without discarding formats
    /// supplied by a graph builder. This is used by renderer hosts to wire the
    /// presentation format into custom graphs.
    pub fn add_pipeline_format(&mut self, format: PipelineFormatSet) {
        if !self.pipeline_formats.contains(&format) {
            self.pipeline_formats.push(format);
            if self.locked {
                self.prepare_pipeline_registries();
            }
        }
    }

    pub fn with_xr_mode(&mut self, active: bool) -> &mut Self {
        self.xr_active = active;
        // The pool must know *before* `lock()`/`init_transients()` allocates:
        // in XR mode every pool texture is created as a 2-layer array so the
        // passes' D2Array views match the `multiview_mask = 0b11` the executor
        // forces on them.
        self.pool.set_xr_mode(active);
        self
    }

    /// Returns true when at least one pass reconstructs the renderer's
    /// subpixel camera-jitter sequence.
    pub fn requires_camera_jitter(&self) -> bool {
        self.passes.iter().any(|pass| pass.requires_camera_jitter())
    }

    // ── Public API ──────────────────────────────────────────────────────

    /// Store opaque data (e.g. a GraphRebuilder) on the graph so the Renderer
    /// can retrieve it later without the caller having to pass it explicitly.
    pub fn set_graph_data<T: Send + Sync + 'static>(&mut self, data: T) {
        self.graph_data = Some(Box::new(data));
    }

    /// Take the stored opaque data, if it matches type `T`.
    pub fn take_graph_data<T: Send + Sync + 'static>(&mut self) -> Option<T> {
        self.graph_data.take().map(|b| *b.downcast::<T>().unwrap())
    }

    /// Declares which pass types of this graph may be swapped on their own or
    /// together when only some passes need replacing (shader hot reload). A
    /// graph without a policy is always rebuilt whole.
    pub fn set_swap_policy(&mut self, policy: crate::graph::SwapPolicy) {
        self.swap_policy = Some(policy);
    }

    /// The policy set by [`set_swap_policy`](Self::set_swap_policy), if any.
    pub fn swap_policy(&self) -> Option<&crate::graph::SwapPolicy> {
        self.swap_policy.as_ref()
    }

    /// Identity of every pass in execution-list order, for matching this
    /// graph's passes against another graph's.
    pub fn pass_identities(&self) -> Vec<crate::graph::PassIdentity> {
        self.passes
            .iter()
            .map(|pass| crate::graph::PassIdentity {
                type_id: pass.as_any().type_id(),
                name: pass.name(),
                type_name: pass.type_name(),
            })
            .collect()
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        if self.output_w == width && self.output_h == height && self.resources_allocated {
            return;
        }
        self.internal_w = width;
        self.internal_h = height;
        self.output_w = width;
        self.output_h = height;
        self.resize_pending = true;

        if self.locked {
            self.locked = false;
            self.lock(width, height);
            for pass in &mut self.passes {
                pass.on_resize(&self.device, width, height);
            }
        } else {
            self.pool.clear();
            // Cached reflected groups may bind the textures just dropped.
            self.reflected_group_cache.clear();
            self.collect_declarations();
            let (writes, reads, _) = self.chain_read_write_sets();
            self.parallel_layers = compute_parallel_layers(&writes, &reads);
            self.allocate_textures();
            self.prepare_pipeline_registries();
            self.detect_subpass_chains();
            self.resources_allocated = true;
            for pass in &mut self.passes {
                pass.on_resize(&self.device, width, height);
            }
            self.rebuild_gpu_render_bundles();
        }
    }

    pub fn init_transients(&mut self, width: u32, height: u32) {
        self.internal_w = width;
        self.internal_h = height;
        self.output_w = width;
        self.output_h = height;
        self.pool.clear();
        // Cached reflected groups may bind the textures just dropped.
        self.reflected_group_cache.clear();
        self.collect_declarations();
        let (writes, reads, _) = self.chain_read_write_sets();
        self.parallel_layers = compute_parallel_layers(&writes, &reads);
        self.allocate_textures();
        self.prepare_pipeline_registries();
        self.detect_subpass_chains();
        self.resources_allocated = true;
        self.rebuild_gpu_render_bundles();
    }

    /// Registers `name` as supplied by the host rather than by any pass in
    /// the graph. Called once at graph-build time by whoever writes the
    /// value every frame (today: `helio`'s `Renderer`, for
    /// `billboards`/`vg`/`corona_emitters`/`main_scene`).
    ///
    /// `validate_dependencies()` treats every registered external input as
    /// available from pass index 0, replacing the hardcoded literal list —
    /// a resource that no `declare_external_input` call and no pass's
    /// `write_group`/`write_color` covers is now a real validation error
    /// instead of a silent hardcoded exception.
    pub fn declare_external_input(&mut self, name: &'static str) {
        self.external_inputs.insert(name);
    }

    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        assert!(!self.locked, "RenderGraph: cannot add_pass() after lock()");
        let type_id = pass.as_any().type_id();
        self.pass_index_map
            .entry(type_id)
            .or_insert(self.passes.len());
        self.passes.push(pass);
        self.gpu_render_bundles.push(None);
    }

    /// Append a pass to a graph that may already be locked, for passes that
    /// only exist once something has happened at runtime (a finished bake).
    /// A locked graph is relocked like [`Self::replace_pass_at`] does, so
    /// the schedule and resource declarations include the new pass.
    pub fn add_pass_live(&mut self, pass: Box<dyn RenderPass>) {
        if !self.locked {
            self.add_pass(pass);
            return;
        }
        self.locked = false;
        self.add_pass(pass);
        self.gpu_render_bundles.clear();
        self.pass_cache.clear();
        if let Some(pass) = self.passes.last_mut() {
            pass.on_resize(&self.device, self.output_w, self.output_h);
        }
        self.lock(self.output_w, self.output_h);
        self.resize_pending = true;
    }

    pub fn find_pass_mut<T: RenderPass + 'static>(&mut self) -> Option<&mut T> {
        let idx = *self.pass_index_map.get(&TypeId::of::<T>())?;
        self.passes[idx].as_any_mut().downcast_mut::<T>()
    }

    pub fn find_pass<T: RenderPass + 'static>(&self) -> Option<&T> {
        let idx = *self.pass_index_map.get(&TypeId::of::<T>())?;
        self.passes[idx].as_any().downcast_ref::<T>()
    }

    /// Preserve opt-in streaming state when rebuilding a graph on this device.
    /// New pass configuration and graph resources remain authoritative.
    pub fn inherit_persistent_state(&mut self, previous: &mut RenderGraph) {
        let mut used = std::collections::HashSet::new();
        for pass in &mut self.passes {
            for (index, old) in previous.passes.iter_mut().enumerate() {
                if !used.contains(&index)
                    && pass.name() == old.name()
                    && pass.as_any().type_id() == old.as_any().type_id()
                    && pass.inherit_persistent_state(old.as_mut())
                {
                    used.insert(index);
                    pass.on_resize(&self.device, self.internal_w, self.internal_h);
                    break;
                }
            }
        }
    }

    /// Find the index of the first pass matching type `T`.
    pub fn pass_index_of<T: RenderPass + 'static>(&self) -> Option<usize> {
        self.passes
            .iter()
            .position(|p| (*p).as_any().downcast_ref::<T>().is_some())
    }

    /// Propagate a game/editor-mode toggle to every pass via
    /// `RenderPass::set_editor_mode` — a plain virtual dispatch, no downcasting,
    /// so this works without the graph (or its caller) knowing which concrete
    /// pass types actually care. See that trait method's docs for why callers
    /// should call this every frame rather than only on the transition.
    pub fn set_editor_mode(&mut self, enabled: bool) {
        for pass in &mut self.passes {
            pass.set_editor_mode(enabled);
        }
    }

    /// Broadcast generic per-frame projection inputs to every pass.
    ///
    /// Passes interpret only the fields they own; the graph and renderer
    /// facade never downcast into concrete pass implementations.
    pub fn set_frame_inputs(&mut self, inputs: &crate::RenderFrameInputs<'_>) {
        for pass in &mut self.passes {
            pass.set_frame_inputs(inputs);
        }
    }

    /// Publish pass-owned resources required before the first pass executes.
    ///
    /// These resources are borrowed from graph-owned pass state and therefore
    /// need the same narrowly-scoped lifetime extension used by the executor's
    /// output publication path. The graph remains the sole owner of this
    /// bridge; callers only receive the typed registry view.
    /// Whether a pass overwrites every pixel of the frame target before
    /// anything reads it (see [`RenderPass::initializes_target`]), so the
    /// host need not clear the target first.
    pub fn initializes_target(&self) -> bool {
        self.passes.iter().any(|pass| pass.initializes_target())
    }

    pub fn publish_frame_inputs<'a>(
        &self,
        camera: &crate::GpuCameraUniforms,
        registry: &mut crate::ResourceRegistry<'a>,
    ) {
        let frame_ptr = registry as *mut crate::ResourceRegistry<'a>;
        for pass in &self.passes {
            unsafe {
                pass.publish_frame_inputs(camera, &mut *frame_ptr);
            }
        }
    }

    /// Replace the pass at `index` with a new one.
    ///
    /// On a locked graph this re-locks it, which recreates every pooled
    /// texture and the reflected pipelines of every pass; only the replaced
    /// pass gets `on_resize`. Callers that need the rest of the graph left
    /// exactly as it is (shader hot reload) use
    /// [`swap_passes_from`](Self::swap_passes_from) instead.
    pub fn replace_pass_at(&mut self, index: usize, pass: Box<dyn RenderPass>) {
        if index < self.passes.len() {
            // Pipelines in the cache are keyed by pass name, so neither the
            // old nor the new pass may be handed the other's.
            self.pipeline_cache
                .invalidate_pass(self.passes[index].name());
            self.pipeline_cache.invalidate_pass(pass.name());
            self.passes[index] = pass;
            // Replacing one type can also expose a later instance of the old
            // type. Rebuild the first-instance map rather than patching one key.
            self.pass_index_map.clear();
            for (i, pass) in self.passes.iter().enumerate() {
                self.pass_index_map
                    .entry(pass.as_any().type_id())
                    .or_insert(i);
            }
            // The replacement can publish different resources to later passes.
            // Chain membership alone cannot detect stale dependent bundles.
            self.gpu_render_bundles.clear();
            self.pass_cache.clear();
            if self.locked {
                self.passes[index].on_resize(&self.device, self.output_w, self.output_h);
                self.locked = false;
                self.lock(self.output_w, self.output_h);
                self.resize_pending = true;
            } else if self.resources_allocated {
                self.passes[index].on_resize(&self.device, self.output_w, self.output_h);
                self.init_transients(self.output_w, self.output_h);
                self.resize_pending = true;
            }
        }
    }

    /// Checks whether `picks` (`(index in self, index in replacement)`) can be
    /// swapped in with [`swap_passes_from`](Self::swap_passes_from), without
    /// changing anything. The error says why not.
    ///
    /// A swap leaves the schedule, the pooled textures and every other pass
    /// alone, so it is only sound when the incoming pass has the same shape as
    /// the one it replaces: same type and name, same declared reads, writes,
    /// resources and chaining behaviour. A change of shader text cannot alter
    /// those (they come from Rust code and configuration); one that does is
    /// refused rather than risking a stale schedule.
    pub fn check_swap(
        &self,
        replacement: &RenderGraph,
        picks: &[(usize, usize)],
    ) -> std::result::Result<(), String> {
        if !self.locked || !replacement.locked {
            return Err("a graph is not locked".into());
        }
        if self.gpu_render_bundles.iter().any(Option::is_some) {
            // A prebuilt bundle bakes in pipelines and the bindings earlier
            // passes published; rebuilding it needs the whole schedule.
            return Err("the graph has prebuilt render bundles".into());
        }
        if self.pass_cache.len() != self.passes.len()
            || self.pipeline_registries.len() != self.passes.len()
            || self.reflected_pipelines.len() != self.passes.len()
            || replacement.reflected_pipelines.len() != replacement.passes.len()
        {
            return Err("the graph's per-pass tables are out of step with its passes".into());
        }
        let mut seen_live = std::collections::HashSet::new();
        let mut seen_new = std::collections::HashSet::new();
        for &(live, new) in picks {
            if !seen_live.insert(live) || !seen_new.insert(new) {
                return Err("a pass was picked twice".into());
            }
            let (Some(old), Some(fresh)) = (self.passes.get(live), replacement.passes.get(new))
            else {
                return Err(format!("pass index {live}/{new} is out of range"));
            };
            if old.as_any().type_id() != fresh.as_any().type_id() || old.name() != fresh.name() {
                return Err(format!(
                    "pass '{}' is not the same pass as '{}'",
                    old.name(),
                    fresh.name()
                ));
            }
            if PassInterface::of(&**old) != PassInterface::of(&**fresh) {
                return Err(format!(
                    "pass '{}' declares different resources or scheduling after the edit",
                    old.name()
                ));
            }
            if self.reflected_pipelines[live].is_some() != replacement.reflected_pipelines[new].is_some()
            {
                return Err(format!("pass '{}' changed its reflected bindings", old.name()));
            }
        }
        Ok(())
    }

    /// Moves the passes `picks` selects out of `replacement` into this graph,
    /// leaving every other pass of this graph, its pooled textures and its
    /// schedule untouched.
    ///
    /// Each incoming pass takes over what its predecessor opted to pass on
    /// ([`RenderPass::inherit_persistent_state`]), and brings along the
    /// reflected pipeline the replacement built for it. The graph's pipeline
    /// cache entries for the swapped pass are invalidated and its recipes
    /// re-declared from the new instance.
    ///
    /// Validated by [`check_swap`](Self::check_swap) first: on `Err` nothing
    /// has changed and `replacement` is intact. On `Ok`, `replacement` has been
    /// emptied of passes and should be dropped.
    pub fn swap_passes_from(
        &mut self,
        replacement: &mut RenderGraph,
        picks: &[(usize, usize)],
    ) -> std::result::Result<(), String> {
        self.check_swap(replacement, picks)?;

        let mut incoming: Vec<Option<Box<dyn RenderPass>>> =
            std::mem::take(&mut replacement.passes)
                .into_iter()
                .map(Some)
                .collect();
        for &(live, new) in picks {
            let mut fresh = incoming[new].take().expect("check_swap rejects repeated picks");
            if fresh.inherit_persistent_state(&mut *self.passes[live]) {
                fresh.on_resize(&self.device, self.internal_w, self.internal_h);
            }
            self.pipeline_cache.invalidate_pass(self.passes[live].name());
            self.passes[live] = fresh;
            self.pipeline_registries[live] = self.pipeline_registry_for(&*self.passes[live]);
            self.reflected_pipelines[live] = replacement.reflected_pipelines[new].take();
            if let Some(cache) = self.reflected_group_cache.get_mut(live) {
                // Keyed by layout, which the new pipeline owns.
                cache.clear();
            }
        }
        Ok(())
    }

    pub fn iter_passes_mut<T: RenderPass + 'static>(&mut self) -> impl Iterator<Item = &mut T> {
        self.passes
            .iter_mut()
            .filter_map(|p| p.as_any_mut().downcast_mut::<T>())
    }

    pub fn collect_debug_views(&self) -> Vec<crate::DebugViewDescriptor> {
        self.passes
            .iter()
            .flat_map(|p| p.debug_views().iter().copied())
            .collect()
    }

    /// Propagate a renderer-wide debug mode change to every pass.
    pub fn set_debug_mode(&mut self, mode: u32) {
        for pass in &mut self.passes {
            pass.set_debug_mode(mode);
        }
    }

    pub fn validate_dependencies(&self) -> std::result::Result<(), String> {
        use std::collections::HashSet;
        let mut available: HashSet<&str> = HashSet::new();
        available.extend(self.external_inputs.iter().copied());

        for (i, pass) in self.passes.iter().enumerate() {
            let name = pass.name();
            let (reads, writes) = self.dependency_declarations(pass.as_ref());
            for resource in reads {
                if !available.contains(resource) {
                    return Err(format!(
                        "RenderGraph validation failed: pass '{}' (index {}) reads '{}' \
                         but no prior pass writes it. Available: {:?}",
                        name, i, resource, available
                    ));
                }
            }
            for resource in writes {
                available.insert(resource);
            }
        }
        Ok(())
    }

    fn dependency_declarations<'a>(
        &self,
        pass: &'a dyn RenderPass,
    ) -> (Vec<&'a str>, Vec<&'a str>) {
        let mut reads = pass.reads().to_vec();
        let mut writes = pass.writes().to_vec();
        let mut builder = crate::graph::ResourceBuilder::new();
        pass.declare_resources(&mut builder);
        for declaration in builder.declarations() {
            let target = match declaration.access {
                crate::graph::ResourceAccess::Read => &mut reads,
                crate::graph::ResourceAccess::Write => &mut writes,
            };
            if !target.contains(&declaration.name) {
                target.push(declaration.name);
            }
        }
        (reads, writes)
    }

    pub fn dump_dependency_graph(&self) {
        eprintln!("digraph RenderGraph {{");
        for (i, pass) in self.passes.iter().enumerate() {
            eprintln!("  {} [label=\"{}\"];", i, pass.name());
            let (reads, _) = self.dependency_declarations(pass.as_ref());
            for resource in reads {
                for j in (0..i).rev() {
                    let (_, writes) = self.dependency_declarations(self.passes[j].as_ref());
                    if writes.contains(&resource) {
                        eprintln!("  {} -> {} [label=\"{}\"];", j, i, resource);
                        break;
                    }
                }
            }
        }
        eprintln!("}}");
    }

    pub fn profiler(&self) -> &Profiler {
        &self.profiler
    }

    /// Mutable profiler access for renderer-owned command buffers that are
    /// submitted outside the graph (for example the target clear).
    pub fn profiler_mut(&mut self) -> &mut Profiler {
        &mut self.profiler
    }

    /// Collect an owned snapshot of all resource and pass data for a debug
    /// overlay or inspector.
    pub fn collect_frame_debug_data(&self) -> FrameDebugData {
        let mut data = FrameDebugData::default();
        data.frame_count = self.frame_count;
        data.delta_time = self.delta_time;

        let mut total_bytes = 0u64;
        let mut alias_groups: HashMap<&str, Vec<&str>> = HashMap::new();

        for (name, rl) in &self.resources {
            let bpp = format_bpp(rl.format);
            let bytes =
                rl.width as u64 * rl.height as u64 * rl.depth_or_array_layers as u64 * bpp as u64
                    / 8;
            total_bytes += bytes;
            let alias = rl.alias_group.as_deref().unwrap_or("-").to_string();
            if rl.alias_group.is_some() {
                alias_groups
                    .entry(rl.alias_group.as_ref().unwrap())
                    .or_default()
                    .push(name);
            }
            data.resources.push(DebugResourceInfo {
                name: name.clone(),
                width: rl.width,
                height: rl.height,
                layers: rl.depth_or_array_layers,
                format_name: format_name(rl.format).to_string(),
                size_kb: bytes / 1024,
                alias,
                chain_local: rl.chain_local,
                first_write_pass: rl.first_write_pass,
                last_read_pass: rl.last_read_pass,
            });
        }
        data.total_vram_kb = total_bytes / 1024;
        data.physical_vram_kb = self.pool.physical_vram_bytes() / 1024;

        for (group, members) in &alias_groups {
            let t: u64 = members
                .iter()
                .filter_map(|n| {
                    self.resources.get(*n).map(|rl| {
                        let bpp = format_bpp(rl.format);
                        rl.width as u64
                            * rl.height as u64
                            * rl.depth_or_array_layers as u64
                            * bpp as u64
                            / 8
                    })
                })
                .sum();
            let saved = t * (members.len().saturating_sub(1) as u64);
            data.passes.push(DebugPassInfo {
                index: 999,
                name: format!(
                    "alias group '{}': {} members, ~{} KB saved",
                    group,
                    members.len(),
                    saved / 1024
                ),
                kind: String::new(),
                writes: Vec::new(),
                reads: Vec::new(),
                chain_marker: String::new(),
                parallel_layer: 0,
            });
        }

        let mut pass_chain: Vec<Option<usize>> = vec![None; self.passes.len()];
        for (ci, chain) in self.subpass_chains.iter().enumerate() {
            for pi in chain.clone() {
                pass_chain[pi] = Some(ci);
            }
        }

        for (i, pass) in self.passes.iter().enumerate() {
            let writes: Vec<String> = self
                .resources
                .iter()
                .filter(|(_, rl)| rl.first_write_pass == i)
                .map(|(n, _)| n.clone())
                .collect();
            let reads: Vec<String> = self
                .dependency_declarations(pass.as_ref())
                .0
                .into_iter()
                .map(str::to_owned)
                .collect();
            let r_or_c = if writes.is_empty() { "C" } else { "R" };
            let marker = match pass_chain[i] {
                Some(ci) => {
                    let chain = &self.subpass_chains[ci];
                    if i == chain.start {
                        format!("[{}.{}]", ci, chain.len())
                    } else {
                        format!("|.{}", chain.len())
                    }
                }
                None => String::new(),
            };
            data.passes.push(DebugPassInfo {
                index: i,
                name: pass.name().to_string(),
                kind: r_or_c.to_string(),
                writes,
                reads,
                chain_marker: marker,
                parallel_layer: self
                    .parallel_layers
                    .iter()
                    .position(|layer| layer.contains(&i))
                    .unwrap_or(0),
            });
        }

        for (ci, chain) in self.subpass_chains.iter().enumerate() {
            let names: Vec<String> = self.passes[chain.start..chain.end]
                .iter()
                .map(|p| p.name().to_string())
                .collect();
            data.subpass_chains
                .push(format!("chain {}: {}", ci, names.join(" → ")));
        }

        data
    }

    /// Combine the current graph capture with the latest CPU/GPU timing
    /// snapshot into an owned host-facing payload.
    ///
    /// GPU timings are asynchronous; a pass has `None` for a timing that is
    /// not present in the latest completed profiler snapshot.
    pub fn collect_graph_timeline(&self) -> crate::GraphTimelineData {
        let debug = self.collect_frame_debug_data();
        let timing = self.profiler.timing_snapshot();
        let mut timing_by_name = std::collections::HashMap::new();
        for pass in &timing.passes {
            timing_by_name.insert(pass.name, (pass.cpu_ms, pass.gpu_ms));
        }
        let passes = debug
            .passes
            .into_iter()
            .filter(|pass| pass.index < self.passes.len())
            .map(|pass| {
                let (cpu_ms, gpu_ms) = timing_by_name
                    .get(self.passes[pass.index].name())
                    .copied()
                    .unwrap_or((None, None));
                crate::GraphTimelinePass {
                    index: pass.index,
                    name: pass.name,
                    reads: pass.reads,
                    writes: pass.writes,
                    cpu_ms,
                    gpu_ms,
                    parallel_layer: pass.parallel_layer,
                    chain_marker: pass.chain_marker,
                }
            })
            .collect();
        crate::GraphTimelineData {
            frame_count: debug.frame_count,
            total_vram_kb: debug.total_vram_kb,
            physical_vram_kb: debug.physical_vram_kb,
            passes,
            resources: debug.resources,
        }
    }

    pub fn execute(
        &mut self,
        scene: &dyn SceneInput,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) -> Result<wgpu::SubmissionIndex> {
        let mut registry = crate::ResourceRegistry::empty();
        self.execute_with_registry(scene, target, depth, &mut registry)
    }

    /// Executes the graph against the host-provided transient resource registry.
    ///
    /// External inputs are written with typed [`crate::ResourceKey`] values
    /// before execution, and every pass observes the same registry through
    /// `PassContext::registry` and `PrepareContext::registry`.
    pub fn execute_with_registry(
        &mut self,
        scene: &dyn SceneInput,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        registry: &mut crate::ResourceRegistry<'_>,
    ) -> Result<wgpu::SubmissionIndex> {
        // Descriptor slices retained by passes are frame-scoped. Reuse the
        // executor-owned arena before recording begins; it remains alive until
        // this method returns and the command buffers have been submitted.
        #[cfg(not(target_arch = "wasm32"))]
        profiling::profile_scope!("RenderGraph::execute_with_registry");
        self.frame_storage.reset();
        assert!(
            self.locked,
            "RenderGraph::execute() requires lock() to be called first"
        );

        // External device owners drive wgpu polling. Consume callbacks from
        // that host cadence before reserving a bounded readback slot for this
        // frame; this never polls or waits.
        #[cfg(not(target_arch = "wasm32"))]
        let timestamps_scope =
            profiling::ProfileScope::new_static("RenderGraph: read pending GPU timestamps");
        if !self.owns_device {
            self.profiler.read_gpu_timestamps_deferred();
        }
        self.profiler.clear_cpu_timings();
        #[cfg(not(target_arch = "wasm32"))]
        drop(timestamps_scope);

        #[cfg(not(target_arch = "wasm32"))]
        let encoders_scope = profiling::ProfileScope::new_static("RenderGraph: create command encoders");
        let mut encoder = scene
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Graph"),
            });
        let mut compute_encoder =
            scene
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Compute Graph"),
                });
        #[cfg(not(target_arch = "wasm32"))]
        drop(encoders_scope);

        // Compute and graphics are submitted as separate command buffers. Keep
        // their frame spans separate: a timestamp that begins on one encoder
        // and ends on the other includes queue/encoder gaps and is not a GPU
        // work duration.
        self.profiler
            .begin_gpu_pass(&mut compute_encoder, "__graph_compute");
        registry.reset_tracking("RenderGraph");
        // Resolves every pass's reflected bind groups, reusing last frame's
        // where they bind the same resources.
        let reflected_groups: Vec<Vec<wgpu::BindGroup>> = {
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: create_reflected_groups");
            (0..self.passes.len())
                .map(|pass_index| self.create_reflected_groups(pass_index, registry))
                .collect::<Result<Vec<_>>>()?
        };
        let resized_this_frame = self.resize_pending;

        let mut chain_rp: Option<std::mem::ManuallyDrop<wgpu::RenderPass<'_>>> = None;
        // Label of the GPU timing span around the open chain, if any; closed
        // wherever `chain_rp` is dropped (see `CachedPass::chain_label`).
        let mut chain_span: Option<&'static str> = None;
        // End the open chain's render pass, then its timing span. The span's
        // end timestamp must come after the pass closes: the encoder accepts
        // no other commands while a render pass is open.
        macro_rules! close_chain {
            () => {
                if let Some(mut rp) = chain_rp.take() {
                    unsafe {
                        std::mem::ManuallyDrop::drop(&mut rp);
                    }
                }
                if let Some(label) = chain_span.take() {
                    self.profiler.end_gpu_pass(&mut encoder, label);
                }
            };
        }
        let mut chain_patch: Vec<Option<wgpu::RenderPassColorAttachment<'static>>> = Vec::new();

        // Encoder segments. The graphics encoder is cut at pass boundaries
        // (never inside an open chain) and each cut segment goes straight to
        // the finish pool; the breakdown diagnostic instead cuts both encoders
        // at every boundary and finishes them here, timed. Either way the
        // frame submits all compute, then all graphics, in recording order —
        // the same order as the two whole encoders.
        let finish_breakdown = self.finish_breakdown_enabled;
        self.finish_breakdown.clear();
        let finish_segment_budget = self.finish_segment_budget;
        #[cfg(not(target_arch = "wasm32"))]
        let finish_pool = (!finish_breakdown).then(|| {
            Arc::clone(
                self.encoder_finish_pool
                    .get_or_init(|| Arc::new(EncoderFinishPool::new())),
            )
        });
        #[cfg(not(target_arch = "wasm32"))]
        let (finish_reply_tx, finish_reply_rx) = mpsc::channel::<FinishReply>();
        #[cfg(not(target_arch = "wasm32"))]
        let mut finishes_in_flight = 0usize;
        let mut compute_segments: Vec<FrameBuffer> = Vec::new();
        // Indexed by segment; `None` while its finish is still in flight.
        let mut graphics_segments: Vec<Option<FrameBuffer>> = Vec::new();
        // Reusable command buffers encoded this frame but not cached.
        let mut frame_reusable: Vec<wgpu::ReusableCommandBuffer> = Vec::new();
        let mut segment_passes: Vec<&'static str> = Vec::new();
        let mut segment_recording = std::time::Duration::ZERO;
        // Whether the compute encoder holds commands (the frame's opening
        // timestamp, or a directly recorded pass's) that a cached unit's
        // compute commands must follow.
        let mut compute_pending = true;
        let new_encoder = |label: &'static str| {
            scene
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) })
        };
        // Close the current segment and carry on recording into fresh
        // encoders. Only valid while no chain render pass is open.
        macro_rules! cut_segment {
            () => {
                if !segment_passes.is_empty() {
                    let done_graphics =
                        std::mem::replace(&mut encoder, new_encoder("Render Graph"));
                    let passes = std::mem::take(&mut segment_passes);
                    segment_recording = std::time::Duration::ZERO;
                    let index = graphics_segments.len();
                    graphics_segments.push(None);
                    if finish_breakdown {
                        let done_compute =
                            std::mem::replace(&mut compute_encoder, new_encoder("Compute Graph"));
                        #[cfg(not(target_arch = "wasm32"))]
                        let _finish_scope = profiling::is_profiling_enabled().then(|| {
                            profiling::ProfileScope::new(format!(
                                "encoder.finish: {}",
                                passes.join(" + ")
                            ))
                        });
                        let start = std::time::Instant::now();
                        compute_segments.push(FrameBuffer::Once(done_compute.finish()));
                        let compute = start.elapsed();
                        let start = std::time::Instant::now();
                        graphics_segments[index] = Some(FrameBuffer::Once(done_graphics.finish()));
                        let graphics = start.elapsed();
                        self.finish_breakdown.push(FinishSegment {
                            passes,
                            compute,
                            graphics,
                        });
                    } else {
                        // The encoder comes back if no thread could take it.
                        #[cfg(not(target_arch = "wasm32"))]
                        let unsent = match finish_pool.as_ref() {
                            Some(pool) => pool.submit(index, done_graphics, &finish_reply_tx).err(),
                            None => Some(done_graphics),
                        };
                        #[cfg(target_arch = "wasm32")]
                        let unsent = Some(done_graphics);
                        match unsent {
                            Some(done_graphics) => {
                                graphics_segments[index] =
                                    Some(FrameBuffer::Once(done_graphics.finish()));
                            }
                            None => {
                                #[cfg(not(target_arch = "wasm32"))]
                                {
                                    finishes_in_flight += 1;
                                }
                            }
                        }
                    }
                }
            };
        }
        // Whether this pass boundary should close the current segment.
        #[cfg(not(target_arch = "wasm32"))]
        let pipelined = finish_pool.is_some();
        #[cfg(target_arch = "wasm32")]
        let pipelined = false;

        self.profiler
            .begin_gpu_pass(&mut encoder, "__graph_graphics");

        if self.unit_caches.len() != self.passes.len() {
            self.reset_recording_cache();
        }
        let recording_cache_decision =
            self.recording_cache_decision(finish_breakdown);
        let use_recording_cache = recording_cache_decision.is_ok();
        self.recording_cache_active = use_recording_cache;
        self.recording_cache_inactive_reason = recording_cache_decision.err();
        if use_recording_cache {
            for unit in &mut self.unit_caches {
                unit.evict_stale(self.frame_count);
            }
        }

        {
            // Every pass declares the optional resources it will read before
            // any pass executes, so producers earlier in the graph can skip
            // outputs nobody needs this frame.
            self.frame_demands.clear();
            for pass in self.passes.iter_mut() {
                let plan_ctx = PrepareContext {
                    device: scene.device(),
                    queue: scene.queue(),
                    frame_num: scene.frame_count(),
                    camera: scene.camera(),
                    camera_data: scene.camera_data(),
                    camera_generation: scene.camera_generation(),
                    scene_buffers: scene.scene_buffers(),
                    registry: &*registry,
                    resize: resized_this_frame,
                    width: self.internal_w,
                    height: self.internal_h,
                    delta_time: self.delta_time,
                    time: self.frame_time,
                    time_delta: self.frame_time_delta,
                    world_origin: scene.world_origin(),
                    owns_device: self.owns_device,
                };
                pass.declare_frame_demands(&plan_ctx, &mut self.frame_demands);
            }
            // SAFETY: `frame_demands` is graph-owned and not modified again
            // until the next frame's `execute_with_registry`; the registry
            // entry is frame-scoped (same lifetime bridge as `publish`).
            let demands: &crate::FrameDemands =
                unsafe { &*std::ptr::addr_of!(self.frame_demands) };
            registry.write(
                crate::ResourceKey::new(crate::FRAME_DEMANDS),
                demands,
                "RenderGraph",
            );
            // Raw pointer, not a borrow: `self.passes.iter_mut()` below holds
            // `self.passes` mutably for the loop body, and `pre_pass_actions`
            // is graph-owned and immutable for the duration of execution -- each
            // `unsafe { &*pre_pass_actions_ptr }` reborrow gets its own
            // inferred lifetime, long enough to satisfy `publish_group`'s
            // `'a` (tied to `registry`) rather than the shorter, unrelated
            // lifetime a plain `self.pre_pass_actions.get(..)` borrow would carry.
            let pre_pass_actions_ptr: *const Vec<Vec<PrePassAction>> = &self.pre_pass_actions;
            let pass_count = self.passes.len();
            // Recording cache: where each unit starts, and the one being
            // recorded (see `recording_cache`).
            let unit_ends = if use_recording_cache {
                self.recording_units()
            } else {
                self.recording_cache_direct.clear();
                Vec::new()
            };
            let mut open_unit: Option<OpenUnit> = None;
            let xr_active = self.xr_active;

            // prepare() and the graph-owned outputs it publishes, before
            // execute().
            macro_rules! prepare_pass {
                ($pass:expr, $pass_index:expr) => {
                    {
                        let _scope = self.profiler.scope($pass.name());
                        let prepare_ctx = PrepareContext {
                            device: scene.device(),
                            queue: scene.queue(),
                            frame_num: scene.frame_count(),
                            camera: scene.camera(),
                            camera_data: scene.camera_data(),
                            camera_generation: scene.camera_generation(),
                            scene_buffers: scene.scene_buffers(),
                            registry: &*registry,
                            resize: resized_this_frame,
                            width: self.internal_w,
                            height: self.internal_h,
                            delta_time: self.delta_time,
                            time: self.frame_time,
                            time_delta: self.frame_time_delta,
                            world_origin: scene.world_origin(),
                            owns_device: self.owns_device,
                        };
                        #[cfg(not(target_arch = "wasm32"))]
                        let _prepare_scope = profiling::is_profiling_enabled().then(|| {
                            profiling::ProfileScope::new(format!("{}::prepare", $pass.name()))
                        });
                        $pass.prepare(&prepare_ctx)?;
                    }

                    // Populate graph-owned output textures into ResourceRegistry BEFORE execute().
                    if let Some(actions) = unsafe { &*pre_pass_actions_ptr }.get($pass_index) {
                        for action in actions {
                            match action {
                                PrePassAction::Route { name, view } => {
                                    registry.route_named_texture(name, view, "Graph");
                                }
                                PrePassAction::Group { name, members } => {
                                    // Generic: the core resolves a `write_group`'s
                                    // members to concrete views but has no notion of
                                    // what they mean — only the owning pass (this
                                    // pass, since `Group` actions are always stored
                                    // at their group's first-write pass index) knows
                                    // how to publish them into its own bespoke
                                    // `ResourceRegistry` field (e.g. `.gbuffer`).
                                    let views: Vec<&wgpu::TextureView> =
                                        members.iter().map(|(_, v)| v).collect();
                                    $pass.publish_group(*name, &views, registry);
                                }
                            }
                        }
                    }
                };
            }
            // A GPU timing span on one of the open unit's streams. The
            // profiler's query indices restart every frame, so a frame of the
            // same shape records the same timestamps and still hits.
            macro_rules! unit_span {
                ($begin:ident, $stream:expr, $label:expr) => {
                    self.profiler.$begin(
                        &mut crate::cmd::CommandRecorder::from_stream(std::ptr::NonNull::from(
                            &mut $stream,
                        )),
                        $label,
                    )
                };
            }
            // End the open unit's render pass, then its chain's timing span,
            // as `close_chain!` does on the encoder.
            macro_rules! close_unit_pass {
                ($unit:expr) => {
                    if $unit.render_pass_open {
                        $unit.graphics.push(Cmd::EndRenderPass);
                        $unit.render_pass_open = false;
                    }
                    if let Some(label) = $unit.chain_span.take() {
                        unit_span!(end_gpu_pass_cmds, $unit.graphics, label);
                    }
                };
            }
            // execute() into the open unit's streams.
            macro_rules! record_execute {
                ($unit:expr, $pass:expr, $pass_index:expr, $transparent:expr, $subpass_index:expr, $subpass_count:expr) => {{
                    let render_pass_open = $unit.render_pass_open;
                    let mut ctx = PassContext {
                        encoder_ptr: std::ptr::null_mut(),
                        queue: scene.queue(),
                        compute_encoder_ptr: std::ptr::null_mut(),
                        target,
                        depth,
                        camera: scene.camera(),
                        camera_data: scene.camera_data(),
                        camera_generation: scene.camera_generation(),
                        scene_buffers: scene.scene_buffers(),
                        profiler: &mut self.profiler,
                        frame_num: scene.frame_count(),
                        width: self.internal_w,
                        height: self.internal_h,
                        device: scene.device(),
                        registry: &*registry,
                        owns_device: self.owns_device,
                        resource_pool: &self.pool,
                        subpass_index: $subpass_index,
                        subpass_count: $subpass_count,
                        active_render_pass: None,
                        active_compute_pass: None,
                        recorded: Some(RecordedStreams {
                            graphics: std::ptr::NonNull::from(&mut $unit.graphics),
                            compute: std::ptr::NonNull::from(&mut $unit.compute),
                            render_pass_open,
                        }),
                        pipeline_cache: &self.pipeline_cache,
                        pipelines: &self.pipeline_registries[$pass_index],
                        reflected_bind_groups: &reflected_groups[$pass_index],
                        reflected_pipeline: self.reflected_pipelines[$pass_index].as_ref(),
                        #[cfg(debug_assertions)]
                        chain_transparent: $transparent,
                    };
                    ctx.apply_reflected_bind_groups();
                    #[cfg(not(target_arch = "wasm32"))]
                    profiling::profile_scope!($pass.name());
                    $pass.execute(&mut ctx)?;
                }};
            }

            for (pass_index, pass) in self.passes.iter_mut().enumerate() {
                // Recording cache: a unit's passes record into streams of
                // their own. When they match a cached recording, its command
                // buffers are submitted again instead of encoding the unit.
                if open_unit.is_none() {
                    if let Some(end) = unit_ends.get(pass_index).copied().flatten() {
                        // The unit gets command buffers of its own, so what the
                        // encoders hold is submitted before them. Consecutive
                        // units leave nothing to cut.
                        close_chain!();
                        if segment_passes.is_empty() && graphics_segments.is_empty() {
                            // Only the frame's opening timestamp; it still has
                            // to come first.
                            segment_passes.push("__graph_graphics");
                        }
                        cut_segment!();
                        if compute_pending {
                            let done_compute =
                                std::mem::replace(&mut compute_encoder, new_encoder("Compute Graph"));
                            compute_segments.push(FrameBuffer::Once(done_compute.finish()));
                            compute_pending = false;
                        }
                        let cache = &mut self.unit_caches[pass_index];
                        cache.label = self
                            .pass_cache
                            .get(pass_index)
                            .and_then(|cached| cached.as_ref())
                            .filter(|cached| cached.chain_range.len() > 1)
                            .map_or(pass.name(), |cached| cached.chain_label);
                        let (compute, graphics) = cache.take_scratch();
                        open_unit = Some(OpenUnit {
                            first: pass_index,
                            end,
                            compute,
                            graphics,
                            render_pass_open: false,
                            chain_span: None,
                        });
                    }
                }
                if let Some(unit) = open_unit.as_mut() {
                    let pass_name = pass.name();
                    // Timed from after prepare(), as on the direct path.
                    let mut execute_start = std::time::Instant::now();
                    if let Some(bundle) = &self.gpu_render_bundles[pass_index] {
                        // As the direct path: no prepare(), and the bundle
                        // replayed in the pass's own render pass.
                        let desc = pass.render_pass_descriptor_with_pool_and_storage(
                            target,
                            depth,
                            &*registry,
                            &self.pool,
                            &mut self.frame_storage,
                        );
                        unit_span!(begin_gpu_pass_cmds, unit.graphics, pass_name);
                        unit_span!(begin_gpu_pass_cmds, unit.compute, pass_name);
                        if let Some(desc) = desc {
                            unit.graphics
                                .push(Cmd::BeginRenderPass(Box::new(RenderPassDesc::capture(&desc))));
                            unit.graphics.push(Cmd::ExecuteBundles(vec![bundle.clone()]));
                            unit.graphics.push(Cmd::EndRenderPass);
                        } else {
                            record_execute!(unit, pass, pass_index, false, 0, 0);
                        }
                        unit_span!(end_gpu_pass_cmds, unit.compute, pass_name);
                        unit_span!(end_gpu_pass_cmds, unit.graphics, pass_name);
                    } else {
                        prepare_pass!(pass, pass_index);
                        execute_start = std::time::Instant::now();
                        let desc = pass.render_pass_descriptor_with_pool_and_storage(
                            target,
                            depth,
                            &*registry,
                            &self.pool,
                            &mut self.frame_storage,
                        );
                        let cached = self.pass_cache.get(pass_index).and_then(|c| c.as_ref());
                        let chain = cached.filter(|c| !c.chain_range.is_empty());
                        match (desc, chain) {
                            (Some(desc), Some(c)) => {
                                // A fused chain: one render pass, timed as one
                                // span, as the direct path records it.
                                if pass_index == c.chain_range.start {
                                    unit_span!(begin_gpu_pass_cmds, unit.graphics, c.chain_label);
                                    unit.chain_span = Some(c.chain_label);
                                    unit.graphics.push(Cmd::BeginRenderPass(captured_render_pass(
                                        &desc,
                                        &c.store_ops,
                                        xr_active,
                                    )));
                                    unit.render_pass_open = true;
                                }
                                record_execute!(
                                    unit,
                                    pass,
                                    pass_index,
                                    false,
                                    c.subpass_index,
                                    c.subpass_count
                                );
                                if pass_index + 1 >= c.chain_range.end {
                                    close_unit_pass!(unit);
                                }
                            }
                            (Some(desc), None) => {
                                close_unit_pass!(unit);
                                unit_span!(begin_gpu_pass_cmds, unit.graphics, pass_name);
                                unit_span!(begin_gpu_pass_cmds, unit.compute, pass_name);
                                let store_ops = cached.map_or(&[][..], |c| &c.store_ops[..]);
                                unit.graphics.push(Cmd::BeginRenderPass(captured_render_pass(
                                    &desc, store_ops, xr_active,
                                )));
                                unit.render_pass_open = true;
                                record_execute!(unit, pass, pass_index, false, 0, 0);
                                close_unit_pass!(unit);
                                unit_span!(end_gpu_pass_cmds, unit.compute, pass_name);
                                unit_span!(end_gpu_pass_cmds, unit.graphics, pass_name);
                            }
                            (None, _) => {
                                let bridged = self
                                    .chain_membership
                                    .get(pass_index)
                                    .copied()
                                    .unwrap_or(false)
                                    && pass.chain_transparent();
                                if !bridged {
                                    close_unit_pass!(unit);
                                    unit_span!(begin_gpu_pass_cmds, unit.graphics, pass_name);
                                }
                                unit_span!(begin_gpu_pass_cmds, unit.compute, pass_name);
                                record_execute!(unit, pass, pass_index, bridged, 0, 0);
                                unit_span!(end_gpu_pass_cmds, unit.compute, pass_name);
                                if !bridged {
                                    unit_span!(end_gpu_pass_cmds, unit.graphics, pass_name);
                                }
                            }
                        }
                    }
                    if pass_index + 1 >= unit.end {
                        close_unit_pass!(unit);
                        let unit = open_unit.take().expect("the unit is open");
                        let resolved = self.unit_caches[unit.first].resolve(
                            scene.device(),
                            unit.compute,
                            unit.graphics,
                            self.frame_count,
                        );
                        if resolved.backend_unsupported {
                            self.recording_cache_unsupported = true;
                        }
                        let mut frame_buffer = |buffer: UnitBuffer, compute: bool| match buffer {
                            UnitBuffer::Cached(variant) => FrameBuffer::Cached {
                                pass: unit.first,
                                variant,
                                compute,
                            },
                            UnitBuffer::Once(buffer) => FrameBuffer::Once(buffer),
                            UnitBuffer::Uncached(buffer) => {
                                frame_reusable.push(buffer);
                                FrameBuffer::Uncached(frame_reusable.len() - 1)
                            }
                        };
                        if let Some(buffer) = resolved.compute {
                            compute_segments.push(frame_buffer(buffer, true));
                        }
                        if let Some(buffer) = resolved.graphics {
                            graphics_segments.push(Some(frame_buffer(buffer, false)));
                        }
                    }
                    pass.publish(registry);
                    self.profiler
                        .record_external_cpu_timing(pass_name, execute_start.elapsed());
                    continue;
                }
                // Recorded directly from here on: the encoders hold commands
                // a later unit's must follow.
                compute_pending = true;

                // A chain's passes share one render pass on the encoder, so
                // they stay in one segment until it closes. The final pass
                // always starts a segment of its own: the render thread
                // finishes the last segment itself, so keep it small.
                if chain_rp.is_none()
                    && (finish_breakdown
                        || (pipelined
                            && (segment_recording >= finish_segment_budget
                                || pass_index + 1 == pass_count)))
                {
                    cut_segment!();
                }
                segment_passes.push(pass.name());
                if let Some(bundle) = &self.gpu_render_bundles[pass_index] {
                    let pass_name = pass.name();
                    let execute_start = std::time::Instant::now();
                    let desc = pass.render_pass_descriptor_with_pool_and_storage(
                        target,
                        depth,
                        &*registry,
                        &self.pool,
                        &mut self.frame_storage,
                    );
                    if let Some(desc) = desc {
                        self.profiler.begin_gpu_pass(&mut encoder, pass_name);
                        self.profiler.begin_gpu_pass(&mut compute_encoder, pass_name);
                        let mut pass_encoder = encoder.begin_render_pass(&desc);
                        pass_encoder.execute_bundles(std::iter::once(bundle));
                    } else {
                        self.profiler.begin_gpu_pass(&mut encoder, pass_name);
                        self.profiler.begin_gpu_pass(&mut compute_encoder, pass_name);
                        let mut ctx = PassContext {
                            encoder_ptr: &mut encoder as *mut _,
                            queue: scene.queue(),
                            compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                            target,
                            depth,
                            camera: scene.camera(),
                            camera_data: scene.camera_data(),
                            camera_generation: scene.camera_generation(),
                            scene_buffers: scene.scene_buffers(),
                            profiler: &mut self.profiler,
                            frame_num: scene.frame_count(),
                            width: self.internal_w,
                            height: self.internal_h,
                            device: scene.device(),
                            registry: &*registry,
                            owns_device: self.owns_device,
                            resource_pool: &self.pool,
                            subpass_index: 0,
                            subpass_count: 0,
                            active_render_pass: None,
                            active_compute_pass: None,
                            recorded: None,
                            pipeline_cache: &self.pipeline_cache,
                            pipelines: &self.pipeline_registries[pass_index],
                            reflected_bind_groups: &reflected_groups[pass_index],
                            reflected_pipeline: self.reflected_pipelines[pass_index].as_ref(),
                            #[cfg(debug_assertions)]
                            chain_transparent: false,
                        };
                        ctx.apply_reflected_bind_groups();
                        #[cfg(not(target_arch = "wasm32"))]
                        profiling::profile_scope!(pass.name());
                        pass.execute(&mut ctx)?;
                    }

                    self.profiler.end_gpu_pass(&mut compute_encoder, pass_name);
                    self.profiler.end_gpu_pass(&mut encoder, pass_name);
                    pass.publish(registry);
                    let recorded = execute_start.elapsed();
                    segment_recording += recorded;
                    self.profiler.record_external_cpu_timing(pass_name, recorded);
                    continue;
                }

                prepare_pass!(pass, pass_index);

                // execute()
                let pass_name = pass.name();
                let execute_start = std::time::Instant::now();

                // Migrated path: executor manages render pass (pass implements render_pass_descriptor).
                let desc = pass.render_pass_descriptor_with_pool_and_storage(
                    target,
                    depth,
                    &*registry,
                    &self.pool,
                    &mut self.frame_storage,
                );
                // Which encoders get this pass's own begin/end timestamps.
                let mut timed_main = false;
                let mut timed_compute = false;
                if let Some(desc) = desc {
                    let cache = self.pass_cache.get(pass_index).and_then(|c| c.as_ref());
                    // Chains fuse whether or not GPU timing is on (Helio#298):
                    // a chain is timed as one span around its single hardware
                    // pass, since nothing can be written into the encoder
                    // while that pass is open. Its members keep CPU timings.
                    let is_chained = cache.map_or(false, |c| !c.chain_range.is_empty());
                    if !is_chained {
                        close_chain!();
                        self.profiler.begin_gpu_pass(&mut encoder, pass_name);
                        self.profiler.begin_gpu_pass(&mut compute_encoder, pass_name);
                        timed_main = true;
                        timed_compute = true;
                    }

                    if is_chained {
                        let c = cache.unwrap();
                        if pass_index == c.chain_range.start {
                            self.profiler.begin_gpu_pass(&mut encoder, c.chain_label);
                            chain_span = Some(c.chain_label);
                            chain_patch.clear();
                            chain_patch.extend(desc.color_attachments.iter().enumerate().map(
                                |(i, opt)| {
                                    let mut a = opt.clone();
                                    if let Some(store) = c.store_ops.get(i).copied().flatten() {
                                        if let Some(ref mut att) = a {
                                            att.ops.store = store;
                                        }
                                    }
                                    unsafe {
                                        std::mem::transmute::<
                                            Option<wgpu::RenderPassColorAttachment<'_>>,
                                            Option<wgpu::RenderPassColorAttachment<'static>>,
                                        >(a)
                                    }
                                },
                            ));
                            let chain_desc = wgpu::RenderPassDescriptor {
                                label: desc.label,
                                color_attachments: &chain_patch,
                                depth_stencil_attachment: desc.depth_stencil_attachment,
                                timestamp_writes: desc.timestamp_writes,
                                occlusion_query_set: desc.occlusion_query_set,
                                multiview_mask: if self.xr_active {
                                    Some(std::num::NonZeroU32::new(0b11).unwrap())
                                } else {
                                    desc.multiview_mask
                                },
                            };
                            let rp = unsafe {
                                let enc = &mut *std::ptr::addr_of_mut!(encoder);
                                enc.begin_render_pass(&chain_desc)
                            };
                            chain_rp = Some(std::mem::ManuallyDrop::new(rp));
                        }

                        let mut ctx = PassContext {
                            encoder_ptr: std::ptr::addr_of_mut!(encoder),
                            queue: scene.queue(),
                            compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                            target,
                            depth,
                            camera: scene.camera(),
                            camera_data: scene.camera_data(),
                            camera_generation: scene.camera_generation(),
                            scene_buffers: scene.scene_buffers(),
                            profiler: &mut self.profiler,
                            frame_num: scene.frame_count(),
                            width: self.internal_w,
                            height: self.internal_h,
                            device: scene.device(),
                            registry: &*registry,
                            owns_device: self.owns_device,
                            resource_pool: &self.pool,
                            subpass_index: c.subpass_index,
                            subpass_count: c.subpass_count,
                            active_render_pass: chain_rp
                                .as_mut()
                                .map(|rp| &mut **rp as *mut _ as *mut _),
                            active_compute_pass: None,
                            recorded: None,
                            pipeline_cache: &self.pipeline_cache,
                            pipelines: &self.pipeline_registries[pass_index],
                            reflected_bind_groups: &reflected_groups[pass_index],
                            reflected_pipeline: self.reflected_pipelines[pass_index].as_ref(),
                            #[cfg(debug_assertions)]
                            chain_transparent: false,
                        };
                        ctx.apply_reflected_bind_groups();
                        #[cfg(not(target_arch = "wasm32"))]
                        profiling::profile_scope!(pass.name());
                        pass.execute(&mut ctx)?;

                        if pass_index + 1 >= c.chain_range.end {
                            close_chain!();
                        }
                    } else {
                        let standalone_atts: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> =
                            desc.color_attachments
                                .iter()
                                .enumerate()
                                .map(|(i, opt)| {
                                    let mut a = opt.clone();
                                    if let Some(store) =
                                        cache.and_then(|c| c.store_ops.get(i).copied()).flatten()
                                    {
                                        if let Some(ref mut att) = a {
                                            att.ops.store = store;
                                        }
                                    }
                                    a
                                })
                                .collect();
                        let standalone_desc = wgpu::RenderPassDescriptor {
                            label: desc.label,
                            color_attachments: &standalone_atts,
                            depth_stencil_attachment: desc.depth_stencil_attachment,
                            timestamp_writes: desc.timestamp_writes,
                            occlusion_query_set: desc.occlusion_query_set,
                            multiview_mask: if self.xr_active {
                                Some(std::num::NonZeroU32::new(0b11).unwrap())
                            } else {
                                desc.multiview_mask
                            },
                        };

                        let mut rp = unsafe {
                            let enc = &mut *std::ptr::addr_of_mut!(encoder);
                            enc.begin_render_pass(&standalone_desc)
                        };
                        {
                            let mut ctx = PassContext {
                                encoder_ptr: std::ptr::addr_of_mut!(encoder),
                                queue: scene.queue(),
                                compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                                target,
                                depth,
                                camera: scene.camera(),
                                camera_data: scene.camera_data(),
                                camera_generation: scene.camera_generation(),
                                scene_buffers: scene.scene_buffers(),
                                profiler: &mut self.profiler,
                                frame_num: scene.frame_count(),
                                width: self.internal_w,
                                height: self.internal_h,
                                device: scene.device(),
                                registry: &*registry,
                                owns_device: self.owns_device,
                                resource_pool: &self.pool,
                                subpass_index: 0,
                                subpass_count: 0,
                                active_render_pass: Some(&mut rp as *mut _ as *mut _),
                                active_compute_pass: None,
                                recorded: None,
                                pipeline_cache: &self.pipeline_cache,
                                pipelines: &self.pipeline_registries[pass_index],
                                reflected_bind_groups: &reflected_groups[pass_index],
                                reflected_pipeline: self.reflected_pipelines[pass_index].as_ref(),
                                #[cfg(debug_assertions)]
                                chain_transparent: false,
                            };
                            ctx.apply_reflected_bind_groups();
                            #[cfg(not(target_arch = "wasm32"))]
                            profiling::profile_scope!(pass.name());
                            pass.execute(&mut ctx)?;
                        }
                    }
                } else {
                    let bridged = self
                        .chain_membership
                        .get(pass_index)
                        .copied()
                        .unwrap_or(false)
                        && pass.chain_transparent();
                    if bridged {
                        // Inside an open chain: the main encoder is locked by
                        // the chain's render pass, and a chain-transparent
                        // pass only records on the compute encoder anyway.
                        self.profiler.begin_gpu_pass(&mut compute_encoder, pass_name);
                        timed_compute = true;
                    } else {
                        close_chain!();
                        self.profiler.begin_gpu_pass(&mut encoder, pass_name);
                        self.profiler.begin_gpu_pass(&mut compute_encoder, pass_name);
                        timed_main = true;
                        timed_compute = true;
                    }

                    let mut ctx = PassContext {
                        encoder_ptr: std::ptr::addr_of_mut!(encoder),
                        queue: scene.queue(),
                        compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                        target,
                        depth,
                        camera: scene.camera(),
                        camera_data: scene.camera_data(),
                        camera_generation: scene.camera_generation(),
                        scene_buffers: scene.scene_buffers(),
                        profiler: &mut self.profiler,
                        frame_num: scene.frame_count(),
                        width: self.internal_w,
                        height: self.internal_h,
                        device: scene.device(),
                        registry: &*registry,
                        owns_device: self.owns_device,
                        resource_pool: &self.pool,
                        subpass_index: 0,
                        subpass_count: 0,
                        active_render_pass: None,
                        active_compute_pass: None,
                        recorded: None,
                        pipeline_cache: &self.pipeline_cache,
                        pipelines: &self.pipeline_registries[pass_index],
                        reflected_bind_groups: &reflected_groups[pass_index],
                        reflected_pipeline: self.reflected_pipelines[pass_index].as_ref(),
                        #[cfg(debug_assertions)]
                        chain_transparent: bridged,
                    };
                    ctx.apply_reflected_bind_groups();
                    #[cfg(not(target_arch = "wasm32"))]
                    profiling::profile_scope!(pass.name());
                    pass.execute(&mut ctx)?;
                }

                // execute() may record raw commands on either encoder even
                // without a render-pass descriptor. Close the nested scopes
                // in reverse order; the profiler sums both stream durations.
                if timed_compute {
                    self.profiler.end_gpu_pass(&mut compute_encoder, pass_name);
                }
                if timed_main {
                    self.profiler.end_gpu_pass(&mut encoder, pass_name);
                }

                pass.publish(registry);
                let recorded = execute_start.elapsed();
                segment_recording += recorded;
                self.profiler.record_external_cpu_timing(pass_name, recorded);
            }
            debug_assert!(open_unit.is_none(), "a recording-cache unit ends at its last pass");
        }

        close_chain!();
        self.profiler.end_gpu_pass(&mut encoder, "__graph_graphics");
        self.profiler
            .end_gpu_pass(&mut compute_encoder, "__graph_compute");
        // Resolve after the final graphics timestamp, not before graphics runs.
        self.profiler
            .resolve_gpu_queries(&mut encoder, self.frame_count);
        // The breakdown times the last pass's segment on its own; it carries
        // the graph's closing timestamps and query resolve.
        if finish_breakdown {
            cut_segment!();
        }
        let command_buffers = {
            // Finishing runs wgpu's full command validation and encoding.
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: encoder.finish");
            // The compute encoder is small; let the pool take it while this
            // thread finishes the last graphics segment.
            #[cfg(not(target_arch = "wasm32"))]
            let compute_unsent = match finish_pool.as_ref() {
                Some(pool) => pool
                    .submit(COMPUTE_SEGMENT, compute_encoder, &finish_reply_tx)
                    .err(),
                None => Some(compute_encoder),
            };
            #[cfg(target_arch = "wasm32")]
            let compute_unsent = Some(compute_encoder);
            #[cfg(not(target_arch = "wasm32"))]
            if compute_unsent.is_none() {
                finishes_in_flight += 1;
            }
            let mut compute_last = compute_unsent.map(|compute| compute.finish());
            let graphics_last = {
                #[cfg(not(target_arch = "wasm32"))]
                profiling::profile_scope!("RenderGraph: graphics encoder.finish (last segment)");
                encoder.finish()
            };
            #[cfg(not(target_arch = "wasm32"))]
            {
                profiling::profile_scope!("RenderGraph: wait for encoder-finish threads");
                // Drop this frame's own sender so a job lost with a dead
                // worker ends the wait instead of blocking it forever.
                drop(finish_reply_tx);
                let mut first_panic = None;
                for _ in 0..finishes_in_flight {
                    match finish_reply_rx.recv() {
                        Ok((COMPUTE_SEGMENT, Ok(buffer))) => compute_last = Some(buffer),
                        Ok((index, Ok(buffer))) => {
                            graphics_segments[index] = Some(FrameBuffer::Once(buffer))
                        }
                        Ok((_, Err(payload))) => {
                            first_panic.get_or_insert(payload);
                        }
                        Err(_) => panic!("helio encoder-finish thread exited mid-frame"),
                    }
                }
                if let Some(payload) = first_panic {
                    std::panic::resume_unwind(payload);
                }
            }
            let mut command_buffers = compute_segments;
            command_buffers.extend(compute_last.map(FrameBuffer::Once));
            command_buffers.extend(graphics_segments.into_iter().map(|segment| {
                segment.expect("every graphics segment is finished before submit")
            }));
            command_buffers.push(FrameBuffer::Once(graphics_last));
            command_buffers
        };
        let submission_index = {
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: queue.submit (graph)");
            let unit_caches = &self.unit_caches;
            let items = command_buffers.into_iter().map(|buffer| match buffer {
                FrameBuffer::Once(buffer) => wgpu::SubmitItem::Once(buffer),
                FrameBuffer::Cached { pass, variant, compute } => {
                    let recording = &unit_caches[pass].variants[variant];
                    let buffer = if compute { &recording.compute } else { &recording.graphics };
                    wgpu::SubmitItem::Reusable(buffer.as_ref().expect("cached stream is not empty"))
                }
                FrameBuffer::Uncached(index) => wgpu::SubmitItem::Reusable(&frame_reusable[index]),
            });
            scene.queue().submit_mixed(items)
        };
        {
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: upload::finish_frame");
            crate::upload::finish_frame();
        }

        #[cfg(not(target_arch = "wasm32"))]
        let readback_scope = profiling::ProfileScope::new_static("RenderGraph: read GPU timestamps");
        if self.owns_device {
            self.profiler.read_gpu_timestamps_blocking(scene.device());
        } else {
            self.profiler.read_gpu_timestamps_deferred();
        }
        #[cfg(not(target_arch = "wasm32"))]
        drop(readback_scope);
        {
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: update_snapshot");
            self.profiler
                .update_snapshot(self.frame_count, self.passes.iter().map(|pass| pass.name()));
        }

        self.frame_count += 1;
        self.resize_pending = false;

        Ok(submission_index)
    }

    /// Finalize the graph after all passes have been added.
    fn prepare_pipeline_registries(&mut self) {
        let registries: Vec<PipelineRegistry> = self
            .passes
            .iter()
            .map(|pass| self.pipeline_registry_for(&**pass))
            .collect();
        self.pipeline_registries = registries;
    }

    /// Declares `pass`'s pipeline recipes, schedules every format variant the
    /// host enumerated, and returns the registry of the ones that are ready.
    fn pipeline_registry_for(&self, pass: &dyn RenderPass) -> PipelineRegistry {
        let device = self.device.clone();
        let mut declarations = crate::graph::PipelineRecipeBuilder::new();
        pass.declare_pipelines(&mut declarations);
        let mut registry = PipelineRegistry::new();
        for recipe in declarations.into_recipes() {
            let key_for_builder = recipe.key.clone();
            let build = Arc::clone(&recipe.build);
            for formats in &self.pipeline_formats {
                if formats.color_formats.len() != recipe.key.color_formats.len() {
                    continue;
                }
                let variant_key = recipe.key.with_formats(formats);
                let variant_for_builder = variant_key.clone();
                let build = Arc::clone(&build);
                let device = device.clone();
                self.pipeline_cache
                    .try_get_or_schedule(variant_key, move |driver_cache| {
                        build(&device, &variant_for_builder, driver_cache)
                    });
            }
            let build = Arc::clone(&build);
            let device = device.clone();
            if let Some(pipeline) = self
                .pipeline_cache
                .try_get_or_schedule(recipe.key, move |driver_cache| {
                    build(&device, &key_for_builder, driver_cache)
                })
            {
                registry.insert(recipe.handle, pipeline);
            }
        }
        registry
    }

    /// Finalize the graph after all passes have been added.
    pub fn lock(&mut self, width: u32, height: u32) {
        assert!(!self.locked, "RenderGraph::lock() called twice");
        self.internal_w = width;
        self.internal_h = height;
        self.output_w = width;
        self.output_h = height;
        self.pool.clear();
        // Cached reflected groups may bind the textures just dropped.
        self.reflected_group_cache.clear();
        self.collect_declarations();
        let (writes, reads, _) = self.chain_read_write_sets();
        self.parallel_layers = compute_parallel_layers(&writes, &reads);

        self.reflected_pipelines = self
            .passes
            .iter()
            .map(|pass| {
                pass.reflected_shader().map(|shader| {
                    let mut pipeline =
                        crate::shader::create_reflected_pipeline(&self.device, pass.name(), shader)
                            .unwrap_or_else(|error| {
                                panic!("{}: reflected shader setup failed: {error}", pass.name())
                            });
                    pass.declare_bindings(&mut pipeline.overrides);
                    pipeline
                })
            })
            .collect();

        // Phase 1: first texture allocation (no alias groups).
        self.allocate_textures();

        self.prepare_pipeline_registries();

        // Build a "canon" `ResourceRegistry` for the attachment probe below by
        // replaying the exact same pre-pass routing the real per-frame loop
        // performs (see `execute_with_registry`), generically: plain
        // named routes via `route_named_texture`, and any `write_group`
        // bundle via the owning pass's `publish_group` — core never
        // special-cases a specific group's name here.
        let mut canon = crate::ResourceRegistry::empty();
        for (pi, actions) in self.pre_pass_actions.iter().enumerate() {
            let Some(pass) = self.passes.get(pi) else {
                continue;
            };
            for action in actions {
                match action {
                    PrePassAction::Route { name, view } => {
                        canon.route_named_texture(name, view, "Graph");
                    }
                    PrePassAction::Group { name, members } => {
                        let views: Vec<&wgpu::TextureView> =
                            members.iter().map(|(_, v)| v).collect();
                        pass.publish_group(*name, &views, &mut canon);
                    }
                }
            }
        }

        let dummy_target = {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Lock Dummy Target"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let dummy_depth = self.pool.get_view("depth").cloned().unwrap_or_else(|| {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Lock Dummy Depth"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        });

        let probes: Vec<Option<(usize, Vec<usize>)>> = self
            .passes
            .iter()
            .map(|pass| {
                let desc = pass.render_pass_descriptor_with_pool_and_storage(
                    &dummy_target,
                    &dummy_depth,
                    &canon,
                    &self.pool,
                    &mut self.frame_storage,
                )?;
                let color_len = desc.color_attachments.len();
                let mut signature: Vec<usize> = desc
                    .color_attachments
                    .iter()
                    .map(|opt| {
                        opt.as_ref()
                            .map(|a| a.view as *const wgpu::TextureView as usize)
                            .unwrap_or(0)
                    })
                    .collect();
                signature.push(
                    desc.depth_stencil_attachment
                        .as_ref()
                        .map(|d| d.view as *const wgpu::TextureView as usize)
                        .unwrap_or(0),
                );
                Some((color_len, signature))
            })
            .collect();
        let attachments: Vec<Option<Vec<usize>>> = probes
            .iter()
            .map(|p| p.as_ref().map(|(_, sig)| sig.clone()))
            .collect();

        drop(canon);

        // Phase 2: detect chains and compute chain_local BEFORE final allocation.
        // Save previous membership so incremental bundle rebuild can find
        // the first divergent pass and rebuild from there.
        self.prev_chain_membership = self.chain_membership.clone();
        self.detect_subpass_chains_probed(&attachments);
        self.chain_membership = vec![false; self.passes.len()];
        for chain in &self.subpass_chains {
            for pi in chain.clone() {
                self.chain_membership[pi] = true;
            }
        }
        for rl in self.resources.values_mut() {
            rl.chain_local = self
                .subpass_chains
                .iter()
                .any(|c| c.start <= rl.first_write_pass && rl.last_read_pass < c.end);
        }

        // Phase 3: assign alias groups so chain-local and non-chain-local
        // resources never share a physical allocation.  This prevents the
        // situation where a chain-local resource is forced to use
        // StoreOp::Store because its backing memory is shared with a
        // non-chain-local resource.
        self.assign_chain_aware_alias_groups();

        // Phase 4: re-allocate textures with chain-aware alias groups.
        self.pool.clear();
        // Cached reflected groups may bind the textures just dropped.
        self.reflected_group_cache.clear();
        self.allocate_textures();
        self.resources_allocated = true;

        // Phase 5: detect chain membership changes for incremental bundle rebuild.
        // Only advance the generation if membership actually changed.
        let membership_dirty = self.passes.len() != self.prev_chain_membership.len()
            || self
                .chain_membership
                .iter()
                .zip(&self.prev_chain_membership)
                .any(|(c, p)| c != p);
        if membership_dirty {
            self.chain_generation = self.chain_generation.wrapping_add(1);
        }

        // Phase 6: build pass cache and render bundles.
        self.pass_cache = probes
            .into_iter()
            .enumerate()
            .map(|(pi, probe)| {
                let (color_len, _) = probe?;
                let chain = self.subpass_chains.iter().find(|c| c.contains(&pi));
                let chain_range = chain.cloned().unwrap_or(0..0);
                let subpass_index = chain.map_or(0, |c| (pi - c.start) as u32);
                let subpass_count = chain.map_or(0, |c| c.len() as u32);
                let store_ops: Vec<Option<wgpu::StoreOp>> = vec![None; color_len];
                let chain_label = match chain {
                    Some(c) => {
                        let names: Vec<&'static str> =
                            self.passes[c.clone()].iter().map(|pass| pass.name()).collect();
                        super::scheduling::chain_label(&names)
                    }
                    None => "",
                };
                Some(CachedPass {
                    store_ops,
                    subpass_index,
                    subpass_count,
                    chain_range,
                    chain_label,
                })
            })
            .collect();
        self.reset_recording_cache();

        self.rebuild_gpu_render_bundles_incremental();

        {
            let mut w_set: Vec<Vec<&str>> = Vec::with_capacity(self.passes.len());
            let mut r_set: Vec<Vec<&str>> = Vec::with_capacity(self.passes.len());
            for p in self.passes.iter() {
                let mut w: Vec<&str> = p.writes().to_vec();
                let mut r: Vec<&str> = p.reads().to_vec();
                let mut b = crate::graph::ResourceBuilder::new();
                p.declare_resources(&mut b);
                for d in b.declarations() {
                    match d.access {
                        crate::graph::ResourceAccess::Read => {
                            if !r.contains(&d.name) {
                                r.push(d.name);
                            }
                        }
                        crate::graph::ResourceAccess::Write => {
                            if !w.contains(&d.name) {
                                w.push(d.name);
                            }
                        }
                    }
                }
                w_set.push(w);
                r_set.push(r);
            }
            eprintln!(
                "[RenderGraph] {} passes, {} chain(s):",
                self.passes.len(),
                self.subpass_chains.len()
            );
            for i in 0..self.passes.len() {
                let name = self.passes[i].name();
                let is_chain_start = self.subpass_chains.iter().any(|c| c.start == i);
                let is_chain_mid = self.subpass_chains.iter().any(|c| i > c.start && i < c.end);
                let marker = if is_chain_start {
                    " ──chain──►"
                } else if is_chain_mid {
                    " │         "
                } else {
                    "           "
                };
                let w_str = if w_set[i].is_empty() {
                    "–".to_string()
                } else {
                    w_set[i].join(",")
                };
                let r_str = if r_set[i].is_empty() {
                    "–".to_string()
                } else {
                    r_set[i].join(",")
                };
                eprintln!("  {:>2}. {:<28} W: {}  R: {}", i, name, w_str, r_str);
                if i + 1 < self.passes.len() {
                    let can_fuse = w_set[i].iter().any(|w| r_set[i + 1].contains(w));
                    let is_fused = self
                        .subpass_chains
                        .iter()
                        .any(|c| c.contains(&i) && c.contains(&(i + 1)));
                    if is_fused && !can_fuse && self.passes[i + 1].chain_transparent() {
                        eprintln!(
                            "  {:>2}.{:>2} CHAINED  (bridged over transparent pass '{}')",
                            "",
                            "",
                            self.passes[i + 1].name()
                        );
                    } else {
                        let why = if can_fuse {
                            let common: Vec<&str> = w_set[i]
                                .iter()
                                .filter(|w| r_set[i + 1].contains(w))
                                .copied()
                                .collect();
                            format!("fusable via {}", common.join(","))
                        } else {
                            let mut reasons = Vec::new();
                            for w in &w_set[i] {
                                if !r_set[i + 1].contains(w) {
                                    reasons.push(format!("{} not read by next", w));
                                }
                            }
                            if reasons.is_empty() {
                                reasons.push("no writes from this pass".to_string());
                            }
                            reasons.join("; ")
                        };
                        if is_fused {
                            eprintln!("  {:>2}.{:>2} CHAINED  ({})", "", "", why);
                        } else if can_fuse {
                            eprintln!("  {:>2}.{:>2} NOT CHAINED — both must implement render_pass_descriptor. ({})", "", "", why);
                        }
                    }
                }
                eprintln!("  {}", marker);
            }
        }
        self.locked = true;
    }

    /// Rebuild all GPU render bundles from scratch.
    fn rebuild_gpu_render_bundles(&mut self) {
        self.gpu_render_bundles.clear();
        let mut base = crate::ResourceRegistry::empty();
        for pass in &mut self.passes {
            let bundle = pass.build_gpu_render_bundle(&self.device, &base);
            self.gpu_render_bundles.push(bundle);
            pass.publish(&mut base);
        }
        drop(base);
        self.last_bundle_chain_gen = vec![self.chain_generation; self.passes.len()];
    }

    /// Incrementally rebuild bundles from the first pass whose chain
    /// membership changed.  When membership is stable (chains haven't
    /// changed) and the bundle count matches, this is a no-op —
    /// `set_render_size()` doesn't force a rebuild.
    fn rebuild_gpu_render_bundles_incremental(&mut self) {
        if self.passes.is_empty() {
            self.rebuild_gpu_render_bundles();
            return;
        }

        // First lock: bundle vec hasn't been sized yet — full rebuild.
        if self.gpu_render_bundles.len() != self.passes.len() {
            self.rebuild_gpu_render_bundles();
            return;
        }

        // Find the first pass whose chain membership diverged from the
        // previous frame.  Only passes at or after this index need
        // rebuilding because earlier passes' publishing is unchanged.
        let first_dirty = self
            .prev_chain_membership
            .iter()
            .zip(&self.chain_membership)
            .position(|(old, new)| old != new);

        // If `prev_chain_membership` and `chain_membership` have different
        // lengths (passes added or removed) the zipped scan won't detect it,
        // so fall back to the boundary at the shorter length.
        let start = first_dirty.unwrap_or_else(|| {
            self.prev_chain_membership
                .len()
                .min(self.chain_membership.len())
        });

        if start == self.passes.len() {
            return; // no change — existing bundles are valid
        }

        // Rebuild from `start` to end.  Passes before `start` keep
        // their existing bundles.  Rebuild the cumulative base from
        // the surviving prefix.
        let mut base = crate::ResourceRegistry::empty();
        let (prefix, suffix) = self.passes.split_at_mut(start);
        for pass in prefix.iter_mut() {
            pass.publish(&mut base);
        }
        self.gpu_render_bundles.truncate(start);
        for pass in suffix.iter_mut() {
            let bundle = pass.build_gpu_render_bundle(&self.device, &base);
            self.gpu_render_bundles.push(bundle);
            pass.publish(&mut base);
        }
        self.last_bundle_chain_gen.truncate(start);
        drop(base);
        self.last_bundle_chain_gen
            .resize(self.passes.len(), self.chain_generation);
    }
}
