use crate::graph::executor::{format_bpp, format_name};
use crate::graph::resource::GraphTexturePool;
use crate::graph::{PipelineFormatCache, PipelineFormatSet, PipelineRegistry};
use crate::{
    PassContext, PrepareContext, Profiler, RenderFrameStorage, RenderPass, Result, SceneInput,
};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::{mpsc, Arc, Mutex, OnceLock};

use super::resource_lifetime::ResourceLifetime;
use super::scheduling::{compute_parallel_layers, CachedPass, PrePassAction};
use super::{DebugPassInfo, DebugResourceInfo, FrameDebugData};

/// Per-frame vectors reused across successful frames. They are drained before
/// they go back to the graph, so they keep capacity, never command buffers or
/// resources; a frame that fails drops its own.
#[derive(Default)]
struct FrameScratch {
    compute: Vec<wgpu::CommandBuffer>,
    graphics: Vec<Option<wgpu::CommandBuffer>>,
    segment_passes: Vec<&'static str>,
    reflected_groups: Vec<Vec<wgpu::BindGroup>>,
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

/// What kind of object a declared resource is, where the pass says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeclaredKind {
    /// Only a name (`reads()`, `writes()`): the declaration gives no kind.
    Unknown,
    Texture,
    Depth,
    /// `ResourceBuilder::write_buffer`.
    Buffer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeclaredResource {
    pub name: &'static str,
    pub kind: DeclaredKind,
}

/// What one pass declares, see [`RenderGraph::pass_declaration`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PassDeclaration {
    pub pass: &'static str,
    /// The Rust type path of the pass ([`RenderPass::type_name`]).
    pub type_name: &'static str,
    pub reads: Vec<DeclaredResource>,
    pub writes: Vec<DeclaredResource>,
    /// SceneDB buffer keys the pass consumes ([`RenderPass::scene_buffers`]).
    pub scene_buffers: Vec<&'static str>,
}

pub struct RenderGraph {
    /// The scheduled passes: the members of the current frame, in build
    /// order. Everything per pass (pipelines, caches, chains) is indexed by
    /// position in this list.
    pub(crate) passes: Vec<Box<dyn RenderPass>>,
    /// Build-order slot of each scheduled pass, so a repartition can merge
    /// members and dormant passes back into build order.
    member_slots: Vec<usize>,
    /// Passes that are not members of the frame (see
    /// [`RenderPass::is_member`]), with their build-order slots.
    dormant: Vec<(usize, Box<dyn RenderPass>)>,
    next_slot: usize,
    /// Per-frame membership answers, kept to avoid an allocation a frame.
    membership_scratch: Vec<bool>,
    /// The readers behind [`SceneContent`](crate::graph::SceneContent).
    scene_rows: crate::graph::SceneRows,
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
    frame_scratch: FrameScratch,
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
    /// Persistent secondary-view graphs, keyed by the pass that owns each
    /// capture (planar reflections, portals, editor probes, ...).
    secondary_graphs: HashMap<&'static str, RenderGraph>,
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
        groups: &mut Vec<wgpu::BindGroup>,
    ) -> Result<()> {
        let Some(pipeline) = self
            .reflected_pipelines
            .get(pass_index)
            .and_then(Option::as_ref)
        else {
            groups.clear();
            return Ok(());
        };
        if self.reflected_group_cache.len() < self.passes.len() {
            self.reflected_group_cache
                .resize_with(self.passes.len(), Vec::new);
        }
        crate::shader::reflection::reuse_or_create_reflected_bind_groups_into(
            self.passes[pass_index].name(),
            &pipeline.bindings,
            &pipeline.layouts,
            &pipeline.overrides,
            registry,
            &self.device,
            &mut self.reflected_group_cache[pass_index],
            groups,
        )
        .map_err(|error| {
            crate::Error::ResourceNotFound(format!("{}: {error}", self.passes[pass_index].name()))
        })
    }

    pub fn new(device: &std::sync::Arc<wgpu::Device>, queue: &wgpu::Queue) -> Self {
        Self {
            passes: Vec::new(),
            member_slots: Vec::new(),
            dormant: Vec::new(),
            next_slot: 0,
            membership_scratch: Vec::new(),
            scene_rows: Default::default(),
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
            frame_scratch: FrameScratch::default(),
            chain_membership: Vec::new(),
            prev_chain_membership: Vec::new(),
            chain_generation: 0,
            last_bundle_chain_gen: Vec::new(),
            locked: false,
            xr_active: false,
            pass_cache: Vec::new(),
            frame_count: 0,
            secondary_graphs: HashMap::new(),
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

    /// Construct an externally driven graph from the borrowed handles exposed
    /// to render passes. wgpu handles are cheap shared handles, so this keeps
    /// secondary graphs owned by a pass from having to retain a frontend's
    /// `Arc<Device>` just to create their executor.
    pub fn new_with_external_device_ref(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let device = std::sync::Arc::new(device.clone());
        let queue = queue.clone();
        let mut graph = Self::new(&device, &queue);
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

    /// [`resize`](Self::resize) to `width` x `height` at both resolutions.
    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.resize(crate::graph::RenderSize::uniform(width, height));
    }

    /// Resizes the graph in place: its textures are reallocated at the new
    /// resolutions and every pass, scheduled or dormant, gets
    /// [`RenderPass::on_resize`]. Passes, pipelines and the schedule are kept,
    /// so a resize costs the textures, not a rebuild.
    pub fn resize(&mut self, size: crate::graph::RenderSize) {
        if self.render_size() == size && self.resources_allocated {
            return;
        }
        let started = std::time::Instant::now();
        self.set_size(size);
        self.resize_pending = true;
        if self.locked {
            self.reallocate_textures();
        } else {
            self.allocate_transients();
        }
        let textures = started.elapsed();
        let device = self.device.clone();
        for pass in self.all_passes_mut() {
            pass.on_resize(&device, size);
        }
        let passes = started.elapsed() - textures;
        self.gpu_render_bundles.clear();
        self.rebuild_gpu_render_bundles();
        let elapsed = started.elapsed();
        if elapsed >= std::time::Duration::from_millis(100) {
            eprintln!(
                "[RenderGraph] slow resize to {:?} (output {:?}) took {:.1} ms (textures {:.1} ms,                  pass callbacks {:.1} ms, bundles {:.1} ms)",
                size.internal,
                size.output,
                elapsed.as_secs_f64() * 1000.0,
                textures.as_secs_f64() * 1000.0,
                passes.as_secs_f64() * 1000.0,
                (elapsed - textures - passes).as_secs_f64() * 1000.0,
            );
        }
    }

    fn set_size(&mut self, size: crate::graph::RenderSize) {
        [self.internal_w, self.internal_h] = size.internal;
        [self.output_w, self.output_h] = size.output;
    }

    /// Reallocates the graph textures of a locked graph at its current size,
    /// keeping its chains and passes (the textures' alias groups depend only
    /// on the schedule).
    fn reallocate_textures(&mut self) {
        self.pool.clear();
        // Cached reflected groups may bind the textures just dropped.
        self.reflected_group_cache.clear();
        self.collect_declarations();
        for rl in self.resources.values_mut() {
            rl.chain_local = self
                .subpass_chains
                .iter()
                .any(|c| c.start <= rl.first_write_pass && rl.last_read_pass < c.end);
        }
        self.assign_chain_aware_alias_groups();
        self.allocate_textures();
        self.resources_allocated = true;
        // Recorded units bind the textures just replaced.
    }

    pub fn init_transients(&mut self, width: u32, height: u32) {
        self.set_size(crate::graph::RenderSize::uniform(width, height));
        self.allocate_transients();
    }

    /// Allocates the graph textures of a graph that is not locked.
    fn allocate_transients(&mut self) {
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
        self.member_slots.push(self.next_slot);
        self.next_slot += 1;
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
        let size = self.render_size();
        if let Some(pass) = self.passes.last_mut() {
            pass.on_resize(&self.device, size);
        }
        self.schedule();
        self.resize_pending = true;
    }

    /// The first pass of type `T`, scheduled or dormant (see
    /// [`RenderPass::is_member`]): configuration reaches a pass whether or not
    /// the current scene uses it.
    pub fn find_pass_mut<T: RenderPass + 'static>(&mut self) -> Option<&mut T> {
        let slot = self.first_slot_of(TypeId::of::<T>())?;
        self.pass_at_slot_mut(slot)?
            .as_any_mut()
            .downcast_mut::<T>()
    }

    pub fn find_pass<T: RenderPass + 'static>(&self) -> Option<&T> {
        let slot = self.first_slot_of(TypeId::of::<T>())?;
        self.all_passes()
            .find(|(s, _)| *s == slot)
            .and_then(|(_, pass)| pass.as_any().downcast_ref::<T>())
    }

    /// Build-order slot of the first pass of `type_id`, scheduled or dormant.
    fn first_slot_of(&self, type_id: TypeId) -> Option<usize> {
        let member = self
            .pass_index_map
            .get(&type_id)
            .map(|&index| self.member_slots[index]);
        let dormant = self
            .dormant
            .iter()
            .find(|(_, pass)| pass.as_any().type_id() == type_id)
            .map(|(slot, _)| *slot);
        match (member, dormant) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn pass_at_slot_mut(&mut self, slot: usize) -> Option<&mut Box<dyn RenderPass>> {
        if let Some(index) = self.member_slots.iter().position(|&s| s == slot) {
            return self.passes.get_mut(index);
        }
        self.dormant
            .iter_mut()
            .find(|(s, _)| *s == slot)
            .map(|(_, pass)| pass)
    }

    /// Every pass with its build-order slot: the scheduled ones, then the
    /// dormant ones.
    fn all_passes(&self) -> impl Iterator<Item = (usize, &Box<dyn RenderPass>)> {
        self.member_slots
            .iter()
            .copied()
            .zip(self.passes.iter())
            .chain(self.dormant.iter().map(|(slot, pass)| (*slot, pass)))
    }

    fn all_passes_mut(&mut self) -> impl Iterator<Item = &mut Box<dyn RenderPass>> {
        self.passes
            .iter_mut()
            .chain(self.dormant.iter_mut().map(|(_, pass)| pass))
    }

    /// Number of passes that are not members of the current frame.
    pub fn dormant_count(&self) -> usize {
        self.dormant.len()
    }

    /// Names of the scheduled passes, in execution order.
    pub fn member_names(&self) -> Vec<&'static str> {
        self.passes.iter().map(|pass| pass.name()).collect()
    }

    /// Preserve opt-in streaming state when rebuilding a graph on this device.
    /// New pass configuration and graph resources remain authoritative.
    /// Dormant passes inherit too, from scheduled or dormant predecessors.
    pub fn inherit_persistent_state(&mut self, previous: &mut RenderGraph) {
        let size = self.render_size();
        let device = self.device.clone();
        let mut used = std::collections::HashSet::new();
        let mut olds: Vec<&mut Box<dyn RenderPass>> = previous.all_passes_mut().collect();
        for pass in self.all_passes_mut() {
            for (index, old) in olds.iter_mut().enumerate() {
                if !used.contains(&index)
                    && pass.name() == old.name()
                    && pass.as_any().type_id() == old.as_any().type_id()
                    && pass.inherit_persistent_state(&mut ***old)
                {
                    used.insert(index);
                    pass.on_resize(&device, size);
                    break;
                }
            }
        }
        drop(olds);

        // Same scene: start with the members the old graph had, and its row
        // readers, rather than a readback later.
        self.scene_rows = std::mem::take(&mut previous.scene_rows);
        let members: Vec<(TypeId, &'static str)> = previous
            .passes
            .iter()
            .map(|pass| (pass.as_any().type_id(), pass.name()))
            .collect();
        let join: Vec<bool> = self
            .dormant
            .iter()
            .map(|(_, pass)| members.contains(&(pass.as_any().type_id(), pass.name())))
            .collect();
        if join.contains(&true) {
            let stay = vec![true; self.passes.len()];
            self.repartition(&stay, &join, true);
            if self.locked {
                self.reschedule_members();
            }
        }
    }

    /// The graph's current resolutions.
    pub fn render_size(&self) -> crate::graph::RenderSize {
        crate::graph::RenderSize::new(
            [self.internal_w, self.internal_h],
            [self.output_w, self.output_h],
        )
    }

    /// Whether this graph has completed [`Self::lock`] and can execute.
    pub fn is_locked(&self) -> bool {
        self.locked
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
        for pass in self.all_passes_mut() {
            pass.set_editor_mode(enabled);
        }
    }

    /// Broadcast generic per-frame projection inputs to every pass.
    ///
    /// Passes interpret only the fields they own; the graph and renderer
    /// facade never downcast into concrete pass implementations.
    pub fn set_frame_inputs(&mut self, inputs: &crate::RenderFrameInputs<'_>) {
        for pass in self.all_passes_mut() {
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
            let size = self.render_size();
            if self.locked {
                self.passes[index].on_resize(&self.device, size);
                self.locked = false;
                self.schedule();
                self.resize_pending = true;
            } else if self.resources_allocated {
                self.passes[index].on_resize(&self.device, size);
                self.allocate_transients();
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
            if self.reflected_pipelines[live].is_some()
                != replacement.reflected_pipelines[new].is_some()
            {
                return Err(format!(
                    "pass '{}' changed its reflected bindings",
                    old.name()
                ));
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
            let mut fresh = incoming[new]
                .take()
                .expect("check_swap rejects repeated picks");
            if fresh.inherit_persistent_state(&mut *self.passes[live]) {
                fresh.on_resize(&self.device, self.render_size());
            }
            self.pipeline_cache
                .invalidate_pass(self.passes[live].name());
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

    /// Every pass of type `T`, scheduled or dormant.
    pub fn iter_passes_mut<T: RenderPass + 'static>(&mut self) -> impl Iterator<Item = &mut T> {
        self.all_passes_mut()
            .filter_map(|p| p.as_any_mut().downcast_mut::<T>())
    }

    /// The debug views of every pass, dormant ones included: a view joins its
    /// pass when it is selected.
    pub fn collect_debug_views(&self) -> Vec<crate::DebugViewDescriptor> {
        self.all_passes()
            .flat_map(|(_, p)| p.debug_views().iter().copied())
            .collect()
    }

    pub fn collect_debug_view_legends(&self) -> Vec<(u32, crate::DebugViewLegend)> {
        self.all_passes()
            .flat_map(|(_, pass)| {
                pass.debug_views().iter().filter_map(|view| {
                    pass.debug_view_legend(view.debug_mode)
                        .map(|legend| (view.debug_mode, legend))
                })
            })
            .collect()
    }

    pub fn collect_debug_view_categories(&self) -> Vec<(u32, &'static str)> {
        self.all_passes()
            .flat_map(|(_, pass)| {
                pass.debug_views()
                    .iter()
                    .map(|view| (view.debug_mode, pass.debug_view_category(view.debug_mode)))
                    .filter(|(_, category)| !category.is_empty())
            })
            .collect()
    }

    /// Propagate a renderer-wide debug mode change to every pass.
    pub fn set_debug_mode(&mut self, mode: u32) {
        for pass in self.all_passes_mut() {
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

    /// Resources declared with [`declare_external_input`](Self::declare_external_input):
    /// supplied by the host each frame rather than written by a pass.
    pub fn external_input_names(&self) -> Vec<&'static str> {
        self.external_inputs.iter().copied().collect()
    }

    /// Number of passes added so far.
    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    /// The size passes are prepared and recorded at, set by
    /// [`lock`](Self::lock).
    pub fn internal_size(&self) -> (u32, u32) {
        (self.internal_w, self.internal_h)
    }

    /// What pass `index` declares, for tools: the resources it reads and
    /// writes (from `reads()`/`writes()` and `declare_resources`) with their
    /// kind where the pass declares one, and the SceneDB buffers it consumes.
    pub fn pass_declaration(&self, index: usize) -> Option<PassDeclaration> {
        let pass = self.passes.get(index)?;
        let mut declaration = PassDeclaration {
            pass: pass.name(),
            type_name: pass.type_name(),
            reads: Vec::new(),
            writes: Vec::new(),
            scene_buffers: pass.scene_buffers().to_vec(),
        };
        let mut push = |list: &mut Vec<DeclaredResource>,
                        name: &'static str,
                        kind: DeclaredKind| {
            match list.iter_mut().find(|existing| existing.name == name) {
                // A precise kind beats the unknown one a bare name implies.
                Some(existing) if existing.kind == DeclaredKind::Unknown => existing.kind = kind,
                Some(_) => {}
                None => list.push(DeclaredResource { name, kind }),
            }
        };
        for &name in pass.reads() {
            push(&mut declaration.reads, name, DeclaredKind::Unknown);
        }
        for &name in pass.writes() {
            push(&mut declaration.writes, name, DeclaredKind::Unknown);
        }
        let mut builder = crate::graph::ResourceBuilder::new();
        pass.declare_resources(&mut builder);
        for resource in builder.declarations() {
            match resource.access {
                crate::graph::ResourceAccess::Read => {
                    push(&mut declaration.reads, resource.name, DeclaredKind::Unknown)
                }
                crate::graph::ResourceAccess::Write => {
                    let kind = match resource.format {
                        None => DeclaredKind::Buffer,
                        Some(crate::graph::ResourceFormat::Depth32Float) => DeclaredKind::Depth,
                        Some(_) => DeclaredKind::Texture,
                    };
                    // A group member is published under the group's name, which
                    // is what readers ask for.
                    push(
                        &mut declaration.writes,
                        resource.group.unwrap_or(resource.name),
                        kind,
                    );
                }
            }
        }
        Some(declaration)
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
        // The frame's members: passes join and leave with the scene's content
        // before anything of the frame is recorded.
        if self.update_membership(scene) {
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: reschedule for new members");
            self.reschedule_members();
        }

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
        let encoders_scope =
            profiling::ProfileScope::new_static("RenderGraph: create command encoders");
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
        let mut reflected_groups = std::mem::take(&mut self.frame_scratch.reflected_groups);
        reflected_groups.resize_with(self.passes.len(), Vec::new);
        {
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: create_reflected_groups");
            for (pass_index, groups) in reflected_groups.iter_mut().enumerate() {
                self.create_reflected_groups(pass_index, registry, groups)?;
            }
        }
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
        let mut chain_patch: smallvec::SmallVec<
            [Option<wgpu::RenderPassColorAttachment<'static>>; 8],
        > = smallvec::SmallVec::new();

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
        // `graphics_segments` is indexed by segment; `None` while its finish
        // is still in flight.
        let FrameScratch {
            compute: mut compute_segments,
            graphics: mut graphics_segments,
            mut segment_passes,
            reflected_groups: _,
        } = std::mem::take(&mut self.frame_scratch);
        let mut segment_recording = std::time::Duration::ZERO;
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
                    let passes = if finish_breakdown {
                        std::mem::take(&mut segment_passes)
                    } else {
                        segment_passes.clear();
                        Vec::new()
                    };
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
                        compute_segments.push(done_compute.finish());
                        let compute = start.elapsed();
                        let start = std::time::Instant::now();
                        graphics_segments[index] = Some(done_graphics.finish());
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
                                graphics_segments[index] = Some(done_graphics.finish());
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
                    world_origin: scene.world_origin(),
                    scene_buffers: scene.scene_buffers(),
                    registry: &*registry,
                    resize: resized_this_frame,
                    width: self.internal_w,
                    height: self.internal_h,
                    delta_time: self.delta_time,
                    time: self.frame_time,
                    time_delta: self.frame_time_delta,
                    owns_device: self.owns_device,
                };
                pass.declare_frame_demands(&plan_ctx, &mut self.frame_demands);
            }
            // SAFETY: `frame_demands` is graph-owned and not modified again
            // until the next frame's `execute_with_registry`; the registry
            // entry is frame-scoped (same lifetime bridge as `publish`).
            let demands: &crate::FrameDemands = unsafe { &*std::ptr::addr_of!(self.frame_demands) };
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

            // prepare() and the graph-owned outputs it publishes, before
            // execute().
            macro_rules! prepare_pass {
                ($pass:expr, $pass_index:expr) => {{
                    let _scope = self.profiler.scope($pass.name());
                    let prepare_ctx = PrepareContext {
                        device: scene.device(),
                        queue: scene.queue(),
                        frame_num: scene.frame_count(),
                        camera: scene.camera(),
                        camera_data: scene.camera_data(),
                        camera_generation: scene.camera_generation(),
                        world_origin: scene.world_origin(),
                        scene_buffers: scene.scene_buffers(),
                        registry: &*registry,
                        resize: resized_this_frame,
                        width: self.internal_w,
                        height: self.internal_h,
                        delta_time: self.delta_time,
                        time: self.frame_time,
                        time_delta: self.frame_time_delta,
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
                                // A declared external input may also be the
                                // attachment supplied to a writer (for
                                // example an imported camera target). Keep the
                                // seeded view in that case so the graph routes
                                // the caller-owned target through the writer.
                                let external_view = self.external_inputs.contains(name)
                                    && registry
                                        .get(crate::ResourceKey::<&wgpu::TextureView>::new(name))
                                        .is_some();
                                if !external_view {
                                    registry.route_named_texture(name, view, "Graph");
                                }
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
                }};
            }
            for (pass_index, pass) in self.passes.iter_mut().enumerate() {
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
                        self.profiler
                            .begin_gpu_pass(&mut compute_encoder, pass_name);
                        let mut pass_encoder = encoder.begin_render_pass(&desc);
                        pass_encoder.execute_bundles(std::iter::once(bundle));
                    } else {
                        self.profiler.begin_gpu_pass(&mut encoder, pass_name);
                        self.profiler
                            .begin_gpu_pass(&mut compute_encoder, pass_name);
                        let mut ctx = PassContext {
                            encoder_ptr: &mut encoder as *mut _,
                            queue: scene.queue(),
                            compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                            target,
                            depth,
                            camera: scene.camera(),
                            camera_data: scene.camera_data(),
                            camera_generation: scene.camera_generation(),
                            world_origin: scene.world_origin(),
                            scene_buffers: scene.scene_buffers(),
                            profiler: &mut self.profiler,
                            frame_num: scene.frame_count(),
                            width: self.internal_w,
                            height: self.internal_h,
                            device: scene.device(),
                            secondary_graphs: &mut self.secondary_graphs as *mut _,
                            registry: &*registry,
                            owns_device: self.owns_device,
                            resource_pool: &self.pool,
                            subpass_index: 0,
                            subpass_count: 0,
                            active_render_pass: None,
                            active_compute_pass: None,
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
                    self.profiler
                        .record_external_cpu_timing(pass_name, recorded);
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
                        self.profiler
                            .begin_gpu_pass(&mut compute_encoder, pass_name);
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
                            world_origin: scene.world_origin(),
                            scene_buffers: scene.scene_buffers(),
                            profiler: &mut self.profiler,
                            frame_num: scene.frame_count(),
                            width: self.internal_w,
                            height: self.internal_h,
                            device: scene.device(),
                            secondary_graphs: &mut self.secondary_graphs as *mut _,
                            registry: &*registry,
                            owns_device: self.owns_device,
                            resource_pool: &self.pool,
                            subpass_index: c.subpass_index,
                            subpass_count: c.subpass_count,
                            active_render_pass: chain_rp
                                .as_mut()
                                .map(|rp| &mut **rp as *mut _ as *mut _),
                            active_compute_pass: None,
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
                                world_origin: scene.world_origin(),
                                scene_buffers: scene.scene_buffers(),
                                profiler: &mut self.profiler,
                                frame_num: scene.frame_count(),
                                width: self.internal_w,
                                height: self.internal_h,
                                device: scene.device(),
                                secondary_graphs: &mut self.secondary_graphs as *mut _,
                                registry: &*registry,
                                owns_device: self.owns_device,
                                resource_pool: &self.pool,
                                subpass_index: 0,
                                subpass_count: 0,
                                active_render_pass: Some(&mut rp as *mut _ as *mut _),
                                active_compute_pass: None,
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
                        self.profiler
                            .begin_gpu_pass(&mut compute_encoder, pass_name);
                        timed_compute = true;
                    } else {
                        close_chain!();
                        self.profiler.begin_gpu_pass(&mut encoder, pass_name);
                        self.profiler
                            .begin_gpu_pass(&mut compute_encoder, pass_name);
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
                        world_origin: scene.world_origin(),
                        scene_buffers: scene.scene_buffers(),
                        profiler: &mut self.profiler,
                        frame_num: scene.frame_count(),
                        width: self.internal_w,
                        height: self.internal_h,
                        device: scene.device(),
                        secondary_graphs: &mut self.secondary_graphs as *mut _,
                        registry: &*registry,
                        owns_device: self.owns_device,
                        resource_pool: &self.pool,
                        subpass_index: 0,
                        subpass_count: 0,
                        active_render_pass: None,
                        active_compute_pass: None,
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
                self.profiler
                    .record_external_cpu_timing(pass_name, recorded);
            }
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
                        Ok((index, Ok(buffer))) => graphics_segments[index] = Some(buffer),
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
            command_buffers.extend(compute_last);
            command_buffers.extend(
                graphics_segments.drain(..).map(|segment| {
                    segment.expect("every graphics segment is finished before submit")
                }),
            );
            command_buffers.push(graphics_last);
            command_buffers
        };
        let mut command_buffers = command_buffers;
        let submission_index = {
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: queue.submit (graph)");
            scene.queue().submit(command_buffers.drain(..))
        };
        segment_passes.clear();
        for groups in &mut reflected_groups {
            groups.clear();
        }
        self.frame_scratch = FrameScratch {
            compute: command_buffers,
            graphics: graphics_segments,
            segment_passes,
            reflected_groups,
        };
        {
            #[cfg(not(target_arch = "wasm32"))]
            profiling::profile_scope!("RenderGraph: upload::finish_frame");
            crate::upload::finish_frame();
        }

        #[cfg(not(target_arch = "wasm32"))]
        let readback_scope =
            profiling::ProfileScope::new_static("RenderGraph: read GPU timestamps");
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

        // Once the first pipelines have been built, save the driver cache so
        // other processes (and the next run) skip compiling them. No-op unless
        // persistence was enabled.
        if self.frame_count == 300 {
            let _ = self.pipeline_cache.persist();
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

    /// [`lock_sized`](Self::lock_sized) with both resolutions `width` x `height`.
    pub fn lock(&mut self, width: u32, height: u32) {
        self.lock_sized(crate::graph::RenderSize::uniform(width, height));
    }

    /// Finishes the graph after every pass has been added: decides the first
    /// members (with no scene yet, so passes that need scene content start
    /// dormant; see [`RenderPass::is_member`]), then schedules them and
    /// allocates their textures at `size`.
    pub fn lock_sized(&mut self, size: crate::graph::RenderSize) {
        assert!(!self.locked, "RenderGraph::lock() called twice");
        self.set_size(size);
        let mut content = self.scene_rows.content(None, None, None, self.frame_count);
        let stay: Vec<bool> = self
            .passes
            .iter_mut()
            .map(|pass| pass.is_member(&mut content))
            .collect();
        let join: Vec<bool> = self
            .dormant
            .iter_mut()
            .map(|(_, pass)| pass.is_member(&mut content))
            .collect();
        // Nothing has run yet, so nothing leaves.
        self.repartition(&stay, &join, false);
        self.schedule();
    }

    /// Schedules the graph's new members, keeping every graph texture whose
    /// resource is unchanged (same views, so bind groups built over them stay
    /// valid). If one had to change, every pass is told as on a resize.
    fn reschedule_members(&mut self) {
        let started = std::time::Instant::now();
        let before = self.pool.views();
        self.pool.retire();
        self.locked = false;
        self.schedule_reporting(false);
        let changed = before
            .iter()
            .any(|(name, view)| self.pool.get_view(name).is_some_and(|now| now != view));
        if changed {
            let size = self.render_size();
            let device = self.device.clone();
            for pass in self.all_passes_mut() {
                pass.on_resize(&device, size);
            }
            self.resize_pending = true;
        }
        eprintln!(
            "[RenderGraph] {} scheduled, {} dormant ({:.2} ms{}): {}",
            self.passes.len(),
            self.dormant.len(),
            started.elapsed().as_secs_f64() * 1000.0,
            if changed { ", textures moved" } else { "" },
            self.member_names().join(", ")
        );
    }

    /// [`schedule_reporting`](Self::schedule_reporting), printing the schedule.
    fn schedule(&mut self) {
        self.schedule_reporting(true);
    }

    /// Asks every pass whether it is a member of this frame and moves those
    /// that changed. True when the members changed (the graph must be
    /// scheduled again).
    fn update_membership(&mut self, scene: &dyn SceneInput) -> bool {
        let mut content = self.scene_rows.content(
            Some(scene.device().as_ref()),
            Some(scene.queue().as_ref()),
            Some(scene.scene_buffers()),
            self.frame_count,
        );
        let mut changed = false;
        self.membership_scratch.clear();
        for pass in self.passes.iter_mut() {
            let stays = pass.is_member(&mut content);
            changed |= !stays;
            self.membership_scratch.push(stays);
        }
        let members = self.passes.len();
        for (_, pass) in self.dormant.iter_mut() {
            let joins = pass.is_member(&mut content);
            changed |= joins;
            self.membership_scratch.push(joins);
        }
        if !changed {
            return false;
        }
        let scratch = std::mem::take(&mut self.membership_scratch);
        self.repartition(&scratch[..members], &scratch[members..], true);
        self.membership_scratch = scratch;
        true
    }

    /// Moves passes between the schedule and the dormant list: scheduled pass
    /// `i` stays when `stay[i]`, dormant pass `j` joins when `join[j]`. The
    /// schedule keeps build order. With `leave`, passes that leave get
    /// [`RenderPass::on_leave`]. The caller schedules the graph again.
    fn repartition(&mut self, stay: &[bool], join: &[bool], leave: bool) {
        let members = std::mem::take(&mut self.passes);
        let slots = std::mem::take(&mut self.member_slots);
        let dormant = std::mem::take(&mut self.dormant);
        let mut all: Vec<(usize, Box<dyn RenderPass>, bool)> =
            Vec::with_capacity(members.len() + dormant.len());
        for ((slot, mut pass), &stays) in slots.into_iter().zip(members).zip(stay) {
            if leave && !stays {
                pass.on_leave(&self.device);
            }
            all.push((slot, pass, stays));
        }
        for ((slot, pass), &joins) in dormant.into_iter().zip(join) {
            all.push((slot, pass, joins));
        }
        all.sort_by_key(|(slot, _, _)| *slot);
        self.pass_index_map.clear();
        for (slot, pass, member) in all {
            if member {
                self.pass_index_map
                    .entry(pass.as_any().type_id())
                    .or_insert(self.passes.len());
                self.passes.push(pass);
                self.member_slots.push(slot);
            } else {
                self.dormant.push((slot, pass));
            }
        }
        // Everything per pass is rebuilt by the next schedule.
        self.gpu_render_bundles = (0..self.passes.len()).map(|_| None).collect();
        self.pass_cache.clear();
        self.chain_membership.clear();
        self.prev_chain_membership.clear();
        self.last_bundle_chain_gen.clear();
    }

    /// Schedules the current members: their declared resources, chains,
    /// pipelines and textures. Sizes are the graph's current ones. `report`
    /// prints the schedule.
    fn schedule_reporting(&mut self, report: bool) {
        assert!(!self.locked, "RenderGraph::lock() called twice");
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
                        let names: Vec<&'static str> = self.passes[c.clone()]
                            .iter()
                            .map(|pass| pass.name())
                            .collect();
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

        self.rebuild_gpu_render_bundles_incremental();
        self.pool.drop_retired();

        if report {
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
