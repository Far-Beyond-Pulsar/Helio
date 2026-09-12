use crate::graph::executor::{format_bpp, format_name};
use crate::graph::resource::GraphTexturePool;
use crate::graph::{PipelineFormatCache, PipelineFormatSet, PipelineRegistry};
use crate::{PassContext, PrepareContext, Profiler, RenderPass, Result, SceneInput};
use std::any::TypeId;
use std::collections::HashMap;
use std::sync::Arc;

use super::resource_lifetime::ResourceLifetime;
use super::scheduling::{compute_parallel_layers, CachedPass, PrePassAction};
use super::{DebugPassInfo, DebugResourceInfo, FrameDebugData};

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
    pub(crate) pre_pass_actions: Vec<Vec<PrePassAction>>,
    pub(crate) device: std::sync::Arc<wgpu::Device>,
    pub(crate) internal_w: u32,
    pub(crate) internal_h: u32,
    pub(crate) output_w: u32,
    pub(crate) output_h: u32,
    delta_time: f32,
    owns_device: bool,
    gpu_render_bundles: Vec<Option<wgpu::RenderBundle>>,
    resources_allocated: bool,
    pub(crate) subpass_chains: Vec<std::ops::Range<usize>>,
    pub(crate) parallel_layers: Vec<Vec<usize>>,
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
    /// Set by [`set_render_size`](Self::set_render_size) and consumed only
    /// after the first successful frame at the new size. Passes use this
    /// one-frame pulse through [`PrepareContext::resize`] to rebuild resources
    /// that depend on graph-owned texture dimensions.
    resize_pending: bool,
    /// Opaque storage for cross-crate data (e.g. a GraphRebuilder).
    /// Set by graph builders, consumed by the Renderer on construction.
    graph_data: Option<Box<dyn std::any::Any + Send + Sync>>,
    /// Resource names registered via [`declare_external_input`](Self::declare_external_input) —
    /// resources supplied by the host rather than written by any pass in the
    /// graph. `validate_dependencies` treats every name in this set as
    /// available from pass index 0. See `docs/helio_3_0_spec.md` §6.
    external_inputs: std::collections::HashSet<&'static str>,
    /// Worker timestamp profilers whose deferred readbacks have not completed
    /// yet. This is populated for externally-owned devices whose host drives
    /// device polling.
    pending_worker_profilers: Vec<Profiler>,
}
impl RenderGraph {
    fn create_reflected_groups(
        &self,
        pass_index: usize,
        registry: &libhelio::ResourceRegistry<'_>,
    ) -> Result<Vec<wgpu::BindGroup>> {
        let Some(pipeline) = self
            .reflected_pipelines
            .get(pass_index)
            .and_then(Option::as_ref)
        else {
            return Ok(Vec::new());
        };
        crate::shader::create_reflected_bind_groups_with_layouts(
            self.passes[pass_index].name(),
            &pipeline.bindings,
            &pipeline.layouts,
            &pipeline.overrides,
            registry,
            &self.device,
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
            pre_pass_actions: Vec::new(),
            device: device.clone(),
            internal_w: 0,
            internal_h: 0,
            output_w: 0,
            output_h: 0,
            delta_time: 0.0,
            owns_device: true,
            gpu_render_bundles: Vec::new(),
            resources_allocated: false,
            subpass_chains: Vec::new(),
            parallel_layers: Vec::new(),
            chain_membership: Vec::new(),
            prev_chain_membership: Vec::new(),
            chain_generation: 0,
            last_bundle_chain_gen: Vec::new(),
            locked: false,
            xr_active: false,
            pass_cache: Vec::new(),
            frame_count: 0,
            resize_pending: false,
            graph_data: None,
            external_inputs: std::collections::HashSet::new(),
            pending_worker_profilers: Vec::new(),
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

    pub fn find_pass_mut<T: RenderPass + 'static>(&mut self) -> Option<&mut T> {
        let idx = *self.pass_index_map.get(&TypeId::of::<T>())?;
        self.passes[idx].as_any_mut().downcast_mut::<T>()
    }

    pub fn find_pass<T: RenderPass + 'static>(&self) -> Option<&T> {
        let idx = *self.pass_index_map.get(&TypeId::of::<T>())?;
        self.passes[idx].as_any().downcast_ref::<T>()
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

    /// Replace the pass at `index` with a new one.
    pub fn replace_pass_at(&mut self, index: usize, pass: Box<dyn RenderPass>) {
        if index < self.passes.len() {
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
        let frame_resources = libhelio::FrameResources::empty();
        let mut registry = libhelio::ResourceRegistry::empty();
        self.execute_with_resources(scene, target, depth, &frame_resources, &mut registry)
    }

    pub fn execute_with_frame_resources(
        &mut self,
        scene: &dyn SceneInput,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        frame_resources: &libhelio::FrameResources<'_>,
    ) -> Result<wgpu::SubmissionIndex> {
        let mut registry = libhelio::ResourceRegistry::empty();
        self.execute_with_resources(scene, target, depth, frame_resources, &mut registry)
    }

    /// Records graphs with no fused render chains in dependency-layer order.
    /// Each pass receives private encoders and a private profiler, while
    /// publication back into the frame contract remains deterministic and
    /// occurs on the caller thread after the layer joins.
    fn execute_parallel_layers<'a, 'b>(
        &mut self,
        scene: &dyn SceneInput,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        visible: &mut libhelio::FrameResources<'a>,
        registry: &mut libhelio::ResourceRegistry<'b>,
        reflected_groups: &[Vec<wgpu::BindGroup>],
        resized_this_frame: bool,
    ) -> Result<(
        Vec<wgpu::CommandBuffer>,
        Vec<(&'static str, std::time::Duration)>,
        Vec<Profiler>,
    )> {
        let mut command_buffers = Vec::new();
        let mut cpu_timings = Vec::new();
        let mut worker_profilers = Vec::new();
        let layers = self.parallel_layers.clone();
        let internal_w = self.internal_w;
        let internal_h = self.internal_h;
        let delta_time = self.delta_time;
        let owns_device = self.owns_device;
        let reflected_pipelines = &self.reflected_pipelines;
        let (passes, pre_pass_actions) = (&mut self.passes, &self.pre_pass_actions);
        let pipeline_registries = &self.pipeline_registries;
        let pool = &self.pool;
        let pipeline_cache = &self.pipeline_cache;
        for layer in layers {
            for &pass_index in &layer {
                let actions_ptr = pre_pass_actions
                    .get(pass_index)
                    .map(|actions| actions as *const Vec<PrePassAction>);
                if let Some(actions_ptr) = actions_ptr {
                    // The action list is graph-owned and immutable for the
                    // duration of execution; using its raw pointer prevents
                    // the borrow from spanning the separate pass mutation.
                    for action in unsafe { &*actions_ptr } {
                        match action {
                            PrePassAction::Route { name, view } => {
                                visible.route_named_texture(name, view, "Graph");
                            }
                            PrePassAction::Group { name, members } => {
                                let views: Vec<&wgpu::TextureView> =
                                    members.iter().map(|(_, view)| view).collect();
                                (&*passes[pass_index]).publish_group(*name, &views, visible);
                            }
                        }
                    }
                }
                {
                    let prepare_ctx = PrepareContext {
                        device: scene.device(),
                        queue: scene.queue(),
                        frame_num: scene.frame_count(),
                        scene: scene.resources(),
                        scene_buffers: scene.scene_buffers(),
                        frame_resources: visible,
                        registry: &*registry,
                        resize: resized_this_frame,
                        width: internal_w,
                        height: internal_h,
                        delta_time,
                    };
                    passes[pass_index].prepare(&prepare_ctx)?;
                }
            }

            let visible_ref: &libhelio::FrameResources<'_> = &*visible;
            let registry_ref: &libhelio::ResourceRegistry<'_> = &*registry;
            let device = scene.device().clone();
            let queue = scene.queue().clone();
            let scene_resources = scene.resources();
            let scene_buffers = scene.scene_buffers();
            let pipelines = pipeline_registries;
            let width = internal_w;
            let height = internal_h;
            let frame_num = scene.frame_count();
            let handles = std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(layer.len());
                for &pass_index in &layer {
                    // Every index in a computed layer is unique. Converting
                    // the disjoint mutable reference to an integer lets the
                    // scoped worker carry it without requiring a global lock;
                    // the exclusive graph borrow and unique layer indices are
                    // the safety proof for this narrow boundary.
                    let pass_address =
                        (&mut passes[pass_index]) as *mut Box<dyn RenderPass> as usize;
                    let pipeline_registry = &pipelines[pass_index];
                    let worker_device = device.clone();
                    let worker_queue = queue.clone();
                    let worker_scene = scene_resources;
                    handles.push(scope.spawn(move || {
                        let mut encoder =
                            worker_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Helio Parallel Render Pass"),
                            });
                        let mut compute_encoder =
                            worker_device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                                label: Some("Helio Parallel Compute Pass"),
                            });
                        let mut local_profiler = Profiler::new(&worker_device, &worker_queue);
                        let cpu_start = std::time::Instant::now();
                        let pass = unsafe { &mut *(pass_address as *mut Box<dyn RenderPass>) };
                        let pass_name = pass.name();
                        local_profiler.begin_gpu_pass(&mut compute_encoder, pass_name);
                        if let Some(desc) =
                            pass.render_pass_descriptor_with_pool(target, depth, visible_ref, pool)
                        {
                            let attachments: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> =
                                desc.color_attachments.iter().cloned().collect();
                            let standalone_desc = wgpu::RenderPassDescriptor {
                                label: desc.label,
                                color_attachments: &attachments,
                                depth_stencil_attachment: desc.depth_stencil_attachment,
                                timestamp_writes: desc.timestamp_writes,
                                occlusion_query_set: desc.occlusion_query_set,
                                multiview_mask: desc.multiview_mask,
                            };
                            let encoder_ptr: *mut wgpu::CommandEncoder = &mut encoder;
                            let mut render_pass = encoder.begin_render_pass(&standalone_desc);
                            let mut ctx = PassContext {
                                encoder_ptr,
                                compute_encoder_ptr: &mut compute_encoder,
                                target,
                                depth,
                                scene: worker_scene,
                                scene_buffers,
                                profiler: &mut local_profiler,
                                frame_num,
                                width,
                                height,
                                device: &worker_device,
                                resources: visible_ref,
                                registry: registry_ref,
                                owns_device,
                                resource_pool: pool,
                                subpass_index: 0,
                                subpass_count: 0,
                                active_render_pass: Some(&mut render_pass as *mut _ as *mut _),
                                active_compute_pass: None,
                                pipeline_cache,
                                pipelines: pipeline_registry,
                                reflected_bind_groups: &reflected_groups[pass_index],
                                reflected_pipeline: reflected_pipelines[pass_index].as_ref(),
                                #[cfg(debug_assertions)]
                                chain_transparent: false,
                            };
                            ctx.apply_reflected_bind_groups();
                            pass.execute(&mut ctx)?;
                        } else {
                            let mut ctx = PassContext {
                                encoder_ptr: &mut encoder,
                                compute_encoder_ptr: &mut compute_encoder,
                                target,
                                depth,
                                scene: worker_scene,
                                scene_buffers,
                                profiler: &mut local_profiler,
                                frame_num,
                                width,
                                height,
                                device: &worker_device,
                                resources: visible_ref,
                                registry: registry_ref,
                                owns_device,
                                resource_pool: pool,
                                subpass_index: 0,
                                subpass_count: 0,
                                active_render_pass: None,
                                active_compute_pass: None,
                                pipeline_cache,
                                pipelines: pipeline_registry,
                                reflected_bind_groups: &reflected_groups[pass_index],
                                reflected_pipeline: reflected_pipelines[pass_index].as_ref(),
                                #[cfg(debug_assertions)]
                                chain_transparent: false,
                            };
                            ctx.apply_reflected_bind_groups();
                            pass.execute(&mut ctx)?;
                        }
                        local_profiler.end_gpu_pass(&mut compute_encoder, pass_name);
                        // The profiler owns this query set and resolve buffer;
                        // resolve it into the worker command stream before the
                        // encoder is finished. The parent merges the samples
                        // after submission, when wgpu permits readback.
                        local_profiler.resolve_gpu_queries(&mut compute_encoder, frame_num);
                        Ok::<_, crate::Error>((
                            encoder.finish(),
                            compute_encoder.finish(),
                            (pass.name(), cpu_start.elapsed()),
                            local_profiler,
                        ))
                    }));
                }
                handles
                    .into_iter()
                    .map(|handle| {
                        handle.join().map_err(|_| {
                            crate::Error::InvalidPassConfig(
                                "parallel render pass worker panicked".to_string(),
                            )
                        })?
                    })
                    .collect::<Result<Vec<_>>>()
            })?;

            for (pass_index, (encoder, compute_encoder, cpu_timing, worker_profiler)) in
                layer.iter().copied().zip(handles)
            {
                command_buffers.push(compute_encoder);
                command_buffers.push(encoder);
                cpu_timings.push(cpu_timing);
                worker_profilers.push(worker_profiler);
                let pass_ptr = &passes[pass_index] as *const Box<dyn RenderPass>;
                let frame_ptr: *mut libhelio::FrameResources<'a> =
                    unsafe { std::mem::transmute(visible as *mut libhelio::FrameResources<'_>) };
                unsafe {
                    (&*pass_ptr).publish(&mut *frame_ptr);
                    (&*pass_ptr).publish_registry(registry);
                }
            }
        }
        Ok((command_buffers, cpu_timings, worker_profilers))
    }

    /// Executes the graph with both the legacy frame-resource shim and the
    /// phase 3 open resource registry.
    ///
    /// `registry` is supplied by the host so external inputs can be written
    /// with typed [`libhelio::ResourceKey`] values before execution. Existing
    /// passes continue to see `frame_resources` unchanged.
    pub fn execute_with_resources(
        &mut self,
        scene: &dyn SceneInput,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        frame_resources: &libhelio::FrameResources<'_>,
        registry: &mut libhelio::ResourceRegistry<'_>,
    ) -> Result<wgpu::SubmissionIndex> {
        assert!(
            self.locked,
            "RenderGraph::execute() requires lock() to be called first"
        );

        // External device owners drive wgpu polling. Consume callbacks from
        // that host cadence before reserving a bounded readback slot for this
        // frame; this never polls or waits.
        if !self.owns_device {
            self.profiler.read_gpu_timestamps_deferred();
        }
        // Worker profilers use private query sets. On an externally-owned
        // device their mappings may complete several frames after submission,
        // so keep them alive until the host poll cadence delivers a result.
        if !self.pending_worker_profilers.is_empty() {
            let mut pending = Vec::new();
            for mut worker in self.pending_worker_profilers.drain(..) {
                if self.owns_device {
                    worker.read_gpu_timestamps_blocking(scene.device());
                } else {
                    worker.read_gpu_timestamps_deferred();
                }
                if worker.has_completed_gpu_timings() {
                    let samples = worker.get_gpu_timings().to_vec();
                    self.profiler.merge_external_gpu_timings(&samples);
                } else {
                    pending.push(worker);
                }
            }
            self.pending_worker_profilers = pending;
        }
        self.profiler.clear_cpu_timings();

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

        // Compute is submitted first, graphics second. Span BOTH command
        // buffers; per-pass markers on compute alone omit all graphics work.
        self.profiler
            .begin_gpu_pass(&mut compute_encoder, "__graph_frame");
        let mut visible_frame_resources = *frame_resources;
        registry.reset_tracking("RenderGraph");
        let reflected_groups: Vec<Vec<wgpu::BindGroup>> = self
            .passes
            .iter()
            .enumerate()
            .map(|(pass_index, _)| self.create_reflected_groups(pass_index, registry))
            .collect::<Result<Vec<_>>>()?;
        let resized_this_frame = self.resize_pending;

        let use_parallel_recording = self.subpass_chains.is_empty()
            && self.gpu_render_bundles.iter().all(Option::is_none)
            && !self.parallel_layers.is_empty();
        let (parallel_command_buffers, parallel_cpu_timings, mut worker_profilers) =
            if use_parallel_recording {
                self.execute_parallel_layers(
                    scene,
                    target,
                    depth,
                    &mut visible_frame_resources,
                    registry,
                    &reflected_groups,
                    resized_this_frame,
                )?
            } else {
                (Vec::new(), Vec::new(), Vec::new())
            };
        for (name, duration) in parallel_cpu_timings {
            self.profiler.record_external_cpu_timing(name, duration);
        }

        let mut chain_rp: Option<std::mem::ManuallyDrop<wgpu::RenderPass<'_>>> = None;
        let mut chain_patch: Vec<Option<wgpu::RenderPassColorAttachment<'static>>> = Vec::new();

        if !use_parallel_recording {
            for (pass_index, pass) in self.passes.iter_mut().enumerate() {
                if let Some(bundle) = &self.gpu_render_bundles[pass_index] {
                    let pass_name = pass.name();
                    self.profiler
                        .begin_gpu_pass(&mut compute_encoder, pass_name);

                    if let Some(desc) = pass.render_pass_descriptor_with_pool(
                        target,
                        depth,
                        &visible_frame_resources,
                        &self.pool,
                    ) {
                        let mut pass_encoder = encoder.begin_render_pass(&desc);
                        pass_encoder.execute_bundles(std::iter::once(bundle));
                    } else {
                        let scene_resources = scene.resources();
                        let mut ctx = PassContext {
                            encoder_ptr: &mut encoder as *mut _,
                            compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                            target,
                            depth,
                            scene: scene_resources,
                            scene_buffers: scene.scene_buffers(),
                            profiler: &mut self.profiler,
                            frame_num: scene.frame_count(),
                            width: self.internal_w,
                            height: self.internal_h,
                            device: scene.device(),
                            resources: &visible_frame_resources,
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
                        pass.execute(&mut ctx)?;
                    }

                    self.profiler.end_gpu_pass(&mut compute_encoder, pass_name);
                    pass.publish(&mut visible_frame_resources);
                    pass.publish_registry(registry);
                    continue;
                }

                // prepare()
                {
                    let _scope = self.profiler.scope(pass.name());
                    let prepare_ctx = PrepareContext {
                        device: scene.device(),
                        queue: scene.queue(),
                        frame_num: scene.frame_count(),
                        scene: scene.resources(),
                        scene_buffers: scene.scene_buffers(),
                        frame_resources: &visible_frame_resources,
                        registry: &*registry,
                        resize: resized_this_frame,
                        width: self.internal_w,
                        height: self.internal_h,
                        delta_time: self.delta_time,
                    };
                    pass.prepare(&prepare_ctx)?;
                }

                // Populate graph-owned output textures into FrameResources BEFORE execute().
                if let Some(actions) = self.pre_pass_actions.get(pass_index) {
                    for action in actions {
                        match action {
                            PrePassAction::Route { name, view } => {
                                visible_frame_resources.route_named_texture(name, view, "Graph");
                            }
                            PrePassAction::Group { name, members } => {
                                // Generic: the core resolves a `write_group`'s
                                // members to concrete views but has no notion of
                                // what they mean — only the owning pass (this
                                // pass, since `Group` actions are always stored
                                // at their group's first-write pass index) knows
                                // how to publish them into its own bespoke
                                // `FrameResources` field (e.g. `.gbuffer`).
                                let views: Vec<&wgpu::TextureView> =
                                    members.iter().map(|(_, v)| v).collect();
                                pass.publish_group(*name, &views, &mut visible_frame_resources);
                            }
                        }
                    }
                }

                // execute()
                let pass_name = pass.name();
                self.profiler
                    .begin_gpu_pass(&mut compute_encoder, pass_name);

                // Migrated path: executor manages render pass (pass implements render_pass_descriptor).
                if let Some(desc) = pass.render_pass_descriptor_with_pool(
                    target,
                    depth,
                    &visible_frame_resources,
                    &self.pool,
                ) {
                    let cache = self.pass_cache.get(pass_index).and_then(|c| c.as_ref());
                    let is_chained = cache.map_or(false, |c| !c.chain_range.is_empty());

                    if is_chained {
                        let c = cache.unwrap();
                        if pass_index == c.chain_range.start {
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

                        let scene_resources = scene.resources();
                        let mut ctx = PassContext {
                            encoder_ptr: std::ptr::addr_of_mut!(encoder),
                            compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                            target,
                            depth,
                            scene: scene_resources,
                            scene_buffers: scene.scene_buffers(),
                            profiler: &mut self.profiler,
                            frame_num: scene.frame_count(),
                            width: self.internal_w,
                            height: self.internal_h,
                            device: scene.device(),
                            resources: &visible_frame_resources,
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
                        pass.execute(&mut ctx)?;

                        if pass_index + 1 >= c.chain_range.end {
                            if let Some(mut rp) = chain_rp.take() {
                                unsafe {
                                    std::mem::ManuallyDrop::drop(&mut rp);
                                }
                            }
                        }
                    } else {
                        if let Some(mut rp) = chain_rp.take() {
                            unsafe {
                                std::mem::ManuallyDrop::drop(&mut rp);
                            }
                        }

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
                            let scene_resources = scene.resources();
                            let mut ctx = PassContext {
                                encoder_ptr: std::ptr::addr_of_mut!(encoder),
                                compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                                target,
                                depth,
                                scene: scene_resources,
                                scene_buffers: scene.scene_buffers(),
                                profiler: &mut self.profiler,
                                frame_num: scene.frame_count(),
                                width: self.internal_w,
                                height: self.internal_h,
                                device: scene.device(),
                                resources: &visible_frame_resources,
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
                    if !bridged {
                        if let Some(mut rp) = chain_rp.take() {
                            unsafe {
                                std::mem::ManuallyDrop::drop(&mut rp);
                            }
                        }
                    }

                    let scene_resources = scene.resources();
                    let mut ctx = PassContext {
                        encoder_ptr: std::ptr::addr_of_mut!(encoder),
                        compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                        target,
                        depth,
                        scene: scene_resources,
                        scene_buffers: scene.scene_buffers(),
                        profiler: &mut self.profiler,
                        frame_num: scene.frame_count(),
                        width: self.internal_w,
                        height: self.internal_h,
                        device: scene.device(),
                        resources: &visible_frame_resources,
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
                    pass.execute(&mut ctx)?;
                }

                self.profiler.end_gpu_pass(&mut compute_encoder, pass_name);

                pass.publish(&mut visible_frame_resources);
                pass.publish_registry(registry);
            }
        }

        if let Some(mut rp) = chain_rp.take() {
            unsafe {
                std::mem::ManuallyDrop::drop(&mut rp);
            }
        }
        self.profiler.end_gpu_pass(&mut encoder, "__graph_frame");
        // Resolve after the final graphics timestamp, not before graphics runs.
        self.profiler
            .resolve_gpu_queries(&mut encoder, self.frame_count);
        let mut command_buffers = vec![compute_encoder.finish(), encoder.finish()];
        command_buffers.extend(parallel_command_buffers);
        let submission_index = scene.queue().submit(command_buffers);
        crate::upload::finish_frame();

        if self.owns_device {
            self.profiler.read_gpu_timestamps_blocking(scene.device());
        } else {
            self.profiler.read_gpu_timestamps_deferred();
        }
        for mut worker in worker_profilers.drain(..) {
            if self.owns_device {
                worker.read_gpu_timestamps_blocking(scene.device());
            } else {
                worker.read_gpu_timestamps_deferred();
            }
            if worker.has_completed_gpu_timings() {
                let samples = worker.get_gpu_timings().to_vec();
                self.profiler.merge_external_gpu_timings(&samples);
            } else if !self.owns_device && worker.gpu_timing_supported() {
                self.pending_worker_profilers.push(worker);
            }
        }
        self.profiler
            .update_snapshot(self.frame_count, self.passes.iter().map(|pass| pass.name()));

        self.frame_count += 1;
        self.resize_pending = false;

        Ok(submission_index)
    }

    /// Finalize the graph after all passes have been added.
    fn prepare_pipeline_registries(&mut self) {
        let device = self.device.clone();
        let mut registries = Vec::with_capacity(self.passes.len());
        for pass in &self.passes {
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
            registries.push(registry);
        }
        self.pipeline_registries = registries;
    }

    /// Finalize the graph after all passes have been added.
    pub fn lock(&mut self, width: u32, height: u32) {
        assert!(!self.locked, "RenderGraph::lock() called twice");
        self.internal_w = width;
        self.internal_h = height;
        self.output_w = width;
        self.output_h = height;
        self.pool.clear();
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

        // Build a "canon" `FrameResources` for the attachment probe below by
        // replaying the exact same pre-pass routing the real per-frame loop
        // performs (see `execute_with_frame_resources`), generically: plain
        // named routes via `route_named_texture`, and any `write_group`
        // bundle via the owning pass's `publish_group` — core never
        // special-cases a specific group's name here.
        let mut canon = libhelio::FrameResources::empty();
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
                let desc = pass.render_pass_descriptor_with_pool(
                    &dummy_target,
                    &dummy_depth,
                    &canon,
                    &self.pool,
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
                Some(CachedPass {
                    store_ops,
                    subpass_index,
                    subpass_count,
                    chain_range,
                })
            })
            .collect();

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
        let mut base = libhelio::FrameResources::empty();
        for pass in &mut self.passes {
            let bundle = pass.build_gpu_render_bundle(&self.device, &base);
            self.gpu_render_bundles.push(bundle);
            pass.publish(&mut base);
        }
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
        let mut base = libhelio::FrameResources::empty();
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
        self.last_bundle_chain_gen
            .resize(self.passes.len(), self.chain_generation);
    }
}
