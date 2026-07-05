use crate::graph::resource::{GraphTexturePool, TextureDescriptor};
use crate::graph::ResourceBuilder;
use crate::{GpuScene, PassContext, PrepareContext, Profiler, RenderPass, Result};
use libhelio::GBufferViews;
use std::any::TypeId;
use std::collections::HashMap;

/// Per-resource debug info for the debug overlay.
#[derive(Clone)]
pub struct DebugResourceInfo {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    pub format_name: String,
    pub size_kb: u64,
    pub alias: String,
}

/// Per-pass debug info for the debug overlay.
#[derive(Clone)]
pub struct DebugPassInfo {
    pub index: usize,
    pub name: String,
    pub kind: String, // "C" or "R"
    pub writes: Vec<String>,
    pub chain_marker: String,
}

/// All debug data for a single frame.
#[derive(Clone, Default)]
pub struct FrameDebugData {
    pub resources: Vec<DebugResourceInfo>,
    pub total_vram_kb: u64,
    pub passes: Vec<DebugPassInfo>,
    pub subpass_chains: Vec<String>,
    pub frame_count: u64,
    pub delta_time: f32,
}

fn format_bpp(fmt: wgpu::TextureFormat) -> u32 {
    use wgpu::TextureFormat::*;
    match fmt {
        R8Unorm | R8Snorm | R8Uint | R8Sint => 8,
        R16Unorm | R16Snorm | R16Uint | R16Sint | R16Float | Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint => 16,
        R32Uint | R32Sint | R32Float | Rg16Unorm | Rg16Snorm | Rg16Uint | Rg16Sint | Rg16Float | Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Rgba8Sint | Bgra8UnormSrgb => 32,
        Rg32Uint | Rg32Sint | Rg32Float | Rgba16Unorm | Rgba16Snorm | Rgba16Uint | Rgba16Sint | Rgba16Float => 64,
        Rgba32Uint | Rgba32Sint | Rgba32Float => 128,
        Depth32Float => 32,
        _ => 32,
    }
}

fn format_name(fmt: wgpu::TextureFormat) -> &'static str {
    use wgpu::TextureFormat::*;
    match fmt {
        Rgba16Float => "Rgba16Float",
        Rgba8Unorm => "Rgba8Unorm",
        Rgba8UnormSrgb => "Rgba8UnormSrgb",
        Bgra8UnormSrgb => "Bgra8UnormSrgb",
        R32Float => "R32Float",
        R16Float => "R16Float",
        R8Unorm => "R8Unorm",
        Rg16Float => "Rg16Float",
        Depth32Float => "Depth32Float",
        _ => "Other",
    }
}

/// Pre-computed per-pass data populated at graph lock time.
struct CachedPass {
    /// Store op override for each color attachment (None = use pass's original).
    store_ops: Vec<Option<wgpu::StoreOp>>,
    /// Subpass index within a chain (0 for standalone / chain start).
    subpass_index: u32,
    /// Range of the chain (start..end), or 0..0 for standalone.
    chain_range: std::ops::Range<usize>,
}

struct ResourceLifetime {
    first_write_pass: usize,
    #[allow(dead_code)]
    last_read_pass: usize,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    depth_or_array_layers: u32,
    mip_level_count: u32,
    extra_usage: wgpu::TextureUsages,
    alias_group: Option<String>,
}

/// An action to perform on FrameResources before a pass executes.
enum PrePassAction {
    /// Route a single named view into the appropriate FrameResources slot.
    Route { name: String, view: wgpu::TextureView },
    /// Write a GBufferViews struct composed of 4 individual views.
    Gbuffer {
        albedo: wgpu::TextureView,
        normal: wgpu::TextureView,
        orm: wgpu::TextureView,
        emissive: wgpu::TextureView,
    },
}

pub struct RenderGraph {
    passes: Vec<Box<dyn RenderPass>>,
    pass_index_map: HashMap<TypeId, usize>,
    profiler: Profiler,
    pool: GraphTexturePool,
    resources: HashMap<String, ResourceLifetime>,
    /// Per-pass index: actions to perform before that pass runs.
    pre_pass_actions: Vec<Vec<PrePassAction>>,
    device: std::sync::Arc<wgpu::Device>,
    internal_w: u32,
    internal_h: u32,
    output_w: u32,
    output_h: u32,
    delta_time: f32,
    owns_device: bool,
    gpu_render_bundles: Vec<Option<wgpu::RenderBundle>>,
    resources_allocated: bool,
    /// Detected subpass-fusible chains. Each entry is a contiguous range of
    /// pass indices whose resources share a single render pass (tile-memory
    /// optimization). The executor opens one render pass per chain instead of
    /// one per migrated pass. Consecutive chained passes draw into the same
    /// active render pass without closing/reopening it.
    subpass_chains: Vec<std::ops::Range<usize>>,
    /// When true, `add_pass()` panics and `lock()` has been called.
    locked: bool,
    /// Pre-computed per-pass data (store ops, chain info) populated by `lock()`.
    pass_cache: Vec<Option<CachedPass>>,
    /// Frame counter for periodic stats reporting.
    frame_count: u64,
}

impl RenderGraph {
    pub fn new(device: &std::sync::Arc<wgpu::Device>, queue: &wgpu::Queue) -> Self {
        Self {
            passes: Vec::new(),
            pass_index_map: HashMap::new(),
            profiler: Profiler::new(device, queue),
            pool: GraphTexturePool::new(),
            resources: HashMap::new(),
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
            locked: false,
            pass_cache: Vec::new(),
            frame_count: 0,
        }
    }

    pub fn new_with_external_device(device: &std::sync::Arc<wgpu::Device>, queue: &wgpu::Queue) -> Self {
        let mut graph = Self::new(device, queue);
        graph.owns_device = false;
        graph
    }

    pub fn set_delta_time(&mut self, dt: f32) {
        self.delta_time = dt;
    }

    // ── Declaration collection ──────────────────────────────────────────

    fn collect_declarations(&mut self) {
        self.resources.clear();
        let mut builders: Vec<ResourceBuilder> = (0..self.passes.len())
            .map(|_| ResourceBuilder::new())
            .collect();
        for (i, pass) in self.passes.iter().enumerate() {
            pass.declare_resources(&mut builders[i]);
            // Inject read declarations from legacy reads() so the graph sees
            // every pass's resource dependencies for chain detection.
            for &name in pass.reads() {
                builders[i].read(name);
            }
        }
        self.build_resource_lifetimes(&builders);
    }

    fn build_resource_lifetimes(&mut self, builders: &[ResourceBuilder]) {
        #[derive(Clone)]
        struct DeclWrite {
            name: String,
            format: Option<wgpu::TextureFormat>,
            size: crate::graph::ResourceSize,
            pass_index: usize,
            layers: u32,
            extra_usage: wgpu::TextureUsages,
        }
        let mut writes: Vec<DeclWrite> = Vec::new();

        for (i, builder) in builders.iter().enumerate() {
            for d in builder.declarations() {
                if matches!(d.access, crate::graph::ResourceAccess::Write) {
                    let fmt = d.format.map(|f| f.to_wgpu());
                    writes.push(DeclWrite {
                        name: d.name.to_string(),
                        format: fmt,
                        size: d.size.unwrap_or(crate::graph::ResourceSize::MatchSurface),
                        pass_index: i,
                        layers: d.layers,
                        extra_usage: d.extra_usage,
                    });
                }
            }
        }

        for w in &writes {
            let mut last_read = w.pass_index;
            for (j, builder) in builders.iter().enumerate() {
                for d in builder.declarations() {
                    if d.access == crate::graph::ResourceAccess::Read && d.name == w.name && j > last_read {
                        last_read = j;
                    }
                }
            }

            let (width, height) = match w.size {
                crate::graph::ResourceSize::MatchSurface => (self.internal_w, self.internal_h),
                crate::graph::ResourceSize::Output => (self.output_w, self.output_h),
                crate::graph::ResourceSize::Absolute { width, height } => (width, height),
                crate::graph::ResourceSize::Scaled { divisor } => {
                    (self.output_w / divisor.max(1), self.output_h / divisor.max(1))
                }
            };
            let fmt = w.format.unwrap_or(wgpu::TextureFormat::Rgba16Float);

            let mip_level_count = if fmt == wgpu::TextureFormat::R32Float {
                let max_dim = width.max(height);
                (u32::BITS - max_dim.leading_zeros()).max(1).min(12)
            } else {
                1
            };

            self.resources.entry(w.name.clone()).or_insert(ResourceLifetime {
                first_write_pass: w.pass_index,
                last_read_pass: last_read,
                format: fmt,
                width,
                height,
                depth_or_array_layers: w.layers.max(1),
                mip_level_count,
                extra_usage: w.extra_usage,
                alias_group: None,
            });
        }
    }

    // ── Texture allocation ──────────────────────────────────────────────

    fn allocate_textures(&mut self) {
        self.pre_pass_actions.clear();
        if self.resources.is_empty() {
            return;
        }

        // Allocate all textures into the pool.
        for (name, rl) in &self.resources {
            let usage = if rl.format == wgpu::TextureFormat::R32Float {
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING
            } else {
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
            } | rl.extra_usage;
            let tex_desc = TextureDescriptor {
                name: name.clone(),
                format: rl.format,
                width: rl.width,
                height: rl.height,
                depth_or_array_layers: rl.depth_or_array_layers,
                mip_level_count: rl.mip_level_count,
                sample_count: 1,
                usage,
                alias_group: rl.alias_group.clone(),
            };
            self.pool.allocate(&self.device, tex_desc);
        }

        // Build per-pass pre-populate actions.
        let mut actions: Vec<Vec<PrePassAction>> = (0..self.passes.len()).map(|_| Vec::new()).collect();
        for (name, rl) in &self.resources {
            let pi = rl.first_write_pass;
            if pi >= actions.len() { continue; }
            if let Some(view) = self.pool.get_view(name) {
                actions[pi].push(PrePassAction::Route {
                    name: name.clone(),
                    view: wgpu::TextureView::clone(view),
                });
            }
        }

        // Post-process: detect compound resources (e.g. GBuffer) and replace
        // individual Route actions with a single Gbuffer action.
        for pi in 0..actions.len() {
            let mut albedo_idx = None;
            let mut normal_idx = None;
            let mut orm_idx = None;
            let mut emissive_idx = None;

            for (j, action) in actions[pi].iter().enumerate() {
                if let PrePassAction::Route { name, .. } = action {
                    match name.as_str() {
                        "gbuffer_albedo" => albedo_idx = Some(j),
                        "gbuffer_normal" => normal_idx = Some(j),
                        "gbuffer_orm" => orm_idx = Some(j),
                        "gbuffer_emissive" => emissive_idx = Some(j),
                        _ => {}
                    }
                }
            }

            if let (Some(a), Some(n), Some(o), Some(e)) = (albedo_idx, normal_idx, orm_idx, emissive_idx) {
                // Extract the views. We need to take ownership; since the Vec
                // will be rebuilt, we clone them from the existing Route entries.
                let albedo_v = match &actions[pi][a] { PrePassAction::Route { view, .. } => wgpu::TextureView::clone(view), _ => unreachable!() };
                let normal_v = match &actions[pi][n] { PrePassAction::Route { view, .. } => wgpu::TextureView::clone(view), _ => unreachable!() };
                let orm_v = match &actions[pi][o] { PrePassAction::Route { view, .. } => wgpu::TextureView::clone(view), _ => unreachable!() };
                let emissive_v = match &actions[pi][e] { PrePassAction::Route { view, .. } => wgpu::TextureView::clone(view), _ => unreachable!() };

                // Remove the individual Route entries (pop larger indices first).
                let mut indices = vec![a, n, o, e];
                indices.sort_by(|a, b| b.cmp(a));
                for idx in indices {
                    actions[pi].remove(idx);
                }

                // Add the compound Gbuffer entry.
                actions[pi].push(PrePassAction::Gbuffer {
                    albedo: albedo_v,
                    normal: normal_v,
                    orm: orm_v,
                    emissive: emissive_v,
                });
            }
        }

        self.pre_pass_actions = actions;
    }

    // ── Public API ──────────────────────────────────────────────────────

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        if self.output_w == width && self.output_h == height && self.resources_allocated {
            return;
        }
        self.output_w = width;
        self.output_h = height;

        if self.locked {
            // Re-lock with the new size.
            self.locked = false;
            self.lock(width, height);
            for pass in &mut self.passes {
                pass.on_resize(&self.device, width, height);
            }
        } else {
            self.pool.clear();
            self.collect_declarations();
            self.allocate_textures();
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
        self.allocate_textures();
        self.detect_subpass_chains();
        self.resources_allocated = true;
        self.rebuild_gpu_render_bundles();
    }

    /// Detect chains of adjacent passes where each writes a resource the next
    /// reads. These could be fused into a single render pass with `next_subpass()`
    /// to keep inter-pass data in tile memory.
    /// Uses a greedy forward scan to build maximal chains.
    fn detect_subpass_chains(&mut self) {
        self.subpass_chains.clear();
        // Greedy chain builder: chain A→B if B reads() any resource that A writes().
        let mut i = 0;
        while i < self.passes.len() {
            let chain_start = i;
            while i + 1 < self.passes.len() {
                let writes_i: &[&str] = self.passes[i].writes();
                let reads_next: &[&str] = self.passes[i + 1].reads();
                let can_fuse = writes_i.iter().any(|w| reads_next.contains(w));
                if !can_fuse { break; }
                i += 1;
            }
            let chain_len = i + 1 - chain_start;
            if chain_len >= 2 {
                self.subpass_chains.push(chain_start..i + 1);
            }
            i += 1;
        }
    }

    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        assert!(!self.locked, "RenderGraph: cannot add_pass() after lock()");
        let type_id = pass.as_any().type_id();
        self.pass_index_map.entry(type_id).or_insert(self.passes.len());
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

    pub fn validate_dependencies(&self) -> std::result::Result<(), String> {
        use std::collections::HashSet;
        let mut available: HashSet<&str> = HashSet::new();
        available.insert("main_scene");
        available.insert("vg");
        available.insert("billboards");
        available.insert("corona_emitters");
        available.insert("depth_texture");

        for (i, pass) in self.passes.iter().enumerate() {
            let name = pass.name();
            for &resource in pass.reads() {
                if !available.contains(resource) {
                    return Err(format!(
                        "RenderGraph validation failed: pass '{}' (index {}) reads '{}' \
                         but no prior pass writes it. Available: {:?}",
                        name, i, resource, available
                    ));
                }
            }
            for &resource in pass.writes() {
                available.insert(resource);
            }
        }
        Ok(())
    }

    pub fn dump_dependency_graph(&self) {
        eprintln!("digraph RenderGraph {{");
        for (i, pass) in self.passes.iter().enumerate() {
            eprintln!("  {} [label=\"{}\"];", i, pass.name());
            for &resource in pass.reads() {
                for j in (0..i).rev() {
                    if self.passes[j].writes().contains(&resource) {
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

    /// Collect a snapshot of all resource and pass data for the debug overlay.
    pub fn collect_frame_debug_data(&self) -> FrameDebugData {
        let mut data = FrameDebugData::default();
        data.frame_count = self.frame_count;
        data.delta_time = self.delta_time;

        let mut total_bytes = 0u64;
        let mut alias_groups: HashMap<&str, Vec<&str>> = HashMap::new();

        for (name, rl) in &self.resources {
            let bpp = format_bpp(rl.format);
            let bytes = rl.width as u64 * rl.height as u64 * rl.depth_or_array_layers as u64 * bpp as u64 / 8;
            total_bytes += bytes;
            let alias = rl.alias_group.as_deref().unwrap_or("-").to_string();
            if rl.alias_group.is_some() {
                alias_groups.entry(rl.alias_group.as_ref().unwrap()).or_default().push(name);
            }
            data.resources.push(DebugResourceInfo {
                name: name.clone(),
                width: rl.width,
                height: rl.height,
                layers: rl.depth_or_array_layers,
                format_name: format_name(rl.format).to_string(),
                size_kb: bytes / 1024,
                alias,
            });
        }
        data.total_vram_kb = total_bytes / 1024;

        // Alias group summary lines
        for (group, members) in &alias_groups {
            let t: u64 = members.iter().filter_map(|n| {
                self.resources.get(*n).map(|rl| {
                    let bpp = format_bpp(rl.format);
                    rl.width as u64 * rl.height as u64 * rl.depth_or_array_layers as u64 * bpp as u64 / 8
                })
            }).sum();
            let saved = t * (members.len().saturating_sub(1) as u64);
            data.passes.push(DebugPassInfo {
                index: 999,
                name: format!("alias group '{}': {} members, ~{} KB saved", group, members.len(), saved / 1024),
                kind: String::new(),
                writes: Vec::new(),
                chain_marker: String::new(),
            });
        }

        // Pass pipeline
        let mut pass_chain: Vec<Option<usize>> = vec![None; self.passes.len()];
        for (ci, chain) in self.subpass_chains.iter().enumerate() {
            for pi in chain.clone() {
                pass_chain[pi] = Some(ci);
            }
        }

        for (i, pass) in self.passes.iter().enumerate() {
            let writes: Vec<String> = self.resources.iter()
                .filter(|(_, rl)| rl.first_write_pass == i)
                .map(|(n, _)| n.clone())
                .collect();
            let r_or_c = if writes.is_empty() { "C" } else { "R" };
            let marker = match pass_chain[i] {
                Some(ci) => {
                    let chain = &self.subpass_chains[ci];
                    if i == chain.start { format!("[{}.{}]", ci, chain.len()) }
                    else { format!("|.{}", chain.len()) }
                }
                None => String::new(),
            };
            data.passes.push(DebugPassInfo {
                index: i,
                name: pass.name().to_string(),
                kind: r_or_c.to_string(),
                writes,
                chain_marker: marker,
            });
        }

        for (ci, chain) in self.subpass_chains.iter().enumerate() {
            let names: Vec<String> = self.passes[chain.start..chain.end].iter().map(|p| p.name().to_string()).collect();
            data.subpass_chains.push(format!("chain {}: {}", ci, names.join(" → ")));
        }

        data
    }

    pub fn execute(
        &mut self,
        scene: &GpuScene,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) -> Result<()> {
        let frame_resources = libhelio::FrameResources::empty();
        self.execute_with_frame_resources(scene, target, depth, &frame_resources)
    }

    pub fn execute_with_frame_resources(
        &mut self,
        scene: &GpuScene,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        frame_resources: &libhelio::FrameResources<'_>,
    ) -> Result<()> {
        if !self.resources_allocated {
            self.init_transients(scene.width.max(1), scene.height.max(1));
        }

        self.profiler.clear_cpu_timings();

        let mut encoder = scene
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Graph"),
            });
        let mut compute_encoder = scene
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Compute Graph"),
            });

        let mut visible_frame_resources = *frame_resources;

        // Subpass chain tracking: keep a render pass open across consecutive
        // migrated passes that form a write→read chain (tile-memory optimization).
        let mut chain_rp: Option<std::mem::ManuallyDrop<wgpu::RenderPass<'_>>> = None;

        for (pass_index, pass) in self.passes.iter_mut().enumerate() {
            // GPU-only prebuilt path.
            if let Some(bundle) = &self.gpu_render_bundles[pass_index] {
                let pass_name = pass.name();
                self.profiler.begin_gpu_pass(&mut compute_encoder, pass_name);

                if let Some(desc) = pass.render_pass_descriptor(target, depth, &visible_frame_resources) {
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
                        profiler: &mut self.profiler,
                        frame_num: scene.frame_count,
                        width: self.internal_w,
                        height: self.internal_h,
                        device: &scene.device,
                        resources: &visible_frame_resources,
                        owns_device: self.owns_device,
                        resource_pool: &self.pool,
                        subpass_index: 0,
                        active_render_pass: None,
                        active_compute_pass: None,
                    };
                    pass.execute(&mut ctx)?;
                }

                self.profiler.end_gpu_pass(&mut compute_encoder, pass_name);
                pass.publish(&mut visible_frame_resources);
                continue;
            }

            // prepare()
            {
                let _scope = self.profiler.scope(pass.name());
                let prepare_ctx = PrepareContext {
                    device: &scene.device,
                    queue: &scene.queue,
                    frame_num: scene.frame_count,
                    scene,
                    frame_resources: &visible_frame_resources,
                    resize: false,
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
                            route_named_texture(name, view, &mut visible_frame_resources);
                        }
                        PrePassAction::Gbuffer { albedo, normal, orm, emissive } => {
                            visible_frame_resources.gbuffer.write(
                                GBufferViews { albedo, normal, orm, emissive },
                                "Graph",
                            );
                        }
                    }
                }
            }

            // execute()
            let pass_name = pass.name();
            self.profiler.begin_gpu_pass(&mut compute_encoder, pass_name);

            // Migrated path: executor manages render pass (pass implements render_pass_descriptor).
            if let Some(desc) = pass.render_pass_descriptor(target, depth, &visible_frame_resources) {
                let cache = self.pass_cache.get(pass_index).and_then(|c| c.as_ref());
                let chains_enabled = std::env::var("HELIO_DISABLE_CHAINS").is_err();
                let is_chained = chains_enabled && cache.map_or(false, |c| !c.chain_range.is_empty());

                if is_chained {
                    let c = cache.unwrap();
                    if pass_index == c.chain_range.start {
                        // First pass in chain: apply cached store-op overrides.
                        let chain_atts: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> =
                            desc.color_attachments.iter().enumerate().map(|(i, opt)| {
                                let mut a = opt.clone();
                                if let (Some(att), Some(store)) = (&mut a, c.store_ops.get(i).copied().flatten()) {
                                    att.ops.store = store;
                                }
                                a
                            }).collect();
                        let chain_desc = wgpu::RenderPassDescriptor {
                            label: desc.label,
                            color_attachments: &chain_atts,
                            depth_stencil_attachment: desc.depth_stencil_attachment,
                            timestamp_writes: desc.timestamp_writes,
                            occlusion_query_set: desc.occlusion_query_set,
                            multiview_mask: desc.multiview_mask,
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
                        profiler: &mut self.profiler,
                        frame_num: scene.frame_count,
                        width: self.internal_w,
                        height: self.internal_h,
                        device: &scene.device,
                        resources: &visible_frame_resources,
                        owns_device: self.owns_device,
                        resource_pool: &self.pool,
                        subpass_index: c.subpass_index,
                        active_render_pass: chain_rp.as_mut().map(|rp| &mut **rp as *mut _ as *mut _),
                        active_compute_pass: None,
                    };
                    pass.execute(&mut ctx)?;

                    if pass_index + 1 >= c.chain_range.end {
                        if let Some(mut rp) = chain_rp.take() {
                            unsafe { std::mem::ManuallyDrop::drop(&mut rp); }
                        }
                    }
                } else {
                    // Standalone (or unlocked graph): close any active chain first.
                    if let Some(mut rp) = chain_rp.take() {
                        unsafe { std::mem::ManuallyDrop::drop(&mut rp); }
                    }

                    // Apply cached store-op overrides (no-ops if cache is None).
                    let standalone_atts: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> =
                        desc.color_attachments.iter().enumerate().map(|(i, opt)| {
                            let mut a = opt.clone();
                            if let (Some(att), Some(store)) = (&mut a, cache.and_then(|c| c.store_ops.get(i).copied()).flatten()) {
                                att.ops.store = store;
                            }
                            a
                        }).collect();
                    let standalone_desc = wgpu::RenderPassDescriptor {
                        label: desc.label,
                        color_attachments: &standalone_atts,
                        depth_stencil_attachment: desc.depth_stencil_attachment,
                        timestamp_writes: desc.timestamp_writes,
                        occlusion_query_set: desc.occlusion_query_set,
                        multiview_mask: desc.multiview_mask,
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
                            profiler: &mut self.profiler,
                            frame_num: scene.frame_count,
                            width: self.internal_w,
                            height: self.internal_h,
                            device: &scene.device,
                            resources: &visible_frame_resources,
                            owns_device: self.owns_device,
                            resource_pool: &self.pool,
                            subpass_index: 0,
                            active_render_pass: Some(&mut rp as *mut _ as *mut _),
                            active_compute_pass: None,
                        };
                        pass.execute(&mut ctx)?;
                    }
                }
            } else {
                // Legacy path: close any active chain render pass first.
                if let Some(mut rp) = chain_rp.take() {
                    unsafe { std::mem::ManuallyDrop::drop(&mut rp); }
                }

                // Legacy path: pass opens its own render/compute pass via begin_render_pass.
                let scene_resources = scene.resources();
                let mut ctx = PassContext {
                    encoder_ptr: std::ptr::addr_of_mut!(encoder),
                        compute_encoder_ptr: std::ptr::addr_of_mut!(compute_encoder),
                    target,
                    depth,
                    scene: scene_resources,
                    profiler: &mut self.profiler,
                    frame_num: scene.frame_count,
                    width: self.internal_w,
                    height: self.internal_h,
                    device: &scene.device,
                    resources: &visible_frame_resources,
                    owns_device: self.owns_device,
                    resource_pool: &self.pool,
                    subpass_index: 0,
                    active_render_pass: None,
                    active_compute_pass: None,
                };
                pass.execute(&mut ctx)?;
            }

            self.profiler.end_gpu_pass(&mut compute_encoder, pass_name);

            pass.publish(&mut visible_frame_resources);
        }

        self.profiler.resolve_gpu_queries(&mut compute_encoder);
        scene.queue.submit([compute_encoder.finish(), encoder.finish()]);
        crate::upload::finish_frame();

        if self.owns_device {
            self.profiler.read_gpu_timestamps_blocking(&scene.device);
        } else {
            self.profiler.read_gpu_timestamps_deferred();
        }

        self.frame_count += 1;

        Ok(())
    }

    /// Finalize the graph after all passes have been added.
    ///
    /// Pre-computes resource lifetimes, subpass chains, and per-pass descriptor
    /// data so that `execute()` has zero HashMap lookups and zero heap allocations
    /// for the hot path. Panics if `add_pass()` is called after this.
    pub fn lock(&mut self, width: u32, height: u32) {
        assert!(!self.locked, "RenderGraph::lock() called twice");

        self.internal_w = width;
        self.internal_h = height;
        self.output_w = width;
        self.output_h = height;

        self.pool.clear();
        self.collect_declarations();
        // Detect chains BEFORE texture allocation so the allocator can choose
        // to skip resources whose entire lifetime is within a single chain.
        self.detect_subpass_chains();
        self.allocate_textures();
        self.resources_allocated = true;
        self.rebuild_gpu_render_bundles();

        // Build canonical FrameResources from graph-owned textures.
        let mut canon = libhelio::FrameResources::empty();
        for (name, _) in &self.resources {
            if let Some(view) = self.pool.get_view(name) {
                route_named_texture(name, view, &mut canon);
            }
        }
        // Also collect GBuffer views if present.
        if canon.gbuffer.get().is_none() {
            let a = self.pool.get_view("gbuffer_albedo");
            let n = self.pool.get_view("gbuffer_normal");
            let o = self.pool.get_view("gbuffer_orm");
            let e = self.pool.get_view("gbuffer_emissive");
            if let (Some(a), Some(n), Some(o), Some(e)) = (a, n, o, e) {
                canon.gbuffer.write(
                    libhelio::GBufferViews { albedo: &a, normal: &n, orm: &o, emissive: &e },
                    "Graph",
                );
            }
        }

        // Build reverse view→name map from the pool for the descriptor scan.
        let mut v2n: std::collections::HashMap<usize, &str> = std::collections::HashMap::new();
        for (name, _) in &self.resources {
            if let Some(view) = self.pool.get_view(name) {
                v2n.insert(view as *const _ as usize, name.as_str());
            }
        }

        // Create a tiny throw-away texture for the `target` parameter so we can
        // call render_pass_descriptor() at lock time (the actual target differs
        // every frame and is only used as the output attachment — it never
        // appears in v2n so it gets no store-op override).
        let dummy_target = {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Lock Dummy Target"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };
        // Depth from the graph pool if available, otherwise a 1×1 stand-in.
        let dummy_depth = self.pool.get_view("depth").cloned()
            .unwrap_or_else(|| {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Lock Dummy Depth"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                tex.create_view(&wgpu::TextureViewDescriptor::default())
            });

        // Pre-compute per-pass cache.
        self.pass_cache = self.passes.iter().enumerate().map(|(pi, pass)| {
            let desc = pass.render_pass_descriptor(&dummy_target, &dummy_depth, &canon)?;

            let chain = self.subpass_chains.iter().find(|c| c.contains(&pi));
            let chain_range = chain.cloned().unwrap_or(0..0);
            let subpass_index = chain.map_or(0, |c| (pi - c.start) as u32);

            // Determine store ops for each color attachment.
            let mut store_ops: Vec<Option<wgpu::StoreOp>> = Vec::new();
            for opt in desc.color_attachments.iter() {
                let op = opt.as_ref().and_then(|att| {
                    let key = att.view as *const _ as usize;
                    let name = v2n.get(&key)?;
                    let rl = self.resources.get(*name)?;
                    // If every reader is within this chain, discard.
                    if rl.last_read_pass < chain_range.end {
                        Some(wgpu::StoreOp::Discard)
                    } else {
                        None
                    }
                });
                store_ops.push(op);
            }

            Some(CachedPass { store_ops, subpass_index, chain_range })
        }).collect();

        // Log detected chains.
        if !self.subpass_chains.is_empty() {
            let names: Vec<String> = self.subpass_chains.iter().map(|c| {
                let ns: Vec<&str> = self.passes[c.start..c.end].iter().map(|p| p.name()).collect();
                format!("  chain {}: {}", c.start, ns.join(" → "))
            }).collect();
            eprintln!("[RenderGraph] {} subpass chain(s):", self.subpass_chains.len());
            for n in &names { eprintln!("{}", n); }
        }

        self.locked = true;
    }

    fn rebuild_gpu_render_bundles(&mut self) {
        self.gpu_render_bundles.clear();
        let mut base = libhelio::FrameResources::empty();
        for pass in &mut self.passes {
            let bundle = pass.build_gpu_render_bundle(&self.device, &base);
            self.gpu_render_bundles.push(bundle);
            pass.publish(&mut base);
        }
    }
}

// ── Standalone routing function ───────────────────────────────────────

fn route_named_texture<'a>(name: &str, view: &'a wgpu::TextureView, frame: &mut libhelio::FrameResources<'a>) {
    match name {
        "pre_aa" => frame.pre_aa.write(view, "Graph"),
        "ssao" => frame.ssao.write(view, "Graph"),
        "hiz" => frame.hiz.write(view, "Graph"),
        "sky_lut" => frame.sky_lut.write(view, "Graph"),
        "gbuffer_lightmap_uv" => frame.gbuffer_lightmap_uv.write(view, "Graph"),
        "water_sim_texture" => frame.water_sim_texture.write(view, "Graph"),
        "water_caustics" => frame.water_caustics.write(view, "Graph"),
        "rc_cascades" => frame.rc_view.write(view, "Graph"),
        "shadow_atlas" => frame.shadow_atlas.write(view, "Graph"),
        "static_shadow_atlas" => frame.static_shadow_atlas.write(view, "Graph"),
        // Individual GBuffer textures are handled by Gbuffer action, not here.
        "gbuffer_albedo" | "gbuffer_normal" | "gbuffer_orm" | "gbuffer_emissive" => {}
        _ => {}
    }
}
