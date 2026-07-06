use crate::graph::executor::{format_bpp, format_name};
use crate::graph::resource::GraphTexturePool;
use crate::{GpuScene, PassContext, PrepareContext, Profiler, RenderPass, Result};
use libhelio::GBufferViews;
use std::any::TypeId;
use std::collections::HashMap;

use super::resource_lifetime::ResourceLifetime;
use super::scheduling::{CachedPass, PrePassAction};
use super::{DebugPassInfo, DebugResourceInfo, FrameDebugData};

pub struct RenderGraph {
    pub(crate) passes: Vec<Box<dyn RenderPass>>,
    pass_index_map: HashMap<TypeId, usize>,
    profiler: Profiler,
    pub(crate) pool: GraphTexturePool,
    pub(crate) resources: HashMap<String, ResourceLifetime>,
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
    chain_membership: Vec<bool>,
    locked: bool,
    pass_cache: Vec<Option<CachedPass>>,
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
            chain_membership: Vec::new(),
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

    // ── Public API ──────────────────────────────────────────────────────

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        if self.output_w == width && self.output_h == height && self.resources_allocated {
            return;
        }
        self.output_w = width;
        self.output_h = height;

        if self.locked {
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
                chain_local: rl.chain_local,
                first_write_pass: rl.first_write_pass,
                last_read_pass: rl.last_read_pass,
            });
        }
        data.total_vram_kb = total_bytes / 1024;

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
        assert!(self.locked, "RenderGraph::execute() requires lock() to be called first");

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

        let mut chain_rp: Option<std::mem::ManuallyDrop<wgpu::RenderPass<'_>>> = None;
        let mut chain_patch: Vec<Option<wgpu::RenderPassColorAttachment<'static>>> = Vec::new();

        for (pass_index, pass) in self.passes.iter_mut().enumerate() {
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
                let is_chained = cache.map_or(false, |c| !c.chain_range.is_empty());

                if is_chained {
                    let c = cache.unwrap();
                    if pass_index == c.chain_range.start {
                        chain_patch.clear();
                        chain_patch.extend(desc.color_attachments.iter().enumerate().map(|(i, opt)| {
                            let mut a = opt.clone();
                            if let Some(store) = c.store_ops.get(i).copied().flatten() {
                                if let Some(ref mut att) = a {
                                    att.ops.store = store;
                                }
                            }
                            unsafe { std::mem::transmute::<
                                Option<wgpu::RenderPassColorAttachment<'_>>,
                                Option<wgpu::RenderPassColorAttachment<'static>>,
                            >(a) }
                        }));
                        let chain_desc = wgpu::RenderPassDescriptor {
                            label: desc.label,
                            color_attachments: &chain_patch,
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
                    if let Some(mut rp) = chain_rp.take() {
                        unsafe { std::mem::ManuallyDrop::drop(&mut rp); }
                    }

                    let standalone_atts: Vec<Option<wgpu::RenderPassColorAttachment<'_>>> =
                        desc.color_attachments.iter().enumerate().map(|(i, opt)| {
                            let mut a = opt.clone();
                            if let Some(store) = cache.and_then(|c| c.store_ops.get(i).copied()).flatten() {
                                if let Some(ref mut att) = a {
                                    att.ops.store = store;
                                }
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
                let bridged = self.chain_membership.get(pass_index).copied().unwrap_or(false)
                    && pass.chain_transparent();
                if !bridged {
                    if let Some(mut rp) = chain_rp.take() {
                        unsafe { std::mem::ManuallyDrop::drop(&mut rp); }
                    }
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
    pub fn lock(&mut self, width: u32, height: u32) {
        assert!(!self.locked, "RenderGraph::lock() called twice");
        self.internal_w = width;
        self.internal_h = height;
        self.output_w = width;
        self.output_h = height;
        self.pool.clear();
        self.collect_declarations();
        self.allocate_textures();
        self.resources_allocated = true;
        self.rebuild_gpu_render_bundles();

        let mut canon = libhelio::FrameResources::empty();
        for (name, _) in &self.resources {
            if let Some(view) = self.pool.get_view(name) {
                route_named_texture(name, view, &mut canon);
            }
        }
        if canon.gbuffer.get().is_none() {
            if let (Some(a), Some(n), Some(o), Some(e)) = (
                self.pool.get_view("gbuffer_albedo"),
                self.pool.get_view("gbuffer_normal"),
                self.pool.get_view("gbuffer_orm"),
                self.pool.get_view("gbuffer_emissive"),
            ) {
                canon.gbuffer.write(libhelio::GBufferViews { albedo: a, normal: n, orm: o, emissive: e }, "Graph");
            }
        }
        let mut v2n = std::collections::HashMap::new();
        for (name, _) in &self.resources {
            if let Some(view) = self.pool.get_view(name) {
                v2n.insert(view as *const _ as usize, name.as_str());
            }
        }

        let dummy_target = {
            let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Lock Dummy Target"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            tex.create_view(&wgpu::TextureViewDescriptor::default())
        };
        let dummy_depth = self.pool.get_view("depth").cloned()
            .unwrap_or_else(|| {
                let tex = self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("Lock Dummy Depth"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1, sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                });
                tex.create_view(&wgpu::TextureViewDescriptor::default())
            });

        let probes: Vec<Option<(usize, Vec<usize>)>> = self.passes.iter().map(|pass| {
            let desc = pass.render_pass_descriptor(&dummy_target, &dummy_depth, &canon)?;
            let color_len = desc.color_attachments.len();
            let mut signature: Vec<usize> = desc.color_attachments.iter().map(|opt| {
                opt.as_ref().map(|a| a.view as *const wgpu::TextureView as usize).unwrap_or(0)
            }).collect();
            signature.push(
                desc.depth_stencil_attachment.as_ref()
                    .map(|d| d.view as *const wgpu::TextureView as usize)
                    .unwrap_or(0)
            );
            Some((color_len, signature))
        }).collect();
        let attachments: Vec<Option<Vec<usize>>> = probes.iter()
            .map(|p| p.as_ref().map(|(_, sig)| sig.clone()))
            .collect();

        self.detect_subpass_chains_probed(&attachments);
        self.chain_membership = vec![false; self.passes.len()];
        for chain in &self.subpass_chains {
            for pi in chain.clone() {
                self.chain_membership[pi] = true;
            }
        }
        for rl in self.resources.values_mut() {
            rl.chain_local = self.subpass_chains.iter().any(|c| {
                c.start <= rl.first_write_pass && rl.last_read_pass < c.end
            });
        }

        self.pass_cache = probes.into_iter().enumerate().map(|(pi, probe)| {
            let (color_len, _) = probe?;
            let chain = self.subpass_chains.iter().find(|c| c.contains(&pi));
            let chain_range = chain.cloned().unwrap_or(0..0);
            let subpass_index = chain.map_or(0, |c| (pi - c.start) as u32);
            let store_ops: Vec<Option<wgpu::StoreOp>> = vec![None; color_len];
            Some(CachedPass { store_ops, subpass_index, chain_range })
        }).collect();

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
                        crate::graph::ResourceAccess::Read => { if !r.contains(&d.name) { r.push(d.name); } }
                        crate::graph::ResourceAccess::Write => { if !w.contains(&d.name) { w.push(d.name); } }
                    }
                }
                w_set.push(w);
                r_set.push(r);
            }
            eprintln!("[RenderGraph] {} passes, {} chain(s):", self.passes.len(), self.subpass_chains.len());
            for i in 0..self.passes.len() {
                let name = self.passes[i].name();
                let is_chain_start = self.subpass_chains.iter().any(|c| c.start == i);
                let is_chain_mid   = self.subpass_chains.iter().any(|c| i > c.start && i < c.end);
                let marker = if is_chain_start { " ──chain──►" } else if is_chain_mid { " │         " } else { "           " };
                let w_str = if w_set[i].is_empty() { "–".to_string() } else { w_set[i].join(",") };
                let r_str = if r_set[i].is_empty() { "–".to_string() } else { r_set[i].join(",") };
                eprintln!("  {:>2}. {:<28} W: {}  R: {}", i, name, w_str, r_str);
                if i + 1 < self.passes.len() {
                    let can_fuse = w_set[i].iter().any(|w| r_set[i + 1].contains(w));
                    let is_fused = self.subpass_chains.iter().any(|c| c.contains(&i) && c.contains(&(i + 1)));
                    if is_fused && !can_fuse && self.passes[i + 1].chain_transparent() {
                        eprintln!("  {:>2}.{:>2} CHAINED  (bridged over transparent pass '{}')", "", "", self.passes[i + 1].name());
                    } else {
                        let why = if can_fuse {
                            let common: Vec<&str> = w_set[i].iter().filter(|w| r_set[i + 1].contains(w)).copied().collect();
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
        "gbuffer_albedo" | "gbuffer_normal" | "gbuffer_orm" | "gbuffer_emissive" => {}
        _ => {}
    }
}
