use crate::graph::ResourceBuilder;

use super::execution::RenderGraph;
use super::scheduling::PrePassAction;

pub(crate) struct ResourceLifetime {
    pub(crate) first_write_pass: usize,
    #[allow(dead_code)]
    pub(crate) last_read_pass: usize,
    pub(crate) format: wgpu::TextureFormat,
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) depth_or_array_layers: u32,
    pub(crate) mip_level_count: u32,
    pub(crate) extra_usage: wgpu::TextureUsages,
    pub(crate) alias_group: Option<String>,
    pub(crate) chain_local: bool,
}

impl RenderGraph {
    pub(crate) fn collect_declarations(&mut self) {
        self.resources.clear();
        self.resource_groups.clear();
        let mut builders: Vec<ResourceBuilder> = (0..self.passes.len())
            .map(|_| ResourceBuilder::new())
            .collect();
        for (i, pass) in self.passes.iter().enumerate() {
            pass.declare_resources(&mut builders[i]);
            for &name in pass.reads() {
                builders[i].read(name);
            }
        }
        self.build_resource_lifetimes(&builders);
    }

    pub(crate) fn build_resource_lifetimes(&mut self, builders: &[ResourceBuilder]) {
        // Record `write_group` membership in declaration order — a plain
        // `Vec` built by walking `builders` (deterministic), not the
        // `self.resources` hash map (iteration order is unspecified) — so
        // `allocate_textures()` can later combine each group's writes into
        // one `PrePassAction::Group` with members in the order the pass
        // declared them, for any arity, without pattern-matching names.
        self.resource_groups.clear();
        for (i, builder) in builders.iter().enumerate() {
            for d in builder.declarations() {
                if d.access != crate::graph::ResourceAccess::Write {
                    continue;
                }
                let Some(group) = d.group else { continue };
                match self
                    .resource_groups
                    .iter_mut()
                    .find(|(pi, g, _)| *pi == i && *g == group)
                {
                    Some((_, _, members)) => members.push(d.name),
                    None => self.resource_groups.push((i, group, vec![d.name])),
                }
            }
        }

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
                    if d.access == crate::graph::ResourceAccess::Read
                        && d.name == w.name
                        && j > last_read
                    {
                        last_read = j;
                    }
                }
            }

            let (width, height) = match w.size {
                crate::graph::ResourceSize::MatchSurface => (self.internal_w, self.internal_h),
                crate::graph::ResourceSize::Output => (self.output_w, self.output_h),
                crate::graph::ResourceSize::Absolute { width, height } => (width, height),
                crate::graph::ResourceSize::Scaled { divisor } => (
                    self.output_w / divisor.max(1),
                    self.output_h / divisor.max(1),
                ),
                crate::graph::ResourceSize::ScaledInternal { divisor } => (
                    (self.internal_w / divisor.max(1)).max(1),
                    (self.internal_h / divisor.max(1)).max(1),
                ),
            };
            let fmt = w.format.unwrap_or(wgpu::TextureFormat::Rgba16Float);

            let mip_level_count = if fmt == wgpu::TextureFormat::R32Float {
                let max_dim = width.max(height);
                (u32::BITS - max_dim.leading_zeros()).max(1).min(12)
            } else {
                1
            };

            self.resources
                .entry(w.name.clone())
                .or_insert(ResourceLifetime {
                    first_write_pass: w.pass_index,
                    last_read_pass: last_read,
                    format: fmt,
                    width,
                    height,
                    depth_or_array_layers: w.layers.max(1),
                    mip_level_count,
                    extra_usage: w.extra_usage,
                    alias_group: None,
                    chain_local: false,
                });
        }
    }

    /// Assign alias groups so chain-local and non-chain-local resources never
    /// share a physical allocation.  Runs after `chain_local` is computed on
    /// every `ResourceLifetime`, before the final `allocate_textures()` call.
    pub(crate) fn assign_chain_aware_alias_groups(&mut self) {
        #[derive(Clone)]
        struct AliasGroup {
            name: String,
            chain_local: bool,
            format: wgpu::TextureFormat,
            width: u32,
            height: u32,
            depth_or_array_layers: u32,
            mip_level_count: u32,
            extra_usage: wgpu::TextureUsages,
            last_read_pass: usize,
        }

        let mut names: Vec<String> = self.resources.keys().cloned().collect();
        names.sort();
        names.sort_by_key(|name| self.resources[name].first_write_pass);

        let mut groups = Vec::<AliasGroup>::new();
        for (resource_index, name) in names.iter().enumerate() {
            let resource = &self.resources[name];
            let compatible = |group: &AliasGroup| {
                group.chain_local == resource.chain_local
                    && group.format == resource.format
                    && group.width >= resource.width
                    && group.height >= resource.height
                    && group.depth_or_array_layers >= resource.depth_or_array_layers
                    && group.mip_level_count >= resource.mip_level_count
                    && group.extra_usage.contains(resource.extra_usage)
                    && group.last_read_pass < resource.first_write_pass
            };

            let group_index = groups.iter().position(compatible).unwrap_or_else(|| {
                let name = if resource.chain_local {
                    format!("chain_alias_{resource_index}")
                } else {
                    format!("frame_alias_{}", groups.len())
                };
                groups.push(AliasGroup {
                    name,
                    chain_local: resource.chain_local,
                    format: resource.format,
                    width: resource.width,
                    height: resource.height,
                    depth_or_array_layers: resource.depth_or_array_layers,
                    mip_level_count: resource.mip_level_count,
                    extra_usage: resource.extra_usage,
                    last_read_pass: resource.last_read_pass,
                });
                groups.len() - 1
            });

            let group = &mut groups[group_index];
            group.last_read_pass = resource.last_read_pass;
            self.resources
                .get_mut(name)
                .expect("resource was collected from the map")
                .alias_group = Some(group.name.clone());
        }
    }

    pub(crate) fn allocate_textures(&mut self) {
        use crate::graph::resource::TextureDescriptor;

        self.pre_pass_actions.clear();
        if self.resources.is_empty() {
            return;
        }

        let mut allocation_order: Vec<&String> = self.resources.keys().collect();
        allocation_order.sort_by_key(|name| self.resources[*name].first_write_pass);
        let mut active: Vec<&str> = Vec::new();
        for name in allocation_order {
            let rl = &self.resources[name];
            active.retain(|active_name| {
                let active_rl = &self.resources[*active_name];
                if active_rl.last_read_pass < rl.first_write_pass {
                    self.pool.release(active_name);
                    false
                } else {
                    true
                }
            });
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
            active.push(name.as_str());
        }

        let mut actions: Vec<Vec<PrePassAction>> =
            (0..self.passes.len()).map(|_| Vec::new()).collect();
        for (name, rl) in &self.resources {
            let pi = rl.first_write_pass;
            if pi >= actions.len() {
                continue;
            }
            if let Some(view) = self.pool.get_view(name) {
                actions[pi].push(PrePassAction::Route {
                    name: name.clone(),
                    view: wgpu::TextureView::clone(view),
                });
            }
        }

        // Combine each declared `write_group`'s individual `Route` entries
        // into one `PrePassAction::Group`, generically over the group's name
        // and arity — not a scan for specific string literals. Any pass that
        // calls `write_group` gets this for free.
        for (pi, group_name, member_names) in &self.resource_groups {
            let pi = *pi;
            if pi >= actions.len() {
                continue;
            }
            let mut members = Vec::with_capacity(member_names.len());
            for &member_name in member_names {
                let Some(idx) = actions[pi].iter().position(
                    |a| matches!(a, PrePassAction::Route { name, .. } if name == member_name),
                ) else {
                    continue;
                };
                if let PrePassAction::Route { view, .. } = actions[pi].remove(idx) {
                    members.push((member_name, view));
                }
            }
            if !members.is_empty() {
                actions[pi].push(PrePassAction::Group {
                    name: *group_name,
                    members,
                });
            }
        }

        self.pre_pass_actions = actions;
    }
}
