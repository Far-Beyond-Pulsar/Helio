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
        let mut chain_group_gen: u32 = 0;
        for rl in self.resources.values_mut() {
            if rl.chain_local {
                // Every chain-local resource gets its own alias group keyed
                // on its first_write_pass.  Resources written by the same
                // pass are never alive concurrently so they may alias.
                let g = format!("chain_local_{}", rl.first_write_pass);
                rl.alias_group = Some(g);
            } else {
                rl.alias_group = None;
            }
        }
    }

    pub(crate) fn allocate_textures(&mut self) {
        use crate::graph::resource::TextureDescriptor;

        self.pre_pass_actions.clear();
        if self.resources.is_empty() {
            return;
        }

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
                let Some(idx) = actions[pi].iter().position(|a| {
                    matches!(a, PrePassAction::Route { name, .. } if name == member_name)
                }) else {
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
