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
                chain_local: false,
            });
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
                let albedo_v = match &actions[pi][a] { PrePassAction::Route { view, .. } => wgpu::TextureView::clone(view), _ => unreachable!() };
                let normal_v = match &actions[pi][n] { PrePassAction::Route { view, .. } => wgpu::TextureView::clone(view), _ => unreachable!() };
                let orm_v = match &actions[pi][o] { PrePassAction::Route { view, .. } => wgpu::TextureView::clone(view), _ => unreachable!() };
                let emissive_v = match &actions[pi][e] { PrePassAction::Route { view, .. } => wgpu::TextureView::clone(view), _ => unreachable!() };

                let mut indices = vec![a, n, o, e];
                indices.sort_by(|a, b| b.cmp(a));
                for idx in indices {
                    actions[pi].remove(idx);
                }

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
}
