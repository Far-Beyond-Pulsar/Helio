//! Stored voxel terrain integrated with Helio's GBuffer and lighting graph.
use crate::{Params, World};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::{Arc, Mutex};
mod residency;
mod terrain;

pub const GBUFFER_FORMATS: [wgpu::TextureFormat; 8] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rg16Float,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba16Float,
    wgpu::TextureFormat::Rgba16Float,
];
pub struct GBufferTargets<'a> {
    pub colors: [&'a wgpu::TextureView; 8],
    pub depth: &'a wgpu::TextureView,
}
#[derive(Clone, Copy)]
pub enum DepthConvention {
    Forward,
    Reverse,
}
#[derive(Clone)]
pub struct EngineVoxelFrame {
    pub params: Params,
    pub world: Arc<World>,
    pub raytraced_sun: bool,
}
pub type SharedVoxelFrame = Arc<Mutex<Option<EngineVoxelFrame>>>;

/// Graph entry that does not allocate the planet's residency buffers until a
/// matching SceneDB source supplies its first frame.
pub struct LazyEngineVoxelPass {
    source: SharedVoxelFrame,
    active: Option<EngineVoxelPass>,
}

impl LazyEngineVoxelPass {
    pub fn new(source: SharedVoxelFrame) -> Self {
        Self {
            source,
            active: None,
        }
    }

    pub fn ready(&self) -> bool {
        self.active.as_ref().is_some_and(EngineVoxelPass::ready)
    }

    pub fn chunk_jobs_pending(&self) -> usize {
        self.active
            .as_ref()
            .map_or(0, EngineVoxelPass::chunk_jobs_pending)
    }
}

impl RenderPass for LazyEngineVoxelPass {
    fn name(&self) -> &'static str {
        "TinyVoxelGBuffer"
    }
    fn reads(&self) -> &'static [&'static str] {
        &[
            "gbuffer",
            "gbuffer_lightmap_uv",
            "gbuffer_sss",
            "gbuffer_extra",
            "gbuffer_velocity",
        ]
    }
    fn writes(&self) -> &'static [&'static str] {
        &[
            "gbuffer",
            "gbuffer_lightmap_uv",
            "gbuffer_sss",
            "gbuffer_extra",
            "gbuffer_velocity",
            "directional_visibility",
        ]
    }
    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        for name in self.reads() {
            builder.read(name);
        }
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if self.active.is_none()
            && self
                .source
                .lock()
                .map_err(|_| {
                    helio_core::Error::InvalidPassConfig("Voxel frame source was poisoned".into())
                })?
                .is_some()
        {
            self.active = Some(EngineVoxelPass::with_frame_source(
                ctx.device,
                ctx.queue,
                ctx.width,
                ctx.height,
                Arc::clone(&self.source),
            ));
        }
        Ok(())
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        match &mut self.active {
            Some(pass) => pass.execute(ctx),
            None => Ok(()),
        }
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some(pass) = &self.active {
            pass.publish(frame);
        }
    }
    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if let Some(pass) = &mut self.active {
            pass.on_resize(device, width, height);
        }
    }
}

pub struct EngineVoxelPass {
    pub terrain: terrain::StoredTerrain,
    frame_source: Option<SharedVoxelFrame>,
    frame_params: Option<Params>,
    visibility_active: bool,
}
impl EngineVoxelPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        Self::with_depth(device, queue, width, height, DepthConvention::Forward)
    }
    pub fn with_depth(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        depth: DepthConvention,
    ) -> Self {
        Self {
            terrain: terrain::StoredTerrain::new(device, queue, width, height, depth),
            frame_source: None,
            frame_params: None,
            visibility_active: false,
        }
    }
    pub fn with_frame_source(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        source: SharedVoxelFrame,
    ) -> Self {
        let mut pass = Self::new(device, queue, width, height);
        pass.frame_source = Some(source);
        pass
    }
    pub fn set_frame(&mut self, params: Params) {
        self.frame_params = Some(params);
    }
    pub fn primary_traversal_mode(&self) -> &'static str {
        "stored-bricks"
    }
    pub fn chunk_jobs_pending(&self) -> usize {
        self.terrain.stats().pending
    }
    pub fn ready(&self) -> bool {
        self.terrain.stats().ready
    }
    pub fn set_stage_profiling(&mut self, enabled: bool) {
        self.terrain.set_stage_profiling(enabled);
    }
    pub fn stage_profiler(&self) -> Option<&helio_core::profiling::GpuProfiler> {
        self.terrain.profiler.as_ref()
    }
    pub fn visibility_direction(&self) -> Option<&wgpu::Texture> {
        self.visibility_active.then(|| self.terrain.direction())
    }
    pub fn visibility_diagnostics(&self) -> Option<&wgpu::Texture> {
        self.visibility_active.then(|| self.terrain.sun())
    }
    pub fn encode_gbuffer(
        &mut self,
        params: &Params,
        camera: &wgpu::Buffer,
        _frame: u64,
        encoder: &mut wgpu::CommandEncoder,
        targets: GBufferTargets<'_>,
    ) {
        self.terrain
            .encode(params, camera, encoder, targets, self.visibility_active);
    }
}
impl RenderPass for EngineVoxelPass {
    fn name(&self) -> &'static str {
        "TinyVoxelGBuffer"
    }
    fn reads(&self) -> &'static [&'static str] {
        &[
            "gbuffer",
            "gbuffer_lightmap_uv",
            "gbuffer_sss",
            "gbuffer_extra",
            "gbuffer_velocity",
        ]
    }
    fn writes(&self) -> &'static [&'static str] {
        &[
            "gbuffer",
            "gbuffer_lightmap_uv",
            "gbuffer_sss",
            "gbuffer_extra",
            "gbuffer_velocity",
            "directional_visibility",
        ]
    }
    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        for name in self.reads() {
            builder.read(name);
        }
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if self.visibility_active {
            frame.write_texture_binding(
                "directional_visibility",
                &self.terrain.sun_view,
                self.name(),
            );
        }
    }
    fn on_resize(&mut self, _: &wgpu::Device, width: u32, height: u32) {
        self.terrain.resize(width, height);
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if let Some(p) = &mut self.terrain.profiler {
            p.read_timestamps_deferred();
        }
        if let Some(source) = &self.frame_source {
            let frame = source
                .lock()
                .map_err(|_| {
                    helio_core::Error::InvalidPassConfig("Voxel frame source was poisoned".into())
                })?
                .clone();
            let Some(frame) = frame else {
                self.visibility_active = false;
                return Ok(());
            };
            self.terrain.set_world(frame.world);
            self.frame_params = Some(frame.params);
            self.visibility_active = frame.raytraced_sun;
        }
        let mut params = self.frame_params.ok_or_else(|| {
            helio_core::Error::InvalidPassConfig(
                "Voxel pass requires a high-precision frame".into(),
            )
        })?;
        params.screen[0] = ctx.width as f32;
        params.screen[1] = ctx.height as f32;
        let missing = |name: &str| helio_core::Error::ResourceNotFound(name.into());
        let g = ctx
            .registry
            .get::<helio_core::ViewGroup<'_, 4>>(helio_core::ResourceKey::new("gbuffer"))
            .ok_or_else(|| missing("gbuffer"))?;
        let targets = GBufferTargets {
            colors: [
                g.views[0],
                g.views[1],
                g.views[2],
                g.views[3],
                ctx.registry
                    .get::<&wgpu::TextureView>(helio_core::ResourceKey::new("gbuffer_lightmap_uv"))
                    .ok_or_else(|| missing("gbuffer_lightmap_uv"))?,
                ctx.registry
                    .get::<&wgpu::TextureView>(helio_core::ResourceKey::new("gbuffer_sss"))
                    .ok_or_else(|| missing("gbuffer_sss"))?,
                ctx.registry
                    .get::<&wgpu::TextureView>(helio_core::ResourceKey::new("gbuffer_extra"))
                    .ok_or_else(|| missing("gbuffer_extra"))?,
                ctx.registry
                    .get::<&wgpu::TextureView>(helio_core::ResourceKey::new("gbuffer_velocity"))
                    .ok_or_else(|| missing("gbuffer_velocity"))?,
            ],
            depth: ctx.depth,
        };
        self.encode_gbuffer(
            &params,
            ctx.camera,
            ctx.frame_num,
            unsafe { &mut *ctx.encoder_ptr },
            targets,
        );
        if let Some(p) = &mut self.terrain.profiler {
            p.resolve_queries(unsafe { &mut *ctx.encoder_ptr }, ctx.frame_num);
        }
        Ok(())
    }
}
