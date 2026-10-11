//! Chunk trees in Helio's render graph: residency, primary visibility and
//! the gbuffer, from the same frame source as [`crate::engine::PlanetPass`].
//! Lighting is Helio's deferred lighting, its sun shadowed by walks
//! through the chunk trees (`directional_visibility`).

use super::{residency, ChunkWorld, FrameTimings, ResidencyConfig, UpdateStats};
use crate::engine::{PlanetFrame, SharedPlanetFrame};
use glam::{DVec3, Mat4};
use helio_core::{PassContext, PrepareContext, RenderPass};
use helio_core::Result as HelioResult;
use std::sync::{Arc, Mutex};

/// What a [`ChunkPass`] reports to its host.
#[derive(Default)]
pub struct ChunkPassState {
    pub stats: UpdateStats,
    /// Frames whose GPU times came back, until the host takes them.
    pub timings: Vec<FrameTimings>,
    /// Nothing is left to generate for the current eye.
    pub settled: bool,
}

pub type SharedChunkState = Arc<Mutex<ChunkPassState>>;

pub struct ChunkPass {
    source: SharedPlanetFrame,
    state: SharedChunkState,
    config: ResidencyConfig,
    world: Option<ChunkWorld>,
    /// The planet the world was built for (recipe fingerprint).
    built_for: Option<u64>,
    shown: Option<PlanetFrame>,
    last_eye: Option<(DVec3, std::time::Instant)>,
    /// The last frame's view-projection and eye.
    last_view: Option<(Mat4, DVec3)>,
}

impl ChunkPass {
    /// A pass drawing the frames `source` gives, reporting to `state`.
    pub fn new(source: SharedPlanetFrame, state: SharedChunkState, config: ResidencyConfig) -> Self {
        Self { source, state, config, world: None, built_for: None, shown: None, last_eye: None, last_view: None }
    }
}

const READS: &[&str] = &["gbuffer", "gbuffer_lightmap_uv", "gbuffer_sss", "gbuffer_extra", "gbuffer_velocity"];
const WRITES: &[&str] = &["gbuffer", "gbuffer_lightmap_uv", "gbuffer_sss", "gbuffer_extra", "gbuffer_velocity", "directional_visibility"];

impl RenderPass for ChunkPass {
    fn name(&self) -> &'static str {
        "VoxelChunks"
    }
    fn reads(&self) -> &'static [&'static str] {
        READS
    }
    fn writes(&self) -> &'static [&'static str] {
        WRITES
    }
    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        for name in READS {
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
        // The last frame was submitted: its results can be mapped.
        if let Some(world) = &mut self.world {
            world.submitted();
            let timings = world.take_timings();
            if let Ok(mut state) = self.state.lock() {
                state.timings.extend(timings);
            }
        }
        let frame = self.source.lock().map_err(|_| helio_core::Error::InvalidPassConfig("planet frame source poisoned".into()))?.clone();
        let Some(frame) = frame else {
            self.shown = None;
            return Ok(());
        };
        let recipe = crate::journal::recipe_fingerprint(frame.planet.recipe());
        // Edits follow the planet; another recipe or history, a new world.
        if self.built_for != Some(recipe) || self.world.as_mut().is_some_and(|w| !w.sync_edits(&frame.planet)) {
            self.world = Some(ChunkWorld::new(ctx.device, ctx.queue, &frame.planet));
            self.built_for = Some(recipe);
            self.last_eye = None;
            self.last_view = None;
        }
        self.shown = Some(frame);
        Ok(())
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some(view) = self.world.as_ref().and_then(ChunkWorld::sun_view) {
            frame.write_texture_binding("directional_visibility", view, self.name());
        }
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let (Some(world), Some(frame)) = (&mut self.world, self.shown.clone()) else { return Ok(()) };
        let missing = |name: &str| helio_core::Error::ResourceNotFound(name.into());
        let g = ctx.registry.get::<helio_core::ViewGroup<'_, 4>>(helio_core::ResourceKey::new("gbuffer")).ok_or_else(|| missing("gbuffer"))?;
        macro_rules! view {
            ($name:literal) => {
                ctx.registry.get::<&wgpu::TextureView>(helio_core::ResourceKey::new($name)).ok_or_else(|| missing($name))?
            };
        }
        let gbuffer = [g.views[0], g.views[1], g.views[2], g.views[3], view!("gbuffer_lightmap_uv"), view!("gbuffer_sss"), view!("gbuffer_extra"), view!("gbuffer_velocity")];
        let camera = ctx.camera_data;
        let view_proj = Mat4::from_cols_array(&camera.view_proj);
        let proj = Mat4::from_cols_array(&camera.proj);
        // The last frame's view-projection is relative to its eye: move it
        // to this one's.
        let prev = Mat4::from_cols_array(&camera.prev_view_proj);
        let prev_view_proj = match self.last_view {
            Some((_, eye)) => prev * Mat4::from_translation((frame.eye - eye).as_vec3()),
            None => view_proj,
        };
        self.last_view = Some((view_proj, frame.eye));
        let now = std::time::Instant::now();
        let velocity = match self.last_eye {
            Some((eye, at)) => (frame.eye - eye) / now.duration_since(at).as_secs_f64().max(1e-3),
            None => DVec3::ZERO,
        };
        self.last_eye = Some((frame.eye, now));
        // In view: the cone around the camera's forward reaching the
        // frustum's corners.
        let tan_y = 1.0 / f64::from(proj.y_axis.y);
        let tan_x = 1.0 / f64::from(proj.x_axis.x);
        let view = residency::View {
            forward: glam::Vec3::from_slice(&camera.forward_far[..3]).as_dvec3().normalize(),
            cos_half: (tan_x * tan_x + tan_y * tan_y).sqrt().atan().cos(),
        };
        let near = camera.position_near[3];
        let mut cmds = ctx.graphics_cmds();
        let encoder = &mut cmds;
        let stats = world.update(encoder, frame.eye, velocity, Some(view), ctx.frame_num, &self.config);
        world.draw(encoder, ctx.frame_num, frame.eye, view_proj, prev_view_proj, near, frame.sun, [ctx.width, ctx.height]);
        world.publish(encoder, gbuffer, ctx.depth);
        if let Some(picks) = &frame.picks {
            world.copy_picks(encoder, picks);
        }
        if let Ok(mut state) = self.state.lock() {
            state.stats = stats;
            state.settled = world.settled();
        }
        Ok(())
    }
}
