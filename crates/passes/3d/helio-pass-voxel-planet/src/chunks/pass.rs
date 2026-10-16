//! Chunk trees in Helio's render graph: residency, primary visibility, sun
//! and sky light, and the gbuffer, from a host's [`SharedPlanetFrame`].
//! Lighting is Helio's deferred lighting, its sun shadowed by walks through
//! the chunk trees (`directional_visibility`).

use super::{residency, ChunkWorld, FrameTimings, ResidencyConfig, UpdateStats};
use crate::engine::{PlanetFrame, SharedPlanetFrame, TerrainAppearance};
use glam::{DVec3, Mat4, Vec3};
use helio_core::Result as HelioResult;
use helio_core::{
    DebugViewDescriptor, DebugViewLegend, DebugViewLegendStop, PassContext, PrepareContext,
    RenderPass,
};

pub struct ChunkPass {
    source: SharedPlanetFrame,
    config: ResidencyConfig,
    /// The host's art settings (none: the terrain generator's).
    appearance: Option<TerrainAppearance>,
    debug_mode: u32,
    world: Option<ChunkWorld>,
    /// The planet the world was built for (recipe fingerprint).
    built_for: Option<u64>,
    shown: Option<PlanetFrame>,
    last_eye: Option<(DVec3, std::time::Instant)>,
    stats: UpdateStats,
    /// Frames whose GPU times came back, until taken.
    timings: Vec<FrameTimings>,
    /// Generation's milliseconds last frame (`ResidencyConfig::frame_ms`).
    budget_ms: Option<f64>,
    /// The last frame's CPU milliseconds: prepare, execute.
    cpu_ms: [f64; 2],
}

impl ChunkPass {
    /// A pass drawing the frames `source` gives.
    pub fn new(source: SharedPlanetFrame) -> Self {
        Self::with_config(source, ResidencyConfig::default())
    }

    /// With residency settings `config` (its pixel angle follows the
    /// camera).
    pub fn with_config(source: SharedPlanetFrame, config: ResidencyConfig) -> Self {
        Self {
            source,
            config,
            appearance: None,
            debug_mode: 0,
            world: None,
            built_for: None,
            shown: None,
            last_eye: None,
            stats: UpdateStats::default(),
            timings: Vec::new(),
            budget_ms: None,
            cpu_ms: [0.0; 2],
        }
    }

    /// The host's art settings; whether they changed.
    pub fn set_appearance(&mut self, appearance: Option<TerrainAppearance>) -> bool {
        if self.appearance == appearance {
            return false;
        }
        self.appearance = appearance;
        if let (Some(world), Some(frame)) = (&mut self.world, &self.shown) {
            world.set_appearance(&appearance.unwrap_or_else(|| frame.planet.field().appearance()));
        }
        true
    }

    /// A host viewport should keep rendering until residency settles.
    pub fn needs_frame(&self) -> bool {
        let has_source = self.source.try_lock().map_or(true, |s| s.is_some());
        has_source && self.world.as_ref().is_none_or(|w| !w.settled())
    }

    /// Whether a planet's chunks are held.
    pub fn has_world(&self) -> bool {
        self.world.is_some()
    }

    /// The planet's chunks, for diagnostics (trace counts, a pixel's trace).
    pub fn world(&self) -> Option<&ChunkWorld> {
        self.world.as_ref()
    }

    /// Whether nothing is left to generate for the current eye.
    pub fn settled(&self) -> bool {
        self.world.as_ref().is_some_and(ChunkWorld::settled)
    }

    /// The last frame's residency.
    pub fn stats(&self) -> &UpdateStats {
        &self.stats
    }

    /// The last frame's CPU milliseconds in the pass: prepare (edits,
    /// results), execute (residency, draw).
    pub fn cpu_ms(&self) -> [f64; 2] {
        self.cpu_ms
    }

    /// GPU times of frames that came back since the last call.
    pub fn take_timings(&mut self) -> Vec<FrameTimings> {
        std::mem::take(&mut self.timings)
    }

    /// One line for a host's diagnostics.
    pub fn diagnostics(&self) -> String {
        let s = &self.stats;
        let gpu = self.timings.last();
        format!(
            "chunks settled={} drawn_nodes={} drawn_chunks={} resident_chunks={} fading={} waiting={} footprints={} chunks={} seen={} reached={} budget={} coarse={:.3} worst_px={:.2} worst={} plan_age={} plan_ms={:.1} switch_px={:.2} out_of_slots={} out_of_leaves={} cpu_ms={:?} draw_ms={:?} shadow_ms={:?} generate_ms={:?}",
            self.settled(),
            s.drawn_nodes,
            s.drawn_chunks,
            s.resident_chunks,
            s.fading_nodes,
            s.requests,
            s.generated_footprints,
            s.generated_chunks,
            s.seen,
            s.reached,
            s.budget,
            s.coarse_share,
            s.worst_px,
            s.worst.map_or("-".to_string(), |w| format!("{}/{}/{}/{}/{}/exact{}/guess{}/kids{}of{}", w.key.face, w.key.level, w.key.cx, w.key.cy, w.key.cz, u8::from(w.exact), w.guess, w.generated, w.children)),
            s.plan_age,
            s.plan_ms,
            s.switch_px,
            s.out_of_slots,
            s.out_of_leaves,
            s.cpu_ms,
            gpu.map(|t| t.draw),
            gpu.map(|t| t.shadow),
            gpu.map(|t| t.generate),
        )
    }
}

const READS: &[&str] = &[
    "gbuffer",
    "gbuffer_lightmap_uv",
    "gbuffer_sss",
    "gbuffer_extra",
    "gbuffer_velocity",
];
const WRITES: &[&str] = &[
    "gbuffer",
    "gbuffer_lightmap_uv",
    "gbuffer_sss",
    "gbuffer_extra",
    "gbuffer_velocity",
    "directional_visibility",
];

/// GPU times kept for a host that does not take them.
const KEEP_TIMINGS: usize = 256;

impl RenderPass for ChunkPass {
    fn name(&self) -> &'static str {
        "VoxelChunks"
    }
    fn inherit_persistent_state(&mut self, previous: &mut dyn RenderPass) -> bool {
        let Some(previous) = previous.as_any_mut().downcast_mut::<Self>() else {
            return false;
        };
        if !std::sync::Arc::ptr_eq(&self.source, &previous.source) {
            return false;
        }
        self.world = previous.world.take();
        self.debug_mode = previous.debug_mode;
        if let Some(world) = &mut self.world {
            world.set_debug_mode(self.debug_mode);
        }
        self.built_for = previous.built_for.take();
        self.shown = previous.shown.take();
        // A graph rebuild must not revert the host's art settings.
        self.appearance = previous.appearance;
        self.world.is_some()
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
        let started = std::time::Instant::now();
        let result = self.prepare_frame(ctx);
        self.cpu_ms[0] = started.elapsed().as_secs_f64() * 1e3;
        result
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some(view) = self.world.as_ref().and_then(ChunkWorld::sun_view) {
            frame.write_texture_binding("directional_visibility", view, self.name());
        }
    }
    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let started = std::time::Instant::now();
        let result = self.execute_frame(ctx);
        self.cpu_ms[1] = started.elapsed().as_secs_f64() * 1e3;
        result
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.debug_mode = mode;
        if let Some(world) = &mut self.world {
            world.set_debug_mode(mode);
        }
    }

    fn debug_views(&self) -> &'static [DebugViewDescriptor] {
        static VIEWS: &[DebugViewDescriptor] = &[DebugViewDescriptor {
            name: "Voxel Trace Cost",
            debug_mode: 32,
            description: "Voxel surfaces shaded by primary-ray traversal work; red is cheaper, yellow and white are more expensive",
        }];
        VIEWS
    }

    fn debug_view_legend(&self, debug_mode: u32) -> Option<DebugViewLegend> {
        static STOPS: &[DebugViewLegendStop] = &[
            DebugViewLegendStop {
                label: "0",
                color: [0, 0, 0],
            },
            DebugViewLegendStop {
                label: "64",
                color: [255, 0, 0],
            },
            DebugViewLegendStop {
                label: "128",
                color: [255, 255, 0],
            },
            DebugViewLegendStop {
                label: "192+",
                color: [255, 255, 255],
            },
        ];

        (debug_mode == 32).then_some(DebugViewLegend {
            title: "Voxel trace cost",
            unit: "ray steps",
            stops: STOPS,
        })
    }
}

impl ChunkPass {
    fn prepare_frame(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // The last frame was submitted: its results can be mapped.
        if let Some(world) = &mut self.world {
            world.set_polls_device(ctx.owns_device);
            world.set_plain_colour(false);
            world.submitted();
            self.timings.extend(world.take_timings());
            let excess = self.timings.len().saturating_sub(KEEP_TIMINGS);
            self.timings.drain(..excess);
        }
        let frame = self
            .source
            .lock()
            .map_err(|_| {
                helio_core::Error::InvalidPassConfig("planet frame source poisoned".into())
            })?
            .clone();
        // No planet: its chunks and storage go.
        let Some(frame) = frame else {
            self.shown = None;
            self.world = None;
            self.built_for = None;
            return Ok(());
        };
        let recipe = crate::record::recipe_fingerprint(frame.planet.recipe());
        // Edits follow the planet; another recipe or history, a new world.
        if self.built_for != Some(recipe)
            || self
                .world
                .as_mut()
                .is_some_and(|w| !w.sync_edits(&frame.planet))
        {
            let mut world = ChunkWorld::new(ctx.device, ctx.queue, &frame.planet);
            world.set_polls_device(ctx.owns_device);
            world.set_plain_colour(false);
            world.set_debug_mode(self.debug_mode);
            if let Some(appearance) = &self.appearance {
                world.set_appearance(appearance);
            }
            self.world = Some(world);
            self.built_for = Some(recipe);
            self.last_eye = None;
        }
        self.shown = Some(frame);
        Ok(())
    }

    fn execute_frame(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let (Some(world), Some(frame)) = (&mut self.world, self.shown.clone()) else {
            return Ok(());
        };
        let missing = |name: &str| helio_core::Error::ResourceNotFound(name.into());
        let g = ctx
            .registry
            .get::<helio_core::ViewGroup<'_, 4>>(helio_core::ResourceKey::new("gbuffer"))
            .ok_or_else(|| missing("gbuffer"))?;
        macro_rules! view {
            ($name:literal) => {
                ctx.registry
                    .get::<&wgpu::TextureView>(helio_core::ResourceKey::new($name))
                    .ok_or_else(|| missing($name))?
            };
        }
        let gbuffer = [
            g.views[0],
            g.views[1],
            g.views[2],
            g.views[3],
            view!("gbuffer_lightmap_uv"),
            view!("gbuffer_sss"),
            view!("gbuffer_extra"),
            view!("gbuffer_velocity"),
        ];
        let camera = ctx.camera_data;
        // Chunks are drawn relative to the eye. The frame's coordinates may
        // put the camera off their origin (a host's camera-relative frames
        // keep an origin while the camera moves within a cell of it): its
        // view-projection, moved by the camera's position, is the eye's.
        let at = Vec3::from_slice(&camera.position_near[..3]);
        let view_proj = Mat4::from_cols_array(&camera.view_proj) * Mat4::from_translation(at);
        let proj = Mat4::from_cols_array(&camera.proj);
        // The last frame's, which the renderer keeps in this frame's
        // coordinates (rebased when the origin moves), the same way.
        let prev_view_proj =
            Mat4::from_cols_array(&camera.prev_view_proj) * Mat4::from_translation(at);
        let now = std::time::Instant::now();
        let interval = self
            .last_eye
            .map(|(_, at)| now.duration_since(at).as_secs_f64().max(1e-3));
        let velocity = match (self.last_eye, interval) {
            (Some((eye, _)), Some(dt)) => (frame.eye - eye) / dt,
            _ => DVec3::ZERO,
        };
        self.last_eye = Some((frame.eye, now));
        // In view: the cone around the camera's forward reaching the
        // frustum's corners. A pixel's angle sets the cut's detail.
        let tan_y = 1.0 / f64::from(proj.y_axis.y);
        let tan_x = 1.0 / f64::from(proj.x_axis.x);
        let view = residency::View {
            forward: glam::Vec3::from_slice(&camera.forward_far[..3])
                .as_dvec3()
                .normalize(),
            cos_half: (tan_x * tan_x + tan_y * tan_y).sqrt().atan().cos(),
        };
        self.config.plan.pixel_angle = 2.0 * tan_y / f64::from(ctx.height.max(1));
        let near = camera.position_near[3];
        let mut cmds = ctx.graphics_cmds();
        let encoder = &mut cmds;
        // Generation keeps its pace per second at any frame rate: a frame
        // slower than a 60th of a second generates for longer, up to twice.
        let mut config = self.config;
        config.generate_ms = config
            .generate_ms
            .map(|ms| ms * interval.map_or(1.0, |dt| (dt * 60.0).clamp(1.0, 2.0)));
        // It grows into what the last measured frame left of `frame_ms` on
        // the GPU and the CPU (its own generation counted in), by half a
        // millisecond a frame at most (measurements lag), up to three times
        // that; it falls at once.
        if let (Some(base), Some(frame_ms)) = (config.generate_ms, config.frame_ms) {
            let busy = match (ctx.last_frame_gpu_ms(), ctx.last_frame_cpu_ms()) {
                (Some(g), Some(c)) => Some(f64::from(g.max(c))),
                (g, c) => g.or(c).map(f64::from),
            };
            let room = busy.map_or(base, |busy| {
                (self.timings.last().map_or(0.0, |t| t.generate) + (frame_ms - busy) * 0.8)
                    .clamp(base, 3.0 * base)
            });
            let budget = room.min(self.budget_ms.unwrap_or(base) + 0.5);
            self.budget_ms = Some(budget);
            config.generate_ms = Some(budget);
        }
        self.stats = world.update(
            encoder,
            frame.eye,
            velocity,
            Some(view),
            ctx.frame_num,
            &config,
        );
        world.draw(
            encoder,
            ctx.frame_num,
            frame.eye,
            view_proj,
            prev_view_proj,
            near,
            frame.sun,
            [ctx.width, ctx.height],
        );
        world.publish(encoder, gbuffer, ctx.depth);
        if let Some(picks) = &frame.picks {
            world.copy_picks(encoder, picks);
        }
        Ok(())
    }
}
