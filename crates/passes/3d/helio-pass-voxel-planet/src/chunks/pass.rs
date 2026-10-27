//! Chunk trees in Helio's render graph: residency, primary visibility, sun
//! and sky light, and the gbuffer, from a host's [`SharedPlanetFrame`].
//! Lighting is Helio's deferred lighting, its sun shadowed by walks through
//! the chunk trees (`directional_visibility`).

use super::{residency, ChunkWorld, FrameTimings, ResidencyConfig, UpdateStats};
use crate::engine::{PlanetFrame, SharedPlanetFrame, TerrainAppearance};
use glam::{DVec3, Mat4, Vec2, Vec3};
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
    /// A save's export underway: the resident edited chunks are packing.
    exporting: bool,
    /// How the frame's kept chunks went into the last world built: how
    /// many, or why they were refused.
    kept_loaded: Option<Result<usize, String>>,
    last_eye: [Option<(DVec3, std::time::Instant)>; 2],
    /// Stereo draws can share a head-centred residency viewpoint while tracing
    /// from each actual eye. Inter-eye distance must not become movement velocity.
    residency_eye: Option<DVec3>,
    stats: UpdateStats,
    /// Frames whose GPU times came back, until taken.
    timings: Vec<FrameTimings>,
    /// Generation's milliseconds last frame (`ResidencyConfig::frame_ms`).
    budget_ms: Option<f64>,
    /// The last frame's CPU milliseconds: prepare, execute.
    cpu_ms: [f64; 2],
    /// Voxel materials as the host's scene materials (`set_scene_materials`).
    scene: Option<(
        super::scene::VoxelMaterialMap,
        super::scene::SceneMaterialSource,
    )>,
    /// Every hit walks its own sun ray (`set_steady_shadows`).
    steady_shadows: bool,
    /// The eye's world position this frame (`helio_pass_gi::eye`), where
    /// the GI clipmap the terrain traces rays of is centred.
    gi_eye: Option<DVec3>,
    /// The last frame's projection jitter (NDC).
    last_jitter: [Vec2; 2],
    xr_mode: bool,
    /// The terrain as the ground foliage grows on (`chunks::ground`): shared with a
    /// foliage placement pass ([`Self::foliage_ground`]), its tiling and its answers.
    planet_ground: std::sync::Arc<super::ground::PlanetGround>,
    foliage_ground: std::sync::Arc<helio_pass_foliage_place::GroundLink>,
}

impl ChunkPass {
    /// A pass drawing the frames `source` gives.
    pub fn new(source: SharedPlanetFrame) -> Self {
        Self::with_config(source, ResidencyConfig::default())
    }

    /// With residency settings `config` (its pixel angle follows the
    /// camera).
    pub fn with_config(source: SharedPlanetFrame, config: ResidencyConfig) -> Self {
        let planet_ground = std::sync::Arc::new(super::ground::PlanetGround::default());
        let foliage_ground =
            std::sync::Arc::new(helio_pass_foliage_place::GroundLink::answered_elsewhere(
                Box::new(planet_ground.clone()),
            ));
        Self {
            source,
            config,
            appearance: None,
            debug_mode: 0,
            world: None,
            built_for: None,
            shown: None,
            exporting: false,
            kept_loaded: None,
            last_eye: [None; 2],
            residency_eye: None,
            stats: UpdateStats::default(),
            timings: Vec::new(),
            budget_ms: None,
            cpu_ms: [0.0; 2],
            scene: None,
            steady_shadows: false,
            gi_eye: None,
            last_jitter: [Vec2::ZERO; 2],
            xr_mode: false,
            planet_ground,
            foliage_ground,
        }
    }

    /// The terrain as the ground foliage grows on: give it to a foliage placement pass
    /// (`FoliagePlacePass::set_ground`); this pass answers its queries and reports where
    /// the terrain changed under it.
    pub fn foliage_ground(&self) -> std::sync::Arc<helio_pass_foliage_place::GroundLink> {
        self.foliage_ground.clone()
    }

    /// Voxel materials shown as the host's scene materials: `map` says which
    /// scene material rows each voxel material's faces use, `source` reads
    /// the frame's material table from the context (see
    /// `helio_default_passes::voxel_scene_materials`). Devices without
    /// texture binding arrays keep the appearance table's colours.
    pub fn set_scene_materials(
        &mut self,
        map: super::scene::VoxelMaterialMap,
        source: super::scene::SceneMaterialSource,
    ) {
        self.scene = Some((map, source));
    }

    /// Steady shadows: every hit walks its own sun ray and the sky rays hold
    /// still, instead of samples that move each frame for the denoiser to
    /// gather (about three times dearer; a reference for the denoised
    /// shadows).
    pub fn set_steady_shadows(&mut self, on: bool) {
        self.steady_shadows = on;
    }

    /// Override the position used for residency/prefetch, without changing the
    /// frame's tracing eye. Use the head centre for both draws of a stereo pair.
    pub fn set_residency_eye(&mut self, eye: Option<DVec3>) {
        self.residency_eye = eye;
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
    /// A save's export too, until answered.
    pub fn needs_frame(&self) -> bool {
        let Ok(source) = self.source.try_lock() else {
            return true;
        };
        let Some(frame) = source.as_ref() else {
            return false;
        };
        let exporting = frame
            .export
            .as_ref()
            .is_some_and(|e| e.try_lock().map_or(true, |e| e.wanted));
        exporting || self.world.as_ref().is_none_or(|w| !w.settled())
    }

    /// How the frame's kept chunks went into the last world built: how
    /// many it keeps, or why they were refused (`None`: it had none).
    pub fn kept_loaded(&self) -> Option<&Result<usize, String>> {
        self.kept_loaded.as_ref()
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
            "chunks settled={} kept_loaded={:?} kept_chunks={} kept_bytes={} detail_pages={} detail_capacity={} out_of_details={} drawn_nodes={} drawn_chunks={} resident_chunks={} evicted={} deferred={} fading={} waiting={} footprints={} chunks={} seen={} reached={} budget={} coarse={:.3} worst_px={:.2} worst={} plan_age={} plan_ms={:.1} switch_px={:.2} out_of_slots={} out_of_leaves={} cpu_ms={:?} draw_ms={:?} shadow_ms={:?} denoise_ms={:?} generate_ms={:?}",
            self.settled(),
            self.kept_loaded,
            s.kept_chunks,
            s.kept_bytes,
            s.detail_pages.0,
            s.detail_pages.1,
            s.out_of_details,
            s.drawn_nodes,
            s.drawn_chunks,
            s.resident_chunks,
            s.evicted_chunks,
            s.evictions_deferred,
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
            gpu.map(|t| t.denoise),
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
    "material_textures",
    "tile_light_lists",
    "tile_light_counts",
    helio_pass_gi::GI_FRAME,
    helio_pass_gi::GI_RAYS,
    helio_pass_gi::GI_SUN,
    helio_pass_gi::GI_CLIPMAP,
    ATMOSPHERE_FRAME,
];
/// helio-pass-sky's frame of the atmosphere (its sun's disk).
const ATMOSPHERE_FRAME: &str = "atmosphere_frame";
const WRITES: &[&str] = &[
    "gbuffer",
    "gbuffer_lightmap_uv",
    "gbuffer_sss",
    "gbuffer_extra",
    "gbuffer_velocity",
    "directional_visibility",
    "local_visibility",
    // helio-pass-tsr's STOCHASTIC_MASK.
    "tsr_stochastic",
    helio_pass_gi::GI_HITS,
    helio_pass_shadow_matrix::SUN_OCCLUSION_MAPS,
    helio_pass_shadow_matrix::SUN_OCCLUSION_PAGES,
    helio_pass_shadow_matrix::SUN_OCCLUSION_FRAME,
];

/// GPU times kept for a host that does not take them.
const KEEP_TIMINGS: usize = 256;

impl RenderPass for ChunkPass {
    fn set_frame_inputs(&mut self, inputs: &helio_core::RenderFrameInputs<'_>) {
        self.xr_mode = inputs.xr_mode;
    }

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
        self.exporting = previous.exporting;
        self.kept_loaded = previous.kept_loaded.take();
        // A graph rebuild must not revert the host's art settings.
        self.appearance = previous.appearance;
        if self.scene.is_none() {
            self.scene = previous.scene.take();
        }
        self.steady_shadows |= previous.steady_shadows;
        self.residency_eye = previous.residency_eye;
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
        self.gi_eye = Some(helio_pass_gi::eye(ctx));
        let result = self.prepare_frame(ctx);
        self.cpu_ms[0] = started.elapsed().as_secs_f64() * 1e3;
        result
    }
    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        if let Some(view) = self.world.as_ref().and_then(ChunkWorld::stochastic_view) {
            frame.write_texture_binding("tsr_stochastic", view, self.name());
        }
        if let Some(view) = self.world.as_ref().and_then(ChunkWorld::sun_view) {
            frame.write_texture_binding("directional_visibility", view, self.name());
        }
        if let Some(buffer) = self.world.as_ref().and_then(ChunkWorld::local_visibility) {
            frame.write_buffer_binding("local_visibility", buffer, self.name());
        }
        if let Some((maps, pages, sun_frame)) =
            self.world.as_ref().and_then(ChunkWorld::sun_occlusion)
        {
            frame.write_texture_binding(
                helio_pass_shadow_matrix::SUN_OCCLUSION_MAPS,
                maps,
                self.name(),
            );
            frame.write_buffer_binding(
                helio_pass_shadow_matrix::SUN_OCCLUSION_PAGES,
                pages,
                self.name(),
            );
            frame.write_buffer_binding(
                helio_pass_shadow_matrix::SUN_OCCLUSION_FRAME,
                sun_frame,
                self.name(),
            );
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
        static VIEWS: &[DebugViewDescriptor] = &[
            DebugViewDescriptor {
                name: "Voxel Trace Cost",
                debug_mode: 32,
                description: "Voxel surfaces shaded by primary-ray traversal work; red is cheaper, yellow and white are more expensive",
            },
            DebugViewDescriptor {
                name: "Voxel Levels",
                debug_mode: 34,
                description: "Voxel surfaces coloured by the octree level drawing them (level 0 finest)",
            },
            DebugViewDescriptor {
                name: "Voxel Levels (Left Half)",
                debug_mode: 35,
                description: "Voxel levels over the left half of the view, shading on the right",
            },
        ];
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
        // Foliage tiles the planet this pass draws (a rebuilt graph's new pass too).
        self.planet_ground.set_grid(*frame.planet.grid());
        let recipe = frame.planet.recipe_fingerprint();
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
            self.kept_loaded = frame
                .kept
                .as_ref()
                .map(|kept| world.load_kept(&frame.planet, kept));
            self.world = Some(world);
            self.built_for = Some(recipe);
            self.last_eye = [None; 2];
            self.exporting = false;
        }
        // A save's export: the resident edited chunks packed (over the
        // next updates), then all kept chunks answered for this planet.
        if let (Some(world), Some(export)) = (&mut self.world, &frame.export) {
            let mut export = export.lock().map_err(|_| {
                helio_core::Error::InvalidPassConfig("kept chunk export poisoned".into())
            })?;
            if !export.wanted {
                self.exporting = false;
            } else if !self.exporting {
                world.keep_resident();
                self.exporting = true;
            } else if world.keeping() == 0 {
                export.answer = Some((frame.planet.clone(), world.kept_chunks(&frame.planet)));
                export.wanted = false;
                self.exporting = false;
            }
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
        let eye_index = if self.xr_mode && camera.jitter_frame[3].to_bits() == 2 {
            1
        } else {
            0
        };
        // The tracer starts every perspective ray at zero. Its matrices must
        // therefore be eye-relative, and its absolute ray origin must be the
        // SAME eye as the renderer camera (including OpenXR's predicted pose
        // and each eye's offset). PlanetFrame.eye can be an older input sample;
        // it remains useful for residency, but cannot anchor the traced view.
        let at = Vec3::from_slice(&camera.position_near[..3]);
        let (trace_eye, view_proj, prev_view_proj) =
            trace_camera(camera, ctx.world_origin);
        let proj = Mat4::from_cols_array(&camera.proj);
        let jitter = Vec2::from_slice(&camera.jitter_frame[..2]);
        let jitter = [
            jitter,
            std::mem::replace(&mut self.last_jitter[eye_index], jitter),
        ];
        let now = std::time::Instant::now();
        let residency_eye = self.residency_eye.unwrap_or(frame.eye);
        let interval =
            self.last_eye[eye_index].map(|(_, at)| now.duration_since(at).as_secs_f64().max(1e-3));
        let velocity = match (self.last_eye[eye_index], interval) {
            (Some((eye, _)), Some(dt)) => (residency_eye - eye) / dt,
            _ => DVec3::ZERO,
        };
        self.last_eye[eye_index] = Some((residency_eye, now));
        // In view: the cone around the camera's forward reaching the
        // frustum's corners. A pixel's angle sets the cut's detail.
        let tan_y = 1.0 / f64::from(proj.y_axis.y);
        let view = residency::View {
            forward: glam::Vec3::from_slice(&camera.forward_far[..3])
                .as_dvec3()
                .normalize(),
            cos_half: frustum_cos_half(proj, Vec2::from_slice(&camera.jitter_frame[..2])),
        };
        self.config.plan.pixel_angle = 2.0 * tan_y / f64::from(ctx.height.max(1));
        let near = camera.position_near[3];
        // The scene's lights and their tile lists (light culling's), for
        // the terrain's shadows of them.
        let lights = ctx
            .scene_buffers
            .get(helio_core::BufferKey::of("scene_lights"))
            .map(|h| h.buffer.clone());
        let tile_lists = ctx
            .registry
            .get::<&wgpu::Buffer>(helio_core::ResourceKey::new("tile_light_lists"))
            .map(|b| (*b).clone());
        let tile_counts = ctx
            .registry
            .get::<&wgpu::Buffer>(helio_core::ResourceKey::new("tile_light_counts"))
            .map(|b| (*b).clone());
        // The GI probes' rays, when a GI pass runs before this one.
        let binding = |name: &str| ctx.registry.buffer_binding(name).cloned();
        let clipmap = ctx
            .registry
            .get::<helio_pass_gi::Clipmap>(helio_core::ResourceKey::new(helio_pass_gi::GI_CLIPMAP));
        let gi = match (
            binding(helio_pass_gi::GI_FRAME),
            binding(helio_pass_gi::GI_RAYS),
            binding(helio_pass_gi::GI_HITS),
            binding(helio_pass_gi::GI_SUN),
            clipmap,
            self.gi_eye,
        ) {
            (Some(frame_gi), Some(rays), Some(hits), Some(sun), Some(clipmap), Some(world_eye)) => {
                world.follow_gi(clipmap, world_eye, trace_eye);
                Some((frame_gi, rays, hits, sun))
            }
            _ => None,
        };
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
            residency_eye,
            velocity,
            Some(view),
            ctx.frame_num,
            &config,
        );
        if let Some((map, _)) = &self.scene {
            world.set_material_map(map);
        }
        world.set_steady_shadows(self.steady_shadows);
        // The atmosphere's sun disk sizes the shadows' penumbrae.
        world.set_atmosphere_frame(
            ctx.registry
                .get::<&wgpu::Buffer>(helio_core::ResourceKey::new(ATMOSPHERE_FRAME))
                .map(|b| (*b).clone()),
        );
        // The sun's occlusion by the terrain, for what else the sun lights
        // (fine cells and meshes in this draw, the GI probes' hits,
        // volumetric fog): the pages its readers asked for and lack.
        world.draw_sun_occlusion(encoder, ctx.frame_num, trace_eye, frame.sun);
        world.draw_for_eye(
            encoder,
            ctx.frame_num,
            trace_eye,
            view_proj,
            prev_view_proj,
            jitter,
            near,
            frame.sun,
            [ctx.width, ctx.height],
            Some(ctx.depth),
            match (&lights, &tile_lists, &tile_counts) {
                (Some(lights), Some(tile_lists), Some(tile_counts)) => Some(super::SceneLights {
                    lights,
                    tile_lists,
                    tile_counts,
                    eye: at,
                }),
                _ => None,
            }
            .as_ref(),
            gi.as_ref()
                .map(|(frame, rays, hits, sun)| super::GiRays {
                    frame,
                    rays,
                    hits,
                    sun,
                })
                .as_ref(),
            eye_index,
        );
        // Foliage's ground: where the drawn terrain changed, and this frame's
        // questions (`chunks::ground`) answered against the cut just drawn.
        self.foliage_ground
            .push_changes(world.take_ground_changes());
        if let Some(question) = self.foliage_ground.take_question() {
            let exchange = (
                binding(helio_pass_foliage_place::GROUND_QUERIES),
                binding(helio_pass_foliage_place::GROUND_HITS),
            );
            if let (Some(queries), Some(hits)) = exchange {
                world.probe_ground(
                    encoder,
                    &queries,
                    &hits,
                    &question.tiles,
                    &frame.planet,
                    trace_eye,
                    at,
                );
                self.foliage_ground.answer(question.batch);
            }
        }
        let scene_inputs = self.scene.as_ref().and_then(|(_, source)| source(ctx));
        world.publish_scene(encoder, gbuffer, ctx.depth, scene_inputs);
        if let Some(picks) = &frame.picks {
            world.copy_picks(encoder, picks);
        }
        Ok(())
    }
}

/// The voxel shader unprojects a direction from zero (`pixel_ray`), unlike
/// mesh projection which accepts geometry relative to any origin. Pair that
/// eye-relative matrix with the active camera's absolute position: the
/// frame's world origin plus the camera's position in the frame.
fn trace_camera(camera: &helio_core::GpuCameraUniforms, world_origin: DVec3) -> (DVec3, Mat4, Mat4) {
    let at = Vec3::from_slice(&camera.position_near[..3]);
    let eye = world_origin + at.as_dvec3();
    let from_eye = Mat4::from_translation(at);
    (
        eye,
        Mat4::from_cols_array(&camera.view_proj) * from_eye,
        // Previous VP is rebased into the CURRENT frame coordinates by Helio.
        Mat4::from_cols_array(&camera.prev_view_proj) * from_eye,
    )
}

/// XR frusta are asymmetric: include the offset on each axis so chunks at the
/// wider edge of either eye are not rejected by the residency cone. The
/// projection's per-frame jitter (an NDC translation, `jitter`) is not part of
/// the frustum: it would change the view every frame, and a view that changes
/// is planned again and never settles.
fn frustum_cos_half(proj: Mat4, jitter: Vec2) -> f64 {
    let proj = Mat4::from_translation(-jitter.extend(0.0)) * proj;
    let tan_x = (1.0 + f64::from(proj.z_axis.x).abs()) / f64::from(proj.x_axis.x).abs();
    let tan_y = (1.0 + f64::from(proj.z_axis.y).abs()) / f64::from(proj.y_axis.y).abs();
    (tan_x * tan_x + tan_y * tan_y).sqrt().atan().cos()
}

#[cfg(test)]
mod stereo_tests {
    use super::*;

    #[test]
    fn voxel_rays_and_mesh_projection_agree_for_moving_stereo_eyes() {
        let origin = DVec3::new(6_371_200.0, -1024.0, 2048.0);
        let old_head = Vec3::new(7.0, 2.0, -3.0);
        // Keep the app's earlier input pose unchanged while rendering newer
        // translated/rotated eyes, as OpenXR's predicted display poses do.
        let landmarks = [
            Vec3::new(8.0, 1.0, -10.0),
            Vec3::new(4.0, 4.0, -12.0),
            Vec3::new(10.0, -1.0, -20.0),
        ];
        for motion in [Vec3::ZERO, Vec3::new(0.3, -0.6, 0.2)] {
            for yaw in [0.0, 0.6, -0.4] {
                let orientation = glam::Quat::from_rotation_y(yaw)
                    * glam::Quat::from_rotation_x(-0.2)
                    * glam::Quat::from_rotation_z(0.15);
                for side in [-1.0, 1.0] {
                    let at = old_head + motion + orientation * Vec3::X * (side * 0.032);
                    let pose = Mat4::from_rotation_translation(orientation, at);
                    let mut proj = helio_core::depth::perspective(1.4, 1.1, 0.05);
                    proj.z_axis.x = side * 0.15;
                    let view = pose.inverse();
                    let camera = helio_core::GpuCameraUniforms::new(
                        view,
                        proj,
                        at,
                        0.05,
                        1000.0,
                        0,
                        [0.0; 2],
                        proj * view,
                    );
                    let (eye, vp, _) = trace_camera(&camera, origin);
                    assert!(eye.distance(origin + at.as_dvec3()) < 1e-9);
                    let rays = super::super::ray_plane(vp, [128, 128]);
                    for point in landmarks {
                        // Rasterized foliage sees this fixed world landmark.
                        let ndc = (proj * view).project_point3(point);
                        let expected = (point - at).normalize();
                        // Mirror chunk_draw.wgsl::pixel_ray: it unprojects a
                        // POINT and normalizes it, assuming the eye is zero.
                        let ray = vp
                            .inverse()
                            .project_point3(Vec3::new(ndc.x, ndc.y, 0.5))
                            .normalize();
                        assert!(
                            ray.distance(expected) < 2e-4,
                            "shader ray drift: {ray:?} != {expected:?}"
                        );
                        let pixel =
                            Vec2::new((ndc.x + 1.0) * 64.0 - 0.5, (1.0 - ndc.y) * 64.0 - 0.5);
                        let plane_ray = (Vec3::from_slice(&rays[0])
                            + Vec3::from_slice(&rays[1]) * pixel.x
                            + Vec3::from_slice(&rays[2]) * pixel.y)
                            .normalize();
                        assert!(plane_ray.distance(expected) < 5e-4, "ray plane drift");
                        let world_point = origin + point.as_dvec3();
                        let distance = world_point.distance(eye);
                        assert!((eye + ray.as_dvec3() * distance).distance(world_point) < 0.005);
                    }
                }
            }
        }
    }

    #[test]
    fn voxel_previous_projection_uses_current_eye_after_origin_rebase() {
        let origin = DVec3::new(6_371_200.0, 0.0, 0.0);
        let old_origin = origin - DVec3::new(128.0, 0.0, 0.0);
        let at = Vec3::new(0.1, 1.4, 0.0);
        let old_at = Vec3::new(127.9, 1.8, 0.1);
        let proj = helio_core::depth::perspective(1.4, 1.1, 0.05);
        let old_vp = proj * Mat4::from_translation(-old_at);
        let rebased_previous = old_vp * Mat4::from_translation((origin - old_origin).as_vec3());
        let camera = helio_core::GpuCameraUniforms::new(
            Mat4::from_translation(-at),
            proj,
            at,
            0.05,
            1000.0,
            0,
            [0.0; 2],
            rebased_previous,
        );
        let (eye, _, previous) = trace_camera(&camera, origin);
        let landmark = origin + DVec3::new(1.0, 2.0, -8.0);
        let expected = old_vp.project_point3((landmark - old_origin).as_vec3());
        let actual = previous.project_point3((landmark - eye).as_vec3());
        assert!(actual.distance(expected) < 1e-5);

    }

    #[test]
    fn residency_cone_contains_all_asymmetric_frustum_corners() {
        for offset in [-0.45, 0.0, 0.45] {
            let mut proj = helio_core::depth::perspective(1.4, 1.1, 0.05);
            proj.z_axis.x = offset;
            proj.z_axis.y = 0.2;
            let inverse = proj.inverse();
            let cos_half = frustum_cos_half(proj, Vec2::ZERO);
            for x in [-1.0, 1.0] {
                for y in [-1.0, 1.0] {
                    let ray = inverse.project_point3(Vec3::new(x, y, 1.0)).normalize();
                    assert!(
                        f64::from(-ray.z) + 1e-7 >= cos_half,
                        "corner {ray:?} was culled"
                    );
                }
            }
        }
    }

    #[test]
    fn residency_cone_ignores_projection_jitter() {
        let proj = helio_core::depth::perspective(1.2, 16.0 / 9.0, 0.05);
        let still = frustum_cos_half(proj, Vec2::ZERO);
        for frame in 0..8 {
            let [x, y] = helio_core::temporal::r1_r2_jitter(frame);
            let jitter = Vec2::new(x * 2.0 / 1600.0, y * 2.0 / 900.0);
            let jittered = Mat4::from_translation(jitter.extend(0.0)) * proj;
            assert_eq!(frustum_cos_half(jittered, jitter), still);
        }
    }
}
