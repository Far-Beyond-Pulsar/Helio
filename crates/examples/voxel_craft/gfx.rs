//! The renderer: device and surface, the scene (sun, atmosphere, block
//! materials, the character), the voxel planet pass wired to the world, and
//! drawing a frame for the player's camera.

use crate::player::avatar::Avatar;
use crate::palette;
use crate::v3_demo_common::{directional_light, flush_scene_db, new_scene_db_with_gpu_mirror, scene_db_handle, set_camera_postprocess, spawn_light, update_light};
use glam::{DMat4, DVec3, Vec3};
use helio::{required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera, Renderer, RendererBuilder, RendererConfig};
use helio_default_graphs::{build_default_graph_external_with_voxel_passes, VoxelPassFactory};
use helio_pass_foliage_place::components::FoliageInteractorComponent;
use helio_pass_voxel_planet::chunks::pass::ChunkPass;
use helio_pass_voxel_planet::chunks::ResidencyConfig;
use helio_pass_voxel_planet::engine::{PlanetFrame, SharedPlanetFrame};
use helio_pass_voxel_planet::Planet;
use pulsar_scenedb::{Entity, SceneDb};
use std::sync::{Arc, Mutex};
use winit::window::Window;

/// Side of the cells the renderer's world origin snaps to (Pulsar's
/// `WORLD_ORIGIN_CELL`): scene positions stay small near the camera.
const ORIGIN_CELL: f64 = 1024.0;

pub struct Gfx {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_format: wgpu::TextureFormat,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    pub renderer: Renderer,
    pub scene: SceneDb,
    pub avatar: Avatar,
    /// The player's feet pressing the meadow (a `FoliageInteractorComponent` row).
    feet: Entity,
    source: SharedPlanetFrame,
    sun_light: Entity,
    /// The renderer's world origin of the last frame.
    pub origin: DVec3,
    size: [u32; 2],
    /// The weather's clock at the last frame (game seconds): the renderer's
    /// frame clock (the clouds' wind) advances with it.
    clock: f64,
    /// `VOXEL_CRAFT_TIMINGS=<frames>`: GPU milliseconds a pass, summed over
    /// that many frames and logged as their mean.
    timings: Option<PassTimings>,
}

/// Per-pass GPU time over a window of frames.
struct PassTimings {
    frames: u32,
    window: u32,
    last: u64,
    total: f64,
    passes: std::collections::BTreeMap<&'static str, f64>,
    /// The voxel pass's stages over the window: frames, then draw (seed and
    /// trace), seed, sun shadows, denoiser, generation (ms summed).
    voxel: (u32, [f64; 10]),
}

impl PassTimings {
    /// Adds the renderer's latest completed GPU frame, and logs the means every
    /// `window` frames.
    fn add(&mut self, snapshot: &helio::RenderTimingSnapshot, voxel: &[helio_pass_voxel_planet::chunks::FrameTimings], size: [u32; 2]) {
        for t in voxel {
            self.voxel.0 += 1;
            let [a, b, c, d, e] = t.shadow_stages;
            for (sum, ms) in self.voxel.1.iter_mut().zip([t.draw, t.seed, t.shadow, t.denoise, t.generate, a, b, c, d, e]) {
                *sum += ms;
            }
        }
        let Some(frame) = snapshot.gpu_frame_index.filter(|f| *f != self.last) else { return };
        self.last = frame;
        self.frames += 1;
        self.total += f64::from(snapshot.total_gpu_ms.unwrap_or(0.0));
        for pass in &snapshot.passes {
            *self.passes.entry(pass.name).or_default() += f64::from(pass.gpu_ms.unwrap_or(0.0));
        }
        if self.frames < self.window {
            return;
        }
        let n = f64::from(self.frames);
        let mut passes: Vec<_> = self.passes.iter().map(|(name, ms)| (*name, ms / n)).collect();
        passes.sort_by(|a, b| b.1.total_cmp(&a.1));
        let list: Vec<String> = passes.iter().filter(|(_, ms)| *ms >= 0.02).map(|(name, ms)| format!("{name} {ms:.3}")).collect();
        log::info!("[timings] {}x{} over {} frames: gpu {:.3} ms; {}", size[0], size[1], self.frames, self.total / n, list.join(", "));
        if self.voxel.0 > 0 {
            let [draw, seed, shadow, denoise, generate, sun, sky, hits, local, gi] = self.voxel.1.map(|ms| ms / f64::from(self.voxel.0));
            log::info!("[timings] voxel stages over {} frames: draw {draw:.3} (seed {seed:.3}), shadow {shadow:.3} (sun samples {sun:.3}, sky samples {sky:.3}, hits {hits:.3}, local {local:.3}, GI rays {gi:.3}), denoise {denoise:.3}, generate {generate:.3}", self.voxel.0);
        }
        self.voxel = (0, [0.0; 10]);
        self.frames = 0;
        self.total = 0.0;
        self.passes.clear();
    }
}

/// Residency settings for chunk trees at a window of height `h`.
fn chunk_config(h: u32) -> ResidencyConfig {
    let number = |name: &str, default: f64| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    let mut config = ResidencyConfig::default();
    config.plan.pixel_angle = 2.0 * std::f64::consts::FRAC_PI_8.tan() / f64::from(h);
    // Cells a pixel wide at most: one-block detail wherever it shows.
    config.plan.lod_pixels = number("VOXEL_CRAFT_LOD", 0.6);
    // Far enough for a real horizon.
    config.plan.max_distance = number("VOXEL_CRAFT_DISTANCE", 12_000.0);
    config
}

/// The sun along `direction` (from it), of `color`: it lights the haze too,
/// shadowed by everything (shafts).
fn sun_light(direction: [f32; 3], color: [f32; 3]) -> helio::GpuLight {
    let mut light = directional_light(direction, color, 3.0);
    light.god_rays_enabled = 1;
    light.god_rays_density = 1.0;
    light.god_rays_weight = 1.0;
    light.god_rays_exposure = 1.0;
    // Fully shadowed in the haze.
    light.god_rays_decay = 1.0;
    light
}

fn tsr_quality() -> helio_pass_tsr::TsrQuality {
    match std::env::var("VOXEL_CRAFT_QUALITY").as_deref() {
        Ok("performance") => helio_pass_tsr::TsrQuality::Performance,
        Ok("quality") => helio_pass_tsr::TsrQuality::Quality,
        Ok("native") => helio_pass_tsr::TsrQuality::Native,
        _ => helio_pass_tsr::TsrQuality::Balanced,
    }
}

impl Gfx {
    pub fn new(window: Arc<Window>, planet: &Planet) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: wgpu::InstanceFlags::empty(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });
        let surface = instance.create_surface(window.clone()).expect("surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("a GPU adapter");
        log::info!("adapter: {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("Voxel Craft"),
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .expect("device");
        device.on_uncaptured_error(Arc::new(|e: wgpu::Error| log::error!("[GPU] {e:?}")));
        let (device, queue) = (Arc::new(device), Arc::new(queue));

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps.formats.iter().find(|f| f.is_srgb()).copied().unwrap_or(caps.formats[0]);
        let size = window.inner_size();
        let size = [size.width.max(1), size.height.max(1)];
        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                format: surface_format,
                width: size[0],
                height: size[1],
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: caps.alpha_modes[0],
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );

        // The scene: SceneDB rows for the sun, the air, the blocks' materials
        // and the character.
        let mut scene = new_scene_db_with_gpu_mirror(&device, &queue);
        let palette = palette::load(&mut scene.world);
        // Grass and flowers on the world's grass: the engine's voxel meadow, with stands
        // of tall grass in places and a fresh breeze through it.
        let meadow = helio_pass_voxel_planet::meadow::meadow().with_tall_stands().with_wind([1.0, 0.35], 4.5, 0.8);
        for row in meadow.types {
            let entity = scene.world.spawn();
            scene.world.insert(entity, row);
        }
        let wind = scene.world.spawn();
        scene.world.insert(wind, meadow.wind);
        // The player's feet push it aside, and leave a track that rises behind them.
        let feet = scene.world.spawn();
        scene.world.insert(feet, FoliageInteractorComponent::new(DVec3::ZERO, 0.7, 1.0, Vec3::ZERO, 1.2));
        let sun_light = spawn_light(&mut scene.world, sun_light([-0.3, -0.8, -0.4], [1.0, 0.96, 0.9]));
        let air = scene.world.spawn();
        // Fair-weather cumulus, sized for the small planet: a layer from 0.9
        // to 2.2 km over the ground, drifting with a light breeze
        // (`VOXEL_CRAFT_NO_CLOUDS`: a clear sky).
        let clouds = helio_pass_sky::CloudLayer {
            bottom_km: 0.9,
            top_km: 2.2,
            coverage: 0.45,
            weather_km: 7.0,
            shape_km: 1.8,
            detail_km: 0.3,
            wind: [6.0, 0.0, 2.5],
            ..helio_pass_sky::CloudLayer::cumulus()
        };
        let sky = helio_pass_sky::AtmosphereComponent::earth().around_planet([0.0; 3], planet.grid().radius());
        let sky = if std::env::var_os("VOXEL_CRAFT_NO_CLOUDS").is_some() { sky } else { sky.with_clouds(clouds) };
        scene.world.insert(air, sky);
        // A light ground haze, thinning with altitude over the planet, the
        // sun scatters forward through: shafts past trees and ridges (the
        // terrain's sun occlusion shadows it).
        let haze = scene.world.spawn();
        scene.world.insert(
            haze,
            helio_pass_volumetric_fog::GlobalFogComponent { mode: 1, extinction: 0.001, height_falloff: 0.01, anisotropy: 0.6, ..Default::default() }
                .around_planet([0.0; 3], planet.grid().radius()),
        );
        let mut post = helio_pass_postprocess::PostProcessSettings::default();
        // AgX with its punchy look: sunlit leaves and grass roll toward
        // white, not lime, and the sky stays blue.
        post.tonemap_operator = helio::TonemapOperator::AgxPunchy;
        post.tonemap_exposure = 2.0;
        post.color_contrast = [1.06; 3];
        post.exposure_mode = helio_pass_postprocess::ExposureMode::Manual;
        set_camera_postprocess(&mut scene.world, 0, &post);
        let avatar = Avatar::spawn(&mut scene.world, |block| palette.hand_material(block));
        scene.world.flush_gpu_mirror(&queue);

        // The planet pass draws the world's latest snapshot.
        let source: SharedPlanetFrame = Arc::new(Mutex::new(None));
        let pass_source = source.clone();
        let appearance = palette.appearance;
        let config = chunk_config(size[1]);
        let factory: VoxelPassFactory = Arc::new(move |_, _, _, _| {
            let mut pass = ChunkPass::with_config(pass_source.clone(), config);
            pass.set_appearance(Some(appearance));
            Box::new(pass)
        });
        let mut renderer_config = RendererConfig::new(size[0], size[1], surface_format).with_tsr_quality(tsr_quality());
        // The meadow, grown on the terrain the planet pass draws
        // (`VOXEL_CRAFT_NO_FOLIAGE`: bare turf, for A/B measurements).
        renderer_config.enable_foliage = std::env::var_os("VOXEL_CRAFT_NO_FOLIAGE").is_none();
        renderer_config.foliage_tufts_per_m2 = Some(meadow.tufts_per_m2);
        renderer_config.foliage_blades_per_tuft = Some(meadow.blades_per_tuft);
        renderer_config.foliage_range = Some(meadow.range);
        let handle = scene_db_handle(&scene);
        let mut renderer = RendererBuilder::new(renderer_config, handle)
            .with_external_device()
            .with_pass_build_context(Box::new(move |ctx| build_default_graph_external_with_voxel_passes(ctx, vec![factory.clone()])))
            .build(device.clone(), queue.clone(), size[0], size[1], surface_format);
        renderer.set_ambient([0.45, 0.58, 0.8], 0.35);
        // `VOXEL_CRAFT_DEBUG=<mode>`: one of the renderer's debug views.
        if let Some(mode) = std::env::var("VOXEL_CRAFT_DEBUG").ok().and_then(|v| v.parse().ok()) {
            renderer.set_debug_mode(mode);
        }
        let timings = std::env::var("VOXEL_CRAFT_TIMINGS").ok().and_then(|v| v.parse().ok()).map(|window: u32| PassTimings {
            frames: 0,
            window: window.max(1),
            last: u64::MAX,
            total: 0.0,
            passes: Default::default(),
            voxel: (0, [0.0; 10]),
        });
        Self { window, surface, surface_format, device, queue, renderer, scene, avatar, feet, source, sun_light, origin: DVec3::ZERO, size, clock: 0.0, timings }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.size = [width, height];
        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                format: self.surface_format,
                width,
                height,
                present_mode: wgpu::PresentMode::Fifo,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
                color_space: wgpu::SurfaceColorSpace::Auto,
            },
        );
        self.renderer.set_render_size(width, height);
    }

    /// Whether the planet pass is still generating terrain for this view.
    pub fn settled(&self) -> bool {
        self.renderer.find_pass::<ChunkPass>().is_none_or(ChunkPass::settled)
    }

    /// Where the renderer's origin is for a camera at `eye`.
    pub fn origin_for(eye: DVec3) -> DVec3 {
        (eye / ORIGIN_CELL).round() * ORIGIN_CELL
    }

    /// Draws a frame of `view`.
    pub fn draw(&mut self, view: &View) {
        let (planet, eye, forward) = (view.planet, view.eye, view.look);
        let sun = view.sun.normalize();
        *self.source.lock().unwrap() = Some(PlanetFrame { eye, planet: planet.clone(), sun: sun.as_vec3(), picks: None, kept: None, export: None });
        // The sun's light follows the sky.
        update_light(
            &mut self.scene.world,
            self.sun_light,
            sun_light(
                (-sun).as_vec3().to_array(),
                // Warm at the horizon, white overhead.
                {
                    let up = eye.normalize();
                    let height = sun.dot(up).clamp(-0.2, 1.0) as f32;
                    let warm = (1.0 - height * 2.2).clamp(0.0, 1.0);
                    [1.0, 0.97 - 0.25 * warm, 0.92 - 0.5 * warm]
                },
            ),
        );
        self.renderer.set_frame_clock_delta(Some((view.weather - self.clock).clamp(0.0, 0.1) as f32));
        self.clock = view.weather;
        self.origin = Self::origin_for(eye);
        self.renderer.set_world_origin(Some(self.origin));
        let origin = self.origin;
        // The character and what it holds, and the targeted block's outline.
        self.avatar.pose(&mut self.scene.world, &mut self.renderer, view.player, (eye, forward), origin, view.time);
        if let Some(mut feet) = self.scene.world.get_mut::<FoliageInteractorComponent>(self.feet) {
            feet.set_motion(view.player.position, view.player.velocity.as_vec3());
        }
        self.renderer.debug_clear();
        for (a, b) in view.outline {
            self.renderer.debug_line((*a - origin).as_vec3().to_array(), (*b - origin).as_vec3().to_array(), [0.05, 0.05, 0.05, 1.0]);
        }
        // A crosshair fixed in front of the eye.
        if !view.player.third_person {
            let look = forward.normalize();
            let right = look.cross(eye.normalize()).normalize_or_zero();
            let above = right.cross(look);
            let centre = eye + look * 0.6;
            let reach = 0.0045;
            for (a, b) in [(-right, right), (-above, above)] {
                self.renderer.debug_line(((centre + a * reach) - origin).as_vec3().to_array(), ((centre + b * reach) - origin).as_vec3().to_array(), [1.0, 1.0, 1.0, 0.9]);
            }
        }

        let up = eye.normalize().as_vec3();
        let forward = forward.normalize().as_vec3();
        let up = if forward.dot(up).abs() > 0.999 { up.any_orthonormal_vector() } else { up };
        let near = (planet.air_clearance(eye) * 0.25).clamp(0.05, 50_000.0) as f32;
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        let mut camera = Camera::perspective_look_at(Vec3::ZERO, forward, up, 70f32.to_radians(), aspect, near, 40_000_000.0);
        let local = eye - origin;
        camera.position = local.as_vec3();
        camera.view = (camera.view.as_dmat4() * DMat4::from_translation(-local)).as_mat4();

        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
            _ => return,
        };
        let target = output.texture.create_view(&Default::default());
        flush_scene_db(&self.scene, &self.queue);
        if let Err(error) = self.renderer.render(&camera, &target) {
            log::error!("render: {error:?}");
        }
        if let Some(timings) = &mut self.timings {
            let voxel = self.renderer.find_pass_mut::<ChunkPass>().map(ChunkPass::take_timings).unwrap_or_default();
            timings.add(self.renderer.timing_snapshot(), &voxel, self.size);
        }
        if let Some(path) = view.screenshot {
            self.save_screenshot(&output.texture, path);
        }
        self.queue.present(output);
    }

    /// Writes the frame just drawn to `path` (PNG).
    fn save_screenshot(&self, texture: &wgpu::Texture, path: &str) {
        let [w, h] = self.size;
        let pixels = self.read_pixels(texture);
        let _ = image::save_buffer(path, &pixels, w, h, image::ColorType::Rgba8);
    }

    /// The frame just drawn as RGBA8 rows.
    fn read_pixels(&self, texture: &wgpu::Texture) -> Vec<u8> {
        let [w, h] = self.size;
        let row = (w * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: u64::from(row) * u64::from(h), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        let bgra = matches!(self.surface_format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
        let mut pixels = Vec::with_capacity((w * h * 4) as usize);
        for line in data.chunks(row as usize).take(h as usize) {
            for p in line[..(w * 4) as usize].chunks(4) {
                pixels.extend_from_slice(&if bgra { [p[2], p[1], p[0], 255] } else { [p[0], p[1], p[2], 255] });
            }
        }
        drop(data);
        pixels
    }
}

/// What a frame shows.
pub struct View<'a> {
    pub planet: &'a Arc<Planet>,
    pub player: &'a crate::player::Player,
    /// The camera: position and look direction (planet-centred).
    pub eye: DVec3,
    pub look: DVec3,
    /// Direction to the sun.
    pub sun: DVec3,
    /// Edges to outline (the targeted block), planet-centred.
    pub outline: &'a [(DVec3, DVec3)],
    /// Seconds since the game started (the character's animation).
    pub time: f64,
    /// Game seconds (they hold still in the tour and race with `T`): the
    /// clouds drift with them.
    pub weather: f64,
    /// Save this frame to a PNG at this path.
    pub screenshot: Option<&'a str>,
}
