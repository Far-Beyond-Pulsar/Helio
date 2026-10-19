//! The renderer: device and surface, the scene (sun, atmosphere, block
//! materials, the character), the voxel planet pass wired to the world, and
//! drawing a frame for the player's camera.

use crate::player::avatar::Avatar;
use crate::textures::{self, BlockTextures};
use crate::v3_demo_common::{directional_light, flush_scene_db, new_scene_db_with_gpu_mirror, scene_db_handle, set_camera_postprocess, spawn_light, update_light};
use glam::{DMat4, DVec3, Vec3};
use helio::{required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera, Renderer, RendererBuilder, RendererConfig};
use helio_default_graphs::{build_default_graph_external_with_voxel_passes, voxel_scene_materials, VoxelPassFactory};
use helio_pass_voxel_planet::chunks::pass::ChunkPass;
use helio_pass_voxel_planet::chunks::residency::PlanConfig;
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
    source: SharedPlanetFrame,
    sun_light: Entity,
    /// The renderer's world origin of the last frame.
    pub origin: DVec3,
    size: [u32; 2],
}

/// Residency settings for chunk trees at a window of height `h`.
fn chunk_config(h: u32) -> ResidencyConfig {
    let number = |name: &str, default: f64| std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default);
    ResidencyConfig {
        plan: PlanConfig {
            pixel_angle: 2.0 * std::f64::consts::FRAC_PI_8.tan() / f64::from(h),
            // Cells a pixel wide at most: one-block detail wherever it shows.
            lod_pixels: number("VOXEL_CRAFT_LOD", 1.2),
            offscreen_pixels: 2.0,
            view_guard: 0.35,
            coarsen: 0.8,
            // Far enough for a real horizon.
            max_distance: number("VOXEL_CRAFT_DISTANCE", 12_000.0),
            prefetch_seconds: 0.5,
            keep_frames: 120,
            motion_flow: 16.0,
            motion_max: 3.0,
        },
        tiles_per_frame: number("VOXEL_CRAFT_TILES", 256.0) as usize,
        generate_ms: ResidencyConfig::default().generate_ms,
        frame_ms: ResidencyConfig::default().frame_ms,
    }
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
    pub fn new(window: Arc<Window>, planet: &Planet, blocks: impl FnOnce(&wgpu::Device, &wgpu::Queue, &mut pulsar_scenedb::World) -> BlockTextures) -> Self {
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
        let textures = blocks(&device, &queue, &mut scene.world);
        let sun_light = spawn_light(&mut scene.world, directional_light([-0.3, -0.8, -0.4], [1.0, 0.96, 0.9], 3.0));
        let air = scene.world.spawn();
        scene.world.insert(air, helio_pass_sky::AtmosphereComponent::earth().around_planet([0.0; 3], planet.grid().radius()));
        let mut post = helio_pass_postprocess::PostProcessSettings::default();
        post.tonemap_operator = helio::TonemapOperator::Aces;
        post.tonemap_exposure = std::env::var("VOXEL_CRAFT_EXPOSURE").ok().and_then(|v| v.parse().ok()).unwrap_or(3.0);
        post.color_saturation = [1.12; 3];
        post.color_contrast = [1.06; 3];
        post.exposure_mode = helio_pass_postprocess::ExposureMode::Manual;
        set_camera_postprocess(&mut scene.world, 0, &post);
        let avatar = Avatar::spawn(&mut scene.world, |block| textures.side_material(block));
        scene.world.flush_gpu_mirror(&queue);

        // The planet pass draws the world's latest snapshot.
        let source: SharedPlanetFrame = Arc::new(Mutex::new(None));
        let pass_source = source.clone();
        let (map, appearance) = (textures.map, textures.appearance);
        let config = chunk_config(size[1]);
        let factory: VoxelPassFactory = Arc::new(move |_, _, _, _| {
            let mut pass = ChunkPass::with_config(pass_source.clone(), config);
            pass.set_appearance(Some(appearance));
            pass.set_scene_materials(map, voxel_scene_materials());
            Box::new(pass)
        });
        let mut renderer_config = RendererConfig::new(size[0], size[1], surface_format).with_tsr_quality(tsr_quality());
        renderer_config.enable_foliage = false;
        let handle = scene_db_handle(&scene).with_texture_store(textures.store).expect("texture store");
        let mut renderer = RendererBuilder::new(renderer_config, handle)
            .with_external_device()
            .with_pass_build_context(Box::new(move |ctx| build_default_graph_external_with_voxel_passes(ctx, vec![factory.clone()])))
            .build(device.clone(), queue.clone(), size[0], size[1], surface_format);
        textures::configure_sampler(&mut renderer);
        renderer.set_ambient([0.45, 0.58, 0.8], 0.35);
        Self { window, surface, surface_format, device, queue, renderer, scene, avatar, source, sun_light, origin: DVec3::ZERO, size }
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

    /// The pass's one-line diagnostics.
    pub fn diagnostics(&self) -> String {
        self.renderer.find_pass::<ChunkPass>().map(ChunkPass::diagnostics).unwrap_or_default()
    }

    /// Where the renderer's origin is for a camera at `eye`.
    pub fn origin_for(eye: DVec3) -> DVec3 {
        (eye / ORIGIN_CELL).round() * ORIGIN_CELL
    }

    /// Draws a frame of `view`.
    pub fn draw(&mut self, view: &View) {
        let (planet, eye, forward) = (view.planet, view.eye, view.look);
        let sun = view.sun.normalize();
        *self.source.lock().unwrap() = Some(PlanetFrame { eye, planet: planet.clone(), sun: sun.as_vec3(), picks: None });
        // The sun's light follows the sky.
        update_light(
            &mut self.scene.world,
            self.sun_light,
            directional_light(
                (-sun).as_vec3().to_array(),
                // Warm at the horizon, white overhead.
                {
                    let up = eye.normalize();
                    let height = sun.dot(up).clamp(-0.2, 1.0) as f32;
                    let warm = (1.0 - height * 2.2).clamp(0.0, 1.0);
                    [1.0, 0.97 - 0.25 * warm, 0.92 - 0.5 * warm]
                },
                3.0,
            ),
        );
        self.origin = Self::origin_for(eye);
        self.renderer.set_world_origin(Some(self.origin));
        let origin = self.origin;
        // The character and what it holds, and the targeted block's outline.
        self.avatar.pose(&mut self.scene.world, &mut self.renderer, view.player, (eye, forward), origin, view.time);
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
        if let Some(path) = view.screenshot {
            self.save_screenshot(&output.texture, path);
        }
        if view.luma {
            let pixels = self.read_pixels(&output.texture);
            let sum: u64 = pixels.chunks(4).map(|p| (u64::from(p[0]) * 3 + u64::from(p[1]) * 6 + u64::from(p[2])) / 10).sum();
            log::info!("LUMA {:.2}", sum as f64 / (pixels.len() / 4) as f64);
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
    pub time: f64,
    /// Save this frame to a PNG at this path.
    pub screenshot: Option<&'a str>,
    /// Log the frame's mean luminance and the planet pass's state.
    pub luma: bool,
}
