//! Voxel planet terrain in Helio's renderer, drawn from chunk trees
//! (docs/chunk-trees.md): held views with GPU timings, streaming flights,
//! digging, picks and captures.
//!
//! cargo run -p helio-default-graphs --release --example voxel_flight -- OUTPUT [WIDTH HEIGHT [native|quality]]
//!
//! Modes (environment):
//! * `HELIO_VOXEL_FLIGHT_HOLD=ground|mountain_slope|tunnel|face_edge|cave|eye`
//!   settles one view (`HELIO_VOXEL_FLIGHT_EYE=x,y,z,fx,fy,fz` for `eye`)
//!   and holds it for `HELIO_VOXEL_FLIGHT_HOLD_SECONDS`: drawn by
//!   `ChunkWorld` alone (`CHUNKHOLD`: its GPU stages), or with
//!   `HELIO_VOXEL_CHUNKS_ENGINE=1` by the renderer's chunk pass
//!   (`CHUNKENGINE`: the frame as a host draws it). With
//!   `HELIO_VOXEL_CHUNKS_DIG=n` the view is dug and undone, with
//!   `HELIO_VOXEL_CHUNKS_BLAST=r` blasted by one ball `r` m across.
//! * `HELIO_VOXEL_CHUNKS_FLY=descent|cruise` streams along a path.
//! * `HELIO_VOXEL_FLIGHT_STRESS=ascent|descent|cruise|orbit|jump|dig|yoyo|dive|glide` flies through the
//!   renderer in real time, every frame recorded (`stress_<mode>.csv`),
//!   samples saved with settled references to compare.
//!
//! World: `HELIO_VOXEL_FLIGHT_VOXEL` (0.1 m), `_SEED` (7), `_PRESET`
//! (earth|moon|desert), `_HEIGHTFIELD`, `_NO_CAVES`, `_NO_OVERHANGS`,
//! `_GRAIN=scale_m,octaves,ratio`, `_SUN=x,y,z`, `_SUN_ELEVATION=deg` (the
//! terrain's sun that high over each view's horizon toward `_SUN`'s azimuth;
//! the scene light keeps `_SUN`), `_NO_SUN`,
//! `_LOOK=exposure,contrast,saturation`, `_DEBUG=<mode>`.
//!
//! Frames are pipelined like a game loop (at most `FRAMES_IN_FLIGHT` ahead
//! of the GPU): waiting for every frame lets the driver lower its clocks,
//! which inflates every GPU timing.

use glam::{DVec3, Vec3};
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera, Renderer,
    RendererBuilder, RendererConfig,
};
use helio_default_graphs::{build_default_graph_external_with_voxel_passes, VoxelPassFactory};
use helio_pass_voxel_planet::chunks::pass::ChunkPass;
use helio_pass_voxel_planet::engine::{PlanetFrame, SharedPicks, SharedPlanetFrame};
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Planet, PlanetRecipe};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Side of the cells the engine frames' world origin snaps to (Pulsar's
/// `WORLD_ORIGIN_CELL`).
const ORIGIN_CELL: f64 = 1024.0;

const FRAMES_IN_FLIGHT: usize = 2;

struct Flight {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    source: SharedPlanetFrame,
    planet: Arc<Planet>,
    target: wgpu::Texture,
    size: [u32; 2],
    sun: Vec3,
    /// Degrees over the eye's horizon (`HELIO_VOXEL_FLIGHT_SUN_ELEVATION`).
    sun_elevation: Option<f32>,
    shadows: bool,
    output: PathBuf,
    /// Pipelined frames not yet known complete.
    in_flight: std::collections::VecDeque<Arc<std::sync::atomic::AtomicBool>>,
    /// Pick requests drawn frames answer (`pick_probe`).
    picks: Option<SharedPicks>,
}

/// The planet's up at `eye`.
fn up_for(eye: DVec3) -> Vec3 {
    eye.normalize().as_vec3()
}

/// Horizontal heading at `eye` (0 = local east).
fn tangent(eye: DVec3, heading: f64) -> Vec3 {
    let up = up_for(eye).as_dvec3();
    let east = DVec3::Y.cross(up).try_normalize().unwrap_or(DVec3::X);
    let north = up.cross(east);
    (east * heading.cos() + north * heading.sin()).as_vec3()
}

fn look(eye: DVec3, heading: f64, pitch_deg: f64) -> Vec3 {
    let up = up_for(eye);
    let h = tangent(eye, heading);
    let p = pitch_deg.to_radians() as f32;
    (h * p.cos() + up * p.sin()).normalize()
}

impl Flight {
    /// The sun for an eye at `eye`: `HELIO_VOXEL_FLIGHT_SUN_ELEVATION`
    /// degrees over its horizon toward `sun`'s azimuth, else `sun`.
    fn sun_at(&self, eye: DVec3) -> Vec3 {
        let Some(elevation) = self.sun_elevation else { return self.sun };
        let up = up_for(eye);
        let level = (self.sun - up * self.sun.dot(up)).try_normalize().unwrap_or_else(|| up.any_orthonormal_vector());
        let e = elevation.to_radians();
        (level * e.cos() + up * e.sin()).normalize()
    }

    fn new(output: &Path, size: [u32; 2], quality: helio_pass_tsr::TsrQuality, planet: Planet) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).expect("GPU required");
        eprintln!("VOXEL_FLIGHT_ADAPTER {:?}", adapter.get_info());
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: required_wgpu_features(adapter.features()),
            required_limits: required_wgpu_limits(adapter.limits()),
            experimental_features: required_experimental_features(adapter.features()),
            ..Default::default()
        }))
        .unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let context = EngineGpuContext::new(device.clone(), queue.clone());
        let mut store = SceneGpuStore::new(
            &context,
            SceneGpuConfig { classes: vec![], tombstone_headroom: 0, max_cells_metadata: 0 },
        );
        helio_pass_sky::AtmosphereComponent::register_gpu_columns_growable(&mut store, 4, &device);
        helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(&mut store, 16, &device);
        helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(&mut store, 16, &device);
        helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(&mut store, 16, &device);
        helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(&mut store, 16, &device);
        let mirror = GpuMirrorHandle::new(Arc::new(store), queue.clone());
        let mut scene = pulsar_scenedb::SceneDb::new();
        scene.world.attach_gpu_mirror(mirror.clone());
        // HELIO_VOXEL_FLIGHT_SUN="x,y,z": direction towards the sun.
        let sun_dir = std::env::var("HELIO_VOXEL_FLIGHT_SUN")
            .ok()
            .and_then(|v| {
                let c: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                (c.len() == 3).then(|| Vec3::new(c[0], c[1], c[2]))
            })
            .unwrap_or(Vec3::new(0.35, 0.75, 0.45))
            .normalize();
        let light = scene.world.spawn();
        scene.world.insert(
            light,
            helio_pass_forward_lit::LightComponent::from(helio::GpuLight {
                position_range: [0.0, 0.0, 0.0, f32::MAX],
                direction_outer: [-sun_dir.x, -sun_dir.y, -sun_dir.z, 0.0],
                // The sun above the air: the atmosphere colours it.
                color_intensity: [1.0, 1.0, 1.0, 3.0],
                shadow_index: u32::MAX,
                light_type: helio::LightType::Directional as u32,
                ..Default::default()
            }),
        );
        // Earth's air around the planet, centred on the world origin.
        let air = scene.world.spawn();
        scene.world.insert(air, helio_pass_sky::AtmosphereComponent::earth().around_planet([0.0; 3], planet.grid().radius()));
        scene.world.flush_gpu_mirror(&queue);
        let source: SharedPlanetFrame = Arc::new(Mutex::new(None));
        let pass_source = source.clone();
        // The renderer's pass generates as a host's does: the engine's
        // milliseconds a frame unless the environment sets them.
        let mut config = chunk_config(size[1], 256);
        config.generate_ms = config.generate_ms.or(helio_pass_voxel_planet::chunks::ResidencyConfig::default().generate_ms);
        config.frame_ms = config.frame_ms.or(helio_pass_voxel_planet::chunks::ResidencyConfig::default().frame_ms);
        let factory: VoxelPassFactory = Arc::new(move |_, _, _, _| Box::new(ChunkPass::with_config(pass_source.clone(), config)));
        let mut config = RendererConfig::new(size[0], size[1], wgpu::TextureFormat::Rgba8Unorm).with_tsr_quality(quality);
        config.enable_foliage = false;
        let mut renderer = RendererBuilder::new(config, mirror)
            .with_external_device()
            .with_pass_build_context(Box::new(move |ctx| build_default_graph_external_with_voxel_passes(ctx, vec![factory.clone()])))
            .build(device.clone(), queue.clone(), size[0], size[1], config.surface_format);
        // HELIO_VOXEL_FLIGHT_LOOK="exposure,contrast,saturation": the camera
        // post-process of an outdoor look (ACES tone map and a grade).
        if let Ok(look) = std::env::var("HELIO_VOXEL_FLIGHT_LOOK") {
            let v: Vec<f32> = look.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            let at = |i: usize, d: f32| v.get(i).copied().unwrap_or(d);
            let mut post = helio_pass_postprocess::PostProcessSettings::default();
            post.tonemap_operator = helio::TonemapOperator::Aces;
            post.tonemap_exposure = at(0, 1.0);
            post.color_contrast = [at(1, 1.0); 3];
            post.color_saturation = [at(2, 1.0); 3];
            let camera = scene.world.spawn();
            scene.world.insert(camera, helio_pass_postprocess::CameraPostProcessComponent::new(0, &post));
            scene.world.flush_gpu_mirror(&queue);
        }
        // The mirror owns the uploaded rows for the lifetime of the flight.
        std::mem::forget(scene);
        if let Some(mode) = std::env::var("HELIO_VOXEL_FLIGHT_DEBUG").ok().and_then(|v| v.parse().ok()) {
            renderer.set_debug_mode(mode);
        }
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("flight output"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self {
            device,
            queue,
            renderer,
            source,
            planet: Arc::new(planet),
            target,
            size,
            sun: sun_dir,
            sun_elevation: std::env::var("HELIO_VOXEL_FLIGHT_SUN_ELEVATION").ok().and_then(|v| v.parse().ok()),
            shadows: std::env::var_os("HELIO_VOXEL_FLIGHT_NO_SUN").is_none(),
            output: output.to_path_buf(),
            in_flight: Default::default(),
            picks: None,
        }
    }

    /// Draw a frame from `eye` along `forward`, blocking only on the frame
    /// `FRAMES_IN_FLIGHT` behind it (a game loop keeps the GPU fed): the
    /// milliseconds recording it took and those waiting.
    fn draw(&mut self, eye: DVec3, forward: Vec3) -> (f64, f64) {
        let started = Instant::now();
        *self.source.lock().unwrap() = Some(PlanetFrame { eye, planet: self.planet.clone(), sun: self.sun_at(eye), shadows: self.shadows, picks: self.picks.clone() });
        // As a host's camera-relative frames: the origin stays on a cell
        // corner while the camera moves within the cell, off the origin.
        let origin = (eye / ORIGIN_CELL).round() * ORIGIN_CELL;
        self.renderer.set_world_origin(Some(origin));
        let up = up_for(eye);
        let forward = forward.normalize();
        let up = if forward.dot(up).abs() > 0.999 { up.any_orthonormal_vector() } else { up };
        let near = (self.planet.air_clearance(eye) * 0.25).clamp(0.05, 50_000.0) as f32;
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        let mut camera = Camera::perspective_look_at(Vec3::ZERO, forward, up, std::f32::consts::FRAC_PI_4, aspect, near, 40_000_000.0);
        let local = eye - origin;
        camera.position = local.as_vec3();
        camera.view = (camera.view.as_dmat4() * glam::DMat4::from_translation(-local)).as_mat4();
        self.renderer.render(&camera, &self.target.create_view(&Default::default())).unwrap();
        let recorded = Instant::now();
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        self.queue.on_submitted_work_done(move || flag.store(true, std::sync::atomic::Ordering::Release));
        self.in_flight.push_back(done);
        while self.in_flight.len() > FRAMES_IN_FLIGHT {
            let oldest = self.in_flight.pop_front().unwrap();
            while !oldest.load(std::sync::atomic::Ordering::Acquire) {
                let _ = self.device.poll(wgpu::PollType::Poll);
                std::thread::sleep(std::time::Duration::from_micros(50));
            }
        }
        ((recorded - started).as_secs_f64() * 1e3, recorded.elapsed().as_secs_f64() * 1e3)
    }

    /// The last frame as `<out>/<name>.png`.
    fn capture(&self, name: &str) {
        let row = (self.size[0] * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row) * u64::from(self.size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(self.size[1]) },
            },
            self.target.size(),
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        let pixels: Vec<u8> = data.chunks(row as usize).flat_map(|r| r[..self.size[0] as usize * 4].to_vec()).collect();
        image::save_buffer(self.output.join(format!("{name}.png")), &pixels, self.size[0], self.size[1], image::ColorType::Rgba8).unwrap();
    }
}

fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[((v.len() as f64 - 1.0) * p).round() as usize]
}

fn land_near(planet: &Planet, face: u8, fi: f64, fj: f64, min_height_m: f64) -> DVec3 {
    let grid = planet.grid();
    let n = f64::from(grid.cells());
    let min_top = (min_height_m / grid.voxel_size()) as i32;
    for step in 0..4000 {
        let a = fi + 0.002 * f64::from(step % 60);
        let b = fj + 0.002 * f64::from(step / 60);
        let (i, j) = ((a * n) as i32, (b * n) as i32);
        if planet.column_top(face, i, j, 0) > min_top {
            return grid.direction(face, f64::from(i) + 0.5, f64::from(j) + 0.5);
        }
    }
    panic!("no land");
}

/// Highest terrain within `span` (face fraction) of face coordinates
/// (fi, fj), sampled on a coarse level: (direction, height above datum m).
fn highest_near(planet: &Planet, face: u8, fi: f64, fj: f64, span: f64) -> (DVec3, f64) {
    let grid = planet.grid();
    let level = 10u32;
    let n = f64::from(grid.cells() >> level);
    let size = f64::from(1u32 << level);
    let mut best = (DVec3::ZERO, f64::MIN);
    for a in 0..96 {
        for b in 0..96 {
            let u = (fi + span * (f64::from(a) / 95.0 * 2.0 - 1.0)).clamp(0.0, 0.999);
            let v = (fj + span * (f64::from(b) / 95.0 * 2.0 - 1.0)).clamp(0.0, 0.999);
            let (i, j) = ((u * n) as i32, (v * n) as i32);
            let h = f64::from(planet.column_top(face, i, j, level)) * grid.voxel_size() * size;
            if h > best.1 {
                best = (grid.direction(face, (f64::from(i) + 0.5) * size, (f64::from(j) + 0.5) * size), h);
            }
        }
    }
    best
}

/// Residency settings for chunk trees at the flight's resolution
/// (`HELIO_VOXEL_LOD_PIXELS`, `HELIO_VOXEL_CHUNKS_BUDGET` tiles a frame).
fn chunk_config(h: u32, budget: usize) -> helio_pass_voxel_planet::chunks::ResidencyConfig {
    use helio_pass_voxel_planet::chunks::{residency::PlanConfig, ResidencyConfig};
    let lod: f64 = std::env::var("HELIO_VOXEL_LOD_PIXELS").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let budget = std::env::var("HELIO_VOXEL_CHUNKS_BUDGET").ok().and_then(|v| v.parse().ok()).unwrap_or(budget);
    ResidencyConfig {
        plan: PlanConfig {
            pixel_angle: 2.0 * std::f64::consts::FRAC_PI_8.tan() / f64::from(h),
            lod_pixels: lod,
            coarsen: 0.8,
            max_distance: 400_000.0,
            prefetch_seconds: 0.5,
            keep_frames: 120,
        },
        tiles_per_frame: budget,
        // `HELIO_VOXEL_CHUNKS_GENERATE_MS`: GPU generation a frame.
        generate_ms: std::env::var("HELIO_VOXEL_CHUNKS_GENERATE_MS").ok().and_then(|v| v.parse().ok()),
        // `HELIO_VOXEL_CHUNKS_FRAME_MS`: the frame's GPU milliseconds
        // generation may grow into (the renderer pass only).
        frame_ms: std::env::var("HELIO_VOXEL_CHUNKS_FRAME_MS").ok().and_then(|v| v.parse().ok()),
    }
}

/// Run chunk residency for a still eye until nothing is left to generate.
fn chunk_settle(flight: &Flight, world: &mut helio_pass_voxel_planet::chunks::ChunkWorld, eye: DVec3, frame: &mut u64) -> f64 {
    let config = chunk_config(flight.size[1], 512);
    let started = Instant::now();
    loop {
        let mut encoder = flight.device.create_command_encoder(&Default::default());
        let cpu = Instant::now();
        let stats = world.update(&mut helio_core::CommandRecorder::from_encoder(&mut encoder), eye, DVec3::ZERO, None, *frame, &config);
        if *frame % 50 == 0 || std::env::var_os("HELIO_VOXEL_CHUNKS_TRACE").is_some() {
            eprintln!("CHUNKSETTLE {} cpu {:.1} ms {stats:?}", *frame, cpu.elapsed().as_secs_f64() * 1e3);
        }
        flight.queue.submit([encoder.finish()]);
        world.submitted();
        *frame += 1;
        if world.settled() {
            return started.elapsed().as_secs_f64();
        }
        if *frame % 8 == 0 {
            let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
        }
    }
}

/// Hold `view` drawn from chunk trees: build the cut for the still eye,
/// then draw it for `HELIO_VOXEL_FLIGHT_HOLD_SECONDS` with pipelined
/// frames and print the median primary time (`CHUNKHOLD`). With
/// `HELIO_VOXEL_CHUNKS_PNG=<path>` the colour target is saved.
fn chunk_hold(flight: &Flight, view: &str, eye: DVec3, forward: Vec3) {
    use helio_pass_voxel_planet::chunks::ChunkWorld;
    let [w, h] = flight.size;
    // Report this mode's errors (the run's validation scope is never popped
    // in hold mode).
    let mut internal = Some(flight.device.push_error_scope(wgpu::ErrorFilter::Internal));
    let mut scope = Some(flight.device.push_error_scope(wgpu::ErrorFilter::Validation));
    let mut world = ChunkWorld::new(&flight.device, &flight.queue, &flight.planet);
    let mut frame = 0u64;
    let seconds = chunk_settle(flight, &mut world, eye, &mut frame);
    eprintln!("CHUNKS settled {view} in {seconds:.2} s ({frame} frames): {} chunks drawn, {:.1} MB", world.chunk_count(), world.resident_bytes() as f64 / 1048576.0);
    let up = up_for(eye);
    let forward = forward.normalize();
    let up = if forward.dot(up).abs() > 0.999 { up.any_orthonormal_vector() } else { up };
    let near = (flight.planet.air_clearance(eye) * 0.25).clamp(0.05, 50_000.0) as f32;
    let camera = Camera::perspective_look_at(Vec3::ZERO, forward, up, std::f32::consts::FRAC_PI_4, w as f32 / h as f32, near, 40_000_000.0);
    let seconds: f64 = std::env::var("HELIO_VOXEL_FLIGHT_HOLD_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(120.0);
    let started = Instant::now();
    let mut pending = std::collections::VecDeque::new();
    let mut times = Vec::new();
    let mut seed_times = Vec::new();
    let mut shadow_times = Vec::new();
    let first = frame;
    let mut draw_cpu = Vec::new();
    while started.elapsed().as_secs_f64() < seconds {
        let mut encoder = flight.device.create_command_encoder(&Default::default());
        let cpu = Instant::now();
        world.draw(&mut helio_core::CommandRecorder::from_encoder(&mut encoder), frame, eye, camera.proj * camera.view, camera.proj * camera.view, near, flight.sun_at(eye), [w, h]);
        draw_cpu.push(cpu.elapsed().as_secs_f64() * 1e3);
        pending.push_back(flight.queue.submit([encoder.finish()]));
        world.submitted();
        // Two frames in flight at most.
        if pending.len() > 2 {
            let _ = flight.device.poll(wgpu::PollType::Wait { submission_index: pending.pop_front(), timeout: None });
        }
        // The first frames compile and warm caches.
        for t in world.take_timings().into_iter().filter(|t| t.frame >= first + 30) {
            times.push(t.draw);
            seed_times.push(t.seed);
            shadow_times.push(t.shadow);
        }
        frame += 1;
        if frame == first + 2 {
            if let Some(error) = pollster::block_on(scope.take().unwrap().pop()) {
                eprintln!("CHUNKS validation: {error}");
            }
            if let Some(error) = pollster::block_on(internal.take().unwrap().pop()) {
                eprintln!("CHUNKS internal: {error}");
            }
        }
    }
    eprintln!("CHUNKS draw cpu p50 {:.2} ms", percentile(&draw_cpu, 0.5));
    let (outside, inside) = world.drawn();
    if std::env::var_os("HELIO_VOXEL_CHUNKS_COUNT").is_some() {
        world.take_counts();
        let mut encoder = flight.device.create_command_encoder(&Default::default());
        world.draw(&mut helio_core::CommandRecorder::from_encoder(&mut encoder), frame, eye, camera.proj * camera.view, camera.proj * camera.view, near, flight.sun_at(eye), [w, h]);
        flight.queue.submit([encoder.finish()]);
        let counts = world.take_counts();
        if std::env::var_os("HELIO_VOXEL_CHUNKS_DEBUG_PIXEL").is_some() {
            let mut encoder = flight.device.create_command_encoder(&Default::default());
            world.draw(&mut helio_core::CommandRecorder::from_encoder(&mut encoder), frame + 1, eye, camera.proj * camera.view, camera.proj * camera.view, near, flight.sun_at(eye), [w, h]);
            flight.queue.submit([encoder.finish()]);
            for (n, step) in world.pixel_trace().iter().enumerate() {
                eprintln!("PIXELTRACE {n}: {step:?}");
            }
        }
        let pixels = f64::from(w * h);
        eprintln!(
            "CHUNKCOUNT {view}: seeded {:.2} hits {:.2} (of pixels) steps/seeded {:.1} boxes/seeded {:.2}; missed {:.4} of pixels, {:.1} boxes each, {} lost",
            f64::from(counts.seeded) / pixels,
            f64::from(counts.hits) / pixels,
            f64::from(counts.steps) / f64::from(counts.seeded.max(1)),
            f64::from(counts.boxes) / f64::from(counts.seeded.max(1)),
            f64::from(counts.missed) / pixels,
            f64::from(counts.missed_boxes) / f64::from(counts.missed.max(1)),
            counts.lost
        );
    }
    eprintln!(
        "CHUNKHOLD {view} n={} primary {:.3} p95 {:.3} seed {:.3} shadow {:.3} boxes {outside}+{inside} chunks {}",
        times.len(),
        percentile(&times, 0.5),
        percentile(&times, 0.95),
        percentile(&seed_times, 0.5),
        percentile(&shadow_times, 0.5),
        world.chunk_count(),
    );
    // `HELIO_VOXEL_CHUNKS_PROBE=x,y;x,y`: each pixel's chunk hit against
    // the exact CPU raycast.
    if let Ok(probes) = std::env::var("HELIO_VOXEL_CHUNKS_PROBE") {
        let read = |texture: &wgpu::Texture, x: u32, y: u32| -> [u8; 8] {
            let buffer = flight.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut encoder = flight.device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) } },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            flight.queue.submit([encoder.finish()]);
            buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
            flight.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let data = buffer.slice(..).get_mapped_range().unwrap();
            let mut out = [0u8; 8];
            out.copy_from_slice(&data[..8]);
            out
        };
        let targets = world.targets().unwrap();
        let (visibility, distance) = (&targets.visibility, &targets.distance);
        let inverse = (camera.proj * camera.view).inverse();
        for probe in probes.split(';') {
            let xy: Vec<u32> = probe.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            let (x, y) = (xy[0], xy[1]);
            let v = read(visibility, x, y);
            let t = f32::from_le_bytes(read(distance, x, y)[..4].try_into().unwrap());
            let ndc = Vec3::new((x as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / h as f32 * 2.0, 0.5);
            let far = inverse * ndc.extend(1.0);
            let dir = (far.truncate() / far.w).normalize().as_dvec3();
            let slot = u32::from_le_bytes(v[..4].try_into().unwrap());
            let cell = u32::from_le_bytes(v[4..].try_into().unwrap());
            let cpu = flight.planet.raycast(eye, dir, 200_000.0);
            eprintln!(
                "PROBE {x},{y}: chunk t {:.2} slot {slot} {:?} cell ({},{},{}) axis {} material {} | cpu {:?}",
                if t > 0.0 { t } else { f32::NAN },
                world.chunk_at(slot),
                cell & 63,
                (cell >> 6) & 63,
                (cell >> 12) & 63,
                (cell >> 18) & 7,
                cell >> 21,
                cpu.map(|h| (h.distance, h.cell, flight.planet.material(h.cell)))
            );
            if let Some(h) = cpu {
                eprintln!("  over the cpu hit: {:?}", world.chunks_over(h.cell.face, h.cell.i, h.cell.j));
            }
        }
    }
    if let Ok(path) = std::env::var("HELIO_VOXEL_CHUNKS_PNG") {
        save_chunk_colour(flight, &world, &path);
    }
}

/// `HELIO_VOXEL_CHUNKS_DIG=n`: chunk trees following edits. Settle the
/// view, then dig `n` balls (1.5 m) one a frame along a line across the
/// view centre's hit (from the centre out), then undo them one a frame: what each edit generates
/// again and costs, when it shows (the centre pixel's depth), and whether
/// the undone frame's hits match the first's. Saves
/// `<out>/dig_{before,dug,undone}.png`. With a `blast` radius (m), one ball
/// of it instead, half sunk just beyond the view centre: the frames until
/// the world follows it and their costs.
fn chunk_dig(flight: &Flight, view: &str, eye: DVec3, forward: Vec3, count: usize, blast: Option<f64>) {
    use helio_pass_voxel_planet::chunks::ChunkWorld;
    let [w, h] = flight.size;
    let mut planet = (*flight.planet).clone();
    let mut world = ChunkWorld::new(&flight.device, &flight.queue, &flight.planet);
    let mut frame = 0u64;
    chunk_settle(flight, &mut world, eye, &mut frame);
    let up = up_for(eye);
    let forward = forward.normalize();
    let near = (flight.planet.air_clearance(eye) * 0.25).clamp(0.05, 50_000.0) as f32;
    let camera = Camera::perspective_look_at(Vec3::ZERO, forward, up, std::f32::consts::FRAC_PI_4, w as f32 / h as f32, near, 40_000_000.0);
    let config = chunk_config(h, 512);
    let out = std::path::PathBuf::from(std::env::args().nth(1).unwrap());
    // One frame: follow the planet's edits, update, draw; GPU times and the
    // tiles generated again.
    let step = |world: &mut ChunkWorld, planet: &Planet, frame: &mut u64| {
        assert!(world.sync_edits(planet), "edits follow");
        let mut encoder = flight.device.create_command_encoder(&Default::default());
        let stats = world.update(&mut helio_core::CommandRecorder::from_encoder(&mut encoder), eye, DVec3::ZERO, None, *frame, &config);
        world.draw(&mut helio_core::CommandRecorder::from_encoder(&mut encoder), *frame, eye, camera.proj * camera.view, camera.proj * camera.view, near, flight.sun_at(eye), [w, h]);
        flight.queue.submit([encoder.finish()]);
        world.submitted();
        let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
        world.submitted();
        *frame += 1;
        stats
    };
    let depth_at = |world: &ChunkWorld, x: u32, y: u32| -> f32 {
        let buffer = flight.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = flight.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &world.targets().unwrap().distance, mip_level: 0, origin: wgpu::Origin3d { x, y, z: 0 }, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) } },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        flight.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        flight.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        f32::from_le_bytes(data[..4].try_into().unwrap())
    };
    let centre_depth = |world: &ChunkWorld| depth_at(world, w / 2, h / 2);
    // Down the centre column, the chunk hit against the exact CPU raycast
    // of `planet`: the largest difference (m).
    let worst = |world: &ChunkWorld, planet: &Planet, rows: std::ops::Range<u32>| -> (f64, u32) {
        let inverse = (camera.proj * camera.view).inverse();
        let mut worst = (0.0f64, 0u32);
        for y in rows.step_by(12) {
            let ndc = Vec3::new(0.5 / w as f32, 1.0 - (y as f32 + 0.5) / h as f32 * 2.0, 0.5);
            let far = inverse * ndc.extend(1.0);
            let dir = (far.truncate() / far.w).normalize().as_dvec3();
            let gpu = f64::from(depth_at(world, w / 2, y));
            let cpu = planet.raycast(eye, dir, blast.map_or(1000.0, |r| (r * 4.0).max(1000.0))).map_or(0.0, |h| h.distance);
            if (gpu - cpu).abs() > worst.0 {
                worst = ((gpu - cpu).abs(), y);
            }
        }
        worst
    };
    // Drawn frames find cave mouths: settle what they ask for.
    for _ in 0..4 {
        step(&mut world, &planet, &mut frame);
    }
    while !world.settled() {
        step(&mut world, &planet, &mut frame);
    }
    step(&mut world, &planet, &mut frame);
    eprintln!("CHUNKDIG {view}: rays reached under {} tiles' surface bands", world.deep_tiles());
    // `HELIO_VOXEL_CHUNKS_PROBE=x,y;x,y`: those pixels against the CPU raycast.
    if let Ok(probes) = std::env::var("HELIO_VOXEL_CHUNKS_PROBE") {
        let inverse = (camera.proj * camera.view).inverse();
        for probe in probes.split(';') {
            let xy: Vec<u32> = probe.split(',').filter_map(|v| v.trim().parse().ok()).collect();
            let ndc = Vec3::new((xy[0] as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (xy[1] as f32 + 0.5) / h as f32 * 2.0, 0.5);
            let far = inverse * ndc.extend(1.0);
            let dir = (far.truncate() / far.w).normalize().as_dvec3();
            eprintln!("CHUNKDIG probe {},{}: chunks {:.2} m, cpu {:?}", xy[0], xy[1], depth_at(&world, xy[0], xy[1]), planet.raycast(eye, dir, 1000.0).map(|h| (h.distance, h.cell)));
        }
    }
    let before_path = out.join("dig_before.png");
    save_chunk_colour(flight, &world, before_path.to_str().unwrap());
    let before_hits = world.read_hits();
    let before = centre_depth(&world);
    let hit = planet.raycast(eye, forward.as_dvec3(), 1000.0).expect("the view centre hits ground");
    let at = eye + forward.as_dvec3() * hit.distance;
    let side = forward.as_dvec3().cross(up.as_dvec3()).normalize();
    eprintln!("CHUNKDIG {view}: centre at {:.2} m, {count} balls across it", hit.distance);
    if let Some(radius) = blast {
        let up = up.as_dvec3();
        let ahead = (forward.as_dvec3() - up * forward.as_dvec3().dot(up)).normalize();
        planet.apply(Brush { center: (at + ahead * radius - up * (radius * 0.5)).to_array(), radius, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
        // Frames until the world follows it: tiles, GPU generation, CPU.
        let follow = |world: &mut ChunkWorld, planet: &Planet, frame: &mut u64, what: &str| {
            let (mut frames, mut tiles, mut most, mut gen_max, mut cpu_max, mut gen_sum) = (0, 0, 0, 0.0f64, 0.0f64, 0.0);
            let started = Instant::now();
            loop {
                let stats = step(world, planet, frame);
                let timings = world.take_timings();
                let gen: f64 = timings.iter().map(|t| t.generate).sum();
                let stages = timings.iter().fold([0.0; 3], |s, t| [s[0] + t.stages[0], s[1] + t.stages[1], s[2] + t.stages[2]]);
                frames += 1;
                tiles += stats.generated_tiles;
                most = most.max(stats.generated_tiles);
                gen_max = gen_max.max(gen);
                gen_sum += gen;
                cpu_max = cpu_max.max(stats.cpu_ms.iter().sum::<f64>());
                if std::env::var_os("HELIO_VOXEL_CHUNKS_TRACE").is_some() {
                    eprintln!("CHUNKBLAST frame {frames}: tiles {} chunks {} regen {} requests {} gen {gen:.2} ms {stages:.2?} cpu {:?} drawn {} resident {} slots {}", stats.generated_tiles, stats.generated_chunks, stats.regenerated_tiles, stats.requests, stats.cpu_ms.map(|v| (v * 100.0).round() / 100.0), stats.drawn_chunks, stats.resident_chunks, stats.out_of_slots);
                }
                if world.settled() || frames >= 600 {
                    break;
                }
            }
            eprintln!(
                "CHUNKBLAST {view} {what}: r {radius:.0} m, {frames} frames ({:.2} s) to settle, tiles {tiles} (most a frame {most}), generation {gen_sum:.1} ms (most a frame {gen_max:.2}), update cpu most a frame {cpu_max:.2} ms",
                started.elapsed().as_secs_f64()
            );
        };
        follow(&mut world, &planet, &mut frame, "blasted");
        step(&mut world, &planet, &mut frame);
        let (error, row) = worst(&world, &planet, 0..h);
        eprintln!("CHUNKBLAST {view}: centre column against the CPU raycast: worst {error:.2} m (row {row})");
        save_chunk_colour(flight, &world, out.join("blast.png").to_str().unwrap());
        planet.undo().unwrap();
        follow(&mut world, &planet, &mut frame, "undone");
        step(&mut world, &planet, &mut frame);
        let differ = before_hits.iter().zip(world.read_hits()).filter(|(a, b)| a.map(|h| (h.cell, h.level)) != b.map(|h| (h.cell, h.level))).count();
        eprintln!("CHUNKBLAST {view}: undone frame's hits differ from the first's in {differ} pixels");
        return;
    }
    let (mut tiles, mut gen_ms, mut shown) = (Vec::new(), Vec::new(), None);
    for n in 0..count {
        // From the centre outwards: the first are sealed into the baked
        // store when there are more than the recent log keeps.
        let offset = (n as f64 / 2.0).ceil() * if n % 2 == 0 { 0.6 } else { -0.6 };
        planet.apply(Brush { center: (at + side * offset).to_array(), radius: 1.5, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
        let stats = step(&mut world, &planet, &mut frame);
        tiles.push(stats.generated_tiles as f64);
        gen_ms.extend(world.take_timings().iter().map(|t| t.generate));
        if shown.is_none() && (centre_depth(&world) - before).abs() > 0.05 {
            shown = Some(n);
        }
    }
    let (error, row) = worst(&world, &planet, h / 2..h);
    eprintln!("CHUNKDIG {view}: dug, centre column against the CPU raycast: worst {error:.2} m (row {row})");
    let dug_path = out.join("dig_dug.png");
    save_chunk_colour(flight, &world, dug_path.to_str().unwrap());
    eprintln!(
        "CHUNKDIG {view}: tiles generated again an edit p50 {:.0} max {:.0}, generation p50 {:.2} max {:.2} ms; centre deeper from edit {shown:?} (of {count})",
        percentile(&tiles, 0.5),
        tiles.iter().copied().fold(0.0, f64::max),
        percentile(&gen_ms, 0.5),
        gen_ms.iter().copied().fold(0.0, f64::max)
    );
    for _ in 0..count {
        planet.undo().unwrap();
        step(&mut world, &planet, &mut frame);
    }
    while !world.settled() {
        step(&mut world, &planet, &mut frame);
    }
    step(&mut world, &planet, &mut frame);
    let undone_path = out.join("dig_undone.png");
    save_chunk_colour(flight, &world, undone_path.to_str().unwrap());
    // The hits, not the colours (sun samples turn each frame).
    let differ = before_hits.iter().zip(world.read_hits()).filter(|(a, b)| a.map(|h| (h.cell, h.level)) != b.map(|h| (h.cell, h.level))).count();
    eprintln!("CHUNKDIG {view}: undone frame's hits differ from the first's in {differ} pixels; centre {:.2} m (was {before:.2})", centre_depth(&world));
}

/// A view drawn by the renderer with the chunk pass: settle, hold
/// (`HELIO_VOXEL_FLIGHT_HOLD_SECONDS`), report the chunk draw's GPU time and
/// save the final frame (`<out>/chunks_<view>.png`).
fn chunk_engine_hold(flight: &mut Flight, view: &str, eye: DVec3, forward: Vec3) {
    let started = Instant::now();
    let mut frames = 0;
    loop {
        flight.draw(eye, forward);
        frames += 1;
        let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
        std::thread::sleep(std::time::Duration::from_millis(6));
        if frames > 10 && flight.renderer.find_pass::<ChunkPass>().is_some_and(ChunkPass::settled) {
            break;
        }
        if frames > 6000 {
            eprintln!("CHUNKENGINE {view}: did not settle");
            break;
        }
    }
    let pass = flight.renderer.find_pass_mut::<ChunkPass>().unwrap();
    eprintln!("CHUNKENGINE {view} settled in {:.2} s ({frames} frames): {}", started.elapsed().as_secs_f64(), pass.diagnostics());
    pass.take_timings();
    let seconds: f64 = std::env::var("HELIO_VOXEL_FLIGHT_HOLD_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0);
    let held = Instant::now();
    // The graph's GPU time by pass, over the hold.
    let mut passes: std::collections::BTreeMap<&'static str, (f64, u32)> = Default::default();
    let mut graph = Vec::new();
    let mut seen = 0;
    while held.elapsed().as_secs_f64() < seconds {
        flight.draw(eye, forward);
        let snapshot = flight.renderer.timing_snapshot();
        if snapshot.generation != seen {
            seen = snapshot.generation;
            graph.extend(snapshot.total_gpu_ms.map(f64::from));
            for p in &snapshot.passes {
                if let Some(ms) = p.gpu_ms {
                    let entry = passes.entry(p.name).or_default();
                    entry.0 += f64::from(ms);
                    entry.1 += 1;
                }
            }
        }
    }
    let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
    std::thread::sleep(std::time::Duration::from_millis(50));
    let timings = flight.renderer.find_pass_mut::<ChunkPass>().unwrap().take_timings();
    let draw: Vec<f64> = timings.iter().map(|t| t.draw).collect();
    eprintln!("CHUNKENGINE {view} n={} chunk draw p50 {:.3} p95 {:.3}", draw.len(), percentile(&draw, 0.5), percentile(&draw, 0.95));
    let mut by_pass: Vec<(&str, f64)> = passes.iter().map(|(name, (sum, n))| (*name, sum / f64::from((*n).max(1)))).collect();
    by_pass.sort_by(|a, b| b.1.total_cmp(&a.1));
    eprintln!("CHUNKENGINE {view} graph gpu p50 {:.2} ms; by pass: {}", percentile(&graph, 0.5), by_pass.iter().filter(|p| p.1 >= 0.05).map(|(n, ms)| format!("{n} {ms:.2}")).collect::<Vec<_>>().join(", "));
    flight.capture(&format!("chunks_{view}"));
    pick_probe(flight, view, eye, forward);
}

/// `HELIO_VOXEL_FLIGHT_STRESS=ascent|descent|cruise|orbit|jump|dig|yoyo|dive|glide`: a flight
/// through the renderer in real time (the path follows the wall clock, as a
/// game's camera does: a slow frame moves it further), from 2 m to 400 km
/// or back in `HELIO_VOXEL_FLIGHT_HOLD_SECONDS` (20), at 300 m/s 40 m over
/// the ground, or in orbit (`HELIO_VOXEL_STRESS_ALTITUDE` m, 400 km) at
/// 7.7 km/s looking down towards the horizon. Every frame goes to `<out>/stress_<mode>.csv` (time, altitude,
/// frame interval, the chunk pass's coarse share, worst cell and level
/// switch pixels, waiting tiles, generation, update CPU by phase). Every
/// `HELIO_VOXEL_STRESS_SAMPLE` seconds (1) the frame is saved; after the
/// flight each sample's view is settled and held until TSR converges and
/// saved twice, 30 frames apart (`stress_<mode>_<n>.png`, `_ref.png`,
/// `_ref2.png`: the second pair is the noise floor of a comparison).
fn chunk_stress(flight: &mut Flight, mode: &str, ground: DVec3, heading: f64) {
    let seconds: f64 = std::env::var("HELIO_VOXEL_FLIGHT_HOLD_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let sample: f64 = std::env::var("HELIO_VOXEL_STRESS_SAMPLE").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
    let up = ground.normalize();
    let ahead = tangent(ground, heading).as_dvec3();
    let planet = flight.planet.clone();
    let radius = planet.grid().radius();
    let path = |t: f64| -> (DVec3, Vec3) {
        let u = (t / seconds).clamp(0.0, 1.0);
        if mode == "cruise" {
            let eye = planet.surface_point(ground + ahead * (300.0 * t), 40.0);
            return (eye, (ahead - up * 0.15).normalize().as_vec3());
        }
        if mode == "dig" {
            // 6 m over the ground looking ahead and down; a ball is dug
            // where the view's centre meets it a second in.
            return (planet.surface_point(ground, 6.0), (ahead - up * 0.5).normalize().as_vec3());
        }
        if mode == "glide" {
            // Down from 300 km to 30 km over the flight, looking toward the
            // horizon and drifting ahead (an editor camera eased down over
            // the planet's curve).
            let altitude = 300_000.0 * 0.1f64.powf(u);
            let eye = planet.surface_point(ground + ahead * (altitude * 0.05 * t), altitude);
            return (eye, (ahead - up * 0.35).normalize().as_vec3());
        }
        if mode == "dive" {
            // Straight down from `HELIO_VOXEL_STRESS_ALTITUDE` m (15 000 km,
            // the whole planet in view) to `HELIO_VOXEL_STRESS_LOW` m (20 km)
            // in half a second, a second in, then hovering (an editor camera
            // dropped from orbit).
            let top: f64 = std::env::var("HELIO_VOXEL_STRESS_ALTITUDE").ok().and_then(|v| v.parse().ok()).unwrap_or(15_000_000.0);
            let u = ((t - 1.0) / 0.5).clamp(0.0, 1.0);
            let low: f64 = std::env::var("HELIO_VOXEL_STRESS_LOW").ok().and_then(|v| v.parse().ok()).unwrap_or(20_000.0);
            let altitude = top * (low / top).powf(u);
            let down = (ahead * 0.02 - up).normalize();
            return (planet.surface_point(ground, altitude), down.as_vec3());
        }
        if mode == "yoyo" {
            // Up and down between 2 and 50 km every `HELIO_VOXEL_STRESS_PERIOD`
            // seconds (3) looking steeply down (an editor camera flown up
            // and down: switches across many levels while others fade).
            let period: f64 = std::env::var("HELIO_VOXEL_STRESS_PERIOD").ok().and_then(|v| v.parse().ok()).unwrap_or(3.0);
            let altitude = 2_000.0 * 25f64.powf(0.5 - 0.5 * (std::f64::consts::TAU * t / period).cos());
            return (planet.surface_point(ground + ahead * (300.0 * t), altitude), (ahead - up * 2.0).normalize().as_vec3());
        }
        if mode == "jump" {
            // 40 m over the ground, then at once `HELIO_VOXEL_STRESS_ALTITUDE`
            // (30 km) over it looking down (an editor camera's jump: one
            // switch across many levels).
            if t < 1.0 {
                return (planet.surface_point(ground, 40.0), (ahead - up * 0.15).normalize().as_vec3());
            }
            let altitude: f64 = std::env::var("HELIO_VOXEL_STRESS_ALTITUDE").ok().and_then(|v| v.parse().ok()).unwrap_or(30_000.0);
            return (planet.surface_point(ground, altitude), (ahead - up * 3.0).normalize().as_vec3());
        }
        if mode == "orbit" {
            let altitude: f64 = std::env::var("HELIO_VOXEL_STRESS_ALTITUDE").ok().and_then(|v| v.parse().ok()).unwrap_or(400_000.0);
            // Around the planet along a great circle through the spawn, at
            // `HELIO_VOXEL_STRESS_SPEED` m/s (7 700).
            let speed: f64 = std::env::var("HELIO_VOXEL_STRESS_SPEED").ok().and_then(|v| v.parse().ok()).unwrap_or(7_700.0);
            let angle = speed * t / (radius + altitude);
            let position = (up * angle.cos() + ahead * angle.sin()) * (radius + altitude);
            let (local_up, local_ahead) = (position.normalize(), ahead * angle.cos() - up * angle.sin());
            return (position, (local_ahead - local_up * 0.8).normalize().as_vec3());
        }
        // Exponentially between 2 m and 400 km.
        let altitude = 2.0 * 200_000.0f64.powf(if mode == "ascent" { u } else { 1.0 - u });
        let eye = planet.surface_point(ground + ahead * (altitude * 0.3), altitude);
        (eye, (ahead - up * 0.6).normalize().as_vec3())
    };
    // Settle a view and hold it until TSR converges.
    let settle = |flight: &mut Flight, eye: DVec3, forward: Vec3| {
        let started = Instant::now();
        let mut frames = 0;
        while frames < 10 || !flight.renderer.find_pass::<ChunkPass>().is_some_and(ChunkPass::settled) {
            flight.draw(eye, forward);
            frames += 1;
            if frames > 6000 {
                eprintln!("CHUNKSTRESS {mode}: a view did not settle");
                break;
            }
        }
        for _ in 0..90 {
            flight.draw(eye, forward);
        }
        started.elapsed().as_secs_f64()
    };
    let (eye, forward) = path(0.0);
    eprintln!("CHUNKSTRESS {mode}: loaded in {:.2} s", settle(flight, eye, forward));
    flight.renderer.find_pass_mut::<ChunkPass>().unwrap().take_timings();
    let mut csv = String::from("t,altitude_m,interval_ms,captured,coarse,worst_px,switches,switch_px,waiting,generated,fading,update_cpu_ms,results_ms,plan_ms,evict_ms,generate_ms,drawn_ms,table_ms,record_ms,wait_ms,pass_prepare_ms,pass_execute_ms
");
    let (mut intervals, mut coarse, mut worst, mut switch_px) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut draw_ms, mut generate_ms) = (Vec::new(), Vec::new());
    let mut samples = Vec::new();
    let mut slow = Vec::new();
    let started = Instant::now();
    let mut last = started;
    let mut captured = false;
    let mut traced = false;
    let mut dug = false;
    let mut next_dig = 1.0;
    let mut generated_tiles = 0usize;
    loop {
        let t = started.elapsed().as_secs_f64();
        if t > seconds {
            break;
        }
        let (eye, forward) = path(t);
        // `dig`: one ball, `HELIO_VOXEL_STRESS_DIG` m across (12), close to
        // the camera. `HELIO_VOXEL_STRESS_DIG_EVERY=s` (any mode): a ball at
        // the view's centre every `s` seconds (an editor's brush clicked
        // from high up).
        let every: Option<f64> = std::env::var("HELIO_VOXEL_STRESS_DIG_EVERY").ok().and_then(|v| v.parse().ok());
        let due = match every {
            Some(_) => t >= next_dig,
            None => mode == "dig" && t >= 1.0 && !dug,
        };
        if due {
            dug = true;
            next_dig = t + every.unwrap_or(0.0);
            let radius: f64 = std::env::var("HELIO_VOXEL_STRESS_DIG").ok().and_then(|v| v.parse().ok()).unwrap_or(12.0);
            // Close up, the ground the view's centre meets; from high up
            // (a CPU raycast there takes seconds), the sphere's, snapped
            // to the ground.
            let dir = forward.as_dvec3();
            let at = if every.is_none() {
                eye + dir * flight.planet.raycast(eye, dir, 1000.0).expect("the view centre hits ground").distance
            } else {
                let b = eye.dot(dir);
                let r = flight.planet.grid().radius();
                let disc = b * b - (eye.length_squared() - r * r);
                if disc < 0.0 {
                    eprintln!("CHUNKSTRESS dig: the view centre misses the planet at t {t:.3} s");
                    continue;
                }
                flight.planet.surface_point(eye + dir * (-b - disc.sqrt()), 0.0)
            };
            let hit_distance = (at - eye).length();
            let mut planet = (*flight.planet).clone();
            planet.apply(Brush { center: at.to_array(), radius, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
            flight.planet = Arc::new(planet);
            eprintln!("CHUNKSTRESS dig: ball {radius} m at {hit_distance:.1} m from the eye, t {t:.3} s");
        }
        let (record_ms, wait_ms) = flight.draw(eye, forward);
        let now = Instant::now();
        let interval = (now - last).as_secs_f64() * 1e3;
        last = now;
        let altitude = eye.length() - radius;
        // A slow frame: the renderer's CPU by pass.
        if record_ms > 40.0 {
            let snapshot = flight.renderer.timing_snapshot();
            let mut passes: Vec<(&str, f32)> = snapshot.passes.iter().filter_map(|p| Some((p.name, p.cpu_ms?))).collect();
            passes.sort_by(|a, b| b.1.total_cmp(&a.1));
            passes.truncate(4);
            eprintln!("CHUNKSTRESS {mode}: recording {record_ms:.1} ms at t {t:.2} s: graph cpu {:?} ms, passes {passes:?}", snapshot.total_cpu_ms);
        }
        let pass = flight.renderer.find_pass_mut::<ChunkPass>().unwrap();
        let s = *pass.stats();
        let pass_cpu = pass.cpu_ms();
        for timing in pass.take_timings() {
            draw_ms.push(timing.draw);
            generate_ms.push(timing.generate);
        }
        generated_tiles += s.generated_tiles;
        csv += &format!("{t:.4},{altitude:.2},{interval:.2},{},{:.4},{:.3},{},{:.3},{},{},{},{:.2},{},{record_ms:.2},{wait_ms:.2},{:.2},{:.2}
", u8::from(captured), s.coarse_share, s.worst_px, s.switches, s.switch_px, s.requests, s.generated_tiles, s.fading_tiles, s.cpu_ms.iter().sum::<f64>(), s.cpu_ms.map(|v| format!("{v:.2}")).join(","), pass_cpu[0], pass_cpu[1]);
        // A frame after a capture waited for its readback.
        if !captured {
            intervals.push(interval);
            slow.push((interval, t, altitude));
        }
        // Counting builds: each frame's rays that left the cut or the box
        // budget (missed) and placements that found nothing (lost).
        if std::env::var_os("HELIO_VOXEL_CHUNKS_COUNT").is_some() {
            if let Some(world) = pass.world() {
                // `HELIO_VOXEL_CHUNKS_DEBUG_PIXEL`: that pixel's boxes (read
                // before the counts, which clear them).
                // `_DEBUG_PIXEL_AT=seconds` (1.0): when the pixel is traced.
                let at: f64 = std::env::var("HELIO_VOXEL_CHUNKS_DEBUG_PIXEL_AT").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0);
                let trace = if t > at && !traced && std::env::var_os("HELIO_VOXEL_CHUNKS_DEBUG_PIXEL").is_some() { world.pixel_trace() } else { Vec::new() };
                let c = world.take_counts();
                if c.missed + c.lost > 0 {
                    eprintln!("CHUNKSTRESS {mode}: t {t:.3} s missed {} ({} boxes each) lost {} (last at {:?})", c.missed, c.missed_boxes / c.missed.max(1), c.lost, c.lost_pixel);
                }
                if !trace.is_empty() {
                    traced = true;
                    eprintln!("PIXELTRACE at t {t:.3} s");
                    for (n, step) in trace.iter().enumerate() {
                        eprintln!("PIXELTRACE {n}: {step:?}");
                    }
                }
            }
        }
        coarse.push(s.coarse_share);
        worst.push(s.worst_px);
        if s.switches > 0 {
            switch_px.push(s.switch_px);
        }
        captured = t >= samples.len() as f64 * sample;
        if captured {
            flight.capture(&format!("stress_{mode}_{:02}", samples.len()));
            samples.push((eye, forward));
        }
    }
    std::fs::write(flight.output.join(format!("stress_{mode}.csv")), csv).unwrap();
    slow.sort_by(|a, b| b.0.total_cmp(&a.0));
    let max = |v: &[f64]| v.iter().copied().fold(0.0, f64::max);
    let over = |v: &[f64], x: f64| v.iter().filter(|&&p| p > x).count();
    eprintln!(
        "CHUNKSTRESS {mode}: frames {} | interval p50 {:.1} p95 {:.1} p99 {:.1} max {:.1} ms, over 33 ms {} | gpu draw p50 {:.2} max {:.2}, generate p50 {:.2} max {:.2} ms | coarse mean {:.4} max {:.3} | worst px p50 {:.2} p95 {:.2} max {:.2} | switches in {} frames, px p50 {:.2} p95 {:.2} max {:.2}, over 1.5 px {}, over 3 px {}",
        intervals.len(),
        percentile(&intervals, 0.5),
        percentile(&intervals, 0.95),
        percentile(&intervals, 0.99),
        max(&intervals),
        over(&intervals, 33.3),
        percentile(&draw_ms, 0.5),
        max(&draw_ms),
        percentile(&generate_ms, 0.5),
        max(&generate_ms),
        coarse.iter().sum::<f64>() / coarse.len().max(1) as f64,
        max(&coarse),
        percentile(&worst, 0.5),
        percentile(&worst, 0.95),
        max(&worst),
        switch_px.len(),
        percentile(&switch_px, 0.5),
        percentile(&switch_px, 0.95),
        max(&switch_px),
        over(&switch_px, 1.5),
        over(&switch_px, 3.0),
    );
    // GPU generation per tile over the flight (its timings lag a few frames).
    let generation: f64 = generate_ms.iter().sum();
    eprintln!("CHUNKSTRESS {mode}: generation {generation:.1} ms GPU for {generated_tiles} tiles, {:.1} us a tile", generation * 1e3 / generated_tiles.max(1) as f64);
    for (interval, t, altitude) in slow.iter().take(5) {
        eprintln!("CHUNKSTRESS {mode}: slow frame {interval:.1} ms at t {t:.2} s, altitude {altitude:.0} m");
    }
    for (n, (eye, forward)) in samples.iter().enumerate() {
        settle(flight, *eye, *forward);
        flight.capture(&format!("stress_{mode}_{n:02}_ref"));
        for _ in 0..30 {
            flight.draw(*eye, *forward);
        }
        flight.capture(&format!("stress_{mode}_{n:02}_ref2"));
    }
    // `HELIO_VOXEL_PICKS`: the last sample's view, settled, against the CPU.
    if let Some(&(eye, forward)) = samples.last() {
        pick_probe(flight, mode, eye, forward);
    }
}

/// `HELIO_VOXEL_PICKS=x,y;x,y` (output pixels): the renderer's picks there
/// against the CPU raycast through the same pixel.
fn pick_probe(flight: &mut Flight, view: &str, eye: DVec3, forward: Vec3) {
    use helio_pass_voxel_planet::engine::{PickRequest, Picks};
    let Ok(points) = std::env::var("HELIO_VOXEL_PICKS") else { return };
    let [w, h] = flight.size;
    let picks = std::sync::Arc::new(std::sync::Mutex::new(Picks::default()));
    let pixels: Vec<[u32; 2]> = points.split(';').map(|p| p.split(',').map(|v| v.trim().parse().unwrap()).collect::<Vec<u32>>()).map(|v| [v[0], v[1]]).collect();
    picks.lock().unwrap().requests = pixels.iter().enumerate().map(|(n, p)| PickRequest { id: n as u64, uv: [(p[0] as f32 + 0.5) / w as f32, (p[1] as f32 + 0.5) / h as f32] }).collect();
    flight.picks = Some(picks.clone());
    for _ in 0..12 {
        flight.draw(eye, forward);
        let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
    }
    flight.picks = None;
    let up = up_for(eye);
    let forward = forward.normalize();
    let up = if forward.dot(up).abs() > 0.999 { up.any_orthonormal_vector() } else { up };
    let right = forward.cross(up).normalize();
    let cam_up = right.cross(forward);
    let tan = (std::f32::consts::FRAC_PI_4 * 0.5).tan();
    let results = std::mem::take(&mut picks.lock().unwrap().results);
    for (n, p) in pixels.iter().enumerate() {
        let ndc = [(p[0] as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (p[1] as f32 + 0.5) / h as f32 * 2.0];
        let dir = (forward + right * ndc[0] * tan * (w as f32 / h as f32) + cam_up * ndc[1] * tan).normalize().as_dvec3();
        let cpu = flight.planet.raycast(eye, dir, 100_000.0);
        let pick = results.iter().find(|r| r.id == n as u64).and_then(|r| r.hit);
        eprintln!("PICK {view} {},{}: drawn {:?} | cpu {:?}", p[0], p[1], pick.map(|h| (h.distance, h.level, h.cell, h.entered)), cpu.map(|c| (c.distance, c.cell)));
        if let (Some(c), Some(world)) = (cpu, flight.renderer.find_pass::<ChunkPass>().and_then(ChunkPass::world)) {
            eprintln!("PICK {view} {},{}: chunks over the cpu cell {:?}", p[0], p[1], world.chunks_over(c.cell.face, c.cell.i, c.cell.j));
        }
    }
}

/// `HELIO_VOXEL_CHUNKS_FLY=descent|cruise`: chunk trees streamed along a
/// path at 60 simulated frames a second (`HELIO_VOXEL_CHUNKS_BUDGET` tiles
/// generated a frame): how far the drawn detail lags what is wanted
/// (`coarse`, the share of drawn tiles coarser than 1.5x the target), the
/// primary time and the CPU update time. `HELIO_VOXEL_CHUNKS_PNG=<prefix>`
/// saves a frame each second.
fn chunk_fly(flight: &Flight, mode: &str, ground: DVec3, heading: f64) {
    use helio_pass_voxel_planet::chunks::ChunkWorld;
    let [w, h] = flight.size;
    let config = chunk_config(h, 64);
    let mut world = ChunkWorld::new(&flight.device, &flight.queue, &flight.planet);
    let up = ground.normalize();
    let mut budgets = Vec::new();
    let mut phases = [0.0f64; 6];
    let ahead = tangent(ground, heading).as_dvec3();
    let seconds: f64 = std::env::var("HELIO_VOXEL_FLIGHT_HOLD_SECONDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0);
    // Eye and view direction at time t.
    let path = |t: f64| -> (DVec3, Vec3) {
        match mode {
            "cruise" => {
                // 300 m/s, 40 m over the ground.
                let along = ground + ahead * (300.0 * t);
                let eye = flight.planet.surface_point(along, 40.0);
                (eye, (ahead - up * 0.15).normalize().as_vec3())
            }
            _ => {
                // From 400 km down to 2 m, exponentially.
                let altitude = 400_000.0 * (2.0f64 / 400_000.0).powf((t / seconds).min(1.0));
                let point = ground + ahead * (altitude * 0.3);
                let eye = flight.planet.surface_point(point, altitude.max(2.0));
                (eye, (ahead - up * 0.6).normalize().as_vec3())
            }
        }
    };
    let dt = 1.0 / 60.0;
    let frames = (seconds / dt) as u64;
    // Loaded before the flight starts (a game loads its spawn first).
    let mut start = 0u64;
    let loaded = chunk_settle(flight, &mut world, path(0.0).0, &mut start);
    eprintln!("CHUNKFLY {mode} loaded in {loaded:.2} s ({start} frames)");
    let png = std::env::var("HELIO_VOXEL_CHUNKS_PNG").ok();
    let mut pending = std::collections::VecDeque::new();
    let (mut times, mut update_ms, mut coarse, mut requests, mut resident) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), 0);
    let mut fading = Vec::new();
    // Generation by kernel (us), and the tiles it made.
    let (mut stage_us, mut stage_tiles) = ([0.0f64; 3], 0usize);
    let counting = std::env::var_os("HELIO_VOXEL_CHUNKS_COUNT").is_some();
    let mut counts = [0u64; 7];
    let (mut generate_ms, mut per_tile) = (Vec::new(), Vec::new());
    for n in 0..frames {
        let frame = start + n;
        let t = n as f64 * dt;
        let (eye, forward) = path(t);
        let velocity = (path(t + dt).0 - eye) / dt;
        let up_dir = up_for(eye);
        let view_up = if forward.dot(up_dir).abs() > 0.999 { up_dir.any_orthonormal_vector() } else { up_dir };
        let near = (flight.planet.air_clearance(eye) * 0.25).clamp(0.05, 50_000.0) as f32;
        let camera = Camera::perspective_look_at(Vec3::ZERO, forward, view_up, std::f32::consts::FRAC_PI_4, w as f32 / h as f32, near, 40_000_000.0);
        let mut encoder = flight.device.create_command_encoder(&Default::default());
        let cpu = Instant::now();
        let view = helio_pass_voxel_planet::chunks::residency::View { forward: forward.as_dvec3(), cos_half: (std::f64::consts::FRAC_PI_8 * f64::from(w) / f64::from(h) * 1.1).cos() };
        let stats = world.update(&mut helio_core::CommandRecorder::from_encoder(&mut encoder), eye, velocity, Some(view), frame, &config);
        update_ms.push(cpu.elapsed().as_secs_f64() * 1e3);
        world.draw(&mut helio_core::CommandRecorder::from_encoder(&mut encoder), frame, eye, camera.proj * camera.view, camera.proj * camera.view, near, flight.sun_at(eye), [w, h]);
        pending.push_back(flight.queue.submit([encoder.finish()]));
        world.submitted();
        if pending.len() > 2 {
            let _ = flight.device.poll(wgpu::PollType::Wait { submission_index: pending.pop_front(), timeout: None });
        }
        for t in world.take_timings() {
            times.push(t.draw);
            generate_ms.push(t.generate);
            if t.tiles > 0 {
                per_tile.push(t.generate / t.tiles as f64);
                for (sum, ms) in stage_us.iter_mut().zip(t.stages) {
                    *sum += ms * 1e3;
                }
                stage_tiles += t.tiles;
            }
        }
        coarse.push(stats.coarse_share);
        fading.push(stats.fading_tiles as f64);
        // `HELIO_VOXEL_CHUNKS_COUNT`: the traces' counts over the flight.
        if counting && n % 30 == 29 {
            let c = world.take_counts();
            for (sum, v) in counts.iter_mut().zip([c.seeded, c.hits, c.steps, c.boxes, c.missed, c.missed_boxes, c.lost]) {
                *sum += u64::from(v);
            }
        }
        budgets.push(stats.budget as f64);
        for (p, v) in phases.iter_mut().zip(stats.cpu_ms) {
            *p += v / frames as f64;
        }
        requests.push(stats.requests as f64);
        resident = stats.resident_chunks;
        if n % 60 == 0 {
            eprintln!(
                "CHUNKFLY {mode} t {t:5.1} alt {:9.1} coarse {:.3} requests {:5} generated {:3} drawn {:6} resident {:6} evicted {:4} regen {} slots {} leaves {}",
                flight.planet.air_clearance(eye),
                stats.coarse_share,
                stats.requests,
                stats.generated_tiles,
                stats.drawn_chunks,
                stats.resident_chunks,
                stats.evicted_tiles,
                stats.regenerated_tiles,
                stats.out_of_slots,
                stats.out_of_leaves
            );
            if let Some(prefix) = &png {
                let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
                save_chunk_colour(flight, &world, &format!("{prefix}_{mode}_{:03}.png", n / 60));
            }
        }
    }
    let lagging = coarse.iter().filter(|&&c| c > 0.02).count();
    eprintln!("CHUNKFLY {mode} generation us a tile by kernel (tile columns, chunk columns, chunk build): {:.1?}", stage_us.map(|us| us / stage_tiles.max(1) as f64));
    eprintln!("CHUNKFLY {mode} fading tiles mean {:.0} max {:.0}", fading.iter().sum::<f64>() / fading.len().max(1) as f64, fading.iter().copied().fold(0.0, f64::max));
    if counting {
        let [seeded, hits, steps, boxes, missed, missed_boxes, lost] = counts.map(|v| v as f64);
        eprintln!(
            "CHUNKFLY {mode} counts: hits/seeded {:.3} steps/seeded {:.1} boxes/seeded {:.2}; missed/seeded {:.4}, {:.1} boxes each, lost {lost}",
            hits / seeded.max(1.0),
            steps / seeded.max(1.0),
            boxes / seeded.max(1.0),
            missed / seeded.max(1.0),
            missed_boxes / missed.max(1.0)
        );
    }
    eprintln!("CHUNKFLY {mode} update cpu mean by phase (results, plan, evict, generate, drawn, table): {phases:.2?}");
    eprintln!(
        "CHUNKFLY {mode} summary: frames {frames} coarse mean {:.4} max {:.3} frames over 2% {lagging} | requests p50 {:.0} max {:.0} | primary p50 {:.3} p95 {:.3} | generate p50 {:.3} p95 {:.3} ms, {:.1} us a tile, budget p50 {:.0} | update cpu p50 {:.2} p95 {:.2} ms | resident {resident} chunks {:.1} MB",
        coarse.iter().sum::<f64>() / coarse.len().max(1) as f64,
        coarse.iter().copied().fold(0.0, f64::max),
        percentile(&requests, 0.5),
        requests.iter().copied().fold(0.0, f64::max),
        percentile(&times, 0.5),
        percentile(&times, 0.95),
        percentile(&generate_ms, 0.5),
        percentile(&generate_ms, 0.95),
        percentile(&per_tile, 0.5) * 1e3,
        percentile(&budgets, 0.5),
        percentile(&update_ms, 0.5),
        percentile(&update_ms, 0.95),
        world.resident_bytes() as f64 / 1048576.0
    );
}

/// The chunk draw's colour target as a PNG.
fn save_chunk_colour(flight: &Flight, world: &helio_pass_voxel_planet::chunks::ChunkWorld, path: &str) {
    let [w, h] = flight.size;
    let row = (w * 4).div_ceil(256) * 256;
    let buffer = flight.device.create_buffer(&wgpu::BufferDescriptor { label: None, size: u64::from(row) * u64::from(h), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = flight.device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        world.targets().unwrap().colour.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    flight.queue.submit([encoder.finish()]);
    buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    flight.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap().to_vec();
    let mut pixels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let at = (y * row) as usize;
        pixels.extend_from_slice(&data[at..at + (w * 4) as usize]);
    }
    image::save_buffer(path, &pixels, w, h, image::ColorType::Rgba8).unwrap();
}

/// A tunnel carved by 150 brushes (3-7 m balls) from the surface down and
/// ahead: the eye inside it and a view along it.
fn carve_tunnel(flight: &mut Flight, ground: DVec3, heading: f64) -> (DVec3, Vec3) {
    let up = ground.normalize();
    let ahead = tangent(ground, heading).as_dvec3();
    let side = ahead.cross(up);
    let mut planet = (*flight.planet).clone();
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        (rng >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut at = ground - up * 2.0;
    let mut eye = at;
    for n in 0..150 {
        let radius = 3.0 + 4.0 * next();
        planet.apply(Brush { center: at.to_array(), radius, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
        if n == 20 {
            eye = at;
        }
        at += ahead * 2.5 - up * 0.6 + side * ((next() - 0.5) * 2.0);
    }
    flight.planet = Arc::new(planet);
    (eye, (ahead - up * 0.25).normalize().as_vec3())
}

/// Views of generated volumetric terrain near `near`: inside a tunnel (an
/// air pocket with walls within 8 m, below its column's heightfield top) and
/// under an overhang (air below a solid cell above the heightfield top).
fn volume_views(planet: &Planet, near: DVec3) -> Vec<(&'static str, DVec3, Vec3)> {
    use helio_pass_voxel_planet::Cell;
    let grid = *planet.grid();
    let field = planet.field();
    let (base, _) = grid.locate(near);
    let mut rng = 0x2545_F491_4F6C_DD1Du64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let (mut tunnel, mut overhang) = (None, None);
    for _ in 0..200_000 {
        if tunnel.is_some() && overhang.is_some() {
            break;
        }
        let i = base.i + (next() % 20_000) as i32 - 10_000;
        let j = base.j + (next() % 20_000) as i32 - 10_000;
        let (below, above) = field.extent(grid.domain_point(base.face, i, j, 0), 0);
        let top = planet.column_top(base.face, i, j, 0);
        let solid = |di: i32, dj: i32, k: i32| planet.solid(Cell::new(base.face, i + di, j + dj, k));
        if tunnel.is_none() && below > 4 {
            let k = top - 4 - (next() % below as u64) as i32;
            if (-1..=1).all(|a| (-1..=1).all(|b| (-1..=1).all(|c| !solid(a, b, k + c)))) {
                let eye = grid.cell_center(Cell::new(base.face, i, j, k));
                let up = grid.up(eye);
                let ahead = up.any_orthonormal_vector();
                let side = up.cross(ahead);
                if [ahead, -ahead, side, -side, up, -up].iter().all(|d| planet.raycast(eye, *d, 8.0).is_some()) {
                    tunnel = Some(("cave_tunnel", eye, (ahead - up * 0.1).normalize().as_vec3()));
                }
            }
        }
        if overhang.is_none() && above > 2 {
            let k = top + (next() % above as u64) as i32;
            if solid(0, 0, k) && !solid(0, 0, k - 1) && !solid(0, 0, k - 2) {
                let eye = grid.cell_center(Cell::new(base.face, i, j, k - 2));
                let up = grid.up(eye);
                let ahead = up.any_orthonormal_vector();
                overhang = Some(("overhang", eye - ahead * 6.0, (ahead + up * 0.25).normalize().as_vec3()));
            }
        }
    }
    tunnel.into_iter().chain(overhang).collect()
}

/// The highest summit near the spawn and two views of it: 300 m above the
/// ground 12 km away, and on its slope 1.5 km below the summit.
fn mountain(planet: &Planet, heading: f64) -> Vec<(&'static str, DVec3, Vec3)> {
    let (peak_dir, height) = highest_near(planet, 2, 0.47, 0.53, 0.15);
    eprintln!("VOXEL_FLIGHT mountain summit {height:.0} m");
    let peak = planet.surface_point(peak_dir, 0.0);
    let away = tangent(peak, heading + std::f64::consts::PI).as_dvec3();
    let air = planet.surface_point(peak + away * 12_000.0, 300.0);
    let slope = planet.surface_point(peak + away * 1_500.0, 1.7);
    [("mountain_air", air), ("mountain_slope", slope)].into_iter().map(|(name, eye)| (name, eye, (peak - eye).normalize().as_vec3())).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let output = PathBuf::from(args.get(1).expect("OUTPUT directory"));
    std::fs::create_dir_all(&output).unwrap();
    let size = [
        args.get(2).map_or(1280, |s| s.parse().unwrap()),
        args.get(3).map_or(720, |s| s.parse().unwrap()),
    ];
    let quality = match args.get(4).map(String::as_str).unwrap_or("native") {
        "native" => helio_pass_tsr::TsrQuality::Native,
        "quality" => helio_pass_tsr::TsrQuality::Quality,
        other => panic!("unknown quality {other}"),
    };
    let voxel: f64 = std::env::var("HELIO_VOXEL_FLIGHT_VOXEL").ok().map_or(0.1, |v| v.parse().unwrap());
    // HELIO_VOXEL_FLIGHT_HEIGHTFIELD=1: the Earth stack without caves and
    // overhangs (A/B for volumetric columns).
    // HELIO_VOXEL_FLIGHT_SEED=<n>: the Earth stack with another seed (7).
    let seed = std::env::var("HELIO_VOXEL_FLIGHT_SEED").ok().and_then(|v| v.parse().ok()).unwrap_or(7);
    // HELIO_VOXEL_FLIGHT_PRESET=earth|moon|desert: the layer stack (Earth).
    let stack = match std::env::var("HELIO_VOXEL_FLIGHT_PRESET").as_deref() {
        Ok("moon") => helio_pass_voxel_planet::layers::TerrainLayers::moon(),
        Ok("desert") => helio_pass_voxel_planet::layers::TerrainLayers::desert(),
        Ok("earth") | Err(_) => helio_pass_voxel_planet::layers::TerrainLayers::earth(),
        Ok(other) => panic!("unknown preset {other}"),
    };
    // HELIO_VOXEL_FLIGHT_GRAIN=scale_m,octaves,ratio: an extra roughness
    // layer (A/B for ground grain).
    let mut stack = stack;
    // HELIO_VOXEL_FLIGHT_NO_CAVES / _NO_OVERHANGS: A/B for volumetric terms.
    stack.caves.enabled &= std::env::var_os("HELIO_VOXEL_FLIGHT_NO_CAVES").is_none();
    stack.overhangs.enabled &= std::env::var_os("HELIO_VOXEL_FLIGHT_NO_OVERHANGS").is_none();
    if let Ok(grain) = std::env::var("HELIO_VOXEL_FLIGHT_GRAIN") {
        use helio_pass_voxel_planet::layers::{Layer, LayerKind, LayerMask};
        let v: Vec<f64> = grain.split(',').map(|x| x.trim().parse().unwrap()).collect();
        stack.layers.push(Layer { kind: LayerKind::Roughness, mask: LayerMask::AboveDeepSea, scale_km: v[0] / 1000.0, octaves: v[1] as u32, ratio: v[2], ..Layer::default() });
    }
    let terrain = if std::env::var_os("HELIO_VOXEL_FLIGHT_HEIGHTFIELD").is_some() {
        stack.heightfield().source(seed)
    } else {
        stack.source(seed)
    };
    let planet = Planet::new(PlanetRecipe { voxel_size_m: voxel, terrain, ..Default::default() }).unwrap();
    let mut flight = Flight::new(&output, size, quality, planet);

    // Ground spawn on the +Y face.
    let dir = land_near(&flight.planet, 2, 0.47, 0.53, 20.0);
    let ground = flight.planet.surface_point(dir, 1.7);
    let heading = 0.6;
    if let Ok(mode) = std::env::var("HELIO_VOXEL_CHUNKS_FLY") {
        chunk_fly(&flight, &mode, ground, heading);
        return;
    }
    if let Ok(mode) = std::env::var("HELIO_VOXEL_FLIGHT_STRESS") {
        chunk_stress(&mut flight, &mode, ground, heading);
        return;
    }
    if let Ok(view) = std::env::var("HELIO_VOXEL_FLIGHT_HOLD") {
        let (e, f) = match view.as_str() {
            "tunnel" => carve_tunnel(&mut flight, ground, heading),
            "mountain_slope" => {
                let (_, e, f) = mountain(&flight.planet, heading)[1];
                (e, f)
            }
            // Inside the natural cave nearest the spawn (searched 3 km
            // apart, cave regions are ~6 km), along it.
            "cave" => {
                let up = ground.normalize();
                let ahead = tangent(ground, heading).as_dvec3();
                let side = ahead.cross(up);
                let (_, e, f) = (0..49)
                    .map(|n| (n % 7 - 3, n / 7 - 3))
                    .map(|(a, b)| flight.planet.surface_point(ground + ahead * (3000.0 * f64::from(a)) + side * (3000.0 * f64::from(b)), 0.0))
                    .find_map(|near| volume_views(&flight.planet, near).into_iter().find(|v| v.0 == "cave_tunnel"))
                    .expect("a cave within 9 km of the spawn");
                (e, f)
            }
            // `HELIO_VOXEL_FLIGHT_EYE=x,y,z,fx,fy,fz`: any eye (planet-centred)
            // and view direction.
            "eye" => {
                let v: Vec<f64> = std::env::var("HELIO_VOXEL_FLIGHT_EYE").expect("HELIO_VOXEL_FLIGHT_EYE").split(',').map(|x| x.trim().parse().unwrap()).collect();
                (DVec3::new(v[0], v[1], v[2]), DVec3::new(v[3], v[4], v[5]).normalize().as_vec3())
            }
            // 40 m from the +Y face's edge, looking across it.
            "face_edge" => {
                let grid = flight.planet.grid();
                let n = f64::from(grid.cells());
                let eye = flight.planet.surface_point(grid.direction(2, n - 400.0, n * 0.5), 1.7);
                let far = flight.planet.surface_point(grid.direction(2, n + 3000.0, n * 0.5), 0.0);
                (eye, (far - eye).normalize().as_vec3())
            }
            _ => (ground, look(ground, heading, -12.0)),
        };
        if std::env::var_os("HELIO_VOXEL_CHUNKS_ENGINE").is_some() {
            chunk_engine_hold(&mut flight, &view, e, f);
        } else if let Some(count) = std::env::var("HELIO_VOXEL_CHUNKS_DIG").ok().and_then(|v| v.parse().ok()) {
            chunk_dig(&flight, &view, e, f, count, std::env::var("HELIO_VOXEL_CHUNKS_BLAST").ok().and_then(|v| v.parse().ok()));
        } else {
            chunk_hold(&flight, &view, e, f);
        }
        return;
    }
    eprintln!("set HELIO_VOXEL_FLIGHT_HOLD or HELIO_VOXEL_CHUNKS_FLY (see the module docs)");
}
