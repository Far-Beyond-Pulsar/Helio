//! Full-engine voxel planet flight: movement sequences, frame timings,
//! terrain GPU stages, residency/memory, arrival and edit latency, canonical
//! CPU/GPU agreement audits and acceptance gates.
//!
//! cargo run -p helio-default-graphs --release --example voxel_flight -- OUTPUT [WIDTH HEIGHT [native|quality]]
//!
//! Frames are pipelined like a game loop (at most `FRAMES_IN_FLIGHT` ahead of
//! the GPU); a frame's time is the interval between frame completions, and
//! harness-only work (captures, readbacks, audits) is excluded from it. GPU
//! stage timestamps arrive a frame or two later and are attached to the frame
//! that produced them. Waiting for every frame instead lets the GPU idle and
//! the driver lower its clocks, which inflates every GPU timing; `frames.csv`
//! records the graphics and memory clocks per frame (via `nvidia-smi`).
//!
//! Environment:
//! * `HELIO_VOXEL_FLIGHT_SYNC=1` waits for every frame (old behaviour).
//! * `HELIO_VOXEL_FLIGHT_RECORD=1` saves every other movement frame (visual
//!   review run; capture readback perturbs timings, so do not use it for gates).
//! * `HELIO_VOXEL_FLIGHT_VOXEL=0.3` authored base voxel size (0.1..1.0).
//! * `HELIO_VOXEL_FLIGHT_NO_SUN=1` disables traced terrain sunlight.
//! * `HELIO_VOXEL_FLIGHT_QUICK=1` short timing probe with stage summaries.
//! * `HELIO_VOXEL_FLIGHT_GROUND_ONLY=1` ground views and audits only.
//! * `HELIO_VOXEL_FLIGHT_HEAT=1` saves traversal step heatmaps with audits.
//! * `HELIO_VOXEL_FLIGHT_CPU_PROBE=1` per-pass CPU cost of a steady view.
//! * `HELIO_VOXEL_FLIGHT_REVERSAL_AUDIT=1` audits two frames of the fast descent.
//! * `HELIO_VOXEL_FLIGHT_DEBUG=<mode>` Helio debug view.
//! * `HELIO_VOXEL_FLIGHT_HEIGHTFIELD=1` the Earth stack without caves and overhangs.
//! * `HELIO_VOXEL_FLIGHT_SCULPT=1` sculpting stress (editor strokes, edits
//!   piling up in one area): per-stroke frame, brush and generation cost.
//!
//! `gates.json` / `gates.md` evaluate the declared acceptance targets.
use glam::{DVec3, Vec3};
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera, Renderer,
    RendererBuilder, RendererConfig,
};
use helio_default_graphs::{build_default_graph_external_with_voxel_passes, VoxelPassFactory};
use helio_pass_voxel_planet::engine::{PlanetFrame, PlanetPass, SharedPlanetFrame};
use helio_pass_voxel_planet::{grid::Shape, terrain::material, Brush, BrushOp, BrushShape, Planet, PlanetRecipe};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const DT: f64 = 1.0 / 60.0;

#[derive(Clone, Debug, Default)]
struct Sample {
    stage: String,
    /// Frame time: the interval between consecutive frame completions when
    /// pipelined, the synchronized render time otherwise.
    sync_ms: f64,
    submit_ms: f64,
    /// NaN until the frame's GPU timestamps arrive.
    terrain_gpu_ms: f64,
    stages: BTreeMap<&'static str, f64>,
    frame_num: u64,
    /// Pre-formatted CSV columns after the timing columns.
    row: String,
    altitude: f64,
}

/// Frames the CPU may run ahead of the GPU in pipelined mode.
const FRAMES_IN_FLIGHT: usize = 2;

struct Flight {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    source: SharedPlanetFrame,
    planet: Arc<Planet>,
    target: wgpu::Texture,
    size: [u32; 2],
    frame: u64,
    csv: std::fs::File,
    sun: Vec3,
    shadows: bool,
    record: bool,
    output: PathBuf,
    samples: Vec<Sample>,
    clocks: GpuClocks,
    /// Pipelined frames not yet known complete.
    in_flight: std::collections::VecDeque<Arc<std::sync::atomic::AtomicBool>>,
    /// End of the last frame; cleared by harness-only work (captures,
    /// readbacks, audits) so it never counts toward the next frame time.
    last_frame_end: std::cell::Cell<Option<Instant>>,
    /// Wait for every frame (HELIO_VOXEL_FLIGHT_SYNC): lets the GPU idle and
    /// downclock between frames, unlike a game's continuous submission.
    sync_frames: bool,
}

/// Latest GPU graphics and memory clocks (MHz) from a streaming
/// `nvidia-smi` query; zero when unavailable. Drivers lower clocks while the
/// GPU idles between synchronized frames, which scales every GPU timing, so
/// frames record the clock they ran at.
struct GpuClocks {
    latest: Arc<std::sync::atomic::AtomicU64>,
    _child: Option<std::process::Child>,
}

impl GpuClocks {
    fn start() -> Self {
        use std::io::BufRead;
        let latest = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let child = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=clocks.gr,clocks.mem", "--format=csv,noheader,nounits", "-lms", "20"])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null())
            .spawn()
            .ok();
        let mut child = child;
        if let Some(stdout) = child.as_mut().and_then(|c| c.stdout.take()) {
            let latest = latest.clone();
            std::thread::spawn(move || {
                for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
                    let mut parts = line.split(',').map(|p| p.trim().parse::<u64>().unwrap_or(0));
                    let (gr, mem) = (parts.next().unwrap_or(0), parts.next().unwrap_or(0));
                    latest.store(gr | (mem << 32), std::sync::atomic::Ordering::Relaxed);
                }
            });
        }
        Self { latest, _child: child }
    }
    fn read(&self) -> (u64, u64) {
        let v = self.latest.load(std::sync::atomic::Ordering::Relaxed);
        (v & 0xffff_ffff, v >> 32)
    }
}

impl Drop for GpuClocks {
    fn drop(&mut self) {
        if let Some(child) = &mut self._child {
            let _ = child.kill();
        }
    }
}

/// Set while the flight renders a plane world (up is +Y there).
static PLANE_WORLD: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn up_for(eye: DVec3) -> Vec3 {
    if PLANE_WORLD.load(std::sync::atomic::Ordering::Relaxed) {
        Vec3::Y
    } else {
        eye.normalize().as_vec3()
    }
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
    fn new(output: &Path, size: [u32; 2], quality: helio_pass_tsr::TsrQuality, planet: Planet) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
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
        helio_pass_sky::SkyComponent::register_gpu_columns_growable(&mut store, 4, &device);
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
        let factory: VoxelPassFactory = Arc::new(move |_, _, _, _| {
            let mut pass = PlanetPass::new(pass_source.clone());
            pass.set_profiling(true);
            Box::new(pass)
        });
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
        let target = Self::make_target(&device, size);
        let csv = std::fs::File::create(output.join("frames.csv")).unwrap();
        Self {
            device,
            queue,
            renderer,
            source,
            planet: Arc::new(planet),
            target,
            size,
            frame: 0,
            csv,
            sun: sun_dir,
            shadows: std::env::var_os("HELIO_VOXEL_FLIGHT_NO_SUN").is_none(),
            record: std::env::var_os("HELIO_VOXEL_FLIGHT_RECORD").is_some(),
            output: output.to_path_buf(),
            samples: Vec::new(),
            clocks: GpuClocks::start(),
            in_flight: Default::default(),
            last_frame_end: std::cell::Cell::new(None),
            sync_frames: std::env::var_os("HELIO_VOXEL_FLIGHT_SYNC").is_some(),
        }
    }

    fn make_target(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("flight output"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn pass(&mut self) -> &mut PlanetPass {
        self.renderer.find_pass_mut::<PlanetPass>().unwrap()
    }

    fn draw(&mut self, stage: &str, eye: DVec3, forward: Vec3) -> f64 {
        self.draw_with_up(stage, eye, forward, None)
    }

    fn draw_with_up(&mut self, stage: &str, eye: DVec3, forward: Vec3, view_up: Option<Vec3>) -> f64 {
        *self.source.lock().unwrap() = Some(PlanetFrame { eye, planet: self.planet.clone(), sun: self.sun, shadows: self.shadows, picks: None });
        self.renderer.set_world_origin(Some(eye));
        let up = up_for(eye);
        let forward = forward.normalize();
        let up = view_up.unwrap_or_else(|| if forward.dot(up).abs() > 0.999 { up.any_orthonormal_vector() } else { up });
        let near = (self.planet.air_clearance(eye) * 0.25).clamp(0.05, 50_000.0) as f32;
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        let camera = Camera::perspective_look_at(Vec3::ZERO, forward, up, std::f32::consts::FRAC_PI_4, aspect, near, 40_000_000.0);
        let start = Instant::now();
        self.renderer.render(&camera, &self.target.create_view(&Default::default())).unwrap();
        let submit = start.elapsed().as_secs_f64() * 1000.0;
        let done = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let flag = done.clone();
        self.queue.on_submitted_work_done(move || flag.store(true, std::sync::atomic::Ordering::Release));
        self.in_flight.push_back(done);
        if self.sync_frames {
            self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            self.in_flight.clear();
        } else {
            // Keep the GPU fed like a game loop: block only on the frame
            // FRAMES_IN_FLIGHT behind this one.
            while self.in_flight.len() > FRAMES_IN_FLIGHT {
                let oldest = self.in_flight.pop_front().unwrap();
                while !oldest.load(std::sync::atomic::Ordering::Acquire) {
                    let _ = self.device.poll(wgpu::PollType::Poll);
                    std::thread::sleep(std::time::Duration::from_micros(50));
                }
            }
        }
        let now = Instant::now();
        let sync = if self.sync_frames {
            (now - start).as_secs_f64() * 1000.0
        } else {
            (now - self.last_frame_end.get().unwrap_or(start).min(start)).as_secs_f64() * 1000.0
        };
        self.last_frame_end.set(Some(now));
        let frame_num = self.pass().renderer().map_or(0, |r| r.frame_number());
        let stats = self.pass().stats().unwrap_or_default();
        let clocks = self.clocks.read();
        let row = format!(
            "{},{},{},{:.1},{:.1},{:.3},{},{},{},{},{:.4},{:.4},{:.4},{:.1},{},{}",
            stats.resident_columns,
            stats.pending_columns,
            stats.jobs,
            stats.units,
            stats.unit_budget,
            stats.us_per_unit,
            stats.evictions,
            stats.failed_jobs,
            stats.active_levels,
            stats.finest_level,
            stats.plan_cpu_ms,
            stats.upload_cpu_ms,
            stats.encode_cpu_ms,
            stats.logical_bytes as f64 / 1048576.0,
            clocks.0,
            clocks.1
        );
        let altitude = self.planet.grid().height(eye);
        self.samples.push(Sample {
            stage: stage.to_string(),
            sync_ms: sync,
            submit_ms: submit,
            terrain_gpu_ms: f64::NAN,
            stages: BTreeMap::new(),
            frame_num,
            row,
            altitude,
        });
        self.collect_timings(self.sync_frames);
        self.frame += 1;
        sync
    }

    /// Attach completed GPU stage timings to their frame's sample.
    fn collect_timings(&mut self, blocking: bool) {
        let result = self.pass().renderer_mut().and_then(|r| {
            if blocking {
                let timings = r.stage_timings_blocking();
                Some((r.frame_number(), timings))
            } else {
                r.stage_timings_deferred()
            }
        });
        let Some((frame, timings)) = result else { return };
        if let Some(sample) = self.samples.iter_mut().rev().take(16).find(|s| s.frame_num == frame) {
            if sample.terrain_gpu_ms.is_nan() {
                let mut stages = BTreeMap::new();
                for (name, ms) in timings {
                    *stages.entry(name).or_insert(0.0) += ms;
                }
                // `planet_generate` is timed inside `planet_residency`.
                sample.terrain_gpu_ms = stages.iter().filter(|(name, _)| **name != "planet_generate").map(|(_, ms)| ms).sum();
                sample.stages = stages;
            }
        }
    }

    /// Write frames.csv (after the flight: GPU timings arrive late).
    fn write_csv(&mut self) {
        writeln!(self.csv, "frame,stage,altitude_m,sync_ms,cpu_submit_ms,gpu_wait_ms,terrain_gpu_ms,residency_ms,generate_ms,primary_ms,shade_ms,gbuffer_ms,sunlight_ms,resident,pending,jobs,units,unit_budget,us_per_unit,evictions,failed,active_levels,finest_level,plan_cpu_ms,upload_cpu_ms,encode_cpu_ms,logical_mib,gpu_clock_mhz,mem_clock_mhz").unwrap();
        for (index, s) in self.samples.iter().enumerate() {
            if s.terrain_gpu_ms.is_nan() {
                continue;
            }
            let get = |n: &str| s.stages.get(n).copied().unwrap_or(0.0);
            writeln!(
                self.csv,
                "{index},{},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{}",
                s.stage,
                s.altitude,
                s.sync_ms,
                s.submit_ms,
                s.sync_ms - s.submit_ms,
                s.terrain_gpu_ms,
                get("planet_residency"),
                get("planet_generate"),
                get("planet_primary"),
                get("planet_shade"),
                get("planet_gbuffer"),
                get("planet_sunlight"),
                s.row
            )
            .unwrap();
        }
        self.csv.flush().unwrap();
    }

    /// Draw until residency is complete; returns (frames, synchronized ms).
    fn settle(&mut self, stage: &str, eye: DVec3, forward: Vec3) -> (usize, f64) {
        let mut ms = 0.0;
        // Frames before the pipelines finish compiling (on a worker) draw
        // no terrain and do not count.
        let mut frames = 0;
        while frames < 3000 {
            ms += self.draw(stage, eye, forward);
            match self.pass().renderer() {
                Some(r) if r.settled() => return (frames + 1, ms),
                Some(_) => frames += 1,
                None => {}
            }
        }
        panic!("{stage}: residency did not settle: {:?}", self.pass().renderer().map(|r| r.stats()));
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        self.read_range(buffer, 0, buffer.size())
    }

    /// Read `size` bytes at `offset` (both multiples of 4).
    fn read_range(&self, buffer: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
        self.last_frame_end.set(None);
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, size);
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = staging.slice(..).get_mapped_range().unwrap().to_vec();
        data
    }

    /// Sun visibility per pixel (the planet pass's directional visibility).
    fn read_sun(&self) -> Vec<f32> {
        self.last_frame_end.set(None);
        let r = self.renderer.find_pass::<PlanetPass>().unwrap().renderer().unwrap();
        let texture = r.sun_texture();
        let (w, h) = (texture.width(), texture.height());
        let row = (w * 8).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row) * u64::from(h),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap().to_vec();
        let f16 = |bits: u16| -> f32 {
            let sign = if bits & 0x8000 != 0 { -1.0 } else { 1.0 };
            let exp = i32::from((bits >> 10) & 0x1f);
            let frac = f32::from(bits & 0x3ff);
            sign * if exp == 0 { frac * 2f32.powi(-24) } else { (1.0 + frac / 1024.0) * 2f32.powi(exp - 15) }
        };
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let at = (y * row + x * 8) as usize;
                out.push(f16(u16::from_le_bytes([data[at], data[at + 1]])));
            }
        }
        out
    }

    fn capture(&self, name: &str) -> Vec<u8> {
        self.last_frame_end.set(None);
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
        pixels
    }

    /// Hit status counts plus CPU/GPU agreement for sampled primary rays that
    /// land inside the level-0 range (internal render resolution).
    fn audit(&mut self, name: &str, eye: DVec3, forward: Vec3) -> serde_json::Value {
        // Exact pixel-centre rays: the audit frame is rendered without TAA jitter.
        self.renderer.set_jitter_enabled(false);
        self.draw("audit", eye, forward);
        self.renderer.set_jitter_enabled(true);
        let (hits, size, lod0) = {
            let r = self.renderer.find_pass::<PlanetPass>().unwrap().renderer().unwrap();
            (self.read(r.hit_buffer()), r.screen_size(), r.stats().lod0_distance)
        };
        let sun_vis = self.read_sun();
        let mut counts = [0usize; 4];
        let up0 = up_for(eye);
        let forward = forward.normalize();
        let up = if forward.dot(up0).abs() > 0.999 { up0.any_orthonormal_vector() } else { up0 };
        let right = forward.cross(up).normalize();
        let cam_up = right.cross(forward);
        let tan = (std::f32::consts::FRAC_PI_4 * 0.5).tan();
        let aspect = size[0] as f32 / size[1] as f32;
        let (mut compared, mut mismatched) = (0usize, 0usize);
        let mut exact_mismatched = 0usize;
        let (mut sun_compared, mut sun_mismatched) = (0usize, 0usize);
        let mut sun_samples = Vec::new();
        let mut samples = Vec::new();
        let mut stuck = Vec::new();
        for (index, hit) in hits.chunks_exact(32).take((size[0] * size[1]) as usize).enumerate() {
            let w = |i: usize| u32::from_le_bytes(hit[i * 4..i * 4 + 4].try_into().unwrap());
            let info = w(4);
            counts[(info & 3) as usize] += 1;
            if info & 3 == 2 && stuck.len() < 4 {
                stuck.push(format!(
                    "px {},{} t {} face {} level {} i {} j {} k {} normal {}",
                    index as u32 % size[0], index as u32 / size[0], f32::from_bits(w(0)),
                    (info >> 2) & 7, (info >> 5) & 31, w(1) as i32, w(2) as i32, w(3) as i32, (info >> 10) & 7
                ));
            }
            let (x, y) = (index as u32 % size[0], index as u32 / size[0]);
            if x % 7 != 3 || y % 7 != 3 || info & 3 != 1 || ((info >> 5) & 31) != 0 {
                continue;
            }
            let ndc = [(x as f32 + 0.5) / size[0] as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / size[1] as f32 * 2.0];
            let dir = (forward + right * ndc[0] * tan * aspect + cam_up * ndc[1] * tan).normalize().as_dvec3();
            if let Some(cpu) = self.planet.raycast(eye, dir, lod0 * 0.6) {
                compared += 1;
                // Sunlight: an exact CPU shadow ray from the same surface point
                // (sunlit faces only; the GPU marks back faces unlit).
                if self.shadows && cpu.normal.dot(self.sun.as_dvec3()) > 0.05 {
                    let point = eye + dir * cpu.distance + cpu.normal * (self.planet.grid().voxel_size() * 0.02);
                    // Occluders within 200 m (near field; farther shadows are not audited).
                    let lit = self.planet.raycast(point, self.sun.as_dvec3(), 200.0).is_none();
                    let gpu = sun_vis[index];
                    // TAA-rotated 2x2 sharing can legitimately borrow a neighbour's ray.
                    if gpu >= 0.0 {
                        sun_compared += 1;
                        if lit != (gpu > 0.5) {
                            sun_mismatched += 1;
                            if sun_samples.len() < 6 {
                                sun_samples.push(format!("px {x},{y} cell {:?} cpu lit {lit} gpu {gpu}", cpu.cell));
                            }
                        }
                    }
                }
                let t = f64::from(f32::from_bits(w(0)));
                let same = (info >> 2) & 7 == u32::from(cpu.cell.face)
                    && w(1) as i32 == cpu.cell.i
                    && w(2) as i32 == cpu.cell.j
                    && w(3) as i32 == cpu.cell.k;
                // Audit jitter is disabled. Keep the old distance-threshold
                // diagnostic, but it is not an exact-cell acceptance gate.
                exact_mismatched += usize::from(!same);
                if !same && (t - cpu.distance).abs() > self.planet.grid().voxel_size() * 3.0 {
                    mismatched += 1;
                    if samples.len() < 4 {
                        // Canonical kind of the GPU's cell: 1 means the CPU ray
                        // passed a solid cell, 0 the GPU hit an air cell.
                        let face = ((info >> 2) & 7) as u8;
                        let kind = self.planet.kind(helio_pass_voxel_planet::Cell { face, i: w(1) as i32, j: w(2) as i32, k: w(3) as i32 });
                        let (on_ray, _) = self.planet.grid().locate(eye + dir * (t + 0.001));
                        samples.push(format!(
                            "px {x},{y} gpu f{face} ({},{},{}) kind {kind} t {t:.3} ray cell ({},{},{}) kind {} cpu f{} ({},{},{}) t {:.3}",
                            w(1) as i32, w(2) as i32, w(3) as i32, on_ray.i, on_ray.j, on_ray.k, self.planet.kind(on_ray),
                            cpu.cell.face, cpu.cell.i, cpu.cell.j, cpu.cell.k, cpu.distance
                        ));
                    }
                }
            }
        }
        // Traversal work histogram from the diagnostic hit counters.
        let mut work: Vec<[u32; 4]> = hits
            .chunks_exact(32)
            .take((size[0] * size[1]) as usize)
            .map(|h| {
                let u = u32::from_le_bytes(h[24..28].try_into().unwrap());
                let v = u32::from_le_bytes(h[28..32].try_into().unwrap());
                [u & 0xffff, u >> 16, v & 0xffff, v >> 16]
            })
            .collect();
        if std::env::var_os("HELIO_VOXEL_FLIGHT_HEAT").is_some() {
            // Step heatmap: black 0, red 16, yellow 32, white 64+ steps.
            let heat: Vec<u8> = work
                .iter()
                .flat_map(|w| {
                    let s = w[0] as f32;
                    let r = (s / 16.0).min(1.0);
                    let g = ((s - 16.0) / 16.0).clamp(0.0, 1.0);
                    let b = ((s - 32.0) / 32.0).clamp(0.0, 1.0);
                    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8, 255]
                })
                .collect();
            image::save_buffer(self.output.join(format!("{name}-steps.png")), &heat, size[0], size[1], image::ColorType::Rgba8).unwrap();
        }
        let mut stats = serde_json::Map::new();
        for (index, name) in ["steps", "lookups", "block_skips", "locates"].iter().enumerate() {
            work.sort_by_key(|w| w[index]);
            let q = |p: f64| work[((work.len() - 1) as f64 * p) as usize][index];
            let mean = work.iter().map(|w| f64::from(w[index])).sum::<f64>() / work.len() as f64;
            stats.insert((*name).into(), serde_json::json!({"mean": mean, "p50": q(0.5), "p95": q(0.95), "max": q(1.0)}));
        }
        serde_json::json!({"name": name, "mismatch_samples": samples, "stuck": stuck, "work": stats, "miss": counts[0], "hit": counts[1], "exhausted": counts[2], "loading": counts[3], "compared": compared, "mismatched": mismatched, "exact_mismatched": exact_mismatched, "sun_compared": sun_compared, "sun_mismatched": sun_mismatched, "sun_samples": sun_samples})
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

/// The highest summit near the spawn and two views of it: 300 m above the/// ground 12 km away, and on its slope 1.5 km below the summit.
struct Mountain {
    peak: DVec3,
    views: Vec<(&'static str, DVec3, Vec3)>,
}

fn mountain(planet: &Planet, heading: f64) -> Mountain {
    let (peak_dir, height) = highest_near(planet, 2, 0.47, 0.53, 0.15);
    eprintln!("VOXEL_FLIGHT mountain summit {height:.0} m");
    let peak = planet.surface_point(peak_dir, 0.0);
    let away = tangent(peak, heading + std::f64::consts::PI).as_dvec3();
    let air = planet.surface_point(peak + away * 12_000.0, 300.0);
    let slope = planet.surface_point(peak + away * 1_500.0, 1.7);
    let views = [("mountain_air", air), ("mountain_slope", slope)]
        .into_iter()
        .map(|(name, eye)| (name, eye, (peak - eye).normalize().as_vec3()))
        .collect();
    Mountain { peak, views }
}

/// Horizontal direction from `eye` toward the summit.
fn level_toward(range: &Mountain, eye: DVec3) -> DVec3 {
    let toward = (range.peak - eye).normalize();
    let up = eye.normalize();
    (toward - up * toward.dot(up)).normalize()
}

fn record_frame(flight: &Flight, stage: &str, index: usize) {
    if (flight.record && index % 2 == 0) || index % 150 == 0 {
        flight.capture(&format!("{stage}-{index:04}"));
    }
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
    let validation = flight.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut report = serde_json::Map::new();
    let mut audits = Vec::new();
    report.insert("config".into(), serde_json::json!({"size": size, "quality": format!("{quality:?}"), "voxel_m": flight.planet.grid().voxel_size(), "shadows": flight.shadows, "record": flight.record}));

    // Ground spawn on the +Y face.
    let dir = land_near(&flight.planet, 2, 0.47, 0.53, 20.0);
    let ground = flight.planet.surface_point(dir, 1.7);
    let heading = 0.6;
    if let Ok(views) = std::env::var("HELIO_VOXEL_FLIGHT_VIEWS") {
        capture_views(&mut flight, &views, ground, heading);
        return;
    }
    if let Some(km) = std::env::var("HELIO_VOXEL_FLIGHT_LODCMP").ok().and_then(|v| v.parse::<f64>().ok()) {
        lod_compare(&mut flight, km);
        return;
    }
    let forward = look(ground, heading, -12.0);
    let (frames, ms) = flight.settle("ground_load", ground, forward);
    eprintln!("VOXEL_FLIGHT ground load {frames} frames {ms:.1} ms");
    report.insert("cold_ground_load".into(), serde_json::json!({"frames": frames, "sync_ms": ms}));
    for _ in 0..60 {
        flight.draw("ground_warm", ground, forward);
    }
    if std::env::var_os("HELIO_VOXEL_FLIGHT_CPU_PROBE").is_some() {
        // CPU cost of Renderer::render per pass over a steady view.
        let mut totals: BTreeMap<&'static str, (f64, u32)> = BTreeMap::new();
        let mut frame_cpu = Vec::new();
        for _ in 0..240 {
            flight.draw("cpu_probe", ground, forward);
            let snap = flight.renderer.timing_snapshot();
            if let Some(t) = snap.total_cpu_ms {
                frame_cpu.push(f64::from(t));
            }
            for pass in &snap.passes {
                if let Some(ms) = pass.cpu_ms {
                    let e = totals.entry(pass.name).or_insert((0.0, 0));
                    e.0 += f64::from(ms);
                    e.1 += 1;
                }
            }
        }
        let mut list: Vec<_> = totals.into_iter().map(|(n, (ms, c))| (ms / f64::from(c), n)).collect();
        list.sort_by(|a, b| b.0.total_cmp(&a.0));
        eprintln!("CPU total_cpu_ms p50 {:.3}", percentile(&frame_cpu, 0.5));
        for (ms, name) in &list {
            eprintln!("CPU pass {name:36} {ms:8.4} ms");
        }
        return;
    }
    flight.capture("ground");
    audits.push(flight.audit("ground", ground, forward));
    for (name, pitch, h) in [("ground_horizon", 0.0, heading + 1.3), ("ground_down", -45.0, heading + 2.5), ("ground_up", 15.0, heading - 1.0)] {
        let f = look(ground, h, pitch);
        flight.settle(name, ground, f);
        for _ in 0..20 {
            flight.draw(name, ground, f);
        }
        flight.capture(name);
        audits.push(flight.audit(name, ground, f));
    }

    if let Some(deg) = std::env::var("HELIO_VOXEL_FLIGHT_TRIP").ok().and_then(|v| v.parse::<f64>().ok()) {
        editor_trip(&mut flight, deg);
        return;
    }
    if let Some(height) = std::env::var("HELIO_VOXEL_FLIGHT_CRUISE").ok().and_then(|v| v.parse::<f64>().ok()) {
        cruise(&mut flight, height);
        return;
    }
    if let Some(secs) = std::env::var("HELIO_VOXEL_FLIGHT_LONG").ok().and_then(|v| v.parse::<f64>().ok()) {
        long_route(&mut flight, secs);
        return;
    }
    if let Ok(log) = std::env::var("HELIO_VOXEL_FLIGHT_REPLAY") {
        replay(&mut flight, Path::new(&log));
        return;
    }
    if std::env::var_os("HELIO_VOXEL_FLIGHT_EDITOR_PATH").is_some() {
        // An editor-style descent: 10 m/s scaled by height/20 m, from 300 km
        // down to 300 m and then a level cruise, looking 25 degrees down.
        // HELIO_VOXEL_FLIGHT_EDITOR_PATH=<deg> starts that far from the pole.
        // Logs residency progress and captures views.
        let out_deg: f64 = std::env::var("HELIO_VOXEL_FLIGHT_EDITOR_PATH").ok().and_then(|v| v.parse().ok()).unwrap_or(0.0);
        let start = DVec3::new(out_deg.to_radians().sin(), out_deg.to_radians().cos(), 0.0);
        let mut eye = start * (flight.planet.grid().radius() + 300_000.0);
        let dt = 1.0 / 120.0;
        let forward = DVec3::X;
        let mut t = 0.0;
        let mut frame = 0usize;
        while t < 45.0 {
            // Height above the ground directly below (what the editor's
            // speed should follow; air_clearance is only a conservative
            // bound, ~0 anywhere below the highest possible terrain).
            let clearance = eye.length() - flight.planet.surface_point(eye, 0.0).length();
            let speed = 10.0 * (clearance / 20.0).clamp(1.0, 1.0e6);
            let low = clearance < 300.0;
            let up = eye.normalize();
            let ahead = (forward - up * forward.dot(up)).normalize();
            let dir = if low { ahead } else { (ahead - up).normalize() };
            eye += dir * speed * dt;
            let (cell, _) = flight.planet.grid().locate(eye);
            if flight.planet.solid(cell) {
                eye = flight.planet.surface_point(eye, 0.5);
            }
            let up = eye.normalize();
            let look = ((forward - up * forward.dot(up)).normalize() - up * 0.47).normalize().as_vec3();
            flight.draw("editor_path", eye, look);
            if frame % 60 == 0 {
                let stats = flight.pass().stats().unwrap_or_default();
                let up = eye.normalize();
                let from_pole = up.y.acos().to_degrees();
                eprintln!(
                    "PATH t {t:5.1}s height {:9.1} m from_pole {from_pole:5.2} deg speed {speed:8.1} m/s resident {} pending {} jobs {} finest {}",
                    flight.planet.grid().height(eye),
                    stats.resident_columns,
                    stats.pending_columns,
                    stats.jobs,
                    stats.finest_level
                );
                if frame % 240 == 0 {
                    flight.capture(&format!("path_{:03}", (t * 10.0) as u32));
                }
            }
            t += dt;
            frame += 1;
        }
        return;
    }
    if std::env::var_os("HELIO_VOXEL_FLIGHT_SKIM").is_some() {
        // An editor camera held forward and down against the ground: it
        // moves 10 m/s forward and 10 m/s down each 120 Hz frame and is
        // lifted 0.5 m above the surface whenever it enters solid terrain.
        // Editor mode draws the editor overlays (grid) like Pulsar's viewport.
        flight.renderer.set_editor_mode(true);
        let mut eye = ground;
        let mut bad = 0;
        for frame in 0..900 {
            let up = up_for(eye).as_dvec3();
            let ahead = tangent(eye, heading).as_dvec3();
            eye += (ahead - up) * (10.0 / 120.0);
            let (cell, _) = flight.planet.grid().locate(eye);
            if flight.planet.solid(cell) {
                eye = flight.planet.surface_point(eye, 0.5);
            }
            flight.draw("skim", eye, tangent(eye, heading));
            if frame % 30 == 29 {
                let pixels = flight.capture(&format!("skim_{frame:03}"));
                let pink = pixels
                    .chunks(4)
                    .filter(|p| i32::from(p[0]) > i32::from(p[1]) + 40 && i32::from(p[0]) > i32::from(p[2]) + 30)
                    .count() as f64
                    / (pixels.len() / 4) as f64;
                let clearance = flight.planet.air_clearance(eye);
                eprintln!("SKIM frame {frame} clearance {clearance:.3} m pink {:.1}%", pink * 100.0);
                bad += usize::from(pink > 0.01);
            }
        }
        eprintln!("SKIM bad captures {bad}");
        return;
    }
    if std::env::var_os("HELIO_VOXEL_FLIGHT_GROUND_ONLY").is_some() {
        for a in &audits {
            eprintln!("GROUND audit {a}");
        }
        flight.write_csv();
        let error = pollster::block_on(validation.pop());
        assert!(error.is_none(), "GPU validation: {error:?}");
        return;
    }
    if std::env::var_os("HELIO_VOXEL_FLIGHT_SCULPT").is_some() {
        sculpt_stress(&mut flight, ground, heading);
        flight.write_csv();
        return;
    }
    if std::env::var_os("HELIO_VOXEL_FLIGHT_QUICK").is_some() {
        // Timing probe: short walk and an orbit view, then stage summaries.
        let mut eye = ground;
        for i in 0..120 {
            let h = heading + (i as f64 * 0.01).sin() * 0.6;
            eye = flight.planet.surface_point(eye + tangent(eye, h).as_dvec3() * 0.05, 1.7);
            flight.draw("walk", eye, look(eye, h, -8.0));
        }
        let h = heading + (119.0f64 * 0.01).sin() * 0.6;
        audits.push(flight.audit("walk_view", eye, look(eye, h, -8.0)));
        flight.capture("walk_view");
        // Steady low-altitude views (the ascent's heaviest bands).
        for (name, alt, pitch) in [("hover_330", 330.0, -41.0), ("hover_3k", 3000.0, -30.0), ("hover_55k", 55_000.0, -30.0)] {
            let e = ground.normalize() * (ground.length() + alt);
            let f = look(e, heading, pitch);
            flight.settle(name, e, f);
            for _ in 0..30 {
                flight.draw(name, e, f);
            }
            flight.capture(name);
            audits.push(flight.audit(name, e, f));
        }
        let volume = volume_views(&flight.planet, ground);
        eprintln!("QUICK volume views: {:?}", volume.iter().map(|v| v.0).collect::<Vec<_>>());
        for (name, e, f) in mountain(&flight.planet, heading).views.into_iter().chain(volume) {
            flight.settle(name, e, f);
            for _ in 0..30 {
                flight.draw(name, e, f);
            }
            flight.capture(name);
            audits.push(flight.audit(name, e, f));
        }
        let orbit = ground.normalize() * (ground.length() + 300_000.0);
        let orbit_look = look(orbit, heading, -65.0);
        flight.settle("orbit_settle", orbit, orbit_look);
        for _ in 0..30 {
            flight.draw("orbit", orbit, orbit_look);
        }
        flight.capture("orbit");
        let mut groups: BTreeMap<String, Vec<&Sample>> = BTreeMap::new();
        for s in &flight.samples {
            groups.entry(s.stage.clone()).or_default().push(s);
        }
        for (name, list) in &groups {
            let sync: Vec<f64> = list.iter().map(|s| s.sync_ms).collect();
            let terrain: Vec<f64> = list.iter().map(|s| s.terrain_gpu_ms).filter(|v| !v.is_nan()).collect();
            let stage = |k: &str| percentile(&list.iter().filter(|s| !s.terrain_gpu_ms.is_nan()).map(|s| s.stages.get(k).copied().unwrap_or(0.0)).collect::<Vec<_>>(), 0.5);
            eprintln!(
                "QUICK {name:16} n={:4} sync p50 {:7.2} p95 {:7.2} terrain p50 {:6.2} p95 {:6.2} | primary {:6.2} shade {:5.2} sun {:6.2} residency {:5.2} horizon {:5.3}",
                list.len(), percentile(&sync, 0.5), percentile(&sync, 0.95), percentile(&terrain, 0.5), percentile(&terrain, 0.95),
                stage("planet_primary"), stage("planet_shade"), stage("planet_sunlight"), stage("planet_residency"), stage("planet_horizon")
            );
        }
        for a in &audits {
            eprintln!("QUICK audit {a}");
        }
        // Whole-graph pass costs averaged over a steady ground view.
        let mut totals: BTreeMap<&'static str, (f64, u32)> = BTreeMap::new();
        let forward = look(ground, heading, -12.0);
        for _ in 0..60 {
            flight.draw("graph_probe", ground, forward);
            let _ = flight.device.poll(wgpu::PollType::wait_indefinitely());
            for pass in &flight.renderer.timing_snapshot().passes {
                if let Some(ms) = pass.gpu_ms {
                    let e = totals.entry(pass.name).or_insert((0.0, 0));
                    e.0 += f64::from(ms);
                    e.1 += 1;
                }
            }
        }
        let mut list: Vec<_> = totals.into_iter().map(|(n, (ms, c))| (ms / f64::from(c), n)).collect();
        list.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (ms, name) in list.iter().take(16) {
            eprintln!("QUICK pass {name:32} {ms:7.3} ms");
        }
        eprintln!("QUICK graph total {:?}", flight.renderer.gpu_frame_ms());
        // Plane worlds: a finite 4 km plane and an infinite plane.
        PLANE_WORLD.store(true, std::sync::atomic::Ordering::Relaxed);
        for (tag, shape) in [("plane", Shape::Plane), ("infinite", Shape::InfinitePlane)] {
            flight.planet = Arc::new(Planet::new(PlanetRecipe { shape, plane_size_m: 4_096.0, ..Default::default() }).unwrap());
            let ground = flight.planet.surface_point(DVec3::new(300.0, 0.0, -200.0), 1.7);
            for (view, height, pitch) in [("ground", 0.0, -12.0), ("air", 300.0, -30.0)] {
                let name: &'static str = Box::leak(format!("{tag}_{view}").into_boxed_str());
                let e = ground + DVec3::Y * height;
                let f = look(e, heading, pitch);
                flight.settle(name, e, f);
                for _ in 0..30 {
                    flight.draw(name, e, f);
                }
                flight.capture(name);
                if std::env::var_os("HELIO_VOXEL_FLIGHT_SUN_FRAMES").is_some() {
                    // Raw sun visibility of consecutive frames (diagnostics).
                    for n in 0..4 {
                        flight.draw(name, e, f);
                        let sun = flight.read_sun();
                        let (w, h) = (flight.size[0], flight.size[1]);
                        let img: Vec<u8> = sun.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0) as u8).collect();
                        image::save_buffer(flight.output.join(format!("{name}-sun{n}.png")), &img, w, h, image::ColorType::L8).unwrap();
                    }
                }
                audits.push(flight.audit(name, e, f));
            }
        }
        PLANE_WORLD.store(false, std::sync::atomic::Ordering::Relaxed);
        // A moon: another layer stack (craters, basins) with the lunar
        // material table (the renderer's appearance defaults to it).
        flight.planet = Arc::new(Planet::new(PlanetRecipe {
            radius_m: 1_737_400.0,
            terrain: helio_pass_voxel_planet::layers::TerrainLayers::moon().source(7),
            ..Default::default()
        }).unwrap());
        let moon_dir = DVec3::new(0.31, 1.0, 0.17).normalize();
        for (view, height, pitch) in [("moon_ground", 1.7, -12.0), ("moon_330", 330.0, -35.0), ("moon_5k", 5_000.0, -40.0), ("moon_orbit", 300_000.0, -70.0)] {
            let e = flight.planet.surface_point(moon_dir, height);
            let f = look(e, heading, pitch);
            flight.settle(view, e, f);
            for _ in 0..30 {
                flight.draw(view, e, f);
            }
            flight.capture(view);
            audits.push(flight.audit(view, e, f));
        }
        let mut groups: BTreeMap<String, Vec<&Sample>> = BTreeMap::new();
        for s in &flight.samples {
            if s.stage.starts_with("plane") || s.stage.starts_with("infinite") || s.stage.starts_with("moon") {
                groups.entry(s.stage.clone()).or_default().push(s);
            }
        }
        for (name, list) in &groups {
            let terrain: Vec<f64> = list.iter().map(|s| s.terrain_gpu_ms).filter(|v| !v.is_nan()).collect();
            let stage = |k: &str| percentile(&list.iter().filter(|s| !s.terrain_gpu_ms.is_nan()).map(|s| s.stages.get(k).copied().unwrap_or(0.0)).collect::<Vec<_>>(), 0.5);
            eprintln!(
                "QUICK {name:16} n={:4} terrain p50 {:6.2} p95 {:6.2} | primary {:6.2} shade {:5.2} sun {:6.2}",
                list.len(), percentile(&terrain, 0.5), percentile(&terrain, 0.95),
                stage("planet_primary"), stage("planet_shade"), stage("planet_sunlight")
            );
        }
        for a in audits.iter().filter(|a| a["name"].as_str().is_some_and(|n| n.starts_with("plane") || n.starts_with("infinite") || n.starts_with("moon"))) {
            eprintln!("QUICK audit {a}");
        }
        flight.write_csv();
        let error = pollster::block_on(validation.pop());
        assert!(error.is_none(), "GPU validation: {error:?}");
        return;
    }
    // Walking, running and vehicle speed over terrain, following the surface.
    let mut eye = ground;
    for (stage, speed, count) in [("walk", 1.5, 360usize), ("run", 12.0, 360), ("vehicle", 60.0, 360)] {
        for i in 0..count {
            let h = heading + (i as f64 * 0.004).sin() * 0.6;
            let step = tangent(eye, h).as_dvec3() * speed * DT;
            eye = flight.planet.surface_point(eye + step, 1.7);
            flight.draw(stage, eye, look(eye, h, -8.0));
            record_frame(&flight, stage, i);
        }
    }
    for i in 0..240 {
        let h = heading + i as f64 / 240.0 * std::f64::consts::TAU;
        flight.draw("rotate", eye, look(eye, h, -5.0));
        record_frame(&flight, "rotate", i);
    }
    let base = eye;
    let top = 300_000.0f64;
    for i in 0..600 {
        let t = i as f64 / 599.0;
        let alt = 1.7 * (top / 1.7).powf(t);
        let e = base.normalize() * (base.length() + alt - 1.7);
        flight.draw("ascent", e, look(e, heading, -20.0 - 50.0 * t));
        record_frame(&flight, "ascent", i);
    }
    let orbit = base.normalize() * (base.length() + top);
    let orbit_look = look(orbit, heading, -65.0);
    flight.settle("orbit_settle", orbit, orbit_look);
    for _ in 0..60 {
        flight.draw("orbit", orbit, orbit_look);
    }
    flight.capture("orbit");
    audits.push(flight.audit("orbit", orbit, orbit_look));
    let limb = look(orbit, heading, -8.0);
    flight.settle("orbit_limb", orbit, limb);
    flight.capture("orbit-limb");
    for i in 0..600 {
        let t = i as f64 / 599.0;
        let alt = top * (1.7 / top).powf(t);
        let e = base.normalize() * (base.length() + alt - 1.7);
        flight.draw("descent", e, look(e, heading, -70.0 + 58.0 * t));
        record_frame(&flight, "descent", i);
    }
    let arrive_look = look(base, heading, -12.0);
    let at_stop = flight.capture("arrival-stop");
    let (mut arrival_ms, mut arrival_frames, mut at_250) = (0.0, 0usize, None);
    loop {
        arrival_ms += flight.draw("arrival", base, arrive_look);
        arrival_frames += 1;
        if at_250.is_none() && arrival_ms >= 250.0 {
            at_250 = Some(flight.capture("arrival-250ms"));
        }
        if flight.pass().renderer().is_some_and(|r| r.settled()) || arrival_frames > 3000 {
            break;
        }
    }
    for _ in 0..30 {
        flight.draw("arrival_settled", base, arrive_look);
    }
    let settled = flight.capture("arrival-settled");
    let diff = |a: &[u8], b: &[u8]| {
        let changed = a
            .chunks_exact(4)
            .zip(b.chunks_exact(4))
            .filter(|(p, q)| (0..3).any(|c| (i32::from(p[c]) - i32::from(q[c])).abs() > 24))
            .count();
        changed as f64 / (a.len() / 4) as f64
    };
    let at_250 = at_250.unwrap_or_else(|| settled.clone());
    report.insert("arrival".into(), serde_json::json!({
        "frames_to_settle": arrival_frames,
        "sync_ms_to_settle": arrival_ms,
        "changed_pixels_stop_vs_settled": diff(&at_stop, &settled),
        "changed_pixels_250ms_vs_settled": diff(&at_250, &settled),
    }));
    audits.push(flight.audit("arrival", base, arrive_look));

    let mut alt = 1.7f64;
    let mut index = 0usize;
    for (target, frames) in [(5_000.0, 90), (20.0, 90), (60_000.0, 120), (3.0, 120)] {
        let start = alt;
        for f in 0..frames {
            let t = (f + 1) as f64 / frames as f64;
            alt = start * (target / start).powf(t);
            let e = base.normalize() * (base.length() + alt - 1.7);
            flight.draw("reversal", e, look(e, heading + t, -30.0));
            record_frame(&flight, "reversal", index);
            if std::env::var_os("HELIO_VOXEL_FLIGHT_REVERSAL_AUDIT").is_some() && (index == 336 || index == 350) {
                let a = flight.audit(&format!("reversal_{index}"), e, look(e, heading + t, -30.0));
                eprintln!("REVERSAL audit alt {alt:.0} {a}");
            }
            index += 1;
        }
    }
    let far_dir = land_near(&flight.planet, 4, 0.31, 0.62, 30.0);
    let far = flight.planet.surface_point(far_dir, 1.7);
    let far_look = look(far, 0.2, -10.0);
    let (frames, ms) = flight.settle("teleport", far, far_look);
    report.insert("teleport".into(), serde_json::json!({"frames_to_settle": frames, "sync_ms_to_settle": ms}));
    flight.capture("teleport");
    audits.push(flight.audit("teleport", far, far_look));

    // Mountains: fly toward the highest summit near the spawn at 150 m/s,
    // 300 m above the ground, then run up its slope.
    let range = mountain(&flight.planet, heading);
    let (_, air, air_look) = range.views[0];
    flight.settle("mountain_settle", air, air_look);
    flight.capture("mountain");
    audits.push(flight.audit("mountain", air, air_look));
    let mut eye = air;
    for i in 0..360 {
        let level = level_toward(&range, eye);
        eye = flight.planet.surface_point(eye + level * 150.0 * DT, 300.0);
        flight.draw("mountain_flight", eye, (range.peak - eye).normalize().as_vec3());
        record_frame(&flight, "mountain_flight", i);
    }
    let (_, slope, slope_look) = range.views[1];
    flight.settle("mountain_settle", slope, slope_look);
    let mut eye = slope;
    for i in 0..240 {
        let level = level_toward(&range, eye);
        eye = flight.planet.surface_point(eye + level * 6.0 * DT, 1.7);
        let f = (level + eye.normalize() * 0.1).normalize().as_vec3();
        flight.draw("mountain_walk", eye, f);
        record_frame(&flight, "mountain_walk", i);
    }
    audits.push(flight.audit("mountain_walk", eye, (level_toward(&range, eye) + eye.normalize() * 0.1).normalize().as_vec3()));

    // Destruction: one brush per frame at the aim point. Latency is measured
    // until the GPU centre hit matches the canonical CPU ray cast.
    let aim = look(base, heading, -25.0);
    flight.settle("dig_prepare", base, aim);
    // The probe compares the exact centre ray, so jitter is off while digging.
    flight.renderer.set_jitter_enabled(false);
    let mut latencies = Vec::new();
    for n in 0..40 {
        let Some(hit) = flight.planet.raycast(base, aim.as_dvec3(), 200.0) else { break };
        let center = flight.planet.grid().cell_center(hit.cell);
        let add = n % 5 == 4;
        let brush = Brush {
            center: (if add { center + base.normalize() * 1.0 } else { center }).to_array(),
            radius: 0.4 + 0.1 * f64::from(n % 7),
            shape: if n % 3 == 0 { BrushShape::Cube } else { BrushShape::Sphere },
            op: if add { BrushOp::Add } else { BrushOp::Remove },
            material: if add { material::COBBLE } else { 0 },
        };
        let mut planet = (*flight.planet).clone();
        planet.apply(brush).unwrap();
        flight.planet = Arc::new(planet);
        let expected = {
            let r = flight.renderer.find_pass::<PlanetPass>().unwrap().renderer().unwrap();
            let size = r.screen_size();
            let up0 = up_for(base);
            let right = aim.cross(up0).normalize();
            let cam_up = right.cross(aim);
            let tan = (std::f32::consts::FRAC_PI_4 * 0.5).tan();
            let aspect = size[0] as f32 / size[1] as f32;
            let ndc = [
                ((size[0] / 2) as f32 + 0.5) / size[0] as f32 * 2.0 - 1.0,
                1.0 - ((size[1] / 2) as f32 + 0.5) / size[1] as f32 * 2.0,
            ];
            let dir = (aim + right * ndc[0] * tan * aspect + cam_up * ndc[1] * tan).normalize().as_dvec3();
            flight.planet.raycast(base, dir, 200.0).map(|h| h.cell)
        };
        // Wall time from the edit to the first frame that shows it.
        let edited = Instant::now();
        let mut frames = 0;
        loop {
            flight.draw("dig", base, aim);
            frames += 1;
            let hit = {
                let r = flight.renderer.find_pass::<PlanetPass>().unwrap().renderer().unwrap();
                let size = r.screen_size();
                // Only the centre hit: a whole-buffer readback would dominate the latency.
                let at = u64::from(size[1] / 2 * size[0] + size[0] / 2) * 32;
                flight.read_range(r.hit_buffer(), at, 32)
            };
            let w = |i: usize| u32::from_le_bytes(hit[i * 4..i * 4 + 4].try_into().unwrap());
            let got = (w(1) as i32, w(2) as i32, w(3) as i32);
            let matched = expected.is_some_and(|c| (c.i, c.j, c.k) == got);
            if matched || frames >= 30 {
                latencies.push((frames, edited.elapsed().as_secs_f64() * 1000.0, matched));
                break;
            }
        }
    }
    flight.renderer.set_jitter_enabled(true);
    // Let regeneration around the last edits finish before auditing.
    flight.settle("dig_settle", base, aim);
    flight.capture("dig");
    audits.push(flight.audit("dig", base, aim));
    let unmatched = latencies.iter().filter(|l| !l.2).count();
    report.insert("edits".into(), serde_json::json!({
        "count": latencies.len(),
        "unmatched_after_30_frames": unmatched,
        "max_frames": latencies.iter().map(|l| l.0).max(),
        "max_ms": latencies.iter().map(|l| l.1).fold(0.0, f64::max),
        "p95_ms": percentile(&latencies.iter().map(|l| l.1).collect::<Vec<_>>(), 0.95),
    }));
    // Large remote destruction from orbit (no tool distance limit).
    let orbit_hit = flight.planet.raycast(orbit, -orbit.normalize(), f64::INFINITY).expect("orbital edit ray");
    let mut planet = (*flight.planet).clone();
    planet
        .apply(Brush {
            center: flight.planet.grid().cell_center(orbit_hit.cell).to_array(),
            radius: 60.0,
            shape: BrushShape::Sphere,
            op: BrushOp::Remove,
            material: 0,
        })
        .unwrap();
    flight.planet = Arc::new(planet);
    flight.settle("orbital_edit", orbit, orbit_look);
    flight.capture("orbital-edit");
    let crater_view = orbit.normalize() * (base.length() + 150.0) + tangent(orbit, heading + 3.0).as_dvec3() * 40.0;
    let crater_look = (orbit.normalize() * (base.length() - 20.0) - crater_view).as_vec3();
    flight.settle("crater", crater_view, crater_look);
    for _ in 0..16 {
        flight.draw("crater", crater_view, crater_look);
    }
    flight.capture("crater");

    // Resize: the graph rebuild must retain residency.
    let before = flight.pass().stats().unwrap().resident_columns;
    flight.size = [size[0] + 64, size[1] + 36];
    flight.target = Flight::make_target(&flight.device, flight.size);
    flight.renderer.set_render_size(flight.size[0], flight.size[1]);
    flight.draw("resize", base, arrive_look);
    let after = flight.pass().stats().unwrap().resident_columns;
    report.insert("resize".into(), serde_json::json!({"resident_before": before, "resident_after": after}));
    flight.settle("resize", base, arrive_look);
    flight.capture("resized");

    // Authored base-grid replacement.
    let mut grids = Vec::new();
    for size_m in [0.3, 1.0] {
        flight.planet = Arc::new(Planet::new(PlanetRecipe { voxel_size_m: size_m, ..Default::default() }).unwrap());
        let e = flight.planet.surface_point(base.normalize(), 1.7);
        let f = look(e, heading, -12.0);
        let name = format!("grid_{size_m}");
        let (frames, ms) = flight.settle(&name, e, f);
        for _ in 0..16 {
            flight.draw(&name, e, f);
        }
        flight.capture(&format!("grid-{size_m}m"));
        audits.push(flight.audit(&name, e, f));
        grids.push(serde_json::json!({"voxel_m": size_m, "frames": frames, "sync_ms": ms}));
    }
    report.insert("grid_replacement".into(), serde_json::Value::Array(grids));
    let error = pollster::block_on(validation.pop());
    assert!(error.is_none(), "GPU validation: {error:?}");

    // Per-stage statistics and gates.
    let stats = flight.pass().stats().unwrap();
    let mut groups: BTreeMap<String, Vec<&Sample>> = BTreeMap::new();
    for s in &flight.samples {
        groups.entry(s.stage.clone()).or_default().push(s);
    }
    let mut stages = serde_json::Map::new();
    for (name, list) in &groups {
        let sync: Vec<f64> = list.iter().map(|s| s.sync_ms).collect();
        let timed: Vec<&&Sample> = list.iter().filter(|s| !s.terrain_gpu_ms.is_nan()).collect();
        let terrain: Vec<f64> = timed.iter().map(|s| s.terrain_gpu_ms).collect();
        let mut stage_p95 = serde_json::Map::new();
        for key in ["planet_residency", "planet_generate", "planet_primary", "planet_shade", "planet_skylight", "planet_gbuffer", "planet_sunlight"] {
            let v: Vec<f64> = timed.iter().map(|s| s.stages.get(key).copied().unwrap_or(0.0)).collect();
            stage_p95.insert(key.into(), serde_json::json!(percentile(&v, 0.95)));
        }
        stages.insert(
            name.clone(),
            serde_json::json!({
                "frames": list.len(),
                "timed_frames": timed.len(),
                "sync_p50": percentile(&sync, 0.5), "sync_p95": percentile(&sync, 0.95), "sync_p99": percentile(&sync, 0.99), "sync_max": percentile(&sync, 1.0),
                "terrain_gpu_p50": percentile(&terrain, 0.5), "terrain_gpu_p95": percentile(&terrain, 0.95), "terrain_gpu_max": percentile(&terrain, 1.0),
                "terrain_stage_p95": stage_p95,
            }),
        );
    }
    report.insert("stages".into(), serde_json::Value::Object(stages));
    report.insert("audits".into(), serde_json::Value::Array(audits.clone()));
    report.insert(
        "memory".into(),
        serde_json::json!({"logical_mib": stats.logical_bytes as f64 / 1048576.0, "free_pool_pages": stats.free_pages, "pool_pages": stats.pool_pages}),
    );
    let gather = |names: &[&str], terrain: bool| -> Vec<f64> {
        names
            .iter()
            .flat_map(|s| groups.get(*s).into_iter().flatten().map(|x| if terrain { x.terrain_gpu_ms } else { x.sync_ms }))
            .filter(|v| !v.is_nan())
            .collect()
    };
    let warm = gather(&["ground_warm", "orbit", "arrival_settled"], false);
    let movement_names = ["walk", "run", "vehicle", "rotate", "ascent", "descent", "reversal", "mountain_flight", "mountain_walk"];
    let moving = gather(&movement_names, false);
    let mut terrain_names = movement_names.to_vec();
    terrain_names.extend(["ground_warm", "orbit"]);
    let terrain = gather(&terrain_names, true);
    let bad_rays: u64 = audits.iter().map(|a| a["exhausted"].as_u64().unwrap() + a["loading"].as_u64().unwrap()).sum();
    let mismatched: u64 = audits.iter().map(|a| a["mismatched"].as_u64().unwrap()).sum();
    let exact_mismatched: u64 = audits.iter().map(|a| a["exact_mismatched"].as_u64().unwrap()).sum();
    let compared: u64 = audits.iter().map(|a| a["compared"].as_u64().unwrap()).sum();
    let edit_max = report["edits"]["max_ms"].as_f64().unwrap_or(f64::INFINITY);
    let arrival = report["arrival"]["sync_ms_to_settle"].as_f64().unwrap();
    report.insert(
        "near_field_distance_threshold_disagreements".into(),
        serde_json::json!({"compared": compared, "mismatched": mismatched, "threshold_base_voxels": 3}),
    );
    let gates = serde_json::json!([
        {"gate": "warm full-graph sync p95 <= 16.67 ms", "value": percentile(&warm, 0.95), "pass": percentile(&warm, 0.95) <= 16.67},
        {"gate": "movement sync p99 <= 25 ms", "value": percentile(&moving, 0.99), "pass": percentile(&moving, 0.99) <= 25.0},
        {"gate": "terrain GPU p95 <= 5 ms (movement + warm)", "value": percentile(&terrain, 0.95), "pass": percentile(&terrain, 0.95) <= 5.0},
        {"gate": "logical terrain GPU memory <= 1024 MiB", "value": stats.logical_bytes as f64 / 1048576.0, "pass": stats.logical_bytes <= 1 << 30},
        {"gate": "arrival settles <= 250 ms after descent", "value": arrival, "pass": arrival <= 250.0},
        {"gate": "visible local edit <= 100 ms", "value": edit_max, "pass": edit_max <= 100.0 && unmatched == 0},
        {"gate": "no exhausted/loading rays in settled audits", "value": bad_rays, "pass": bad_rays == 0},
        {"gate": "sampled near-field exact CPU/GPU cell agreement", "value": format!("{exact_mismatched}/{compared}"), "pass": compared > 0 && exact_mismatched == 0},
        {"gate": "resize keeps residency", "value": report["resize"].clone(), "pass": after >= before / 2},
    ]);
    report.insert("gates".into(), gates.clone());
    std::fs::write(output.join("gates.json"), serde_json::to_string_pretty(&serde_json::Value::Object(report)).unwrap()).unwrap();
    let mut md = String::from("| Gate | Value | Pass |\n|---|---|---|\n");
    for g in gates.as_array().unwrap() {
        md.push_str(&format!(
            "| {} | {} | {} |\n",
            g["gate"].as_str().unwrap(),
            g["value"],
            if g["pass"].as_bool().unwrap() { "yes" } else { "**no**" }
        ));
    }
    std::fs::write(output.join("gates.md"), &md).unwrap();
    eprintln!("{md}");
    flight.write_csv();
    eprintln!("VOXEL_FLIGHT_COMPLETE frames={}", flight.frame);
}

/// Ray statuses of the last frame: (miss rays that must hit the planet,
/// loading, exhausted, all rays). A miss is certain to be a hole when the
/// ray passes below the deepest possible terrain.
fn holes(flight: &Flight, eye: DVec3, forward: Vec3, mask: Option<&str>) -> (usize, usize, usize, usize) {
    let (hits, size) = {
        let r = flight.renderer.find_pass::<PlanetPass>().unwrap().renderer().unwrap();
        (flight.read(r.hit_buffer()), r.screen_size())
    };
    let up0 = up_for(eye);
    let forward = forward.normalize();
    let up = if forward.dot(up0).abs() > 0.999 { up0.any_orthonormal_vector() } else { up0 };
    let right = forward.cross(up).normalize();
    let cam_up = right.cross(forward);
    let tan = (std::f32::consts::FRAC_PI_4 * 0.5).tan();
    let aspect = size[0] as f32 / size[1] as f32;
    let floor = flight.planet.grid().radius() - 12_000.0;
    let (mut miss, mut loading, mut exhausted, mut total) = (0, 0, 0, 0);
    // Hole mask: red misses, yellow loading, magenta exhausted, grey hits.
    let mut image = vec![0u8; (size[0] * size[1] * 4) as usize];
    for (index, hit) in hits.chunks_exact(32).take((size[0] * size[1]) as usize).enumerate() {
        let status = u32::from_le_bytes(hit[16..20].try_into().unwrap()) & 3;
        image[index * 4..index * 4 + 4].copy_from_slice(match status {
            1 => &[90, 90, 90, 255],
            2 => &[255, 0, 255, 255],
            3 => &[255, 255, 0, 255],
            _ => &[0, 0, 0, 255],
        });
        let info = u32::from_le_bytes(hit[16..20].try_into().unwrap());
        total += 1;
        match info & 3 {
            2 => exhausted += 1,
            3 => loading += 1,
            0 => {
                let (x, y) = (index as u32 % size[0], index as u32 / size[0]);
                let ndc = [(x as f32 + 0.5) / size[0] as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / size[1] as f32 * 2.0];
                let d = (forward + right * ndc[0] * tan * aspect + cam_up * ndc[1] * tan).normalize().as_dvec3();
                // Closest approach of the ray to the planet centre.
                let t = (-eye.dot(d)).max(0.0);
                if (eye + d * t).length() < floor {
                    miss += 1;
                    image[index * 4..index * 4 + 4].copy_from_slice(&[255, 0, 0, 255]);
                }
            }
            _ => {}
        }
    }
    if let Some(name) = mask {
        image::save_buffer(flight.output.join(format!("{name}.png")), &image, size[0], size[1], image::ColorType::Rgba8).unwrap();
    }
    (miss, loading, exhausted, total)
}

/// The editor trip of the user's recordings, at `deg` from the pole: climb
/// from the ground to orbit, fly across in orbit, descend, cruise low.
/// Editor speed is 10 m/s x height/20 m. HELIO_VOXEL_FLIGHT_PROBE=1 reads
/// every frame's rays back and reports holes (slows the frames).
fn editor_trip(flight: &mut Flight, deg: f64) {
    let probe = std::env::var_os("HELIO_VOXEL_FLIGHT_PROBE").is_some();
    // HELIO_VOXEL_FLIGHT_BLOCKY=1: log enlarged-block coverage (readbacks).
    let measure_blocks = std::env::var_os("HELIO_VOXEL_FLIGHT_BLOCKY").is_some();
    let capture_every: usize = std::env::var("HELIO_VOXEL_FLIGHT_TRIP_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(120);
    let no_horizon = std::env::var_os("HELIO_VOXEL_FLIGHT_NO_HORIZON").is_some();
    let low_height: f64 = std::env::var("HELIO_VOXEL_FLIGHT_TRIP_LOW").ok().and_then(|v| v.parse().ok()).unwrap_or(30.0);
    let audit_at: Vec<f64> = std::env::var("HELIO_VOXEL_FLIGHT_AUDIT_AT")
        .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    // HELIO_VOXEL_FLIGHT_TRIP_FROM=<km>: start on the flank of the nearest
    // high summit, that far from it (rock, scree and snow instead of meadow).
    let start = match std::env::var("HELIO_VOXEL_FLIGHT_TRIP_FROM").ok().and_then(|v| v.parse::<f64>().ok()) {
        Some(km) => {
            let range = mountain(&flight.planet, 0.0);
            let away = tangent(range.peak, std::f64::consts::PI).as_dvec3();
            (range.peak + away * km * 1000.0).normalize()
        }
        None => DVec3::new(deg.to_radians().sin(), deg.to_radians().cos(), 0.0),
    };
    let mut eye = flight.planet.surface_point(start, 1.7);
    // HELIO_VOXEL_FLIGHT_TRIP_FPS: frames per second of flight (120; the
    // editor runs near 60, which halves the streaming budget per metre).
    let dt = 1.0 / std::env::var("HELIO_VOXEL_FLIGHT_TRIP_FPS").ok().and_then(|v| v.parse::<f64>().ok()).unwrap_or(120.0);
    let mut t = 0.0;
    let mut frame = 0usize;
    let (mut worst, mut worst_at) = (0.0f64, String::new());
    // HELIO_VOXEL_FLIGHT_TRIP_CLIMB: climb seconds (14: ~9 km, 30: ~5000 km;
    // height grows e^(t/2) at the editor's speed).
    let climb: f64 = std::env::var("HELIO_VOXEL_FLIGHT_TRIP_CLIMB").ok().and_then(|v| v.parse().ok()).unwrap_or(14.0);
    let phases: [(&str, f64); 5] = [("settle", 2.0), ("climb", climb), ("orbit", 10.0), ("descend", climb), ("low", 20.0)];
    let mut phase_end = 0.0;
    let mut previous_rings = String::new();
    let trip_end: f64 = std::env::var("HELIO_VOXEL_FLIGHT_TRIP_END").ok().and_then(|v| v.parse().ok()).unwrap_or(f64::INFINITY);
    for (name, duration) in phases {
        phase_end += duration;
        while t < phase_end && t < trip_end {
            let up = eye.normalize();
            let ahead = (DVec3::X - up * DVec3::X.dot(up)).try_normalize().unwrap_or(DVec3::Z);
            let height = eye.length() - flight.planet.surface_point(eye, 0.0).length();
            let speed = 10.0 * (height / 20.0).clamp(1.0, 1.0e6);
            let (dir, look) = match name {
                "settle" => (DVec3::ZERO, (ahead - up * 0.2).normalize()),
                "climb" => (up, (ahead - up * 0.6).normalize()),
                "orbit" => (ahead, (ahead - up * 1.2).normalize()),
                "descend" => (-up, (ahead - up * 0.6).normalize()),
                _ => {
                    // Low flight: hold HELIO_VOXEL_FLIGHT_TRIP_LOW metres (30)
                    // over the ground at the editor's speed.
                    let hold = (low_height - height) * 0.5;
                    ((ahead * speed + up * hold) / speed.max(1.0), (ahead - up * 0.25).normalize())
                }
            };
            eye += dir * speed * dt;
            let (cell, _) = flight.planet.grid().locate(eye);
            if flight.planet.solid(cell) {
                eye = flight.planet.surface_point(eye, 0.5);
            }
            let look = look.as_vec3();
            if no_horizon {
                if let Some(r) = flight.pass().renderer_mut() {
                    r.settings_mut().horizon = false;
                }
            }
            flight.draw(name, eye, look);
            let submit = flight.samples.last().map_or(0.0, |s| s.submit_ms);
            if submit > 25.0 {
                // Where a slow frame's CPU time went.
                let stats = flight.pass().stats().unwrap_or_default();
                let snap = flight.renderer.timing_snapshot();
                let mut passes: Vec<(f32, &str)> = snap.passes.iter().filter_map(|p| p.cpu_ms.map(|ms| (ms, p.name))).collect();
                passes.sort_by(|a, b| b.0.total_cmp(&a.0));
                eprintln!(
                    "SLOW {name} t {t:.2} submit {submit:.1} total_cpu {:?} plan {:.2} upload {:.2} encode {:.2} top {:?}",
                    snap.total_cpu_ms, stats.plan_cpu_ms, stats.upload_cpu_ms, stats.encode_cpu_ms, &passes[..passes.len().min(4)]
                );
            }
            // HELIO_VOXEL_FLIGHT_AUDIT_AT="t1,t2,..": audit traversal work at
            // those trip times (HELIO_VOXEL_FLIGHT_HEAT saves step heatmaps).
            if audit_at.iter().any(|a| (t - a).abs() < dt * 0.5) {
                flight.capture(&format!("audit_{name}_{:05}", (t * 100.0) as u32));
                eprintln!("AUDIT t {t:.2} h {height:.0} {}", flight.audit(&format!("audit_{name}_{:05}", (t * 100.0) as u32), eye, look));
            }
            if probe {
                let rings = {
                    let planet = flight.planet.clone();
                    let pass = flight.pass();
                    let lod0 = pass.stats().map_or(0.0, |s| s.lod0_distance);
                    pass.renderer().map(|r| {
                        let cut = 100.0f64.max(0.75 * planet.air_clearance(eye));
                        let (fallback, rings) = r.sky_rings(eye, lod0, cut);
                        fallback
                            .iter()
                            .zip(&rings)
                            .enumerate()
                            .filter(|(_, (f, _))| **f > 0.0)
                            .map(|(l, (f, r))| format!("L{l}:fb {f:.0} ring {r:.5}"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                };
                let rings = rings.unwrap_or_default();
                let (miss, loading, exhausted, total) = holes(flight, eye, look, None);
                let bad = (miss + loading + exhausted) as f64 / total as f64;
                if bad > worst {
                    worst = bad;
                    worst_at = format!("{name} t {t:.2} h {height:.0}");
                }
                if bad > 0.002 {
                    let stats = flight.pass().stats().unwrap_or_default();
                    eprintln!(
                        "HOLES {name} t {t:6.2} h {height:9.0} miss {miss} loading {loading} exhausted {exhausted} ({:.2}%) resident {} pending {} levels {} finest {}",
                        bad * 100.0, stats.resident_columns, stats.pending_columns, stats.active_levels, stats.finest_level
                    );
                    eprintln!("  rings now  {rings}
  rings prev {previous_rings}");
                    if bad > 0.001 {
                        holes(flight, eye, look, Some(&format!("mask_{name}_{:05}", (t * 100.0) as u32)));
                        flight.capture(&format!("hole_{name}_{:05}", (t * 100.0) as u32));
                    }
                }
                previous_rings = rings;
            } else if frame % 30 == 0 {
                let stats = flight.pass().stats().unwrap_or_default();
                let blocks = if measure_blocks {
                    let (a, b, w) = blocky(flight);
                    format!(" blocky>2px {:.2}% >4px {:.2}% widest {w:.1}", a * 100.0, b * 100.0)
                } else {
                    String::new()
                };
                eprintln!(
                    "TRIP {name} t {t:6.2} h {height:9.0} speed {speed:9.0} resident {} pending {} jobs {} levels {} finest {} plan {:.2} upload {:.2}{blocks}",
                    stats.resident_columns, stats.pending_columns, stats.jobs, stats.active_levels, stats.finest_level, stats.plan_cpu_ms, stats.upload_cpu_ms
                );
            }
            // HELIO_VOXEL_FLIGHT_TRIP_EVERY=<frames>: capture cadence (120).
            if frame % capture_every == 0 {
                flight.capture(&format!("trip_{name}_{:05}", (t * 100.0) as u32));
            }
            t += dt;
            frame += 1;
        }
    }
    if probe {
        eprintln!("HOLES worst {:.2}% at {worst_at}", worst * 100.0);
    }
    flight.write_csv();
}

/// Replays the full camera pose timeline of new Pulsar editor logs (old logs
/// retain their altitude-only fallback). This is an offscreen reproduction,
/// not native presentation timing.
/// Replays the altitude timeline of a Pulsar editor session
/// (`PULSAR_VOXEL_STATS=1` engine log) at 60 frames per second of log time,
/// over one ground point (HELIO_VOXEL_FLIGHT_REPLAY_DEG from the pole, 30),
/// looking 30 degrees down. HELIO_VOXEL_FLIGHT_REPLAY_FROM / _TO limit it
/// to log times (seconds of the day, UTC).
fn log_vector(line: &str, key: &str) -> Option<DVec3> {
    let raw = line.split(key).nth(1)?.trim_start().strip_prefix('[')?.split(']').next()?;
    let values: Vec<f64> = raw.split(',').map(str::trim).map(str::parse).collect::<Result<_, _>>().ok()?;
    if values.len() != 3 || !values.iter().all(|v| v.is_finite()) { return None; }
    Some(DVec3::new(values[0], values[1], values[2]))
}

fn replay(flight: &mut Flight, log: &Path) {
    let text = std::fs::read_to_string(log).expect("replay log");
    let mut points: Vec<(f64, f64, Option<DVec3>, Option<DVec3>, Option<DVec3>)> = Vec::new();
    for line in text.lines().filter(|l| l.contains("VOXEL_STATS")) {
        let Some(time) = line.get(11..26) else { continue };
        let parts: Vec<f64> = time.split(':').filter_map(|v| v.parse().ok()).collect();
        let Some(alt) = line.split("altitude=").nth(1).and_then(|v| v.split(' ').next()).and_then(|v| v.parse().ok()) else { continue };
        if parts.len() != 3 {
            continue;
        }
        let mut t = parts[0] * 3600.0 + parts[1] * 60.0 + parts[2];
        if let Some(&(last, ..)) = points.last() {
            if t < last - 43_200.0 {
                t += 86_400.0;
            }
        }
        points.push((t, alt, log_vector(line, "eye="), log_vector(line, "forward="), log_vector(line, "up=")));
    }
    assert!(points.len() >= 2, "replay needs at least two valid VOXEL_STATS samples");
    eprintln!("REPLAY {} pose samples (legacy altitude samples use a fixed location)", points.iter().filter(|p| p.2.is_some()).count());
    let env = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<f64>().ok());
    let from = env("HELIO_VOXEL_FLIGHT_REPLAY_FROM").unwrap_or(points[0].0);
    let to = env("HELIO_VOXEL_FLIGHT_REPLAY_TO").unwrap_or(points.last().unwrap().0);
    let deg = env("HELIO_VOXEL_FLIGHT_REPLAY_DEG").unwrap_or(30.0);
    let start = DVec3::new(deg.to_radians().sin(), deg.to_radians().cos(), 0.0);
    let ground = flight.planet.surface_point(start, 0.0);
    let up = ground.normalize();
    let ahead = (DVec3::X - up * DVec3::X.dot(up)).normalize();
    let forward = (ahead - up * 0.6).normalize().as_vec3();
    let altitude_at = |t: f64| {
        let i = points.partition_point(|p| p.0 <= t).clamp(1, points.len() - 1);
        let (a, b) = (points[i - 1], points[i]);
        let f = ((t - a.0) / (b.0 - a.0).max(1e-6)).clamp(0.0, 1.0);
        a.1 + (b.1 - a.1) * f
    };
    let mut t = from;
    let mut frame = 0usize;
    while t < to {
        let i = points.partition_point(|p| p.0 <= t).clamp(1, points.len() - 1);
        let (a, b) = (points[i - 1], points[i]);
        let blend = ((t - a.0) / (b.0 - a.0).max(1e-6)).clamp(0.0, 1.0);
        let lerp = |a: Option<DVec3>, b: Option<DVec3>| a.zip(b).map(|(a,b)| a.lerp(b, blend));
        let eye = lerp(a.2, b.2).unwrap_or(ground + up * altitude_at(t).max(1.7));
        let forward = lerp(a.3, b.3).and_then(DVec3::try_normalize).map_or(forward, |v| v.as_vec3());
        let view_up = lerp(a.4, b.4).and_then(DVec3::try_normalize).map(|v| v.as_vec3());
        flight.draw_with_up("replay", eye, forward, view_up);
        if frame % 30 == 0 {
            let stats = flight.pass().stats().unwrap_or_default();
            let (a, b, w) = blocky(flight);
            eprintln!(
                "REPLAY t {t:9.2} h {:9.1} resident {} pending {} jobs {} levels {} finest {} plan {:.2} upload {:.2} blocky>2px {:.2}% >4px {:.2}% widest {w:.1}",
                altitude_at(t), stats.resident_columns, stats.pending_columns, stats.jobs, stats.active_levels, stats.finest_level, stats.plan_cpu_ms, stats.upload_cpu_ms, a * 100.0, b * 100.0
            );
        }
        if frame % 300 == 0 {
            flight.capture(&format!("replay_{:07}", (t * 10.0) as u64));
        }
        t += 1.0 / 60.0;
        frame += 1;
    }
    flight.capture("replay_end");
}

/// Level flight at `height` metres over the ground at the editor's speed
/// (10 m/s x height / 20 m) for HELIO_VOXEL_FLIGHT_CRUISE_SECS (20), then a
/// stop: logs how far residency lags while moving and how long it takes to
/// converge afterwards.
fn cruise(flight: &mut Flight, height: f64) {
    let secs: f64 = std::env::var("HELIO_VOXEL_FLIGHT_CRUISE_SECS").ok().and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let deg: f64 = std::env::var("HELIO_VOXEL_FLIGHT_REPLAY_DEG").ok().and_then(|v| v.parse().ok()).unwrap_or(30.0);
    let start = DVec3::new(deg.to_radians().sin(), deg.to_radians().cos(), 0.0);
    let r = flight.planet.surface_point(start, 0.0).length();
    let mut eye = start * (r + height);
    let speed = std::env::var("HELIO_VOXEL_FLIGHT_CRUISE_SPEED").ok()
        .and_then(|v| v.parse::<f64>().ok()).filter(|v| v.is_finite() && *v > 0.0)
        .unwrap_or(10.0 * (height / 20.0).max(1.0));
    let capture_every = std::env::var("HELIO_VOXEL_FLIGHT_CRUISE_EVERY").ok()
        .and_then(|v| v.parse::<usize>().ok()).filter(|v| *v > 0).unwrap_or(300);
    let dt = 1.0 / 60.0;
    let view = |eye: DVec3| {
        let up = eye.normalize();
        let ahead = (DVec3::X - up * DVec3::X.dot(up)).try_normalize().unwrap_or(DVec3::Z);
        (up, ahead, (ahead - up * 0.6).normalize().as_vec3())
    };
    let (_, _, f) = view(eye);
    flight.settle("cruise_settle", eye, f);
    let mut t = 0.0;
    let mut frame = 0usize;
    let mut converged = None;
    while t < secs + 30.0 {
        let (_, ahead, f) = view(eye);
        let moving = t < secs;
        if moving {
            eye += ahead * speed * dt;
            let ground = flight.planet.surface_point(eye, 0.0).length();
            eye = eye.normalize() * (ground + height);
        }
        let stage = if moving { "cruise" } else { "cruise_stop" };
        flight.draw(stage, eye, f);
        let stats = flight.pass().stats().unwrap_or_default();
        if !moving && converged.is_none() && stats.pending_columns == 0 {
            converged = Some(t - secs);
        }
        if frame % 30 == 0 {
            let (a, b, w) = blocky(flight);
            eprintln!(
                "CRUISE {stage} t {t:6.2} speed {speed:6.0} resident {} pending {} jobs {} units {:.0} budget {:.0} us/unit {:.3} plan {:.2} upload {:.2} blocky>2px {:.2}% >4px {:.2}% widest {w:.1}",
                stats.resident_columns, stats.pending_columns, stats.jobs, stats.units, stats.unit_budget, stats.us_per_unit, stats.plan_cpu_ms, stats.upload_cpu_ms, a * 100.0, b * 100.0
            );
        }
        if frame % capture_every == 0 {
            flight.capture(&format!("cruise_{:05}", (t * 100.0) as u32));
        }
        if moving && t + dt >= secs {
            flight.capture("cruise-arrival");
        }
        if converged.is_some() && t > secs + 2.0 {
            break;
        }
        t += dt;
        frame += 1;
    }
    flight.capture("cruise_end");
    flight.write_csv();
    eprintln!("CRUISE converged {converged:?} s after stopping");
}

/// Sustained heavy travel (HELIO_VOXEL_FLIGHT_LONG=<seconds>): editor speed
/// (10 m/s scaled by height over 20 m) times a boost, repeatedly diving from
/// orbit-like heights to tens of metres and climbing again while turning,
/// then a stop. Logs residency health (pool, table probe runs, queued
/// diffs) and enlarged blocks, and when the stopped view refines.
fn long_route(flight: &mut Flight, secs: f64) {
    let dt = 1.0 / 60.0;
    // (time fraction, height m, boost)
    let plan: [(f64, f64, f64); 13] = [
        (0.00, 40.0, 2.0), (0.10, 40.0, 3.0), (0.17, 3000.0, 2.0), (0.27, 3000.0, 2.0),
        (0.33, 60.0, 3.0), (0.47, 60.0, 3.0), (0.53, 25_000.0, 2.0), (0.63, 25_000.0, 2.0),
        (0.73, 30.0, 3.0), (0.80, 400.0, 3.0), (0.87, 30.0, 3.0), (0.93, 400.0, 3.0), (1.00, 30.0, 3.0),
    ];
    let at = |f: f64| {
        let i = plan.iter().rposition(|p| p.0 <= f).unwrap().min(plan.len() - 2);
        let (a, b) = (plan[i], plan[i + 1]);
        let u = ((f - a.0) / (b.0 - a.0)).clamp(0.0, 1.0);
        // Interpolate heights geometrically (climb rate follows height).
        ((a.1.ln() + (b.1.ln() - a.1.ln()) * u).exp(), a.2 + (b.2 - a.2) * u)
    };
    let start = DVec3::new(0.5f64.sin(), 0.5f64.cos(), 0.0);
    let ground = |flight: &Flight, p: DVec3| flight.planet.surface_point(p, 0.0).length();
    let mut eye = start * (ground(flight, start) + plan[0].1);
    let mut heading = DVec3::X;
    let view = |eye: DVec3, heading: DVec3, t: f64| {
        let up = eye.normalize();
        let ahead = (heading - up * heading.dot(up)).try_normalize().unwrap_or(DVec3::Z);
        let side = up.cross(ahead);
        // Sweep the view left and right while flying.
        let yaw = 0.6 * (t * 0.4).sin();
        let look = (ahead * yaw.cos() + side * yaw.sin() - up * 0.5).normalize();
        (up, ahead, look.as_vec3())
    };
    flight.settle("long_settle", eye, view(eye, heading, 0.0).2);
    let mut t = 0.0;
    let mut frame = 0usize;
    let mut refined = None;
    let log = |flight: &mut Flight, stage: &str, t: f64, height: f64, speed: f64| {
        let stats = flight.pass().stats().unwrap_or_default();
        let (longest, beyond, diffs) = flight.renderer.find_pass::<PlanetPass>().unwrap().renderer().unwrap().residency_health();
        let (a, b, w) = blocky(flight);
        eprintln!(
            "LONG {stage} t {t:6.1} h {height:8.0} v {speed:7.0} resident {} pending {} diffs {diffs} jobs {} units {:.0} budget {:.0} failed {} free_pages {}/{} free_units {:.0}% recycles {} lod_pressure {:.2} probe {longest} beyond64 {beyond} plan {:.2} late {} reranked {} blocky>2px {:.2}% >4px {:.2}% widest {w:.1}",
            stats.resident_columns, stats.pending_columns, stats.jobs, stats.units, stats.unit_budget, stats.failed_jobs, stats.free_pages, stats.pool_pages, stats.free_units as f64 / (f64::from(stats.pool_pages) * 512.0) * 100.0, stats.recycles, stats.lod_pressure, stats.plan_cpu_ms, stats.late_plans, stats.reranked, a * 100.0, b * 100.0
        );
        b
    };
    while t < secs + 40.0 {
        let moving = t < secs;
        let (height, boost) = at((t / secs).min(1.0));
        let clearance = eye.length() - ground(flight, eye);
        let speed = if moving { boost * 10.0 * (clearance / 20.0).max(1.0) } else { 0.0 };
        if moving {
            let (up, ahead, _) = view(eye, heading, t);
            // Turn slowly so new terrain keeps entering the view.
            let turn = 0.25 * (t * 0.07).sin();
            heading = (ahead + up.cross(ahead) * turn * dt).normalize();
            eye += ahead * speed * dt;
            // Follow the planned height at a bounded climb rate.
            let target = ground(flight, eye) + height;
            let r = eye.length();
            let rate = (target - r).clamp(-2.0 * clearance.max(20.0), 2.0 * clearance.max(20.0));
            eye = eye.normalize() * (r + rate * dt).max(ground(flight, eye) + 5.0);
        }
        let stage = if moving { "move" } else { "stop" };
        let (_, _, f) = view(eye, heading, t.min(secs));
        flight.draw(stage, eye, f);
        if frame % if moving { 60 } else { 30 } == 0 {
            let over4 = log(flight, stage, t, clearance, speed);
            if !moving && refined.is_none() && over4 < 0.005 {
                refined = Some(t - secs);
            }
        }
        if frame % 600 == 0 {
            flight.capture(&format!("long_{:05}", (t * 10.0) as u32));
        }
        if refined.is_some() && t > secs + 5.0 {
            break;
        }
        t += dt;
        frame += 1;
    }
    flight.capture("long_end");
    eprintln!("LONG refined {refined:?} s after stopping");
}

/// Enlarged blocks on screen: the share of terrain pixels of the last frame
/// drawn by a coarser-than-base level whose cell is wider than 2 and 4
/// pixels (by design such cells are at most ~2 px; wider ones mean a finer
/// level was not resident yet), and the widest.
fn blocky(flight: &mut Flight) -> (f64, f64, f64) {
    let (hits, size, voxel) = {
        let r = flight.renderer.find_pass::<PlanetPass>().unwrap().renderer().unwrap();
        (flight.read(r.hit_buffer()), r.screen_size(), flight.planet.grid().voxel_size())
    };
    let angle = 2.0 * (std::f64::consts::FRAC_PI_4 * 0.5).tan() / f64::from(size[1]);
    let (mut terrain, mut over2, mut over4, mut widest) = (0usize, 0usize, 0usize, 0.0f64);
    for hit in hits.chunks_exact(32).take((size[0] * size[1]) as usize) {
        let w = |i: usize| u32::from_le_bytes(hit[i * 4..i * 4 + 4].try_into().unwrap());
        let info = w(4);
        if info & 3 != 1 {
            continue;
        }
        let t = f64::from(f32::from_bits(w(0))).max(0.05);
        let level = (info >> 5) & 31;
        terrain += 1;
        // Base cells are never enlarged (near the eye they are simply close).
        if level == 0 {
            continue;
        }
        let px = voxel * f64::from(1u32 << level) / (t * angle);
        over2 += usize::from(px > 2.0);
        over4 += usize::from(px > 4.0);
        widest = widest.max(px);
    }
    let n = terrain.max(1) as f64;
    (over2 as f64 / n, over4 as f64 / n, widest)
}

/// One fixed view over a mountain flank (`km` from the nearest summit),
/// rendered with finer levels switched off progressively (`lod_pixels`
/// 1, 2, 4, 8, 16: each doubling moves every level one step coarser):
/// surface colours must not depend on the level that draws them.
fn lod_compare(flight: &mut Flight, km: f64) {
    // HELIO_VOXEL_FLIGHT_VIEWS_POLE: over the pole's hills instead (the
    // Pulsar example), at lower heights.
    let pole = std::env::var_os("HELIO_VOXEL_FLIGHT_VIEWS_POLE").is_some();
    let ground = if pole {
        DVec3::Y
    } else {
        let range = mountain(&flight.planet, 0.0);
        let away = tangent(range.peak, std::f64::consts::PI).as_dvec3();
        (range.peak + away * km * 1000.0).normalize()
    };
    let heights: &[f64] = if pole { &[60.0, 300.0] } else { &[300.0, 1500.0] };
    for &height in heights {
        let eye = flight.planet.surface_point(ground, height);
        let up = eye.normalize();
        let ahead = (DVec3::X - up * DVec3::X.dot(up)).normalize();
        let forward = (ahead - up * 1.2).normalize().as_vec3();
        for lod in [1.0f32, 0.5, 0.25, 0.125, 0.0625] {
            if let Some(r) = flight.pass().renderer_mut() {
                r.settings_mut().lod_pixels = lod;
            }
            let name = format!("lodcmp_{}m_{}", height as u32, (lod * 1000.0) as u32);
            flight.settle(&name, eye, forward);
            for _ in 0..20 {
                flight.draw(&name, eye, forward);
            }
            let pixels = flight.capture(&name);
            let (mut grey, mut green, mut n) = (0usize, 0usize, 0usize);
            for p in pixels.chunks(4) {
                let (r, g, b) = (i32::from(p[0]), i32::from(p[1]), i32::from(p[2]));
                n += 1;
                if g > r + 25 && g > b + 25 {
                    green += 1;
                } else if (r - g).abs() < 20 && (g - b).abs() < 20 && r < 230 {
                    grey += 1;
                }
            }
            let (a, b, w) = blocky(flight);
            eprintln!(
                "LODCMP h {height} lod_pixels {lod}: green {:.1}% grey {:.1}% | cells >2px {:.1}% >4px {:.1}% widest {w:.1}",
                green as f64 * 100.0 / n as f64, grey as f64 * 100.0 / n as f64, a * 100.0, b * 100.0
            );
        }
    }
}

/// Sculpting as the editor does it: stamps every 0.6 radius along a drag
/// (`stroke_fill`), several per frame, in laps of a ring around the aim
/// point, so edits pile up in one area as in a long session.
/// The ground point of the column nearest `ground` (rings of 64 cells, up to
/// 64 km) whose generated volume (caves, overhangs) reaches its surface.
/// The nearest column (on a 64-cell lattice) whose generated volume extent
/// (level cells below and above its top) passes `wanted`, 1.7 m above it.
fn volume_region_ground(planet: &Planet, ground: DVec3, what: &str, wanted: impl Fn(u8, i32, i32, i32, i32) -> bool) -> DVec3 {
    let grid = *planet.grid();
    let (base, _) = grid.locate(ground);
    for ring in 0..1000i32 {
        for step in 0..(8 * ring).max(1) {
            let side = step / (2 * ring).max(1);
            let along = step % (2 * ring).max(1) - ring;
            let (di, dj) = match side { 0 => (along, -ring), 1 => (ring, along), 2 => (-along, ring), _ => (-ring, -along) };
            let (i, j) = (base.i + di * 64, base.j + dj * 64);
            if !(0..grid.cells()).contains(&i) || !(0..grid.cells()).contains(&j) {
                continue;
            }
            let (below, above) = planet.field().extent(grid.domain_point(base.face, i, j, 0), 0);
            if wanted(base.face, i, j, below, above) {
                let cell = helio_pass_voxel_planet::Cell::new(base.face, i, j, planet.column_top(base.face, i, j, 0));
                eprintln!("VIEWS {what} region {} m away", grid.cell_center(cell).distance(ground).round());
                return planet.surface_point(grid.cell_center(cell), 1.7);
            }
        }
    }
    ground
}

fn sculpt_stress(flight: &mut Flight, ground: DVec3, heading: f64) {
    let up = ground.normalize();
    let eye = ground + up * 8.0;
    let forward = look(eye, heading, -40.0);
    flight.settle("sculpt_prepare", eye, forward);
    let target = flight.planet.raycast(eye, forward.as_dvec3(), 200.0).expect("aim at the ground").cell;
    let target = flight.planet.grid().cell_center(target);
    let side = tangent(target, heading).as_dvec3();
    let ahead = up.cross(side);
    let ring = 6.0;
    let strokes = [
        ("sculpt_dig_r1", 1.0, BrushShape::Sphere, BrushOp::Remove, 3usize, 240usize),
        ("sculpt_dig_r4", 4.0, BrushShape::Sphere, BrushOp::Remove, 2, 120),
        ("sculpt_build_r1", 1.0, BrushShape::Cube, BrushOp::Add, 3, 120),
    ];
    for (stage, radius, shape, op, per_frame, frames) in strokes {
        let spacing = radius * 0.6;
        let mut stamp = 0usize;
        let (mut wall, mut apply) = (Vec::new(), Vec::new());
        for _ in 0..frames {
            let started = Instant::now();
            let mut planet = (*flight.planet).clone();
            for _ in 0..per_frame {
                let angle = stamp as f64 * spacing / ring;
                let point = target + (side * angle.cos() + ahead * angle.sin()) * ring;
                let surface = flight.planet.surface_point(point, 0.0);
                let center = if op == BrushOp::Add { surface + up * radius } else { surface - up * radius * 0.3 };
                planet
                    .apply(Brush { center: center.to_array(), radius, shape, op, material: if op == BrushOp::Add { material::BRICK } else { 0 } })
                    .unwrap();
                stamp += 1;
            }
            flight.planet = Arc::new(planet);
            apply.push(started.elapsed().as_secs_f64() * 1000.0);
            flight.draw(stage, eye, forward);
            wall.push(started.elapsed().as_secs_f64() * 1000.0);
        }
        let settle_started = Instant::now();
        let (settle_frames, _) = flight.settle(&format!("{stage}_settle"), eye, forward);
        let settle_ms = settle_started.elapsed().as_secs_f64() * 1000.0;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
        let stats = flight.pass().stats().unwrap();
        eprintln!(
            "SCULPT {stage}: {} stamps (total edits {}), frame wall mean {:.2} p95 {:.2} ms, apply mean {:.3} ms, settle {settle_frames} frames {settle_ms:.0} ms, us/unit {:.2}",
            stamp,
            flight.planet.edits().len(),
            mean(&wall),
            percentile(&wall, 0.95),
            mean(&apply),
            stats.us_per_unit,
        );
    }
    flight.capture("sculpt");
}

/// HELIO_VOXEL_FLIGHT_VIEWS="height:pitch,...": settles and captures views
/// above the ground site (`view_<height>_<pitch>.png`), skipping the ground
/// audits.
fn capture_views(flight: &mut Flight, views: &str, ground: DVec3, heading: f64) {
    // HELIO_VOXEL_FLIGHT_VIEWS_CAVES=1: from the nearest cave region
    // (columns whose generated volume reaches the surface).
    let mut ground = if std::env::var_os("HELIO_VOXEL_FLIGHT_VIEWS_CAVES").is_some() {
        volume_region_ground(&flight.planet, ground, "cave", |_, _, _, below, _| below > 0)
    } else {
        ground
    };
    // HELIO_VOXEL_FLIGHT_VIEWS_OVERHANGS=1: from the nearest hillside (over
    // 20 degrees across 4 m) of an overhang region at full strength (rock
    // shelves, undercut ledges).
    if std::env::var_os("HELIO_VOXEL_FLIGHT_VIEWS_OVERHANGS").is_some() {
        let planet = flight.planet.clone();
        ground = volume_region_ground(&flight.planet, ground, "overhang hillside", |face, i, j, _, above| {
            let top = |di: i32, dj: i32| planet.column_top(face, i + di, j + dj, 0);
            above >= 50 && (top(20, 0) - top(-20, 0)).abs().max((top(0, 20) - top(0, -20)).abs()) >= 15
        });
    }
    // HELIO_VOXEL_FLIGHT_VIEWS_MOUNTAIN=<km>: on the flank of the nearest
    // high summit, that far from it, looking at it.
    let mut range = None;
    // HELIO_VOXEL_FLIGHT_VIEWS_POLE=<bearing rad>: at the north pole (the
    // Pulsar example's spawn), looking along that bearing (x towards z).
    let mut pole_bearing = None;
    if let Some(bearing) = std::env::var("HELIO_VOXEL_FLIGHT_VIEWS_POLE").ok().and_then(|v| v.parse::<f64>().ok()) {
        ground = flight.planet.surface_point(DVec3::Y, 0.0);
        pole_bearing = Some(bearing);
    }
    if let Some(km) = std::env::var("HELIO_VOXEL_FLIGHT_VIEWS_MOUNTAIN").ok().and_then(|v| v.parse::<f64>().ok()) {
        let r = mountain(&flight.planet, heading);
        let away = tangent(r.peak, heading + std::f64::consts::PI).as_dvec3();
        ground = flight.planet.surface_point(r.peak + away * km * 1000.0, 0.0);
        range = Some(r);
    }
    // HELIO_VOXEL_FLIGHT_VIEWS_AHEAD=<m>,<right m>: moves the ground site
    // along the view heading (non-pole views), to visit what a view showed.
    if let Ok(ahead) = std::env::var("HELIO_VOXEL_FLIGHT_VIEWS_AHEAD") {
        let v: Vec<f64> = ahead.split(',').map(|x| x.trim().parse().unwrap()).collect();
        let forward = tangent(ground, heading).as_dvec3();
        let right = tangent(ground, heading - std::f64::consts::FRAC_PI_2).as_dvec3();
        ground = flight.planet.surface_point(ground + forward * v[0] + right * v.get(1).copied().unwrap_or(0.0), 0.0);
    }
    // HELIO_VOXEL_FLIGHT_LOD_PIXELS=<f>: the renderer's lod_pixels (2: every
    // level one step finer), to compare a view across levels.
    if let Some(lod) = std::env::var("HELIO_VOXEL_FLIGHT_LOD_PIXELS").ok().and_then(|v| v.parse::<f32>().ok()) {
        if let Some(r) = flight.pass().renderer_mut() {
            r.settings_mut().lod_pixels = lod;
        }
    }
    for view in views.split(',') {
        let mut parts = view.split(':').map(|v| v.trim().parse::<f64>().unwrap());
        let (height, pitch) = (parts.next().unwrap(), parts.next().unwrap_or(-20.0));
        let eye = flight.planet.surface_point(ground, height);
        let forward = match &range {
            _ if pole_bearing.is_some() => {
                let (b, p) = (pole_bearing.unwrap(), pitch.to_radians());
                (DVec3::new(b.cos() * p.cos(), p.sin(), b.sin() * p.cos())).as_vec3().normalize()
            }
            Some(r) => {
                let p = (pitch as f32).to_radians();
                (level_toward(r, eye).as_vec3() * p.cos() + up_for(eye) * p.sin()).normalize()
            }
            None => look(eye, heading, pitch),
        };
        let name = format!("view_{height}_{}", -pitch);
        eprintln!("VIEWS {name}: eye {:.1} m from the centre, ground {:.1} m above the datum", eye.length(), eye.length() - height - flight.planet.grid().radius());
        flight.settle(&name, eye, forward);
        // HELIO_VOXEL_FLIGHT_VIEWS_HOLD=<frames>: a still camera that long
        // (an editor keeps rendering ~90 frames after the camera stops).
        let hold = std::env::var("HELIO_VOXEL_FLIGHT_VIEWS_HOLD").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
        for _ in 0..hold {
            flight.draw(&name, eye, forward);
        }
        flight.capture(&name);
        // HELIO_VOXEL_FLIGHT_VIEWS_FLY=<m/s>,<s>: then flies on along the view
        // at that height above the ground, capturing every second (what an
        // editor camera shows while refinement catches up).
        if let Ok(fly) = std::env::var("HELIO_VOXEL_FLIGHT_VIEWS_FLY") {
            let v: Vec<f64> = fly.split(',').map(|x| x.trim().parse().unwrap()).collect();
            let up = eye.normalize();
            let along = (forward.as_dvec3() - up * forward.as_dvec3().dot(up)).normalize();
            let frames = (v[1] / DT) as usize;
            for f in 1..=frames {
                let site = flight.planet.surface_point(eye + along * v[0] * f as f64 * DT, 0.0);
                let at = flight.planet.surface_point(site, height);
                flight.draw(&name, at, forward);
                if f % 60 == 0 {
                    flight.capture(&format!("{name}_fly_{}", f / 60));
                }
            }
        }
    }
    flight.write_csv();
}
