//! Full-engine voxel planet flight: movement sequences, synchronized frame
//! timings, terrain GPU stages, residency/memory, arrival and edit latency,
//! canonical CPU/GPU agreement audits and acceptance gates.
//!
//! cargo run -p helio-default-graphs --release --example voxel_flight -- OUTPUT [WIDTH HEIGHT [native|quality]]
//!
//! Environment:
//! * `HELIO_VOXEL_FLIGHT_RECORD=1` saves every other movement frame (visual
//!   review run; capture readback perturbs timings, so do not use it for gates).
//! * `HELIO_VOXEL_FLIGHT_VOXEL=0.3` authored base voxel size (0.1..1.0).
//! * `HELIO_VOXEL_FLIGHT_NO_SUN=1` disables traced terrain sunlight.
//!
//! `frames.csv` separates CPU submission from the synchronized GPU wait. Terrain
//! stage timestamps are read back for the frame that produced them.
//! `gates.json` / `gates.md` evaluate the declared acceptance targets.
use glam::{DVec3, Vec3};
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera, Renderer,
    RendererBuilder, RendererConfig,
};
use helio_default_graphs::{build_default_graph_external_with_voxel_passes, VoxelPassFactory};
use helio_pass_voxel_planet::engine::{PlanetFrame, PlanetPass, SharedPlanetFrame};
use helio_pass_voxel_planet::{field, Brush, BrushOp, BrushShape, Planet, PlanetRecipe};
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
    sync_ms: f64,
    terrain_gpu_ms: f64,
    stages: BTreeMap<&'static str, f64>,
}

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
}

fn up_for(eye: DVec3) -> Vec3 {
    eye.normalize().as_vec3()
}

/// Horizontal heading at `eye` (0 = local east).
fn tangent(eye: DVec3, heading: f64) -> Vec3 {
    let up = eye.normalize();
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
        helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(&mut store, 16, &device);
        helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(&mut store, 16, &device);
        helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(&mut store, 16, &device);
        helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(&mut store, 16, &device);
        let mirror = GpuMirrorHandle::new(Arc::new(store), queue.clone());
        let mut scene = pulsar_scenedb::SceneDb::new();
        scene.world.attach_gpu_mirror(mirror.clone());
        let sun_dir = Vec3::new(0.35, 0.75, 0.45).normalize();
        let light = scene.world.spawn();
        scene.world.insert(
            light,
            helio_pass_forward_lit::LightComponent::from(helio::GpuLight {
                position_range: [0.0, 0.0, 0.0, f32::MAX],
                direction_outer: [-sun_dir.x, -sun_dir.y, -sun_dir.z, 0.0],
                color_intensity: [1.0, 0.94, 0.82, 3.0],
                shadow_index: u32::MAX,
                light_type: helio::LightType::Directional as u32,
                ..Default::default()
            }),
        );
        scene.world.flush_gpu_mirror(&queue);
        // The mirror owns the uploaded rows for the lifetime of the flight.
        std::mem::forget(scene);
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
            .with_ambient([0.6, 0.72, 0.95], 1.4)
            .with_external_device()
            .with_pass_build_context(Box::new(move |ctx| build_default_graph_external_with_voxel_passes(ctx, vec![factory.clone()])))
            .build(device.clone(), queue.clone(), size[0], size[1], config.surface_format);
        renderer.set_fallback_sky_enabled(true);
        if let Some(mode) = std::env::var("HELIO_VOXEL_FLIGHT_DEBUG").ok().and_then(|v| v.parse().ok()) {
            renderer.set_debug_mode(mode);
        }
        let target = Self::make_target(&device, size);
        let mut csv = std::fs::File::create(output.join("frames.csv")).unwrap();
        writeln!(csv, "frame,stage,altitude_m,sync_ms,cpu_submit_ms,gpu_wait_ms,terrain_gpu_ms,residency_ms,primary_ms,shade_ms,gbuffer_ms,sunlight_ms,resident,pending,jobs,evictions,failed,active_levels,finest_level,plan_cpu_ms,upload_cpu_ms,encode_cpu_ms,logical_mib").unwrap();
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
        *self.source.lock().unwrap() = Some(PlanetFrame { eye, planet: self.planet.clone(), sun: self.sun, shadows: self.shadows });
        self.renderer.set_world_origin(Some(eye));
        let up = up_for(eye);
        // Hemisphere fill around the local vertical with a sunlit-grass bounce.
        self.renderer.set_ambient_hemisphere(up.to_array(), Some([0.3, 0.34, 0.2]));
        let forward = forward.normalize();
        let up = if forward.dot(up).abs() > 0.999 { up.any_orthonormal_vector() } else { up };
        let near = (self.planet.air_clearance(eye) * 0.25).clamp(0.05, 50_000.0) as f32;
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        let camera = Camera::perspective_look_at(Vec3::ZERO, forward, up, std::f32::consts::FRAC_PI_4, aspect, near, 40_000_000.0);
        let start = Instant::now();
        self.renderer.render(&camera, &self.target.create_view(&Default::default())).unwrap();
        let submit = start.elapsed().as_secs_f64() * 1000.0;
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let sync = start.elapsed().as_secs_f64() * 1000.0;
        let timings = self.pass().renderer_mut().map(|r| r.stage_timings_blocking()).unwrap_or_default();
        let stats = self.pass().stats().unwrap_or_default();
        let mut stages = BTreeMap::new();
        for (name, ms) in timings {
            *stages.entry(name).or_insert(0.0) += ms;
        }
        let terrain: f64 = stages.values().sum();
        let get = |n: &str| stages.get(n).copied().unwrap_or(0.0);
        let altitude = eye.length() - self.planet.grid().radius();
        writeln!(
            self.csv,
            "{},{stage},{altitude:.3},{sync:.4},{submit:.4},{:.4},{terrain:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.1}",
            self.frame,
            sync - submit,
            get("planet_residency"),
            get("planet_primary"),
            get("planet_shade"),
            get("planet_gbuffer"),
            get("planet_sunlight"),
            stats.resident_columns,
            stats.pending_columns,
            stats.jobs,
            stats.evictions,
            stats.failed_jobs,
            stats.active_levels,
            stats.finest_level,
            stats.plan_cpu_ms,
            stats.upload_cpu_ms,
            stats.encode_cpu_ms,
            stats.logical_bytes as f64 / 1048576.0
        )
        .unwrap();
        self.samples.push(Sample { stage: stage.to_string(), sync_ms: sync, terrain_gpu_ms: terrain, stages });
        self.frame += 1;
        sync
    }

    /// Draw until residency is complete; returns (frames, synchronized ms).
    fn settle(&mut self, stage: &str, eye: DVec3, forward: Vec3) -> (usize, f64) {
        let mut ms = 0.0;
        for frames in 1..=3000 {
            ms += self.draw(stage, eye, forward);
            if self.pass().renderer().is_some_and(|r| r.settled()) {
                return (frames, ms);
            }
        }
        panic!("{stage}: residency did not settle");
    }

    fn read(&self, buffer: &wgpu::Buffer) -> Vec<u8> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        self.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = staging.slice(..).get_mapped_range().unwrap().to_vec();
        data
    }

    fn capture(&self, name: &str) -> Vec<u8> {
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
        let mut counts = [0usize; 4];
        let up0 = up_for(eye);
        let forward = forward.normalize();
        let up = if forward.dot(up0).abs() > 0.999 { up0.any_orthonormal_vector() } else { up0 };
        let right = forward.cross(up).normalize();
        let cam_up = right.cross(forward);
        let tan = (std::f32::consts::FRAC_PI_4 * 0.5).tan();
        let aspect = size[0] as f32 / size[1] as f32;
        let (mut compared, mut mismatched) = (0usize, 0usize);
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
                let t = f64::from(f32::from_bits(w(0)));
                let same = (info >> 2) & 7 == u32::from(cpu.cell.face)
                    && w(1) as i32 == cpu.cell.i
                    && w(2) as i32 == cpu.cell.j
                    && w(3) as i32 == cpu.cell.k;
                // TAA jitter moves the GPU sample by up to half a pixel.
                if !same && (t - cpu.distance).abs() > self.planet.grid().voxel_size() * 3.0 {
                    mismatched += 1;
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
        serde_json::json!({"name": name, "stuck": stuck, "work": stats, "miss": counts[0], "hit": counts[1], "exhausted": counts[2], "loading": counts[3], "compared": compared, "mismatched": mismatched})
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
    let planet = Planet::new(PlanetRecipe { voxel_size_m: voxel, ..Default::default() }).unwrap();
    let mut flight = Flight::new(&output, size, quality, planet);
    let validation = flight.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut report = serde_json::Map::new();
    let mut audits = Vec::new();
    report.insert("config".into(), serde_json::json!({"size": size, "quality": format!("{quality:?}"), "voxel_m": flight.planet.grid().voxel_size(), "shadows": flight.shadows, "record": flight.record}));

    // Ground spawn on the +Y face (the fallback sky assumes +Y up).
    let dir = land_near(&flight.planet, 2, 0.47, 0.53, 20.0);
    let ground = flight.planet.surface_point(dir, 1.7);
    let heading = 0.6;
    let forward = look(ground, heading, -12.0);
    let (frames, ms) = flight.settle("ground_load", ground, forward);
    eprintln!("VOXEL_FLIGHT ground load {frames} frames {ms:.1} ms");
    report.insert("cold_ground_load".into(), serde_json::json!({"frames": frames, "sync_ms": ms}));
    for _ in 0..60 {
        flight.draw("ground_warm", ground, forward);
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

    if std::env::var_os("HELIO_VOXEL_FLIGHT_GROUND_ONLY").is_some() {
        for a in &audits {
            eprintln!("GROUND audit {a}");
        }
        let error = pollster::block_on(validation.pop());
        assert!(error.is_none(), "GPU validation: {error:?}");
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
            let terrain: Vec<f64> = list.iter().map(|s| s.terrain_gpu_ms).collect();
            let stage = |k: &str| percentile(&list.iter().map(|s| s.stages.get(k).copied().unwrap_or(0.0)).collect::<Vec<_>>(), 0.5);
            eprintln!(
                "QUICK {name:16} n={:4} sync p50 {:7.2} p95 {:7.2} terrain p50 {:6.2} p95 {:6.2} | primary {:6.2} shade {:5.2} sun {:6.2} residency {:5.2}",
                list.len(), percentile(&sync, 0.5), percentile(&sync, 0.95), percentile(&terrain, 0.5), percentile(&terrain, 0.95),
                stage("planet_primary"), stage("planet_shade"), stage("planet_sunlight"), stage("planet_residency")
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
            material: if add { field::material::COBBLE } else { 0 },
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
        let (mut frames, mut ms) = (0, 0.0);
        loop {
            ms += flight.draw("dig", base, aim);
            frames += 1;
            let hit = {
                let r = flight.renderer.find_pass::<PlanetPass>().unwrap().renderer().unwrap();
                let size = r.screen_size();
                let at = (size[1] / 2 * size[0] + size[0] / 2) as usize * 32;
                flight.read(r.hit_buffer())[at..at + 32].to_vec()
            };
            let w = |i: usize| u32::from_le_bytes(hit[i * 4..i * 4 + 4].try_into().unwrap());
            let got = (w(1) as i32, w(2) as i32, w(3) as i32);
            let matched = expected.is_some_and(|c| (c.i, c.j, c.k) == got);
            if matched || frames >= 30 {
                latencies.push((frames, ms, matched));
                break;
            }
        }
    }
    flight.renderer.set_jitter_enabled(true);
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
        let terrain: Vec<f64> = list.iter().map(|s| s.terrain_gpu_ms).collect();
        let mut stage_p95 = serde_json::Map::new();
        for key in ["planet_residency", "planet_primary", "planet_shade", "planet_gbuffer", "planet_sunlight"] {
            let v: Vec<f64> = list.iter().map(|s| s.stages.get(key).copied().unwrap_or(0.0)).collect();
            stage_p95.insert(key.into(), serde_json::json!(percentile(&v, 0.95)));
        }
        stages.insert(
            name.clone(),
            serde_json::json!({
                "frames": list.len(),
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
            .collect()
    };
    let warm = gather(&["ground_warm", "orbit", "arrival_settled"], false);
    let movement_names = ["walk", "run", "vehicle", "rotate", "ascent", "descent", "reversal"];
    let moving = gather(&movement_names, false);
    let mut terrain_names = movement_names.to_vec();
    terrain_names.extend(["ground_warm", "orbit"]);
    let terrain = gather(&terrain_names, true);
    let bad_rays: u64 = audits.iter().map(|a| a["exhausted"].as_u64().unwrap() + a["loading"].as_u64().unwrap()).sum();
    let mismatched: u64 = audits.iter().map(|a| a["mismatched"].as_u64().unwrap()).sum();
    let compared: u64 = audits.iter().map(|a| a["compared"].as_u64().unwrap()).sum();
    let edit_max = report["edits"]["max_ms"].as_f64().unwrap_or(f64::INFINITY);
    let arrival = report["arrival"]["sync_ms_to_settle"].as_f64().unwrap();
    let gates = serde_json::json!([
        {"gate": "warm full-graph sync p95 <= 16.67 ms", "value": percentile(&warm, 0.95), "pass": percentile(&warm, 0.95) <= 16.67},
        {"gate": "movement sync p99 <= 25 ms", "value": percentile(&moving, 0.99), "pass": percentile(&moving, 0.99) <= 25.0},
        {"gate": "terrain GPU p95 <= 5 ms (movement + warm)", "value": percentile(&terrain, 0.95), "pass": percentile(&terrain, 0.95) <= 5.0},
        {"gate": "logical terrain GPU memory <= 1024 MiB", "value": stats.logical_bytes as f64 / 1048576.0, "pass": stats.logical_bytes <= 1 << 30},
        {"gate": "arrival settles <= 250 ms after descent", "value": arrival, "pass": arrival <= 250.0},
        {"gate": "visible local edit <= 100 ms", "value": edit_max, "pass": edit_max <= 100.0 && unmatched == 0},
        {"gate": "no exhausted/loading rays in settled audits", "value": bad_rays, "pass": bad_rays == 0},
        {"gate": "near-field CPU/GPU cell agreement", "value": format!("{mismatched}/{compared}"), "pass": compared > 0 && mismatched * 1000 <= compared},
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
    flight.csv.flush().unwrap();
    eprintln!("VOXEL_FLIGHT_COMPLETE frames={}", flight.frame);
}
