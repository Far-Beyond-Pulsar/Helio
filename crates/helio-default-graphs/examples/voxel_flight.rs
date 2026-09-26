//! Deterministic full-graph terrain captures and synchronized frame timings.
//! cargo run -p helio-default-graphs --release --example voxel_flight -- OUTPUT [WIDTH HEIGHT]
//! Captures are actual render output. CSV times include CPU submission and GPU
//! completion, exclude readback/PNG encoding, and do not include presentation.
use glam::{DVec3, Vec3};
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera, Renderer,
    RendererBuilder, RendererConfig,
};
use helio_default_graphs::{build_default_graph_external_with_voxel_passes, VoxelPassFactory};
use helio_pass_tiny_voxel::{
    engine::{EngineVoxelFrame, LazyEngineVoxelPass, SharedVoxelFrame},
    world::render_origin,
    Params, World,
};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};
use std::{
    fs,
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

struct Flight {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: Renderer,
    source: SharedVoxelFrame,
    world: Arc<World>,
    target: wgpu::Texture,
    size: [u32; 2],
    frame: usize,
    csv: fs::File,
}
impl Flight {
    async fn new(output: &Path, size: [u32; 2]) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("GPU required");
        eprintln!("VOXEL_FLIGHT_ADAPTER {:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: required_wgpu_features(adapter.features()),
                required_limits: required_wgpu_limits(adapter.limits()),
                experimental_features: required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let context = EngineGpuContext::new(device.clone(), queue.clone());
        let mut store = SceneGpuStore::new(
            &context,
            SceneGpuConfig {
                classes: vec![],
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        );
        helio_pass_sky::SkyComponent::register_gpu_columns_growable(&mut store, 4, &device);
        helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(&mut store, 16, &device);
        helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(
            &mut store, 16, &device,
        );
        helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(
            &mut store, 16, &device,
        );
        helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(
            &mut store, 16, &device,
        );
        let mirror = GpuMirrorHandle::new(Arc::new(store), queue.clone());
        let mut scene = pulsar_scenedb::SceneDb::new();
        scene.world.attach_gpu_mirror(mirror.clone());
        let sun = scene.world.spawn();
        scene.world.insert(
            sun,
            helio_pass_forward_lit::LightComponent::from(helio::GpuLight {
                position_range: [0.0, 0.0, 0.0, f32::MAX],
                direction_outer: [-0.4, -0.8, -0.3, 0.0],
                color_intensity: [1.0, 0.96, 0.88, 3.0],
                shadow_index: u32::MAX,
                light_type: helio::LightType::Directional as u32,
                ..Default::default()
            }),
        );
        scene.world.flush_gpu_mirror(&queue);
        let source: SharedVoxelFrame = Arc::new(Mutex::new(None));
        let pass_source = source.clone();
        let factory: VoxelPassFactory =
            Arc::new(move |_, _, _, _| Box::new(LazyEngineVoxelPass::new(pass_source.clone())));
        let mut config = RendererConfig::new(size[0], size[1], wgpu::TextureFormat::Rgba8Unorm)
            .with_tsr_quality(helio_pass_tsr::TsrQuality::Native);
        config.enable_foliage = false;
        let mut renderer = RendererBuilder::new(config, mirror)
            .with_ambient([0.5, 0.5, 0.6], 1.0)
            .with_external_device()
            .with_pass_build_context(Box::new(move |ctx| {
                build_default_graph_external_with_voxel_passes(ctx, vec![factory])
            }))
            .build(
                device.clone(),
                queue.clone(),
                size[0],
                size[1],
                config.surface_format,
            );
        renderer.set_fallback_sky_enabled(true);
        let target = Self::target(&device, size);
        let mut csv = fs::File::create(output.join("frames.csv")).unwrap();
        writeln!(csv, "frame,stage,x,y,z,sync_frame_ms,ready,refining,planning,pending,generated,reused,bricks,pixel_budget").unwrap();
        Self {
            device,
            queue,
            renderer,
            source,
            world: Arc::new(World::default()),
            target,
            size,
            frame: 0,
            csv,
        }
    }
    fn target(device: &wgpu::Device, size: [u32; 2]) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("voxel flight output"),
            size: wgpu::Extent3d {
                width: size[0],
                height: size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }
    fn draw(&mut self, stage: &str, eye: DVec3, forward: Vec3) -> f64 {
        let validation = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let start = Instant::now();
        let forward = forward.normalize();
        let right = forward.cross(Vec3::Y).normalize();
        let up = right.cross(forward);
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        let origin = render_origin(eye);
        let fraction = std::array::from_fn(|a| {
            if a < 3 {
                (eye[a] / 0.1 - f64::from(origin[a])) as f32
            } else {
                0.0
            }
        });
        *self.source.lock().unwrap() = Some(EngineVoxelFrame {
            params: Params {
                origin: [origin[0], origin[1], origin[2], 0],
                fraction,
                radial: [0.0, 1.0, 0.0, 0.0],
                right: [right.x, right.y, right.z, aspect],
                up: [up.x, up.y, up.z, 0.41421356],
                forward: [forward.x, forward.y, forward.z, 0.0],
                screen: [self.size[0] as f32, self.size[1] as f32, 0.0, 0.0],
                lighting: [0.4, 0.8, 0.3, 0.0],
                settings: [30_000_000.0, 0.0, 1.0, 0.0],
            },
            world: self.world.clone(),
            raytraced_sun: false,
        });
        // Camera matrices never contain Earth-sized f32 translations.
        self.renderer.set_world_origin(Some(eye));
        let near = (self.world.air_clearance(eye) * 0.25).max(0.05) as f32;
        let camera = Camera::perspective_look_at(
            Vec3::ZERO,
            forward,
            up,
            std::f32::consts::FRAC_PI_4,
            aspect,
            near,
            30_000_000.0,
        );
        self.renderer
            .render(&camera, &self.target.create_view(&Default::default()))
            .unwrap();
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        let error = pollster::block_on(validation.pop());
        assert!(
            error.is_none(),
            "frame {} ({stage}) GPU validation: {error:?}",
            self.frame
        );
        let stats = self
            .renderer
            .find_pass::<LazyEngineVoxelPass>()
            .unwrap()
            .stats()
            .unwrap();
        if stage != "ground_load" {
            assert!(stats.ready, "{stage}: terrain disappeared during movement");
        }
        writeln!(
            self.csv,
            "{},{},{:.6},{:.6},{:.6},{:.4},{},{},{},{},{},{},{},{:.4}",
            self.frame,
            stage,
            eye.x,
            eye.y,
            eye.z,
            ms,
            stats.ready,
            stats.refining,
            stats.planning,
            stats.pending,
            stats.generated,
            stats.reused,
            stats.bricks,
            stats.pixel_budget
        )
        .unwrap();
        self.frame += 1;
        ms
    }
    fn settle(&mut self, stage: &str, eye: DVec3, direction: Vec3) {
        let start = Instant::now();
        loop {
            self.draw(stage, eye, direction);
            if !self
                .renderer
                .find_pass::<LazyEngineVoxelPass>()
                .unwrap()
                .needs_frame()
            {
                break;
            }
            assert!(
                start.elapsed() < Duration::from_secs(120),
                "{stage}: residency timeout"
            );
        }
        eprintln!(
            "VOXEL_FLIGHT_SETTLED stage={stage} load_ms={:.2}",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
    fn capture(&self, path: &Path) -> Vec<u8> {
        let row = (self.size[0] * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel flight readback"),
            size: u64::from(row) * u64::from(self.size[1]),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let hits = self
            .renderer
            .find_pass::<LazyEngineVoxelPass>()
            .unwrap()
            .primary_hit_buffer()
            .unwrap();
        let hit_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel flight hit audit"),
            size: hits.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(hits, 0, &hit_buffer, 0, hits.size());
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(self.size[1]),
                },
            },
            self.target.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        hit_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let hit_data = hit_buffer.slice(..).get_mapped_range().unwrap();
        let mut counts = [0usize; 4];
        for hit in hit_data.chunks_exact(32) {
            counts[(u32::from_le_bytes(hit[12..16].try_into().unwrap()) & 3) as usize] += 1;
        }
        fs::write(
            path.with_extension("hits.csv"),
            format!(
                "empty,solid,exhausted,loading\n{},{},{},{}\n",
                counts[0], counts[1], counts[2], counts[3]
            ),
        )
        .unwrap();
        assert_eq!(
            counts[2],
            0,
            "{}: voxel traversal exhausted",
            path.display()
        );
        if path.file_stem().unwrap() != "composition-sentinel" {
            assert!(
                counts[1] > 0 && counts[3] == 0,
                "{}: terrain missing: {counts:?}",
                path.display()
            );
        }
        let data = buffer.slice(..).get_mapped_range().unwrap();
        let pixels: Vec<u8> = data
            .chunks(row as usize)
            .flat_map(|r| r[..self.size[0] as usize * 4].iter().copied())
            .collect();
        image::save_buffer(
            path,
            &pixels,
            self.size[0],
            self.size[1],
            image::ColorType::Rgba8,
        )
        .unwrap();
        pixels
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let output = Path::new(args.get(1).expect("OUTPUT directory required"));
    fs::create_dir_all(output).unwrap();
    let size = [
        args.get(2).map_or(1280, |s| s.parse().unwrap()),
        args.get(3).map_or(720, |s| s.parse().unwrap()),
    ];
    let mut flight = pollster::block_on(Flight::new(output, size));
    let validation = flight
        .device
        .push_error_scope(wgpu::ErrorFilter::Validation);
    let ground = flight.world.ground_spawn(0.0, 0.0, 3.0);
    let forward = Vec3::new(0.0, -0.15, -1.0);
    // A known final-stage effect must survive DOF on the very first frame.
    // This catches the real composition regression, even when every isolated
    // TSR/postprocess shader and GPU validation test passes.
    flight
        .renderer
        .find_pass_mut::<helio_pass_postprocess::PostProcessPass>()
        .unwrap()
        .set_user_shader(Some("vec3<f32>(1.0, 0.0, 1.0)"));
    flight.draw("ground_load", ground, forward);
    let sentinel = flight.capture(&output.join("composition-sentinel.png"));
    assert!(
        sentinel
            .chunks_exact(4)
            .all(|p| p[0] >= 250 && p[1] <= 5 && p[2] >= 250),
        "the final postprocess result did not survive full graph composition"
    );
    flight
        .renderer
        .find_pass_mut::<helio_pass_postprocess::PostProcessPass>()
        .unwrap()
        .clear_user_effects(&flight.device);
    flight.settle("ground_load", ground, forward);
    for i in 0..120 {
        let eye = ground + DVec3::new(i as f64 * 0.04, 0.0, -i as f64 * 0.03);
        flight.draw("walk", eye, forward);
        if i % 30 == 0 {
            flight.capture(&output.join(format!("walk-{i:03}.png")));
        }
    }
    for (name, altitude) in [("200m", 200.0), ("1km", 1_000.0), ("orbit", 300_000.0)] {
        let eye = ground + DVec3::Y * altitude;
        let look = Vec3::new(0.0, -0.8, -1.0);
        flight.settle(name, eye, look);
        let mut times = Vec::new();
        for _ in 0..60 {
            times.push(flight.draw(name, eye, look));
        }
        times.sort_by(f64::total_cmp);
        flight.capture(&output.join(format!("{name}.png")));
        eprintln!(
            "VOXEL_FLIGHT_STEADY stage={name} p50_ms={:.3} p95_ms={:.3} max_ms={:.3}",
            times[30], times[57], times[59]
        );
    }
    // Continuous descent: do not settle between frames or hide arrival/loading.
    for i in 0..240 {
        let altitude = 300_000.0_f64.powf(1.0 - i as f64 / 239.0) - 1.0;
        flight.draw(
            "descent",
            ground + DVec3::Y * altitude,
            Vec3::new(0.0, -0.8, -1.0),
        );
        if i % 30 == 0 || i == 239 {
            flight.capture(&output.join(format!("descent-{i:03}.png")));
        }
    }
    flight.settle("returned_ground", ground, forward);
    flight.capture(&output.join("returned-ground.png"));
    // Rebuild the graph at a different aspect and assert the resident cut survives.
    let before = flight
        .renderer
        .find_pass::<LazyEngineVoxelPass>()
        .unwrap()
        .stats()
        .unwrap();
    flight.size = [size[0] + 64, size[1] + 36];
    flight.target = Flight::target(&flight.device, flight.size);
    flight
        .renderer
        .set_render_size(flight.size[0], flight.size[1]);
    flight.draw("resize", ground, forward);
    let after = flight
        .renderer
        .find_pass::<LazyEngineVoxelPass>()
        .unwrap()
        .stats()
        .unwrap();
    assert!(
        after.ready && after.generated >= before.generated,
        "resize discarded resident terrain"
    );
    flight.settle("resize_settle", ground, forward);
    flight.capture(&output.join("resized.png"));
    // Destroy a target from orbit, then inspect its local geometry through the
    // same full graph. Editing is not clipped to the camera draw distance.
    let target_eye = ground + DVec3::new(0.0, 300_000.0, -8.0);
    let (cell, _, distance) = flight
        .world
        .raycast(target_eye, -DVec3::Y, f64::INFINITY)
        .unwrap();
    assert!(distance > 290_000.0);
    let mut edited = (*flight.world).clone();
    edited
        .apply_edit(helio_pass_tiny_voxel::world::Edit {
            cell,
            radius: 4.0,
            material: 0,
        })
        .unwrap();
    assert_eq!(edited.material(cell), 0);
    flight.world = Arc::new(edited);
    let inspect = Vec3::new(0.0, -0.4, -1.0);
    flight.settle("orbital_edit", ground, inspect);
    for _ in 0..30 {
        flight.draw("orbital_edit", ground, inspect);
    }
    flight.capture(&output.join("orbital-edit.png"));
    let mut coarse = (*flight.world).clone();
    coarse.set_voxel_size(1.0).unwrap();
    flight.world = Arc::new(coarse);
    flight.settle("1m_base", ground, inspect);
    for _ in 0..30 {
        flight.draw("1m_base", ground, inspect);
    }
    flight.capture(&output.join("1m-base.png"));
    let error = pollster::block_on(validation.pop());
    assert!(error.is_none(), "GPU validation errors: {error:?}");
    flight.csv.flush().unwrap();
    eprintln!("VOXEL_FLIGHT_COMPLETE frames={}", flight.frame);
}
