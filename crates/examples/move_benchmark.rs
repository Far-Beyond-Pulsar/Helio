//! Headless object-move benchmark.
//!
//! Measures what one moving object costs per frame as the scene grows. Moving a
//! single object should cost the same in a 1k-object scene as in a 64k-object
//! scene; any stage whose time grows with scene size is a scaling bug.
//!
//! Each configuration builds a lit scene, warms the renderer up, then times
//! three workloads over the same frames:
//!
//! * `idle`          -- nothing changes (baseline).
//! * `move_one`      -- one object is moved through `World::get_mut`, the
//!                      per-row path SceneDB mirrors directly.
//! * `editor_resync` -- one object is moved and then every object row is
//!                      re-inserted, which is what Pulsar-Native's
//!                      `sync_static_mesh_rows` does on every frame the scene
//!                      revision changes (i.e. every frame of a gizmo drag).
//!
//! Stages timed per frame (milliseconds, wall clock):
//!
//! * `update`  -- CPU SceneDB mutation.
//! * `flush`   -- `World::flush_gpu_mirror`.
//! * `rt`      -- `SceneDbRayTracing::prepare` (RT mode only).
//! * `rt_gpu`  -- blocking wait for the acceleration-structure build it
//!                submitted (RT mode only).
//! * `render`  -- `Renderer::render` CPU recording and submission.
//! * `gpu`     -- blocking wait for the frame to finish on the device.
//!
//! Usage (all flags optional):
//!
//! ```text
//! move_benchmark --scene grid --sizes 1000,4000,16000 --lights 64 \
//!                --modes ss,rt --frames 40 --warmup 8 --out bench_out
//! move_benchmark --scene cathedral_large --modes ss,rt
//! move_benchmark --mesh-movability static     # tag every mesh entity (none|static|stationary|movable|dynamic)
//! move_benchmark --graph default --editor   # Pulsar-Native editor viewport graph
//!                                            # (add --no-ray-query on lavapipe)
//! ```
//!
//! No window or surface is created. On a machine without a GPU, Mesa's
//! lavapipe works: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`.
//! Results are printed as a Markdown table and written as CSV to `--out`.

#![allow(dead_code)]

mod architectural_mesh;
mod cathedral_large;
mod hlfs_capture;
mod v3_demo_common;

use glam::{Mat4, Quat, Vec3};
use helio::{Camera, MeshUpload, PackedVertex, Renderer, RendererBuilder, RendererConfig};
use helio_pass_gbuffer::StaticObjectComponent;
use pulsar_scenedb::{Entity, SceneDb, World};
use std::sync::Arc;
use std::time::Instant;
use v3_demo_common::{
    make_material, new_scene_db_with_gpu_mirror, point_light, scene_db_handle, spawn_light,
    spawn_material, spawn_mesh, spawn_object,
};

// `cathedral_large` reads these from its parent module.
const LARGE_CHANDELIER_Z: &[f32] = &[-54.0, -36.0, -18.0, 0.0, 18.0, 36.0, 54.0];
const LARGE_CANDLES: &[(f32, f32, f32)] = &[
    (-4.0, 1.6, -64.0), (-2.0, 1.6, -63.5), (0.0, 1.6, -64.0),
    (2.0, 1.6, -63.5), (4.0, 1.6, -64.0),
];

const WIDTH: u32 = 320;
const HEIGHT: u32 = 180;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    ScreenSpace,
    RayTraced,
}

impl Mode {
    fn label(self) -> &'static str {
        match self {
            Mode::ScreenSpace => "ss",
            Mode::RayTraced => "rt",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Workload {
    Idle,
    MoveOne,
    EditorResync,
}

impl Workload {
    const ALL: [Workload; 3] = [Workload::Idle, Workload::MoveOne, Workload::EditorResync];
    fn label(self) -> &'static str {
        match self {
            Workload::Idle => "idle",
            Workload::MoveOne => "move_one",
            Workload::EditorResync => "editor_resync",
        }
    }
}

struct Args {
    scene: String,
    sizes: Vec<usize>,
    lights: usize,
    unique_meshes: usize,
    sphere_segments: u32,
    modes: Vec<Mode>,
    workloads: Vec<Workload>,
    frames: usize,
    warmup: usize,
    out: String,
    graph: String,
    editor: bool,
    no_ray_query: bool,
    mesh_movability: Option<helio::Movability>,
}

fn parse_args() -> Args {
    let mut args = Args {
        scene: "grid".into(),
        sizes: vec![1_000, 4_000, 16_000],
        lights: 64,
        unique_meshes: 64,
        sphere_segments: 12,
        modes: vec![Mode::ScreenSpace, Mode::RayTraced],
        workloads: Workload::ALL.to_vec(),
        frames: 40,
        warmup: 8,
        out: "move_benchmark_out".into(),
        graph: "hlfs".into(),
        editor: false,
        no_ray_query: false,
        mesh_movability: None,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let value = raw.get(i + 1).cloned().unwrap_or_default();
        match raw[i].as_str() {
            "--scene" => args.scene = value,
            "--sizes" => {
                args.sizes = value
                    .split(',')
                    .map(|s| s.trim().parse().expect("--sizes takes integers"))
                    .collect()
            }
            "--lights" => args.lights = value.parse().expect("--lights takes an integer"),
            "--unique-meshes" => {
                args.unique_meshes = value.parse().expect("--unique-meshes takes an integer")
            }
            "--segments" => {
                args.sphere_segments = value.parse().expect("--segments takes an integer")
            }
            "--modes" => {
                args.modes = value
                    .split(',')
                    .map(|s| match s.trim() {
                        "ss" => Mode::ScreenSpace,
                        "rt" => Mode::RayTraced,
                        other => panic!("unknown mode {other}; use ss or rt"),
                    })
                    .collect()
            }
            "--workloads" => {
                args.workloads = value
                    .split(',')
                    .map(|s| {
                        *Workload::ALL
                            .iter()
                            .find(|w| w.label() == s.trim())
                            .unwrap_or_else(|| panic!("unknown workload {s}"))
                    })
                    .collect()
            }
            "--frames" => args.frames = value.parse().expect("--frames takes an integer"),
            "--warmup" => args.warmup = value.parse().expect("--warmup takes an integer"),
            "--out" => args.out = value,
            "--graph" => args.graph = value,
            "--mesh-movability" => {
                args.mesh_movability = match value.as_str() {
                    "none" => None,
                    "static" => Some(helio::Movability::Static),
                    "stationary" => Some(helio::Movability::Stationary),
                    "movable" => Some(helio::Movability::Movable),
                    "dynamic" => Some(helio::Movability::Dynamic),
                    other => panic!("unknown movability {other}"),
                }
            }
            "--editor" => {
                args.editor = true;
                i += 1;
                continue;
            }
            "--no-ray-query" => {
                args.no_ray_query = true;
                i += 1;
                continue;
            }
            "--help" | "-h" => {
                eprintln!("see the module doc at the top of move_benchmark.rs");
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    assert!(args.frames > 0, "--frames must be positive");
    args
}

// ── Scene construction ────────────────────────────────────────────────────────

/// Built scene plus the handles the workloads need.
struct BenchScene {
    objects: Vec<Entity>,
    /// Object moved by `move_one`/`editor_resync`.
    mover: Entity,
    mover_origin: Vec3,
    camera: Camera,
    triangles: usize,
}

fn uv_sphere(radius: f32, segments: u32, squash: f32) -> MeshUpload {
    let rings = segments.max(3);
    let sectors = (segments * 2).max(3);
    let mut vertices = Vec::with_capacity(((rings + 1) * (sectors + 1)) as usize);
    let mut indices = Vec::with_capacity((rings * sectors * 6) as usize);
    for r in 0..=rings {
        let v = r as f32 / rings as f32;
        let phi = v * std::f32::consts::PI;
        for s in 0..=sectors {
            let u = s as f32 / sectors as f32;
            let theta = u * std::f32::consts::TAU;
            let n = Vec3::new(phi.sin() * theta.cos(), phi.cos(), phi.sin() * theta.sin());
            let p = Vec3::new(n.x * radius, n.y * radius * squash, n.z * radius);
            let tangent = Vec3::new(-theta.sin(), 0.0, theta.cos());
            vertices.push(PackedVertex::from_components(
                p.to_array(),
                n.to_array(),
                [u, v],
                tangent.to_array(),
                1.0,
            ));
        }
    }
    let stride = sectors + 1;
    for r in 0..rings {
        for s in 0..sectors {
            let a = r * stride + s;
            let b = a + stride;
            indices.extend_from_slice(&[a, b, a + 1, a + 1, b, b + 1]);
        }
    }
    MeshUpload { vertices, indices }
}

fn bench_camera(target: Vec3, eye: Vec3) -> Camera {
    Camera::perspective_look_at(
        eye,
        target,
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        WIDTH as f32 / HEIGHT as f32,
        0.1,
        2_000.0,
    )
}

/// `n` objects on a square grid over a floor, sharing `unique` meshes and 8
/// materials, lit by `lights` point lights spread over the grid.
fn populate_grid(world: &mut World, n: usize, unique: usize, segments: u32, lights: usize) -> BenchScene {
    let unique = unique.clamp(1, n.max(1));
    let materials: Vec<Entity> = (0..8)
        .map(|i| {
            let hue = i as f32 / 8.0;
            spawn_material(
                world,
                make_material(
                    [0.4 + 0.5 * hue, 0.6 - 0.3 * hue, 0.3 + 0.2 * hue, 1.0],
                    0.3 + 0.08 * i as f32,
                    if i % 3 == 0 { 0.8 } else { 0.0 },
                    [0.0; 3],
                    0.0,
                ),
            )
        })
        .collect();
    let mut triangles = 0;
    let meshes: Vec<(Entity, usize)> = (0..unique)
        .map(|i| {
            let upload = uv_sphere(0.4, segments, 0.6 + (i % 7) as f32 * 0.15);
            let tris = upload.indices.len() / 3;
            (spawn_mesh(world, upload), tris)
        })
        .collect();

    let side = (n as f32).sqrt().ceil() as usize;
    let spacing = 1.5;
    let half = side as f32 * spacing * 0.5;

    let floor_material = spawn_material(world, make_material([0.5, 0.5, 0.48, 1.0], 0.8, 0.0, [0.0; 3], 0.0));
    let floor_mesh = spawn_mesh(world, v3_demo_common::box_mesh([0.0; 3], [half + 2.0, 0.1, half + 2.0]));
    triangles += 12;
    let floor = spawn_object(world, floor_mesh, floor_material, Mat4::from_translation(Vec3::new(0.0, -0.6, 0.0)), half * 1.5)
        .expect("floor");

    let mut objects = vec![floor];
    for i in 0..n {
        let (x, z) = (i % side, i / side);
        let position = Vec3::new(x as f32 * spacing - half, 0.0, z as f32 * spacing - half);
        let (mesh, tris) = meshes[i % unique];
        triangles += tris;
        let transform = Mat4::from_scale_rotation_translation(
            Vec3::ONE,
            Quat::from_rotation_y(i as f32 * 0.37),
            position,
        );
        objects.push(spawn_object(world, mesh, materials[i % materials.len()], transform, 0.6).expect("grid object"));
    }

    let light_side = (lights as f32).sqrt().ceil().max(1.0) as usize;
    for i in 0..lights {
        let (x, z) = (i % light_side, i / light_side);
        let step = 2.0 * half / light_side as f32;
        let position = [x as f32 * step - half + step * 0.5, 3.0, z as f32 * step - half + step * 0.5];
        let hue = i as f32 / lights.max(1) as f32;
        spawn_light(world, point_light(position, [1.0, 0.7 + 0.3 * hue, 0.5 + 0.5 * hue], 40.0, step * 1.5 + 4.0));
    }

    // Move an object near the camera so the move is actually visible.
    let mover = objects[1 + n / 2];
    let mover_origin = world
        .get::<StaticObjectComponent>(mover)
        .map(|o| o.transform().w_axis.truncate())
        .unwrap_or(Vec3::ZERO);
    let camera = bench_camera(Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, half.max(6.0) * 0.8, half.max(6.0) * 1.2));
    BenchScene { objects, mover, mover_origin, camera, triangles }
}

/// The HLFS large cathedral with its interior lights plus one small movable prop.
fn populate_cathedral_large(world: &mut World) -> BenchScene {
    v3_demo_common::spawn_indoor_cathedral_sky(world);
    cathedral_large::populate(world);
    for &z in LARGE_CHANDELIER_Z {
        spawn_light(world, point_light([0.0, 31.0, z], [1.0, 0.92, 0.78], 160.0, 22.0));
    }
    for &(x, y, z) in LARGE_CANDLES {
        spawn_light(world, point_light([x, y, z], [1.0, 0.6, 0.15], 8.0, 4.0));
    }
    let material = spawn_material(world, make_material([0.8, 0.2, 0.2, 1.0], 0.4, 0.0, [0.0; 3], 0.0));
    let mesh = spawn_mesh(world, uv_sphere(0.6, 16, 1.0));
    let mover_origin = Vec3::new(0.0, 1.0, 50.0);
    let mover = spawn_object(world, mesh, material, Mat4::from_translation(mover_origin), 0.6).expect("mover");
    let objects: Vec<Entity> = world
        .query::<(&StaticObjectComponent,)>()
        .map(|(entity, _)| entity)
        .collect();
    let triangles = objects
        .iter()
        .filter_map(|e| world.get::<StaticObjectComponent>(*e))
        .map(|o| o.index_count as usize / 3)
        .sum();
    let camera = bench_camera(Vec3::new(0.0, 8.0, -40.0), Vec3::new(0.0, 2.3, 60.0));
    BenchScene { objects, mover, mover_origin, camera, triangles }
}

/// Every opaque object casts ray-traced shadows and every light traces them.
fn enable_ray_shadows(world: &mut World, objects: &[Entity]) {
    let lights: Vec<_> = world
        .query::<(&helio_pass_forward_lit::LightComponent,)>()
        .map(|(id, _)| id)
        .collect();
    for id in lights {
        let mut component = world.get_mut::<helio_pass_forward_lit::LightComponent>(id).unwrap();
        let mut light: helio_pass_forward_lit::GpuLight = (*component).into();
        light.set_ray_traced_shadows(true);
        *component = light.into();
    }
    for &id in objects {
        let material_slot = world.get::<StaticObjectComponent>(id).unwrap().material_slot;
        let blended = world
            .query::<(&helio_pass_gbuffer::MaterialComponent,)>()
            .find(|(entity, _)| entity.index() == material_slot)
            .is_some_and(|(entity, (m,))| {
                m.flags & helio_mats::FLAG_ALPHA_BLEND != 0
                    && world.get::<helio_pass_hlfs::RayTransmission>(entity).is_none()
            });
        if !blended {
            world.get_mut::<StaticObjectComponent>(id).unwrap().flags |=
                helio_pass_object_batch::INSTANCE_FLAG_CASTS_SHADOW;
        }
    }
}

// ── Workloads ─────────────────────────────────────────────────────────────────

fn mover_transform(origin: Vec3, frame: usize) -> Mat4 {
    let t = frame as f32 * 0.15;
    Mat4::from_translation(origin + Vec3::new(t.sin() * 0.8, 0.3 + 0.3 * t.cos(), 0.0))
}

fn move_one(world: &mut World, scene: &BenchScene, frame: usize) {
    let transform = mover_transform(scene.mover_origin, frame);
    let mut object = world.get_mut::<StaticObjectComponent>(scene.mover).expect("mover row");
    let bounds = [transform.w_axis.x, transform.w_axis.y, transform.w_axis.z, object.bounds[3]];
    *object = object.with_transform(transform, bounds);
}

/// Mirrors Pulsar-Native `engine_backend::scene::sync_static_mesh_rows`: every
/// live mesh row is rebuilt with `StaticObjectComponent::new` and re-inserted,
/// whether or not it changed.
fn editor_resync(world: &mut World, scene: &BenchScene, frame: usize) {
    let mover_transform = mover_transform(scene.mover_origin, frame);
    for &entity in &scene.objects {
        let row = *world.get::<StaticObjectComponent>(entity).expect("object row");
        let transform = if entity == scene.mover { mover_transform } else { row.transform() };
        let bounds = [transform.w_axis.x, transform.w_axis.y, transform.w_axis.z, row.bounds[3]];
        world.insert(
            entity,
            StaticObjectComponent::new(
                row.mesh_slot,
                row.mesh_generation,
                row.material_slot,
                row.material_generation,
                transform,
                bounds,
                row.index_count,
                row.first_index,
                row.vertex_offset,
                row.material_class,
                row.graph_hash(),
                row.flags,
            ),
        );
    }
}

// ── Measurement ───────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
struct FrameTiming {
    update: f64,
    flush: f64,
    rt: f64,
    rt_gpu: f64,
    render: f64,
    gpu: f64,
}

impl FrameTiming {
    fn total(&self) -> f64 {
        self.update + self.flush + self.rt + self.rt_gpu + self.render + self.gpu
    }
}

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

struct Bench {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    scene_db: SceneDb,
    renderer: Renderer,
    acceleration: Option<helio_pass_hlfs::SceneDbRayTracing>,
    view: wgpu::TextureView,
    scene: BenchScene,
}

impl Bench {
    fn frame(&mut self, workload: Workload, frame: usize) -> FrameTiming {
        let mut timing = FrameTiming::default();
        let t = Instant::now();
        match workload {
            Workload::Idle => {}
            Workload::MoveOne => move_one(&mut self.scene_db.world, &self.scene, frame),
            Workload::EditorResync => editor_resync(&mut self.scene_db.world, &self.scene, frame),
        }
        timing.update = ms(t);

        let t = Instant::now();
        self.scene_db.world.flush_gpu_mirror(&self.queue);
        timing.flush = ms(t);

        if let Some(acceleration) = self.acceleration.as_mut() {
            let t = Instant::now();
            acceleration.prepare(&self.scene_db.world).expect("RT preparation");
            timing.rt = ms(t);
            // Wait for the acceleration-structure build so its device time is
            // not attributed to the next submission in `render`.
            let t = Instant::now();
            self.device.poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
            timing.rt_gpu = ms(t);
            self.renderer
                .set_ray_tracing_frame_with_transmission(acceleration.tlas(), acceleration.transmission());
        }

        let t = Instant::now();
        self.renderer.render(&self.scene.camera, &self.view).expect("render");
        timing.render = ms(t);

        let t = Instant::now();
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("device poll");
        timing.gpu = ms(t);
        timing
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

fn median_of(frames: &[FrameTiming], f: impl Fn(&FrameTiming) -> f64) -> f64 {
    let mut values: Vec<f64> = frames.iter().map(f).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    percentile(&values, 0.5)
}

struct Row {
    scene: String,
    objects: usize,
    triangles: usize,
    lights: usize,
    mode: Mode,
    workload: Workload,
    frames: Vec<FrameTiming>,
}

async fn device(no_ray_query: bool) -> (Arc<wgpu::Device>, Arc<wgpu::Queue>, wgpu::AdapterInfo) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await
        .expect("no wgpu adapter; on a GPU-less machine set VK_ICD_FILENAMES to lavapipe");
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("move_benchmark"),
            // lavapipe's compiler crashes on the radiance-cascades ray-query
            // pipeline; `--no-ray-query` runs the default graph on its fallback.
            required_features: helio::required_wgpu_features(if no_ray_query {
                adapter.features() - wgpu::Features::EXPERIMENTAL_RAY_QUERY
            } else {
                adapter.features()
            }),
            required_limits: helio::required_wgpu_limits(adapter.limits()),
            experimental_features: helio::required_experimental_features(adapter.features()),
            ..Default::default()
        })
        .await
        .expect("device");
    device.on_uncaptured_error(Arc::new(|error| panic!("wgpu error: {error}")));
    device.set_device_lost_callback(|reason, message| eprintln!("device lost: {reason:?}: {message}"));
    (Arc::new(device), Arc::new(queue), info)
}

fn build_bench(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    args: &Args,
    size: usize,
    mode: Mode,
) -> Bench {
    let mut scene_db = new_scene_db_with_gpu_mirror(device, queue);
    let scene = match args.scene.as_str() {
        "grid" => populate_grid(&mut scene_db.world, size, args.unique_meshes, args.sphere_segments, args.lights),
        "cathedral_large" => populate_cathedral_large(&mut scene_db.world),
        other => panic!("unknown scene {other}; use grid or cathedral_large"),
    };
    if let Some(movability) = args.mesh_movability {
        let meshes: Vec<Entity> = scene_db
            .world
            .query::<(&helio_pass_gbuffer::MeshComponent,)>()
            .map(|(entity, _)| entity)
            .collect();
        for mesh in meshes {
            scene_db.world.insert(mesh, movability);
        }
    }
    if mode == Mode::RayTraced {
        // The renderer captures SceneDB's light flags at build time.
        enable_ray_shadows(&mut scene_db.world, &scene.objects);
    }
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let config = RendererConfig::new(WIDTH, HEIGHT, format)
        .with_render_scale(1.0)
        .with_shadow_quality(helio::ShadowQuality::High);
    let mut renderer = RendererBuilder::new(config, scene_db_handle(&scene_db))
        .with_external_device()
        .with_editor_mode(args.editor)
        .with_pass_build_context(match args.graph.as_str() {
            "hlfs" => Box::new(helio_default_graphs::build_hlfs_graph_with_context),
            // The graph Pulsar-Native's editor viewport builds.
            "default" => Box::new(helio_default_graphs::build_default_graph_external_with_context),
            other => panic!("unknown graph {other}; use hlfs or default"),
        })
        .build(device.clone(), queue.clone(), WIDTH, HEIGHT, format);
    renderer.set_ambient([0.08, 0.08, 0.09], 1.0);
    renderer.set_frame_delta_override(Some(1.0 / 60.0));
    let acceleration = (mode == Mode::RayTraced).then(|| {
        if let Some(pass) = renderer.find_pass_mut::<helio_pass_hlfs::HlfsPass>() {
            pass.set_config(
                device,
                helio_pass_hlfs::HlfsConfig {
                    mode: helio_pass_hlfs::HlfsMode::RayTraced,
                    ..Default::default()
                },
            );
        }
        helio_pass_hlfs::SceneDbRayTracing::new(device.clone(), queue.clone())
    });
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("move_benchmark target"),
        size: wgpu::Extent3d { width: WIDTH, height: HEIGHT, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    Bench { device: device.clone(), queue: queue.clone(), scene_db, renderer, acceleration, view, scene }
}

fn main() {
    env_logger::init();
    let args = parse_args();
    let (device, queue, info) = pollster::block_on(device(args.no_ray_query));
    eprintln!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);
    let rt_supported = device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);

    let sizes = if args.scene == "grid" { args.sizes.clone() } else { vec![0] };
    let mut rows = Vec::new();
    for &size in &sizes {
        for &mode in &args.modes {
            if mode == Mode::RayTraced && !rt_supported {
                eprintln!("skipping rt: adapter has no ray-query support");
                continue;
            }
            let t = Instant::now();
            let mut bench = build_bench(&device, &queue, &args, size, mode);
            let objects = bench.scene.objects.len();
            eprintln!(
                "[{} objects={} tris={} mode={}] built in {:.1}s",
                args.scene,
                objects,
                bench.scene.triangles,
                mode.label(),
                t.elapsed().as_secs_f64()
            );
            for frame in 0..args.warmup {
                bench.frame(Workload::Idle, frame);
            }
            for &workload in &args.workloads {
                // One unmeasured frame lets any one-off transition (the first
                // move after idle) settle out of the steady-state numbers.
                bench.frame(workload, 0);
                let frames: Vec<FrameTiming> =
                    (1..=args.frames).map(|frame| bench.frame(workload, frame)).collect();
                eprintln!(
                    "    {:<14} median total {:8.2} ms",
                    workload.label(),
                    median_of(&frames, FrameTiming::total)
                );
                rows.push(Row {
                    scene: args.scene.clone(),
                    objects,
                    triangles: bench.scene.triangles,
                    lights: bench
                        .scene_db
                        .world
                        .query::<(&helio_pass_forward_lit::LightComponent,)>()
                        .count(),
                    mode,
                    workload,
                    frames,
                });
            }
        }
    }
    report(&args, &info, &rows);
}

fn report(args: &Args, info: &wgpu::AdapterInfo, rows: &[Row]) {
    let mut table = String::new();
    table.push_str(&format!(
        "Adapter: {} ({:?}), graph {}{}, mesh movability {}, {}x{}, {} measured frames per row (median ms; total also p95)\n\n",
        info.name, info.backend, args.graph, if args.editor { " (editor mode)" } else { "" },
        args.mesh_movability.map_or("unset".to_string(), |m| format!("{m:?}")), WIDTH, HEIGHT, args.frames
    ));
    table.push_str("| scene | objects | tris | lights | mode | workload | update | flush | rt | rt_gpu | render | gpu | total | total p95 |\n");
    table.push_str("|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    let mut csv = String::from("scene,objects,triangles,lights,mode,workload,frame,update_ms,flush_ms,rt_ms,rt_gpu_ms,render_ms,gpu_ms,total_ms\n");
    for row in rows {
        let mut totals: Vec<f64> = row.frames.iter().map(FrameTiming::total).collect();
        totals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        table.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} |\n",
            row.scene,
            row.objects,
            row.triangles,
            row.lights,
            row.mode.label(),
            row.workload.label(),
            median_of(&row.frames, |f| f.update),
            median_of(&row.frames, |f| f.flush),
            median_of(&row.frames, |f| f.rt),
            median_of(&row.frames, |f| f.rt_gpu),
            median_of(&row.frames, |f| f.render),
            median_of(&row.frames, |f| f.gpu),
            percentile(&totals, 0.5),
            percentile(&totals, 0.95),
        ));
        for (i, f) in row.frames.iter().enumerate() {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
                row.scene,
                row.objects,
                row.triangles,
                row.lights,
                row.mode.label(),
                row.workload.label(),
                i,
                f.update,
                f.flush,
                f.rt,
                f.rt_gpu,
                f.render,
                f.gpu,
                f.total()
            ));
        }
    }
    println!("{table}");
    std::fs::create_dir_all(&args.out).expect("create --out directory");
    let stem = format!(
        "{}/{}_{}_{}",
        args.out,
        args.scene,
        args.graph,
        args.mesh_movability.map_or("unset".to_string(), |m| format!("{m:?}").to_lowercase())
    );
    std::fs::write(format!("{stem}_summary.md"), &table).expect("write summary");
    std::fs::write(format!("{stem}_frames.csv"), csv).expect("write csv");
    eprintln!("wrote {stem}_summary.md and {stem}_frames.csv");
}
