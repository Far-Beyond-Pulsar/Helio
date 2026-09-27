//! Headless multi-resolution render benchmark and frame capture.
//!
//! Renders representative scenes at several output resolutions, records the
//! wall-clock cost of every frame and the graph profiler's per-pass CPU/GPU
//! timings, and saves the last frame of each run as a PNG so a before/after
//! pair can be diffed with `scripts/frame_diff.py`.
//!
//! Every run is deterministic: a fixed frame delta, a fixed frame count and a
//! fixed camera, so the same build renders the same pixels. Temporal effects
//! (volumetric fog accumulation, TSR/FXAA history, exposure adaptation) are
//! given `--warmup` frames to settle before measuring.
//!
//! Scenes:
//!
//! * `fog_hall`        -- colonnade with a shadowed volumetric sun, local fog,
//!                        point lights, metallic/rough spheres, emissive
//!                        fixtures (bloom) and alpha-blended glass panes.
//! * `cathedral_large` -- the HLFS large cathedral (416k tris, many lights,
//!                        stained glass) with its chandelier and candle lights.
//! * `sky`             -- a floor under an open sky: the fixed per-pixel cost
//!                        of the graph with almost no geometry.
//!
//! Usage (all flags optional):
//!
//! ```text
//! resolution_bench --scenes fog_hall,cathedral_large,sky \
//!                  --res 1920x1080,2560x1440,3840x2160 \
//!                  --frames 6 --warmup 6 --scale 0.75 --out bench_out --tag before
//! ```
//!
//! `--pulsar-columns` registers the SceneDB columns Pulsar-Native's editor
//! registers up front (billboards, decals, water volumes and hitboxes), so
//! their buffers exist while empty, as they do in the editor.
//!
//! `--scale` is the renderer's internal render scale (the editor uses the
//! `RendererConfig` default, 0.75). `--editor` renders in editor mode (light
//! billboards, grid). `--tsr` switches FXAA for TSR. `--no-capture` skips PNGs.
//! `--no-ray-query` is needed on lavapipe, whose compiler crashes on the
//! radiance-cascades ray-query pipeline.
//!
//! No window or surface is created. On a machine without a GPU, Mesa's
//! lavapipe works: `VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`.
//! lavapipe executes "GPU" work on the CPU, so its GPU numbers track shader
//! and fill work (per-pixel cost) rather than a real GPU's bandwidth; use
//! them for relative before/after comparisons of the same build machine.
//!
//! Output (under `--out/<tag>/`): `<scene>_<w>x<h>.png`, `frames.csv`
//! (per-frame CPU/GPU wall time), `passes.csv` (median per-pass CPU/GPU ms)
//! and a Markdown summary on stdout.

#![allow(dead_code)]

mod architectural_mesh;
mod cathedral_large;
mod hlfs_capture;
mod v3_demo_common;

use glam::{Mat4, Vec3};
use helio::{Camera, Renderer, RendererBuilder, RendererConfig};
use pulsar_scenedb::{SceneDb, World};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;
use v3_demo_common::*;

// `cathedral_large` reads these from its parent module.
const LARGE_CHANDELIER_Z: &[f32] = &[-54.0, -36.0, -18.0, 0.0, 18.0, 36.0, 54.0];
const LARGE_CANDLES: &[(f32, f32, f32)] = &[
    (-4.0, 1.6, -64.0),
    (-2.0, 1.6, -63.5),
    (0.0, 1.6, -64.0),
    (2.0, 1.6, -63.5),
    (4.0, 1.6, -64.0),
];

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

struct Args {
    scenes: Vec<String>,
    resolutions: Vec<(u32, u32)>,
    frames: usize,
    warmup: usize,
    scale: f32,
    out: String,
    tag: String,
    editor: bool,
    tsr: bool,
    capture: bool,
    no_ray_query: bool,
    pulsar_columns: bool,
}

fn parse_args() -> Args {
    let mut args = Args {
        scenes: vec!["fog_hall".into(), "cathedral_large".into(), "sky".into()],
        resolutions: vec![(1920, 1080), (2560, 1440), (3840, 2160)],
        frames: 6,
        warmup: 6,
        scale: 0.75,
        out: "resolution_bench_out".into(),
        tag: "run".into(),
        editor: false,
        tsr: false,
        capture: true,
        no_ray_query: false,
        pulsar_columns: false,
    };
    let raw: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < raw.len() {
        let value = raw.get(i + 1).cloned().unwrap_or_default();
        let switch = match raw[i].as_str() {
            "--editor" => Some(&mut args.editor),
            "--tsr" => Some(&mut args.tsr),
            "--no-ray-query" => Some(&mut args.no_ray_query),
            "--pulsar-columns" => Some(&mut args.pulsar_columns),
            _ => None,
        };
        if let Some(switch) = switch {
            *switch = true;
            i += 1;
            continue;
        }
        match raw[i].as_str() {
            "--no-capture" => {
                args.capture = false;
                i += 1;
                continue;
            }
            "--scenes" => args.scenes = value.split(',').map(|s| s.trim().to_string()).collect(),
            "--res" => {
                args.resolutions = value
                    .split(',')
                    .map(|r| {
                        let (w, h) = r.trim().split_once('x').expect("--res takes WxH,WxH");
                        (w.parse().expect("width"), h.parse().expect("height"))
                    })
                    .collect()
            }
            "--frames" => args.frames = value.parse().expect("--frames takes an integer"),
            "--warmup" => args.warmup = value.parse().expect("--warmup takes an integer"),
            "--scale" => args.scale = value.parse().expect("--scale takes a float"),
            "--out" => args.out = value,
            "--tag" => args.tag = value,
            "--help" | "-h" => {
                eprintln!("see the module doc at the top of resolution_bench.rs");
                std::process::exit(0);
            }
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    assert!(args.frames > 0, "--frames must be positive");
    args
}

// ── Scenes ────────────────────────────────────────────────────────────────────

fn fog_hall(world: &mut World, aspect: f32) -> Camera {
    const HALF_X: f32 = 7.0;
    const HALF_Z: f32 = 30.0;
    const ROOF_Y: f32 = 9.0;
    spawn_sky(world, [0.8, 0.9, 1.0]);
    let stone = spawn_material(world, make_material([0.62, 0.60, 0.58, 1.0], 0.85, 0.0, [0.0; 3], 0.0));
    let floor = spawn_mesh(world, plane_mesh([0.0; 3], 40.0));
    spawn_object(world, floor, stone, Mat4::IDENTITY, 40.0).unwrap();
    let roof = spawn_mesh(world, box_mesh([0.0; 3], [HALF_X + 1.0, 0.3, HALF_Z]));
    spawn_object(world, roof, stone, Mat4::from_translation(Vec3::new(0.0, ROOF_Y, 0.0)), HALF_Z).unwrap();
    let pillar = spawn_mesh(world, box_mesh([0.0; 3], [0.6, ROOF_Y * 0.5, 0.6]));
    for i in 0..=12 {
        let z = -HALF_Z + i as f32 * 5.0;
        for side in [-1.0_f32, 1.0] {
            let at = Vec3::new(side * HALF_X, ROOF_Y * 0.5, z);
            spawn_object(world, pillar, stone, Mat4::from_translation(at), ROOF_Y * 0.5).unwrap();
        }
    }

    // Spheres across the roughness/metallic range.
    let sphere = spawn_mesh(world, sphere_mesh([0.0; 3], 0.7));
    for i in 0..10 {
        let t = i as f32 / 9.0;
        let material = spawn_material(
            world,
            make_material([0.9, 0.55 + 0.3 * t, 0.3, 1.0], 0.1 + 0.8 * t, if i % 2 == 0 { 1.0 } else { 0.0 }, [0.0; 3], 0.0),
        );
        let at = Vec3::new(-3.0 + (i % 5) as f32 * 1.5, 0.7, -4.0 - (i / 5) as f32 * 6.0);
        spawn_object(world, sphere, material, Mat4::from_translation(at), 0.7).unwrap();
    }

    // Emissive fixtures (bloom) with a point light under each.
    let fixture = spawn_mesh(world, box_mesh([0.0; 3], [0.4, 0.06, 0.4]));
    let glow = spawn_material(world, make_material([1.0, 0.8, 0.5, 1.0], 0.5, 0.0, [1.0, 0.7, 0.35], 8.0));
    for i in 0..8 {
        let z = -HALF_Z + 4.0 + i as f32 * 7.0;
        spawn_object(world, fixture, glow, Mat4::from_translation(Vec3::new(0.0, ROOF_Y - 0.5, z)), 0.5).unwrap();
        spawn_light(world, point_light([0.0, ROOF_Y - 0.8, z], [1.0, 0.75, 0.45], 25.0, 9.0));
    }

    // Alpha-blended glass panes in front of the camera.
    let pane = spawn_mesh(world, box_mesh([0.0; 3], [1.2, 1.5, 0.03]));
    for (i, tint) in [[0.2, 0.5, 1.0, 0.35], [1.0, 0.3, 0.2, 0.45], [0.3, 1.0, 0.4, 0.3]].into_iter().enumerate() {
        let mut material = make_material(tint, 0.05, 0.0, [0.0; 3], 0.0);
        material.flags |= helio_mats::FLAG_ALPHA_BLEND | helio_mats::FLAG_TRANSPARENT_ONLY;
        let material = spawn_material(world, material);
        let at = Vec3::new(-2.5 + i as f32 * 2.5, 1.6, 4.0 - i as f32 * 2.0);
        spawn_object(world, pane, material, Mat4::from_translation(at), 2.0).unwrap();
    }

    let sun_dir = Vec3::new(0.55, -0.6, -0.35).normalize();
    spawn_light(
        world,
        volumetric_light(directional_light(sun_dir.to_array(), [1.0, 0.9, 0.75], 4.0), SHADOW_BASES[0]),
    );
    let mut haze = LocalFogVolumeComponent::new(
        [-HALF_X - 2.0, 0.0, -HALF_Z - 2.0],
        [HALF_X + 2.0, ROOF_Y + 1.0, HALF_Z + 2.0],
        GlobalFogComponent {
            extinction: 0.04,
            albedo: [0.62, 0.70, 0.85],
            anisotropy: 0.35,
            height_falloff: 0.25,
            ..Default::default()
        },
    );
    haze.edge_fade = 1.5;
    let entity = world.spawn();
    world.insert(entity, haze);
    set_volumetric_quality(world, 1, 120.0);

    let eye = Vec3::new(0.0, 2.0, 16.0);
    Camera::perspective_look_at(eye, eye + Vec3::new(0.0, -0.05, -1.0), Vec3::Y, std::f32::consts::FRAC_PI_4, aspect, 0.1, 500.0)
}

fn cathedral(world: &mut World, aspect: f32) -> Camera {
    spawn_indoor_cathedral_sky(world);
    cathedral_large::populate(world);
    for &z in LARGE_CHANDELIER_Z {
        spawn_light(world, point_light([0.0, 31.0, z], [1.0, 0.92, 0.78], 160.0, 22.0));
    }
    for &(x, y, z) in LARGE_CANDLES {
        spawn_light(world, point_light([x, y, z], [1.0, 0.6, 0.15], 8.0, 4.0));
    }
    Camera::perspective_look_at(
        Vec3::new(0.0, 2.3, 60.0),
        Vec3::new(0.0, 8.0, -40.0),
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        aspect,
        0.1,
        2_000.0,
    )
}

fn sky(world: &mut World, aspect: f32) -> Camera {
    spawn_sky(world, [0.8, 0.9, 1.0]);
    let material = spawn_material(world, make_material([0.5, 0.5, 0.48, 1.0], 0.8, 0.0, [0.0; 3], 0.0));
    let floor = spawn_mesh(world, plane_mesh([0.0; 3], 200.0));
    spawn_object(world, floor, material, Mat4::IDENTITY, 200.0).unwrap();
    spawn_light(world, directional_light([0.3, -0.8, -0.4], [1.0, 0.95, 0.9], 3.0));
    Camera::perspective_look_at(
        Vec3::new(0.0, 3.0, 10.0),
        Vec3::new(0.0, 2.0, -10.0),
        Vec3::Y,
        std::f32::consts::FRAC_PI_4,
        aspect,
        0.1,
        1_000.0,
    )
}

// ── Measurement ───────────────────────────────────────────────────────────────

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
    let features = if no_ray_query {
        adapter.features() - wgpu::Features::EXPERIMENTAL_RAY_QUERY
    } else {
        adapter.features()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("resolution_bench"),
            required_features: helio::required_wgpu_features(features),
            required_limits: helio::required_wgpu_limits(adapter.limits()),
            experimental_features: helio::required_experimental_features(adapter.features()),
            ..Default::default()
        })
        .await
        .expect("device");
    device.on_uncaptured_error(Arc::new(|error| panic!("wgpu error: {error}")));
    (Arc::new(device), Arc::new(queue), info)
}

struct RunResult {
    scene: String,
    width: u32,
    height: u32,
    internal: (u32, u32),
    cpu_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
    /// (pass, cpu samples, gpu samples) in graph execution order.
    passes: Vec<(&'static str, Vec<f64>, Vec<f64>)>,
    graph_vram_kb: u64,
    build_s: f64,
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    sorted[sorted.len() / 2]
}

fn run(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    args: &Args,
    scene_name: &str,
    (width, height): (u32, u32),
) -> RunResult {
    let build_start = Instant::now();
    // Pulsar-Native's editor registers these columns up front (see
    // engine_backend's helio_bridge), so their buffers exist while empty.
    let pulsar_columns = args.pulsar_columns;
    let mut scene_db: SceneDb = new_scene_db_with_gpu_mirror_and(device, queue, |store| {
        if pulsar_columns {
            helio_pass_billboard::BillboardComponent::register_gpu_columns_growable(store, 1024, device);
            helio_pass_decal::DecalComponent::register_gpu_columns_growable(store, 256, device);
            helio_pass_water_sim::WaterVolumeComponent::register_gpu_columns_growable(store, 64, device);
            helio_pass_water_sim::WaterHitboxComponent::register_gpu_columns_growable(store, 256, device);
        }
    });
    let aspect = width as f32 / height as f32;
    let camera = match scene_name {
        "fog_hall" => fog_hall(&mut scene_db.world, aspect),
        "cathedral_large" => cathedral(&mut scene_db.world, aspect),
        "sky" => sky(&mut scene_db.world, aspect),
        other => panic!("unknown scene {other}"),
    };
    let mut config = RendererConfig::new(width, height, FORMAT).with_render_scale(args.scale);
    if args.tsr {
        config = config.with_tsr_quality(helio_pass_tsr::TsrQuality::Quality).with_render_scale(args.scale);
    }
    let internal = (config.internal_width(), config.internal_height());
    let mut renderer: Renderer = RendererBuilder::new(config, scene_db_handle(&scene_db))
        .with_external_device()
        .with_editor_mode(args.editor)
        .with_pass_build_context(Box::new(helio_default_graphs::build_default_graph_external_with_context))
        .build(device.clone(), queue.clone(), width, height, FORMAT);
    renderer.set_ambient([0.08, 0.08, 0.09], 1.0);
    renderer.set_frame_delta_override(Some(1.0 / 60.0));

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resolution_bench target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());
    let build_s = build_start.elapsed().as_secs_f64();

    let mut result = RunResult {
        scene: scene_name.to_string(),
        width,
        height,
        internal,
        cpu_ms: Vec::new(),
        gpu_ms: Vec::new(),
        passes: Vec::new(),
        graph_vram_kb: 0,
        build_s,
    };
    for frame in 0..args.warmup + args.frames {
        flush_scene_db(&scene_db, queue);
        let t = Instant::now();
        renderer.render(&camera, &view).expect("render");
        let cpu = t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let gpu = t.elapsed().as_secs_f64() * 1e3;
        // The external-device graph reads timestamps a frame late, so the
        // snapshot after frame N describes frame N-1: skip one extra frame.
        if frame > args.warmup {
            for pass in &renderer.timing_snapshot().passes {
                let index = match result.passes.iter().position(|(name, ..)| *name == pass.name) {
                    Some(index) => index,
                    None => {
                        result.passes.push((pass.name, Vec::new(), Vec::new()));
                        result.passes.len() - 1
                    }
                };
                let entry = &mut result.passes[index];
                if let Some(ms) = pass.cpu_ms {
                    entry.1.push(ms as f64);
                }
                if let Some(ms) = pass.gpu_ms {
                    entry.2.push(ms as f64);
                }
            }
        }
        if frame >= args.warmup {
            result.cpu_ms.push(cpu);
            result.gpu_ms.push(gpu);
        }
    }
    result.graph_vram_kb = renderer.graph_timeline().physical_vram_kb;

    if args.capture {
        let dir = format!("{}/{}", args.out, args.tag);
        std::fs::create_dir_all(&dir).unwrap();
        let path = format!("{dir}/{scene_name}_{width}x{height}.png");
        save_png(device, queue, &target, width, height, &path);
    }
    result
}

fn save_png(device: &wgpu::Device, queue: &wgpu::Queue, target: &wgpu::Texture, width: u32, height: u32, path: &str) {
    let row = (width * 4).div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resolution_bench readback"),
        size: u64::from(row * height),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        target.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(height) },
        },
        wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = readback.slice(..).get_mapped_range().unwrap();
    let mut pixels = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height as usize {
        let start = y * row as usize;
        pixels.extend_from_slice(&mapped[start..start + width as usize * 4]);
    }
    drop(mapped);
    readback.unmap();
    image::save_buffer(path, &pixels, width, height, image::ColorType::Rgba8).unwrap();
}

fn main() {
    env_logger::init();
    let args = parse_args();
    let (device, queue, info) = pollster::block_on(device(args.no_ray_query));
    eprintln!("adapter: {} ({:?}, {:?})", info.name, info.device_type, info.backend);

    let dir = format!("{}/{}", args.out, args.tag);
    std::fs::create_dir_all(&dir).unwrap();
    let mut frames_csv = std::fs::File::create(format!("{dir}/frames.csv")).unwrap();
    writeln!(frames_csv, "scene,width,height,frame,cpu_ms,gpu_ms").unwrap();
    let mut passes_csv = std::fs::File::create(format!("{dir}/passes.csv")).unwrap();
    writeln!(passes_csv, "scene,width,height,pass,cpu_ms,gpu_ms").unwrap();

    let mut results = Vec::new();
    for scene in &args.scenes {
        for &resolution in &args.resolutions {
            let result = run(&device, &queue, &args, scene, resolution);
            eprintln!(
                "[{} {}x{} internal {}x{}] built {:.1}s, cpu {:.2} ms, gpu {:.2} ms",
                result.scene,
                result.width,
                result.height,
                result.internal.0,
                result.internal.1,
                result.build_s,
                median(&result.cpu_ms),
                median(&result.gpu_ms)
            );
            for (i, (cpu, gpu)) in result.cpu_ms.iter().zip(&result.gpu_ms).enumerate() {
                writeln!(frames_csv, "{},{},{},{i},{cpu:.3},{gpu:.3}", result.scene, result.width, result.height).unwrap();
            }
            for (pass, cpu, gpu) in &result.passes {
                writeln!(
                    passes_csv,
                    "{},{},{},{pass},{:.4},{:.4}",
                    result.scene,
                    result.width,
                    result.height,
                    median(cpu),
                    median(gpu)
                )
                .unwrap();
            }
            results.push(result);
        }
    }

    println!("\n## resolution_bench `{}`\n", args.tag);
    println!(
        "Adapter: {} ({:?}), scale {}, {} measured frames after {} warmup (median ms){}{}\n",
        info.name,
        info.backend,
        args.scale,
        args.frames,
        args.warmup,
        if args.editor { ", editor mode" } else { "" },
        if args.tsr { ", TSR" } else { "" }
    );
    println!("| scene | output | internal | render CPU | GPU wait | graph GPU | graph VRAM MiB |");
    println!("|---|---|---|---:|---:|---:|---:|");
    for r in &results {
        let graph_gpu = r
            .passes
            .iter()
            .filter(|(name, ..)| *name == "__graph_compute" || *name == "__graph_graphics")
            .map(|(_, _, gpu)| median(gpu))
            .filter(|v| v.is_finite())
            .sum::<f64>();
        println!(
            "| {} | {}x{} | {}x{} | {:.2} | {:.2} | {:.2} | {:.1} |",
            r.scene,
            r.width,
            r.height,
            r.internal.0,
            r.internal.1,
            median(&r.cpu_ms),
            median(&r.gpu_ms),
            graph_gpu,
            r.graph_vram_kb as f64 / 1024.0
        );
    }
    for r in &results {
        println!("\n### {} {}x{} — top passes by GPU ms\n", r.scene, r.width, r.height);
        println!("| pass | CPU ms | GPU ms |");
        println!("|---|---:|---:|");
        let mut rows: Vec<_> = r
            .passes
            .iter()
            .filter(|(name, ..)| !name.starts_with("__"))
            .map(|(name, cpu, gpu)| (*name, median(cpu), median(gpu)))
            .collect();
        rows.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        for (name, cpu, gpu) in rows.iter().take(15) {
            println!("| {name} | {cpu:.3} | {gpu:.3} |");
        }
    }
}
