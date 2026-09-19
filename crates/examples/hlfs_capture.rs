//! Deterministic offscreen camera path through a populated scene.
use helio::{Camera, RendererBuilder, RendererConfig};
use pulsar_scenedb::{Entity, World};
use std::sync::Arc;

pub fn run(directory: &str, populate: fn(&mut World) -> (Vec<Entity>, Vec<Entity>)) {
    let capture_frames = std::env::var("HLFS_CAPTURE_FRAMES")
        .map(|value| {
            value
                .parse::<u32>()
                .expect("HLFS_CAPTURE_FRAMES must be an integer")
        })
        .unwrap_or(100);
    assert!(
        capture_frames > 16,
        "capture needs more than 16 warmup frames"
    );
    let ray_traced = std::env::var_os("HLFS_RT").is_some();
    let presampled = std::env::var_os("HLFS_PRESAMPLED").is_some();
    assert!(
        !presampled || ray_traced,
        "HLFS_PRESAMPLED requires HLFS_RT"
    );

    let reference = std::env::var_os("HLFS_REFERENCE").is_some();
    let performance = std::env::var_os("HLFS_PERFORMANCE").is_some();
    let sample_count = std::env::var("HLFS_SAMPLE_COUNT").ok().map(|value| {
        value
            .parse::<u32>()
            .expect("HLFS_SAMPLE_COUNT must be an integer")
    });
    let fxaa = std::env::var_os("HLFS_FXAA").is_some();
    pollster::block_on(async {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&Default::default())
            .await
            .expect("adapter");
        eprintln!("Capture adapter: {:?}", adapter.get_info());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: helio::required_wgpu_features(adapter.features()),
                required_limits: helio::required_wgpu_limits(adapter.limits()),
                experimental_features: helio::required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let (width, height) = match std::env::var("HLFS_RESOLUTION").as_deref() {
            Ok("1440p") => (2560, 1440),
            Ok("4k") => (3840, 2160),
            Ok(other) => panic!("unsupported HLFS_RESOLUTION: {other}; use 1440p or 4k"),
            Err(_) => (640, 360),
        };
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let config = RendererConfig::new(width, height, format)
            .with_shadow_quality(helio::ShadowQuality::High);
        let mut scene_db = crate::v3_demo_common::new_scene_db_with_gpu_mirror(&device, &queue);
        let (chandelier_light_ids, candle_light_ids) = populate(&mut scene_db.world);
        if ray_traced {
            let ids: Vec<_> = scene_db
                .world
                .query::<(&helio_pass_forward_lit::LightComponent,)>()
                .map(|(id, _)| id)
                .collect();
            for id in ids {
                let mut component = scene_db
                    .world
                    .get_mut::<helio_pass_forward_lit::LightComponent>(id)
                    .unwrap();
                let mut light: helio_pass_forward_lit::GpuLight = (*component).into();
                light.set_ray_traced_shadows(std::env::var_os("HLFS_UNSHADOWED").is_none());
                *component = light.into();
            }
            let ids: Vec<_> = scene_db
                .world
                .query::<(&helio_pass_gbuffer::StaticObjectComponent,)>()
                .map(|(id, _)| id)
                .collect();
            for id in ids {
                scene_db
                    .world
                    .get_mut::<helio_pass_gbuffer::StaticObjectComponent>(id)
                    .unwrap()
                    .flags |= helio_pass_object_batch::INSTANCE_FLAG_CASTS_SHADOW;
            }
        }
        let mut acceleration =
            helio_pass_hlfs::SceneDbRayTracing::new(device.clone(), queue.clone());
        let mut renderer =
            RendererBuilder::new(config, crate::v3_demo_common::scene_db_handle(&scene_db))
                .with_editor_mode(false)
                .with_pass_build_context(Box::new(move |ctx| {
                    if fxaa {
                        helio_default_graphs::build_fxaa_hlfs_graph_with_context(ctx)
                    } else {
                        helio_default_graphs::build_hlfs_graph_with_context(ctx)
                    }
                }))
                .build(device.clone(), queue.clone(), width, height, format);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Cathedral capture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        std::fs::create_dir_all(directory).unwrap();
        let mut frame_times = Vec::new();
        for frame in 0..capture_frames {
            if ray_traced || reference || performance || presampled || sample_count.is_some() {
                let pass = renderer
                    .find_pass_mut::<helio_pass_hlfs::HlfsPass>()
                    .expect("HLFS pass");
                pass.set_config(
                    &device,
                    helio_pass_hlfs::HlfsConfig {
                        mode: if ray_traced {
                            helio_pass_hlfs::HlfsMode::RayTraced
                        } else {
                            helio_pass_hlfs::HlfsMode::ScreenSpace
                        },
                        debug_mode: if reference {
                            helio_pass_hlfs::HlfsDebugMode::Reference
                        } else {
                            helio_pass_hlfs::HlfsDebugMode::Final
                        },
                        samples_per_pixel: sample_count.unwrap_or(if presampled {
                            1
                        } else if performance {
                            4
                        } else {
                            2
                        }),
                        ..if presampled {
                            helio_pass_hlfs::HlfsConfig::ray_traced_presampled()
                        } else if performance {
                            helio_pass_hlfs::HlfsConfig::performance()
                        } else {
                            Default::default()
                        }
                    },
                );
            }
            let t = (frame as f32 / 99.0).clamp(0.0, 1.0);
            let position = glam::Vec3::new(2.0 * t, 2.0, 24.0 - 18.0 * t);
            let camera = Camera::perspective_look_at(
                position,
                glam::Vec3::new(0.0, 5.0, -20.0),
                glam::Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                width as f32 / height as f32,
                0.1,
                200.0,
            );
            let start = std::time::Instant::now();
            crate::v3_demo_common::flush_scene_db(&scene_db, &queue);
            if ray_traced {
                let tlas = acceleration
                    .prepare(&scene_db.world)
                    .expect("SceneDB RT geometry");
                renderer.set_ray_tracing_frame(Some(tlas));
            }
            renderer.render(&camera, &view).expect("cathedral frame");
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            if frame >= 16 {
                frame_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            if frame == 0 {
                eprintln!(
                    "Scene: {} chandelier lights, {} candle lights",
                    chandelier_light_ids.len(),
                    candle_light_ids.len()
                );
            }
            if [0, 31, 63, 99].contains(&frame) {
                let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Capture readback"),
                    size: u64::from(width * height * 4),
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.copy_texture_to_buffer(
                    texture.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(width * 4),
                            rows_per_image: Some(height),
                        },
                    },
                    texture.size(),
                );
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let bytes = buffer.slice(..).get_mapped_range().unwrap();
                image::save_buffer(
                    std::path::Path::new(directory).join(format!(
                        "cathedral-{}{frame:03}.png",
                        if reference { "reference-" } else { "" }
                    )),
                    &bytes,
                    width,
                    height,
                    image::ColorType::Rgba8,
                )
                .unwrap();
                eprintln!("Captured cathedral frame {frame}");
            }
        }
        frame_times.sort_by(f64::total_cmp);
        eprintln!("Serialized frame latency (CPU + GPU, excluding capture readback): median_ms={:.3} p95_ms={:.3}", frame_times[frame_times.len()/2], frame_times[frame_times.len()*95/100]);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    });
}
