//! Deterministic offscreen camera path through a populated scene.
use helio::{Camera, LightId, Renderer, RendererBuilder, RendererConfig};
use std::sync::Arc;

pub fn run(directory: &str, populate: fn(&mut Renderer) -> (Vec<LightId>, Vec<LightId>)) {
    let reference = std::env::var_os("HLFS_REFERENCE").is_some();
    let performance = std::env::var_os("HLFS_PERFORMANCE").is_some();
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
        let (width, height) = (640, 360);
        let format = wgpu::TextureFormat::Rgba8UnormSrgb;
        let config = RendererConfig::new(width, height, format)
            .with_shadow_quality(helio::ShadowQuality::High);
        let mut renderer = RendererBuilder::new(config)
            .with_editor_mode(true)
            .with_graph(Box::new(move |d, q, s, c, ds, cb, cs| {
                if fxaa {
                    helio_default_graphs::build_fxaa_hlfs_graph(d, q, s, c, ds, cb, cs, None)
                } else {
                    helio_default_graphs::build_hlfs_graph(d, q, s, c, ds, cb, cs, None)
                }
            }))
            .build(device.clone(), queue.clone(), width, height, format);
        let _ = populate(&mut renderer);
        renderer.set_editor_mode(false);
        // Populate before rebuilding so passes see the actual scene resources.
        let build_graph = if fxaa {
            helio_default_graphs::build_fxaa_hlfs_graph
        } else {
            helio_default_graphs::build_hlfs_graph
        };
        let graph = build_graph(
            &device,
            &queue,
            renderer.scene(),
            config,
            renderer.debug_state(),
            renderer.debug_camera_buf(),
            renderer.cull_stats_buf(),
            None,
        );
        renderer.set_graph(graph);
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
        for frame in 0..100 {
            if reference || performance {
                let pass = renderer
                    .find_pass_mut::<helio_pass_hlfs::HlfsPass>()
                    .expect("HLFS pass");
                pass.set_config(
                    &device,
                    helio_pass_hlfs::HlfsConfig {
                        debug_mode: if reference {
                            helio_pass_hlfs::HlfsDebugMode::Reference
                        } else {
                            helio_pass_hlfs::HlfsDebugMode::Final
                        },
                        ..if performance {
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
            renderer.render(&camera, &view).expect("cathedral frame");
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            if frame >= 16 {
                frame_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            if frame == 0 {
                eprintln!(
                    "Scene: {} lights, {} movable",
                    renderer.scene().gpu_scene().lights.len(),
                    renderer.scene().gpu_scene().movable_light_count
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
