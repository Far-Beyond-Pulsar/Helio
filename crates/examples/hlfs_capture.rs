//! Deterministic offscreen camera path through a populated scene.
use helio::{Camera, RendererBuilder, RendererConfig};
use pulsar_scenedb::{Entity, World};
use std::sync::Arc;

#[path = "architectural_materials.rs"]
pub mod architectural_materials;

pub fn run(directory: &str, populate: fn(&mut World) -> (Vec<Entity>, Vec<Entity>)) {
    run_scene(directory, "cathedral", populate, |t, aspect| {
        Camera::perspective_look_at(
            glam::Vec3::new(2.0 * t, 2.0, 24.0 - 18.0 * t),
            glam::Vec3::new(0.0, 5.0, -20.0), glam::Vec3::Y,
            std::f32::consts::FRAC_PI_4, aspect, 0.1, 200.0,
        )
    });
}

pub fn run_scene(
    directory: &str,
    name: &str,
    populate: fn(&mut World) -> (Vec<Entity>, Vec<Entity>),
    camera_path: fn(f32, f32) -> Camera,
) {
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
    let capture_tail = std::env::var("HLFS_CAPTURE_TAIL").map(|value|
        value.parse::<u32>().expect("HLFS_CAPTURE_TAIL must be an integer")).unwrap_or(1);
    assert!(capture_tail > 0 && capture_tail <= capture_frames, "invalid capture tail count");
    let presampled = std::env::var_os("HLFS_PRESAMPLED").is_some();
    let temporal_resampling = std::env::var_os("HLFS_TEMPORAL_RIS").is_some();
    assert!(!temporal_resampling || presampled, "HLFS_TEMPORAL_RIS requires HLFS_PRESAMPLED");
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
    let tsr_reactivity = std::env::var("HLFS_TSR_REACTIVITY").ok().map(|value| {
        let value = value.parse::<f32>().expect("HLFS_TSR_REACTIVITY must be a number");
        assert!(value.is_finite() && (0.0..=1.0).contains(&value), "invalid TSR reactivity");
        assert!(std::env::var_os("HLFS_TSR_NATIVE").is_some(), "reactivity requires TSR");
        value
    });
    let candidate_count = std::env::var("HLFS_CANDIDATE_COUNT").ok().map(|value| {
        value.parse::<u32>().expect("HLFS_CANDIDATE_COUNT must be an integer")
    });
    let fixed_camera = std::env::var("HLFS_FIXED_CAMERA").ok().map(|value| {
        let t = value.parse::<f32>().expect("HLFS_FIXED_CAMERA must be a number in [0, 1]");
        assert!(t.is_finite() && (0.0..=1.0).contains(&t), "invalid fixed camera position");
        t
    });
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
        let mut config = RendererConfig::new(width, height, format)
            // Capture resolution denotes native internal rendering by default.
            // Do not silently inherit the renderer's interactive 75% preset.
            .with_render_scale(1.0)
            .with_shadow_quality(helio::ShadowQuality::High)
            .with_ssr(std::env::var_os("HLFS_SSR").is_some());
        if std::env::var_os("HLFS_TSR_NATIVE").is_some() {
            assert!(!fxaa, "select one AA method for a controlled capture");
            config = config.with_tsr_quality(helio_pass_tsr::TsrQuality::Native);
        }
        if let Ok(value) = std::env::var("HLFS_RENDER_SCALE") {
            let scale = value.parse::<f32>().expect("HLFS_RENDER_SCALE must be a number");
            assert!(scale.is_finite() && (0.25..=1.0).contains(&scale), "invalid render scale");
            assert!(std::env::var_os("HLFS_TSR_NATIVE").is_none() || scale == 1.0,
                "native TSR requires render scale 1.0");
            config = config.with_render_scale(scale);
        }
        let render_scale = config.render_scale;
        let internal_size = (config.internal_width(), config.internal_height());
        let mut scene_db = crate::v3_demo_common::new_scene_db_with_gpu_mirror(&device, &queue);
        let (chandelier_light_ids, candle_light_ids) = populate(&mut scene_db.world);
        if ray_traced {
            enable_ray_shadows(&mut scene_db.world);
        }
        let mut acceleration =
            helio_pass_hlfs::SceneDbRayTracing::new(device.clone(), queue.clone());
        let mut scene_handle=crate::v3_demo_common::scene_db_handle(&scene_db);
        let architectural_store = architectural_materials::load(&device, &queue, &mut scene_db.world);
        let has_architectural_textures = architectural_store.is_some();
        if let Some(store) = architectural_store {
            scene_handle = scene_handle.with_texture_store(store).unwrap();
        }
        let mut diagnostic_texture_store=None;
        let texture_lifecycle=std::env::var_os("HLFS_TEXTURE_LIFECYCLE").is_some();
        assert!(!texture_lifecycle || std::env::var_os("HLFS_TEXTURE_CHECKER").is_some(), "texture lifecycle requires checker mode");
        // Diagnostic material texture, not an architectural art asset. Keep
        // ownership in SceneDB's store and refer to its slot from material rows.
        if std::env::var_os("HLFS_TEXTURE_CHECKER").is_some() {
            let mut store=pulsar_scenedb::gpu::TextureStore::new(1);
            let slot=register_checker(&device,&queue,&mut store,false);
            let materials:Vec<_>=scene_db.world.query::<(&helio_pass_gbuffer::MaterialComponent,)>()
                .filter(|(_, (material,))|material.base_color[3]>=1.0).map(|(entity,_)|entity).collect();
            for entity in materials {
                scene_db.world.get_mut::<helio_pass_gbuffer::MaterialComponent>(entity).unwrap().tex_base_color=slot;
            }
            let store=Arc::new(std::sync::RwLock::new(store));
            scene_handle=scene_handle.with_texture_store(store.clone()).unwrap();
            diagnostic_texture_store=Some(store);
        }
        let mut renderer =
            RendererBuilder::new(config, scene_handle)
                .with_editor_mode(false)
                .with_pass_build_context(Box::new(move |ctx| {
                    if fxaa {
                        helio_default_graphs::build_fxaa_hlfs_graph_with_context(ctx)
                    } else {
                        helio_default_graphs::build_hlfs_graph_with_context(ctx)
                    }
                }))
                .build(device.clone(), queue.clone(), width, height, format);
        if has_architectural_textures {
            architectural_materials::configure_sampler(&mut renderer);
        }
        if std::env::var_os("HLFS_NO_TRANSPARENCY_REACTIVITY").is_some() {
            renderer.find_pass_mut::<helio_pass_tsr::TsrPass>()
                .expect("HLFS_NO_TRANSPARENCY_REACTIVITY requires TSR")
                .set_transparency_reactivity(false);
        }
        if let Some(value) = tsr_reactivity {
            renderer.find_pass_mut::<helio_pass_tsr::TsrPass>().expect("TSR pass").set_reactivity(value);
        }
        renderer.set_ambient([0.05, 0.05, 0.08], 1.0);
        // Camera motion advances by frame index. Temporal filters and animated
        // passes must use the same fixed clock, independent of capture readback.
        renderer.set_frame_delta_override(Some(1.0 / 60.0));
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
        let timing = std::env::var_os("HLFS_CAPTURE_TIMINGS").map(|_| {
            let pass = renderer.find_pass_mut::<helio_pass_hlfs::HlfsPass>().expect("HLFS pass");
            assert!(pass.enable_timing(&device), "GPU timestamps unavailable");
            let query = pass.timing_query().unwrap().clone();
            let resolve = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Capture HLFS timestamps"), size: 56,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let read = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Capture HLFS timestamp readback"), size: 56,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (query, resolve, read)
        });
        let mut timing_csv = String::from("frame,coarse_ms,fine_ms,sampling_ms,temporal_ms,spatial_ms,composite_ms,hlfs_only_ms\n");
        for frame in 0..capture_frames {
            if texture_lifecycle {
                let mut store=diagnostic_texture_store.as_ref().unwrap().write().unwrap();
                if frame==48 { store.unregister(0).unwrap(); }
                if frame==80 { assert_eq!(register_checker(&device,&queue,&mut store,true),0); }
                if frame+1==capture_frames { eprintln!("Diagnostic texture uploads: {}",store.upload_count()); }
            }
            if ray_traced || reference || performance || presampled || sample_count.is_some() || candidate_count.is_some() {
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
                        temporal_resampling,
                        candidates_per_sample: candidate_count.unwrap_or(if presampled { 2 } else { 8 }),
                        samples_per_pixel: sample_count.unwrap_or(if presampled {
                            helio_pass_hlfs::HlfsConfig::ray_traced_presampled().samples_per_pixel
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
            let t = fixed_camera.unwrap_or((frame as f32 / 99.0).clamp(0.0, 1.0));
            let camera = camera_path(t, width as f32 / height as f32);
            let start = std::time::Instant::now();
            crate::v3_demo_common::flush_scene_db(&scene_db, &queue);
            if ray_traced {
                acceleration
                    .prepare(&scene_db.world)
                    .expect("SceneDB RT geometry");
                renderer.set_ray_tracing_frame_with_transmission(acceleration.tlas(), acceleration.transmission());
            }
            renderer.render(&camera, &view).expect("cathedral frame");
            if frame == 0 {
                let pass = renderer.find_pass_mut::<helio_pass_hlfs::HlfsPass>().expect("HLFS pass");
                let size = pass.output_texture().size();
                assert_eq!((size.width, size.height), internal_size,
                    "HLFS target must match the configured internal resolution");
                let sample_scale = pass.config().sample_scale;
                let sample_width = size.width.div_ceil(sample_scale);
                let sample_height = size.height.div_ceil(sample_scale);
                let aa = if fxaa { "fxaa" } else if std::env::var_os("HLFS_TSR_NATIVE").is_some() { "tsr_native" } else { "none" };
                let transparency_reactivity = aa == "tsr_native" && std::env::var_os("HLFS_NO_TRANSPARENCY_REACTIVITY").is_none();
                let metadata = format!(
                    "{{\n  \"output\": [{width}, {height}],\n  \"internal\": [{}, {}],\n  \"hlfs_sampling\": [{sample_width}, {sample_height}],\n  \"render_scale\": {render_scale},\n  \"sample_scale\": {sample_scale},\n  \"aa\": \"{aa}\",\n  \"stone_textures\": {has_architectural_textures},\n  \"transparency_reactivity\": {transparency_reactivity},\n  \"reference\": {reference},\n  \"fixed_delta_seconds\": 0.016666666666666666\n}}\n",
                    size.width, size.height);
                eprintln!("Capture dimensions: {metadata}");
                std::fs::write(std::path::Path::new(directory).join("capture-config.json"), metadata).unwrap();
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            if frame >= 16 {
                frame_times.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            // Resolve outside the serialized-frame interval. This measures only
            // HLFS's six GPU stages: it does not include SceneDB or TLAS work.
            if let Some((query, resolve, read)) = &timing {
                let mut encoder = device.create_command_encoder(&Default::default());
                encoder.resolve_query_set(query, 0..7, resolve, 0);
                encoder.copy_buffer_to_buffer(resolve, 0, read, 0, 56);
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                read.slice(..).map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let bytes = read.slice(..).get_mapped_range().unwrap();
                let ticks: &[u64] = bytemuck::cast_slice(&bytes);
                let stages: [f64; 6] = std::array::from_fn(|i|
                    (ticks[i + 1] - ticks[i]) as f64 * queue.get_timestamp_period() as f64 / 1e6);
                timing_csv.push_str(&format!("{frame},{},{},{},{},{},{},{}\n",
                    stages[0],stages[1],stages[2],stages[3],stages[4],stages[5],stages.iter().sum::<f64>()));
                drop(bytes);
                read.unmap();
            }
            if frame == 0 {
                eprintln!(
                    "Scene: {} chandelier lights, {} candle lights",
                    chandelier_light_ids.len(),
                    candle_light_ids.len()
                );
            }
            if [0, 31, 63, 99].contains(&frame) || frame >= capture_frames - capture_tail {
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
                        "{name}-{}{frame:03}.png",
                        if reference { "reference-" } else { "" }
                    )),
                    &bytes,
                    width,
                    height,
                    image::ColorType::Rgba8,
                )
                .unwrap();
                eprintln!("Captured {name} frame {frame}");
            }
        }
        frame_times.sort_by(f64::total_cmp);
        eprintln!("Serialized frame latency (CPU + GPU, excluding capture readback): median_ms={:.3} p95_ms={:.3}", frame_times[frame_times.len()/2], frame_times[frame_times.len()*95/100]);
        if timing.is_some() {
            std::fs::write(std::path::Path::new(directory).join("hlfs-gpu-timings.csv"), timing_csv).unwrap();
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    });
}

/// Include opaque geometry and explicitly authored thin-sheet RT materials.
pub fn enable_ray_shadows(world: &mut World) {
    let ids: Vec<_> = world
        .query::<(&helio_pass_forward_lit::LightComponent,)>()
        .map(|(id, _)| id)
        .collect();
    for id in ids {
        let mut component = world
            .get_mut::<helio_pass_forward_lit::LightComponent>(id)
            .unwrap();
        let mut light: helio_pass_forward_lit::GpuLight = (*component).into();
        light.set_ray_traced_shadows(std::env::var_os("HLFS_UNSHADOWED").is_none());
        *component = light.into();
    }
    let ids: Vec<_> = world
        .query::<(&helio_pass_gbuffer::StaticObjectComponent,)>()
        .map(|(id, _)| id)
        .collect();
    for id in ids {
        let object = world
            .get::<helio_pass_gbuffer::StaticObjectComponent>(id)
            .unwrap();
        let transparent = world
            .query::<(&helio_pass_gbuffer::MaterialComponent,)>()
            .any(|(entity, (material,))| {
                entity.index() == object.material_slot
                    && material.flags & helio_mats::FLAG_ALPHA_BLEND != 0
                    && world.get::<helio_pass_hlfs::RayTransmission>(entity).is_none()
            });
        // Display alpha alone is not a transmission model.
        if !transparent {
            world
                .get_mut::<helio_pass_gbuffer::StaticObjectComponent>(id)
                .unwrap()
                .flags |= helio_pass_object_batch::INSTANCE_FLAG_CASTS_SHADOW;
        }
    }
}

/// A deliberately obvious texture for binding/lifecycle validation only.
fn register_checker(device: &wgpu::Device, queue: &wgpu::Queue,
    store: &mut pulsar_scenedb::gpu::TextureStore, replacement: bool) -> u32 {
    let mut texels=Vec::new();
    for y in 0..4 { for x in 0..4 {
        let value=if (x+y)%2==0 {255} else {45};
        texels.extend_from_slice(&if replacement {[45,value,45,255]} else {[value,value,value,255]});
    }}
    store.register(device,queue,&wgpu::TextureDescriptor {
        label:Some("Material texture diagnostic checker"),
        size:wgpu::Extent3d {width:4,height:4,depth_or_array_layers:1},
        mip_level_count:1,sample_count:1,dimension:wgpu::TextureDimension::D2,
        format:wgpu::TextureFormat::Rgba8UnormSrgb,
        usage:wgpu::TextureUsages::TEXTURE_BINDING|wgpu::TextureUsages::COPY_DST,
        view_formats:&[],
    },&texels).unwrap()
}
