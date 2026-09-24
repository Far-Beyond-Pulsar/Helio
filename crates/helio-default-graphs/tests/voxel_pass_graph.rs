use std::sync::{Arc, Mutex};

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera,
    RendererBuilder, RendererConfig,
};
use helio_default_graphs::{build_default_graph_external_with_voxel_passes, VoxelPassFactory};
use helio_pass_tiny_voxel::{
    engine::{EngineVoxelFrame, LazyEngineVoxelPass, SharedVoxelFrame},
    world::render_origin,
    Params, World,
};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

#[test]
fn optional_voxel_pass_builds_and_renders_in_the_deferred_graph() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = instance.request_adapter(&Default::default()).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: voxel default graph");
            return;
        };
        let features = required_wgpu_features(adapter.features());
        let limits = required_wgpu_limits(adapter.limits());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_features: features,
                required_limits: limits,
                experimental_features: required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let gpu_context = EngineGpuContext::new(Arc::clone(&device), Arc::clone(&queue));
        let mut gpu_store = SceneGpuStore::new(
            &gpu_context,
            SceneGpuConfig {
                classes: Vec::new(),
                tombstone_headroom: 0,
                max_cells_metadata: 0,
            },
        );
        helio_pass_sky::SkyComponent::register_gpu_columns_growable(&mut gpu_store, 4, &device);
        helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(
            &mut gpu_store,
            16,
            &device,
        );
        helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(
            &mut gpu_store,
            16,
            &device,
        );
        helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(
            &mut gpu_store,
            16,
            &device,
        );
        helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(
            &mut gpu_store,
            16,
            &device,
        );
        let mirror = GpuMirrorHandle::new(Arc::new(gpu_store), Arc::clone(&queue));
        let frame: SharedVoxelFrame = Arc::new(Mutex::new(None));
        let pass_source = Arc::clone(&frame);
        let factory: VoxelPassFactory = Arc::new(move |_, _, _, _| {
            Box::new(LazyEngineVoxelPass::new(Arc::clone(&pass_source)))
        });
        let mut config = RendererConfig::new(640, 360, wgpu::TextureFormat::Rgba8Unorm);
        config.enable_foliage = false;
        let mut renderer = RendererBuilder::new(config, mirror)
            .with_ambient([0.5, 0.5, 0.6], 1.0)
            .with_external_device()
            .with_pass_build_context(Box::new(move |ctx| {
                build_default_graph_external_with_voxel_passes(ctx, vec![factory])
            }))
            .build(
                Arc::clone(&device),
                Arc::clone(&queue),
                640,
                360,
                config.surface_format,
            );
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel graph smoke target"),
            size: wgpu::Extent3d {
                width: 640,
                height: 360,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let eye = glam::Vec3::new(0.0, 6_371_003.0, 0.0);
        let camera = Camera::perspective_look_at(
            eye,
            eye - glam::Vec3::Y,
            glam::Vec3::Z,
            std::f32::consts::FRAC_PI_4,
            640.0 / 360.0,
            0.1,
            10_000.0,
        );
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        renderer.render(&camera, &view).unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let validation_error = validation_scope.pop().await;
        assert!(
            validation_error.is_none(),
            "empty graph GPU validation: {validation_error:?}"
        );

        // Activate the pass on the next frame. Traversal is disabled here so
        // this checks pipeline and attachment compatibility independently of
        // residency warmup; the crate's GPU tests exercise brick generation.
        *frame.lock().unwrap() = Some(EngineVoxelFrame {
            params: Params {
                origin: [0, 63_710_030, 0, 0],
                fraction: [0.0; 4],
                radial: [0.0, 1.0, 0.0, 0.0],
                right: [1.0, 0.0, 0.0, 640.0 / 360.0],
                up: [0.0, 0.0, -1.0, 0.41421357],
                forward: [0.0, -1.0, 0.0, 0.0],
                screen: [640.0, 360.0, 0.0, 0.0],
                lighting: [0.0, 1.0, 0.0, 0.0],
                settings: [10_000.0, 0.0, 0.0, 0.0],
            },
            world: Arc::new(World::default()),
            raytraced_sun: false,
        });
        let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        renderer.render(&camera, &view).unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let validation_error = validation_scope.pop().await;
        assert!(
            validation_error.is_none(),
            "active graph GPU validation: {validation_error:?}"
        );

        if let Ok(path) = std::env::var("HELIO_VOXEL_CAPTURE") {
            let world = Arc::new(World::default());
            let eye64 = world.ground_spawn(0.0, 0.0, 3.0);
            let eye = eye64.as_vec3();
            let forward = glam::Vec3::new(0.0, -0.15, -1.0).normalize();
            let right = forward.cross(glam::Vec3::Y).normalize();
            let up = right.cross(forward).normalize();
            let camera = Camera::perspective_look_at(
                eye,
                eye + forward,
                glam::Vec3::Y,
                std::f32::consts::FRAC_PI_4,
                640.0 / 360.0,
                0.1,
                10_000.0,
            );
            let origin = render_origin(eye64);
            let fraction = [
                (eye64.x / 0.1 - f64::from(origin[0])) as f32,
                (eye64.y / 0.1 - f64::from(origin[1])) as f32,
                (eye64.z / 0.1 - f64::from(origin[2])) as f32,
                0.0,
            ];
            *frame.lock().unwrap() = Some(EngineVoxelFrame {
                params: Params {
                    origin: [origin[0], origin[1], origin[2], 0],
                    fraction,
                    radial: [0.0, 1.0, 0.0, 0.0],
                    right: [right.x, right.y, right.z, 640.0 / 360.0],
                    up: [up.x, up.y, up.z, 0.41421357],
                    forward: [forward.x, forward.y, forward.z, 0.0],
                    screen: [640.0, 360.0, 0.0, 0.0],
                    lighting: [0.4, 0.8, 0.3, 0.0],
                    settings: [10_000.0, 0.0, 1.0, 0.0],
                },
                world,
                raytraced_sun: false,
            });
            let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            for _ in 0..240 {
                renderer.render(&camera, &view).unwrap();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                if renderer
                    .find_pass::<LazyEngineVoxelPass>()
                    .is_some_and(LazyEngineVoxelPass::ready)
                {
                    break;
                }
            }
            assert!(
                renderer.find_pass::<LazyEngineVoxelPass>().unwrap().ready(),
                "voxel cut still loading after 240 frames; {} jobs pending",
                renderer
                    .find_pass::<LazyEngineVoxelPass>()
                    .unwrap()
                    .chunk_jobs_pending()
            );
            let validation_error = validation_scope.pop().await;
            assert!(
                validation_error.is_none(),
                "capture GPU validation: {validation_error:?}"
            );
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Voxel graph capture"),
                size: 640 * 360 * 4,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_texture_to_buffer(
                target.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &readback,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(640 * 4),
                        rows_per_image: Some(360),
                    },
                },
                wgpu::Extent3d {
                    width: 640,
                    height: 360,
                    depth_or_array_layers: 1,
                },
            );
            queue.submit([encoder.finish()]);
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.recv().unwrap().unwrap();
            let bytes = slice.get_mapped_range().unwrap().to_vec();
            image::save_buffer(path, &bytes, 640, 360, image::ColorType::Rgba8).unwrap();
        }
    });
}
