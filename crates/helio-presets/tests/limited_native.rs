use std::sync::Arc;

use glam::Vec3;
use helio::{required_wgpu_limits, Camera, RendererBuilder, RendererConfig};
use helio_core::TEXTURE_BINDING_ARRAYS;

const PORTABLE_SAMPLED_TEXTURE_LIMIT: u32 = 16;

#[test]
fn default_graph_renders_on_a_native_non_bindless_device_at_the_16_texture_boundary() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: limited native default graph");
            return;
        };
        if !adapter
            .features()
            .contains(wgpu::Features::INDIRECT_FIRST_INSTANCE)
        {
            eprintln!("GPU_VALIDATION_SKIPPED_MISSING_INDIRECT_FIRST_INSTANCE: limited native default graph");
            return;
        }

        let mut limits = required_wgpu_limits(adapter.limits());
        limits.max_sampled_textures_per_shader_stage = PORTABLE_SAMPLED_TEXTURE_LIMIT;
        limits.max_samplers_per_shader_stage = limits
            .max_samplers_per_shader_stage
            .min(PORTABLE_SAMPLED_TEXTURE_LIMIT);
        run_default_graph(
            adapter,
            wgpu::Features::INDIRECT_FIRST_INSTANCE,
            limits,
            "Limited Native",
        )
        .await;
    });
}

#[test]
fn default_graph_renders_on_a_device_with_texture_binding_arrays() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: bindless native default graph");
            return;
        };
        let required_features = wgpu::Features::INDIRECT_FIRST_INSTANCE | TEXTURE_BINDING_ARRAYS;
        if !adapter.features().contains(required_features) {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_BINDLESS_TIER: bindless native default graph");
            return;
        }
        let limits = required_wgpu_limits(adapter.limits());
        run_default_graph(adapter, required_features, limits, "Bindless Native").await;
    });
}

async fn run_default_graph(
    adapter: wgpu::Adapter,
    required_features: wgpu::Features,
    required_limits: wgpu::Limits,
    label: &'static str,
) {
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features,
            required_limits,
            experimental_features: wgpu::ExperimentalFeatures::disabled(),
            ..Default::default()
        })
        .await
        .expect("the selected native capability tier must create a device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let scene_db = scene_db_with_gpu_mirror(&device, &queue);
    let config = RendererConfig::new(32, 32, wgpu::TextureFormat::Rgba8Unorm);
    let mut renderer = RendererBuilder::new(
        config.clone(),
        scene_db
            .world
            .gpu_mirror()
            .cloned()
            .expect("mirror attached above"),
    )
    .with_pass_build_context(Box::new(helio_presets::deferred))
    .with_external_device()
    .build(
        Arc::clone(&device),
        Arc::clone(&queue),
        config.width,
        config.height,
        config.surface_format,
    );
    let construction_error = validation_scope.pop().await;
    assert!(
        construction_error.is_none(),
        "graph construction: {construction_error:?}"
    );
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    scene_db.world.flush_gpu_mirror(&queue);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Limited Native Render Target"),
        size: wgpu::Extent3d {
            width: config.width,
            height: config.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let camera = Camera::perspective_look_at(
        Vec3::new(0.0, 1.0, 3.0),
        Vec3::ZERO,
        Vec3::Y,
        60.0_f32.to_radians(),
        1.0,
        0.1,
        100.0,
    );

    renderer
        .render(&camera, &target_view)
        .expect("the complete selected-tier default graph must render");

    let resized_target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Limited Native Resized Render Target"),
        size: wgpu::Extent3d {
            width: 48,
            height: 24,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    renderer.set_render_size(48, 24);
    renderer
        .render(&camera, &resized_target.create_view(&Default::default()))
        .expect("the complete selected-tier default graph must render after resize");
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let validation_error = validation_scope.pop().await;
    assert!(
        validation_error.is_none(),
        "selected-tier default graph validation failed: {validation_error:?}"
    );
}

/// An empty SceneDB with a GPU mirror and the columns the default graph
/// reads, the way every frontend now hands scene data to the renderer.
fn scene_db_with_gpu_mirror(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
) -> pulsar_scenedb::SceneDb {
    let mut scene_db = pulsar_scenedb::SceneDb::new();
    let ctx = pulsar_scenedb::gpu::EngineGpuContext::new(device.clone(), queue.clone());
    let mut store = pulsar_scenedb::gpu::SceneGpuStore::new(
        &ctx,
        pulsar_scenedb::gpu::SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    let mirror = pulsar_scenedb::gpu::GpuMirrorHandle::new(Arc::new(store), queue.clone());
    scene_db.world.attach_gpu_mirror(mirror);
    scene_db
}

async fn request_test_adapter(instance: &wgpu::Instance) -> Option<wgpu::Adapter> {
    for force_fallback_adapter in [false, true] {
        if let Ok(adapter) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            return Some(adapter);
        }
    }
    None
}
