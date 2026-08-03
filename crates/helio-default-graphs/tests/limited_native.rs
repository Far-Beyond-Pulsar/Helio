use std::sync::{Arc, Mutex};

use glam::Vec3;
use helio::{
    required_wgpu_limits, Camera, DebugCameraUniform, DebugDrawState, MaterialBindingMode,
    Renderer, RendererConfig, Scene, BINDLESS_MATERIAL_FEATURES, EXPANDED_MATERIAL_TEXTURE_RESERVE,
    MAX_MATERIAL_TEXTURES,
};
use helio_default_graphs::build_default_graph_external;

const PORTABLE_SAMPLED_TEXTURE_LIMIT: u32 = 16;

#[test]
fn default_graph_renders_on_a_native_non_bindless_device_at_the_16_texture_boundary() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
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
            MaterialBindingMode::Expanded,
            PORTABLE_SAMPLED_TEXTURE_LIMIT as usize - EXPANDED_MATERIAL_TEXTURE_RESERVE,
            "Limited Native",
        )
        .await;
    });
}

#[test]
fn default_graph_retains_the_bindless_material_tier_when_supported() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: bindless native default graph");
            return;
        };
        let required_features =
            wgpu::Features::INDIRECT_FIRST_INSTANCE | BINDLESS_MATERIAL_FEATURES;
        if !adapter.features().contains(required_features) {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_BINDLESS_TIER: bindless native default graph");
            return;
        }
        let limits = required_wgpu_limits(adapter.limits());
        let expected_max = MAX_MATERIAL_TEXTURES
            .min(limits.max_sampled_textures_per_shader_stage as usize)
            .min(limits.max_samplers_per_shader_stage as usize);
        run_default_graph(
            adapter,
            required_features,
            limits,
            MaterialBindingMode::BindingArray,
            expected_max,
            "Bindless Native",
        )
        .await;
    });
}

async fn run_default_graph(
    adapter: wgpu::Adapter,
    required_features: wgpu::Features,
    required_limits: wgpu::Limits,
    expected_mode: MaterialBindingMode,
    expected_max_textures: usize,
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
        .expect("the selected native material capability tier must create a device");
    let device = Arc::new(device);
    let queue = Arc::new(queue);
    let validation_scope = device.push_error_scope(wgpu::ErrorFilter::Validation);

    let scene = Scene::new(Arc::clone(&device), Arc::clone(&queue));
    assert_eq!(scene.material_binding_config().mode, expected_mode);
    assert_eq!(
        scene.material_binding_config().max_textures,
        expected_max_textures
    );

    let config = RendererConfig::new(32, 32, wgpu::TextureFormat::Rgba8Unorm);
    let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));
    let debug_camera = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Limited Native Debug Camera"),
        size: core::mem::size_of::<DebugCameraUniform>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let cull_stats = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Limited Native Cull Stats"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let graph = build_default_graph_external(
        &device,
        &queue,
        &scene,
        config,
        Arc::clone(&debug_state),
        &debug_camera,
        &cull_stats,
        None,
    );

    #[allow(deprecated)]
    let mut renderer = Renderer::new_with_external_device(
        Arc::clone(&device),
        Arc::clone(&queue),
        config.surface_format,
        config.width,
        config.height,
        config.render_scale,
        config,
        scene,
        graph,
        debug_state,
        debug_camera,
        cull_stats,
    );
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
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let validation_error = validation_scope.pop().await;
    assert!(
        validation_error.is_none(),
        "selected-tier default graph validation failed: {validation_error:?}"
    );
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
