use std::sync::{Arc, Mutex};

use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits,
    DebugCameraUniform, DebugDrawState, GraphRebuilder, RendererConfig, Scene,
};
use helio_default_graphs::{
    build_default_graph_external, build_default_graph_external_with_planetary_voxels,
};
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_planetary_voxel::{
    PlanetaryVoxelGpuConfig, PlanetaryVoxelRenderConfig, PlanetaryVoxelRenderPass,
    TransvoxelGpuExtractorConfig, TransvoxelGpuTransitionExtractorConfig,
};
use helio_pass_voxel_mesh::VoxelMeshPass;

#[test]
fn planetary_extension_is_opt_in_ordered_and_preserved_by_rebuild() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = request_test_adapter(&instance).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: planetary default graph extension");
            return;
        };
        let features = required_wgpu_features(adapter.features());
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Planetary Default Graph Extension Device"),
                required_features: features,
                required_limits: required_wgpu_limits(adapter.limits()),
                experimental_features: required_experimental_features(adapter.features()),
                ..Default::default()
            })
            .await
            .expect("Helio-compatible adapter must create a device");
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let scene = Scene::new(Arc::clone(&device), Arc::clone(&queue));
        let config = RendererConfig::new(32, 32, wgpu::TextureFormat::Rgba8UnormSrgb);
        let debug_state = Arc::new(Mutex::new(DebugDrawState::default()));
        let debug_camera = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Extension Debug Camera"),
            size: core::mem::size_of::<DebugCameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let cull_stats = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Planetary Extension Cull Stats"),
            size: 32,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let default_graph = build_default_graph_external(
            &device,
            &queue,
            &scene,
            config,
            Arc::clone(&debug_state),
            &debug_camera,
            &cull_stats,
            None,
        );
        assert!(default_graph
            .find_pass::<PlanetaryVoxelRenderPass>()
            .is_none());
        let default_dependency_result = default_graph.validate_dependencies();

        let planetary_config = test_planetary_config();
        let mut graph = build_default_graph_external_with_planetary_voxels(
            &device,
            &queue,
            &scene,
            config,
            Arc::clone(&debug_state),
            &debug_camera,
            &cull_stats,
            None,
            planetary_config,
        )
        .expect("bounded planetary graph configuration must build");
        assert_planetary_pass_contract(&graph, planetary_config, &default_dependency_result);

        let rebuilder = graph
            .take_graph_data::<GraphRebuilder>()
            .expect("default graph carries its resize rebuilder");
        let rebuilt = rebuilder(
            &device,
            &queue,
            &scene,
            RendererConfig::new(48, 24, config.surface_format),
            debug_state,
            &debug_camera,
            &cull_stats,
        );
        assert_planetary_pass_contract(&rebuilt, planetary_config, &default_dependency_result);
    });
}

fn assert_planetary_pass_contract(
    graph: &helio_core::RenderGraph,
    expected_config: PlanetaryVoxelRenderConfig,
    expected_dependency_result: &Result<(), String>,
) {
    assert_same_dependency_result(&graph.validate_dependencies(), expected_dependency_result);
    let deferred = graph
        .pass_index_of::<DeferredLightPass>()
        .expect("default graph contains deferred lighting");
    let planetary = graph
        .pass_index_of::<PlanetaryVoxelRenderPass>()
        .expect("opt-in graph contains the planetary pass");
    let voxel = graph
        .pass_index_of::<VoxelMeshPass>()
        .expect("default graph contains the retained voxel mesh pass");
    assert!(deferred < planetary && planetary < voxel);
    assert_eq!(
        graph
            .find_pass::<PlanetaryVoxelRenderPass>()
            .expect("planetary pass remains discoverable")
            .config(),
        expected_config
    );
}

fn assert_same_dependency_result(actual: &Result<(), String>, expected: &Result<(), String>) {
    match (actual, expected) {
        (Ok(()), Ok(())) => {}
        (Err(actual), Err(expected)) => assert_eq!(
            actual.split("Available:").next(),
            expected.split("Available:").next(),
            "the opt-in pass must not change the first dependency-validation failure"
        ),
        _ => panic!("the opt-in pass must not add a dependency-validation regression"),
    }
}

fn test_planetary_config() -> PlanetaryVoxelRenderConfig {
    PlanetaryVoxelRenderConfig {
        residency: PlanetaryVoxelGpuConfig::new(2, 8, 8, 2, 4)
            .expect("test residency configuration is valid"),
        max_surface_pages: 2,
        max_pending_surfaces: 2,
        regular: TransvoxelGpuExtractorConfig::new(1_024, 2_048)
            .expect("test regular extraction configuration is valid"),
        transition: TransvoxelGpuTransitionExtractorConfig::new(512, 1_536)
            .expect("test transition extraction configuration is valid"),
        max_surface_bytes: 16 * 1024 * 1024,
    }
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
