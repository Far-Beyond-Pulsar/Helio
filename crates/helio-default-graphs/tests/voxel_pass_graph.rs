//! The voxel planet pass inside the default deferred graph: pipeline and
//! attachment compatibility, residency settling, source removal and resize.
use std::sync::{Arc, Mutex};

use glam::Vec3;
use helio::{
    required_experimental_features, required_wgpu_features, required_wgpu_limits, Camera, RendererBuilder,
    RendererConfig,
};
use helio_default_graphs::{build_default_graph_external_with_lighting_passes, GraphPassFactory, VoxelPassFactory};
use helio_pass_voxel_planet::engine::{PlanetFrame, PlanetPass, SharedPlanetFrame};
use helio_pass_voxel_planet::{Planet, PlanetRecipe};
use pulsar_scenedb::gpu::{EngineGpuContext, GpuMirrorHandle, SceneGpuConfig, SceneGpuStore};

struct FinalResourceConsumer {
    expected: [u32; 2],
    observed: Arc<Mutex<Vec<[u32; 2]>>>,
}
impl helio_core::RenderPass for FinalResourceConsumer {
    fn name(&self) -> &'static str {
        "FinalResourceConsumer"
    }
    fn reads(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }
    fn execute(&mut self, ctx: &mut helio_core::PassContext) -> helio_core::Result<()> {
        let texture = ctx.resource_pool.get_texture("pre_aa").unwrap();
        let extent = [texture.width(), texture.height()];
        assert_eq!(extent, self.expected);
        assert!(ctx.registry.texture_view(helio_core::ResourceKey::new("pre_aa")).is_some());
        self.observed.lock().unwrap().push(extent);
        Ok(())
    }
}

#[test]
fn planet_pass_builds_settles_and_resizes_in_the_deferred_graph() {
    pollster::block_on(async {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Ok(adapter) = instance.request_adapter(&Default::default()).await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: voxel default graph");
            return;
        };
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
        let gpu_context = EngineGpuContext::new(Arc::clone(&device), Arc::clone(&queue));
        let mut gpu_store =
            SceneGpuStore::new(&gpu_context, SceneGpuConfig { classes: Vec::new(), tombstone_headroom: 0, max_cells_metadata: 0 });
        helio_pass_sky::SkyComponent::register_gpu_columns_growable(&mut gpu_store, 4, &device);
        helio_pass_gbuffer::MeshComponent::register_gpu_columns_growable(&mut gpu_store, 16, &device);
        helio_pass_gbuffer::MaterialComponent::register_gpu_columns_growable(&mut gpu_store, 16, &device);
        helio_pass_gbuffer::StaticObjectComponent::register_gpu_columns_growable(&mut gpu_store, 16, &device);
        helio_pass_forward_lit::LightComponent::register_gpu_columns_growable(&mut gpu_store, 16, &device);
        let mirror = GpuMirrorHandle::new(Arc::new(gpu_store), Arc::clone(&queue));

        let source: SharedPlanetFrame = Arc::new(Mutex::new(None));
        let pass_source = Arc::clone(&source);
        let factory: VoxelPassFactory = Arc::new(move |_, _, _, _| Box::new(PlanetPass::new(Arc::clone(&pass_source))));
        let mut config = RendererConfig::new(640, 360, wgpu::TextureFormat::Rgba8Unorm);
        config.enable_foliage = false;
        let observed = Arc::new(Mutex::new(Vec::new()));
        let captured = observed.clone();
        let final_factory: GraphPassFactory = Arc::new(move |_, _, width, height| {
            Box::new(FinalResourceConsumer { expected: [width, height], observed: captured.clone() })
        });
        let mut renderer = RendererBuilder::new(config, mirror)
            .with_ambient([0.5, 0.5, 0.6], 1.0)
            .with_external_device()
            .with_pass_build_context(Box::new(move |ctx| {
                build_default_graph_external_with_lighting_passes(ctx, vec![factory], Vec::new(), vec![final_factory])
            }))
            .build(Arc::clone(&device), Arc::clone(&queue), 640, 360, config.surface_format);
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Voxel graph smoke target"),
            size: wgpu::Extent3d { width: 640, height: 360, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());

        // The camera sits at the world origin, which the renderer places at
        // the planet eye.
        let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
        let eye = planet.surface_point(glam::DVec3::new(0.2, 1.0, 0.3), 1.7);
        let up = eye.normalize().as_vec3();
        let forward = (up.any_orthonormal_vector() - up * 0.2).normalize();
        let camera = Camera::perspective_look_at(Vec3::ZERO, forward, up, std::f32::consts::FRAC_PI_4, 640.0 / 360.0, 0.05, 40_000_000.0);
        renderer.set_world_origin(Some(eye));

        // Without a planet frame the pass is inert.
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        renderer.render(&camera, &view).unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let error = validation.pop().await;
        assert!(error.is_none(), "empty graph GPU validation: {error:?}");
        assert!(renderer.find_pass::<PlanetPass>().unwrap().renderer().is_none());

        // With a frame it streams the planet around the eye until settled.
        *source.lock().unwrap() = Some(PlanetFrame { eye, planet: planet.clone(), sun: up, shadows: true });
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut frames = 0;
        loop {
            renderer.render(&camera, &view).unwrap();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            frames += 1;
            if !renderer.find_pass::<PlanetPass>().unwrap().needs_frame() || frames >= 600 {
                break;
            }
        }
        let error = validation.pop().await;
        assert!(error.is_none(), "active graph GPU validation: {error:?}");
        let stats = renderer.find_pass::<PlanetPass>().unwrap().stats().unwrap();
        assert!(frames < 600, "residency still streaming after {frames} frames: {stats:?}");
        assert!(stats.resident_columns > 1000, "{stats:?}");
        assert_eq!(stats.pending_columns, 0);
        eprintln!("VOXEL_GRAPH_SETTLED frames={frames} resident={}", stats.resident_columns);

        // Removing the source drops the planet renderer.
        *source.lock().unwrap() = None;
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        renderer.render(&camera, &view).unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(validation.pop().await.is_none());
        let pass = renderer.find_pass::<PlanetPass>().unwrap();
        assert!(pass.renderer().is_none());
        assert!(!pass.needs_frame());
        assert!(observed.lock().unwrap().contains(&[config.internal_width(), config.internal_height()]));

        // A resize rebuilds the graph; downstream passes see the new size.
        *source.lock().unwrap() = Some(PlanetFrame { eye, planet, sun: up, shadows: true });
        renderer.set_render_size(320, 180);
        let resized = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("final consumer resize"),
            size: wgpu::Extent3d { width: 320, height: 180, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: config.surface_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
        renderer.render(&camera, &resized.create_view(&Default::default())).unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert!(validation.pop().await.is_none());
        let resized_config = RendererConfig { width: 320, height: 180, ..config };
        assert_eq!(
            observed.lock().unwrap().last(),
            Some(&[resized_config.internal_width(), resized_config.internal_height()])
        );
        assert!(renderer.find_pass::<PlanetPass>().unwrap().renderer().is_some());
    });
}
