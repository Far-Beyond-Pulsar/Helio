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

struct Editor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    renderer: helio::Renderer,
    source: SharedPlanetFrame,
    target: wgpu::Texture,
}

fn editor(width: u32, height: u32) -> Option<Editor> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: editor overlays");
        return None;
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: required_wgpu_features(adapter.features()),
        required_limits: required_wgpu_limits(adapter.limits()),
        experimental_features: required_experimental_features(adapter.features()),
        ..Default::default()
    }))
    .unwrap();
    let (device, queue) = (Arc::new(device), Arc::new(queue));
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
    let mut config = RendererConfig::new(width, height, wgpu::TextureFormat::Rgba8Unorm);
    config.enable_foliage = false;
    let mut renderer = RendererBuilder::new(config, mirror)
        .with_ambient([0.5, 0.5, 0.6], 1.0)
        .with_external_device()
        .with_pass_build_context(Box::new(move |ctx| {
            build_default_graph_external_with_lighting_passes(ctx, vec![factory], Vec::new(), Vec::new())
        }))
        .build(Arc::clone(&device), Arc::clone(&queue), width, height, config.surface_format);
    renderer.set_editor_mode(true);
    renderer.set_fallback_sky_enabled(true);
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("editor overlay target"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: config.surface_format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    Some(Editor { device, queue, renderer, source, target })
}

impl Editor {
    fn render(&mut self, camera: &Camera) {
        self.renderer.render(camera, &self.target.create_view(&Default::default())).unwrap();
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    /// Fractions of pixels in the grid's axis colours (red x axis, green z
    /// axis) and in its neutral grey line colour.
    fn rgba(&self) -> Vec<[u8; 4]> {
        let size = self.target.size();
        let row = (size.width * 4).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: u64::from(row * size.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size.height) },
            },
            size,
        );
        self.queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        (0..size.height as usize).flat_map(|y| {
            data[y * row as usize..][..size.width as usize * 4].chunks_exact(4)
                .map(|px| <[u8; 4]>::try_from(px).unwrap()).collect::<Vec<_>>()
        }).collect()
    }

    fn overlay_fractions(&self) -> (f64, f64) {
        let pixels = self.rgba();
        let (mut axis, mut grey) = (0usize, 0usize);
        for px in &pixels {
            let [r,g,b] = [i32::from(px[0]), i32::from(px[1]), i32::from(px[2])];
            axis += usize::from((r > g + 40 && r > b + 30) || (g > r + 60 && g > b + 60));
            grey += usize::from((r - g).abs() < 12 && (g - b).abs() < 16 && r < 170);
        }
        (axis as f64 / pixels.len() as f64, grey as f64 / pixels.len() as f64)
    }
}

/// Editor overlays in a camera-relative frame (world origin at the eye) must
/// stay in world space. The grid used to put its y = 0 plane through the eye:
/// pressed against voxel ground with a tiny near plane, every pixel hit it
/// next to the x axis and the view turned the axis' red.
#[test]
fn editor_overlays_stay_in_world_space_in_camera_relative_frames() {
    let Some(mut editor) = editor(320, 180) else { return };
    let planet = Arc::new(Planet::new(PlanetRecipe::default()).unwrap());
    let ground = planet.surface_point(glam::DVec3::new(0.2, 1.0, 0.3), 0.0);
    let up = ground.normalize();
    let forward = up.any_orthonormal_vector().as_vec3();
    *editor.source.lock().unwrap() = Some(PlanetFrame { eye: ground, planet: planet.clone(), sun: up.as_vec3(), shadows: true });
    let validation = editor.device.push_error_scope(wgpu::ErrorFilter::Validation);
    // An editor camera held against the ground: a few centimetres of
    // clearance, the smallest near plane and a planetary far plane.
    for step in 0..120 {
        let eye = ground + up * (0.02 + 0.004 * f64::from(step % 10));
        editor.source.lock().unwrap().as_mut().unwrap().eye = eye;
        editor.renderer.set_world_origin(Some(eye));
        let camera =
            Camera::perspective_look_at(Vec3::ZERO, forward, up.as_vec3(), std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.05, 40_000_000.0);
        editor.render(&camera);
        if step % 20 == 19 {
            let (axis, _) = editor.overlay_fractions();
            assert!(axis < 0.001, "step {step}: {:.1}% of the view in grid axis colours", axis * 100.0);
        }
    }
    assert!(pollster::block_on(validation.pop()).is_none());

    // World-space frames still show the grid: above an empty scene, looking
    // down at the origin, the lines and both axes appear.
    *editor.source.lock().unwrap() = None;
    editor.renderer.set_world_origin(None);
    let eye = Vec3::new(3.0, 6.0, 3.0);
    let camera = Camera::perspective_look_at(eye, Vec3::ZERO, Vec3::Y, std::f32::consts::FRAC_PI_4, 16.0 / 9.0, 0.1, 1000.0);
    for _ in 0..4 {
        editor.render(&camera);
    }
    let (axis, grey) = editor.overlay_fractions();
    assert!(axis > 0.001 && grey > 0.01, "grid missing: axis {axis:.4}, lines {grey:.4}");
}

/// The same local camera and sun at two poles must see the same atmosphere.
/// The previous fixed +Y fallback produced a brown lower-hemisphere sky at X.
#[test]
fn planetary_sky_follows_the_world_eye_and_sun_through_resize() {
    let Some(mut editor) = editor(256, 144) else { return };
    editor.renderer.set_editor_mode(false);
    let cases = [
        ([0.0, 6_374_000.0, 0.0], Vec3::new(0.6, 0.8, 0.0), Vec3::Y, [0.4, 0.8, 0.2]),
        ([6_374_000.0, 0.0, 0.0], Vec3::new(0.8, -0.6, 0.0), Vec3::X, [0.8, -0.4, 0.2]),
    ];
    let mut means = Vec::new();
    for (eye, forward, up, sun) in cases {
        editor.renderer.set_planetary_sky(Some(helio_pass_sky::PlanetarySky::earth_like(eye, 6_371_000.0, sun)));
        let camera = Camera::perspective_look_at(Vec3::ZERO, forward, up, std::f32::consts::FRAC_PI_4, 16.0/9.0, 0.05, 40_000_000.0);
        for _ in 0..8 { editor.render(&camera); }
        let pixels = editor.rgba();
        let mean: [f64;3] = std::array::from_fn(|c| pixels.iter().map(|p| f64::from(p[c])).sum::<f64>() / pixels.len() as f64);
        eprintln!("planetary sky mean {mean:?}");
        means.push(mean);
    }
    assert!(means[0][2] > means[0][0] + 10.0, "day sky must be blue");
    for c in 0..3 { assert!((means[0][c] - means[1][c]).abs() < 12.0, "rotated sky mismatch: {means:?}"); }
    editor.renderer.set_render_size(320, 180);
    editor.target = editor.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("resized planetary sky"), size: wgpu::Extent3d { width: 320, height: 180, depth_or_array_layers: 1 },
        mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC, view_formats: &[],
    });
    let camera = Camera::perspective_look_at(Vec3::ZERO, Vec3::new(0.8,-0.6,0.0), Vec3::X, std::f32::consts::FRAC_PI_4, 16.0/9.0, 0.05, 40_000_000.0);
    for _ in 0..8 { editor.render(&camera); }
    let pixels = editor.rgba();
    for c in 0..3 {
        let mean = pixels.iter().map(|p| f64::from(p[c])).sum::<f64>() / pixels.len() as f64;
        assert!((mean - means[1][c]).abs() < 12.0, "sky reset during resize");
    }
}
