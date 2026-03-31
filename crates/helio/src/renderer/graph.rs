use std::sync::Arc;

use helio_pass_billboard::BillboardPass;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_depth_prepass::DepthPrepassPass;
use helio_pass_hiz::HiZBuildPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_simple_cube::SimpleCubePass;
use helio_pass_sky_lut::SkyLutPass;
use helio_pass_sky::SkyPass;
use helio_pass_taa::TaaPass;
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_pass_hlfs::HlfsPass;
use helio_pass_water_caustics::WaterCausticsPass;
use helio_pass_water_surface::WaterSurfacePass;
use helio_pass_underwater::UnderwaterPass;
use helio_v3::RenderGraph;

use crate::scene::Scene;
use crate::renderer::debug::DebugDrawPass;
use crate::renderer::config::RendererConfig;

/// Spotlight icon embedded at compile time — used as the editor billboard sprite.
static SPOTLIGHT_PNG: &[u8] = include_bytes!("../../../../spotlight.png");

pub fn build_default_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<crate::renderer::debug::DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    debug_depth_test: bool,
) -> RenderGraph {
    let gpu_scene = scene.gpu_scene();
    let mut graph = RenderGraph::new(device, queue);

    let camera_buf = gpu_scene.camera.buffer();

    let hiz_pass = HiZBuildPass::new(device, config.internal_width(), config.internal_height());
    let hiz_view = Arc::clone(&hiz_pass.hiz_view);
    let hiz_sampler = Arc::clone(&hiz_pass.hiz_sampler);

    let shadow_dirty_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Flags"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let shadow_hashes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Hashes"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    graph.add_pass(Box::new(ShadowMatrixPass::new(
        device,
        gpu_scene.lights.buffer(),
        gpu_scene.shadow_matrices.buffer(),
        camera_buf,
        &shadow_dirty_buf,
        &shadow_hashes_buf,
    )));

    graph.add_pass(Box::new(ShadowPass::new(device)));

    let has_sky = scene.sky_context().has_sky;
    if has_sky {
        let sky_lut_pass = SkyLutPass::new(device, camera_buf);
        let sky_lut_view = sky_lut_pass.sky_lut_view.clone();
        graph.add_pass(Box::new(sky_lut_pass));

        graph.add_pass(Box::new(SkyPass::new(
            device,
            camera_buf,
            &sky_lut_view,
            config.internal_width(),
            config.internal_height(),
            config.surface_format,
        )));
    }

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        true,
    )));

    graph.add_pass(Box::new(OcclusionCullPass::new(device, hiz_view, hiz_sampler)));

    graph.add_pass(Box::new(DepthPrepassPass::new(device, wgpu::TextureFormat::Depth32Float)));

    graph.add_pass(Box::new(hiz_pass));

    graph.add_pass(Box::new(LightCullPass::new(device, config.internal_width(), config.internal_height())));

    graph.add_pass(Box::new(GBufferPass::new(device, config.internal_width(), config.internal_height())));

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));

    let mut deferred_light_pass = DeferredLightPass::new(
        device,
        queue,
        camera_buf,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
    );
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(deferred_light_pass));

    // Water rendering passes (caustics → surface → underwater)
    // Add passes unconditionally so water appears when water volumes are inserted at runtime.
    graph.add_pass(Box::new(WaterCausticsPass::new(device)));
    graph.add_pass(Box::new(WaterSurfacePass::new(
        device,
        camera_buf,
        config.internal_width(),
        config.internal_height(),
    )));
    graph.add_pass(Box::new(UnderwaterPass::new(device, camera_buf)));

    let spotlight = image::load_from_memory(SPOTLIGHT_PNG)
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (sw, sh) = spotlight.dimensions();
    let mut billboard_pass = BillboardPass::new_with_sprite_rgba(
        device,
        queue,
        camera_buf,
        config.surface_format,
        spotlight.as_raw(),
        sw,
        sh,
    );
    billboard_pass.set_occluded_by_geometry(true);
    graph.add_pass(Box::new(billboard_pass));

    graph.add_pass(Box::new(TaaPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        config.width,
        config.height,
        config.surface_format,
    )));

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        false,
    )));

    graph.set_render_size(config.width, config.height);

    graph
}

pub fn build_hlfs_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<crate::renderer::debug::DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    _debug_depth_test: bool,
) -> RenderGraph {
    let gpu_scene = scene.gpu_scene();
    let mut graph = RenderGraph::new(device, queue);

    let camera_buf = gpu_scene.camera.buffer();

    let hiz_pass = HiZBuildPass::new(device, config.internal_width(), config.internal_height());
    let hiz_view = Arc::clone(&hiz_pass.hiz_view);
    let hiz_sampler = Arc::clone(&hiz_pass.hiz_sampler);

    let shadow_dirty_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Flags"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let shadow_hashes_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Hashes"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    graph.add_pass(Box::new(ShadowMatrixPass::new(
        device,
        gpu_scene.lights.buffer(),
        gpu_scene.shadow_matrices.buffer(),
        camera_buf,
        &shadow_dirty_buf,
        &shadow_hashes_buf,
    )));

    graph.add_pass(Box::new(ShadowPass::new(device)));

    let has_sky = scene.sky_context().has_sky;
    if has_sky {
        let sky_lut_pass = SkyLutPass::new(device, camera_buf);
        let sky_lut_view = sky_lut_pass.sky_lut_view.clone();
        graph.add_pass(Box::new(sky_lut_pass));

        graph.add_pass(Box::new(SkyPass::new(
            device,
            camera_buf,
            &sky_lut_view,
            config.internal_width(),
            config.internal_height(),
            config.surface_format,
        )));
    }

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        true,
    )));

    graph.add_pass(Box::new(OcclusionCullPass::new(device, hiz_view, hiz_sampler)));

    graph.add_pass(Box::new(DepthPrepassPass::new(device, wgpu::TextureFormat::Depth32Float)));

    graph.add_pass(Box::new(hiz_pass));

    graph.add_pass(Box::new(GBufferPass::new(device, config.internal_width(), config.internal_height())));

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));

    // ── HLFS Pass: Replaces traditional deferred lighting ──
    let mut hlfs_pass = HlfsPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        config.surface_format,
    );
    hlfs_pass.set_shadow_quality(config.shadow_quality, queue);
    graph.add_pass(Box::new(hlfs_pass));

    // Always include water passes in HLFS pipeline, enabling runtime insertion of water volumes.
    graph.add_pass(Box::new(WaterCausticsPass::new(device)));
    graph.add_pass(Box::new(WaterSurfacePass::new(
        device,
        camera_buf,
        config.internal_width(),
        config.internal_height(),
    )));
    graph.add_pass(Box::new(UnderwaterPass::new(device, camera_buf)));

    let spotlight = image::load_from_memory(SPOTLIGHT_PNG)
        .unwrap_or_else(|_| image::DynamicImage::new_rgba8(1, 1))
        .into_rgba8();
    let (sw, sh) = spotlight.dimensions();
    let mut billboard_pass = BillboardPass::new_with_sprite_rgba(
        device,
        queue,
        camera_buf,
        config.surface_format,
        spotlight.as_raw(),
        sw,
        sh,
    );
    billboard_pass.set_occluded_by_geometry(true);
    graph.add_pass(Box::new(billboard_pass));

    graph.add_pass(Box::new(TaaPass::new(
        device,
        config.internal_width(),
        config.internal_height(),
        config.width,
        config.height,
        config.surface_format,
    )));

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state.clone(),
        false,
        false,
    )));

    graph.set_render_size(config.width, config.height);

    graph
}

pub fn build_simple_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
) -> RenderGraph {
    let mut graph = RenderGraph::new(device, queue);
    graph.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));
    graph
}

pub fn create_depth_resources(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Helio Depth Texture"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
