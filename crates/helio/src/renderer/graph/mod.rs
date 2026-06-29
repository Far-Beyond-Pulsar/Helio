mod default;
mod fxaa;
mod fxaa_hlfs;
mod hlfs;
mod simple;

pub use default::{build_default_graph, build_default_graph_external};
pub use fxaa::{build_fxaa_graph, build_fxaa_graph_external};
pub use fxaa_hlfs::{build_fxaa_hlfs_graph, build_fxaa_hlfs_graph_external};
pub use hlfs::build_hlfs_graph;
pub use simple::build_simple_graph;

use std::sync::Arc;

use helio_pass_billboard::BillboardPass;
use helio_pass_corona::CoronaPass;
use helio_pass_hiz::HiZBuildPass;
use helio_pass_indirect_dispatch::IndirectDispatchPass;
use helio_pass_occlusion_cull::OcclusionCullPass;
use helio_pass_gbuffer::GBufferPass;
use helio_pass_shadow::ShadowPass;
use helio_pass_shadow_cull::ShadowCullPass;
use helio_pass_shadow_dirty::ShadowDirtyPass;
use helio_pass_shadow_matrix::ShadowMatrixPass;
use helio_pass_sky_lut::SkyLutPass;
use helio_pass_sky::SkyPass;
use helio_pass_virtual_geometry::VirtualGeometryPass;
use helio_pass_debug_overlay::{DebugOverlayPass, DebugOverlayState};
use helio_pass_perf_overlay::{PerfOverlayAnalyzerPass, PerfOverlayPass, PerfOverlayShared};
use helio_pass_water_sim::WaterSimPass;
use helio_v3::RenderGraph;

use crate::scene::Scene;
use crate::renderer::debug::{DebugDrawPass, DebugDrawState};
use crate::renderer::config::RendererConfig;

/// Spotlight icon embedded at compile time — used as the editor billboard sprite.
static SPOTLIGHT_PNG: &[u8] = include_bytes!("../../../../../spotlight.png");

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
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Shared early-pipeline setup: shadow passes, sky, debug draw (pre-geometry),
/// indirect dispatch, occlusion cull, HiZ, and perf overlay analyzer.
///
/// Returns `(perf_overlay_shared,)` for use by later pipeline stages.
/// `cull_stats_buf` is a storage buffer with 8 atomic u32 counters for culling statistics,
/// updated by IndirectDispatchPass and OcclusionCullPass. The same buffer handle should be
/// used by the renderer for readback via a staging buffer (COPY_DST | MAP_READ).
fn add_common_early_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    scene: &Scene,
    config: &RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    w: u32,
    h: u32,
) -> Arc<std::sync::Mutex<PerfOverlayShared>> {
    let gpu_scene = scene.gpu_scene();
    let camera_buf = gpu_scene.camera.buffer();

    let hiz_pass = HiZBuildPass::new(device, w, h);
    let hiz_sampler = Arc::clone(&hiz_pass.hiz_sampler);

    let shadow_dirty_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Shadow Dirty Flags"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    }));
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

    let shadow_dirty_pass = ShadowDirtyPass::new(device);
    let face_dirty_buf = Arc::clone(&shadow_dirty_pass.face_dirty_buf);
    let face_geom_count_buf = Arc::clone(&shadow_dirty_pass.face_geom_count_buf);
    graph.add_pass(Box::new(shadow_dirty_pass));

    // ShadowCullPass: per-face frustum culling for shadow geometry
    let shadow_cull_pass = ShadowCullPass::new(device, Arc::clone(&face_dirty_buf));
    let face_cull_indirect = Arc::clone(&shadow_cull_pass.face_indirect_buf);
    let face_cull_counts = Arc::clone(&shadow_cull_pass.face_counts_buf);
    graph.add_pass(Box::new(shadow_cull_pass));

    graph.add_pass(Box::new(ShadowPass::new(
        device,
        face_dirty_buf,
        face_geom_count_buf,
        face_cull_indirect,
        face_cull_counts,
        config.shadow_atlas_size,
    )));

    if scene.sky_context().has_sky {
        graph.add_pass(Box::new(SkyLutPass::new(device, camera_buf)));

        graph.add_pass(Box::new(SkyPass::new(
            device,
            camera_buf,
            config.surface_format,
        )));
    }

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state,
        false,
        true,
    )));

    graph.add_pass(Box::new(IndirectDispatchPass::new(
        device,
        cull_stats_buf.clone(),
    )));
    graph.add_pass(Box::new(hiz_pass));
    let mut occlusion_cull = OcclusionCullPass::new(
        device,
        hiz_sampler,
        w,
        h,
        cull_stats_buf.clone(),
    );
    // Wire static HiZ metadata from the baked voxel grid to the occlusion pass.
    if let Some(meta) = graph.find_pass::<HiZBuildPass>()
        .and_then(|p| p.static_hiz_metadata())
    {
        occlusion_cull.set_static_hiz_metadata(
            meta.world_bounds_min,
            meta.world_bounds_max,
            meta.grid_resolution,
        );
    }
    graph.add_pass(Box::new(occlusion_cull));

    let perf_overlay_shared = PerfOverlayShared::new(device, w, h);
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf_overlay_shared))));

    perf_overlay_shared
}

/// Shared geometry passes: GBuffer + VirtualGeometry.
fn add_geometry_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    scene: &Scene,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
) {
    let camera_buf = scene.gpu_scene().camera.buffer();

    graph.add_pass(Box::new(GBufferPass::new(device)));

    let mut vg_pass = VirtualGeometryPass::new(device, camera_buf);
    vg_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(vg_pass));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));
}

/// Shared late-pipeline passes: billboard, water sim.
fn add_late_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    w: u32,
    h: u32,
) {
    let camera_buf = scene.gpu_scene().camera.buffer();

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
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    graph.add_pass(Box::new(CoronaPass::new(
        device,
        queue,
        camera_buf,
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    graph.add_pass(Box::new(WaterSimPass::new(
        device,
        camera_buf,
        w,
        h,
        config.surface_format,
    )));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));
}

/// Shared final passes: perf overlay, debug draw, and debug info overlay.
/// If `debug_overlay` is `Some`, a `DebugOverlayPass` is appended at the end.
fn add_final_passes(
    graph: &mut RenderGraph,
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    config: &RendererConfig,
    perf: &Arc<std::sync::Mutex<PerfOverlayShared>>,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) {
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(perf))));

    let mut perf_overlay_pass = PerfOverlayPass::new(
        device,
        Arc::clone(perf),
        config.surface_format,
    );
    perf_overlay_pass.set_mode(config.perf_overlay_mode);
    graph.add_pass(Box::new(perf_overlay_pass));

    graph.add_pass(Box::new(DebugDrawPass::new(
        device,
        debug_camera_buf,
        config.surface_format,
        debug_state,
        false,
        false,
    )));

    if let Some(shared) = debug_overlay {
        graph.add_pass(Box::new(DebugOverlayPass::new(
            device,
            queue,
            Arc::clone(shared),
            config.surface_format,
            config.width,
            config.height,
        )));
    }
}

fn new_graph(device: &Arc<wgpu::Device>, queue: &Arc<wgpu::Queue>, owns_device: bool) -> RenderGraph {
    if owns_device {
        RenderGraph::new(device, queue)
    } else {
        RenderGraph::new_with_external_device(device, queue)
    }
}
