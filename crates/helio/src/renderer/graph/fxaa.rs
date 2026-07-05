use std::sync::Arc;

use helio_pass_debug_overlay::DebugOverlayState;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_fxaa::FxaaPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_perf_overlay::{PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass};
use helio_v3::RenderGraph;

use super::{
    add_common_early_passes, add_final_passes, add_geometry_passes, add_late_passes, new_graph,
};
use crate::renderer::config::RendererConfig;
use crate::renderer::debug::DebugDrawState;
use crate::scene::Scene;

/// Full-resolution pipeline with FXAA instead of TAA/TSR.
///
/// Renders everything at native display resolution (`render_scale = 1.0`)
/// and applies a single FXAA post-process pass for anti-aliasing.
pub fn build_fxaa_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_fxaa_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        true,
        debug_overlay,
    )
}

pub fn build_fxaa_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_fxaa_graph_internal(
        device,
        queue,
        scene,
        config,
        debug_state,
        debug_camera_buf,
        cull_stats_buf,
        false,
        debug_overlay,
    )
}

fn build_fxaa_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    cull_stats_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    let w = config.width;
    let h = config.height;

    let mut graph = new_graph(device, queue, owns_device);

    let perf = add_common_early_passes(
        &mut graph,
        device,
        scene,
        &config,
        debug_state.clone(),
        debug_camera_buf,
        cull_stats_buf,
        w,
        h,
    );

    graph.add_pass(Box::new(LightCullPass::new(device, w, h)));

    add_geometry_passes(&mut graph, device, scene, &config, &perf);

    let camera_buf = scene.gpu_scene().camera.buffer();
    let mut deferred_light_pass =
        DeferredLightPass::new(device, queue, camera_buf, config.surface_format);
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(deferred_light_pass));
    graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(Arc::clone(
        &perf,
    ))));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(Arc::clone(&perf))));

    add_late_passes(&mut graph, device, queue, scene, &config, &perf, w, h);

    graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));

    add_final_passes(
        &mut graph,
        device,
        queue,
        &config,
        &perf,
        debug_state,
        debug_camera_buf,
        debug_overlay,
    );

    graph.init_transients(w, h);
    graph
}
