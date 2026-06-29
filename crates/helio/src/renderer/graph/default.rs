use std::sync::Arc;

use helio_pass_debug_overlay::DebugOverlayState;
use helio_pass_deferred_light::DeferredLightPass;
use helio_pass_light_cull::LightCullPass;
use helio_pass_perf_overlay::{PerfOverlayAnalyzerPass, PerfOverlayCostAnalyzerPass};
use helio_pass_taa::TaaPass;
use helio_v3::RenderGraph;

use crate::scene::Scene;
use crate::renderer::debug::DebugDrawState;
use crate::renderer::config::RendererConfig;
use super::{add_common_early_passes, add_geometry_passes, add_late_passes, add_final_passes, new_graph};

pub fn build_default_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_default_graph_internal(device, queue, scene, config, debug_state, debug_camera_buf, true, debug_overlay)
}

/// Same as `build_default_graph` but marks the graph as operating against an
/// externally-owned device.  Blocking `device.poll` calls are replaced with a
/// single non-blocking tick so Helio never races with the device owner's event
/// loop (e.g. GPUI's winit event loop).
pub fn build_default_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    build_default_graph_internal(device, queue, scene, config, debug_state, debug_camera_buf, false, debug_overlay)
}

fn build_default_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    owns_device: bool,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = new_graph(device, queue, owns_device);

    let perf = add_common_early_passes(
        &mut graph, device, scene, &config, debug_state.clone(), debug_camera_buf, iw, ih,
    );

    graph.add_pass(Box::new(LightCullPass::new(device, iw, ih)));

    add_geometry_passes(&mut graph, device, scene, &config, &perf);

    let camera_buf = scene.gpu_scene().camera.buffer();
    let mut deferred_light_pass = DeferredLightPass::new(
        device, queue, camera_buf, config.surface_format,
    );
    deferred_light_pass.set_shadow_quality(config.shadow_quality, queue);
    deferred_light_pass.debug_mode = config.debug_mode;
    graph.add_pass(Box::new(deferred_light_pass));
    graph.add_pass(Box::new(PerfOverlayCostAnalyzerPass::new(perf.clone())));
    graph.add_pass(Box::new(PerfOverlayAnalyzerPass::new(perf.clone())));

    add_late_passes(&mut graph, device, queue, scene, &config, &perf, iw, ih);

    graph.add_pass(Box::new(TaaPass::new(
        device, iw, ih, config.width, config.height, config.surface_format,
    )));

    add_final_passes(&mut graph, device, queue, &config, &perf, debug_state, debug_camera_buf, debug_overlay);

    graph.init_transients(iw, ih);
    graph
}
