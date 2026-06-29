use std::sync::Arc;

use helio_pass_debug_overlay::DebugOverlayState;
use helio_pass_hlfs::HlfsPass;
use helio_pass_perf_overlay::PerfOverlayAnalyzerPass;
use helio_pass_taa::TaaPass;
use helio_v3::RenderGraph;

use crate::scene::Scene;
use crate::renderer::debug::DebugDrawState;
use crate::renderer::config::RendererConfig;
use super::{add_common_early_passes, add_geometry_passes, add_late_passes, add_final_passes};

pub fn build_hlfs_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    debug_overlay: Option<&Arc<std::sync::Mutex<DebugOverlayState>>>,
) -> RenderGraph {
    let iw = config.internal_width();
    let ih = config.internal_height();

    let mut graph = RenderGraph::new(device, queue);

    let perf = add_common_early_passes(
        &mut graph, device, scene, &config, debug_state.clone(), debug_camera_buf, iw, ih,
    );

    add_geometry_passes(&mut graph, device, scene, &config, &perf);

    let mut hlfs_pass = HlfsPass::new(device, iw, ih, config.surface_format);
    hlfs_pass.set_shadow_quality(config.shadow_quality, queue);
    graph.add_pass(Box::new(hlfs_pass));

    add_late_passes(&mut graph, device, queue, scene, &config, &perf, iw, ih);

    graph.add_pass(Box::new(TaaPass::new(
        device, iw, ih, config.width, config.height, config.surface_format,
    )));

    add_final_passes(&mut graph, device, queue, &config, &perf, debug_state, debug_camera_buf, debug_overlay);

    graph.init_transients(iw, ih);
    graph
}
