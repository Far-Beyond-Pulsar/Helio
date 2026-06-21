use std::sync::Arc;

use helio_pass_fxaa::FxaaPass;
use helio_pass_hlfs::HlfsPass;
use helio_v3::RenderGraph;

use crate::scene::Scene;
use crate::renderer::debug::DebugDrawState;
use crate::renderer::config::RendererConfig;
use super::{add_common_early_passes, add_geometry_passes, add_late_passes, add_final_passes, new_graph};

/// Full-resolution pipeline with HLFS lighting and FXAA anti-aliasing.
///
/// Combines HLFS (O(1) shading cost relative to light count) with FXAA
/// (single-pass spatial AA). No temporal jitter or upscaling.
pub fn build_fxaa_hlfs_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
) -> RenderGraph {
    build_fxaa_hlfs_graph_internal(device, queue, scene, config, debug_state, debug_camera_buf, true)
}

pub fn build_fxaa_hlfs_graph_external(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
) -> RenderGraph {
    build_fxaa_hlfs_graph_internal(device, queue, scene, config, debug_state, debug_camera_buf, false)
}

fn build_fxaa_hlfs_graph_internal(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    scene: &Scene,
    config: RendererConfig,
    debug_state: Arc<std::sync::Mutex<DebugDrawState>>,
    debug_camera_buf: &wgpu::Buffer,
    owns_device: bool,
) -> RenderGraph {
    let w = config.width;
    let h = config.height;

    let mut graph = new_graph(device, queue, owns_device);

    let perf = add_common_early_passes(
        &mut graph, device, scene, &config, debug_state.clone(), debug_camera_buf, w, h,
    );

    add_geometry_passes(&mut graph, device, scene, &config, &perf, w, h);

    let mut hlfs_pass = HlfsPass::new(device, w, h, config.surface_format);
    hlfs_pass.set_shadow_quality(config.shadow_quality, queue);
    graph.add_pass(Box::new(hlfs_pass));

    add_late_passes(&mut graph, device, queue, scene, &config, &perf, w, h);

    graph.add_pass(Box::new(FxaaPass::new(device, config.surface_format)));

    add_final_passes(&mut graph, device, &config, &perf, debug_state, debug_camera_buf);

    graph.init_transients(w, h);
    graph
}
