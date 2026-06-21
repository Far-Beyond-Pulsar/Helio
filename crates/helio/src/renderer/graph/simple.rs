use std::sync::Arc;

use helio_pass_simple_cube::SimpleCubePass;
use helio_v3::RenderGraph;

pub fn build_simple_graph(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
    surface_format: wgpu::TextureFormat,
) -> RenderGraph {
    let mut graph = RenderGraph::new(device, queue);
    graph.add_pass(Box::new(SimpleCubePass::new(device, surface_format)));
    graph
}
