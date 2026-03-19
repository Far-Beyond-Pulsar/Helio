use std::sync::Arc;

use helio_v3::{RenderGraph, RenderPass, Result as HelioResult};

use crate::mesh::MeshBuffers;
use crate::scene::Scene;

#[derive(Debug, Clone, Copy)]
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
}

impl RendererConfig {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

pub struct Renderer {
    graph: RenderGraph,
    scene: Scene,
}

impl Renderer {
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        config: RendererConfig,
    ) -> Self {
        let mut scene = Scene::new(device.clone(), queue.clone());
        scene.set_render_size(config.width, config.height);
        Self {
            graph: RenderGraph::new(&device, &queue),
            scene,
        }
    }

    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    pub fn mesh_buffers(&self) -> MeshBuffers<'_> {
        self.scene.mesh_buffers()
    }

    pub fn add_pass(&mut self, pass: Box<dyn RenderPass>) {
        self.graph.add_pass(pass);
    }

    pub fn set_render_size(&mut self, width: u32, height: u32) {
        self.scene.set_render_size(width, height);
    }

    pub fn render(
        &mut self,
        target: &wgpu::TextureView,
        depth: &wgpu::TextureView,
    ) -> HelioResult<()> {
        self.scene.flush();
        self.graph.execute(self.scene.gpu_scene(), target, depth)?;
        self.scene.advance_frame();
        Ok(())
    }
}
