use helio_core::{gpu, Scene, Camera, CameraGpuData};
use helio_lighting::LightingSystem;
use crate::{RendererConfig, RenderPath, FrameGraph, GBuffer, DeferredRenderer, ForwardRenderer};
use glam::Mat4;
use std::sync::Arc;

pub struct Renderer {
    pub config: RendererConfig,
    pub context: Arc<gpu::Context>,
    pub frame_graph: FrameGraph,
    pub gbuffer: Option<GBuffer>,
    pub deferred_renderer: Option<DeferredRenderer>,
    pub forward_renderer: Option<ForwardRenderer>,
    pub camera_buffer: Option<gpu::Buffer>,
    pub frame_index: u32,
    prev_view_proj: Mat4,
}

impl Renderer {
    pub fn new(context: Arc<gpu::Context>, config: RendererConfig) -> Self {
        let mut renderer = Self {
            config: config.clone(),
            context: context.clone(),
            frame_graph: FrameGraph::new(),
            gbuffer: None,
            deferred_renderer: None,
            forward_renderer: None,
            camera_buffer: None,
            frame_index: 0,
            prev_view_proj: Mat4::IDENTITY,
        };

        renderer.camera_buffer = Some(context.create_buffer(gpu::BufferDesc {
            name: "camera_data",
            size: std::mem::size_of::<CameraGpuData>() as u64,
            memory: gpu::Memory::Shared,
        }));

        match config.render_path {
            RenderPath::Deferred => {
                renderer.gbuffer = Some(GBuffer::new(&context, config.width, config.height));
                renderer.deferred_renderer = Some(DeferredRenderer::new(&context));
            }
            RenderPath::Forward | RenderPath::ForwardPlus => {
                renderer.forward_renderer = Some(ForwardRenderer::new(&context, &config));
            }
            RenderPath::VisibilityBuffer => {
                
            }
        }

        renderer
    }

    pub fn render(
        &mut self,
        scene: &Scene,
        lighting: &mut LightingSystem,
        target_view: gpu::TextureView,
    ) {
        self.update_camera_buffer(&scene.camera);
        lighting.update_gpu_data(&self.context);

        let mut encoder = self.context.create_command_encoder(gpu::CommandEncoderDesc {
            name: "frame",
            buffer_count: 1,
        });
        encoder.start();

        match self.config.render_path {
            RenderPath::Deferred => {
                if let (Some(gbuffer), Some(deferred)) = (&mut self.gbuffer, &mut self.deferred_renderer) {
                    gbuffer.clear(&mut encoder);
                    deferred.render(
                        &mut encoder,
                        scene,
                        lighting,
                        gbuffer,
                        target_view,
                        self.camera_buffer.unwrap(),
                    );
                }
            }
            RenderPath::Forward | RenderPath::ForwardPlus => {
                if let Some(forward) = &mut self.forward_renderer {
                    forward.render(
                        &mut encoder,
                        scene,
                        lighting,
                        target_view,
                        self.camera_buffer.unwrap(),
                    );
                }
            }
            RenderPath::VisibilityBuffer => {
                
            }
        }

        let sync_point = self.context.submit(&mut encoder);
        self.context.wait_for(&sync_point, !0);
        self.context.destroy_command_encoder(&mut encoder);

        self.frame_index += 1;
    }

    fn update_camera_buffer(&mut self, camera: &Camera) {
        if let Some(buffer) = self.camera_buffer {
            let view_proj = camera.view_projection_matrix();
            let _gpu_data = CameraGpuData::from_camera(
                camera,
                self.prev_view_proj,
                self.frame_index,
            );
            self.prev_view_proj = view_proj;

            // Buffer mapping will be handled by renderer implementation
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;

        if let Some(mut gbuffer) = self.gbuffer.take() {
            gbuffer.cleanup(&self.context);
            self.gbuffer = Some(GBuffer::new(&self.context, width, height));
        }
    }

    pub fn cleanup(&mut self) {
        if let Some(buffer) = self.camera_buffer.take() {
            self.context.destroy_buffer(buffer);
        }
        if let Some(mut gbuffer) = self.gbuffer.take() {
            gbuffer.cleanup(&self.context);
        }
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.cleanup();
    }
}
