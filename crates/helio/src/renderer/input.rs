//! Renderer-side implementation of the generic SceneInput boundary.

use helio_core::{GpuCameraUniforms, SceneBufferProjection, SceneInput};
use pulsar_scenedb::gpu::GpuMirrorHandle;
use std::sync::Arc;

/// Read-only frame input assembled from the frontend SceneDB GPU mirror.
///
/// This type intentionally has no typed scene fields. Component buffers are
/// discovered and consumed by passes through `BufferKey`; camera is the only
/// universal render input carried explicitly by the core contract.
pub(crate) struct SceneInputAdapter<'a> {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    camera_buffer: &'a wgpu::Buffer,
    camera_data: &'a GpuCameraUniforms,
    camera_generation: u64,
    frame_count: u64,
    buffers: SceneBufferProjection,
    world_origin: Option<glam::DVec3>,
}

impl<'a> SceneInputAdapter<'a> {
    pub(crate) fn from_scene_db(
        mirror: &'a GpuMirrorHandle,
        camera_buffer: &'a wgpu::Buffer,
        camera_data: &'a GpuCameraUniforms,
        camera_generation: u64,
        frame_count: u64,
        world_origin: Option<glam::DVec3>,
    ) -> Self {
        Self {
            device: mirror.store().device_arc(),
            queue: Arc::new(mirror.queue().clone()),
            camera_buffer,
            camera_data,
            camera_generation,
            frame_count,
            buffers: {
                let mut buffers = SceneBufferProjection::from_store_all(mirror.store());
                // The entity-generation mirror lives on the handle, not in the
                // store's registry; consumers joining rows across entities
                // check recorded generations against it.
                let generations = mirror.generations();
                let mut buffer = None;
                generations.with_buffer(&mut |b| buffer = Some(b.clone()));
                if let Some(buffer) = buffer {
                    buffers.insert(
                        helio_core::ENTITY_GENERATIONS_KEY,
                        helio_core::BufferHandle {
                            buffer,
                            epoch: generations.epoch(),
                            row_bytes: 4,
                            // The mirror reports reallocation, not uploads.
                            content_generation: generations.epoch(),
                        },
                    );
                }
                buffers
            },
            world_origin,
        }
    }

    /// Run the renderer's scene derivations, publishing their outputs into
    /// this frame's scene buffers, and submit their work ahead of the graph.
    pub(crate) fn run_derivations(&mut self, derivations: &mut [Box<dyn helio_core::SceneDerivation>]) {
        if let Some(commands) = helio_core::run_scene_derivations(
            derivations,
            &self.device,
            &self.queue,
            &mut self.buffers,
        ) {
            self.queue.submit(std::iter::once(commands));
        }
    }
}

impl SceneInput for SceneInputAdapter<'_> {
    fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }

    fn frame_count(&self) -> u64 {
        self.frame_count
    }

    fn camera(&self) -> &wgpu::Buffer {
        self.camera_buffer
    }

    fn camera_data(&self) -> &GpuCameraUniforms {
        self.camera_data
    }

    fn camera_generation(&self) -> u64 {
        self.camera_generation
    }

    fn scene_buffers(&self) -> &SceneBufferProjection {
        &self.buffers
    }

    fn world_origin(&self) -> Option<glam::DVec3> {
        self.world_origin
    }
}
