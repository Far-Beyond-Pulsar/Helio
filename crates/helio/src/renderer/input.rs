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
}

impl<'a> SceneInputAdapter<'a> {
    pub(crate) fn from_scene_db(
        mirror: &'a GpuMirrorHandle,
        camera_buffer: &'a wgpu::Buffer,
        camera_data: &'a GpuCameraUniforms,
        camera_generation: u64,
        frame_count: u64,
    ) -> Self {
        Self {
            device: mirror.store().device_arc(),
            queue: Arc::new(mirror.queue().clone()),
            camera_buffer,
            camera_data,
            camera_generation,
            frame_count,
            buffers: SceneBufferProjection::from_store_all(mirror.store()),
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
}
