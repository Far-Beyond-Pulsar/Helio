use bytemuck::Zeroable;
use helio_core::{GpuCameraUniforms, SceneInput};
use std::sync::{Arc, OnceLock};

/// Test-only equivalent of the frontend compatibility bridge. Keeping this
/// outside helio-core verifies that the core API has no concrete-scene impl
/// -- `helio-core` has zero knowledge of any specific scene-object type, so
/// this adapter carries only the truly generic per-frame handles `SceneInput`
/// asks for (device/queue/frame count/camera); everything else a pass needs
/// comes through `scene_buffers`/`libhelio::FrameResources` instead.
pub struct SceneInputAdapter {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub frame_count: u64,
    camera_buf: wgpu::Buffer,
    camera_data: GpuCameraUniforms,
}

impl SceneInputAdapter {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let camera_data = GpuCameraUniforms::zeroed();
        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Test Camera Buffer"),
            size: std::mem::size_of::<GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            device,
            queue,
            frame_count: 0,
            camera_buf,
            camera_data,
        }
    }
}

impl SceneInput for SceneInputAdapter {
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
        &self.camera_buf
    }
    fn camera_data(&self) -> &GpuCameraUniforms {
        &self.camera_data
    }
    fn camera_generation(&self) -> u64 {
        0
    }
    fn scene_buffers(&self) -> &helio_core::SceneBufferProjection {
        static EMPTY: OnceLock<helio_core::SceneBufferProjection> = OnceLock::new();
        EMPTY.get_or_init(helio_core::SceneBufferProjection::empty)
    }
}
