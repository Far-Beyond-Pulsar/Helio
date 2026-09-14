//! Minimal `helio_core::SceneInput` adapter for the standalone 2D sprite
//! demos, which drive `helio_core::RenderGraph` directly and deliberately
//! bypass `helio::Renderer`/SceneDB entirely (see each demo's module docs).
//!
//! `RenderGraph::execute` takes `&dyn SceneInput` for its API shape only --
//! `SpriteBatchPass`/`SpriteCullPass` never read scene buffers or camera
//! data through it -- so this carries only the truly generic per-frame
//! handles the trait asks for and an empty `SceneBufferProjection`. This is
//! the example-crate twin of `helio-core`'s own test-only
//! `tests/support.rs` adapter.

use bytemuck::Zeroable;
use helio_core::{GpuCameraUniforms, SceneBufferProjection, SceneInput};
use std::sync::{Arc, OnceLock};

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
            label: Some("Sprite Demo Camera Buffer"),
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
    fn scene_buffers(&self) -> &SceneBufferProjection {
        static EMPTY: OnceLock<SceneBufferProjection> = OnceLock::new();
        EMPTY.get_or_init(SceneBufferProjection::empty)
    }
}
