use helio_core::{GpuScene, SceneInput, SceneResources};
use std::sync::{Arc, OnceLock};

/// Test-only equivalent of the frontend compatibility bridge. Keeping this
/// outside helio-core verifies that the core API has no concrete-scene impl.
pub struct SceneInputAdapter<'a>(pub &'a GpuScene);

impl SceneInput for SceneInputAdapter<'_> {
    fn device(&self) -> &Arc<wgpu::Device> {
        &self.0.device
    }
    fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.0.queue
    }
    fn frame_count(&self) -> u64 {
        self.0.frame_count
    }
    fn resources(&self) -> SceneResources<'_> {
        self.0.resources()
    }
    fn scene_buffers(&self) -> &helio_core::SceneBufferProjection {
        static EMPTY: OnceLock<helio_core::SceneBufferProjection> = OnceLock::new();
        EMPTY.get_or_init(helio_core::SceneBufferProjection::empty)
    }
}
