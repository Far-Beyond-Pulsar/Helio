//! The renderer's read-only scene projection boundary.

use helio_core::{SceneBufferProjection, SceneInput, SceneResources};
use pulsar_scenedb::gpu::GpuMirrorHandle;
use std::sync::Arc;

use super::Scene;

/// Read-only GPU projection supplied by the frontend-owned SceneDB.
///
/// Frame-produced resources are deliberately not part of this value; they are
/// passed independently to `RenderGraph::execute_with_resources`.
pub struct SceneDbProjection<'a> {
    mirror: &'a GpuMirrorHandle,
    transient_resources: SceneResources<'a>,
    frame_count: u64,
    queue: &'a Arc<wgpu::Queue>,
    buffers: SceneBufferProjection,
}

impl<'a> SceneDbProjection<'a> {
    /// `transient_scene` supplies only the current frame's derived resources
    /// while `mirror` supplies every persistent, entity-stable GPU column.
    /// The names are intentionally explicit: no caller may mistake the
    /// renderer-local frame bundle for the authoritative scene database.
    pub(crate) fn new(transient_scene: &'a Scene, mirror: &'a GpuMirrorHandle) -> Self {
        Self {
            mirror,
            transient_resources: transient_scene.gpu_scene().resources(),
            frame_count: transient_scene.gpu_scene().frame_count,
            queue: &transient_scene.gpu_scene().queue,
            buffers: SceneBufferProjection::from_store_all(mirror.store()),
        }
    }

    fn device(&self) -> Arc<wgpu::Device> { self.mirror.store().device_arc() }
}

/// Graph input backed by the SceneDB projection.
pub struct SceneInputAdapter<'a> {
    device: Arc<wgpu::Device>,
    queue: &'a Arc<wgpu::Queue>,
    projection: SceneResources<'a>,
    frame_count: u64,
    buffers: SceneBufferProjection,
}

impl<'a> SceneInputAdapter<'a> {
    pub(crate) fn from_scene_db(projection: SceneDbProjection<'a>) -> Self {
        Self {
            device: projection.device(),
            queue: projection.queue,
            projection: projection.transient_resources,
            frame_count: projection.frame_count,
            buffers: projection.buffers,
        }
    }

}

impl SceneInput for SceneInputAdapter<'_> {
    fn device(&self) -> &Arc<wgpu::Device> { &self.device }
    fn queue(&self) -> &Arc<wgpu::Queue> { self.queue }
    fn frame_count(&self) -> u64 { self.frame_count }
    fn resources(&self) -> SceneResources<'_> { self.projection }
    fn scene_buffers(&self) -> &SceneBufferProjection { &self.buffers }
}
