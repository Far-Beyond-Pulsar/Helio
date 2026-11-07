//! What Helio's own examples share: a SceneDB world with a GPU mirror (every
//! renderer reads the scene through one, even a graph that draws no scene
//! content), the per-frame flush, and the default renderer.
//!
//! Helio's examples exercise what Helio is without any object kind: a
//! custom graph and the debug draw. Scenes of meshes, lights and materials
//! are the meshes plugin's examples (Pulsar `plugins/vendor/meshes/examples`).

#![allow(dead_code)] // each example uses part of it

use helio::{Renderer, RendererBuilder, RendererConfig, SceneDbHandle};
use std::sync::Arc;

/// A SceneDB world with a GPU mirror on `device`, attached to the inspector.
pub fn new_scene_db_with_gpu_mirror(
    device: &Arc<wgpu::Device>,
    queue: &Arc<wgpu::Queue>,
) -> pulsar_scenedb::SceneDb {
    let mut scene_db = pulsar_scenedb::SceneDb::new();
    let ctx = pulsar_scenedb::gpu::EngineGpuContext::new(device.clone(), queue.clone());
    let gpu_store = pulsar_scenedb::gpu::SceneGpuStore::new(
        &ctx,
        pulsar_scenedb::gpu::SceneGpuConfig {
            classes: Vec::new(),
            tombstone_headroom: 0,
            max_cells_metadata: 0,
        },
    );
    let mirror = pulsar_scenedb::gpu::GpuMirrorHandle::new(Arc::new(gpu_store), queue.clone());
    scene_db.world.attach_gpu_mirror(mirror);
    scenedb_inspector_agent::install_world(&mut scene_db.world);
    scene_db
}

/// Frame-boundary SceneDB sync: uploads the rows written since the last
/// frame, then feeds an attached inspector.
pub fn flush_scene_db(
    scene_db: &pulsar_scenedb::SceneDb,
    queue: &wgpu::Queue,
) -> Option<pulsar_scenedb::gpu::SyncStats> {
    let stats = scene_db.world.flush_gpu_mirror(queue);
    scene_db.world.publish_inspector_snapshot();
    stats
}

/// The `SceneDbHandle` to pass to `RendererBuilder::new` for a world made by
/// [`new_scene_db_with_gpu_mirror`].
pub fn scene_db_handle(scene_db: &pulsar_scenedb::SceneDb) -> SceneDbHandle {
    scene_db
        .world
        .gpu_mirror()
        .cloned()
        .expect("new_scene_db_with_gpu_mirror always attaches a mirror")
}

/// The deferred preset on Helio's XR-aware device.
pub fn build_default_renderer_with_context(
    scene_db: &pulsar_scenedb::SceneDb,
    graphics: &mut helio::HelioGraphicsContext<'_>,
    config: RendererConfig,
) -> Renderer {
    let builder = RendererBuilder::new(config.clone(), scene_db_handle(scene_db))
        .with_pass_build_context(Box::new(helio_presets::deferred));
    graphics
        .build_renderer(builder, config)
        .expect("Helio renderer")
        .0
}
