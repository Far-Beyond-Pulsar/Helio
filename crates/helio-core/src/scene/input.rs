//! Generic scene input boundary used by graph execution.

use super::SceneResources;
use pulsar_scenedb::gpu::{BufferHandle, BufferKey, SceneGpuStore};
use std::sync::Arc;

/// Type-agnostic, read-only GPU buffers supplied by the frontend SceneDB.
/// Buffer contents are interpreted by the pass that declared the key; core
/// never needs to know the component type or layout.
#[derive(Clone, Default)]
pub struct SceneBufferProjection {
    entries: Vec<(BufferKey, BufferHandle)>,
}

impl SceneBufferProjection {
    pub fn empty() -> Self { Self::default() }

    pub fn from_store(store: &SceneGpuStore, keys: impl IntoIterator<Item = BufferKey>) -> Self {
        let entries = keys
            .into_iter()
            .filter_map(|key| store.resolve_buffer_handle(key).map(|buffer| (key, buffer)))
            .collect();
        Self { entries }
    }

    /// Snapshot every currently registered SceneDB buffer. Keys remain the
    /// only contract; this does not inspect or deserialize buffer contents.
    pub fn from_store_all(store: &SceneGpuStore) -> Self {
        Self::from_store(store, store.buffer_registry().keys())
    }

    pub fn get(&self, key: BufferKey) -> Option<&BufferHandle> {
        self.entries.iter().find(|(candidate, _)| *candidate == key).map(|(_, buffer)| buffer)
    }

    pub fn contains(&self, key: BufferKey) -> bool { self.get(key).is_some() }
}

/// Borrowed GPU scene input consumed by [`crate::RenderGraph`].
pub trait SceneInput {
    /// GPU device used to record this frame.
    fn device(&self) -> &Arc<wgpu::Device>;
    /// GPU queue used to submit this frame.
    fn queue(&self) -> &Arc<wgpu::Queue>;
    /// Monotonic frontend frame number.
    fn frame_count(&self) -> u64;
    /// Borrow the resource projection consumed by existing passes.
    fn resources(&self) -> SceneResources<'_>;
    /// Frontend-owned, type-erased SceneDB buffers. Every graph execution has
    /// this projection, even when a focused test intentionally supplies no
    /// keys. This keeps the graph on one SceneDB input path rather than
    /// retaining a legacy/no-SceneDB branch.
    fn scene_buffers(&self) -> &SceneBufferProjection;
}
