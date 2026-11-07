//! Generic scene input boundary used by graph execution.

use crate::GpuCameraUniforms;
use pulsar_scenedb::gpu::SceneGpuStore;
pub use pulsar_scenedb::gpu::{BufferHandle, BufferKey, FlushedRows};
use std::marker::PhantomData;
use std::sync::Arc;

/// The frontend's entity-generation buffer: one `u32` per entity index, the
/// entity's current generation (SceneDB's GPU liveness mirror). The
/// renderer publishes it under this key; consumers joining a row that
/// records another entity's `(index, generation)` check it here.
pub const ENTITY_GENERATIONS_KEY: BufferKey = BufferKey::of("world_entity_generations");

/// The current render view's camera uniform buffer, also exposed to scene
/// derivations that make per-view decisions (for example authored mesh LOD).
pub const CAMERA_BUFFER_KEY: BufferKey = BufferKey::of("scene_camera");

/// A GPU buffer the renderer reads: its key and the row layout its shaders
/// expect. Helio declares contracts and defines no components: whoever
/// produces the data (a SceneDB component an engine registers, or a scene
/// derivation) writes rows of `R` under the key
/// (docs/architecture/layered-rewrite.md, phase 1).
pub struct BufferContract<R: 'static> {
    key: &'static str,
    _row: PhantomData<fn() -> R>,
}

impl<R: 'static> BufferContract<R> {
    /// Bytes per row; a buffer published under the key must use this stride.
    pub const ROW_BYTES: u64 = std::mem::size_of::<R>() as u64;

    pub const fn new(key: &'static str) -> Self {
        Self {
            key,
            _row: PhantomData,
        }
    }

    pub const fn key(&self) -> BufferKey {
        BufferKey::of(self.key)
    }

    pub const fn name(&self) -> &'static str {
        self.key
    }

    /// Compile-time check for a producer: `layout` is a component or row
    /// stored under `key` (`const _: () = CONTRACT.assert_producer::<C>("k");`).
    /// Panics (a const-evaluation error) when the key or the size differ.
    pub const fn assert_producer<C>(&self, key: &str) {
        assert!(
            str_eq(self.key, key),
            "producer's buffer key differs from the contract's"
        );
        assert!(
            std::mem::size_of::<C>() == std::mem::size_of::<R>(),
            "producer's row size differs from the contract's"
        );
    }

    /// The buffer published under this contract, if any. A buffer whose row
    /// stride is not `R`'s is an error naming the contract, never a silent
    /// misread.
    pub fn resolve<'a>(
        &self,
        scene: &'a SceneBufferProjection,
    ) -> Result<Option<&'a BufferHandle>, ContractError> {
        match scene.get(self.key()) {
            Some(handle) if handle.row_bytes != 0 && handle.row_bytes != Self::ROW_BYTES => {
                Err(ContractError {
                    key: self.key,
                    expected_row_bytes: Self::ROW_BYTES,
                    found_row_bytes: handle.row_bytes,
                })
            }
            found => Ok(found),
        }
    }
}

impl<R: 'static> Clone for BufferContract<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R: 'static> Copy for BufferContract<R> {}

impl<R: 'static> std::fmt::Debug for BufferContract<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "BufferContract({:?}, {} B rows)",
            self.key,
            Self::ROW_BYTES
        )
    }
}

/// A variable-length pool the renderer reads: elements of `T` under `key`,
/// addressed by `(offset, count)` ranges a row of another contract carries
/// (mesh vertices and indices, portal chains).
pub type PoolContract<T> = BufferContract<T>;

/// A buffer published under a contract's key with a different row stride.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContractError {
    pub key: &'static str,
    pub expected_row_bytes: u64,
    pub found_row_bytes: u64,
}

impl std::fmt::Display for ContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "buffer `{}` has {}-byte rows; its contract expects {}",
            self.key, self.found_row_bytes, self.expected_row_bytes
        )
    }
}

impl std::error::Error for ContractError {}

/// CPU data the renderer reads, by key: the CPU half of a
/// [`BufferContract`], for what is not a GPU row (a voxel world, its pick
/// requests). Whoever owns the data (an engine's component) publishes an
/// `Arc<T>` under the key each frame; the feature that declared the
/// contract reads it. The type is the reader's, so a publisher of another
/// type is not seen.
pub struct PayloadContract<T: 'static> {
    key: &'static str,
    _payload: PhantomData<fn() -> T>,
}

impl<T: std::any::Any + Send + Sync> PayloadContract<T> {
    pub const fn new(key: &'static str) -> Self {
        Self {
            key,
            _payload: PhantomData,
        }
    }

    pub const fn key(&self) -> &'static str {
        self.key
    }
}

/// A frame's payloads ([`PayloadContract`]): what the host published, by
/// key. Cheap to clone (shared values).
#[derive(Clone, Default)]
pub struct ScenePayloads {
    entries: Vec<(&'static str, Arc<dyn std::any::Any + Send + Sync>)>,
}

impl ScenePayloads {
    /// Publish `value` under `key`, replacing what was there: for hosts
    /// that publish by key without naming the reader's type.
    pub fn insert_any(&mut self, key: &'static str, value: Arc<dyn std::any::Any + Send + Sync>) {
        match self.entries.iter_mut().find(|(k, _)| *k == key) {
            Some((_, existing)) => *existing = value,
            None => self.entries.push((key, value)),
        }
    }

    /// Publish `value` under `contract`.
    pub fn insert<T: std::any::Any + Send + Sync>(
        &mut self,
        contract: &PayloadContract<T>,
        value: Arc<T>,
    ) {
        self.insert_any(contract.key, value);
    }

    /// The payload published under `contract`, if it is of its type.
    pub fn get<T: std::any::Any + Send + Sync>(
        &self,
        contract: &PayloadContract<T>,
    ) -> Option<Arc<T>> {
        self.entries
            .iter()
            .find(|(k, _)| *k == contract.key)
            .and_then(|(_, value)| value.clone().downcast::<T>().ok())
    }

    /// Removes and returns the payload published under `contract`: a
    /// payload good for one frame.
    pub fn take<T: std::any::Any + Send + Sync>(
        &mut self,
        contract: &PayloadContract<T>,
    ) -> Option<Arc<T>> {
        let index = self.entries.iter().position(|(k, _)| *k == contract.key)?;
        self.entries
            .remove(index)
            .1
            .downcast::<T>()
            .ok()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

/// Type-agnostic, read-only GPU buffers supplied by the frontend SceneDB,
/// and the CPU payloads it published ([`PayloadContract`]). Contents are
/// interpreted by the pass that declared the key; core never needs to know
/// the component type or layout.
#[derive(Clone, Default)]
pub struct SceneBufferProjection {
    entries: Vec<(BufferKey, BufferHandle)>,
    /// The rows each dirty-tracked buffer received in its last mirror flush.
    flushed: Vec<(BufferKey, FlushedRows)>,
    payloads: ScenePayloads,
}

impl SceneBufferProjection {
    /// Construct a projection from frontend-owned GPU handles, including focused
    /// test fixtures. Handles are owned snapshots and survive buffer replacement.
    pub fn from_handles(entries: impl IntoIterator<Item = (BufferKey, BufferHandle)>) -> Self {
        Self {
            entries: entries.into_iter().collect(),
            flushed: Vec::new(),
            payloads: ScenePayloads::default(),
        }
    }

    pub fn empty() -> Self {
        Self::default()
    }
    pub fn from_store(store: &SceneGpuStore, keys: impl IntoIterator<Item = BufferKey>) -> Self {
        let mut entries = Vec::new();
        let mut flushed = Vec::new();
        for key in keys {
            if let Some(handle) = store.resolve_buffer_handle(key) {
                entries.push((key, handle));
                if let Some(rows) = store.flushed_rows(key) {
                    flushed.push((key, rows));
                }
            }
        }
        Self {
            entries,
            flushed,
            payloads: ScenePayloads::default(),
        }
    }

    /// The payload published under `contract` this frame.
    pub fn payload<T: std::any::Any + Send + Sync>(
        &self,
        contract: &PayloadContract<T>,
    ) -> Option<Arc<T>> {
        self.payloads.get(contract)
    }

    /// This frame's payloads, replacing the previous ones.
    pub fn set_payloads(&mut self, payloads: ScenePayloads) {
        self.payloads = payloads;
    }

    /// The rows `key`'s buffer received in its last mirror flush, when SceneDB
    /// kept them (dirty-tracked columns). A consumer that processed the
    /// previous `sequence` can visit only these rows instead of all of them.
    pub fn flushed_rows(&self, key: BufferKey) -> Option<&FlushedRows> {
        self.flushed
            .iter()
            .find(|(candidate, _)| *candidate == key)
            .map(|(_, rows)| rows)
    }

    /// Snapshot every currently registered SceneDB buffer. Keys remain the
    /// only contract; this does not inspect or deserialize buffer contents.
    pub fn from_store_all(store: &SceneGpuStore) -> Self {
        Self::from_store(store, store.buffer_registry().keys())
    }

    pub fn get(&self, key: BufferKey) -> Option<&BufferHandle> {
        self.entries
            .iter()
            .find(|(candidate, _)| *candidate == key)
            .map(|(_, buffer)| buffer)
    }

    pub fn contains(&self, key: BufferKey) -> bool {
        self.get(key).is_some()
    }

    /// Publish `handle` under `key`, replacing any buffer already there.
    /// The frontend adds buffers SceneDB keeps outside its store registry
    /// (the entity-generation mirror); scene derivations add their outputs.
    ///
    /// Replacing a buffer drops the flushed-row record kept for it: those
    /// rows described the frontend's buffer, not the one that now stands in
    /// for it (a derivation that rewrites a column, e.g. composing parented
    /// transforms, moves rows the flush never named).
    pub fn insert(&mut self, key: BufferKey, handle: BufferHandle) {
        match self
            .entries
            .iter_mut()
            .find(|(candidate, _)| *candidate == key)
        {
            Some((_, existing)) => {
                *existing = handle;
                self.flushed.retain(|(candidate, _)| *candidate != key);
            }
            None => self.entries.push((key, handle)),
        }
    }

    /// Record `rows` as the rows `key`'s buffer received this frame, for a
    /// buffer a derivation published (see `SceneDerivationOutput::flushed`).
    pub fn set_flushed_rows(&mut self, key: BufferKey, rows: FlushedRows) {
        self.flushed.retain(|(candidate, _)| *candidate != key);
        self.flushed.push((key, rows));
    }

    /// Changes whenever SceneDB reports new contents (`content_generation`)
    /// or a reallocation (`epoch`) in any buffer here -- "did anything in the
    /// scene change" with no GPU readback. Independent of entry order.
    ///
    /// Passes that reuse camera-dependent results while the camera holds
    /// still pair this with `camera_generation`: a still camera does not mean
    /// a still scene.
    pub fn content_signature(&self) -> u64 {
        self.entries.iter().fold(0u64, |sum, (_, handle)| {
            let mut h = handle.epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15);
            h ^= handle.content_generation.rotate_left(29);
            h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
            sum.wrapping_add(h ^ (h >> 31))
        })
    }
}

/// Borrowed GPU scene input consumed by [`crate::RenderGraph`].
pub trait SceneInput {
    /// GPU device used to record this frame.
    fn device(&self) -> &Arc<wgpu::Device>;
    /// GPU queue used to submit this frame.
    fn queue(&self) -> &Arc<wgpu::Queue>;
    /// Monotonic frontend frame number.
    fn frame_count(&self) -> u64;
    /// The active camera's GPU-bound uniform buffer. Camera is a genuine
    /// per-frame universal (exactly one per frame, needed by nearly every
    /// pass), not a specific scene-object type -- unlike the removed
    /// `SceneResources` type-bag, this is a first-class field for the same
    /// reason `frame_num`/`width`/`height` already are.
    fn camera(&self) -> &wgpu::Buffer;
    /// CPU-side mirror of the same frame's camera data, for passes that need
    /// scalar values (position, near/far, jitter, ...) alongside the buffer.
    fn camera_data(&self) -> &GpuCameraUniforms;
    /// See `helio_core::PassContext::camera_generation`'s doc.
    fn camera_generation(&self) -> u64;
    /// Frontend-owned, type-erased SceneDB buffers. Every graph execution has
    /// this projection, even when a focused test intentionally supplies no
    /// keys. This keeps the graph on one SceneDB input path rather than
    /// retaining a legacy/no-SceneDB branch.
    fn scene_buffers(&self) -> &SceneBufferProjection;
    /// Double-precision world position of this frame's coordinate origin.
    /// Frames are always camera-relative: the camera and every position in a
    /// pass row are relative to it (the frame's space), so f32 keeps its
    /// precision near the eye however far from the world's origin it is.
    /// Frontends write rows in that space (joins take the origin off f64 or
    /// f32-pair world positions). Planet centres are the one exception:
    /// world anchors, which the passes rebase by this origin.
    fn world_origin(&self) -> glam::DVec3;
}

#[cfg(test)]
mod payload_tests {
    use super::*;

    const ONE_FRAME: PayloadContract<u32> = PayloadContract::new("one_frame");
    const KEPT: PayloadContract<u32> = PayloadContract::new("kept");

    #[test]
    fn a_taken_payload_is_gone_and_the_others_stay() {
        let mut payloads = ScenePayloads::default();
        payloads.insert(&ONE_FRAME, Arc::new(7));
        payloads.insert(&KEPT, Arc::new(9));
        assert_eq!(payloads.take(&ONE_FRAME).as_deref(), Some(&7));
        assert!(payloads.take(&ONE_FRAME).is_none(), "good for one frame");
        assert_eq!(payloads.get(&KEPT).as_deref(), Some(&9));
    }
}
