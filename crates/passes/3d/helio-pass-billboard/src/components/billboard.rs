use std::marker::PhantomData;

use pulsar_scenedb::gpu::{BufferHandle, BufferKey, GpuMirrorHandle};
use pulsar_scenedb::{ComponentId, GpuColumnSet};
use pulsar_scenedb_derive::SceneStore;

/// The authoritative SceneDB record consumed by the billboard pass.
///
/// The packed layout is intentional: the shader reads the three fields as one
/// 48-byte instance record. `World::insert`/`get_mut` update this record and
/// SceneDB owns the deferred GPU upload.
#[derive(SceneStore, bytemuck::Pod, bytemuck::Zeroable, Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[gpu(layout = packed, buffer = "billboard_instances")]
pub struct BillboardComponent {
    #[gpu]
    pub world_pos: [f32; 4],
    #[gpu]
    pub scale_flags: [f32; 4],
    #[gpu]
    pub color: [f32; 4],
}

/// Generic contract between a pass-owned SceneDB record and a consumer.
///
/// The consumer only receives an owned buffer handle plus its allocation
/// epoch. It does not reach into SceneDB internals or require helio-core to
/// know the concrete component type. A changed epoch means a growable SceneDB
/// buffer was reallocated and the consumer must rebuild its bind group.
pub trait SceneGpuRecord: GpuColumnSet + Send + Sync + 'static {
    fn buffer_key() -> BufferKey;
    fn packed_component_id() -> ComponentId;
}

impl SceneGpuRecord for BillboardComponent {
    fn buffer_key() -> BufferKey {
        BufferKey::of("billboard_instances")
    }

    fn packed_component_id() -> ComponentId {
        Self::packed_gpu_component_id()
    }
}

/// A generic, re-resolvable SceneDB GPU input for a pass-owned record.
#[derive(Clone)]
pub struct SceneGpuBinding<T: SceneGpuRecord> {
    handle: BufferHandle,
    _record: PhantomData<T>,
}

impl<T: SceneGpuRecord> SceneGpuBinding<T> {
    /// Resolve the current buffer. `None` means the component has not been
    /// registered yet, which is a valid setup-time state.
    pub fn resolve(mirror: &GpuMirrorHandle) -> Option<Self> {
        mirror
            .store()
            .resolve_buffer_handle(T::buffer_key())
            .map(|handle| Self {
                handle,
                _record: PhantomData,
            })
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.handle.buffer
    }

    pub fn epoch(&self) -> u64 {
        self.handle.epoch
    }

    /// Produce the wgpu entry used by any bind group that consumes the record.
    pub fn bind_group_entry<'a>(&'a self, binding: u32) -> wgpu::BindGroupEntry<'a> {
        wgpu::BindGroupEntry {
            binding,
            resource: self.buffer().as_entire_binding(),
        }
    }
}

/// Concrete input type exported by this pass; its implementation remains
/// generic so other pass-owned SceneDB records can use the same contract.
pub type BillboardSceneBinding = SceneGpuBinding<BillboardComponent>;
