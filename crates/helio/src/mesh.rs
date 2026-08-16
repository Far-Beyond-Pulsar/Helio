use bytemuck::{Pod, Zeroable};
use pulsar_scenedb::gpu::{VarLenBufferRef, VarLenGpuPool, VarLenHandle};
use std::sync::Arc;

use crate::arena::SparsePool;
use crate::handles::MeshId;

/// Determines the lifetime and update policy of mesh geometry on the GPU.
///
/// | Kind    | Can update geometry? | CPU mirror retained? | Use case |
/// |---------|---------------------|----------------------|----------|
/// | Static  | No (upload-once)    | Yes (baking)         | Buildings, terrain, props |
/// | Dynamic | Yes (per-frame OK)  | Yes (dirty tracking) | Skinned characters, morphs, procedural |
///
/// Objects that **move** but keep their shape (rigid bodies) use `MeshKind::Static`
/// geometry combined with `Movability::Movable` on the object. Transform updates
/// go through `update_object_transform()` which is O(1) and never touches mesh data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeshKind {
    /// Geometry is uploaded once and never changed.
    Static,
    /// Geometry can be replaced per-frame via [`MeshPool::update_dynamic_vertices`].
    Dynamic,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, Pod, Zeroable)]
pub struct PackedVertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords0: [f32; 2],
    pub tex_coords1: [f32; 2],
    pub normal: u32,
    pub tangent: u32,
}

// SAFETY: #[repr(C)], every field is itself Pod (f32/u32 arrays and
// scalars), no padding, every bit pattern valid -- the exact same guarantee
// already vouched for by this struct's `bytemuck::Pod` derive above.
// `pulsar_scenedb::Pod` isn't bytemuck's trait (SceneDB defines its own
// marker so its GPU column machinery doesn't take a hard bytemuck
// dependency at that layer) -- both need a real, separate impl, this one
// hand-written since there's no derive for it. This is what lets
// `VarLenGpuPool<PackedVertex>` exist at all (`VarLenGpuPool<T>` bounds
// `T: pulsar_scenedb::Pod`, not `T: bytemuck::Pod` -- see `MeshSubPool`,
// which stores its vertex pool as exactly this type).
unsafe impl pulsar_scenedb::Pod for PackedVertex {}

impl PackedVertex {
    pub fn from_components(
        position: [f32; 3],
        normal: [f32; 3],
        tex_coords: [f32; 2],
        tangent: [f32; 3],
        bitangent_sign: f32,
    ) -> Self {
        Self {
            position,
            bitangent_sign,
            tex_coords0: tex_coords,
            tex_coords1: [0.0, 0.0],
            normal: pack_snorm4x8([normal[0], normal[1], normal[2], 0.0]),
            tangent: pack_snorm4x8([tangent[0], tangent[1], tangent[2], 0.0]),
        }
    }
}

fn pack_snorm4x8(v: [f32; 4]) -> u32 {
    let to_i8 = |x: f32| -> u32 {
        let clamped = x.clamp(-1.0, 1.0);
        let scaled = (clamped * 127.0).round() as i8;
        scaled as u8 as u32
    };

    to_i8(v[0]) | (to_i8(v[1]) << 8) | (to_i8(v[2]) << 16) | (to_i8(v[3]) << 24)
}

#[derive(Debug, Clone)]
pub struct MeshUpload {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
}

/// Upload descriptor for a multi-material (sectioned) mesh.
///
/// All sections share one vertex buffer. Each element of `sections` is an independent
/// index list referencing `vertices`, rendered with its own material per draw call.
/// This mirrors Unreal Engine's Static Mesh section model: one VB/IB, N draw calls.
#[derive(Debug, Clone)]
pub struct SectionedMeshUpload {
    /// The full shared vertex array. All sections index into this.
    pub vertices: Vec<PackedVertex>,
    /// Per-section index lists. `sections[i]` is drawn with the i-th material.
    pub sections: Vec<Vec<u32>>,
}

/// Internal record for a stored multi-material mesh.
/// Sections share the same vertex buffer region but have distinct index ranges.
pub(crate) struct MultiMeshRecord {
    /// One `MeshId` per section (all share the same vertex range in the pool).
    pub section_mesh_ids: Vec<crate::handles::MeshId>,
    /// Number of live [`SectionedObjectId`] instances placed from this mesh.
    pub ref_count: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct MeshSlice {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}

#[derive(Debug, Clone)]
pub(crate) struct MeshRecord {
    pub slice: MeshSlice,
    pub ref_count: u32,
    pub kind: MeshKind,
}

pub struct MeshBuffers<'a> {
    pub vertices: VarLenBufferRef<'a, PackedVertex>,
    pub indices: VarLenBufferRef<'a, u32>,
}

// ── Sub-pool (vertex + index, backed by SceneDB's shared growable pool) ───────
//
// Vertex/index storage delegates entirely to `pulsar_scenedb::gpu::
// VarLenGpuPool` (a growable GPU buffer + `RangeList` freelist suballocator)
// instead of a hand-rolled `GrowableBuffer` + free-list allocator pair --
// those two were an independent reimplementation of exactly what
// `VarLenGpuPool` already is (same element types, same grow-and-suballocate
// shape). `Arc`-wrapped so this same pool instance can later be registered
// into a `SceneGpuStore` (Pulsar-Native#561 Phase D: a `StaticMeshComponent`'s
// own `#[gpu] Vec<PackedVertex>`/`Vec<u32>` fields write into THIS pool
// directly -- see `MeshPool::vertex_pool`/`index_pool` -- so its hydrate-time
// write and Helio's draw calls read the exact same buffer, zero translation).
struct MeshSubPool {
    vertices: Arc<VarLenGpuPool<PackedVertex>>,
    indices: Arc<VarLenGpuPool<u32>>,
    // `VarLenGpuPool` writes synchronously (`queue.write_buffer` on every
    // call) instead of staging into a CPU mirror for a later batched flush
    // the way `GrowableBuffer` did -- so every write needs a `&wgpu::Queue`
    // at the call site. Stored here (captured once at construction, from
    // the SAME `queue` `Scene::new` already receives) specifically so
    // `MeshPool::insert`/`insert_dynamic`/`insert_sectioned`/
    // `update_dynamic_vertices` keep their EXACT existing signatures --
    // none of them took a `queue` parameter before (that's what
    // `Scene::flush(queue)` was for), and threading one through now would
    // ripple into every caller across the whole Helio consumer ecosystem
    // (`Scene::insert_actor`/`insert_mesh` alone have dozens of call
    // sites). This keeps that ripple to zero.
    queue: Arc<wgpu::Queue>,
    // VarLenGpuPool has no CPU mirror to derive a "how many elements are
    // actually live right now" count from (by design -- it's a pure
    // GPU-resident store, see its own module doc). `stats.rs`'s debug HUD
    // wants that number, so it's tracked directly here instead.
    live_vertex_count: usize,
    live_index_count: usize,
}

impl MeshSubPool {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, kind: MeshKind) -> Self {
        let (v_label, i_label, v_cap, i_cap) = match kind {
            MeshKind::Static => (
                "Helio Static Vertex Buffer",
                "Helio Static Index Buffer",
                4096,
                8192,
            ),
            MeshKind::Dynamic => (
                "Helio Dynamic Vertex Buffer",
                "Helio Dynamic Index Buffer",
                512,
                1024,
            ),
        };
        Self {
            vertices: Arc::new(VarLenGpuPool::new_with_usage(
                device.clone(),
                v_label,
                v_cap,
                wgpu::BufferUsages::VERTEX,
            )),
            indices: Arc::new(VarLenGpuPool::new_with_usage(
                device,
                i_label,
                i_cap,
                wgpu::BufferUsages::INDEX,
            )),
            queue,
            live_vertex_count: 0,
            live_index_count: 0,
        }
    }

    /// Allocate and write `vertices`/`indices` as brand-new data (no
    /// previous allocation to free first -- every `MeshPool::insert*` call
    /// is a fresh mesh, never a re-write of an existing one; compare
    /// [`Self::rewrite_dynamic_vertices`], which IS a re-write).
    ///
    /// Returns the two handles the caller stores in the `MeshRecord`'s
    /// `MeshSlice`. Panics (via `expect`) only if the device's own
    /// `max_buffer_size` is exhausted -- the same ultimately-unrecoverable
    /// condition the old `GrowableBuffer`-based pool would have hit as an
    /// uncaught `wgpu` validation panic; this is not a new failure mode,
    /// just a clearer message at the same point.
    fn alloc_and_write(
        &mut self,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> (VarLenHandle, VarLenHandle) {
        let vhandle = self
            .vertices
            .write_var_row(&self.queue, VarLenHandle::default(), vertices)
            .expect("mesh vertex pool exhausted the device's max_buffer_size");
        let ihandle = self
            .indices
            .write_var_row(&self.queue, VarLenHandle::default(), indices)
            .expect("mesh index pool exhausted the device's max_buffer_size");
        self.live_vertex_count += vertices.len();
        self.live_index_count += indices.len();
        (vhandle, ihandle)
    }

    /// In-place, offset-preserving overwrite of a dynamic mesh's vertex data
    /// -- deliberately NOT a free-then-reallocate cycle through
    /// `write_var_row`. A draw call's `GpuDrawCall.vertex_offset` is baked in
    /// once at object-creation time and never re-read per frame, so if this
    /// silently moved the data to a different offset (which
    /// `VarLenGpuPool::write_var_row`'s free+realloc would risk, even though
    /// same-size reuse is likely in practice, it's not a guaranteed contract)
    /// every previously-created draw call referencing this mesh would start
    /// reading the wrong vertices. Requires `new_vertices.len()` to match the
    /// handle's existing `count` exactly -- same constraint the old
    /// `GrowableBuffer::update_range` path enforced via its caller
    /// (`MeshPool::update_dynamic_vertices`).
    fn rewrite_dynamic_vertices(&self, handle: VarLenHandle, new_vertices: &[PackedVertex]) {
        debug_assert_eq!(handle.count as usize, new_vertices.len(), "in-place rewrite must not change element count");
        self.vertices.write_at_offset(&self.queue, handle.offset, new_vertices);
    }

    /// Return the vertex and index ranges of `slice` to the pool's freelist.
    ///
    /// Unlike the old `FreeListAllocator`, `VarLenGpuPool` never shrinks its
    /// backing buffer back down on a tail-free (it has no `truncate`/
    /// `shrink_to_fit` -- see its own module doc: purely grow-only). The
    /// freed range stays tracked and fully reusable by a later allocation;
    /// only the "proactively give VRAM back after a big deletion" behavior
    /// is gone, not correctness. Acceptable trade-off for de-duplicating two
    /// copies of the same allocator, not something masked or silently lost.
    fn free_slice(&mut self, slice: &MeshSlice) {
        self.vertices.free_handle(VarLenHandle { offset: slice.first_vertex, count: slice.vertex_count });
        self.indices.free_handle(VarLenHandle { offset: slice.first_index, count: slice.index_count });
        self.live_vertex_count -= slice.vertex_count as usize;
        self.live_index_count -= slice.index_count as usize;
    }

    fn buffers(&self) -> MeshBuffers<'_> {
        MeshBuffers {
            vertices: self.vertices.read_buffer(),
            indices: self.indices.read_buffer(),
        }
    }

    /// No-op: `VarLenGpuPool::write_var_row`/`write_at_offset` upload
    /// synchronously (`queue.write_buffer`) on every call -- there's no
    /// separate dirty-CPU-mirror step left to flush. Kept only so
    /// `MeshPool::flush` doesn't need to change shape for its caller.
    fn flush(&mut self, _queue: &wgpu::Queue) {}
}

// ── Public MeshPool ───────────────────────────────────────────────────────────

pub struct MeshPool {
    static_sub: MeshSubPool,
    dynamic_sub: MeshSubPool,
    meshes: SparsePool<MeshRecord, MeshId>,
}

impl MeshPool {
    pub fn new(device: std::sync::Arc<wgpu::Device>, queue: std::sync::Arc<wgpu::Queue>) -> Self {
        Self {
            static_sub: MeshSubPool::new(device.clone(), queue.clone(), MeshKind::Static),
            dynamic_sub: MeshSubPool::new(device, queue, MeshKind::Dynamic),
            meshes: SparsePool::new(),
        }
    }

    /// The static sub-pool's shared vertex storage. `Arc`-cloned, not
    /// borrowed: the intended caller is a `SceneGpuStore` registration
    /// (Pulsar-Native#561 Phase D) wanting to hand this SAME pool instance
    /// to a `StaticMeshComponent`'s `#[gpu] Vec<PackedVertex>` field, so its
    /// hydrate-time write and this pool's own draw-time reads share one
    /// buffer, zero translation. Static only (not exposed for `dynamic_sub`)
    /// -- a component's own mesh data is upload-once geometry, the same
    /// `MeshKind::Static` classification every other static mesh already
    /// gets via `insert()`.
    pub fn vertex_pool(&self) -> Arc<VarLenGpuPool<PackedVertex>> {
        Arc::clone(&self.static_sub.vertices)
    }

    /// The static sub-pool's shared index storage — see [`Self::vertex_pool`].
    pub fn index_pool(&self) -> Arc<VarLenGpuPool<u32>> {
        Arc::clone(&self.static_sub.indices)
    }

    /// Test-only: reach the DYNAMIC sub-pool's vertex storage for a readback
    /// assertion. Not part of the public surface -- unlike the static pool
    /// (shared with `StaticMeshComponent`, see [`Self::vertex_pool`]),
    /// nothing outside this module needs to reach dynamic mesh storage
    /// directly.
    #[cfg(test)]
    fn dynamic_buffers_pool_for_test(&self) -> Arc<VarLenGpuPool<PackedVertex>> {
        Arc::clone(&self.dynamic_sub.vertices)
    }

    pub fn insert(&mut self, mesh: MeshUpload) -> MeshId {
        self.insert_with_kind(mesh, MeshKind::Static)
    }

    pub fn insert_dynamic(&mut self, mesh: MeshUpload) -> MeshId {
        self.insert_with_kind(mesh, MeshKind::Dynamic)
    }

    fn insert_with_kind(&mut self, mesh: MeshUpload, kind: MeshKind) -> MeshId {
        let sub = match kind {
            MeshKind::Static => &mut self.static_sub,
            MeshKind::Dynamic => &mut self.dynamic_sub,
        };

        let (vhandle, ihandle) = sub.alloc_and_write(&mesh.vertices, &mesh.indices);

        let slice = MeshSlice {
            first_vertex: vhandle.offset,
            vertex_count: vhandle.count,
            first_index: ihandle.offset,
            index_count: ihandle.count,
        };

        let (id, _, _) = self.meshes.insert(MeshRecord { slice, ref_count: 0, kind });
        id
    }

    pub fn insert_sectioned(&mut self, upload: SectionedMeshUpload) -> MultiMeshRecord {
        let sub = &mut self.static_sub;

        // Vertices are shared across all sections — allocate once.
        let vhandle = sub
            .vertices
            .write_var_row(&sub.queue, VarLenHandle::default(), &upload.vertices)
            .expect("mesh vertex pool exhausted the device's max_buffer_size");
        sub.live_vertex_count += upload.vertices.len();

        let section_mesh_ids = upload
            .sections
            .iter()
            .map(|sec_indices| {
                let ihandle = sub
                    .indices
                    .write_var_row(&sub.queue, VarLenHandle::default(), sec_indices)
                    .expect("mesh index pool exhausted the device's max_buffer_size");
                sub.live_index_count += sec_indices.len();

                let (id, _, _) = self.meshes.insert(MeshRecord {
                    slice: MeshSlice {
                        first_vertex: vhandle.offset,
                        vertex_count: vhandle.count,
                        first_index: ihandle.offset,
                        index_count: ihandle.count,
                    },
                    ref_count: 0,
                    kind: MeshKind::Static,
                });
                id
            })
            .collect();

        MultiMeshRecord { section_mesh_ids, ref_count: 0 }
    }

    pub fn update_dynamic_vertices(
        &mut self,
        id: MeshId,
        new_vertices: &[PackedVertex],
    ) -> Result<(), &'static str> {
        let Some(record) = self.meshes.get(id) else {
            return Err("invalid mesh id");
        };
        if record.kind != MeshKind::Dynamic {
            return Err("cannot update static mesh vertices");
        }
        if new_vertices.len() != record.slice.vertex_count as usize {
            return Err("vertex count mismatch: new_vertices.len() must equal the original upload");
        }
        let handle = VarLenHandle { offset: record.slice.first_vertex, count: record.slice.vertex_count };
        self.dynamic_sub.rewrite_dynamic_vertices(handle, new_vertices);
        Ok(())
    }

    pub fn get(&self, id: MeshId) -> Option<&MeshRecord> {
        self.meshes.get(id)
    }

    pub fn get_mut(&mut self, id: MeshId) -> Option<&mut MeshRecord> {
        self.meshes.get_mut_with_slot(id).map(|(_, record)| record)
    }

    /// Remove a mesh and immediately free its vertex/index ranges back into
    /// the pool's freelist, ready for a later allocation to reuse. Unlike
    /// the pre-`VarLenGpuPool` implementation, the underlying buffer is
    /// never proactively truncated back down (see [`MeshSubPool::free_slice`]) --
    /// VRAM is retained, not leaked.
    pub fn remove(&mut self, id: MeshId) -> Option<MeshRecord> {
        let (_, record) = self.meshes.remove(id)?;
        let sub = match record.kind {
            MeshKind::Static => &mut self.static_sub,
            MeshKind::Dynamic => &mut self.dynamic_sub,
        };
        sub.free_slice(&record.slice);
        Some(record)
    }

    pub fn buffers(&self) -> MeshBuffers<'_> {
        self.static_sub.buffers()
    }

    pub fn dynamic_buffers(&self) -> MeshBuffers<'_> {
        self.dynamic_sub.buffers()
    }

    pub fn total_vertex_count(&self) -> usize {
        self.static_sub.live_vertex_count
    }

    pub fn total_index_count(&self) -> usize {
        self.static_sub.live_index_count
    }

    pub fn unique_mesh_count(&self) -> usize {
        self.meshes.live_len()
    }

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        self.static_sub.flush(queue);
        self.dynamic_sub.flush(queue);
    }

    /// Reads a mesh's vertex/index data back from the GPU. Only used behind
    /// `#[cfg(feature = "bake")]` (offline lightmap baking, not a per-frame
    /// path), so a blocking readback here is a non-issue perf-wise -- baking
    /// itself already takes seconds to minutes. `VarLenGpuPool` keeps no CPU
    /// mirror to read this from directly (a deliberate design point, see its
    /// module doc), unlike the old `GrowableBuffer`-backed pool, which is why
    /// this now needs `device`/`queue` params it didn't take before -- see
    /// `Scene::build_static_bake_scene`'s doc for the one call site this
    /// change threads through.
    pub(crate) fn extract_mesh_data(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        id: MeshId,
    ) -> Option<MeshUpload> {
        let record = self.meshes.get(id)?;
        let slice = &record.slice;
        let sub = match record.kind {
            MeshKind::Static => &self.static_sub,
            MeshKind::Dynamic => &self.dynamic_sub,
        };

        let vertices = readback_var_len(device, queue, &sub.vertices, slice.first_vertex, slice.vertex_count);
        let indices = readback_var_len(device, queue, &sub.indices, slice.first_index, slice.index_count);

        Some(MeshUpload { vertices, indices })
    }
}

/// Blocking GPU→CPU readback of `[offset, offset + count)` from a
/// `VarLenGpuPool<T>`. Only ever called from [`MeshPool::extract_mesh_data`]
/// (bake-only, not a hot path) -- see that method's doc for why a blocking
/// wait here is acceptable.
fn readback_var_len<T: bytemuck::Pod + pulsar_scenedb::Pod + Send + Sync + 'static>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pool: &VarLenGpuPool<T>,
    offset: u32,
    count: u32,
) -> Vec<T> {
    if count == 0 {
        return Vec::new();
    }
    let byte_len = count as u64 * std::mem::size_of::<T>() as u64;
    let byte_offset = offset as u64 * std::mem::size_of::<T>() as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("helio-mesh-extract-readback"),
        size: byte_len,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("helio-mesh-extract-readback-encoder"),
    });
    pool.with_buffer(&mut |buf| {
        encoder.copy_buffer_to_buffer(buf, byte_offset, &staging, 0, byte_len);
    });
    queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("mesh extract readback map failed"));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("mesh extract readback poll failed");
    let data = slice.get_mapped_range().expect("mesh extract readback mapped range").to_vec();
    staging.unmap();
    bytemuck::cast_slice(&data).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> (Arc<wgpu::Device>, Arc<wgpu::Queue>) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        }))
        .expect("no adapter -- GPU tests need a local GPU");
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mesh-pool-test"),
            ..Default::default()
        }))
        .expect("device");
        (Arc::new(device), Arc::new(queue))
    }

    fn v(x: f32) -> PackedVertex {
        PackedVertex { position: [x, 0.0, 0.0], ..Default::default() }
    }

    #[test]
    fn insert_then_readback_round_trips_through_the_var_len_pool() {
        let (device, queue) = test_device();
        let mut pool = MeshPool::new(device.clone(), queue.clone());

        let id = pool.insert(MeshUpload { vertices: vec![v(1.0), v(2.0), v(3.0)], indices: vec![0, 1, 2] });
        let record = pool.get(id).expect("just-inserted mesh must be reachable");
        assert_eq!(record.slice.vertex_count, 3);
        assert_eq!(record.slice.index_count, 3);

        let got = readback_var_len::<PackedVertex>(&device, &queue, &pool.vertex_pool(), record.slice.first_vertex, record.slice.vertex_count);
        assert_eq!(got.iter().map(|p| p.position[0]).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn remove_frees_the_slot_and_a_later_insert_reuses_it() {
        let (device, queue) = test_device();
        let mut pool = MeshPool::new(device.clone(), queue.clone());

        let a = pool.insert(MeshUpload { vertices: vec![v(1.0), v(2.0)], indices: vec![0, 1] });
        let a_offset = pool.get(a).unwrap().slice.first_vertex;
        pool.remove(a);

        let b = pool.insert(MeshUpload { vertices: vec![v(9.0), v(9.0)], indices: vec![0, 1] });
        assert_eq!(pool.get(b).unwrap().slice.first_vertex, a_offset, "B must reuse A's freed slot, not append past it");
    }

    #[test]
    fn update_dynamic_vertices_preserves_the_original_offset() {
        // The property a draw call's baked-in `vertex_offset` depends on --
        // see `MeshSubPool::rewrite_dynamic_vertices`'s doc.
        let (device, queue) = test_device();
        let mut pool = MeshPool::new(device.clone(), queue.clone());

        // Insert a second dynamic mesh right after, so a relocation (if one
        // happened) would be observable as a changed offset, not silently
        // masked by both meshes coincidentally landing at 0 again.
        let a = pool.insert_dynamic(MeshUpload { vertices: vec![v(1.0), v(2.0)], indices: vec![0, 1] });
        let _b = pool.insert_dynamic(MeshUpload { vertices: vec![v(5.0)], indices: vec![0] });
        let a_offset = pool.get(a).unwrap().slice.first_vertex;

        pool.update_dynamic_vertices(a, &[v(100.0), v(200.0)]).expect("same-length update must succeed");
        assert_eq!(pool.get(a).unwrap().slice.first_vertex, a_offset, "in-place update must not move the allocation");

        let got = readback_var_len::<PackedVertex>(&device, &queue, &pool.dynamic_buffers_pool_for_test(), a_offset, 2);
        assert_eq!(got.iter().map(|p| p.position[0]).collect::<Vec<_>>(), vec![100.0, 200.0]);
    }

    #[test]
    fn update_dynamic_vertices_rejects_a_length_mismatch() {
        let (device, queue) = test_device();
        let mut pool = MeshPool::new(device, queue);
        let a = pool.insert_dynamic(MeshUpload { vertices: vec![v(1.0), v(2.0)], indices: vec![0, 1] });
        assert!(pool.update_dynamic_vertices(a, &[v(1.0)]).is_err(), "fewer vertices than the original upload must be rejected");
    }

    #[test]
    fn insert_sectioned_shares_one_vertex_allocation_across_all_sections() {
        let (device, queue) = test_device();
        let mut pool = MeshPool::new(device, queue);

        let upload = SectionedMeshUpload {
            vertices: vec![v(1.0), v(2.0), v(3.0), v(4.0)],
            sections: vec![vec![0, 1, 2], vec![1, 2, 3]],
        };
        let record = pool.insert_sectioned(upload);
        assert_eq!(record.section_mesh_ids.len(), 2);

        let s0 = pool.get(record.section_mesh_ids[0]).unwrap().slice;
        let s1 = pool.get(record.section_mesh_ids[1]).unwrap().slice;
        assert_eq!(s0.first_vertex, s1.first_vertex, "both sections must share the same vertex range");
        assert_eq!(s0.vertex_count, s1.vertex_count);
        assert_ne!(s0.first_index, s1.first_index, "each section must have its own index range");
    }
}
