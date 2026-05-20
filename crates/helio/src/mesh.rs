use bytemuck::{Pod, Zeroable};
use helio_v3::GrowableBuffer;

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
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PackedVertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords0: [f32; 2],
    pub tex_coords1: [f32; 2],
    pub normal: u32,
    pub tangent: u32,
}

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
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
}

// ── Free-list range allocator ─────────────────────────────────────────────────

/// First-fit range allocator with coalescing and tail-trimming.
///
/// Tracks free `(start, len)` ranges inside a logically contiguous buffer.
/// On each `free()` call adjacent ranges are merged (O(free_ranges) but
/// typically very small).  When the newly-freed range butts up against the
/// end of the buffer, the tail is trimmed so callers can `truncate()` the
/// physical buffer back to the new high-water mark, actually returning memory.
#[derive(Default)]
struct FreeListAllocator {
    /// Sorted by start offset, coalesced, no overlaps.
    free: Vec<(usize, usize)>,
}

impl FreeListAllocator {
    /// Try to satisfy an allocation of `count` elements using the free list.
    ///
    /// Uses first-fit: picks the first range that is large enough.  Splits
    /// oversized ranges, leaving the remainder on the list.
    ///
    /// Returns `Some(start)` on success or `None` when the caller must append.
    fn alloc(&mut self, count: usize) -> Option<usize> {
        if count == 0 {
            return Some(0);
        }
        let idx = self.free.iter().position(|&(_, len)| len >= count)?;
        let (start, len) = self.free[idx];
        if len == count {
            self.free.remove(idx);
        } else {
            self.free[idx] = (start + count, len - count);
        }
        Some(start)
    }

    /// Mark `[start, start + count)` as free.
    ///
    /// Adjacent ranges are coalesced.  If the resulting free range extends to
    /// `buf_len` (the current logical end of the buffer), the tail is trimmed:
    /// the free entry is removed and the new logical buffer end is returned so
    /// the caller can `truncate()` the physical buffer.
    ///
    /// Returns `Some(new_buf_len)` when a tail-trim occurred, `None` otherwise.
    fn free(&mut self, start: usize, count: usize, buf_len: usize) -> Option<usize> {
        if count == 0 {
            return None;
        }

        // Insert in sorted order.
        let pos = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(pos, (start, count));

        // Coalesce with successor.
        if pos + 1 < self.free.len() {
            let (s, l) = self.free[pos];
            let (ns, nl) = self.free[pos + 1];
            if s + l == ns {
                self.free[pos] = (s, l + nl);
                self.free.remove(pos + 1);
            }
        }
        // Coalesce with predecessor.
        if pos > 0 {
            let prev = pos - 1;
            let (ps, pl) = self.free[prev];
            let (s, l) = self.free[pos.min(self.free.len() - 1)];
            if ps + pl == s {
                self.free[prev] = (ps, pl + l);
                // The coalesced entry is now at `prev`.
                if prev + 1 < self.free.len() {
                    self.free.remove(prev + 1);
                }
            }
        }

        // Tail-trim: if the last free range reaches the buffer end, remove it
        // and report a new (smaller) logical end so the caller can truncate.
        if let Some(&(tail_start, tail_len)) = self.free.last() {
            if tail_start + tail_len == buf_len {
                self.free.pop();
                return Some(tail_start);
            }
        }

        None
    }

    fn clear(&mut self) {
        self.free.clear();
    }
}

// ── Sub-pool (vertex + index + their allocators) ──────────────────────────────

struct MeshSubPool {
    vertices: GrowableBuffer<PackedVertex>,
    indices: GrowableBuffer<u32>,
    vertex_alloc: FreeListAllocator,
    index_alloc: FreeListAllocator,
}

impl MeshSubPool {
    fn new(device: std::sync::Arc<wgpu::Device>, kind: MeshKind) -> Self {
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
            vertices: GrowableBuffer::new(
                device.clone(),
                v_cap,
                wgpu::BufferUsages::VERTEX,
                v_label,
            ),
            indices: GrowableBuffer::new(
                device,
                i_cap,
                wgpu::BufferUsages::INDEX,
                i_label,
            ),
            vertex_alloc: FreeListAllocator::default(),
            index_alloc: FreeListAllocator::default(),
        }
    }

    /// Allocate space for `vcount` vertices and `icount` indices.
    ///
    /// Tries free ranges first; falls back to appending.  Returns the
    /// `(first_vertex, first_index)` slot start.
    fn alloc_and_write(
        &mut self,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> (usize, usize) {
        let vcount = vertices.len();
        let icount = indices.len();

        let vstart = if let Some(s) = self.vertex_alloc.alloc(vcount) {
            self.vertices.update_range(s, vertices);
            s
        } else {
            self.vertices.extend_from_slice(vertices).start
        };

        let istart = if let Some(s) = self.index_alloc.alloc(icount) {
            self.indices.update_range(s, indices);
            s
        } else {
            self.indices.extend_from_slice(indices).start
        };

        (vstart, istart)
    }

    /// Return the vertex and index ranges of `slice` to the free list.
    ///
    /// Performs tail-trimming: if the freed range reaches the current logical
    /// end of the buffer, the buffer is truncated immediately, actually
    /// releasing CPU and (on next flush) GPU memory.
    fn free_slice(&mut self, slice: &MeshSlice) {
        let vstart = slice.first_vertex as usize;
        let vcount = slice.vertex_count as usize;
        let istart = slice.first_index as usize;
        let icount = slice.index_count as usize;

        if let Some(new_vlen) = self.vertex_alloc.free(vstart, vcount, self.vertices.live_len()) {
            self.vertices.truncate(new_vlen);
        }
        if let Some(new_ilen) = self.index_alloc.free(istart, icount, self.indices.live_len()) {
            self.indices.truncate(new_ilen);
        }
    }

    fn buffers(&self) -> MeshBuffers<'_> {
        MeshBuffers {
            vertices: self.vertices.buffer(),
            indices: self.indices.buffer(),
        }
    }

    fn flush(&mut self, queue: &wgpu::Queue) {
        self.vertices.flush(queue);
        self.indices.flush(queue);
    }
}

// ── Public MeshPool ───────────────────────────────────────────────────────────

pub struct MeshPool {
    static_sub: MeshSubPool,
    dynamic_sub: MeshSubPool,
    meshes: SparsePool<MeshRecord, MeshId>,
}

impl MeshPool {
    pub fn new(device: std::sync::Arc<wgpu::Device>) -> Self {
        Self {
            static_sub: MeshSubPool::new(device.clone(), MeshKind::Static),
            dynamic_sub: MeshSubPool::new(device, MeshKind::Dynamic),
            meshes: SparsePool::new(),
        }
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

        let (first_vertex, first_index) = sub.alloc_and_write(&mesh.vertices, &mesh.indices);

        let slice = MeshSlice {
            first_vertex: first_vertex as u32,
            vertex_count: mesh.vertices.len() as u32,
            first_index: first_index as u32,
            index_count: mesh.indices.len() as u32,
        };

        let (id, _, _) = self.meshes.insert(MeshRecord { slice, ref_count: 0, kind });
        id
    }

    pub fn insert_sectioned(&mut self, upload: SectionedMeshUpload) -> MultiMeshRecord {
        let sub = &mut self.static_sub;

        // Vertices are shared across all sections — allocate once.
        let first_vertex = if let Some(s) = sub.vertex_alloc.alloc(upload.vertices.len()) {
            sub.vertices.update_range(s, &upload.vertices);
            s
        } else {
            sub.vertices.extend_from_slice(&upload.vertices).start
        };
        let vertex_count = upload.vertices.len() as u32;

        let section_mesh_ids = upload
            .sections
            .iter()
            .map(|sec_indices| {
                let first_index = if let Some(s) = sub.index_alloc.alloc(sec_indices.len()) {
                    sub.indices.update_range(s, sec_indices);
                    s
                } else {
                    sub.indices.extend_from_slice(sec_indices).start
                };

                let (id, _, _) = self.meshes.insert(MeshRecord {
                    slice: MeshSlice {
                        first_vertex: first_vertex as u32,
                        vertex_count,
                        first_index: first_index as u32,
                        index_count: sec_indices.len() as u32,
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
        let start = record.slice.first_vertex as usize;
        self.dynamic_sub.vertices.update_range(start, new_vertices);
        Ok(())
    }

    pub fn get(&self, id: MeshId) -> Option<&MeshRecord> {
        self.meshes.get(id)
    }

    pub fn get_mut(&mut self, id: MeshId) -> Option<&mut MeshRecord> {
        self.meshes.get_mut_with_slot(id).map(|(_, record)| record)
    }

    /// Remove a mesh and immediately free its vertex/index ranges back into the
    /// allocator.  If the freed ranges are at the tail of their buffer, the
    /// buffer is truncated on the spot — no separate "compact" call needed.
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
        self.static_sub.vertices.live_len()
    }

    pub fn total_index_count(&self) -> usize {
        self.static_sub.indices.live_len()
    }

    pub fn unique_mesh_count(&self) -> usize {
        self.meshes.live_len()
    }

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        self.static_sub.flush(queue);
        self.dynamic_sub.flush(queue);
    }

    pub(crate) fn extract_mesh_data(&self, id: MeshId) -> Option<MeshUpload> {
        let record = self.meshes.get(id)?;
        let slice = &record.slice;
        let sub = match record.kind {
            MeshKind::Static => &self.static_sub,
            MeshKind::Dynamic => &self.dynamic_sub,
        };

        let vertex_start = slice.first_vertex as usize;
        let vertex_end = vertex_start + slice.vertex_count as usize;
        let index_start = slice.first_index as usize;
        let index_end = index_start + slice.index_count as usize;

        let vertices = sub.vertices.as_slice().get(vertex_start..vertex_end)?.to_vec();
        let indices = sub.indices.as_slice().get(index_start..index_end)?.to_vec();

        Some(MeshUpload { vertices, indices })
    }
}
