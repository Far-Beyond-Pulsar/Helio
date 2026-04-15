use bytemuck::{Pod, Zeroable};
use helio_v3::GrowableBuffer;

use crate::arena::SparsePool;
use crate::handles::MeshId;

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
}

pub struct MeshBuffers<'a> {
    pub vertices: &'a wgpu::Buffer,
    pub indices: &'a wgpu::Buffer,
}

pub struct MeshPool {
    vertices: GrowableBuffer<PackedVertex>,
    indices: GrowableBuffer<u32>,
    meshes: SparsePool<MeshRecord, MeshId>,
}

impl MeshPool {
    pub fn new(device: std::sync::Arc<wgpu::Device>) -> Self {
        Self {
            vertices: GrowableBuffer::new(
                device.clone(),
                4096,
                wgpu::BufferUsages::VERTEX,
                "Helio Mesh Vertex Buffer",
            ),
            indices: GrowableBuffer::new(
                device,
                8192,
                wgpu::BufferUsages::INDEX,
                "Helio Mesh Index Buffer",
            ),
            meshes: SparsePool::new(),
        }
    }

    pub fn insert(&mut self, mesh: MeshUpload) -> MeshId {
        let vertex_range = self.vertices.extend_from_slice(&mesh.vertices);
        let index_range = self.indices.extend_from_slice(&mesh.indices);
        let slice = MeshSlice {
            first_vertex: vertex_range.start as u32,
            vertex_count: (vertex_range.end - vertex_range.start) as u32,
            first_index: index_range.start as u32,
            index_count: (index_range.end - index_range.start) as u32,
        };
        let (id, _, _) = self.meshes.insert(MeshRecord {
            slice,
            ref_count: 0,
        });
        id
    }

    /// Upload a sectioned mesh: vertices are pushed ONCE into the shared vertex buffer;
    /// each section's index list gets its own contiguous range in the index buffer.
    /// Returns one `MeshId` per section — all share the same `first_vertex`.
    ///
    /// This is the GPU-native implementation of Unreal's Static Mesh sections.
    pub fn insert_sectioned(&mut self, upload: SectionedMeshUpload) -> MultiMeshRecord {
        let vertex_range = self.vertices.extend_from_slice(&upload.vertices);
        let first_vertex = vertex_range.start as u32;
        let vertex_count = (vertex_range.end - vertex_range.start) as u32;

        let section_mesh_ids = upload
            .sections
            .iter()
            .map(|sec_indices| {
                let index_range = self.indices.extend_from_slice(sec_indices);
                let (id, _, _) = self.meshes.insert(MeshRecord {
                    slice: MeshSlice {
                        first_vertex,
                        vertex_count,
                        first_index: index_range.start as u32,
                        index_count: (index_range.end - index_range.start) as u32,
                    },
                    ref_count: 0,
                });
                id
            })
            .collect();

        MultiMeshRecord {
            section_mesh_ids,
            ref_count: 0,
        }
    }

    pub fn get(&self, id: MeshId) -> Option<&MeshRecord> {
        self.meshes.get(id)
    }

    pub fn get_mut(&mut self, id: MeshId) -> Option<&mut MeshRecord> {
        self.meshes.get_mut_with_slot(id).map(|(_, record)| record)
    }

    pub fn remove(&mut self, id: MeshId) -> Option<MeshRecord> {
        self.meshes.remove(id).map(|(_, record)| record)
    }

    pub fn buffers(&self) -> MeshBuffers<'_> {
        MeshBuffers {
            vertices: self.vertices.buffer(),
            indices: self.indices.buffer(),
        }
    }

    /// Total vertices in the shared vertex mega-buffer.
    pub fn total_vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Total indices in the shared index mega-buffer.
    /// Triangles = `total_index_count() / 3`.
    pub fn total_index_count(&self) -> usize {
        self.indices.len()
    }

    /// Number of unique mesh records currently live (sections each count as one).
    pub fn unique_mesh_count(&self) -> usize {
        self.meshes.live_len()
    }

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        self.vertices.flush(queue);
        self.indices.flush(queue);
    }

    /// Extracts a mesh's vertex and index data from the pool.
    ///
    /// Returns None if the mesh ID is invalid. Used internally for baking.
    pub(crate) fn extract_mesh_data(&self, id: MeshId) -> Option<MeshUpload> {
        let record = self.meshes.get(id)?;
        let slice = &record.slice;
        
        let vertex_start = slice.first_vertex as usize;
        let vertex_end = vertex_start + slice.vertex_count as usize;
        let index_start = slice.first_index as usize;
        let index_end = index_start + slice.index_count as usize;
        
        let vertices = self.vertices.as_slice()
            .get(vertex_start..vertex_end)?
            .to_vec();
        let indices = self.indices.as_slice()
            .get(index_start..index_end)?
            .to_vec();
        
        Some(MeshUpload { vertices, indices })
    }
}

