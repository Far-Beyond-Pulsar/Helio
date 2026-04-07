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

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        self.vertices.flush(queue);
        self.indices.flush(queue);
    }
}
