use libhelio::GpuMeshletEntry;

use crate::mesh::PackedVertex;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtualMeshId(pub(crate) u32);

#[derive(Debug, Clone)]
pub struct VirtualMeshLodUpload {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
    pub meshlets: Vec<GpuMeshletEntry>,
}

#[derive(Debug, Clone)]
pub struct VirtualMeshUpload {
    pub lods: Vec<VirtualMeshLodUpload>,
}

#[derive(Debug, Clone, Copy)]
pub struct VirtualObjectDescriptor {
    pub virtual_mesh: VirtualMeshId,
    pub material_id: u32,
    pub transform: glam::Mat4,
    pub bounds: [f32; 4],
    pub flags: u32,
    pub groups: crate::groups::GroupMask,
    pub movability: Option<libhelio::Movability>,
}
