use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelVolume {
    pub local_to_world: [f32; 16],
    pub world_to_local: [f32; 16],
    pub dimensions: [u32; 3],
    pub brick_grid_dim: u32,
    pub voxel_size: f32,
    pub palette_offset: u32,
    pub volume_id: u32,
    pub _pad: [u32; 2],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBrickMeta {
    pub data_offset: u32,
    pub occupancy: u32,
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuBrickMeshlet {
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub vertex_count: u32,
    pub index_count: u32,
    pub brick_index: u32,
    pub volume_id: u32,
    pub _pad: [u32; 2],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelMaterial {
    pub color: [f32; 3],
    pub roughness: f32,
    pub metalness: f32,
    pub emissive: f32,
    pub _pad: [u32; 2],
}

#[derive(Debug, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct GpuVoxelEdit {
    pub volume_id: u32,
    pub op_type: u32,
    pub material: u32,
    pub center: [f32; 3],
    pub radius: f32,
    pub _pad: u32,
}
