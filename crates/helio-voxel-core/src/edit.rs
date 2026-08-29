use crate::gpu_types::GpuVoxelEdit;
use glam::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelOp {
    SetBox,
    AddSphere,
    SubtractSphere,
    Paint,
}

#[derive(Debug, Clone, Copy)]
pub struct VoxelEdit {
    pub op: VoxelOp,
    pub center: Vec3,
    pub radius: f32,
    pub material: u8,
}

impl From<&VoxelEdit> for GpuVoxelEdit {
    fn from(edit: &VoxelEdit) -> Self {
        GpuVoxelEdit {
            volume_id: 0,
            op_type: match edit.op {
                VoxelOp::SetBox => 0,
                VoxelOp::AddSphere => 1,
                VoxelOp::SubtractSphere => 2,
                VoxelOp::Paint => 3,
            },
            material: edit.material as u32,
            center: [edit.center.x, edit.center.y, edit.center.z],
            radius: edit.radius,
            _pad: 0,
        }
    }
}
