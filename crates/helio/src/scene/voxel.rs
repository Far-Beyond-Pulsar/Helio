use bytemuck::Zeroable;
use helio_voxel_core::{
    GpuVoxelEdit, GpuVoxelMaterial, GpuVoxelVolume, VoxelEdit, VoxelOctree, EDIT_RING_CAPACITY,
};

use super::types::VoxelVolumeDescriptor;
use crate::handles::VoxelVolumeId;
use helio_core::GpuScene;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoxelMode {
    Auto,
    Dynamic,
}

impl Default for VoxelMode {
    fn default() -> Self {
        VoxelMode::Auto
    }
}

#[derive(Debug)]
pub struct VoxelVolumeRecord {
    pub id: VoxelVolumeId,
    pub octree: VoxelOctree,
    pub gpu_slot: u32,
    pub meshlet_offset: u32,
    pub meshlet_count: u32,
    pub local_to_world: glam::Mat4,
    pub movability: libhelio::Movability,
    pub mode: VoxelMode,
    pub edit_cooldown: u32,
    pub dirty: bool,
    pub material_palette: Vec<GpuVoxelMaterial>,
}

impl VoxelVolumeRecord {
    pub fn new(id: VoxelVolumeId, gpu_slot: u32, descriptor: &VoxelVolumeDescriptor) -> Self {
        let octree = VoxelOctree::new(descriptor.voxel_size, descriptor.root_extent);
        Self {
            id,
            octree,
            gpu_slot,
            meshlet_offset: 0,
            meshlet_count: 0,
            local_to_world: descriptor.local_to_world,
            movability: descriptor
                .movability
                .unwrap_or(libhelio::Movability::Static),
            mode: descriptor.mode.unwrap_or(VoxelMode::Auto),
            edit_cooldown: 0,
            dirty: true,
            material_palette: descriptor.material_palette.clone(),
        }
    }

    /// Upload this volume's data to the GPU scene buffers
    pub fn upload_to_gpu(&self, gpu_scene: &mut GpuScene, gpu_slot: u32) {
        let volume_gpu = GpuVoxelVolume {
            local_to_world: self.local_to_world.to_cols_array(),
            world_to_local: self.local_to_world.inverse().to_cols_array(),
            dimensions: [
                self.octree.brick_size * 8,
                self.octree.brick_size * 8,
                self.octree.brick_size * 8,
            ],
            brick_grid_dim: self.octree.brick_size,
            voxel_size: self.octree.voxel_size,
            palette_offset: 0,
            volume_id: gpu_slot,
            _pad: [0; 2],
        };
        let idx = gpu_slot as usize;
        while gpu_scene.voxel_volumes.len() <= idx {
            gpu_scene.voxel_volumes.push(GpuVoxelVolume::zeroed());
        }
        gpu_scene.voxel_volumes.update(idx, volume_gpu);
    }

    /// Push edits to the GPU edit ring buffer
    pub fn push_edits_to_gpu(&mut self, gpu_scene: &mut GpuScene, edits: &[VoxelEdit]) {
        for edit in edits {
            let idx = gpu_scene.voxel_ring_write_index as usize;
            let gpu_edit = GpuVoxelEdit {
                volume_id: self.gpu_slot,
                op_type: match edit.op {
                    helio_voxel_core::VoxelOp::SetBox => 0,
                    helio_voxel_core::VoxelOp::AddSphere => 1,
                    helio_voxel_core::VoxelOp::SubtractSphere => 2,
                    helio_voxel_core::VoxelOp::Paint => 3,
                },
                material: edit.material as u32,
                center: [edit.center.x, edit.center.y, edit.center.z],
                radius: edit.radius,
                _pad: 0,
            };
            while gpu_scene.voxel_edit_ring.len() <= idx {
                gpu_scene.voxel_edit_ring.push(GpuVoxelEdit::zeroed());
            }
            gpu_scene.voxel_edit_ring.update(idx, gpu_edit);
            gpu_scene.voxel_ring_write_index =
                ((gpu_scene.voxel_ring_write_index + 1) % EDIT_RING_CAPACITY) as u32;
        }
        gpu_scene.voxel_volumes_generation += 1;
    }

    pub fn edit(&mut self, edit: &VoxelEdit) {
        let center = [edit.center.x, edit.center.y, edit.center.z];
        self.octree.root.mark_sphere_dirty(center, edit.radius);
        self.dirty = true;
        self.edit_cooldown = 0;
    }
}
