use bytemuck;

use crate::arena::DenseRemove;
use crate::handles::PostProcessVolumeId;
use crate::scene::errors::{invalid, Result};
use crate::scene::types::PostProcessVolumeRecord;
use crate::scene::Scene;
use libhelio::{GpuPostProcessVolume, PostProcessVolumeDescriptor};

impl Scene {
    /// Insert a post-process volume into the scene.
    pub fn insert_post_process_volume(
        &mut self,
        desc: PostProcessVolumeDescriptor,
    ) -> Result<PostProcessVolumeId> {
        let gpu = desc.to_gpu();
        let record = PostProcessVolumeRecord { gpu };
        let (id, index) = self.pp_volumes.insert(record);
        self.pp_volumes_dirty = true;
        self.pp_volumes_dirty_range = Some((index, index + 1));
        Ok(id)
    }

    /// Remove a post-process volume from the scene.
    pub fn remove_post_process_volume(&mut self, id: PostProcessVolumeId) -> Result<()> {
        let DenseRemove {
            dense_index, moved, ..
        } = self
            .pp_volumes
            .remove(id)
            .ok_or_else(|| invalid("post-process volume"))?;
        self.pp_volumes_dirty = true;
        if let Some((_, moved_index)) = moved {
            let start = dense_index.min(moved_index);
            let end = dense_index.max(moved_index) + 1;
            self.pp_volumes_dirty_range = Some((start, end));
        } else {
            self.pp_volumes_dirty_range = Some((dense_index, dense_index + 1));
        }
        Ok(())
    }

    /// Update an existing post-process volume.
    pub fn update_post_process_volume(
        &mut self,
        id: PostProcessVolumeId,
        desc: PostProcessVolumeDescriptor,
    ) -> Result<()> {
        let (index, record) = self
            .pp_volumes
            .get_mut_with_index(id)
            .ok_or_else(|| invalid("post-process volume"))?;
        record.gpu = desc.to_gpu();
        self.pp_volumes_dirty = true;
        match self.pp_volumes_dirty_range {
            Some((start, end)) => {
                self.pp_volumes_dirty_range = Some((start.min(index), end.max(index + 1)));
            }
            None => self.pp_volumes_dirty_range = Some((index, index + 1)),
        }
        Ok(())
    }

    /// Returns a zero-copy slice of all GPU post-process volume data.
    pub fn get_post_process_volumes_gpu_slice(&self) -> &[GpuPostProcessVolume] {
        bytemuck::cast_slice(self.pp_volumes.dense.as_slice())
    }

    /// Number of active post-process volumes.
    pub fn post_process_volumes_count(&self) -> u32 {
        self.pp_volumes.dense_len() as u32
    }

    /// Whether post-process volumes have pending GPU updates.
    pub fn post_process_volumes_dirty(&self) -> bool {
        self.pp_volumes_dirty
    }

    /// Dirty range of post-process volumes needing GPU upload.
    pub fn post_process_volumes_dirty_range(&self) -> Option<(usize, usize)> {
        self.pp_volumes_dirty_range
    }

    pub(crate) fn consume_post_process_volumes_dirty_range(&mut self) -> Option<(usize, usize)> {
        self.pp_volumes_dirty_range.take()
    }

    pub(crate) fn clear_post_process_volumes_dirty(&mut self) {
        self.pp_volumes_dirty = false;
    }

    pub(crate) fn mark_post_process_volumes_dirty(&mut self) {
        self.pp_volumes_dirty = true;
    }
}
