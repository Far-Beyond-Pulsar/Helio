use libhelio::GpuReflectionCapture;

use crate::handles::ReflectionCaptureId;
use crate::scene::actor::ReflectionCaptureDescriptor;
use crate::scene::errors::{invalid, Result};
use crate::scene::types::ReflectionCaptureRecord;

impl super::super::Scene {
    pub fn insert_reflection_capture(
        &mut self,
        desc: ReflectionCaptureDescriptor,
    ) -> Result<ReflectionCaptureId> {
        let gpu = GpuReflectionCapture {
            position_influence: [
                desc.position[0],
                desc.position[1],
                desc.position[2],
                desc.influence_radius,
            ],
            box_min: [desc.box_min[0], desc.box_min[1], desc.box_min[2], 0.0],
            box_max: [desc.box_max[0], desc.box_max[1], desc.box_max[2], 0.0],
            cubemap_index: desc.cubemap_index,
            capture_type: desc.capture_type,
            blend_weight: desc.blend_weight,
            _pad0: 0,
            _pad1: [0.0; 4],
        };
        let record = ReflectionCaptureRecord { gpu };
        let (id, dense_index) = self.reflection_captures.insert(record);
        let pushed = self.gpu_scene.reflection_captures.push(gpu);
        debug_assert_eq!(pushed, dense_index);
        Ok(id)
    }

    pub fn update_reflection_capture(
        &mut self,
        id: ReflectionCaptureId,
        desc: &ReflectionCaptureDescriptor,
    ) -> Result<()> {
        let (dense_index, record) = self
            .reflection_captures
            .get_mut_with_index(id)
            .ok_or_else(|| invalid("reflection capture"))?;
        record.gpu = GpuReflectionCapture {
            position_influence: [
                desc.position[0],
                desc.position[1],
                desc.position[2],
                desc.influence_radius,
            ],
            box_min: [desc.box_min[0], desc.box_min[1], desc.box_min[2], 0.0],
            box_max: [desc.box_max[0], desc.box_max[1], desc.box_max[2], 0.0],
            cubemap_index: desc.cubemap_index,
            capture_type: desc.capture_type,
            blend_weight: desc.blend_weight,
            _pad0: 0,
            _pad1: [0.0; 4],
        };
        let updated = self.gpu_scene.reflection_captures.update(dense_index, record.gpu);
        debug_assert!(updated);
        Ok(())
    }

    pub fn remove_reflection_capture(&mut self, id: ReflectionCaptureId) -> bool {
        let removed = self.reflection_captures.remove(id);
        if let Some(removed) = removed {
            let gpu_removed = self
                .gpu_scene
                .reflection_captures
                .swap_remove(removed.dense_index);
            debug_assert!(gpu_removed.is_some());
            true
        } else {
            false
        }
    }
}
