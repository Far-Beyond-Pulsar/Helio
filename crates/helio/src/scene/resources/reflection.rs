use libhelio::{GpuReflectionCapture, ReflectionCaptureMobility};

use crate::handles::ReflectionCaptureId;
use crate::scene::actor::ReflectionCaptureDescriptor;
use crate::scene::errors::{invalid, Result};
use crate::scene::types::ReflectionCaptureRecord;

/// Flatten a descriptor into its GPU form, keeping whatever cubemap layer has
/// already been assigned to this capture.
fn build_gpu(desc: &ReflectionCaptureDescriptor, cubemap_index: i32) -> GpuReflectionCapture {
    let pos = desc.position();
    GpuReflectionCapture {
        position_radius: [pos[0], pos[1], pos[2], desc.influence_radius],
        extents_transition: [
            desc.extents[0],
            desc.extents[1],
            desc.extents[2],
            desc.transition_distance,
        ],
        world_to_local: desc.transform.inverse().to_cols_array_2d(),
        cubemap_index,
        shape: desc.shape as u32,
        mobility: desc.mobility as u32,
        brightness: desc.brightness,
    }
}

impl super::super::Scene {
    /// Insert a reflection capture.
    ///
    /// The capture starts with no cubemap layer (`cubemap_index = -1`) and
    /// contributes nothing until one is assigned — by the probe bake for
    /// static captures, or by the dynamic capture pass for movable ones.
    pub fn insert_reflection_capture(
        &mut self,
        desc: ReflectionCaptureDescriptor,
    ) -> Result<ReflectionCaptureId> {
        let record = ReflectionCaptureRecord {
            gpu: build_gpu(&desc, -1),
        };
        let (id, _) = self.reflection_captures.insert(record);
        self.rebuild_reflection_capture_buffer();
        Ok(id)
    }

    pub fn update_reflection_capture(
        &mut self,
        id: ReflectionCaptureId,
        desc: &ReflectionCaptureDescriptor,
    ) -> Result<()> {
        let (_, record) = self
            .reflection_captures
            .get_mut_with_index(id)
            .ok_or_else(|| invalid("reflection capture"))?;
        // Moving or resizing a capture must not throw away the cubemap it is
        // already bound to.
        record.gpu = build_gpu(desc, record.gpu.cubemap_index);
        self.rebuild_reflection_capture_buffer();
        Ok(())
    }

    pub fn remove_reflection_capture(&mut self, id: ReflectionCaptureId) -> bool {
        if self.reflection_captures.remove(id).is_some() {
            self.rebuild_reflection_capture_buffer();
            true
        } else {
            false
        }
    }

    /// World positions of every static capture, in the order the bake should
    /// produce probes. Feeds [`Scene::assign_reflection_capture_layers`].
    pub fn static_reflection_capture_positions(&self) -> Vec<[f32; 3]> {
        let mut out: Vec<_> = self
            .reflection_captures
            .iter_with_handles()
            .filter(|(_, r)| r.gpu.mobility == ReflectionCaptureMobility::Static as u32)
            .map(|(_, r)| {
                let p = r.gpu.position_radius;
                [p[0], p[1], p[2]]
            })
            .collect();
        out.shrink_to_fit();
        out
    }

    /// Bind static captures to cubemap array layers, in the same order
    /// [`Scene::static_reflection_capture_positions`] reported them.
    ///
    /// This is the handshake that keeps capture actors and baked probes from
    /// drifting apart: the caller never picks a layer index by hand.
    pub fn assign_reflection_capture_layers(&mut self) {
        let ids: Vec<_> = self
            .reflection_captures
            .iter_with_handles()
            .filter(|(_, r)| r.gpu.mobility == ReflectionCaptureMobility::Static as u32)
            .map(|(id, _)| id)
            .collect();
        for (layer, id) in ids.into_iter().enumerate() {
            if let Some((_, record)) = self.reflection_captures.get_mut_with_index(id) {
                record.gpu.cubemap_index = layer as i32;
            }
        }
        self.rebuild_reflection_capture_buffer();
    }

    /// Rewrite the GPU capture buffer, ordered by influence volume, smallest
    /// first.
    ///
    /// The shader blends front-to-back and stops once coverage saturates, so
    /// the smallest (most specific) capture must come first to get first claim
    /// on a pixel — a small capture inside a large room capture is the case
    /// that has to win. Captures number in the dozens and change rarely, so a
    /// full sorted rewrite on edit is cheaper to reason about than keeping
    /// arena order and buffer order in sync.
    fn rebuild_reflection_capture_buffer(&mut self) {
        let mut captures: Vec<GpuReflectionCapture> = self
            .reflection_captures
            .iter_with_handles()
            .map(|(_, r)| r.gpu)
            .collect();
        captures.sort_by(|a, b| {
            a.influence_size()
                .partial_cmp(&b.influence_size())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.gpu_scene.reflection_captures.set_data(captures);
    }
}
