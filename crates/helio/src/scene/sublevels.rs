//! Sublevels: a group of objects rendered through one shared, cheaply movable
//! coordinate-space transform.
//!
//! A sublevel's members keep their ordinary *local* (sublevel-relative)
//! `GpuInstanceData::model`, exactly as an ungrouped object would — nothing
//! about how an object is authored changes. What differs is that the GPU
//! vertex/cull shaders read one extra transform, `coordinate_spaces[slot]`
//! (see `libhelio::{coordinate_space, set_coordinate_space}` and
//! `helio_core::CoordinateSpaceBuffer`), and apply it on top. Moving the whole
//! sublevel is therefore **O(1)**: one matrix write via
//! [`Scene::move_sublevel`]/[`Scene::update_sublevel`], never a walk over
//! member objects. Assigning membership (tagging each member with the
//! sublevel's slot) is the one O(N) step, and it happens only once, at
//! [`Scene::add_sublevel`] — not per frame.
//!
//! Unlike portals, sublevel content needs no clipping: it isn't a duplicate
//! draw of anything, just the same geometry rendered in a different place.

use glam::Mat4;

use crate::groups::GroupId;
use crate::handles::SublevelId;
use crate::scene::errors::{invalid, Result, SceneError};
use crate::scene::Scene;

/// Configuration for [`Scene::add_sublevel`].
#[derive(Debug, Clone, Copy)]
pub struct SublevelDescriptor {
    /// The group whose *current* members become this sublevel's members.
    /// Membership is captured once, at creation — objects added to `group`
    /// afterward are not automatically included (see
    /// [`Scene::refresh_sublevel_membership`] to re-capture explicitly).
    pub group: GroupId,

    /// World-space placement of the sublevel's local origin.
    pub placement: Mat4,
}

/// Internal record for a sublevel.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SublevelRecord {
    pub group: GroupId,
    /// GPU coordinate-space slot this sublevel owns (see
    /// `helio_core::CoordinateSpaceBuffer`). Stable for the sublevel's lifetime.
    pub coordinate_space: u32,
    pub placement: Mat4,
}

impl Scene {
    /// Create a sublevel: allocate a coordinate-space slot, place it, and tag
    /// `desc.group`'s current members with that slot.
    ///
    /// # Performance
    /// - CPU cost: **O(N)** over the dense object array (N = total scene
    ///   objects, not group size) — a one-time membership walk, same shape as
    ///   [`Scene::move_group`](crate::Scene::move_group) but paid once here
    ///   instead of on every call.
    /// - GPU cost: one coordinate-space slot write, plus one instance write
    ///   per member (their `flags` changed) — deferred to the next `flush()`
    ///   if a full rebuild is already pending.
    ///
    /// # Errors
    /// [`SceneError::CoordinateSpaceCapacityExceeded`] if all coordinate-space
    /// slots are already claimed by other sublevels/portals.
    pub fn add_sublevel(&mut self, desc: SublevelDescriptor) -> Result<SublevelId> {
        let slot = self
            .alloc_coordinate_space()
            .ok_or(SceneError::CoordinateSpaceCapacityExceeded)?;
        self.gpu_scene
            .coordinate_spaces
            .update_slot(slot, desc.placement.to_cols_array());

        let record = SublevelRecord {
            group: desc.group,
            coordinate_space: slot,
            placement: desc.placement,
        };
        let (id, _slot_index, _fresh) = self.sublevels.insert(record);

        self.tag_group_with_coordinate_space(desc.group, slot);

        Ok(id)
    }

    /// Re-walks `sublevel`'s group and tags any current member that isn't
    /// already tagged with this sublevel's coordinate space.
    ///
    /// [`Scene::add_sublevel`] captures membership once, at creation; call
    /// this after adding more objects to the same group if they should join
    /// the sublevel too. Same O(N) shape as `add_sublevel`'s initial walk.
    pub fn refresh_sublevel_membership(&mut self, sublevel: SublevelId) -> Result<()> {
        let record = *self
            .sublevels
            .get(sublevel)
            .ok_or_else(|| invalid("sublevel"))?;
        self.tag_group_with_coordinate_space(record.group, record.coordinate_space);
        Ok(())
    }

    fn tag_group_with_coordinate_space(&mut self, group: GroupId, slot: u32) {
        let n = self.objects.dense_len();
        for i in 0..n {
            let Some(r) = self.objects.get_dense_mut(i) else {
                continue;
            };
            if !r.groups.contains(group) {
                continue;
            }
            r.instance.flags = libhelio::set_coordinate_space(r.instance.flags, slot);
            if !self.objects_dirty {
                let gpu_slot = r.draw.first_instance as usize;
                self.gpu_scene.instances.update(gpu_slot, r.instance);
            }
        }
    }

    /// Apply a transform delta to a sublevel's placement. **O(1)** — one
    /// coordinate-space matrix write, no member object is touched.
    ///
    /// `delta` is post-multiplied: `new_placement = delta * old_placement`,
    /// same convention as [`Scene::move_group`](crate::Scene::move_group).
    pub fn move_sublevel(&mut self, sublevel: SublevelId, delta: Mat4) -> Result<()> {
        let (_slot_index, record) = self
            .sublevels
            .get_mut_with_slot(sublevel)
            .ok_or_else(|| invalid("sublevel"))?;
        record.placement = delta * record.placement;
        let (space, placement) = (record.coordinate_space, record.placement);
        self.gpu_scene
            .coordinate_spaces
            .update_slot(space, placement.to_cols_array());
        Ok(())
    }

    /// Set a sublevel's placement directly. **O(1)** — one coordinate-space
    /// matrix write, no member object is touched.
    pub fn update_sublevel(&mut self, sublevel: SublevelId, placement: Mat4) -> Result<()> {
        let (_slot_index, record) = self
            .sublevels
            .get_mut_with_slot(sublevel)
            .ok_or_else(|| invalid("sublevel"))?;
        record.placement = placement;
        let space = record.coordinate_space;
        self.gpu_scene
            .coordinate_spaces
            .update_slot(space, placement.to_cols_array());
        Ok(())
    }

    /// Current world-space placement of a sublevel, e.g. to compose with a
    /// member object's local transform for CPU-side picking.
    pub fn sublevel_placement(&self, sublevel: SublevelId) -> Option<Mat4> {
        self.sublevels.get(sublevel).map(|r| r.placement)
    }

    /// Remove a sublevel: untag its members (they fall back to world space,
    /// coordinate space 0) and free its GPU slot for reuse.
    ///
    /// # Performance
    /// CPU cost: O(N) over the dense object array, same as `add_sublevel`.
    pub fn remove_sublevel(&mut self, sublevel: SublevelId) -> Result<()> {
        let (_slot_index, record) = self
            .sublevels
            .remove(sublevel)
            .ok_or_else(|| invalid("sublevel"))?;

        let n = self.objects.dense_len();
        for i in 0..n {
            let Some(r) = self.objects.get_dense_mut(i) else {
                continue;
            };
            if !r.groups.contains(record.group) {
                continue;
            }
            if libhelio::coordinate_space(r.instance.flags) != record.coordinate_space {
                continue;
            }
            r.instance.flags = libhelio::set_coordinate_space(r.instance.flags, 0);
            if !self.objects_dirty {
                let gpu_slot = r.draw.first_instance as usize;
                self.gpu_scene.instances.update(gpu_slot, r.instance);
            }
        }

        self.free_coordinate_space(record.coordinate_space);
        Ok(())
    }
}
