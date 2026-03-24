//! Group membership management for scene objects.
//!
//! Provides methods for adding/removing objects to/from groups and querying
//! group membership with O(1) GPU visibility buffer updates.

use crate::groups::{GroupId, GroupMask};
use crate::handles::ObjectId;

use super::super::errors::{invalid, Result};
use super::super::helpers::object_is_visible;

impl super::super::Scene {
    /// Set the complete group membership mask for an object.
    ///
    /// Replaces the object's entire group membership with a new mask. The change
    /// is reflected in the GPU visibility buffer immediately (without triggering
    /// a full rebuild).
    ///
    /// # Parameters
    /// - `id`: Object handle
    /// - `mask`: New group membership mask
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the group membership was successfully updated.
    ///
    /// # Performance
    /// - CPU cost: O(1) - updates CPU record and GPU visibility slot
    /// - GPU cost: O(1) - writes to single visibility buffer slot
    /// - Memory: No allocations
    ///
    /// # Visibility Semantics
    ///
    /// An object is **hidden** if **any** of its groups are currently hidden.
    /// Use [`GroupMask::NONE`] for objects that should always be visible.
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::{GroupId, GroupMask};
    ///
    /// // Make object belong to groups 0 and 2
    /// let mask = GroupMask::from_id(GroupId(0))
    ///     .with(GroupId(2));
    /// scene.set_object_groups(obj_id, mask)?;
    ///
    /// // Object is now hidden if group 0 OR group 2 is hidden
    /// ```
    pub fn set_object_groups(&mut self, id: ObjectId, mask: GroupMask) -> Result<()> {
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        record.groups = mask;
        if !self.objects_dirty {
            let vis = if object_is_visible(mask, self.group_hidden) {
                1u32
            } else {
                0u32
            };
            let slot = record.draw.first_instance as usize;
            self.gpu_scene.visibility.update(slot, vis);
        }
        Ok(())
    }

    /// Add an object to a group (additive — other groups are kept).
    ///
    /// Adds the specified group to the object's membership mask without affecting
    /// other group memberships.
    ///
    /// # Parameters
    /// - `id`: Object handle
    /// - `group`: Group to add the object to
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the object was successfully added to the group.
    ///
    /// # Performance
    /// - CPU cost: O(1) - updates CPU record and GPU visibility slot
    /// - GPU cost: O(1) - writes to single visibility buffer slot
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// // Add object to group 1 (keeps existing group memberships)
    /// scene.add_object_to_group(obj_id, GroupId(1))?;
    ///
    /// // Object is now in groups 0, 1, and 2 if it was in 0 and 2 before
    /// ```
    pub fn add_object_to_group(&mut self, id: ObjectId, group: GroupId) -> Result<()> {
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        let new_mask = record.groups.with(group);
        record.groups = new_mask;
        if !self.objects_dirty {
            let vis = if object_is_visible(new_mask, self.group_hidden) {
                1u32
            } else {
                0u32
            };
            let slot = record.draw.first_instance as usize;
            self.gpu_scene.visibility.update(slot, vis);
        }
        Ok(())
    }

    /// Remove an object from a group.
    ///
    /// Removes the specified group from the object's membership mask without affecting
    /// other group memberships.
    ///
    /// # Parameters
    /// - `id`: Object handle
    /// - `group`: Group to remove the object from
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object ID is invalid
    ///
    /// # Returns
    /// `Ok(())` if the object was successfully removed from the group.
    ///
    /// # Performance
    /// - CPU cost: O(1) - updates CPU record and GPU visibility slot
    /// - GPU cost: O(1) - writes to single visibility buffer slot
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// // Remove object from group 1 (keeps other group memberships)
    /// scene.remove_object_from_group(obj_id, GroupId(1))?;
    ///
    /// // Object is now only in groups 0 and 2 if it was in 0, 1, and 2 before
    /// ```
    pub fn remove_object_from_group(&mut self, id: ObjectId, group: GroupId) -> Result<()> {
        let Some((_, record)) = self.objects.get_mut_with_index(id) else {
            return Err(invalid("object"));
        };
        let new_mask = record.groups.without(group);
        record.groups = new_mask;
        if !self.objects_dirty {
            let vis = if object_is_visible(new_mask, self.group_hidden) {
                1u32
            } else {
                0u32
            };
            let slot = record.draw.first_instance as usize;
            self.gpu_scene.visibility.update(slot, vis);
        }
        Ok(())
    }

    /// Return the group membership mask for an object.
    ///
    /// Queries which groups an object currently belongs to.
    ///
    /// # Parameters
    /// - `id`: Object handle
    ///
    /// # Errors
    /// - [`SceneError::InvalidHandle`](super::super::SceneError::InvalidHandle) if the object ID is invalid
    ///
    /// # Returns
    /// The object's current group membership mask.
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// let mask = scene.object_groups(obj_id)?;
    /// if mask.contains(GroupId(0)) {
    ///     println!("Object is in group 0");
    /// }
    /// ```
    pub fn object_groups(&self, id: ObjectId) -> Result<GroupMask> {
        let Some((_, record)) = self.objects.get_with_index(id) else {
            return Err(invalid("object"));
        };
        Ok(record.groups)
    }
}

