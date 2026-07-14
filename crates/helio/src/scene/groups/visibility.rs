//! Group visibility control for batch show/hide operations.
//!
//! Provides methods for hiding/showing entire groups of objects with efficient
//! GPU visibility buffer updates.

use crate::groups::{GroupId, GroupMask};

use super::super::helpers::object_is_visible;

impl super::super::Scene {
    /// Hide all objects that belong to a group.
    ///
    /// Sets the group's hidden bit, making all objects in this group invisible.
    /// Objects in multiple groups are hidden if **any** of their groups is hidden.
    ///
    /// # Parameters
    /// - `group`: Group to hide
    ///
    /// # Performance
    /// - CPU cost: O(1) if already hidden, O(N) if state changes (where N = object count)
    /// - GPU cost: O(N) visibility buffer updates (dirty-tracked, only changed slots)
    /// - Memory: No allocations
    ///
    /// # Idempotent
    ///
    /// Calling this on an already-hidden group is a no-op (O(1)).
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// // Hide all enemies
    /// let group_enemies = GroupId(0);
    /// scene.hide_group(group_enemies);
    ///
    /// // All objects with group_enemies in their mask are now hidden
    /// ```
    pub fn hide_group(&mut self, group: GroupId) {
        if self.group_hidden.contains(group) {
            return; // already hidden — nothing to do
        }
        self.group_hidden = self.group_hidden.with(group);
        self.flush_group_visibility();
    }

    /// Show all objects in a group (unless another one of their groups is hidden).
    ///
    /// Clears the group's hidden bit, making objects in this group visible again
    /// (unless they belong to another hidden group).
    ///
    /// # Parameters
    /// - `group`: Group to show
    ///
    /// # Performance
    /// - CPU cost: O(1) if already visible, O(N) if state changes (where N = object count)
    /// - GPU cost: O(N) visibility buffer updates (dirty-tracked, only changed slots)
    /// - Memory: No allocations
    ///
    /// # Multi-Group Objects
    ///
    /// If an object belongs to groups A and B, and both are hidden:
    /// 1. `show_group(A)` will **not** make the object visible (group B is still hidden)
    /// 2. `show_group(B)` will make the object visible (both groups are now shown)
    ///
    /// # Idempotent
    ///
    /// Calling this on an already-visible group is a no-op (O(1)).
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// // Show all UI elements
    /// let group_ui = GroupId(1);
    /// scene.show_group(group_ui);
    /// ```
    pub fn show_group(&mut self, group: GroupId) {
        if !self.group_hidden.contains(group) {
            return; // already visible — nothing to do
        }
        self.group_hidden = self.group_hidden.without(group);
        self.flush_group_visibility();
    }

    /// Return `true` if a group is currently hidden.
    ///
    /// Queries the hidden state of a specific group.
    ///
    /// # Parameters
    /// - `group`: Group to query
    ///
    /// # Returns
    /// `true` if the group is hidden, `false` if visible.
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    ///
    /// let group_enemies = GroupId(0);
    /// if scene.is_group_hidden(group_enemies) {
    ///     println!("Enemies are hidden");
    /// }
    /// ```
    pub fn is_group_hidden(&self, group: GroupId) -> bool {
        self.group_hidden.contains(group)
    }

    /// Set visibility for multiple groups at once via bitmask.
    ///
    /// Only the bits in `mask` are affected; all other groups keep their current state.
    ///
    /// # Parameters
    /// - `mask`: Bitmask of groups to modify
    /// - `visible`: `true` to show these groups, `false` to hide them
    ///
    /// # Performance
    /// - CPU cost: O(1) if no state change, O(N) if visibility changes (where N = object count)
    /// - GPU cost: O(N) visibility buffer updates (dirty-tracked)
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::{GroupId, GroupMask};
    ///
    /// // Hide groups 0, 1, and 2 in one call
    /// let mask = GroupMask::from_id(GroupId(0))
    ///     .with(GroupId(1))
    ///     .with(GroupId(2));
    /// scene.set_group_visibility(mask, false);
    ///
    /// // Show them all again
    /// scene.set_group_visibility(mask, true);
    /// ```
    pub fn set_group_visibility(&mut self, mask: GroupMask, visible: bool) {
        let new_hidden = if visible {
            GroupMask(self.group_hidden.0 & !mask.0) // clear bits → visible
        } else {
            GroupMask(self.group_hidden.0 | mask.0) // set bits → hidden
        };
        if new_hidden == self.group_hidden {
            return;
        }
        self.group_hidden = new_hidden;
        self.flush_group_visibility();
    }

    /// Internal: re-evaluate GPU visibility for every object when `group_hidden` changes.
    ///
    /// Iterates over all objects and updates their visibility buffer slots based on
    /// the new hidden group mask. Skipped entirely when a full rebuild is pending
    /// (the rebuild will compute fresh visibility from scratch).
    ///
    /// # Performance
    /// - CPU cost: O(N) iteration over dense object array
    /// - GPU cost: O(N) visibility buffer updates (dirty-tracked)
    ///
    /// # Optimization
    ///
    /// If `objects_dirty` is true (rebuild pending), this function returns immediately
    /// because the rebuild will compute correct visibility anyway.
    pub(in crate::scene) fn flush_group_visibility(&mut self) {
        if self.objects_dirty {
            return;
        }
        let group_hidden = self.group_hidden;
        let n = self.objects.dense_len();
        for i in 0..n {
            let Some(r) = self.objects.get_dense(i) else {
                continue;
            };
            let vis = if object_is_visible(r.groups, group_hidden) {
                1u32
            } else {
                0u32
            };
            let slot = r.draw.first_instance as usize;
            self.gpu_scene.visibility.update(slot, vis);
        }
    }
}
