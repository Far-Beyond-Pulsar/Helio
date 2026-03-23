//! Group transform operations for batch movement and transformation.
//!
//! Provides methods for applying transforms to all objects in a group simultaneously.

use glam::{Mat4, Vec3};

use crate::groups::GroupId;

use super::super::helpers::{normal_matrix, sphere_to_aabb};

impl super::super::Scene {
    /// Apply a transform delta to every object in a group.
    ///
    /// `delta` is post-multiplied: `new_model = delta * old_model`.  This lets you
    /// pass a pure translation/rotation/scale matrix and have it applied uniformly.
    ///
    /// # Parameters
    /// - `group`: Group to transform
    /// - `delta`: Transform delta (translation, rotation, scale, or combination)
    ///
    /// # Performance
    /// - CPU cost: O(N) over the dense object array (where N = total objects, not group size)
    /// - GPU cost: O(M) updates (where M = objects in group) when layout is stable
    /// - Memory: No allocations
    ///
    /// # Transform Application
    ///
    /// The delta is applied as: `new_transform = delta * old_transform`
    ///
    /// This means:
    /// - Translation delta moves objects in **world space**
    /// - Rotation delta rotates objects around **world origin**
    /// - Scale delta scales objects from **world origin**
    ///
    /// For local-space transformations, pre-compute the world-space delta from
    /// the object's current transform.
    ///
    /// # Bounds Updates
    ///
    /// Bounding sphere centers are transformed by the delta matrix. The radius
    /// is kept unchanged (this is an approximation - for accurate bounds after
    /// non-uniform scaling, manually update bounds with [`update_object_bounds`](crate::Scene::update_object_bounds)).
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    /// use glam::{Mat4, Vec3};
    ///
    /// let group_enemies = GroupId(0);
    ///
    /// // Move all enemies up by 5 units
    /// scene.move_group(group_enemies, Mat4::from_translation(Vec3::new(0.0, 5.0, 0.0)));
    ///
    /// // Rotate all enemies 90 degrees around Y axis
    /// scene.move_group(group_enemies, Mat4::from_rotation_y(std::f32::consts::FRAC_PI_2));
    ///
    /// // Scale all enemies to 2x size
    /// scene.move_group(group_enemies, Mat4::from_scale(Vec3::splat(2.0)));
    /// ```
    pub fn move_group(&mut self, group: GroupId, delta: Mat4) {
        let n = self.objects.dense_len();
        for i in 0..n {
            let Some(r) = self.objects.get_dense_mut(i) else {
                continue;
            };
            if !r.groups.contains(group) {
                continue;
            }
            let new_transform = delta * Mat4::from_cols_array(&r.instance.model);
            r.instance.model = new_transform.to_cols_array();
            r.instance.normal_mat = normal_matrix(new_transform);
            // Update bounds center (keep radius unchanged).
            let old_center = Vec3::new(
                r.instance.bounds[0],
                r.instance.bounds[1],
                r.instance.bounds[2],
            );
            let new_center = delta.transform_point3(old_center);
            r.instance.bounds[0] = new_center.x;
            r.instance.bounds[1] = new_center.y;
            r.instance.bounds[2] = new_center.z;
            r.aabb = sphere_to_aabb(r.instance.bounds);
            if !self.objects_dirty {
                let slot = r.draw.first_instance as usize;
                self.gpu_scene.instances.update(slot, r.instance);
                self.gpu_scene.aabbs.update(slot, r.aabb);
            }
        }
    }

    /// Translate all objects in a group by a world-space delta.
    ///
    /// Convenience wrapper around [`move_group`](Self::move_group) using a pure
    /// translation matrix.
    ///
    /// # Parameters
    /// - `group`: Group to translate
    /// - `delta`: Translation vector in world space
    ///
    /// # Performance
    /// - CPU cost: O(N) over the dense object array
    /// - GPU cost: O(M) updates (where M = objects in group)
    ///
    /// # Example
    /// ```ignore
    /// use helio::groups::GroupId;
    /// use glam::Vec3;
    ///
    /// let group_ui = GroupId(1);
    ///
    /// // Move all UI elements 10 units to the right
    /// scene.translate_group(group_ui, Vec3::new(10.0, 0.0, 0.0));
    ///
    /// // Move all UI elements back to origin
    /// let current_pos = Vec3::new(10.0, 0.0, 0.0);
    /// scene.translate_group(group_ui, -current_pos);
    /// ```
    pub fn translate_group(&mut self, group: GroupId, delta: Vec3) {
        self.move_group(group, Mat4::from_translation(delta));
    }
}
