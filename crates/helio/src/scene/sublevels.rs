//! Sublevels — self-contained scenes re-rendered off-screen every frame and
//! composited back into the main frame under a single placement transform.
//!
//! See `docs/portals_and_sublevels.md` for the full design. In short: objects
//! belonging to a sublevel's [`GroupId`] keep **sublevel-local** transforms;
//! [`Scene::move_sublevel`] / [`Scene::update_sublevel`] change only a single
//! placement matrix (O(1)), never the member objects' instance data. The
//! actual off-screen render and composite are `SecondaryGBufferPass` /
//! `ProxyCompositePass`; this module owns the CPU-side bookkeeping that feeds
//! them (see [`super::secondary_views`]).
//!
//! # Membership tracking
//!
//! [`Scene::add_sublevel`] walks the group's *current* members once and marks
//! them (`INSTANCE_FLAG_SUBLEVEL_HIDDEN` + a membership nibble, both in
//! [`libhelio::GpuInstanceData::flags`] — see `libhelio::instance` for why
//! that field and not a new one). If you add or remove objects from a
//! sublevel's group *after* creating the sublevel, call
//! [`Scene::refresh_sublevel_membership`] to re-walk it — membership is not
//! automatically tracked on every `set_object_groups` call, the same
//! deliberate trade-off `move_group`'s O(N) walk already makes elsewhere in
//! this file's sibling modules.

use glam::{Mat4, Vec3};

use crate::groups::GroupId;
use crate::handles::{MaterialId, MeshId, ObjectId, SublevelId};
use crate::mesh::{MeshUpload, PackedVertex};

use super::core::Scene;
use super::errors::{invalid, Result};
use super::types::ObjectDescriptor;

/// Describes a new sublevel.
#[derive(Debug, Clone, Copy)]
pub struct SublevelDescriptor {
    /// Objects in this group are rendered as the sublevel's contents. Their
    /// [`libhelio::GpuInstanceData::model`] transforms are interpreted as
    /// **local to the sublevel origin**, not world space.
    pub group: GroupId,
    /// Local → world placement. The only thing [`Scene::move_sublevel`] /
    /// [`Scene::update_sublevel`] ever changes.
    pub placement: Mat4,
    /// Application-defined tag, mirroring [`super::types::ObjectDescriptor::user_tag`].
    pub user_tag: u64,
}

impl Default for SublevelDescriptor {
    fn default() -> Self {
        Self {
            group: GroupId::DEFAULT,
            placement: Mat4::IDENTITY,
            user_tag: 0,
        }
    }
}

pub(in crate::scene) struct SublevelRecord {
    pub group: GroupId,
    pub placement: Mat4,
    /// `Some(0..MAX_SUBLEVEL_VIEWS)` while this sublevel occupies a secondary
    /// camera slot and renders; `None` while it has fallen back to normal
    /// (unhidden, main-pass) rendering because every view slot was already
    /// taken — see [`Scene::add_sublevel`]'s doc comment.
    pub view_slot: Option<u32>,
    /// The coarse shadow-proxy box object standing in for this sublevel's
    /// whole interior (design doc §10 "Shadows") — `None` when the group has
    /// no members yet (nothing to bound) or proxy-asset creation failed.
    /// `local_aabb` is the min/max this proxy was last sized from, in
    /// sublevel-local space (union of member `ObjectRecord::aabb`, which is
    /// already local since member `instance.model` is local — see this
    /// module's doc comment) — kept so `move_sublevel`/`update_sublevel` can
    /// recompute the proxy's *world* transform in O(1) without re-walking
    /// membership.
    pub proxy_object: Option<ObjectId>,
    pub local_aabb: Option<(Vec3, Vec3)>,
    #[allow(dead_code)] // surfaced via a future by-tag lookup; kept for API parity with other resources
    pub user_tag: u64,
}

impl Scene {
    /// Register a sublevel over `desc.group`'s current members.
    ///
    /// If a secondary view slot is free (`< MAX_SUBLEVEL_VIEWS` sublevels
    /// currently active), the sublevel becomes active immediately: its
    /// members are hidden from the main pass and shadow atlas
    /// (`INSTANCE_FLAG_SUBLEVEL_HIDDEN`) and tagged with this sublevel's
    /// membership nibble, ready for `SecondaryGBufferPass` to pick up next
    /// frame.
    ///
    /// If every slot is taken, the sublevel is created **inactive**: its
    /// group is left visible and rendered normally through the main pass,
    /// exactly as if it were an ordinary group. Nothing is ever silently
    /// dropped — an inactive sublevel still exists, can still be moved (the
    /// placement is just unused until a slot frees up), and becomes active
    /// automatically the next time [`Scene::add_sublevel`] or
    /// [`Scene::remove_sublevel`] causes a slot to be reconsidered.
    pub fn add_sublevel(&mut self, desc: SublevelDescriptor) -> SublevelId {
        let view_slot = self.allocate_sublevel_view_slot();
        let (id, _) = self.sublevels.insert(SublevelRecord {
            group: desc.group,
            placement: desc.placement,
            view_slot,
            proxy_object: None,
            local_aabb: None,
            user_tag: desc.user_tag,
        });
        if view_slot.is_some() {
            self.set_sublevel_group_flags(desc.group, view_slot);
        }
        self.sync_sublevel_shadow_proxy(id);
        self.secondary_dirty = true;
        id
    }

    /// Move a sublevel by a placement delta: `new_placement = delta * old_placement`.
    ///
    /// O(1) — writes one matrix on this record. No member object's instance
    /// data is touched; this is the whole point of sublevels (design doc
    /// §1.1). The actual camera-slot re-derivation happens once per frame in
    /// [`Scene::update_secondary_views`](super::secondary_views).
    pub fn move_sublevel(&mut self, id: SublevelId, delta: Mat4) -> Result<()> {
        let record = self.sublevels.get_mut(id).ok_or_else(|| invalid("sublevel"))?;
        record.placement = delta * record.placement;
        self.secondary_dirty = true;
        // O(1): one additional object's transform (the shadow proxy, if any)
        // — not proportional to the sublevel's member count. See
        // `move_sublevel_touches_no_member_instance_data` for what this
        // method must NOT do.
        self.update_sublevel_shadow_proxy_transform(id);
        Ok(())
    }

    /// Set a sublevel's placement directly. O(1), same contract as [`Scene::move_sublevel`].
    pub fn update_sublevel(&mut self, id: SublevelId, placement: Mat4) -> Result<()> {
        let record = self.sublevels.get_mut(id).ok_or_else(|| invalid("sublevel"))?;
        record.placement = placement;
        self.secondary_dirty = true;
        self.update_sublevel_shadow_proxy_transform(id);
        Ok(())
    }

    /// Current placement of a sublevel.
    pub fn sublevel_placement(&self, id: SublevelId) -> Result<Mat4> {
        self.sublevels
            .get(id)
            .map(|r| r.placement)
            .ok_or_else(|| invalid("sublevel"))
    }

    /// Remove a sublevel: clears the hidden/membership flags on its current
    /// members (they resume normal main-pass rendering, at whatever raw
    /// transform they were left with — typically you want to re-place them
    /// with the last placement before removing, if the sublevel should look
    /// unchanged post-removal) and frees its view slot for reuse.
    pub fn remove_sublevel(&mut self, id: SublevelId) -> Result<()> {
        let record = self.sublevels.remove(id).ok_or_else(|| invalid("sublevel"))?;
        self.set_sublevel_group_flags(record.removed.group, None);
        if let Some(proxy) = record.removed.proxy_object {
            // Best-effort: the proxy object always exists if we created it,
            // but tolerate a caller having removed it manually.
            let _ = self.remove_object(proxy);
        }
        self.secondary_dirty = true;
        Ok(())
    }

    /// Re-walk a sublevel's group membership and reapply the hidden +
    /// membership flags.
    ///
    /// Call this after changing which objects belong to a sublevel's group
    /// (`set_object_groups`, `insert_object` into the group, etc.) —
    /// membership is not tracked automatically, matching `move_group`'s O(N)
    /// walk-on-demand shape elsewhere in this crate.
    pub fn refresh_sublevel_membership(&mut self, id: SublevelId) -> Result<()> {
        let record = self.sublevels.get(id).ok_or_else(|| invalid("sublevel"))?;
        let group = record.group;
        let view_slot = record.view_slot;
        self.set_sublevel_group_flags(group, view_slot);
        self.sync_sublevel_shadow_proxy(id);
        Ok(())
    }

    /// Number of sublevels currently occupying a secondary camera slot
    /// (`<= helio_secondary_core::MAX_SUBLEVEL_VIEWS`).
    pub fn active_sublevel_count(&self) -> u32 {
        (0..self.sublevels.dense_len())
            .filter(|&i| {
                self.sublevels
                    .get_dense(i)
                    .is_some_and(|r| r.view_slot.is_some())
            })
            .count() as u32
    }

    /// `(group, placement)` for every currently-active sublevel.
    ///
    /// Read by `ScenePicker::rebuild_instances` (in the top-level `helio`
    /// crate, outside `crate::scene`'s privacy boundary — hence a public
    /// method rather than direct field access) to compose a sublevel
    /// member's placement into its picking transform, mirroring what
    /// `helio_secondary_core::sublevel_camera` does for rendering. Small and
    /// allocation-cheap; call sites rebuild picking data on scene-change, not
    /// every frame.
    pub fn active_sublevel_placements(&self) -> Vec<(GroupId, Mat4)> {
        (0..self.sublevels.dense_len())
            .filter_map(|i| self.sublevels.get_dense(i))
            .filter(|r| r.view_slot.is_some())
            .map(|r| (r.group, r.placement))
            .collect()
    }

    fn allocate_sublevel_view_slot(&self) -> Option<u32> {
        let mut used = [false; helio_secondary_core::MAX_SUBLEVEL_VIEWS as usize];
        for i in 0..self.sublevels.dense_len() {
            if let Some(slot) = self.sublevels.get_dense(i).and_then(|r| r.view_slot) {
                used[slot as usize] = true;
            }
        }
        used.iter().position(|&taken| !taken).map(|i| i as u32)
    }

    /// Set or clear `INSTANCE_FLAG_SUBLEVEL_HIDDEN` + the membership nibble on
    /// every current member of `group`. `view_slot = None` clears both;
    /// `Some(slot)` sets hidden + membership `slot + 1`.
    ///
    /// O(N) over the dense object array — the same shape as `move_group`
    /// (`crates/helio/src/scene/groups/transforms.rs`) — but only ever called
    /// from sublevel create/remove/refresh, never per frame.
    pub(in crate::scene) fn set_sublevel_group_flags(&mut self, group: GroupId, view_slot: Option<u32>) {
        let membership = view_slot.map(|slot| slot + 1).unwrap_or(0);
        let n = self.objects.dense_len();
        for i in 0..n {
            let Some(r) = self.objects.get_dense_mut(i) else {
                continue;
            };
            if !r.groups.contains(group) {
                continue;
            }
            let mut flags = r.instance.flags;
            flags = if view_slot.is_some() {
                flags | libhelio::INSTANCE_FLAG_SUBLEVEL_HIDDEN
            } else {
                flags & !libhelio::INSTANCE_FLAG_SUBLEVEL_HIDDEN
            };
            flags = libhelio::set_sublevel_membership(flags, membership);
            r.instance.flags = flags;
            if !self.objects_dirty {
                self.gpu_scene.instances.update(r.gpu_slot as usize, r.instance);
            }
        }
    }

    // ── Shadow proxy volume (design doc §10 "Shadows") ─────────────────────

    /// (Re)build the sublevel's shadow-proxy box from its current member
    /// AABBs. Called from `add_sublevel`/`refresh_sublevel_membership` —
    /// membership-changing paths only, never per frame or from
    /// `move_sublevel`/`update_sublevel` (those only reposition the existing
    /// proxy, see `update_sublevel_shadow_proxy_transform`).
    fn sync_sublevel_shadow_proxy(&mut self, id: SublevelId) {
        let Some(record) = self.sublevels.get(id) else { return };
        let group = record.group;
        let placement = record.placement;
        let old_proxy = record.proxy_object;

        let Some((min, max)) = self.compute_local_aabb(group) else {
            // No members (yet): drop any existing proxy rather than cast a
            // shadow for an empty sublevel.
            if let Some(proxy) = old_proxy {
                let _ = self.remove_object(proxy);
            }
            if let Some(record) = self.sublevels.get_mut(id) {
                record.proxy_object = None;
                record.local_aabb = None;
            }
            return;
        };

        let (mesh, material) = self.ensure_sublevel_proxy_assets();
        let world_transform = placement * box_local_transform(min, max);
        let world_center = world_transform.w_axis.truncate();
        let radius = (max - min).length() * 0.5;

        if let Some(proxy) = old_proxy {
            let _ = self.update_object_transform(proxy, world_transform);
            let _ = self.update_object_bounds(proxy, [world_center.x, world_center.y, world_center.z, radius]);
        } else {
            let proxy = self.insert_object(ObjectDescriptor {
                mesh,
                material,
                transform: world_transform,
                bounds: [world_center.x, world_center.y, world_center.z, radius],
                flags: libhelio::INSTANCE_FLAG_CASTS_SHADOW | libhelio::INSTANCE_FLAG_SHADOW_ONLY,
                groups: crate::groups::GroupMask::NONE,
                movability: Some(libhelio::Movability::Movable),
                user_tag: 0,
            });
            if let Ok(proxy) = proxy {
                if let Some(record) = self.sublevels.get_mut(id) {
                    record.proxy_object = Some(proxy);
                }
            }
        }
        if let Some(record) = self.sublevels.get_mut(id) {
            record.local_aabb = Some((min, max));
        }
    }

    /// O(1) placement-only update: reposition the existing proxy without
    /// touching its size (`local_aabb` is unchanged by a placement move).
    /// Does nothing if the sublevel has no members (no proxy exists).
    fn update_sublevel_shadow_proxy_transform(&mut self, id: SublevelId) {
        let Some(record) = self.sublevels.get(id) else { return };
        let (Some(proxy), Some((min, max)), placement) = (record.proxy_object, record.local_aabb, record.placement) else {
            return;
        };
        let world_transform = placement * box_local_transform(min, max);
        let world_center = world_transform.w_axis.truncate();
        let radius = (max - min).length() * 0.5;
        let _ = self.update_object_transform(proxy, world_transform);
        let _ = self.update_object_bounds(proxy, [world_center.x, world_center.y, world_center.z, radius]);
    }

    /// Union of every current member's (already sublevel-local — see this
    /// module's doc comment) `ObjectRecord::aabb`. `None` if the group has no
    /// members. O(N) over the dense object array, same shape as
    /// `set_sublevel_group_flags`.
    fn compute_local_aabb(&self, group: GroupId) -> Option<(Vec3, Vec3)> {
        let mut min = Vec3::splat(f32::MAX);
        let mut max = Vec3::splat(f32::MIN);
        let mut any = false;
        for i in 0..self.objects.dense_len() {
            let Some(r) = self.objects.get_dense(i) else { continue };
            if !r.groups.contains(group) {
                continue;
            }
            any = true;
            min = min.min(Vec3::from(r.aabb.min));
            max = max.max(Vec3::from(r.aabb.max));
        }
        any.then_some((min, max))
    }

    /// Lazily create (once) the shared unit-box mesh + flat placeholder
    /// material every sublevel's shadow proxy reuses.
    fn ensure_sublevel_proxy_assets(&mut self) -> (MeshId, MaterialId) {
        if self.sublevel_proxy_mesh.is_none() {
            let id = self.insert_mesh(unit_box_mesh());
            self.sublevel_proxy_mesh = Some(id);
        }
        if self.sublevel_proxy_material.is_none() {
            let id = self.insert_material(libhelio::GpuMaterial {
                base_color: [0.0, 0.0, 0.0, 1.0],
                emissive: [0.0, 0.0, 0.0, 0.0],
                roughness_metallic: [1.0, 0.0, 1.5, 0.0],
                tex_base_color: libhelio::GpuMaterial::NO_TEXTURE,
                tex_normal: libhelio::GpuMaterial::NO_TEXTURE,
                tex_roughness: libhelio::GpuMaterial::NO_TEXTURE,
                tex_emissive: libhelio::GpuMaterial::NO_TEXTURE,
                tex_occlusion: libhelio::GpuMaterial::NO_TEXTURE,
                workflow: 0,
                flags: 0,
                material_class: 0,
                class_params: [0.0; 4],
            });
            self.sublevel_proxy_material = Some(id);
        }
        (self.sublevel_proxy_mesh.unwrap(), self.sublevel_proxy_material.unwrap())
    }
}

/// The local→local transform placing a unit box (`-0.5..0.5` each axis) so it
/// covers `[min, max]`: scale to the AABB's extent, translate to its center.
fn box_local_transform(min: Vec3, max: Vec3) -> Mat4 {
    let extent = (max - min).max(Vec3::splat(0.01)); // guard a degenerate (zero-thickness) AABB
    let center = (min + max) * 0.5;
    Mat4::from_scale_rotation_translation(extent, glam::Quat::IDENTITY, center)
}

/// A closed unit box (`-0.5..0.5` on each axis), 24 vertices (4 per face, for
/// correct flat per-face normals/tangents), 36 indices. Scaled per-sublevel
/// via the shadow-proxy object's own transform (`box_local_transform`) rather
/// than baked per-sublevel — one shared mesh, many instances.
fn unit_box_mesh() -> MeshUpload {
    const H: f32 = 0.5;
    // (normal, tangent, 4 CCW-from-outside corners)
    let faces: [([f32; 3], [f32; 3], [[f32; 3]; 4]); 6] = [
        ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [[H, -H, H], [H, -H, -H], [H, H, -H], [H, H, H]]),
        ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [[-H, -H, -H], [-H, -H, H], [-H, H, H], [-H, H, -H]]),
        ([0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [[-H, H, H], [H, H, H], [H, H, -H], [-H, H, -H]]),
        ([0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [[-H, -H, -H], [H, -H, -H], [H, -H, H], [-H, -H, H]]),
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [[-H, -H, H], [H, -H, H], [H, H, H], [-H, H, H]]),
        ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [[H, -H, -H], [-H, -H, -H], [-H, H, -H], [H, H, -H]]),
    ];
    const FACE_UVS: [[f32; 2]; 4] = [[0.0, 1.0], [1.0, 1.0], [1.0, 0.0], [0.0, 0.0]];

    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    for (normal, tangent, corners) in faces {
        let base = vertices.len() as u32;
        for (corner, uv) in corners.iter().zip(FACE_UVS.iter()) {
            vertices.push(PackedVertex::from_components(*corner, normal, *uv, tangent, 1.0));
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    MeshUpload { vertices, indices }
}

#[cfg(test)]
mod shadow_proxy_tests {
    use super::*;

    #[test]
    fn box_local_transform_covers_the_target_aabb_exactly() {
        let min = Vec3::new(-2.0, 0.0, 1.0);
        let max = Vec3::new(4.0, 3.0, 5.0);
        let transform = box_local_transform(min, max);

        // The unit box's own corners (-0.5..0.5) must map onto exactly [min, max].
        let mapped_min = transform.transform_point3(Vec3::splat(-0.5));
        let mapped_max = transform.transform_point3(Vec3::splat(0.5));
        assert!((mapped_min - min).length() < 1e-4, "{mapped_min:?} != {min:?}");
        assert!((mapped_max - max).length() < 1e-4, "{mapped_max:?} != {max:?}");
    }

    #[test]
    fn box_local_transform_guards_a_degenerate_flat_aabb() {
        // A perfectly flat AABB (e.g. one member object with zero-thickness
        // bounds) must not produce a zero-scale (and therefore singular,
        // un-invertible) transform.
        let flat = box_local_transform(Vec3::ZERO, Vec3::new(5.0, 0.0, 5.0));
        assert!(flat.determinant().abs() > 1e-6, "transform must stay invertible: {flat:?}");
    }

    #[test]
    fn unit_box_mesh_is_a_closed_manifold_with_outward_normals() {
        let mesh = unit_box_mesh();
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
        // Every vertex must lie exactly on the unit box's surface.
        for v in &mesh.vertices {
            let on_surface = v.position.iter().any(|&c| (c.abs() - 0.5).abs() < 1e-5);
            assert!(on_surface, "vertex {:?} is not on the unit box surface", v.position);
        }
    }
}
