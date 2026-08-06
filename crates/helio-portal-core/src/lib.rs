//! Portal pair math: `pair_map`, eye construction, teleport, crossing detection
//! and the oblique near clip for the portal eye.
//!
//! A portal pair has two surfaces A and B in the same world. `pair_map` is the
//! rigid transform that maps A's frame to B's frame; the eye camera that renders
//! the destination side is `pair_map` applied to the viewing camera's pose, and
//! teleporting a point through A maps it by `pair_map`. All math here is CPU-side
//! and unit-tested; the passes build the same values into GPU uniforms.
//!
//! Conventions:
//! - A pose is a rigid `Mat4` built from `Mat4::look_at_rh`; the surface's
//!   outward normal (`forward`) is the -Z axis, matching how Helio derives camera
//!   forward from `-view.z_axis`.
//! - The oblique clip plane is expressed with the **keep side** as `dot(c, x) >= 0`.

#![warn(missing_docs)]
#![forbid(unsafe_code)]

use glam::{Mat4, Vec2, Vec3, Vec4};

// ── Portal pose ───────────────────────────────────────────────────────────────

/// Position + orientation of a portal surface.
///
/// Rigid (rotation + translation, unit scale). Built from `Mat4::look_at_rh` so
/// that `forward()` follows the -Z convention.
#[derive(Debug, Clone, Copy)]
pub struct PortalPose {
    /// Local-frame → world transform. Must be rigid.
    pub transform: Mat4,
}

impl PortalPose {
    /// World-space position of the surface.
    pub fn position(&self) -> Vec3 {
        self.transform.w_axis.truncate()
    }

    /// Surface normal pointing into the portal's own world (the -Z axis).
    pub fn forward(&self) -> Vec3 {
        -self.transform.z_axis.truncate().normalize()
    }

    /// Surface up vector (the +Y axis).
    pub fn up(&self) -> Vec3 {
        self.transform.y_axis.truncate().normalize()
    }

    /// A pose at `position` looking toward `look_at` (the surface's forward).
    pub fn from_look_at(position: Vec3, look_at: Vec3, up: Vec3) -> Self {
        Self {
            transform: glam::camera::rh::view::look_at_mat4(position, look_at, up),
        }
    }
}

// ── Portal pair ───────────────────────────────────────────────────────────────

/// Two portal surfaces connected by a `pair_map`.
#[derive(Debug, Clone, Copy)]
pub struct PortalPair {
    /// Source surface.
    pub a: PortalPose,
    /// Destination surface.
    pub b: PortalPose,
}

impl PortalPair {
    /// The rigid transform mapping A's frame to B's frame.
    pub fn pair_map(&self) -> Mat4 {
        self.b.transform * self.a.transform.inverse()
    }

    /// The inverse mapping (B's frame to A's frame).
    pub fn pair_map_inverse(&self) -> Mat4 {
        self.a.transform * self.b.transform.inverse()
    }

    /// Map a world point through the portal.
    pub fn map_point(&self, world: Vec3) -> Vec3 {
        self.pair_map().transform_point3(world)
    }

    /// Map a world direction through the portal (rotation only, no translation).
    pub fn map_direction(&self, world_dir: Vec3) -> Vec3 {
        self.pair_map().transform_vector3(world_dir)
    }

    /// Map a whole pose (position + orientation) through the portal.
    ///
    /// Since both `pair_map` and `pose.transform` are rigid, the product is rigid
    /// and the resulting pose's forward is the mapped forward.
    pub fn map_pose(&self, pose: &PortalPose) -> PortalPose {
        debug_assert!(pose.transform.is_finite());
        let map = self.pair_map();
        // Position: map the pose origin. Orientation: compose the pair-map's
        // rotation with the pose's own rotation. Both products are rigid.
        let rotation = rotation_of(map) * rotation_of(pose.transform);
        PortalPose {
            transform: Mat4::from_translation(map.transform_point3(pose.position())) * rotation,
        }
    }

    /// The eye pose that renders the destination side.
    ///
    /// A viewer at A looking through A sees the world from B's frame, so the eye
    /// pose is `map_pose(camera_pose)` and its forward is B's outward normal.
    pub fn eye_pose(&self, camera_pose: &PortalPose) -> PortalPose {
        self.map_pose(camera_pose)
    }

    /// Teleport a ray through the portal.
    pub fn teleport_ray(&self, origin: Vec3, direction: Vec3) -> (Vec3, Vec3) {
        (self.map_point(origin), self.map_direction(direction))
    }
}

/// The pure rotation part of a rigid transform.
fn rotation_of(t: Mat4) -> Mat4 {
    Mat4::from_cols_array_2d(&[
        [t.x_axis.x, t.x_axis.y, t.x_axis.z, 0.0],
        [t.y_axis.x, t.y_axis.y, t.y_axis.z, 0.0],
        [t.z_axis.x, t.z_axis.y, t.z_axis.z, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
}

// ── Crossing detection ────────────────────────────────────────────────────────

/// Signed distance from `point` to the plane through `plane_origin` with normal
/// `plane_normal` (positive on the normal side).
pub fn plane_signed_distance(plane_origin: Vec3, plane_normal: Vec3, point: Vec3) -> f32 {
    (point - plane_origin).dot(plane_normal)
}

/// True when the camera crossed `pose`'s plane between `prev_pos` and `cur_pos`
/// and `cur_pos` projects within `±half_extent` of the portal's local X/Y.
pub fn crossing_detected(
    prev_pos: Vec3,
    cur_pos: Vec3,
    pose: &PortalPose,
    half_extent: Vec2,
) -> bool {
    let n = pose.forward();
    let d_prev = plane_signed_distance(pose.position(), n, prev_pos);
    let d_cur = plane_signed_distance(pose.position(), n, cur_pos);
    // Sign change with an epsilon: landing exactly on the plane counts as a
    // crossing (otherwise the player can wedge against the surface).
    let crossed = (d_prev > 0.0) != (d_cur > 0.0) || d_cur.abs() < 1e-5;
    if !crossed {
        return false;
    }
    let local = pose.transform.inverse().transform_point3(cur_pos);
    local.x.abs() <= half_extent.x && local.y.abs() <= half_extent.y
}

// ── Oblique near clip (Oblivion-style) ────────────────────────────────────────

/// The clip plane in eye view space that cuts geometry on the player side of a
/// portal surface.
///
/// The returned plane `c` satisfies `keep  ⇔  dot(c, [x,y,z,1]) >= 0`, where the
/// keep side is the side of the portal the eye should see (the far side). The
/// plane is pushed `near` toward the eye so geometry sitting exactly on the
/// surface is not clipped (z-fighting guard).
///
/// The plane is transformed by `eye.transform.inverse().transpose()`, the
/// standard plane transform for a point transform.
pub fn oblique_clip_plane_view(eye: &PortalPose, portal: &PortalPose, near: f32) -> Vec4 {
    let n = portal.forward();
    // Keep side: n·x >= n·portal.position()  ⇒  plane (n, d), d = -n·pos.
    let d = -n.dot(portal.position());
    // Offset the plane `near` toward the eye (along -n).
    let offset = n * (-near);
    let c = Vec4::new(n.x, n.y, n.z, d - n.dot(offset));
    let v = eye.transform.inverse().transpose();
    v * c
}

/// Replace a projection's depth mapping so the view-space plane `clip` becomes
/// the near plane.
///
/// The third **row** of `proj` (the coefficients that produce clip-space `z`) is
/// replaced by `clip`, making `z_ndc = dot(clip, [x,y,z,1]) / w`. With the keep
/// side `dot(clip, x) >= 0` this yields `z_ndc ∈ [0, 1)` for kept geometry
/// (monotonic in distance, so the standard `LessEqual` depth test stays
/// correct), `z_ndc < 0` for the culled side, and a matrix that remains
/// invertible (so `inv_view_proj` reconstruction is consistent).
///
/// This is the oblique **near** clip: it is correct when the clip plane is a
/// fixed, finite distance in front of the eye (the portal case). It is not a
/// precision-optimal mapping; the far-mapping variant is a documented refinement
/// for when the clip plane approaches the eye.
pub fn oblique_near_projection(proj: Mat4, clip: Vec4) -> Mat4 {
    let mut arr = proj.to_cols_array();
    // Row 2 (glam column-major: row i lives at arr[i + 4j] for column j) — the
    // coefficients of the z-output row. Replacing them makes
    // z_clip = dot(clip, [x,y,z,w]).
    arr[2] = clip.x;
    arr[6] = clip.y;
    arr[10] = clip.z;
    arr[14] = clip.w;
    Mat4::from_cols_array(&arr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose_at(pos: Vec3, forward: Vec3) -> PortalPose {
        PortalPose::from_look_at(pos, pos + forward, Vec3::Y)
    }

    #[test]
    fn pose_forward_follows_negative_z() {
        let p = pose_at(Vec3::ZERO, Vec3::Z);
        assert!((p.forward() - Vec3::Z).length() < 1e-5);
    }

    #[test]
    fn translation_pair_maps_a_to_b() {
        let a = pose_at(Vec3::ZERO, Vec3::Z);
        let b = pose_at(Vec3::new(10.0, 0.0, 0.0), Vec3::Z);
        let pair = PortalPair { a, b };

        // A's own position maps exactly onto B's position.
        let mapped = pair.map_point(a.position());
        assert!((mapped - b.position()).length() < 1e-4);

        // A's forward maps onto B's forward.
        let fwd = pair.map_direction(a.forward());
        assert!((fwd - b.forward()).length() < 1e-4);

        // Round-trip through the inverse restores the original.
        let back = pair.pair_map_inverse().transform_point3(b.position());
        assert!((back - a.position()).length() < 1e-4);
    }

    #[test]
    fn eye_pose_lands_on_b_looking_outward() {
        let a = pose_at(Vec3::ZERO, Vec3::Z);
        let b = pose_at(Vec3::new(10.0, 0.0, 0.0), Vec3::Z);
        let pair = PortalPair { a, b };
        let camera = a; // viewer at A looking through A toward +Z

        let eye = pair.eye_pose(&camera);
        assert!((eye.position() - b.position()).length() < 1e-4);
        assert!((eye.forward() - b.forward()).length() < 1e-4);
    }

    #[test]
    fn rotation_pair_eye_faces_the_other_way() {
        // B is at the origin rotated 180° about Y: it faces -Z.
        let a = pose_at(Vec3::ZERO, Vec3::Z);
        let b = pose_at(Vec3::ZERO, -Vec3::Z);
        let pair = PortalPair { a, b };

        let eye = pair.eye_pose(&a);
        assert!((eye.position() - Vec3::ZERO).length() < 1e-4);
        assert!((eye.forward() - (-Vec3::Z)).length() < 1e-4);
    }

    #[test]
    fn teleport_ray_maps_point_and_direction() {
        let a = pose_at(Vec3::ZERO, Vec3::Z);
        let b = pose_at(Vec3::new(10.0, 0.0, 0.0), Vec3::Z);
        let pair = PortalPair { a, b };
        let (o, d) = pair.teleport_ray(Vec3::new(0.0, 0.0, 5.0), Vec3::Z);
        assert!((o - Vec3::new(10.0, 0.0, 5.0)).length() < 1e-4);
        assert!((d - Vec3::Z).length() < 1e-4);
    }

    #[test]
    fn crossing_triggers_when_walking_through_inside_bounds() {
        let portal = pose_at(Vec3::ZERO, Vec3::Z);
        let prev = Vec3::new(0.0, 0.0, -1.0);
        let cur = Vec3::new(0.0, 0.0, 1.0);
        assert!(crossing_detected(prev, cur, &portal, Vec2::splat(2.0)));
    }

    #[test]
    fn crossing_ignored_outside_bounds() {
        let portal = pose_at(Vec3::ZERO, Vec3::Z);
        let prev = Vec3::new(5.0, 0.0, -1.0);
        let cur = Vec3::new(5.0, 0.0, 1.0);
        assert!(!crossing_detected(prev, cur, &portal, Vec2::splat(2.0)));
    }

    #[test]
    fn crossing_ignored_for_parallel_motion() {
        let portal = pose_at(Vec3::ZERO, Vec3::Z);
        let prev = Vec3::new(0.0, 0.0, -1.0);
        let cur = Vec3::new(0.0, 0.0, -0.5);
        assert!(!crossing_detected(prev, cur, &portal, Vec2::splat(2.0)));
    }

    #[test]
    fn crossing_not_detected_when_stopping_short() {
        let portal = pose_at(Vec3::ZERO, Vec3::Z);
        let prev = Vec3::new(0.0, 0.0, -1.0);
        let cur = Vec3::new(0.0, 0.0, -1e-3);
        // 1mm short of the surface: outside the on-plane epsilon, no sign change.
        assert!(!crossing_detected(prev, cur, &portal, Vec2::splat(2.0)));
    }

    #[test]
    fn crossing_detected_when_landing_within_epsilon() {
        let portal = pose_at(Vec3::ZERO, Vec3::Z);
        let prev = Vec3::new(0.0, 0.0, -1.0);
        let cur = Vec3::new(0.0, 0.0, -1e-6);
        // Within the on-plane epsilon: counts as a crossing so the player can
        // not wedge against the surface.
        assert!(crossing_detected(prev, cur, &portal, Vec2::splat(2.0)));
    }

    #[test]
    fn oblique_projection_keeps_far_side_and_culls_near_side() {
        let eye = pose_at(Vec3::new(0.0, 0.0, -3.0), Vec3::Z);
        let portal = pose_at(Vec3::ZERO, Vec3::Z);
        let proj = glam::camera::rh::proj::directx::perspective(60f32.to_radians(), 16.0 / 9.0, 0.1, 1000.0);

        let clip = oblique_clip_plane_view(&eye, &portal, 0.1);
        let oblique = oblique_near_projection(proj, clip);

        // Sanity: the eye view has w = -z (right-handed), so keep/cull checks use
        // the view-projected clip point directly.
        let vp = |world: Vec3| -> Vec3 {
            let v = oblique * eye.transform * world.extend(1.0);
            v.truncate() / v.w
        };

        // Far side of the portal (keep): NDC z in [0, 1).
        let keep = vp(Vec3::new(0.0, 0.0, 5.0));
        assert!(
            keep.z >= 0.0 && keep.z < 1.0,
            "keep-side point should be visible, got {keep:?}"
        );

        // Player side (cull): NDC z < 0.
        let cull = vp(Vec3::new(0.0, 0.0, -1.0));
        assert!(cull.z < 0.0, "cull-side point should be clipped, got {cull:?}");

        // The modified projection must stay invertible for inv_view_proj.
        let round = oblique * oblique.inverse();
        let id = Mat4::IDENTITY.to_cols_array();
        let round = round.to_cols_array();
        for i in 0..16 {
            assert!((round[i] - id[i]).abs() < 1e-3);
        }
    }

    #[test]
    fn clip_plane_keep_side_matches_portal_forward() {
        let eye = pose_at(Vec3::new(0.0, 0.0, -3.0), Vec3::Z);
        let portal = pose_at(Vec3::ZERO, Vec3::Z);
        let clip = oblique_clip_plane_view(&eye, &portal, 0.1);
        // Keep side is dot(c, x) >= 0. A far-side point must satisfy it, a
        // player-side point must not.
        let far = eye.transform * Vec3::new(0.0, 0.0, 5.0).extend(1.0);
        let near = eye.transform * Vec3::new(0.0, 0.0, -1.0).extend(1.0);
        assert!(clip.dot(far) >= 0.0, "far-side point should be on the keep side");
        assert!(clip.dot(near) < 0.0, "player-side point should be on the cull side");
    }
}
