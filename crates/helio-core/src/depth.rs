//! The engine's depth convention: reversed Z with an infinite far plane.
//!
//! Generic: every pass that tests against, writes, or reads back the main
//! depth buffer shares this one convention, so it is policy of the core rather
//! than of any pass.
//!
//! The camera projection maps the near plane to depth 1 and infinity to depth
//! 0. A float depth buffer stores values near 0 with the most precision, and
//! the reversed mapping puts those values far from the camera, where `1/z`
//! leaves the least. The result is close to uniform *relative* precision at
//! every distance — what a scene spanning centimetres to a planet's horizon
//! needs. The standard mapping (near 0, far 1) spends almost all of its bits
//! within a few metres and leaves about `distance² / near · 2⁻²⁴` per step
//! further out: 100 m of slop at 10 km with a 5 cm near plane.
//!
//! What follows from it:
//! - the buffer clears to [`CLEAR`] (0.0, "nothing drawn" = infinitely far);
//! - nearer surfaces win with [`NEARER`] / [`NEARER_EQUAL`];
//! - a depth bias toward the camera is positive ([`bias_toward_camera`]);
//! - the projection has no far plane, so `GpuCameraUniforms::forward_far.w`
//!   is only the range of distance-limited effects (froxels, fog), never a
//!   clip distance;
//! - frustum culling has five planes, not six ([`frustum_planes`]).
//!
//! Shadow maps are their own views with their own projections and keep the
//! standard mapping; nothing here applies to them.
//!
//! The WGSL side of the same convention lives in the shader prelude
//! (`helio_is_sky`, `helio_view_depth`, `helio_depth_nearest`, ...).

use glam::{Mat4, Vec3, Vec4};

/// Depth-buffer clear value: no surface, i.e. infinitely far.
pub const CLEAR: f32 = 0.0;

/// Depth test that keeps the nearer surface.
pub const NEARER: wgpu::CompareFunction = wgpu::CompareFunction::Greater;

/// Depth test that keeps the nearer surface, and an equally near one.
pub const NEARER_EQUAL: wgpu::CompareFunction = wgpu::CompareFunction::GreaterEqual;

/// The engine's camera projection: right-handed, wgpu `[0, 1]` NDC depth,
/// reversed, with no far plane.
pub fn perspective(fov_y_radians: f32, aspect: f32, near: f32) -> Mat4 {
    glam::camera::rh::proj::directx::perspective_infinite_reverse(fov_y_radians, aspect, near)
}

/// An orthographic camera projection on the engine convention: right-handed,
/// wgpu `[0, 1]` NDC depth, `near` at depth 1 and `far` at depth 0. Unlike a
/// perspective one it needs a far plane — its depth is linear in distance.
pub fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    glam::camera::rh::proj::directx::orthographic(left, right, bottom, top, far, near)
}

/// Positive view-space depth of a depth-buffer value under `proj`.
///
/// Inverts only the projection's z and w rows, so it holds for perspective
/// (finite or infinite) and orthographic projections with wgpu NDC depth,
/// reversed or not, and needs no near/far pair to be kept in sync with it.
/// Sky (`depth` 0) under the engine projection is infinitely far.
pub fn view_depth(proj: &Mat4, depth: f32) -> f32 {
    (proj.w_axis.z - depth * proj.w_axis.w) / (proj.z_axis.z - depth * proj.z_axis.w)
}

/// Whether `proj` follows the engine convention: a nearer point gets a larger
/// depth. Holds for perspective and orthographic projections alike.
pub fn is_reversed(proj: &Mat4) -> bool {
    let depth = |distance: f32| proj.project_point3(Vec3::new(0.0, 0.0, -distance)).z;
    depth(1.0) > depth(2.0)
}

/// A depth bias that pulls a surface toward the camera.
///
/// Arguments are magnitudes; the sign is the convention's business (positive
/// under reversed Z). `clamp` caps the total bias, 0 for no cap.
pub const fn bias_toward_camera(constant: i32, slope_scale: f32, clamp: f32) -> wgpu::DepthBiasState {
    wgpu::DepthBiasState { constant, slope_scale, clamp }
}

/// Frustum planes of a view-projection built on the engine projection
/// (Gribb/Hartmann).
///
/// `view_proj` is flat column-major (`Mat4::to_cols_array`). Returns
/// normalized planes `(nx, ny, nz, d)`, a point `P` being inside when
/// `dot(n, P) + d >= 0`: left, right, bottom, top, near. Normalized because
/// GPU culling compares the signed distance against a bounding radius.
///
/// There is no far plane. The textbook far plane `z >= 0` is row 2 of the
/// matrix, which under an infinite reversed projection is `(0, 0, 0, near)`
/// — no direction, and normalizing it divides by zero. The sixth slot is kept
/// so the GPU layout stays six planes, and holds a plane every point is in
/// front of.
pub fn frustum_planes(view_proj: [f32; 16]) -> [[f32; 4]; 6] {
    let vp = Mat4::from_cols_array(&view_proj);
    let (r0, r1, r2, r3) = (vp.row(0), vp.row(1), vp.row(2), vp.row(3));
    let normalize = |p: Vec4| (p / p.truncate().length()).to_array();
    [
        normalize(r3 + r0), // left:   -w <= x
        normalize(r3 - r0), // right:   x <= w
        normalize(r3 + r1), // bottom: -w <= y
        normalize(r3 - r1), // top:     y <= w
        normalize(r3 - r2), // near:    z <= w (near maps to 1)
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    const NEAR: f32 = 0.05;

    fn depth_at(proj: &Mat4, distance: f32) -> f32 {
        proj.project_point3(Vec3::new(0.0, 0.0, -distance)).z
    }

    #[test]
    fn near_plane_is_one_and_infinity_is_zero() {
        let proj = perspective(1.0, 16.0 / 9.0, NEAR);
        assert!((depth_at(&proj, NEAR) - 1.0).abs() < 1e-6);
        assert!(depth_at(&proj, 1.0e30) < 1e-20);
        assert!(depth_at(&proj, 1.0e30) >= CLEAR);
        assert!(is_reversed(&proj));
        assert!(!is_reversed(&glam::camera::rh::proj::directx::perspective(1.0, 1.0, NEAR, 100.0)));
    }

    #[test]
    fn orthographic_is_reversed_between_its_planes() {
        let proj = orthographic(-2.0, 2.0, -2.0, 2.0, 0.1, 10.0);
        assert!((depth_at(&proj, 0.1) - 1.0).abs() < 1e-6);
        assert!(depth_at(&proj, 10.0).abs() < 1e-6);
        assert!(is_reversed(&proj));
        assert!((view_depth(&proj, depth_at(&proj, 3.0)) - 3.0).abs() < 1e-5);
    }

    #[test]
    fn view_depth_round_trips_from_the_near_plane_to_planet_scale() {
        let proj = perspective(1.0, 16.0 / 9.0, NEAR);
        for distance in [NEAR, 10_000.0, 40_000_000.0] {
            let back = view_depth(&proj, depth_at(&proj, distance));
            assert!((back - distance).abs() <= distance * 1e-5, "{distance} came back as {back}");
        }
        assert_eq!(view_depth(&proj, CLEAR), f32::INFINITY);
    }

    #[test]
    fn view_depth_reads_a_standard_projection_too() {
        let proj = glam::camera::rh::proj::directx::perspective(1.0, 1.0, 0.1, 100.0);
        for distance in [0.1, 3.0, 100.0] {
            assert!((view_depth(&proj, depth_at(&proj, distance)) - distance).abs() < distance * 1e-4);
        }
    }

    #[test]
    fn neighbouring_depths_resolve_at_planet_scale() {
        // The reason for the convention: at 10 km a one-ulp step of the stored
        // depth must be centimetres, not the ~100 m of the standard mapping.
        let proj = perspective(1.0, 16.0 / 9.0, NEAR);
        let d = depth_at(&proj, 10_000.0);
        let distance = |depth: f32| f64::from(proj.w_axis.z) / f64::from(depth);
        let step = distance(d) - distance(f32::from_bits(d.to_bits() + 1));
        assert!(step > 0.0 && step < 0.01, "one ulp at 10 km spans {step} m");
    }

    #[test]
    fn frustum_planes_are_finite_and_cull_only_what_is_outside() {
        let view = glam::camera::rh::view::look_at_mat4(Vec3::new(3.0, 2.0, 5.0), Vec3::ZERO, Vec3::Y);
        let view_proj = perspective(1.0, 16.0 / 9.0, NEAR) * view;
        let planes = frustum_planes(view_proj.to_cols_array());
        assert!(planes.iter().flatten().all(|v| v.is_finite()));
        let inside = |p: Vec3| planes.iter().all(|pl| Vec3::from_slice(pl).dot(p) + pl[3] >= 0.0);
        let eye = Vec3::new(3.0, 2.0, 5.0);
        let forward = (Vec3::ZERO - eye).normalize();
        assert!(inside(Vec3::ZERO));
        assert!(inside(eye + forward * 40_000_000.0), "no far plane");
        assert!(!inside(eye + forward * (NEAR * 0.5)), "in front of the near plane");
        assert!(!inside(eye - forward), "behind the camera");
        assert!(!inside(eye + forward + view.inverse().transform_vector3(Vec3::X) * 10.0));
    }
}
