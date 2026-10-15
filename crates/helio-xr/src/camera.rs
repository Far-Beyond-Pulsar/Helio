//! Conversion of OpenXR per-eye view poses into Helio camera uniforms.
//!
//! The byte layout of [`helio_core::GpuCameraUniforms`] is shared with the WGSL
//! `Camera` struct; we build one per eye and upload the pair with
//! `GpuCameraUniforms::upload_stereo`.

use helio_core::GpuCameraUniforms;

/// A single eye's pose in the engine's world space, plus its projection FOV.
#[derive(Debug, Clone, Copy)]
pub struct ViewPose {
    /// Eye position in engine world space.
    pub eye_position: glam::Vec3,
    /// Eye orientation in engine world space.
    pub eye_orientation: glam::Quat,
    /// Horizontal/vertical half-angles of the eye's projection.
    pub fov: openxr::Fovf,
}

impl ViewPose {
    /// Build an engine-space eye pose from a raw OpenXR view located in the
    /// stage space, transformed into the engine world by `world_from_stage`.
    pub fn from_xr(view: &openxr::View, world_from_stage: &glam::Mat4) -> Self {
        let p = view.pose;
        let orientation = glam::Quat::from_xyzw(
            p.orientation.x,
            p.orientation.y,
            p.orientation.z,
            p.orientation.w,
        );
        let position = glam::Vec3::new(p.position.x, p.position.y, p.position.z);
        let world_rotation = world_from_stage.to_scale_rotation_translation().1;
        Self {
            eye_position: world_from_stage.transform_point3(position),
            eye_orientation: (world_rotation * orientation).normalize(),
            fov: view.fov,
        }
    }

    /// World-space transform of the eye (rotation + translation).
    pub fn view_to_world_matrix(&self) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(self.eye_orientation, self.eye_position)
    }

    /// World → eye view matrix (right-handed).
    pub fn view_matrix(&self) -> glam::Mat4 {
        self.view_to_world_matrix().inverse()
    }

    /// Projection matrix from the OpenXR FOV, on the engine depth convention
    /// (reversed Z, no far plane — see `helio_core::depth`).
    pub fn projection(&self, near: f32) -> glam::Mat4 {
        projection_from_fov(self.fov, near)
    }
}

/// World-space transform of an eye pose.
pub fn view_to_world_matrix(pose: &ViewPose) -> glam::Mat4 {
    pose.view_to_world_matrix()
}

/// Build the two eye `GpuCameraUniforms` (left, right) Helio's stereo
/// `array<Camera, 2>` storage buffer expects.
///
/// Upload them with `GpuCameraUniforms::upload_stereo`.
pub fn xr_view_to_camera(
    left: &ViewPose,
    right: &ViewPose,
    near: f32,
    far: f32,
) -> [GpuCameraUniforms; 2] {
    [
        pose_to_camera(left, near, far),
        pose_to_camera(right, near, far),
    ]
}

fn pose_to_camera(pose: &ViewPose, near: f32, far: f32) -> GpuCameraUniforms {
    let view = pose.view_matrix();
    let proj = pose.projection(near);
    let view_proj = proj * view;
    GpuCameraUniforms::new(
        view,
        proj,
        pose.eye_position,
        near,
        far,
        0,
        [0.0, 0.0],
        view_proj,
    )
}

/// Projection matrix from an OpenXR `Fovf` (the standard asymmetric frustum
/// formula from the OpenXR spec, on the engine's reversed infinite depth).
pub fn projection_from_fov(fov: openxr::Fovf, near: f32) -> glam::Mat4 {
    let l = near * fov.angle_left.tan();
    let r = near * fov.angle_right.tan();
    let t = near * fov.angle_up.tan();
    let b = near * fov.angle_down.tan();

    let m00 = 2.0 * near / (r - l);
    let m11 = 2.0 * near / (t - b);
    let m20 = (r + l) / (r - l);
    let m21 = (t + b) / (t - b);

    // ── Depth convention ─────────────────────────────────────────────────────
    //
    // wgpu/WebGPU clip space is **zero-to-one** in Z, like D3D and Vulkan — not OpenGL's
    // [-1, 1]. An earlier OpenGL form here mapped the near half of the frustum to negative
    // Z, where it failed the clip test and vanished.
    //
    // On top of that the engine depth is reversed with no far plane (`helio_core::depth`):
    // clip z is the constant `near` and clip w is the view distance, so the near plane is
    // depth 1 and infinity depth 0. This matches `helio_core::depth::perspective`, which
    // the flat camera path uses — so both paths agree.
    let depth_translate = near;

    // ── Layout ───────────────────────────────────────────────────────────────
    //
    // `from_cols_array` is column-major: index 11 is col2.w and index 14 is col3.z.
    // The perspective divide term (-1) belongs in **col2.w** and the depth translate in
    // **col3.z**; they were previously swapped, which shears the frustum rather than
    // projecting it. Combined with an asymmetric OpenXR FOV — where `angle_left` and
    // `angle_right` differ in magnitude, so the frustum is genuinely off-centre — that
    // shear is mirrored between the eyes, which is why the black wedges appear in
    // opposite corners and the two views refuse to fuse.
    glam::Mat4::from_cols_array(&[
        // col 0
        m00,
        0.0,
        0.0,
        0.0,
        // col 1
        0.0,
        m11,
        0.0,
        0.0,
        // col 2
        m20,
        m21,
        0.0,
        -1.0,
        // col 3
        0.0,
        0.0,
        depth_translate,
        0.0,
    ])
}

#[cfg(test)]
mod projection_tests {
    use super::projection_from_fov;

    fn fov(l: f32, r: f32, u: f32, d: f32) -> openxr::Fovf {
        openxr::Fovf {
            angle_left: l,
            angle_right: r,
            angle_up: u,
            angle_down: d,
        }
    }

    /// Project a view-space point and return NDC.
    fn ndc(m: glam::Mat4, point: glam::Vec3) -> glam::Vec3 {
        let clip = m * point.extend(1.0);
        clip.truncate() / clip.w
    }

    #[test]
    fn depth_is_reversed_and_infinite() {
        // The engine convention: the near plane is depth 1 and depth falls toward 0 with
        // distance, never reaching it. An OpenGL [-1,1] form puts the near half of the
        // frustum at negative Z, where it fails the clip test and vanishes.
        let near = 0.05_f32;
        let m = projection_from_fov(fov(-0.8, 0.8, 0.8, -0.8), near);

        // Looking down -Z, so the near plane sits at z = -near in view space.
        let at_near = ndc(m, glam::Vec3::new(0.0, 0.0, -near));
        let at_planet = ndc(m, glam::Vec3::new(0.0, 0.0, -4.0e7));

        assert!(
            (at_near.z - 1.0).abs() < 1.0e-4,
            "near plane should map to 1, got {}",
            at_near.z
        );
        assert!(
            at_planet.z > 0.0 && at_planet.z < 1.0e-8,
            "planet-scale distance should approach 0, got {}",
            at_planet.z
        );

        let mid = ndc(m, glam::Vec3::new(0.0, 0.0, -1.0));
        assert!((0.0..=1.0).contains(&mid.z), "z {} outside [0,1]", mid.z);
    }

    #[test]
    fn symmetric_fov_matches_the_engine_perspective() {
        // A symmetric OpenXR frustum is an ordinary perspective projection, so it must
        // agree with the function the flat camera path uses. This is what pins the
        // column-major layout: swapping the -1 and the depth translate shears the matrix,
        // and only a comparison against a known-good projection catches it.
        let near = 0.1_f32;
        let half_v: f32 = 0.6;
        let half_h: f32 = 0.8;

        let m = projection_from_fov(fov(-half_h, half_h, half_v, -half_v), near);
        let aspect = half_h.tan() / half_v.tan();
        let reference = helio_core::depth::perspective(half_v * 2.0, aspect, near);

        for (a, b) in m
            .to_cols_array()
            .iter()
            .zip(reference.to_cols_array().iter())
        {
            assert!((a - b).abs() < 1.0e-4, "{m:?} != {reference:?}");
        }
    }

    #[test]
    fn asymmetric_fov_is_off_centre_in_the_expected_direction() {
        // Each eye's frustum is genuinely asymmetric, and that asymmetry is what gives the
        // two eyes their differing views — so the sign has to be right, not merely
        // non-zero.
        //
        // With `angle_left = -1.0` and `angle_right = +0.6` the frustum reaches further
        // left, so the *image centre* corresponds to a ray at (-1.0 + 0.6) / 2 = -0.2 rad,
        // i.e. left of the view axis. The view axis therefore projects to the **right** of
        // centre. Getting this backwards swaps the eyes' off-centre offsets, which fuses
        // to the wrong depth rather than failing visibly.
        let m = projection_from_fov(fov(-1.0, 0.6, 0.8, -0.8), 0.05);
        let centre = ndc(m, glam::Vec3::new(0.0, 0.0, -1.0));
        assert!(
            centre.x > 0.0,
            "view axis should sit right of centre, got {}",
            centre.x
        );

        // And the mirrored frustum must be mirrored in NDC, by the same magnitude.
        let mirrored = projection_from_fov(fov(-0.6, 1.0, 0.8, -0.8), 0.05);
        let mirrored_centre = ndc(mirrored, glam::Vec3::new(0.0, 0.0, -1.0));
        assert!((mirrored_centre.x + centre.x).abs() < 1.0e-5);
    }
}
