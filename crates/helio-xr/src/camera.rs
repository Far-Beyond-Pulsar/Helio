//! Conversion of OpenXR per-eye view poses into Helio camera uniforms.
//!
//! The byte layout of [`libhelio::GpuCameraUniforms`] is shared with the WGSL
//! `Camera` struct; we build one per eye and upload the pair with
//! `GpuCameraUniforms::upload_stereo`.

use libhelio::GpuCameraUniforms;

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

    /// Projection matrix from the OpenXR FOV (right-handed, Z in [-1, 1] clip
    /// space — the same convention as `glam::Mat4::perspective_rh` which Helio
    /// uses elsewhere).
    pub fn projection(&self, near: f32, far: f32) -> glam::Mat4 {
        projection_from_fov(self.fov, near, far)
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
    [pose_to_camera(left, near, far), pose_to_camera(right, near, far)]
}

fn pose_to_camera(pose: &ViewPose, near: f32, far: f32) -> GpuCameraUniforms {
    let view = pose.view_matrix();
    let proj = pose.projection(near, far);
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
/// formula from the OpenXR spec).
pub fn projection_from_fov(fov: openxr::Fovf, near: f32, far: f32) -> glam::Mat4 {
    let l = near * fov.angle_left.tan();
    let r = near * fov.angle_right.tan();
    let t = near * fov.angle_up.tan();
    let b = near * fov.angle_down.tan();

    let m00 = 2.0 * near / (r - l);
    let m11 = 2.0 * near / (t - b);
    let m20 = (r + l) / (r - l);
    let m21 = (t + b) / (t - b);
    let m22 = -(far + near) / (far - near);
    let m32 = -2.0 * far * near / (far - near);

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
        m22,
        m32,
        // col 3
        0.0,
        0.0,
        -1.0,
        0.0,
    ])
}
