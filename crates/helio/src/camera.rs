//! Universal camera input for renderer frames.

use glam::{Mat4, Vec3};
use libhelio::PostProcessSettings;

/// Camera parameters supplied for a render frame.
#[derive(Debug, Clone)]
pub struct Camera {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
    pub near: f32,
    pub far: f32,
    pub jitter: [f32; 2],
    pub postprocess_settings: PostProcessSettings,
}

impl Camera {
    pub fn from_matrices(view: Mat4, proj: Mat4, position: Vec3, near: f32, far: f32) -> Self {
        Self {
            view,
            proj,
            position,
            near,
            far,
            jitter: [0.0, 0.0],
            postprocess_settings: PostProcessSettings::default(),
        }
    }

    pub fn perspective_look_at(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self::from_matrices(
            glam::camera::rh::view::look_at_mat4(position, target, up),
            glam::camera::rh::proj::directx::perspective(fov_y_radians, aspect, near, far),
            position,
            near,
            far,
        )
    }
}
