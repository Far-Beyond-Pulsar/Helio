//! Camera utilities

use glam::{Mat4, Vec3};

/// Camera data for rendering
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Camera {
    /// Combined view-projection matrix
    pub view_proj: Mat4,
    /// Camera position in world space
    pub position: Vec3,
    /// Elapsed time in seconds
    pub time: f32,
}

impl Camera {
    pub fn new(view_proj: Mat4, position: Vec3, time: f32) -> Self {
        Self {
            view_proj,
            position,
            time,
        }
    }

    /// Create a perspective camera
    pub fn perspective(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y: f32,
        aspect: f32,
        near: f32,
        far: f32,
        time: f32,
    ) -> Self {
        let view = Mat4::look_at_rh(position, target, up);
        let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
        let view_proj = proj * view;

        Self {
            view_proj,
            position,
            time,
        }
    }

    /// Create an orthographic camera
    pub fn orthographic(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32,
        time: f32,
    ) -> Self {
        let view = Mat4::look_at_rh(position, target, up);
        let proj = Mat4::orthographic_rh(left, right, bottom, top, near, far);
        let view_proj = proj * view;

        Self {
            view_proj,
            position,
            time,
        }
    }
}
