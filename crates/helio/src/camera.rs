//! Universal camera input for renderer frames.

use glam::{Mat4, Vec3};

/// Camera parameters supplied for a render frame.
#[derive(Debug, Clone)]
pub struct Camera {
    pub view: Mat4,
    pub proj: Mat4,
    pub position: Vec3,
    pub near: f32,
    pub far: f32,
    pub jitter: [f32; 2],
    /// Stable view identity. Pass-owned SceneDB components use this to select
    /// their settings without teaching the camera about any rendering effect.
    pub view_id: u32,
}

impl Camera {
    /// A camera from its own matrices.
    ///
    /// `proj` must follow the engine depth convention — reversed Z, see
    /// `helio_core::depth` — or every depth test in the frame is backwards.
    /// `far` is the range of distance-limited effects (fog), not a clip plane.
    pub fn from_matrices(view: Mat4, proj: Mat4, position: Vec3, near: f32, far: f32) -> Self {
        debug_assert!(
            helio_core::depth::is_reversed(&proj),
            "camera projection must be reversed-Z; build it with helio_core::depth::perspective"
        );
        Self {
            view,
            proj,
            position,
            near,
            far,
            jitter: [0.0, 0.0],
            view_id: 0,
        }
    }

    /// A perspective camera on the engine projection: reversed Z, no far
    /// plane. `far` only bounds distance-limited effects.
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
            helio_core::depth::perspective(fov_y_radians, aspect, near),
            position,
            near,
            far,
        )
    }
}
