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

impl Camera {
    /// This camera with its field of view widened by `factor` (>= 1): the
    /// tangent extent of the projection grows by `factor` in both axes about
    /// the frame centre, so the centre of the view is unmoved and the same
    /// pixel count covers `factor` times the view. Any projection offset (lens
    /// shift) scales with it. `1` returns the camera unchanged, bit for bit.
    ///
    /// A camera whose view has a lens that looks past the frame renders the
    /// wider view and the lens crops it back; `factor` is the lens's applied
    /// overscan. Set it before `jitter`, which the renderer adds afterwards.
    pub fn with_overscan(mut self, factor: f32) -> Self {
        if factor.is_finite() && factor > 1.0 {
            let inverse = 1.0 / factor;
            for column in [&mut self.proj.x_axis, &mut self.proj.y_axis, &mut self.proj.z_axis, &mut self.proj.w_axis] {
                column.x *= inverse;
                column.y *= inverse;
            }
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera::perspective_look_at(Vec3::ZERO, Vec3::NEG_Z, Vec3::Y, 0.8, 16.0 / 9.0, 0.1, 100.0)
    }

    /// The pixel column (of `width`) the world point `p` lands in.
    fn pixel(camera: &Camera, p: Vec3, width: f32) -> f32 {
        let clip = camera.proj * camera.view * p.extend(1.0);
        (clip.x / clip.w * 0.5 + 0.5) * width
    }

    #[test]
    fn no_overscan_leaves_the_projection_bit_for_bit() {
        for factor in [1.0, 0.5, f32::NAN, f32::INFINITY] {
            assert_eq!(camera().with_overscan(factor).proj.to_cols_array(), camera().proj.to_cols_array());
        }
    }

    #[test]
    fn overscan_widens_the_tangent_extent_about_the_centre() {
        let plain = camera();
        let wide = camera().with_overscan(1.25);
        let tan_y = |c: &Camera| 1.0 / c.proj.y_axis.y;
        let tan_x = |c: &Camera| 1.0 / c.proj.x_axis.x;
        assert!((tan_y(&wide) / tan_y(&plain) - 1.25).abs() < 1e-6);
        assert!((tan_x(&wide) / tan_x(&plain) - 1.25).abs() < 1e-6);
        // The aspect, the depth range and the view are untouched.
        assert!((wide.proj.x_axis.x / wide.proj.y_axis.y - plain.proj.x_axis.x / plain.proj.y_axis.y).abs() < 1e-6);
        assert_eq!(wide.proj.z_axis.z, plain.proj.z_axis.z);
        assert_eq!(wide.proj.w_axis.z, plain.proj.w_axis.z);
        assert_eq!(wide.view, plain.view);
        // A point on the axis stays on the same pixel; one at the old frame's
        // edge moves in to 1 / 1.25 of the half width.
        let on_axis = Vec3::new(0.0, 0.0, -5.0);
        assert!((pixel(&wide, on_axis, 1920.0) - pixel(&plain, on_axis, 1920.0)).abs() < 1e-3);
        let edge = Vec3::new(5.0 * tan_x(&plain), 0.0, -5.0);
        assert!((pixel(&plain, edge, 1920.0) - 1920.0).abs() < 1e-2);
        assert!((pixel(&wide, edge, 1920.0) - (960.0 + 960.0 / 1.25)).abs() < 1e-2);
    }
}
