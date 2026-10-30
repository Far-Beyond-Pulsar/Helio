//! Universal camera input for renderer frames.
//!
//! A camera is placed in world space in double precision (`eye`): a planet's
//! surface is millions of metres from the origin, where an f32 position steps
//! by half a metre. Frames are always camera-relative: the renderer picks a
//! world origin near the eye (`Renderer::origin_for`) and every pass works in
//! coordinates relative to it, where f32 keeps sub-millimetre precision. The
//! camera's view in that frame is [`Camera::view_in`].

use glam::{DMat4, DVec3, Mat4, Vec3};

/// Camera parameters supplied for a render frame.
#[derive(Debug, Clone)]
pub struct Camera {
    /// Where the eye is, in world space.
    pub eye: DVec3,
    /// World directions to view space: the view matrix of this camera with its
    /// eye at the origin (rotation only).
    pub rotation: Mat4,
    pub proj: Mat4,
    pub near: f32,
    pub far: f32,
    pub jitter: [f32; 2],
    /// Stable view identity. Pass-owned SceneDB components use this to select
    /// their settings without teaching the camera about any rendering effect.
    pub view_id: u32,
}

impl Camera {
    /// A camera at `eye` with view rotation `rotation` (world directions to
    /// view space) and projection `proj`.
    ///
    /// `proj` must follow the engine depth convention — reversed Z, see
    /// `helio_core::depth` — or every depth test in the frame is backwards.
    /// `far` is the range of distance-limited effects (fog), not a clip plane.
    pub fn new(eye: DVec3, rotation: Mat4, proj: Mat4, near: f32, far: f32) -> Self {
        debug_assert!(
            helio_core::depth::is_reversed(&proj),
            "camera projection must be reversed-Z; build it with helio_core::depth::perspective"
        );
        Self {
            eye,
            rotation,
            proj,
            near,
            far,
            jitter: [0.0, 0.0],
            view_id: 0,
        }
    }

    /// A camera from a world-space view matrix whose eye is at `position`.
    pub fn from_matrices(view: Mat4, proj: Mat4, position: Vec3, near: f32, far: f32) -> Self {
        let eye = position.as_dvec3();
        let rotation = (view.as_dmat4() * DMat4::from_translation(eye)).as_mat4();
        Self::new(eye, rotation, proj, near, far)
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
        Self::perspective_look_to(
            position.as_dvec3(),
            target - position,
            up,
            fov_y_radians,
            aspect,
            near,
            far,
        )
    }

    /// A perspective camera at world-space `eye` looking along `forward`
    /// (any length).
    pub fn perspective_look_to(
        eye: DVec3,
        forward: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        Self::new(
            eye,
            // The view's rotation must be orthonormal (passes invert it by
            // its transpose), and glam builds it from a unit direction.
            glam::camera::rh::view::look_to_mat4(Vec3::ZERO, forward.normalize(), up),
            helio_core::depth::perspective(fov_y_radians, aspect, near),
            near,
            far,
        )
    }

    /// The eye relative to a frame's world origin `origin`.
    pub fn position_in(&self, origin: DVec3) -> Vec3 {
        (self.eye - origin).as_vec3()
    }

    /// The view matrix in a frame whose world origin is `origin`: the
    /// translation in double precision, so the eye's offset from the origin
    /// keeps its precision however far from the world's origin it is.
    pub fn view_in(&self, origin: DVec3) -> Mat4 {
        (self.rotation.as_dmat4() * DMat4::from_translation(-(self.eye - origin))).as_mat4()
    }

    /// The view-projection in a frame whose world origin is `origin`.
    pub fn view_proj_in(&self, origin: DVec3) -> Mat4 {
        self.proj * self.view_in(origin)
    }

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
            for column in [
                &mut self.proj.x_axis,
                &mut self.proj.y_axis,
                &mut self.proj.z_axis,
                &mut self.proj.w_axis,
            ] {
                column.x *= inverse;
                column.y *= inverse;
            }
        }
        self
    }
}

/// Side of the cells a frame's world origin snaps to. Within one cell of the
/// origin an f32 position steps by under 0.2 mm, and the snapped origin is
/// exact in f32 at any planet's radius.
pub const WORLD_ORIGIN_CELL: f64 = 1024.0;

/// The world origin of a frame with its eye at `eye`: `previous` while the
/// eye stays within a cell of it, else the cell corner nearest the eye.
/// Positions rebased to the origin are recomputed only when it moves, not
/// every frame the eye does, so static draws and shadow caches survive camera
/// motion.
pub fn follow_origin(previous: Option<DVec3>, eye: DVec3) -> DVec3 {
    match previous {
        Some(origin) if (eye - origin).abs().max_element() <= WORLD_ORIGIN_CELL => origin,
        _ => (eye / WORLD_ORIGIN_CELL).round() * WORLD_ORIGIN_CELL,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> Camera {
        Camera::perspective_look_at(
            Vec3::ZERO,
            Vec3::NEG_Z,
            Vec3::Y,
            0.8,
            16.0 / 9.0,
            0.1,
            100.0,
        )
    }

    /// The pixel column (of `width`) the world point `p` lands in.
    fn pixel(camera: &Camera, p: Vec3, width: f32) -> f32 {
        let clip = camera.view_proj_in(DVec3::ZERO) * p.extend(1.0);
        (clip.x / clip.w * 0.5 + 0.5) * width
    }

    #[test]
    fn no_overscan_leaves_the_projection_bit_for_bit() {
        for factor in [1.0, 0.5, f32::NAN, f32::INFINITY] {
            assert_eq!(
                camera().with_overscan(factor).proj.to_cols_array(),
                camera().proj.to_cols_array()
            );
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
        assert!(
            (wide.proj.x_axis.x / wide.proj.y_axis.y - plain.proj.x_axis.x / plain.proj.y_axis.y)
                .abs()
                < 1e-6
        );
        assert_eq!(wide.proj.z_axis.z, plain.proj.z_axis.z);
        assert_eq!(wide.proj.w_axis.z, plain.proj.w_axis.z);
        assert_eq!(wide.rotation, plain.rotation);
        // A point on the axis stays on the same pixel; one at the old frame's
        // edge moves in to 1 / 1.25 of the half width.
        let on_axis = Vec3::new(0.0, 0.0, -5.0);
        assert!((pixel(&wide, on_axis, 1920.0) - pixel(&plain, on_axis, 1920.0)).abs() < 1e-3);
        let edge = Vec3::new(5.0 * tan_x(&plain), 0.0, -5.0);
        assert!((pixel(&plain, edge, 1920.0) - 1920.0).abs() < 1e-2);
        assert!((pixel(&wide, edge, 1920.0) - (960.0 + 960.0 / 1.25)).abs() < 1e-2);
    }

    /// A camera looking along a direction of any length (as
    /// `perspective_look_at`'s target minus eye) has an orthonormal
    /// rotation: passes invert it by its transpose.
    #[test]
    fn look_to_any_length_keeps_the_rotation_orthonormal() {
        let camera = Camera::perspective_look_to(
            DVec3::new(0.0, 200.0, 0.0),
            Vec3::new(1.0, 0.45, 0.2),
            Vec3::Y,
            1.2,
            16.0 / 9.0,
            0.1,
            1e7,
        );
        let r = glam::Mat3::from_mat4(camera.rotation);
        assert!((r * r.transpose()).abs_diff_eq(glam::Mat3::IDENTITY, 1e-5));
        let at = Camera::perspective_look_at(
            Vec3::new(0.0, 200.0, 0.0),
            Vec3::new(3.0, 201.35, 0.6),
            Vec3::Y,
            1.2,
            16.0 / 9.0,
            0.1,
            1e7,
        );
        assert!(at.rotation.abs_diff_eq(camera.rotation, 1e-5));
    }

    /// A camera's world-space view matrix comes back as it was given, from
    /// its rotation and eye.
    #[test]
    fn from_matrices_keeps_the_view() {
        let view = glam::camera::rh::view::look_at_mat4(
            Vec3::new(3.0, 4.0, 5.0),
            Vec3::new(-1.0, 0.5, 2.0),
            Vec3::Y,
        );
        let camera = Camera::from_matrices(
            view,
            helio_core::depth::perspective(1.0, 1.5, 0.1),
            Vec3::new(3.0, 4.0, 5.0),
            0.1,
            100.0,
        );
        assert!(camera.view_in(DVec3::ZERO).abs_diff_eq(view, 1e-5));
    }

    /// Far from the world's origin, a vertex near the eye projects where the
    /// exact (f64) projection puts it in the camera-relative frame; in world
    /// f32 coordinates it is off by the half metre f32 steps by out there.
    #[test]
    fn far_geometry_projects_exactly_relative_to_the_followed_origin() {
        let eye = DVec3::new(
            -3_426_954.099417845,
            4_769_291.620032467,
            -2_476_450.5275892294,
        );
        let forward = Vec3::new(0.9167773, 0.29777223, -0.2661786);
        let camera = Camera::perspective_look_to(
            eye,
            forward,
            Vec3::new(-0.39, 0.81, -0.44),
            0.8,
            1.5,
            0.1,
            1e7,
        );
        let origin = follow_origin(None, eye);
        assert!((eye - origin).abs().max_element() <= WORLD_ORIGIN_CELL / 2.0);
        let vertex = eye + forward.as_dvec3().normalize() * 25.0 + DVec3::new(0.3, 0.0, 0.0);
        let exact = {
            let view = camera.rotation.as_dmat4() * DMat4::from_translation(-eye);
            let clip = camera.proj.as_dmat4() * view * vertex.extend(1.0);
            clip.truncate() / clip.w
        };
        let clip = camera.view_proj_in(origin) * (vertex - origin).as_vec3().extend(1.0);
        let rebased = (clip.truncate() / clip.w).as_dvec3();
        assert!(
            (rebased - exact).truncate().length() < 1e-4,
            "{rebased} vs {exact}"
        );
        // Moving within a cell keeps the origin; leaving it moves it.
        assert_eq!(
            follow_origin(Some(origin), eye + DVec3::new(500.0, -400.0, 300.0)),
            origin
        );
        let far = eye + DVec3::new(5_000.0, 0.0, 0.0);
        assert!(
            (far - follow_origin(Some(origin), far)).abs().max_element() <= WORLD_ORIGIN_CELL / 2.0
        );
    }
}
