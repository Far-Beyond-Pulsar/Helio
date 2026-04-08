//! Camera types and constructors.

use glam::{Mat4, Vec3};

/// Camera parameters for rendering.
///
/// Stores view and projection matrices along with derived parameters needed for
/// various rendering techniques (TAA jitter, frustum culling, etc.).
///
/// # Fields
/// - `view`: View matrix (world-to-camera transform)
/// - `proj`: Projection matrix (camera-to-clip transform)
/// - `position`: Camera position in world space
/// - `near`: Near plane distance
/// - `far`: Far plane distance
/// - `jitter`: Subpixel jitter for temporal anti-aliasing (TAA)
///
/// # Example
/// ```ignore
/// let camera = Camera::perspective_look_at(
///     Vec3::new(0.0, 5.0, 10.0), // position
///     Vec3::ZERO,                 // target
///     Vec3::Y,                    // up
///     60.0_f32.to_radians(),      // fov_y
///     16.0 / 9.0,                 // aspect
///     0.1,                        // near
///     1000.0,                     // far
/// );
/// scene.update_camera(camera);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    /// View matrix (world-to-camera transform, right-handed).
    pub view: Mat4,

    /// Projection matrix (camera-to-clip transform, reversed-Z).
    pub proj: Mat4,

    /// Camera position in world space (used for distance calculations, skybox, etc.).
    pub position: Vec3,

    /// Near plane distance in world units.
    pub near: f32,

    /// Far plane distance in world units.
    pub far: f32,

    /// Subpixel jitter offset for temporal anti-aliasing (TAA).
    ///
    /// Format: `[x, y]` in normalized device coordinates (NDC).
    /// For example, `[0.5 / width, 0.5 / height]` shifts by half a pixel.
    pub jitter: [f32; 2],
}

impl Camera {
    /// Construct a camera from explicit view and projection matrices.
    ///
    /// # Parameters
    /// - `view`: View matrix (world-to-camera transform)
    /// - `proj`: Projection matrix (camera-to-clip transform)
    /// - `position`: Camera position in world space
    /// - `near`: Near plane distance
    /// - `far`: Far plane distance
    ///
    /// # Example
    /// ```ignore
    /// let view = Mat4::look_at_rh(eye, center, up);
    /// let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
    /// let camera = Camera::from_matrices(view, proj, eye, near, far);
    /// ```
    pub fn from_matrices(view: Mat4, proj: Mat4, position: Vec3, near: f32, far: f32) -> Self {
        Self {
            view,
            proj,
            position,
            near,
            far,
            jitter: [0.0, 0.0],
        }
    }

    /// Construct a perspective camera looking at a target point.
    ///
    /// Uses right-handed coordinate system with Y-up convention.
    ///
    /// # Parameters
    /// - `position`: Camera position in world space
    /// - `target`: Point the camera is looking at
    /// - `up`: Up vector (typically `Vec3::Y`)
    /// - `fov_y_radians`: Vertical field of view in radians
    /// - `aspect`: Aspect ratio (width / height)
    /// - `near`: Near plane distance
    /// - `far`: Far plane distance
    ///
    /// # Example
    /// ```ignore
    /// let camera = Camera::perspective_look_at(
    ///     Vec3::new(0.0, 5.0, 10.0),
    ///     Vec3::ZERO,
    ///     Vec3::Y,
    ///     60.0_f32.to_radians(),
    ///     1920.0 / 1080.0,
    ///     0.1,
    ///     1000.0,
    /// );
    /// ```
    pub fn perspective_look_at(
        position: Vec3,
        target: Vec3,
        up: Vec3,
        fov_y_radians: f32,
        aspect: f32,
        near: f32,
        far: f32,
    ) -> Self {
        let view = Mat4::look_at_rh(position, target, up);
        let proj = Mat4::perspective_rh(fov_y_radians, aspect, near, far);
        Self::from_matrices(view, proj, position, near, far)
    }
}

