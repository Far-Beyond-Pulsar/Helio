use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

/// Exactly 144 bytes. Pod-safe for direct `queue.write_buffer`.
///
/// `forward()` is derived from `view_proj_inv`, not from the view matrix column.
/// Using view-matrix columns causes jitter when the matrix is reconstructed from
/// floating point quaternions. The unprojection path is numerically stable.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct Camera {
    pub view_proj:     [[f32; 4]; 4],   // offset   0 — 64 bytes
    pub position:      [f32; 3],         // offset  64 — 12 bytes
    pub time:          f32,              // offset  76 —  4 bytes
    pub view_proj_inv: [[f32; 4]; 4],   // offset  80 — 64 bytes
}                                        // total: 144 bytes

impl Camera {
    /// Build a perspective camera.
    ///
    /// `fov_y` is in radians. `time` is the current elapsed time in seconds
    /// (used by animated materials / sky).
    pub fn perspective(
        eye:    Vec3,
        target: Vec3,
        up:     Vec3,
        fov_y:  f32,
        aspect: f32,
        near:   f32,
        far:    f32,
        time:   f32,
    ) -> Self {
        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::perspective_rh(fov_y, aspect, near, far);
        let view_proj = proj * view;
        let view_proj_inv = view_proj.inverse();
        Self {
            view_proj:     view_proj.to_cols_array_2d(),
            position:      eye.into(),
            time,
            view_proj_inv: view_proj_inv.to_cols_array_2d(),
        }
    }

    /// Build an orthographic camera (useful for shadow maps / debug).
    pub fn orthographic(
        eye:    Vec3,
        target: Vec3,
        up:     Vec3,
        left:   f32,
        right:  f32,
        bottom: f32,
        top:    f32,
        near:   f32,
        far:    f32,
        time:   f32,
    ) -> Self {
        let view = Mat4::look_at_rh(eye, target, up);
        let proj = Mat4::orthographic_rh(left, right, bottom, top, near, far);
        let view_proj = proj * view;
        let view_proj_inv = view_proj.inverse();
        Self {
            view_proj:     view_proj.to_cols_array_2d(),
            position:      eye.into(),
            time,
            view_proj_inv: view_proj_inv.to_cols_array_2d(),
        }
    }

    /// Camera forward direction derived from the inverse view-proj matrix.
    /// Unprojects the NDC z-depth vector (0,0,0) → (0,0,-1) into world space.
    pub fn forward(&self) -> Vec3 {
        let inv = Mat4::from_cols_array_2d(&self.view_proj_inv);
        let near_ws = inv.project_point3(Vec3::new(0.0, 0.0, 0.0));
        let far_ws  = inv.project_point3(Vec3::new(0.0, 0.0, 1.0));
        (far_ws - near_ws).normalize()
    }

    /// Camera right direction (derived from view-proj columns, stable).
    pub fn right(&self) -> Vec3 {
        let vp = Mat4::from_cols_array_2d(&self.view_proj);
        // Right is the first row of the view matrix, extractable from vp columns.
        let r = Vec3::new(vp.x_axis.x, vp.y_axis.x, vp.z_axis.x);
        r.normalize()
    }

    /// Camera up direction.
    pub fn up(&self) -> Vec3 {
        let vp = Mat4::from_cols_array_2d(&self.view_proj);
        let u = Vec3::new(vp.x_axis.y, vp.y_axis.y, vp.z_axis.y);
        u.normalize()
    }

    /// Unproject an NDC coordinate (x, y in [-1,1], z in [0,1]) to world space.
    pub fn unproject(&self, ndc: Vec3) -> Vec3 {
        let inv = Mat4::from_cols_array_2d(&self.view_proj_inv);
        inv.project_point3(ndc)
    }

    /// Extract frustum planes (world space) for CPU culling.
    /// Returns [left, right, bottom, top, near, far] in Ax+By+Cz+D=0 form.
    pub fn frustum_planes(&self) -> [[f32; 4]; 6] {
        let m = Mat4::from_cols_array_2d(&self.view_proj);
        let r = m.row(0);
        let u = m.row(1);
        let f = m.row(2);
        let w = m.row(3);

        let planes = [
            w + r, // left
            w - r, // right
            w + u, // bottom
            w - u, // top
            w + f, // near
            w - f, // far
        ];

        let mut out = [[0f32; 4]; 6];
        for (i, p) in planes.iter().enumerate() {
            let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
            if len > 1e-7 {
                out[i] = [p.x / len, p.y / len, p.z / len, p.w / len];
            }
        }
        out
    }
}

/// 48-byte globals uniform — all per-frame scalars in one upload.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct GlobalsUniform {
    pub frame:             u32,       // offset  0
    pub delta_time:        f32,       // offset  4
    pub light_count:       u32,       // offset  8
    pub ambient_intensity: f32,       // offset 12
    pub ambient_color:     [f32; 3],  // offset 16
    pub csm_split_count:   u32,       // offset 28
    pub rc_world_min:      [f32; 3],  // offset 32
    pub _pad0:             u32,       // offset 44
}                                     // total: 48 bytes
