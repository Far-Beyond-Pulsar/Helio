use glam::{Mat4, Quat, Vec3, Vec4};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    pub fov_y: f32,
    pub aspect_ratio: f32,
    pub near_plane: f32,
    pub far_plane: f32,
    pub exposure: f32,
    pub aperture: f32,
    pub focus_distance: f32,
    pub sensor_width: f32,
    pub sensor_height: f32,
    pub iso: f32,
    pub shutter_speed: f32,
}

impl Camera {
    pub fn new_perspective(fov_y: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            fov_y,
            aspect_ratio,
            near_plane: near,
            far_plane: far,
            exposure: 1.0,
            aperture: 16.0,
            focus_distance: 10.0,
            sensor_width: 36.0,
            sensor_height: 24.0,
            iso: 100.0,
            shutter_speed: 1.0 / 60.0,
        }
    }

    pub fn new_orthographic(width: f32, height: f32, near: f32, far: f32) -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            fov_y: 0.0,
            aspect_ratio: width / height,
            near_plane: near,
            far_plane: far,
            exposure: 1.0,
            aperture: 16.0,
            focus_distance: 10.0,
            sensor_width: width,
            sensor_height: height,
            iso: 100.0,
            shutter_speed: 1.0 / 60.0,
        }
    }

    pub fn view_matrix(&self) -> Mat4 {
        Mat4::from_rotation_translation(self.rotation, self.position).inverse()
    }

    pub fn projection_matrix(&self) -> Mat4 {
        if self.fov_y > 0.0 {
            Mat4::perspective_rh(self.fov_y, self.aspect_ratio, self.near_plane, self.far_plane)
        } else {
            Mat4::orthographic_rh(
                -self.sensor_width * 0.5,
                self.sensor_width * 0.5,
                -self.sensor_height * 0.5,
                self.sensor_height * 0.5,
                self.near_plane,
                self.far_plane,
            )
        }
    }

    pub fn view_projection_matrix(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }

    pub fn forward(&self) -> Vec3 {
        self.rotation * Vec3::NEG_Z
    }

    pub fn right(&self) -> Vec3 {
        self.rotation * Vec3::X
    }

    pub fn up(&self) -> Vec3 {
        self.rotation * Vec3::Y
    }

    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let forward = (target - self.position).normalize();
        let right = forward.cross(up).normalize();
        let up = right.cross(forward);
        let mat3 = glam::Mat3::from_cols(right, up, -forward);
        self.rotation = Quat::from_mat3(&mat3);
    }

    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
    }

    pub fn get_jitter_matrix(jitter_x: f32, jitter_y: f32) -> Mat4 {
        Mat4::from_translation(Vec3::new(jitter_x, jitter_y, 0.0))
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new_perspective(
            std::f32::consts::FRAC_PI_3,
            16.0 / 9.0,
            0.1,
            1000.0,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraGpuData {
    pub view_proj: Mat4,
    pub inv_view_proj: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
    pub prev_view_proj: Mat4,
    pub position: Vec3,
    pub near_plane: f32,
    pub forward: Vec3,
    pub far_plane: f32,
    pub right: Vec3,
    pub fov_y: f32,
    pub up: Vec3,
    pub aspect_ratio: f32,
    pub exposure: f32,
    pub aperture: f32,
    pub focus_distance: f32,
    pub frame_index: u32,
}

impl CameraGpuData {
    pub fn from_camera(camera: &Camera, prev_view_proj: Mat4, frame_index: u32) -> Self {
        let view = camera.view_matrix();
        let projection = camera.projection_matrix();
        let view_proj = projection * view;

        Self {
            view_proj,
            inv_view_proj: view_proj.inverse(),
            view,
            projection,
            prev_view_proj,
            position: camera.position,
            near_plane: camera.near_plane,
            forward: camera.forward(),
            far_plane: camera.far_plane,
            right: camera.right(),
            fov_y: camera.fov_y,
            up: camera.up(),
            aspect_ratio: camera.aspect_ratio,
            exposure: camera.exposure,
            aperture: camera.aperture,
            focus_distance: camera.focus_distance,
            frame_index,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_matrix(matrix: Mat4) -> Self {
        let mut planes = [Vec4::ZERO; 6];

        planes[0] = matrix.row(3) + matrix.row(0);
        planes[1] = matrix.row(3) - matrix.row(0);
        planes[2] = matrix.row(3) + matrix.row(1);
        planes[3] = matrix.row(3) - matrix.row(1);
        planes[4] = matrix.row(3) + matrix.row(2);
        planes[5] = matrix.row(3) - matrix.row(2);

        for plane in &mut planes {
            let len = plane.truncate().length();
            *plane /= len;
        }

        Self { planes }
    }

    pub fn contains_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let distance = plane.dot(center.extend(1.0));
            if distance < -radius {
                return false;
            }
        }
        true
    }

    pub fn contains_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let p = Vec3::new(
                if plane.x >= 0.0 { max.x } else { min.x },
                if plane.y >= 0.0 { max.y } else { min.y },
                if plane.z >= 0.0 { max.z } else { min.z },
            );
            if plane.dot(p.extend(1.0)) < 0.0 {
                return false;
            }
        }
        true
    }
}
