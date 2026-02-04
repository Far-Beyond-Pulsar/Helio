use glam::{Mat4, Vec3, Vec4, Quat};
use bytemuck::{Pod, Zeroable};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProjectionMode {
    Perspective { fov: f32, near: f32, far: f32 },
    Orthographic { size: f32, near: f32, far: f32 },
}

#[derive(Debug, Clone)]
pub struct Camera {
    pub position: Vec3,
    pub rotation: Quat,
    pub projection: ProjectionMode,
    pub aspect_ratio: f32,
    
    // Advanced camera features
    pub exposure: f32,
    pub focal_length: f32,
    pub aperture: f32,
    pub focus_distance: f32,
    pub sensor_size: Vec3,
    
    // Motion blur
    pub shutter_speed: f32,
    pub motion_blur_amount: f32,
    
    // Cached matrices
    view_matrix: Mat4,
    projection_matrix: Mat4,
    view_projection_matrix: Mat4,
    inverse_view_projection: Mat4,
    
    // Previous frame data for temporal effects
    prev_view_projection: Mat4,
}

impl Camera {
    pub fn new_perspective(fov: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        let mut camera = Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            projection: ProjectionMode::Perspective { fov, near, far },
            aspect_ratio,
            exposure: 1.0,
            focal_length: 50.0,
            aperture: 2.8,
            focus_distance: 10.0,
            sensor_size: Vec3::new(36.0, 24.0, 0.0),
            shutter_speed: 1.0 / 60.0,
            motion_blur_amount: 0.5,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_projection_matrix: Mat4::IDENTITY,
            inverse_view_projection: Mat4::IDENTITY,
            prev_view_projection: Mat4::IDENTITY,
        };
        camera.update_matrices();
        camera
    }
    
    pub fn new_orthographic(size: f32, aspect_ratio: f32, near: f32, far: f32) -> Self {
        let mut camera = Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            projection: ProjectionMode::Orthographic { size, near, far },
            aspect_ratio,
            exposure: 1.0,
            focal_length: 50.0,
            aperture: 16.0,
            focus_distance: 10.0,
            sensor_size: Vec3::new(36.0, 24.0, 0.0),
            shutter_speed: 1.0 / 60.0,
            motion_blur_amount: 0.0,
            view_matrix: Mat4::IDENTITY,
            projection_matrix: Mat4::IDENTITY,
            view_projection_matrix: Mat4::IDENTITY,
            inverse_view_projection: Mat4::IDENTITY,
            prev_view_projection: Mat4::IDENTITY,
        };
        camera.update_matrices();
        camera
    }
    
    pub fn update_matrices(&mut self) {
        self.prev_view_projection = self.view_projection_matrix;
        
        let forward = self.rotation * Vec3::NEG_Z;
        let up = self.rotation * Vec3::Y;
        self.view_matrix = Mat4::look_at_rh(
            self.position,
            self.position + forward,
            up,
        );
        
        self.projection_matrix = match self.projection {
            ProjectionMode::Perspective { fov, near, far } => {
                Mat4::perspective_rh(fov, self.aspect_ratio, near, far)
            }
            ProjectionMode::Orthographic { size, near, far } => {
                let half_width = size * self.aspect_ratio * 0.5;
                let half_height = size * 0.5;
                Mat4::orthographic_rh(-half_width, half_width, -half_height, half_height, near, far)
            }
        };
        
        self.view_projection_matrix = self.projection_matrix * self.view_matrix;
        self.inverse_view_projection = self.view_projection_matrix.inverse();
    }
    
    pub fn view_matrix(&self) -> Mat4 {
        self.view_matrix
    }
    
    pub fn projection_matrix(&self) -> Mat4 {
        self.projection_matrix
    }
    
    pub fn view_projection_matrix(&self) -> Mat4 {
        self.view_projection_matrix
    }
    
    pub fn inverse_view_projection(&self) -> Mat4 {
        self.inverse_view_projection
    }
    
    pub fn prev_view_projection(&self) -> Mat4 {
        self.prev_view_projection
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
    
    pub fn frustum_planes(&self) -> [Vec4; 6] {
        extract_frustum_planes(self.view_projection_matrix)
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CameraUniform {
    pub view_projection: Mat4,
    pub inverse_view_projection: Mat4,
    pub view: Mat4,
    pub projection: Mat4,
    pub prev_view_projection: Mat4,
    pub position: Vec3,
    pub exposure: f32,
    pub forward: Vec3,
    pub near_plane: f32,
    pub right: Vec3,
    pub far_plane: f32,
    pub up: Vec3,
    pub aspect_ratio: f32,
    pub frustum_planes: [Vec4; 6],
    pub jitter: Vec4,
}

unsafe impl Pod for CameraUniform {}
unsafe impl Zeroable for CameraUniform {}

impl From<&Camera> for CameraUniform {
    fn from(camera: &Camera) -> Self {
        let (near, far) = match camera.projection {
            ProjectionMode::Perspective { near, far, .. } => (near, far),
            ProjectionMode::Orthographic { near, far, .. } => (near, far),
        };
        
        Self {
            view_projection: camera.view_projection_matrix(),
            inverse_view_projection: camera.inverse_view_projection(),
            view: camera.view_matrix(),
            projection: camera.projection_matrix(),
            prev_view_projection: camera.prev_view_projection(),
            position: camera.position,
            exposure: camera.exposure,
            forward: camera.forward(),
            near_plane: near,
            right: camera.right(),
            far_plane: far,
            up: camera.up(),
            aspect_ratio: camera.aspect_ratio,
            frustum_planes: camera.frustum_planes(),
            jitter: Vec4::ZERO,
        }
    }
}

fn extract_frustum_planes(vp: Mat4) -> [Vec4; 6] {
    let mut planes = [Vec4::ZERO; 6];
    
    // Left
    planes[0] = Vec4::new(
        vp.w_axis.x + vp.x_axis.x,
        vp.w_axis.y + vp.x_axis.y,
        vp.w_axis.z + vp.x_axis.z,
        vp.w_axis.w + vp.x_axis.w,
    );
    
    // Right
    planes[1] = Vec4::new(
        vp.w_axis.x - vp.x_axis.x,
        vp.w_axis.y - vp.x_axis.y,
        vp.w_axis.z - vp.x_axis.z,
        vp.w_axis.w - vp.x_axis.w,
    );
    
    // Bottom
    planes[2] = Vec4::new(
        vp.w_axis.x + vp.y_axis.x,
        vp.w_axis.y + vp.y_axis.y,
        vp.w_axis.z + vp.y_axis.z,
        vp.w_axis.w + vp.y_axis.w,
    );
    
    // Top
    planes[3] = Vec4::new(
        vp.w_axis.x - vp.y_axis.x,
        vp.w_axis.y - vp.y_axis.y,
        vp.w_axis.z - vp.y_axis.z,
        vp.w_axis.w - vp.y_axis.w,
    );
    
    // Near
    planes[4] = Vec4::new(
        vp.w_axis.x + vp.z_axis.x,
        vp.w_axis.y + vp.z_axis.y,
        vp.w_axis.z + vp.z_axis.z,
        vp.w_axis.w + vp.z_axis.w,
    );
    
    // Far
    planes[5] = Vec4::new(
        vp.w_axis.x - vp.z_axis.x,
        vp.w_axis.y - vp.z_axis.y,
        vp.w_axis.z - vp.z_axis.z,
        vp.w_axis.w - vp.z_axis.w,
    );
    
    for plane in &mut planes {
        let length = plane.truncate().length();
        *plane /= length;
    }
    
    planes
}
