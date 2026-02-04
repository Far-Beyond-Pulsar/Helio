use glam::{Mat4, Vec3, Quat};

#[derive(Debug, Clone, Copy)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
    
    local_matrix: Mat4,
    world_matrix: Mat4,
    dirty: bool,
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
            local_matrix: Mat4::IDENTITY,
            world_matrix: Mat4::IDENTITY,
            dirty: true,
        }
    }
}

impl Transform {
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        let mut transform = Self {
            position,
            rotation,
            scale,
            local_matrix: Mat4::IDENTITY,
            world_matrix: Mat4::IDENTITY,
            dirty: true,
        };
        transform.update_matrices();
        transform
    }
    
    pub fn from_position(position: Vec3) -> Self {
        Self::new(position, Quat::IDENTITY, Vec3::ONE)
    }
    
    pub fn from_rotation(rotation: Quat) -> Self {
        Self::new(Vec3::ZERO, rotation, Vec3::ONE)
    }
    
    pub fn from_scale(scale: Vec3) -> Self {
        Self::new(Vec3::ZERO, Quat::IDENTITY, scale)
    }
    
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
        self.dirty = true;
    }
    
    pub fn set_rotation(&mut self, rotation: Quat) {
        self.rotation = rotation;
        self.dirty = true;
    }
    
    pub fn set_scale(&mut self, scale: Vec3) {
        self.scale = scale;
        self.dirty = true;
    }
    
    pub fn translate(&mut self, translation: Vec3) {
        self.position += translation;
        self.dirty = true;
    }
    
    pub fn rotate(&mut self, rotation: Quat) {
        self.rotation = rotation * self.rotation;
        self.dirty = true;
    }
    
    pub fn look_at(&mut self, target: Vec3, up: Vec3) {
        let forward = (target - self.position).normalize();
        let right = forward.cross(up).normalize();
        let new_up = right.cross(forward);
        
        self.rotation = Quat::from_mat3(&glam::Mat3::from_cols(
            right,
            new_up,
            -forward,
        ));
        self.dirty = true;
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
    
    pub fn local_matrix(&mut self) -> Mat4 {
        if self.dirty {
            self.update_matrices();
        }
        self.local_matrix
    }
    
    pub fn world_matrix(&mut self) -> Mat4 {
        if self.dirty {
            self.update_matrices();
        }
        self.world_matrix
    }
    
    fn update_matrices(&mut self) {
        self.local_matrix = Mat4::from_scale_rotation_translation(
            self.scale,
            self.rotation,
            self.position,
        );
        self.world_matrix = self.local_matrix;
        self.dirty = false;
    }
    
    pub fn set_parent_transform(&mut self, parent: &Transform) {
        if self.dirty {
            self.update_matrices();
        }
        self.world_matrix = parent.world_matrix * self.local_matrix;
    }
}
