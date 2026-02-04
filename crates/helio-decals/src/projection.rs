use glam::{Mat4, Vec3};

pub struct DecalProjection {
    pub position: Vec3,
    pub rotation: glam::Quat,
    pub size: Vec3,
}

impl DecalProjection {
    pub fn new(position: Vec3, size: Vec3) -> Self {
        Self {
            position,
            rotation: glam::Quat::IDENTITY,
            size,
        }
    }
    
    pub fn projection_matrix(&self) -> Mat4 {
        let scale = Mat4::from_scale(self.size);
        let rotation = Mat4::from_quat(self.rotation);
        let translation = Mat4::from_translation(self.position);
        
        translation * rotation * scale
    }
}
