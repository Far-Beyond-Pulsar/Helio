use glam::{Mat4, Quat, Vec3};
use crate::gpu;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn new(position: Vec3, rotation: Quat, scale: Vec3) -> Self {
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn from_position(position: Vec3) -> Self {
        Self {
            position,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    pub fn to_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }

    pub fn to_blade_transform(&self) -> gpu::Transform {
        let m = self.to_matrix().transpose();
        gpu::Transform {
            x: m.x_axis.to_array().into(),
            y: m.y_axis.to_array().into(),
            z: m.z_axis.to_array().into(),
        }
    }

    pub fn from_blade_transform(t: gpu::Transform) -> Self {
        let mat = Mat4 {
            x_axis: glam::Vec4::from_array(t.x.into()),
            y_axis: glam::Vec4::from_array(t.y.into()),
            z_axis: glam::Vec4::from_array(t.z.into()),
            w_axis: glam::Vec4::W,
        }
        .transpose();
        
        let (scale, rotation, position) = mat.to_scale_rotation_translation();
        Self {
            position,
            rotation,
            scale,
        }
    }

    pub fn transform_point(&self, point: Vec3) -> Vec3 {
        self.rotation * (self.scale * point) + self.position
    }

    pub fn transform_vector(&self, vector: Vec3) -> Vec3 {
        self.rotation * (self.scale * vector)
    }

    pub fn inverse(&self) -> Self {
        let inv_rotation = self.rotation.inverse();
        let inv_scale = Vec3::ONE / self.scale;
        let inv_position = inv_rotation * (-self.position * inv_scale);
        
        Self {
            position: inv_position,
            rotation: inv_rotation,
            scale: inv_scale,
        }
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl From<Mat4> for Transform {
    fn from(mat: Mat4) -> Self {
        let (scale, rotation, position) = mat.to_scale_rotation_translation();
        Self {
            position,
            rotation,
            scale,
        }
    }
}
