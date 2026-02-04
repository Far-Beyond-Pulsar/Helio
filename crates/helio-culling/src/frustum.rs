use glam::{Vec3, Vec4, Mat4};

pub struct Frustum {
    pub planes: [Vec4; 6],
}

impl Frustum {
    pub fn from_matrix(view_projection: Mat4) -> Self {
        let mut planes = [Vec4::ZERO; 6];
        
        // Extract frustum planes from view-projection matrix
        planes[0] = view_projection.row(3) + view_projection.row(0); // Left
        planes[1] = view_projection.row(3) - view_projection.row(0); // Right
        planes[2] = view_projection.row(3) + view_projection.row(1); // Bottom
        planes[3] = view_projection.row(3) - view_projection.row(1); // Top
        planes[4] = view_projection.row(3) + view_projection.row(2); // Near
        planes[5] = view_projection.row(3) - view_projection.row(2); // Far
        
        // Normalize planes
        for plane in &mut planes {
            let length = plane.truncate().length();
            *plane /= length;
        }
        
        Self { planes }
    }
    
    pub fn test_sphere(&self, center: Vec3, radius: f32) -> bool {
        for plane in &self.planes {
            let dist = plane.dot(Vec4::new(center.x, center.y, center.z, 1.0));
            if dist < -radius {
                return false;
            }
        }
        true
    }
    
    pub fn test_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let p = Vec3::new(
                if plane.x > 0.0 { max.x } else { min.x },
                if plane.y > 0.0 { max.y } else { min.y },
                if plane.z > 0.0 { max.z } else { min.z },
            );
            
            if plane.dot(Vec4::new(p.x, p.y, p.z, 1.0)) < 0.0 {
                return false;
            }
        }
        true
    }
}
