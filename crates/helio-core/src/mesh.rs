use glam::{Vec2, Vec3, Vec4};
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Vertex {
    pub position: Vec3,
    pub normal: Vec3,
    pub tangent: Vec4,
    pub uv0: Vec2,
    pub uv1: Vec2,
    pub color: Vec4,
}

unsafe impl Pod for Vertex {}
unsafe impl Zeroable for Vertex {}

impl Default for Vertex {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            normal: Vec3::Y,
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
            uv0: Vec2::ZERO,
            uv1: Vec2::ZERO,
            color: Vec4::ONE,
        }
    }
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub bounds: BoundingBox,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        let bounds = BoundingBox::from_vertices(&vertices);
        Self {
            vertices,
            indices,
            bounds,
        }
    }
    
    pub fn calculate_tangents(&mut self) {
        for i in (0..self.indices.len()).step_by(3) {
            let i0 = self.indices[i] as usize;
            let i1 = self.indices[i + 1] as usize;
            let i2 = self.indices[i + 2] as usize;
            
            let v0 = &self.vertices[i0];
            let v1 = &self.vertices[i1];
            let v2 = &self.vertices[i2];
            
            let edge1 = v1.position - v0.position;
            let edge2 = v2.position - v0.position;
            
            let delta_uv1 = v1.uv0 - v0.uv0;
            let delta_uv2 = v2.uv0 - v0.uv0;
            
            let f = 1.0 / (delta_uv1.x * delta_uv2.y - delta_uv2.x * delta_uv1.y);
            
            let tangent = Vec3::new(
                f * (delta_uv2.y * edge1.x - delta_uv1.y * edge2.x),
                f * (delta_uv2.y * edge1.y - delta_uv1.y * edge2.y),
                f * (delta_uv2.y * edge1.z - delta_uv1.y * edge2.z),
            ).normalize();
            
            let tangent4 = Vec4::new(tangent.x, tangent.y, tangent.z, 1.0);
            
            self.vertices[i0].tangent = tangent4;
            self.vertices[i1].tangent = tangent4;
            self.vertices[i2].tangent = tangent4;
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub min: Vec3,
    pub max: Vec3,
}

impl BoundingBox {
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }
    
    pub fn from_vertices(vertices: &[Vertex]) -> Self {
        if vertices.is_empty() {
            return Self {
                min: Vec3::ZERO,
                max: Vec3::ZERO,
            };
        }
        
        let mut min = vertices[0].position;
        let mut max = vertices[0].position;
        
        for vertex in vertices.iter().skip(1) {
            min = min.min(vertex.position);
            max = max.max(vertex.position);
        }
        
        Self { min, max }
    }
    
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }
    
    pub fn extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }
    
    pub fn radius(&self) -> f32 {
        self.extents().length()
    }
}

pub mod primitives {
    use super::*;
    
    pub fn create_cube() -> Mesh {
        let vertices = vec![
            // Front
            Vertex { position: Vec3::new(-0.5, -0.5,  0.5), normal: Vec3::NEG_Z, uv0: Vec2::new(0.0, 0.0), ..Default::default() },
            Vertex { position: Vec3::new( 0.5, -0.5,  0.5), normal: Vec3::NEG_Z, uv0: Vec2::new(1.0, 0.0), ..Default::default() },
            Vertex { position: Vec3::new( 0.5,  0.5,  0.5), normal: Vec3::NEG_Z, uv0: Vec2::new(1.0, 1.0), ..Default::default() },
            Vertex { position: Vec3::new(-0.5,  0.5,  0.5), normal: Vec3::NEG_Z, uv0: Vec2::new(0.0, 1.0), ..Default::default() },
            
            // Back
            Vertex { position: Vec3::new( 0.5, -0.5, -0.5), normal: Vec3::Z, uv0: Vec2::new(0.0, 0.0), ..Default::default() },
            Vertex { position: Vec3::new(-0.5, -0.5, -0.5), normal: Vec3::Z, uv0: Vec2::new(1.0, 0.0), ..Default::default() },
            Vertex { position: Vec3::new(-0.5,  0.5, -0.5), normal: Vec3::Z, uv0: Vec2::new(1.0, 1.0), ..Default::default() },
            Vertex { position: Vec3::new( 0.5,  0.5, -0.5), normal: Vec3::Z, uv0: Vec2::new(0.0, 1.0), ..Default::default() },
        ];
        
        let indices = vec![
            0, 1, 2, 2, 3, 0, // Front
            4, 5, 6, 6, 7, 4, // Back
        ];
        
        Mesh::new(vertices, indices)
    }
    
    pub fn create_sphere(subdivisions: u32) -> Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        let rings = subdivisions;
        let sectors = subdivisions * 2;
        
        for ring in 0..=rings {
            let theta = ring as f32 * std::f32::consts::PI / rings as f32;
            let sin_theta = theta.sin();
            let cos_theta = theta.cos();
            
            for sector in 0..=sectors {
                let phi = sector as f32 * 2.0 * std::f32::consts::PI / sectors as f32;
                let sin_phi = phi.sin();
                let cos_phi = phi.cos();
                
                let x = sin_theta * cos_phi;
                let y = cos_theta;
                let z = sin_theta * sin_phi;
                
                let u = sector as f32 / sectors as f32;
                let v = ring as f32 / rings as f32;
                
                vertices.push(Vertex {
                    position: Vec3::new(x * 0.5, y * 0.5, z * 0.5),
                    normal: Vec3::new(x, y, z),
                    uv0: Vec2::new(u, v),
                    ..Default::default()
                });
            }
        }
        
        for ring in 0..rings {
            for sector in 0..sectors {
                let current = ring * (sectors + 1) + sector;
                let next = current + sectors + 1;
                
                indices.push(current);
                indices.push(next);
                indices.push(current + 1);
                
                indices.push(current + 1);
                indices.push(next);
                indices.push(next + 1);
            }
        }
        
        Mesh::new(vertices, indices)
    }
    
    pub fn create_plane(width: f32, height: f32, subdivisions: u32) -> Mesh {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        
        for y in 0..=subdivisions {
            for x in 0..=subdivisions {
                let u = x as f32 / subdivisions as f32;
                let v = y as f32 / subdivisions as f32;
                
                let px = (u - 0.5) * width;
                let pz = (v - 0.5) * height;
                
                vertices.push(Vertex {
                    position: Vec3::new(px, 0.0, pz),
                    normal: Vec3::Y,
                    uv0: Vec2::new(u, v),
                    ..Default::default()
                });
            }
        }
        
        for y in 0..subdivisions {
            for x in 0..subdivisions {
                let i0 = y * (subdivisions + 1) + x;
                let i1 = i0 + 1;
                let i2 = (y + 1) * (subdivisions + 1) + x;
                let i3 = i2 + 1;
                
                indices.push(i0);
                indices.push(i2);
                indices.push(i1);
                
                indices.push(i1);
                indices.push(i2);
                indices.push(i3);
            }
        }
        
        Mesh::new(vertices, indices)
    }
}
