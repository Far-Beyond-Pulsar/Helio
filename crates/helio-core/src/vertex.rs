use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub tex_coords: [f32; 2],
}

impl Vertex {
    pub fn new(position: Vec3, normal: Vec3, tangent: Vec4, tex_coords: Vec2) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            tangent: tangent.to_array(),
            tex_coords: tex_coords.to_array(),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct PackedVertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords: [f32; 2],
    pub normal: u32,
    pub tangent: u32,
}

impl PackedVertex {
    pub fn from_vertex(vertex: &Vertex) -> Self {
        let normal = Self::pack_normal(Vec3::from_array(vertex.normal));
        let tangent_vec = Vec3::from_array([vertex.tangent[0], vertex.tangent[1], vertex.tangent[2]]);
        let tangent = Self::pack_normal(tangent_vec);
        
        Self {
            position: vertex.position,
            bitangent_sign: vertex.tangent[3],
            tex_coords: vertex.tex_coords,
            normal,
            tangent,
        }
    }

    fn pack_normal(normal: Vec3) -> u32 {
        let n = (normal + Vec3::ONE) * 0.5 * 1023.0;
        let x = (n.x as u32).min(1023);
        let y = (n.y as u32).min(1023);
        let z = (n.z as u32).min(1023);
        (z << 20) | (y << 10) | x
    }

    pub fn unpack_normal(packed: u32) -> Vec3 {
        let x = (packed & 0x3FF) as f32 / 1023.0;
        let y = ((packed >> 10) & 0x3FF) as f32 / 1023.0;
        let z = ((packed >> 20) & 0x3FF) as f32 / 1023.0;
        Vec3::new(x, y, z) * 2.0 - Vec3::ONE
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub tangent: [f32; 4],
    pub tex_coords: [f32; 2],
    pub bone_indices: [u32; 4],
    pub bone_weights: [f32; 4],
}

impl SkinnedVertex {
    pub fn new(
        position: Vec3,
        normal: Vec3,
        tangent: Vec4,
        tex_coords: Vec2,
        bone_indices: [u32; 4],
        bone_weights: [f32; 4],
    ) -> Self {
        Self {
            position: position.to_array(),
            normal: normal.to_array(),
            tangent: tangent.to_array(),
            tex_coords: tex_coords.to_array(),
            bone_indices,
            bone_weights,
        }
    }
}
