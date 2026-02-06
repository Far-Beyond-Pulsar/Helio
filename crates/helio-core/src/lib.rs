use bytemuck::{Pod, Zeroable};

#[derive(blade_macros::Vertex, Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct PackedVertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords: [f32; 2],
    pub normal: u32,
    pub tangent: u32,
}

impl PackedVertex {
    pub fn new(position: [f32; 3], normal: [f32; 3]) -> Self {
        let normal_packed = pack_snorm(normal[0], normal[1], normal[2], 0.0);
        Self {
            position,
            bitangent_sign: 1.0,
            tex_coords: [0.0, 0.0],
            normal: normal_packed,
            tangent: pack_snorm(1.0, 0.0, 0.0, 0.0),
        }
    }

    pub fn new_with_uv(position: [f32; 3], normal: [f32; 3], tex_coords: [f32; 2]) -> Self {
        let normal_packed = pack_snorm(normal[0], normal[1], normal[2], 0.0);
        Self {
            position,
            bitangent_sign: 1.0,
            tex_coords,
            normal: normal_packed,
            tangent: pack_snorm(1.0, 0.0, 0.0, 0.0),
        }
    }
}

fn pack_snorm(x: f32, y: f32, z: f32, w: f32) -> u32 {
    let to_snorm = |v: f32| ((v.clamp(-1.0, 1.0) * 127.0) as i8) as u8;
    (to_snorm(x) as u32) 
        | ((to_snorm(y) as u32) << 8) 
        | ((to_snorm(z) as u32) << 16) 
        | ((to_snorm(w) as u32) << 24)
}

pub struct Mesh {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
}

pub fn create_cube_mesh(size: f32) -> Mesh {
    let s = size / 2.0;
    let vertices = vec![
        // Front
        PackedVertex::new([-s, -s, s], [0.0, 0.0, 1.0]),
        PackedVertex::new([s, -s, s], [0.0, 0.0, 1.0]),
        PackedVertex::new([s, s, s], [0.0, 0.0, 1.0]),
        PackedVertex::new([-s, s, s], [0.0, 0.0, 1.0]),
        // Back
        PackedVertex::new([s, -s, -s], [0.0, 0.0, -1.0]),
        PackedVertex::new([-s, -s, -s], [0.0, 0.0, -1.0]),
        PackedVertex::new([-s, s, -s], [0.0, 0.0, -1.0]),
        PackedVertex::new([s, s, -s], [0.0, 0.0, -1.0]),
        // Right
        PackedVertex::new([s, -s, s], [1.0, 0.0, 0.0]),
        PackedVertex::new([s, -s, -s], [1.0, 0.0, 0.0]),
        PackedVertex::new([s, s, -s], [1.0, 0.0, 0.0]),
        PackedVertex::new([s, s, s], [1.0, 0.0, 0.0]),
        // Left
        PackedVertex::new([-s, -s, -s], [-1.0, 0.0, 0.0]),
        PackedVertex::new([-s, -s, s], [-1.0, 0.0, 0.0]),
        PackedVertex::new([-s, s, s], [-1.0, 0.0, 0.0]),
        PackedVertex::new([-s, s, -s], [-1.0, 0.0, 0.0]),
        // Top
        PackedVertex::new([-s, s, s], [0.0, 1.0, 0.0]),
        PackedVertex::new([s, s, s], [0.0, 1.0, 0.0]),
        PackedVertex::new([s, s, -s], [0.0, 1.0, 0.0]),
        PackedVertex::new([-s, s, -s], [0.0, 1.0, 0.0]),
        // Bottom
        PackedVertex::new([-s, -s, -s], [0.0, -1.0, 0.0]),
        PackedVertex::new([s, -s, -s], [0.0, -1.0, 0.0]),
        PackedVertex::new([s, -s, s], [0.0, -1.0, 0.0]),
        PackedVertex::new([-s, -s, s], [0.0, -1.0, 0.0]),
    ];
    
    let indices = vec![
        0, 1, 2, 0, 2, 3,
        4, 5, 6, 4, 6, 7,
        8, 9, 10, 8, 10, 11,
        12, 13, 14, 12, 14, 15,
        16, 17, 18, 16, 18, 19,
        20, 21, 22, 20, 22, 23,
    ];
    
    Mesh { vertices, indices }
}

pub fn create_sphere_mesh(radius: f32, sectors: u32, stacks: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    
    for i in 0..=stacks {
        let stack_angle = std::f32::consts::PI / 2.0 - i as f32 * std::f32::consts::PI / stacks as f32;
        let xy = radius * stack_angle.cos();
        let z = radius * stack_angle.sin();
        
        for j in 0..=sectors {
            let sector_angle = j as f32 * 2.0 * std::f32::consts::PI / sectors as f32;
            let x = xy * sector_angle.cos();
            let y = xy * sector_angle.sin();
            
            let normal = [x / radius, y / radius, z / radius];
            vertices.push(PackedVertex::new([x, y, z], normal));
        }
    }
    
    for i in 0..stacks {
        let k1 = i * (sectors + 1);
        let k2 = k1 + sectors + 1;
        
        for j in 0..sectors {
            if i != 0 {
                indices.push(k1 + j);
                indices.push(k2 + j);
                indices.push(k1 + j + 1);
            }
            if i != (stacks - 1) {
                indices.push(k1 + j + 1);
                indices.push(k2 + j);
                indices.push(k2 + j + 1);
            }
        }
    }
    
    Mesh { vertices, indices }
}

pub fn create_plane_mesh(width: f32, height: f32) -> Mesh {
    let w = width / 2.0;
    let h = height / 2.0;

    let uv_scale = 2.0;
    let vertices = vec![
        PackedVertex::new_with_uv([-w, 0.0, -h], [0.0, 1.0, 0.0], [0.0, 0.0]),
        PackedVertex::new_with_uv([w, 0.0, -h], [0.0, 1.0, 0.0], [uv_scale, 0.0]),
        PackedVertex::new_with_uv([w, 0.0, h], [0.0, 1.0, 0.0], [uv_scale, uv_scale]),
        PackedVertex::new_with_uv([-w, 0.0, h], [0.0, 1.0, 0.0], [0.0, uv_scale]),
    ];

    // Counter-clockwise winding when viewed from above (for backface culling)
    let indices = vec![0, 2, 1, 0, 3, 2];

    Mesh { vertices, indices }
}
