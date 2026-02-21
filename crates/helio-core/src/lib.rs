pub mod texture;

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Vertex layout matching base_geometry.wgsl (32 bytes, packed normals/tangents)
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct PackedVertex {
    pub position: [f32; 3],      // @location(0)
    pub bitangent_sign: f32,     // @location(1)
    pub tex_coords: [f32; 2],    // @location(2)
    pub normal: u32,             // @location(3) — packed via pack4x8snorm
    pub tangent: u32,            // @location(4) — packed via pack4x8snorm
}

impl PackedVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32,
        2 => Float32x2,
        3 => Uint32,
        4 => Uint32,
    ];
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
    pub fn new(position: [f32; 3], normal: [f32; 3], uv: [f32; 2], tangent: [f32; 3], bitangent_sign: f32) -> Self {
        Self {
            position,
            bitangent_sign,
            tex_coords: uv,
            normal: pack_snorm4(normal[0], normal[1], normal[2], 0.0),
            tangent: pack_snorm4(tangent[0], tangent[1], tangent[2], 0.0),
        }
    }
}

fn pack_snorm4(x: f32, y: f32, z: f32, w: f32) -> u32 {
    let xi = ((x.clamp(-1.0, 1.0) * 127.0) as i8) as u8;
    let yi = ((y.clamp(-1.0, 1.0) * 127.0) as i8) as u8;
    let zi = ((z.clamp(-1.0, 1.0) * 127.0) as i8) as u8;
    let wi = ((w.clamp(-1.0, 1.0) * 127.0) as i8) as u8;
    xi as u32 | ((yi as u32) << 8) | ((zi as u32) << 16) | ((wi as u32) << 24)
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct BillboardVertex {
    pub position: [f32; 3],      // @location(0)
    pub tex_coords: [f32; 2],    // @location(1)
}

impl BillboardVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x2,
    ];
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Mesh {
    pub vertices: Vec<PackedVertex>,
    pub indices: Vec<u32>,
    pub name: String,
}

impl Mesh {
    pub fn new(vertices: Vec<PackedVertex>, indices: Vec<u32>) -> Self {
        Self { vertices, indices, name: String::new() }
    }
    pub fn with_name(mut self, name: impl Into<String>) -> Self { self.name = name.into(); self }
}

pub struct BillboardMesh {
    pub vertices: Vec<BillboardVertex>,
    pub indices: Vec<u32>,
}

pub struct MeshBuffer {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub vertex_count: u32,
    pub index_count: u32,
    pub name: String,
}

impl MeshBuffer {
    pub fn from_mesh(device: &wgpu::Device, name: &str, mesh: &Mesh) -> Self {
        let vertex_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_vertices", name)),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        let index_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}_indices", name)),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        Self {
            vertex_buffer,
            index_buffer,
            vertex_count: mesh.vertices.len() as u32,
            index_count: mesh.indices.len() as u32,
            name: name.to_string(),
        }
    }
    pub fn vertex_buffer(&self) -> &wgpu::Buffer { &self.vertex_buffer }
    pub fn index_buffer(&self) -> &wgpu::Buffer { &self.index_buffer }
}

pub use texture::{TextureId, TextureManager, GpuTexture};

pub fn align_to(value: u32, alignment: u32) -> u32 {
    (value + alignment - 1) & !(alignment - 1)
}

// ---- mesh creation helpers ----

pub fn create_cube_mesh(size: f32) -> Mesh {
    let h = size * 0.5;
    let faces: [([f32;3], [f32;3]); 6] = [
        ([ 0.0,  0.0,  1.0], [1.0,  0.0, 0.0]), // +Z front
        ([ 0.0,  0.0, -1.0], [-1.0, 0.0, 0.0]), // -Z back
        ([ 1.0,  0.0,  0.0], [0.0,  1.0, 0.0]), // +X right
        ([-1.0,  0.0,  0.0], [0.0, -1.0, 0.0]), // -X left
        ([ 0.0,  1.0,  0.0], [1.0,  0.0, 0.0]), // +Y top
        ([ 0.0, -1.0,  0.0], [1.0,  0.0, 0.0]), // -Y bottom
    ];
    let uvs = [[0.0f32,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);

    for (face_idx, (normal, tangent)) in faces.iter().enumerate() {
        let n = *normal;
        let t = *tangent;
        // Build 4 corners for this face
        let b = cross3(n, t);
        let corners = [
            add3(scale3(t, -h), add3(scale3(b, -h), scale3(n, h))),
            add3(scale3(t,  h), add3(scale3(b, -h), scale3(n, h))),
            add3(scale3(t,  h), add3(scale3(b,  h), scale3(n, h))),
            add3(scale3(t, -h), add3(scale3(b,  h), scale3(n, h))),
        ];
        let base = (face_idx * 4) as u32;
        for (i, corner) in corners.iter().enumerate() {
            vertices.push(PackedVertex::new(*corner, n, uvs[i], t, 1.0));
        }
        indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
    }
    Mesh::new(vertices, indices)
}

pub fn create_sphere_mesh(radius: f32, lat: u32, lon: u32) -> Mesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for i in 0..=lat {
        let phi = std::f32::consts::PI * i as f32 / lat as f32;
        for j in 0..=lon {
            let theta = 2.0 * std::f32::consts::PI * j as f32 / lon as f32;
            let x = phi.sin() * theta.cos();
            let y = phi.cos();
            let z = phi.sin() * theta.sin();
            let n = [x, y, z];
            let p = [x * radius, y * radius, z * radius];
            let uv = [j as f32 / lon as f32, i as f32 / lat as f32];
            let t = [-phi.sin() * theta.sin(), 0.0, phi.sin() * theta.cos()];
            vertices.push(PackedVertex::new(p, n, uv, t, 1.0));
        }
    }
    for i in 0..lat {
        for j in 0..lon {
            let a = i * (lon + 1) + j;
            let b = a + lon + 1;
            indices.extend_from_slice(&[a, b, a+1, b, b+1, a+1]);
        }
    }
    Mesh::new(vertices, indices)
}

pub fn create_plane_mesh(width: f32, height: f32) -> Mesh {
    let hw = width * 0.5;
    let hh = height * 0.5;
    let n = [0.0f32, 1.0, 0.0];
    let t = [1.0f32, 0.0, 0.0];
    let vertices = vec![
        PackedVertex::new([-hw, 0.0, -hh], n, [0.0, 0.0], t, 1.0),
        PackedVertex::new([ hw, 0.0, -hh], n, [1.0, 0.0], t, 1.0),
        PackedVertex::new([ hw, 0.0,  hh], n, [1.0, 1.0], t, 1.0),
        PackedVertex::new([-hw, 0.0,  hh], n, [0.0, 1.0], t, 1.0),
    ];
    Mesh::new(vertices, vec![0, 2, 1, 0, 3, 2])
}

pub fn create_billboard_quad(size: f32) -> BillboardMesh {
    let h = size * 0.5;
    BillboardMesh {
        vertices: vec![
            BillboardVertex { position: [-h, -h, 0.0], tex_coords: [0.0, 1.0] },
            BillboardVertex { position: [ h, -h, 0.0], tex_coords: [1.0, 1.0] },
            BillboardVertex { position: [ h,  h, 0.0], tex_coords: [1.0, 0.0] },
            BillboardVertex { position: [-h,  h, 0.0], tex_coords: [0.0, 0.0] },
        ],
        indices: vec![0, 1, 2, 0, 2, 3],
    }
}

fn add3(a: [f32;3], b: [f32;3]) -> [f32;3] { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
fn scale3(v: [f32;3], s: f32) -> [f32;3] { [v[0]*s, v[1]*s, v[2]*s] }
fn cross3(a: [f32;3], b: [f32;3]) -> [f32;3] {
    [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
}
