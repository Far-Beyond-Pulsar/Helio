use crate::gpu;
use crate::bounds::Aabb;
use crate::vertex::{Vertex, PackedVertex};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshHandle(pub u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveTopology {
    TriangleList,
    TriangleStrip,
    LineList,
    LineStrip,
    PointList,
}

pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub bounds: Aabb,
    pub topology: PrimitiveTopology,
    pub vertex_buffer: Option<gpu::Buffer>,
    pub index_buffer: Option<gpu::Buffer>,
}

impl Mesh {
    pub fn new(vertices: Vec<Vertex>, indices: Vec<u32>, topology: PrimitiveTopology) -> Self {
        let positions: Vec<_> = vertices.iter().map(|v| glam::Vec3::from_array(v.position)).collect();
        let bounds = if !positions.is_empty() {
            Aabb::from_points(&positions)
        } else {
            Aabb::default()
        };

        Self {
            vertices,
            indices,
            bounds,
            topology,
            vertex_buffer: None,
            index_buffer: None,
        }
    }

    pub fn upload_to_gpu(&mut self, context: &gpu::Context) {
        // GPU upload will be handled by the renderer
        // This is a placeholder for now
    }

    pub fn cleanup_gpu_resources(&mut self, context: &gpu::Context) {
        if let Some(buffer) = self.vertex_buffer.take() {
            context.destroy_buffer(buffer);
        }
        if let Some(buffer) = self.index_buffer.take() {
            context.destroy_buffer(buffer);
        }
    }
}

pub struct MeshBuilder {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    topology: PrimitiveTopology,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            topology: PrimitiveTopology::TriangleList,
        }
    }

    pub fn with_topology(mut self, topology: PrimitiveTopology) -> Self {
        self.topology = topology;
        self
    }

    pub fn add_vertex(&mut self, vertex: Vertex) -> u32 {
        let index = self.vertices.len() as u32;
        self.vertices.push(vertex);
        index
    }

    pub fn add_triangle(&mut self, i0: u32, i1: u32, i2: u32) {
        self.indices.push(i0);
        self.indices.push(i1);
        self.indices.push(i2);
    }

    pub fn build(self) -> Mesh {
        Mesh::new(self.vertices, self.indices, self.topology)
    }
}

impl Default for MeshBuilder {
    fn default() -> Self {
        Self::new()
    }
}

pub fn create_cube_mesh(size: f32) -> Mesh {
    let half = size * 0.5;
    let mut builder = MeshBuilder::new();

    let positions = [
        glam::Vec3::new(-half, -half, -half),
        glam::Vec3::new(half, -half, -half),
        glam::Vec3::new(half, half, -half),
        glam::Vec3::new(-half, half, -half),
        glam::Vec3::new(-half, -half, half),
        glam::Vec3::new(half, -half, half),
        glam::Vec3::new(half, half, half),
        glam::Vec3::new(-half, half, half),
    ];

    let normals = [
        glam::Vec3::NEG_Z,
        glam::Vec3::Z,
        glam::Vec3::NEG_X,
        glam::Vec3::X,
        glam::Vec3::NEG_Y,
        glam::Vec3::Y,
    ];

    let tangent = glam::Vec4::new(1.0, 0.0, 0.0, 1.0);

    let faces = [
        ([0, 1, 2, 3], normals[0]),
        ([5, 4, 7, 6], normals[1]),
        ([4, 0, 3, 7], normals[2]),
        ([1, 5, 6, 2], normals[3]),
        ([4, 5, 1, 0], normals[4]),
        ([3, 2, 6, 7], normals[5]),
    ];

    for (face_indices, normal) in &faces {
        let base_idx = builder.vertices.len() as u32;
        
        let uvs = [
            glam::Vec2::new(0.0, 0.0),
            glam::Vec2::new(1.0, 0.0),
            glam::Vec2::new(1.0, 1.0),
            glam::Vec2::new(0.0, 1.0),
        ];

        for (i, &pos_idx) in face_indices.iter().enumerate() {
            builder.add_vertex(Vertex::new(
                positions[pos_idx],
                *normal,
                tangent,
                uvs[i],
            ));
        }

        builder.add_triangle(base_idx, base_idx + 1, base_idx + 2);
        builder.add_triangle(base_idx, base_idx + 2, base_idx + 3);
    }

    builder.build()
}

pub fn create_sphere_mesh(radius: f32, segments: u32, rings: u32) -> Mesh {
    let mut builder = MeshBuilder::new();

    for ring in 0..=rings {
        let theta = ring as f32 * std::f32::consts::PI / rings as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for segment in 0..=segments {
            let phi = segment as f32 * 2.0 * std::f32::consts::PI / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let position = glam::Vec3::new(
                sin_theta * cos_phi * radius,
                cos_theta * radius,
                sin_theta * sin_phi * radius,
            );
            let normal = position.normalize();
            let tangent = glam::Vec4::new(-sin_phi, 0.0, cos_phi, 1.0);
            let tex_coords = glam::Vec2::new(
                segment as f32 / segments as f32,
                ring as f32 / rings as f32,
            );

            builder.add_vertex(Vertex::new(position, normal, tangent, tex_coords));
        }
    }

    for ring in 0..rings {
        for segment in 0..segments {
            let i0 = ring * (segments + 1) + segment;
            let i1 = i0 + 1;
            let i2 = i0 + (segments + 1);
            let i3 = i2 + 1;

            builder.add_triangle(i0, i1, i2);
            builder.add_triangle(i2, i1, i3);
        }
    }

    builder.build()
}

pub fn create_plane_mesh(width: f32, height: f32) -> Mesh {
    let mut builder = MeshBuilder::new();

    let half_w = width * 0.5;
    let half_h = height * 0.5;

    let vertices = [
        Vertex::new(
            glam::Vec3::new(-half_w, 0.0, -half_h),
            glam::Vec3::Y,
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec2::new(0.0, 0.0),
        ),
        Vertex::new(
            glam::Vec3::new(half_w, 0.0, -half_h),
            glam::Vec3::Y,
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec2::new(1.0, 0.0),
        ),
        Vertex::new(
            glam::Vec3::new(half_w, 0.0, half_h),
            glam::Vec3::Y,
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec2::new(1.0, 1.0),
        ),
        Vertex::new(
            glam::Vec3::new(-half_w, 0.0, half_h),
            glam::Vec3::Y,
            glam::Vec4::new(1.0, 0.0, 0.0, 1.0),
            glam::Vec2::new(0.0, 1.0),
        ),
    ];

    for vertex in vertices {
        builder.add_vertex(vertex);
    }

    builder.add_triangle(0, 1, 2);
    builder.add_triangle(0, 2, 3);

    builder.build()
}
