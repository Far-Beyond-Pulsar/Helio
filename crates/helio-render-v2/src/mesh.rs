//! GPU mesh types and draw-call submission for V2 renderer

use std::sync::Arc;

/// Vertex format matching the geometry.wgsl shader exactly (32 bytes)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PackedVertex {
    pub position: [f32; 3],
    pub bitangent_sign: f32,
    pub tex_coords: [f32; 2],
    pub normal: u32,   // Packed SNORM8x4
    pub tangent: u32,  // Packed SNORM8x4
}

impl PackedVertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], tex_coords: [f32; 2]) -> Self {
        Self {
            position,
            bitangent_sign: 1.0,
            tex_coords,
            normal: pack_snorm8x4(normal[0], normal[1], normal[2], 0.0),
            tangent: pack_snorm8x4(1.0, 0.0, 0.0, 1.0),
        }
    }
}

fn pack_snorm8x4(x: f32, y: f32, z: f32, w: f32) -> u32 {
    let xi = (x.clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let yi = (y.clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let zi = (z.clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    let wi = (w.clamp(-1.0, 1.0) * 127.0) as i8 as u8;
    (xi as u32) | ((yi as u32) << 8) | ((zi as u32) << 16) | ((wi as u32) << 24)
}

/// GPU-resident mesh (owns wgpu vertex + index buffers)
#[derive(Clone)]
pub struct GpuMesh {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
}

impl GpuMesh {
    pub fn new(device: &wgpu::Device, vertices: &[PackedVertex], indices: &[u32]) -> Self {
        use wgpu::util::DeviceExt;
        let vertex_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        let index_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        Self { vertex_buffer, index_buffer, index_count: indices.len() as u32 }
    }

    /// Build a unit cube mesh centered at `center` with half-extent `half_size`
    pub fn cube(device: &wgpu::Device, center: [f32; 3], half_size: f32) -> Self {
        let [cx, cy, cz] = center;
        let h = half_size;

        // 6 faces: (normal, [4 corners in CCW winding viewed from outside])
        let faces: &[([f32; 3], [[f32; 3]; 4])] = &[
            ([0.0, 0.0, 1.0], [[cx-h,cy-h,cz+h],[cx+h,cy-h,cz+h],[cx+h,cy+h,cz+h],[cx-h,cy+h,cz+h]]),
            ([0.0, 0.0,-1.0], [[cx+h,cy-h,cz-h],[cx-h,cy-h,cz-h],[cx-h,cy+h,cz-h],[cx+h,cy+h,cz-h]]),
            ([1.0, 0.0, 0.0], [[cx+h,cy-h,cz+h],[cx+h,cy-h,cz-h],[cx+h,cy+h,cz-h],[cx+h,cy+h,cz+h]]),
            ([-1.0,0.0, 0.0], [[cx-h,cy-h,cz-h],[cx-h,cy-h,cz+h],[cx-h,cy+h,cz+h],[cx-h,cy+h,cz-h]]),
            ([0.0, 1.0, 0.0], [[cx-h,cy+h,cz+h],[cx+h,cy+h,cz+h],[cx+h,cy+h,cz-h],[cx-h,cy+h,cz-h]]),
            ([0.0,-1.0, 0.0], [[cx-h,cy-h,cz-h],[cx+h,cy-h,cz-h],[cx+h,cy-h,cz+h],[cx-h,cy-h,cz+h]]),
        ];

        let uvs: [[f32; 2]; 4] = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];
        let mut vertices = Vec::with_capacity(24);
        let mut indices  = Vec::with_capacity(36);

        for (face_idx, (normal, corners)) in faces.iter().enumerate() {
            let base = (face_idx * 4) as u32;
            for (i, &pos) in corners.iter().enumerate() {
                vertices.push(PackedVertex::new(pos, *normal, uvs[i]));
            }
            indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
        }

        Self::new(device, &vertices, &indices)
    }

    /// Build a flat XZ plane centered at `center` with half-extent `half_extent`
    pub fn plane(device: &wgpu::Device, center: [f32; 3], half_extent: f32) -> Self {
        let [cx, cy, cz] = center;
        let h = half_extent;
        let n = [0.0f32, 1.0, 0.0];
        let vertices = [
            PackedVertex::new([cx-h,cy,cz+h], n, [0.0,0.0]),
            PackedVertex::new([cx+h,cy,cz+h], n, [1.0,0.0]),
            PackedVertex::new([cx+h,cy,cz-h], n, [1.0,1.0]),
            PackedVertex::new([cx-h,cy,cz-h], n, [0.0,1.0]),
        ];
        let indices = [0u32, 1, 2, 0, 2, 3];
        Self::new(device, &vertices, &indices)
    }
}

/// A single queued geometry draw call
#[derive(Clone)]
pub struct DrawCall {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
    pub material_bind_group: Arc<wgpu::BindGroup>,
}

impl DrawCall {
    pub fn new(mesh: &GpuMesh, material: Arc<wgpu::BindGroup>) -> Self {
        Self {
            vertex_buffer: mesh.vertex_buffer.clone(),
            index_buffer: mesh.index_buffer.clone(),
            index_count: mesh.index_count,
            material_bind_group: material,
        }
    }
}
