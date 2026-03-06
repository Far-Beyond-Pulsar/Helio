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
    /// Create a vertex. The tangent is auto-computed as the UV u-direction
    /// (direction from `p0` to `p1` in a quad, or a fallback for standalone vertices).
    pub fn new(position: [f32; 3], normal: [f32; 3], tex_coords: [f32; 2]) -> Self {
        // Fall back to a tangent perpendicular to the normal
        let tangent = normal_to_tangent(normal);
        Self::new_with_tangent(position, normal, tex_coords, tangent)
    }

    /// Create a vertex with an explicit tangent (UV u-direction of the face).
    pub fn new_with_tangent(
        position: [f32; 3],
        normal: [f32; 3],
        tex_coords: [f32; 2],
        tangent: [f32; 3],
    ) -> Self {
        Self {
            position,
            bitangent_sign: 1.0,
            tex_coords,
            normal:  pack_snorm8x4(normal[0],  normal[1],  normal[2],  0.0),
            tangent: pack_snorm8x4(tangent[0], tangent[1], tangent[2], 1.0),
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

/// Compute the UV u-direction tangent from corner 0→1 of a quad.
/// (UVs go [0,0],[1,0],[1,1],[0,1] so corner1-corner0 is the +u direction.)
fn face_tangent(p0: [f32; 3], p1: [f32; 3]) -> [f32; 3] {
    let d = [p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]];
    let len = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]).sqrt().max(1e-8);
    [d[0]/len, d[1]/len, d[2]/len]
}

/// Compute any tangent perpendicular to `normal` (used when no UV reference is available).
fn normal_to_tangent(n: [f32; 3]) -> [f32; 3] {
    // Choose the axis least aligned with n to avoid degeneracy
    let up = if n[1].abs() < 0.9 { [0.0f32, 1.0, 0.0] } else { [1.0f32, 0.0, 0.0] };
    // cross(up, n) gives a vector perpendicular to n in the horizontal plane
    let t = [
        up[1]*n[2] - up[2]*n[1],
        up[2]*n[0] - up[0]*n[2],
        up[0]*n[1] - up[1]*n[0],
    ];
    let len = (t[0]*t[0] + t[1]*t[1] + t[2]*t[2]).sqrt().max(1e-8);
    [t[0]/len, t[1]/len, t[2]/len]
}

/// GPU-resident mesh (owns wgpu vertex + index buffers)
#[derive(Clone)]
pub struct GpuMesh {
    pub vertex_buffer: Arc<wgpu::Buffer>,
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
    pub vertex_count: u32,
    /// World-space centroid of all vertices (used for shadow culling).
    pub bounds_center: [f32; 3],
    /// Radius of the bounding sphere around `bounds_center`.
    pub bounds_radius: f32,
}

impl GpuMesh {
    pub fn new(device: &wgpu::Device, vertices: &[PackedVertex], indices: &[u32]) -> Self {
        use wgpu::util::DeviceExt;
        // Only add BLAS_INPUT when the device actually has ray-tracing enabled;
        // requesting it without the feature causes a validation error.
        let has_rt = device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        let blas_flag = if has_rt { wgpu::BufferUsages::BLAS_INPUT } else { wgpu::BufferUsages::empty() };
        let vertex_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX | blas_flag,
        }));
        let index_buffer = Arc::new(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Mesh Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX | blas_flag,
        }));
        // Bounding sphere: centroid + max-distance radius.
        // Centroid approximation is conservative and fast; the tiny extra radius
        // vs a min-bounding-sphere algorithm only means slightly fewer culled draws.
        let (bounds_center, bounds_radius) = if vertices.is_empty() {
            ([0.0f32; 3], 0.0f32)
        } else {
            let n = vertices.len() as f32;
            let cx = vertices.iter().map(|v| v.position[0]).sum::<f32>() / n;
            let cy = vertices.iter().map(|v| v.position[1]).sum::<f32>() / n;
            let cz = vertices.iter().map(|v| v.position[2]).sum::<f32>() / n;
            let r  = vertices.iter().map(|v| {
                let dx = v.position[0] - cx;
                let dy = v.position[1] - cy;
                let dz = v.position[2] - cz;
                (dx * dx + dy * dy + dz * dz).sqrt()
            }).fold(0.0f32, f32::max);
            ([cx, cy, cz], r)
        };
        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            vertex_count: vertices.len() as u32,
            bounds_center,
            bounds_radius,
        }
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
            // Tangent = UV u-direction = direction from corner[0] to corner[1]
            let tangent = face_tangent(corners[0], corners[1]);
            for (i, &pos) in corners.iter().enumerate() {
                vertices.push(PackedVertex::new_with_tangent(pos, *normal, uvs[i], tangent));
            }
            indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
        }

        Self::new(device, &vertices, &indices)
    }

    /// Build an axis-aligned box with independent half-extents on each axis.
    /// Useful for thin slabs, beams, or any non-uniform rectangular volume.
    pub fn rect3d(device: &wgpu::Device, center: [f32; 3], half_extents: [f32; 3]) -> Self {
        let [cx, cy, cz] = center;
        let [hx, hy, hz] = half_extents;

        let faces: &[([f32; 3], [[f32; 3]; 4])] = &[
            ([0.0, 0.0, 1.0], [[cx-hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz+hz]]),
            ([0.0, 0.0,-1.0], [[cx+hx,cy-hy,cz-hz],[cx-hx,cy-hy,cz-hz],[cx-hx,cy+hy,cz-hz],[cx+hx,cy+hy,cz-hz]]),
            ([1.0, 0.0, 0.0], [[cx+hx,cy-hy,cz+hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy+hy,cz-hz],[cx+hx,cy+hy,cz+hz]]),
            ([-1.0,0.0, 0.0], [[cx-hx,cy-hy,cz-hz],[cx-hx,cy-hy,cz+hz],[cx-hx,cy+hy,cz+hz],[cx-hx,cy+hy,cz-hz]]),
            ([0.0, 1.0, 0.0], [[cx-hx,cy+hy,cz+hz],[cx+hx,cy+hy,cz+hz],[cx+hx,cy+hy,cz-hz],[cx-hx,cy+hy,cz-hz]]),
            ([0.0,-1.0, 0.0], [[cx-hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz-hz],[cx+hx,cy-hy,cz+hz],[cx-hx,cy-hy,cz+hz]]),
        ];

        let uvs: [[f32; 2]; 4] = [[0.0,0.0],[1.0,0.0],[1.0,1.0],[0.0,1.0]];
        let mut vertices = Vec::with_capacity(24);
        let mut indices  = Vec::with_capacity(36);

        for (face_idx, (normal, corners)) in faces.iter().enumerate() {
            let base = (face_idx * 4) as u32;
            let tangent = face_tangent(corners[0], corners[1]);
            for (i, &pos) in corners.iter().enumerate() {
                vertices.push(PackedVertex::new_with_tangent(pos, *normal, uvs[i], tangent));
            }
            indices.extend_from_slice(&[base, base+1, base+2, base, base+2, base+3]);
        }

        Self::new(device, &vertices, &indices)
    }


    pub fn plane(device: &wgpu::Device, center: [f32; 3], half_extent: f32) -> Self {
        let [cx, cy, cz] = center;
        let h = half_extent;
        let n = [0.0f32, 1.0, 0.0];
        let t = [1.0f32, 0.0, 0.0]; // UV u-direction on XZ plane = +X
        let vertices = [
            PackedVertex::new_with_tangent([cx-h,cy,cz+h], n, [0.0,0.0], t),
            PackedVertex::new_with_tangent([cx+h,cy,cz+h], n, [1.0,0.0], t),
            PackedVertex::new_with_tangent([cx+h,cy,cz-h], n, [1.0,1.0], t),
            PackedVertex::new_with_tangent([cx-h,cy,cz-h], n, [0.0,1.0], t),
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
    pub vertex_count: u32,
    pub material_bind_group: Arc<wgpu::BindGroup>,
    pub transparent_blend: bool,
    /// World-space bounding sphere centre, copied from the source `GpuMesh`.
    pub bounds_center: [f32; 3],
    /// Bounding sphere radius, copied from the source `GpuMesh`.
    pub bounds_radius: f32,
    /// Material ID for GPU-driven indirect rendering (0 if not assigned)
    pub material_id: u32,
}

impl DrawCall {
    pub fn new(mesh: &GpuMesh, material: Arc<wgpu::BindGroup>, transparent_blend: bool) -> Self {
        Self {
            vertex_buffer: mesh.vertex_buffer.clone(),
            index_buffer: mesh.index_buffer.clone(),
            index_count: mesh.index_count,
            vertex_count: mesh.vertex_count,
            material_bind_group: material,
            transparent_blend,
            bounds_center: mesh.bounds_center,
            bounds_radius: mesh.bounds_radius,
            material_id: 0,  // Will be assigned by renderer if GPU-driven is enabled
        }
    }
}

/// GPU representation of DrawCall for indirect rendering
/// Matches the WGSL struct in indirect_dispatch.wgsl
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrawCall {
    // Mesh buffer offsets
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub vertex_count: u32,
    
    // Material ID
    pub material_id: u32,
    
    // Flags
    pub transparent_blend: u32,
    
    // Padding to align to vec3f
    pub _pad0: u32,
    pub _pad1: u32,
    
    // Bounding volume (vec3f + f32 in WGSL)
    pub bounds_center: [f32; 3],
    pub bounds_radius: f32,
}
