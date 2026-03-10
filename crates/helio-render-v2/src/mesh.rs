//! GPU mesh types and draw-call submission for V2 renderer

use std::sync::Arc;
use crate::gpu_transfer;

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
    /// Local-space bounding sphere center (centroid of all vertices).
    pub bounds_center: [f32; 3],
    /// Bounding sphere radius around `bounds_center`.
    pub bounds_radius: f32,
    /// Local-space AABB minimum corner.
    pub aabb_min: [f32; 3],
    /// Local-space AABB maximum corner.
    pub aabb_max: [f32; 3],
    /// Base vertex offset inside the unified geometry pool (0 for standalone meshes).
    pub pool_base_vertex: u32,
    /// First index offset inside the unified geometry pool (0 for standalone meshes).
    pub pool_first_index: u32,
    /// True when this mesh was allocated from `GpuBufferPool` and participates in
    /// GPU-driven indirect rendering.
    pub pool_allocated: bool,
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
        let vb_bytes = (vertices.len() * std::mem::size_of::<PackedVertex>()) as u64;
        let ib_bytes = (indices.len() * std::mem::size_of::<u32>()) as u64;
        gpu_transfer::track_alloc(vb_bytes + ib_bytes);
        // Bounding sphere: centroid + max-distance radius.
        // Also compute the exact AABB for tighter Hi-Z occlusion culling.
        let (bounds_center, bounds_radius, aabb_min, aabb_max) = if vertices.is_empty() {
            ([0.0f32; 3], 0.0f32, [0.0f32; 3], [0.0f32; 3])
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

            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            for v in vertices {
                for i in 0..3 {
                    if v.position[i] < min[i] { min[i] = v.position[i]; }
                    if v.position[i] > max[i] { max[i] = v.position[i]; }
                }
            }
            ([cx, cy, cz], r, min, max)
        };
        Self {
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            vertex_count: vertices.len() as u32,
            bounds_center,
            bounds_radius,
            aabb_min,
            aabb_max,
            pool_base_vertex: 0,
            pool_first_index: 0,
            pool_allocated: false,
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

    /// Unit cube in local space, centered at the origin with half-size 0.5.
    ///
    /// Unlike [`Self::cube`] which bakes world-space positions into vertices,
    /// this mesh is designed for GPU instancing: place and scale it with
    /// `SceneObject::with_transform()` and the vertex shader applies the model matrix.
    pub fn unit_cube(device: &wgpu::Device) -> Self {
        Self::cube(device, [0.0, 0.0, 0.0], 0.5)
    }

    /// Local-space box with independent half-extents, centered at the origin.
    ///
    /// Designed for GPU instancing; use [`Self::rect3d`] when you want world-space
    /// positions baked into vertices.
    pub fn unit_rect3d(device: &wgpu::Device, half_extents: [f32; 3]) -> Self {
        Self::rect3d(device, [0.0, 0.0, 0.0], half_extents)
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
        let t = [1.0f32, 0.0, 0.0];
        let vertices = [
            PackedVertex::new_with_tangent([cx-h,cy,cz+h], n, [0.0,0.0], t),
            PackedVertex::new_with_tangent([cx+h,cy,cz+h], n, [1.0,0.0], t),
            PackedVertex::new_with_tangent([cx+h,cy,cz-h], n, [1.0,1.0], t),
            PackedVertex::new_with_tangent([cx-h,cy,cz-h], n, [0.0,1.0], t),
        ];
        let indices = [0u32, 1, 2, 0, 2, 3];
        Self::new(device, &vertices, &indices)
    }

    /// Upload vertices + indices into the unified `GpuBufferPool`.
    /// Returns a pool-allocated `GpuMesh` that participates in `multi_draw_indexed_indirect`.
    pub fn upload_to_pool(
        queue: &wgpu::Queue,
        pool: &mut crate::buffer_pool::GpuBufferPool,
        vertices: &[PackedVertex],
        indices: &[u32],
    ) -> Option<Self> {
        let alloc = pool.alloc(queue, vertices, indices)?;

        let (bounds_center, bounds_radius, aabb_min, aabb_max) = if vertices.is_empty() {
            ([0.0f32; 3], 0.0f32, [0.0f32; 3], [0.0f32; 3])
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
            let mut min = [f32::MAX; 3];
            let mut max = [f32::MIN; 3];
            for v in vertices {
                for i in 0..3 {
                    if v.position[i] < min[i] { min[i] = v.position[i]; }
                    if v.position[i] > max[i] { max[i] = v.position[i]; }
                }
            }
            ([cx, cy, cz], r, min, max)
        };

        Some(Self {
            vertex_buffer:    Arc::clone(&pool.vertex_buffer),
            index_buffer:     Arc::clone(&pool.index_buffer),
            index_count:      alloc.index_count,
            vertex_count:     alloc.vertex_count,
            bounds_center,
            bounds_radius,
            aabb_min,
            aabb_max,
            pool_base_vertex: alloc.base_vertex,
            pool_first_index: alloc.first_index,
            pool_allocated:   true,
        })
    }
}

/// A single queued geometry draw call.
///
/// GPU-driven path: `slot` is the instance-data index used as `first_instance`.
/// `pool_base_vertex` / `pool_first_index` are pool offsets for the draw command.
#[derive(Clone)]
pub struct DrawCall {
    /// Points to the pool vertex buffer (or standalone VB for legacy meshes).
    pub vertex_buffer: Arc<wgpu::Buffer>,
    /// Points to the pool index buffer (or standalone IB for legacy meshes).
    pub index_buffer: Arc<wgpu::Buffer>,
    pub index_count: u32,
    pub vertex_count: u32,
    pub material_bind_group: Arc<wgpu::BindGroup>,
    pub transparent_blend: bool,
    /// World-space bounding sphere centre.
    pub bounds_center: [f32; 3],
    /// Bounding sphere radius.
    pub bounds_radius: f32,
    /// GPU scene slot index — `first_instance` in draw commands so vertex shader
    /// reads `instance_data[slot]` for the model transform.
    pub slot: u32,
    /// Base vertex offset in the pool VB (0 for standalone meshes).
    pub pool_base_vertex: i32,
    /// First index offset in the pool IB (0 for standalone meshes).
    pub pool_first_index: u32,
    /// True when this draw call uses the unified geometry pool VB/IB.
    pub pool_allocated: bool,
}

impl DrawCall {
    pub fn new(mesh: &GpuMesh, slot: u32, material: Arc<wgpu::BindGroup>, transparent_blend: bool) -> Self {
        Self {
            vertex_buffer:    mesh.vertex_buffer.clone(),
            index_buffer:     mesh.index_buffer.clone(),
            index_count:      mesh.index_count,
            vertex_count:     mesh.vertex_count,
            material_bind_group: material,
            transparent_blend,
            bounds_center:    mesh.bounds_center,
            bounds_radius:    mesh.bounds_radius,
            slot,
            pool_base_vertex: mesh.pool_base_vertex as i32,
            pool_first_index: mesh.pool_first_index,
            pool_allocated:   mesh.pool_allocated,
        }
    }
}

/// GPU representation of a draw call for the indirect dispatch compute shader.
///
/// Must exactly match `struct GpuDrawCall` in `indirect_dispatch.wgsl`.
/// 32 bytes, 16-byte struct alignment.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuDrawCall {
    pub slot:          u32,        // offset  0 — → first_instance in indirect cmd
    pub first_index:   u32,        // offset  4 — pool IB offset
    pub base_vertex:   i32,        // offset  8 — pool VB offset
    pub index_count:   u32,        // offset 12
    pub bounds_center: [f32; 3],   // offset 16 — world-space bounding sphere (vec3f @ 16)
    pub bounds_radius: f32,        // offset 28
    // total: 32 bytes
}

// Keep for any code that hasn't been updated yet.
#[allow(dead_code)]
pub const INSTANCE_STRIDE: u64 = std::mem::size_of::<[f32; 16]>() as u64;
