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
///
/// The renderer fills a `DrawCall` for every mesh submitted.  To support
/// hardware instancing we optionally carry an instance buffer and count.  The
/// geometry pass will bind the buffer as a second vertex stream and issue
/// `draw_indexed_instanced` when `instance_count > 1`.

/// Byte stride of a single instance record: one `mat4x4<f32>` (4×4×4 bytes).
pub const INSTANCE_STRIDE: u64 = std::mem::size_of::<[f32; 16]>() as u64;

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
    /// Optional per-instance transform buffer (mat4x4<f32> per instance)
    pub instance_buffer: Option<Arc<wgpu::Buffer>>,
    /// Number of instances encoded in `instance_buffer` (defaults to 1).
    pub instance_count: u32,
    /// Byte offset into `instance_buffer` where this draw's instances begin.
    /// For the unified shared instance buffer this is `batch_start_instance * INSTANCE_STRIDE`.
    pub instance_buffer_offset: u64,
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
            instance_buffer: None,
            instance_count: 1,
            instance_buffer_offset: 0,
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
