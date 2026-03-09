use std::sync::Arc;
use bytemuck::{Pod, Zeroable};

/// Tightly-packed vertex: 32 bytes.
///
/// `normal` and `tangent` are SNORM8×4 (as wgpu::VertexFormat::Snorm8x4),
/// stored as packed u32 for Pod safety on the CPU side.
/// The GPU sees them as `vec4<f32>` via `@builtin(VertexFormat::Snorm8x4)`.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub struct PackedVertex {
    pub position:       [f32; 3],  // offset  0
    pub bitangent_sign: f32,       // offset 12 — +1.0 or -1.0
    pub tex_coords:     [f32; 2],  // offset 16
    pub normal:         u32,       // offset 24 — packed SNORM8x4
    pub tangent:        u32,       // offset 28 — packed SNORM8x4
}                                  // total: 32 bytes

pub const INSTANCE_STRIDE: u64 = 64; // 4×Vec4 = mat4x4 (column-major)

impl PackedVertex {
    /// Create vertex. `normal` and `tangent` are normalised vectors.
    pub fn new(
        position:       [f32; 3],
        tex_coords:     [f32; 2],
        normal:         [f32; 3],
        tangent:        [f32; 3],
        bitangent_sign: f32,
    ) -> Self {
        Self {
            position,
            bitangent_sign,
            tex_coords,
            normal:  pack_snorm8x4(normal[0], normal[1], normal[2], 0.0),
            tangent: pack_snorm8x4(tangent[0], tangent[1], tangent[2], 0.0),
        }
    }

    pub fn vertex_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        use wgpu::VertexFormat::*;
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PackedVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute { format: Float32x3, offset:  0, shader_location: 0 },
                wgpu::VertexAttribute { format: Float32,   offset: 12, shader_location: 1 },
                wgpu::VertexAttribute { format: Float32x2, offset: 16, shader_location: 2 },
                wgpu::VertexAttribute { format: Snorm8x4,  offset: 24, shader_location: 3 },
                wgpu::VertexAttribute { format: Snorm8x4,  offset: 28, shader_location: 4 },
            ],
        }
    }

    /// Instance buffer layout: 4 × Float32x4 = mat4x4 (columns 0..3 at locations 5..8).
    pub fn instance_buffer_layout() -> wgpu::VertexBufferLayout<'static> {
        use wgpu::VertexFormat::Float32x4;
        wgpu::VertexBufferLayout {
            array_stride: INSTANCE_STRIDE,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute { format: Float32x4, offset:  0, shader_location: 5 },
                wgpu::VertexAttribute { format: Float32x4, offset: 16, shader_location: 6 },
                wgpu::VertexAttribute { format: Float32x4, offset: 32, shader_location: 7 },
                wgpu::VertexAttribute { format: Float32x4, offset: 48, shader_location: 8 },
            ],
        }
    }
}

fn pack_snorm8x4(x: f32, y: f32, z: f32, w: f32) -> u32 {
    fn cv(v: f32) -> u8 { (v.clamp(-1.0, 1.0) * 127.0).round() as i8 as u8 }
    u32::from_le_bytes([cv(x), cv(y), cv(z), cv(w)])
}

/// GPU-resident mesh. Allocated once, shared via Arc.
pub struct GpuMesh {
    pub vertex_buffer:  Arc<wgpu::Buffer>,
    pub index_buffer:   Arc<wgpu::Buffer>,
    pub vertex_count:   u32,
    pub index_count:    u32,
    /// AABB sphere — pre-computed at upload time.
    pub bounds_center:  [f32; 3],
    pub bounds_radius:  f32,
}

impl GpuMesh {
    /// Upload vertices + indices, compute bounds sphere.
    pub fn upload(
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        vertices: &[PackedVertex],
        indices:  &[u32],
        label:    Option<&str>,
    ) -> Arc<Self> {
        use wgpu::util::DeviceExt;

        let vb = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::cast_slice(vertices),
            usage:    wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::BLAS_INPUT,
        });
        let ib = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    None,
            contents: bytemuck::cast_slice(indices),
            usage:    wgpu::BufferUsages::INDEX | wgpu::BufferUsages::BLAS_INPUT,
        });

        let (center, radius) = bounds_sphere(vertices);

        Arc::new(GpuMesh {
            vertex_buffer: Arc::new(vb),
            index_buffer:  Arc::new(ib),
            vertex_count:  vertices.len() as u32,
            index_count:   indices.len()  as u32,
            bounds_center: center,
            bounds_radius: radius,
        })
    }
}

fn bounds_sphere(verts: &[PackedVertex]) -> ([f32; 3], f32) {
    if verts.is_empty() { return ([0.0; 3], 0.0); }
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for v in verts {
        for i in 0..3 {
            min[i] = min[i].min(v.position[i]);
            max[i] = max[i].max(v.position[i]);
        }
    }
    let cx = (min[0] + max[0]) * 0.5;
    let cy = (min[1] + max[1]) * 0.5;
    let cz = (min[2] + max[2]) * 0.5;
    let mut r2 = 0.0f32;
    for v in verts {
        let dx = v.position[0] - cx;
        let dy = v.position[1] - cy;
        let dz = v.position[2] - cz;
        r2 = r2.max(dx*dx + dy*dy + dz*dz);
    }
    ([cx, cy, cz], r2.sqrt())
}

/// A single draw call produced by the HISM batch builder.
///
/// In v3 `instance_buffer` is NEVER an Option — the batch builder always
/// allocates at least one instance even if it turns out to be culled.
/// This eliminates the hot branch in every pass.
#[derive(Clone)]
pub struct DrawCall {
    pub hism_handle:          crate::hism::HismHandle,
    pub vertex_buffer:        Arc<wgpu::Buffer>,
    pub index_buffer:         Arc<wgpu::Buffer>,
    pub vertex_count:         u32,
    pub index_count:          u32,
    pub material_bind_group:  Arc<wgpu::BindGroup>,
    pub transparent_blend:    bool,
    pub bounds_center:        [f32; 3],
    pub bounds_radius:        f32,
    pub material_id:          u32,
    /// GPU-side packed mat4×4 transforms.  Stride = INSTANCE_STRIDE (64 bytes).
    pub instance_buffer:      Arc<wgpu::Buffer>,  // v3: NEVER Option
    pub instance_count:       u32,
    pub instance_buffer_offset: u64,
}

impl DrawCall {
    /// Squared world-space distance from a camera position (for sort keys).
    pub fn depth_sq(&self, camera_pos: [f32; 3]) -> f32 {
        let dx = self.bounds_center[0] - camera_pos[0];
        let dy = self.bounds_center[1] - camera_pos[1];
        let dz = self.bounds_center[2] - camera_pos[2];
        dx*dx + dy*dy + dz*dz
    }
}
