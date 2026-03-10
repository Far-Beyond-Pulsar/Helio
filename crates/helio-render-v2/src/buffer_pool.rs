//! Unified geometry buffer pool.
//!
//! All mesh vertices and indices live in two large shared GPU buffers.
//! This eliminates per-mesh VB/IB bind overhead and enables
//! `multi_draw_indexed_indirect` — one call for all opaque geometry.

use std::sync::Arc;
use crate::mesh::PackedVertex;
use crate::gpu_transfer;

/// Allocation returned by [`GpuBufferPool::alloc`].
pub struct MeshAlloc {
    pub base_vertex: u32,
    pub first_index: u32,
    pub vertex_count: u32,
    pub index_count: u32,
}

/// Unified GPU geometry pool.
///
/// Backed by a single large vertex buffer and index buffer shared by **all**
/// meshes.  New meshes are appended with a bump allocator.  This mirrors the
/// Unreal Engine "Virtual Geometry Buffer" / Nanite cluster pool design where
/// a single bound VB+IB covers the entire scene.
///
/// # Capacity
/// Default: 4 M vertices (128 MB @ 32 B each) + 16 M indices (64 MB @ 4 B each).
/// This is enough for hundreds of unique high-poly meshes.
pub struct GpuBufferPool {
    /// Shared vertex buffer — bound once for all pool-resident draws.
    pub vertex_buffer: Arc<wgpu::Buffer>,
    /// Shared index buffer — bound once for all pool-resident draws.
    pub index_buffer: Arc<wgpu::Buffer>,
    vertex_cursor: u32,
    index_cursor: u32,
    pub vertex_capacity: u32,
    pub index_capacity: u32,
}

impl GpuBufferPool {
    pub fn new(device: &wgpu::Device) -> Self {
        const VERTEX_CAPACITY: u32 = 4 * 1024 * 1024;   // 4 M verts × 32 B = 128 MB
        const INDEX_CAPACITY: u32  = 16 * 1024 * 1024;  // 16 M idxs  ×  4 B =  64 MB

        let vb_bytes = VERTEX_CAPACITY as u64 * std::mem::size_of::<PackedVertex>() as u64;
        let ib_bytes = INDEX_CAPACITY  as u64 * std::mem::size_of::<u32>() as u64;

        let vertex_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry Pool Vertex Buffer"),
            size: vb_bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        }));
        let index_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry Pool Index Buffer"),
            size: ib_bytes,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        }));

        gpu_transfer::track_alloc(vb_bytes + ib_bytes);
        log::info!("GpuBufferPool: {} MB VB + {} MB IB", vb_bytes / (1024 * 1024), ib_bytes / (1024 * 1024));

        Self {
            vertex_buffer,
            index_buffer,
            vertex_cursor: 0,
            index_cursor:  0,
            vertex_capacity: VERTEX_CAPACITY,
            index_capacity:  INDEX_CAPACITY,
        }
    }

    /// Upload `vertices` + `indices` into the pool and return their base offsets.
    /// Returns `None` if the pool is full (capacity exhausted).
    pub fn alloc(
        &mut self,
        queue: &wgpu::Queue,
        vertices: &[PackedVertex],
        indices:  &[u32],
    ) -> Option<MeshAlloc> {
        let vc = vertices.len() as u32;
        let ic = indices.len()  as u32;

        if self.vertex_cursor + vc > self.vertex_capacity
            || self.index_cursor + ic > self.index_capacity
        {
            log::error!(
                "GpuBufferPool exhausted: verts {}/{}, idxs {}/{}",
                self.vertex_cursor + vc, self.vertex_capacity,
                self.index_cursor  + ic, self.index_capacity,
            );
            return None;
        }

        let base_vertex = self.vertex_cursor;
        let first_index = self.index_cursor;

        let vb_byte_off = base_vertex as u64 * std::mem::size_of::<PackedVertex>() as u64;
        let ib_byte_off = first_index  as u64 * std::mem::size_of::<u32>() as u64;

        queue.write_buffer(&self.vertex_buffer, vb_byte_off, bytemuck::cast_slice(vertices));
        queue.write_buffer(&self.index_buffer,  ib_byte_off, bytemuck::cast_slice(indices));

        let uploaded = vc as u64 * std::mem::size_of::<PackedVertex>() as u64
                      + ic as u64 * std::mem::size_of::<u32>() as u64;
        gpu_transfer::track_upload(uploaded);

        self.vertex_cursor += vc;
        self.index_cursor  += ic;

        Some(MeshAlloc { base_vertex, first_index, vertex_count: vc, index_count: ic })
    }

    pub fn vertex_used(&self) -> u32 { self.vertex_cursor }
    pub fn index_used(&self)  -> u32 { self.index_cursor  }
}
