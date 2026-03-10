//! Unified geometry buffer pool.
//!
//! All mesh vertices and indices live in two large shared GPU buffers.
//! This eliminates per-mesh VB/IB bind overhead and enables
//! `multi_draw_indexed_indirect` — one call for all opaque geometry.
//!
//! # Dynamic growth
//! The pool starts at a modest initial size and doubles on demand up to a VRAM
//! cap (≈70 % of `device.limits().max_buffer_size`, hard-capped at 4 GiB).
//! When the cap is reached the pool stops growing and `alloc` returns `None`;
//! callers should fall back to standalone (system-memory) buffers.  A warning
//! is emitted on the first overflow and every 100 subsequent overflow calls.
//!
//! Growing reuses the same `SharedPoolBuffer` handle: render passes that hold
//! an `Arc<Mutex<Arc<wgpu::Buffer>>>` reference automatically see the new,
//! larger buffer after a grow without any re-wiring.  The grow copies all
//! existing data via a one-shot command encoder so that previously computed
//! `base_vertex` / `first_index` offsets remain valid.

use std::sync::{Arc, Mutex};
use crate::mesh::PackedVertex;
use crate::gpu_transfer;

/// A growable, shared handle to a pool buffer.
///
/// Render passes store this and lock it each frame to obtain the current
/// `Arc<wgpu::Buffer>`.  When the pool grows the inner `Arc` is atomically
/// swapped so all holders see the new buffer without code changes.
pub type SharedPoolBuffer = Arc<Mutex<Arc<wgpu::Buffer>>>;

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
/// Starts at 4 M vertices (128 MB) + 16 M indices (64 MB) and doubles on
/// demand.  Growth stops at the VRAM cap; subsequent allocs return `None`
/// and callers must fall back to standalone sys-mem buffers.
pub struct GpuBufferPool {
    device: Arc<wgpu::Device>,

    /// Shared handles — inner `Arc` is swapped in-place on each grow.
    pub shared_vertex_buffer: SharedPoolBuffer,
    pub shared_index_buffer:  SharedPoolBuffer,

    vertex_cursor:   u32,
    index_cursor:    u32,
    vertex_capacity: u32,
    index_capacity:  u32,

    /// Total bytes allocated across all pool generations (VB + IB combined).
    allocated_vram_bytes: u64,
    /// Hard cap: stop growing above this many bytes.
    vram_cap_bytes: u64,
    /// Consecutive overflow call count — used to throttle warnings.
    overflow_count: u64,
}

impl GpuBufferPool {
    /// Create a new pool, sizing the VRAM cap from device limits.
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        const INIT_VERTEX_CAP: u32 = 4  * 1024 * 1024;  // 4 M verts × 32 B = 128 MB
        const INIT_INDEX_CAP:  u32 = 16 * 1024 * 1024;  // 16 M idxs  × 4 B  =  64 MB

        // Cap growth at 70 % of the device's single-buffer limit, max 4 GiB.
        let max_buf       = device.limits().max_buffer_size;
        let vram_cap      = (max_buf * 7 / 10).min(4 * 1024 * 1024 * 1024);

        let (vb, ib, vb_bytes, ib_bytes) =
            Self::make_buffers(&device, INIT_VERTEX_CAP, INIT_INDEX_CAP);

        let total = vb_bytes + ib_bytes;
        gpu_transfer::track_alloc(total);
        log::info!(
            "GpuBufferPool: {} MB VB + {} MB IB  (cap {:.0} MB, device max_buf {:.0} MB)",
            vb_bytes / (1024 * 1024), ib_bytes / (1024 * 1024),
            vram_cap as f64 / (1024.0 * 1024.0),
            max_buf  as f64 / (1024.0 * 1024.0),
        );

        Self {
            device,
            shared_vertex_buffer: Arc::new(Mutex::new(Arc::new(vb))),
            shared_index_buffer:  Arc::new(Mutex::new(Arc::new(ib))),
            vertex_cursor:   0,
            index_cursor:    0,
            vertex_capacity: INIT_VERTEX_CAP,
            index_capacity:  INIT_INDEX_CAP,
            allocated_vram_bytes: total,
            vram_cap_bytes:  vram_cap,
            overflow_count:  0,
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn make_buffers(
        device:    &wgpu::Device,
        vert_cap:  u32,
        idx_cap:   u32,
    ) -> (wgpu::Buffer, wgpu::Buffer, u64, u64) {
        let vb_bytes = vert_cap as u64 * std::mem::size_of::<PackedVertex>() as u64;
        let ib_bytes = idx_cap  as u64 * std::mem::size_of::<u32>()          as u64;

        let vb = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry Pool Vertex Buffer"),
            size:  vb_bytes,
            usage: wgpu::BufferUsages::VERTEX
                 | wgpu::BufferUsages::COPY_DST
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });
        let ib = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Geometry Pool Index Buffer"),
            size:  ib_bytes,
            usage: wgpu::BufferUsages::INDEX
                 | wgpu::BufferUsages::COPY_DST
                 | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::BLAS_INPUT,
            mapped_at_creation: false,
        });
        (vb, ib, vb_bytes, ib_bytes)
    }

    fn current_pool_bytes(&self) -> u64 {
        self.vertex_capacity as u64 * std::mem::size_of::<PackedVertex>() as u64
            + self.index_capacity as u64 * std::mem::size_of::<u32>() as u64
    }

    /// Attempt to double the pool capacity.
    ///
    /// Copies all existing data to the new buffers via an immediately-submitted
    /// command encoder so the offsets are valid for the next frame.  Returns
    /// `false` if the doubled size would exceed `vram_cap_bytes` or the device's
    /// per-buffer limit.
    fn grow(&mut self, queue: &wgpu::Queue) -> bool {
        let new_vc = self.vertex_capacity.saturating_mul(2);
        let new_ic = self.index_capacity.saturating_mul(2);

        let new_vb_bytes = new_vc as u64 * std::mem::size_of::<PackedVertex>() as u64;
        let new_ib_bytes = new_ic as u64 * std::mem::size_of::<u32>()          as u64;

        // Reject if the new total would exceed the VRAM cap.
        let new_total = new_vb_bytes + new_ib_bytes;
        if self.allocated_vram_bytes.saturating_add(new_total - self.current_pool_bytes())
            > self.vram_cap_bytes
        {
            return false;
        }
        // Also respect the device's per-buffer size limit.
        let max_buf = self.device.limits().max_buffer_size;
        if new_vb_bytes > max_buf || new_ib_bytes > max_buf {
            return false;
        }

        let (new_vb, new_ib, _, _) =
            Self::make_buffers(&self.device, new_vc, new_ic);
        let new_vb = Arc::new(new_vb);
        let new_ib = Arc::new(new_ib);

        // Copy existing data to the new buffers.
        let copy_vb_bytes =
            self.vertex_cursor as u64 * std::mem::size_of::<PackedVertex>() as u64;
        let copy_ib_bytes =
            self.index_cursor  as u64 * std::mem::size_of::<u32>() as u64;

        let old_vb = self.shared_vertex_buffer.lock().unwrap().clone();
        let old_ib = self.shared_index_buffer .lock().unwrap().clone();

        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("Pool Growth Copy") },
        );
        if copy_vb_bytes > 0 {
            encoder.copy_buffer_to_buffer(&old_vb, 0, &new_vb, 0, copy_vb_bytes);
        }
        if copy_ib_bytes > 0 {
            encoder.copy_buffer_to_buffer(&old_ib, 0, &new_ib, 0, copy_ib_bytes);
        }
        queue.submit(std::iter::once(encoder.finish()));

        let added = new_total - self.current_pool_bytes();
        gpu_transfer::track_alloc(added);

        log::info!(
            "GpuBufferPool: grew to {} MB VB ({} M verts) + {} MB IB ({} M idxs)  \
             [{:.0} MB total allocated]",
            new_vb_bytes / (1024 * 1024), new_vc / (1024 * 1024),
            new_ib_bytes / (1024 * 1024), new_ic / (1024 * 1024),
            (self.allocated_vram_bytes + added) as f64 / (1024.0 * 1024.0),
        );

        // Swap the shared handles — all render passes and future GpuMeshes see
        // the new buffer automatically.
        *self.shared_vertex_buffer.lock().unwrap() = new_vb;
        *self.shared_index_buffer .lock().unwrap() = new_ib;

        self.allocated_vram_bytes += added;
        self.vertex_capacity = new_vc;
        self.index_capacity  = new_ic;

        true
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Upload `vertices` + `indices` into the pool and return their base offsets.
    ///
    /// Grows the pool (up to the VRAM cap) automatically before returning `None`.
    /// When `None` is returned the caller should fall back to a standalone
    /// sys-mem buffer via [`crate::mesh::GpuMesh::upload_standalone`].
    pub fn alloc(
        &mut self,
        queue:    &wgpu::Queue,
        vertices: &[PackedVertex],
        indices:  &[u32],
    ) -> Option<MeshAlloc> {
        let vc = vertices.len() as u32;
        let ic = indices.len()  as u32;

        // Grow until the request fits or we hit the VRAM cap.
        while self.vertex_cursor + vc > self.vertex_capacity
            || self.index_cursor  + ic > self.index_capacity
        {
            if !self.grow(queue) {
                self.overflow_count = self.overflow_count.wrapping_add(1);
                if self.overflow_count == 1 || self.overflow_count % 100 == 0 {
                    log::warn!(
                        "GpuBufferPool: VRAM cap reached ({:.0} MB used / {:.0} MB cap). \
                         Mesh overflowing to system memory \
                         [verts {}/{}, idxs {}/{}, overflow #{}].",
                        self.allocated_vram_bytes as f64 / (1024.0 * 1024.0),
                        self.vram_cap_bytes        as f64 / (1024.0 * 1024.0),
                        self.vertex_cursor, self.vertex_capacity,
                        self.index_cursor,  self.index_capacity,
                        self.overflow_count,
                    );
                }
                return None;
            }
        }

        let base_vertex = self.vertex_cursor;
        let first_index = self.index_cursor;

        let vb_byte_off = base_vertex as u64 * std::mem::size_of::<PackedVertex>() as u64;
        let ib_byte_off = first_index  as u64 * std::mem::size_of::<u32>() as u64;

        // Write into the current (possibly just-grown) buffer.
        let vb = self.shared_vertex_buffer.lock().unwrap().clone();
        let ib = self.shared_index_buffer .lock().unwrap().clone();
        queue.write_buffer(&vb, vb_byte_off, bytemuck::cast_slice(vertices));
        queue.write_buffer(&ib, ib_byte_off, bytemuck::cast_slice(indices));

        let uploaded = vc as u64 * std::mem::size_of::<PackedVertex>() as u64
                     + ic as u64 * std::mem::size_of::<u32>() as u64;
        gpu_transfer::track_upload(uploaded);

        self.vertex_cursor += vc;
        self.index_cursor  += ic;

        Some(MeshAlloc { base_vertex, first_index, vertex_count: vc, index_count: ic })
    }

    /// Snapshot of the current vertex buffer `Arc` (for `GpuMesh` construction).
    pub fn current_vertex_buffer(&self) -> Arc<wgpu::Buffer> {
        self.shared_vertex_buffer.lock().unwrap().clone()
    }

    /// Snapshot of the current index buffer `Arc` (for `GpuMesh` construction).
    pub fn current_index_buffer(&self) -> Arc<wgpu::Buffer> {
        self.shared_index_buffer.lock().unwrap().clone()
    }

    pub fn vertex_used(&self) -> u32 { self.vertex_cursor }
    pub fn index_used(&self)  -> u32 { self.index_cursor  }
    pub fn vertex_capacity(&self) -> u32 { self.vertex_capacity }
    pub fn index_capacity(&self)  -> u32 { self.index_capacity  }
}
