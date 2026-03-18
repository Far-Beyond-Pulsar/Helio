//! GPU scene buffer managers with dirty tracking.
//!
//! Each manager wraps a `wgpu::Buffer` with a CPU-side `Vec` mirror.
//! Dirty tracking ensures `flush()` is a no-op when data hasn't changed.

use libhelio::{
    GpuCameraUniforms, GpuInstanceData, GpuInstanceAabb, GpuDrawCall,
    GpuLight, GpuMaterial, GpuShadowMatrix, DrawIndexedIndirectArgs,
};
use bytemuck::Zeroable;
use std::sync::Arc;

/// A grow-only GPU storage buffer with dirty-tracked CPU mirror.
///
/// - `flush()` is O(1) when clean (no-op)
/// - Automatically reallocates with 2× growth when capacity is exceeded
/// - Buffer usage includes `STORAGE | COPY_DST` (+ optionally `INDIRECT`)
pub struct GrowableBuffer<T: bytemuck::Pod> {
    buf: wgpu::Buffer,
    data: Vec<T>,
    dirty: bool,
    capacity: usize,
    usage: wgpu::BufferUsages,
    label: &'static str,
    device: Arc<wgpu::Device>,
}

impl<T: bytemuck::Pod> GrowableBuffer<T> {
    pub fn new(device: Arc<wgpu::Device>, initial_capacity: usize, usage: wgpu::BufferUsages, label: &'static str) -> Self {
        let byte_size = (initial_capacity * std::mem::size_of::<T>()).max(64) as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_size,
            usage: usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buf,
            data: Vec::with_capacity(initial_capacity),
            dirty: false,
            capacity: initial_capacity,
            usage,
            label,
            device,
        }
    }

    /// Returns a reference to the underlying GPU buffer.
    pub fn buffer(&self) -> &wgpu::Buffer { &self.buf }

    /// Returns the number of elements currently stored.
    pub fn len(&self) -> usize { self.data.len() }

    /// Returns true if there are no elements.
    pub fn is_empty(&self) -> bool { self.data.is_empty() }

    /// Replaces the entire contents. Marks dirty.
    pub fn set_data(&mut self, data: Vec<T>) {
        self.data = data;
        self.dirty = true;
    }

    /// Pushes one element. Marks dirty.
    pub fn push(&mut self, item: T) {
        self.data.push(item);
        self.dirty = true;
    }

    /// Flushes dirty data to GPU. O(1) if clean.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        if !self.dirty { return; }
        if self.data.is_empty() { self.dirty = false; return; }

        let bytes = bytemuck::cast_slice(&self.data);
        // Grow buffer if needed
        if self.data.len() > self.capacity {
            self.capacity = self.data.len() * 2;
            let new_size = (self.capacity * std::mem::size_of::<T>()).max(64) as u64;
            self.buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: new_size,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
        queue.write_buffer(&self.buf, 0, bytes);
        self.dirty = false;
    }

    /// Marks clean without flushing (use when buffer was written by GPU).
    pub fn mark_clean(&mut self) { self.dirty = false; }
}

// ─── Camera buffer ────────────────────────────────────────────────────────────

/// Single-element uniform buffer for the camera.
pub struct GpuCameraBuffer {
    buf: wgpu::Buffer,
    data: GpuCameraUniforms,
    dirty: bool,
}

impl GpuCameraBuffer {
    pub fn new(device: &wgpu::Device) -> Self {
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera Uniform"),
            size: std::mem::size_of::<GpuCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            buf,
            data: GpuCameraUniforms::zeroed(),
            dirty: true,
        }
    }

    pub fn buffer(&self) -> &wgpu::Buffer { &self.buf }

    pub fn update(&mut self, camera: GpuCameraUniforms) {
        self.data = camera;
        self.dirty = true;
    }

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        if !self.dirty { return; }
        queue.write_buffer(&self.buf, 0, bytemuck::bytes_of(&self.data));
        self.dirty = false;
    }
}

// ─── Typed manager aliases ────────────────────────────────────────────────────

/// Storage buffer for per-instance data.
pub struct GpuInstanceBuffer(pub GrowableBuffer<GpuInstanceData>);
/// Storage buffer for per-instance AABBs (for GPU culling).
pub struct GpuAabbBuffer(pub GrowableBuffer<GpuInstanceAabb>);
/// Storage buffer for draw call templates (source for indirect dispatch).
pub struct GpuDrawCallBuffer(pub GrowableBuffer<GpuDrawCall>);
/// Storage buffer for GPU lights.
pub struct GpuLightBuffer(pub GrowableBuffer<GpuLight>);
/// Storage buffer for GPU materials.
pub struct GpuMaterialBuffer(pub GrowableBuffer<GpuMaterial>);
/// Storage buffer for shadow matrices.
pub struct GpuShadowMatrixBuffer(pub GrowableBuffer<GpuShadowMatrix>);
/// Indirect draw command buffer (written by GPU compute, read by render passes).
pub struct GpuIndirectBuffer(pub GrowableBuffer<DrawIndexedIndirectArgs>);
/// Storage buffer for per-instance visibility bitmask (u32 per instance, 1=visible).
pub struct GpuVisibilityBuffer(pub GrowableBuffer<u32>);

impl GpuInstanceBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(device, 4096, wgpu::BufferUsages::STORAGE, "Instance Buffer"))
    }
    pub fn buffer(&self) -> &wgpu::Buffer { self.0.buffer() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn set_data(&mut self, data: Vec<GpuInstanceData>) { self.0.set_data(data); }
    pub fn flush(&mut self, queue: &wgpu::Queue) { self.0.flush(queue); }
}

impl GpuAabbBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(device, 4096, wgpu::BufferUsages::STORAGE, "AABB Buffer"))
    }
    pub fn buffer(&self) -> &wgpu::Buffer { self.0.buffer() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn set_data(&mut self, data: Vec<GpuInstanceAabb>) { self.0.set_data(data); }
    pub fn flush(&mut self, queue: &wgpu::Queue) { self.0.flush(queue); }
}

impl GpuDrawCallBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(device, 4096, wgpu::BufferUsages::STORAGE, "DrawCall Buffer"))
    }
    pub fn buffer(&self) -> &wgpu::Buffer { self.0.buffer() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn set_data(&mut self, data: Vec<GpuDrawCall>) { self.0.set_data(data); }
    pub fn flush(&mut self, queue: &wgpu::Queue) { self.0.flush(queue); }
}

impl GpuLightBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(device, 1024, wgpu::BufferUsages::STORAGE, "Light Buffer"))
    }
    pub fn buffer(&self) -> &wgpu::Buffer { self.0.buffer() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn set_data(&mut self, data: Vec<GpuLight>) { self.0.set_data(data); }
    pub fn flush(&mut self, queue: &wgpu::Queue) { self.0.flush(queue); }
}

impl GpuMaterialBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(device, 2048, wgpu::BufferUsages::STORAGE, "Material Buffer"))
    }
    pub fn buffer(&self) -> &wgpu::Buffer { self.0.buffer() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn set_data(&mut self, data: Vec<GpuMaterial>) { self.0.set_data(data); }
    pub fn flush(&mut self, queue: &wgpu::Queue) { self.0.flush(queue); }
}

impl GpuShadowMatrixBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(device, 256, wgpu::BufferUsages::STORAGE, "Shadow Matrix Buffer"))
    }
    pub fn buffer(&self) -> &wgpu::Buffer { self.0.buffer() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn set_data(&mut self, data: Vec<GpuShadowMatrix>) { self.0.set_data(data); }
    pub fn flush(&mut self, queue: &wgpu::Queue) { self.0.flush(queue); }
}

impl GpuIndirectBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
            "Indirect Draw Buffer",
        ))
    }
    pub fn buffer(&self) -> &wgpu::Buffer { self.0.buffer() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn flush(&mut self, queue: &wgpu::Queue) { self.0.flush(queue); }
}

impl GpuVisibilityBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(device, 4096, wgpu::BufferUsages::STORAGE, "Visibility Buffer"))
    }
    pub fn buffer(&self) -> &wgpu::Buffer { self.0.buffer() }
    pub fn len(&self) -> usize { self.0.len() }
    pub fn flush(&mut self, queue: &wgpu::Queue) { self.0.flush(queue); }
}
