//! GPU scene buffer managers with dirty tracking.
//!
//! Each manager wraps a `wgpu::Buffer` with a CPU-side `Vec` mirror.
//! Dirty tracking ensures `flush()` is a no-op when data hasn't changed.

use crate::upload;
use bytemuck::Zeroable;
use libhelio::{
    DrawIndexedIndirectArgs, GpuCameraUniforms, GpuDrawCall, GpuInstanceAabb, GpuInstanceData,
    GpuLight, GpuMaterial, GpuShadowMatrix,
};
use std::sync::Arc;

/// A grow-only GPU storage buffer with dirty-tracked CPU mirror.
///
/// - `flush()` is O(1) when clean (no-op)
/// - Automatically reallocates with 2× growth when capacity is exceeded
/// - Buffer usage includes `STORAGE | COPY_DST` (+ optionally `INDIRECT`)
pub struct GrowableBuffer<T: bytemuck::Pod> {
    buf: wgpu::Buffer,
    data: Vec<T>,
    dirty_range: Option<(usize, usize)>,
    capacity: usize,
    usage: wgpu::BufferUsages,
    label: &'static str,
    device: Arc<wgpu::Device>,
    buffer_version: u64,
}

impl<T: bytemuck::Pod> GrowableBuffer<T> {
    pub fn new(
        device: Arc<wgpu::Device>,
        initial_capacity: usize,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Self {
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
            dirty_range: None,
            capacity: initial_capacity,
            usage,
            label,
            device,
            buffer_version: 0,
        }
    }

    /// Returns a reference to the underlying GPU buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    /// Returns the buffer version, incremented each time the buffer is reallocated.
    ///
    /// Passes can use this to detect when bind groups need to be recreated.
    pub fn buffer_version(&self) -> u64 {
        self.buffer_version
    }

    /// Returns the number of elements currently stored.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if there are no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a read-only view of the CPU-side data (mirrors the GPU buffer).
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    fn mark_dirty_range(&mut self, start: usize, end: usize) {
        if start >= end {
            return;
        }
        match &mut self.dirty_range {
            Some((dirty_start, dirty_end)) => {
                *dirty_start = (*dirty_start).min(start);
                *dirty_end = (*dirty_end).max(end);
            }
            None => {
                self.dirty_range = Some((start, end));
            }
        }
    }

    /// Replaces the entire contents. Marks dirty.
    pub fn set_data(&mut self, data: Vec<T>) {
        self.data = data;
        self.dirty_range = (!self.data.is_empty()).then_some((0, self.data.len()));
    }

    /// Pushes one element and returns its index.
    pub fn push(&mut self, item: T) -> usize {
        let index = self.data.len();
        self.data.push(item);
        self.mark_dirty_range(index, index + 1);
        index
    }

    /// Appends a slice of elements and returns the written index range.
    pub fn extend_from_slice(&mut self, items: &[T]) -> std::ops::Range<usize>
    where
        T: Copy,
    {
        let start = self.data.len();
        self.data.extend_from_slice(items);
        let end = self.data.len();
        self.mark_dirty_range(start, end);
        start..end
    }

    /// Updates one element in-place. Returns `false` if the index is out of bounds.
    pub fn update(&mut self, index: usize, item: T) -> bool {
        let Some(slot) = self.data.get_mut(index) else {
            return false;
        };
        *slot = item;
        self.mark_dirty_range(index, index + 1);
        true
    }

    /// Removes one element in O(1) by swap-removing it. Returns the removed item.
    pub fn swap_remove(&mut self, index: usize) -> Option<T> {
        if index >= self.data.len() {
            return None;
        }
        let last_index = self.data.len() - 1;
        let removed = self.data.swap_remove(index);
        if index < self.data.len() {
            self.mark_dirty_range(index, index + 1);
        } else if index < last_index {
            self.mark_dirty_range(index, index);
        }
        Some(removed)
    }

    /// Flushes dirty data to GPU. O(1) if clean.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        let Some((start, end)) = self.dirty_range else {
            return;
        };
        if self.data.is_empty() {
            self.dirty_range = None;
            return;
        }

        // Grow buffer if needed
        if self.data.len() > self.capacity {
            self.capacity = self.data.len() * 2;
            let new_size = (self.capacity * std::mem::size_of::<T>()).max(64) as u64;
            self.buffer_version += 1;
            self.buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(self.label),
                size: new_size,
                usage: self.usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            upload::write_buffer(queue, &self.buf, 0, bytemuck::cast_slice(&self.data));
            self.dirty_range = None;
            return;
        }
        let end = end.min(self.data.len());
        if start >= end {
            self.dirty_range = None;
            return;
        }
        let byte_offset = (start * std::mem::size_of::<T>()) as u64;
        upload::write_buffer(
            queue,
            &self.buf,
            byte_offset,
            bytemuck::cast_slice(&self.data[start..end]),
        );
        self.dirty_range = None;
    }

    /// Marks clean without flushing (use when buffer was written by GPU).
    pub fn mark_clean(&mut self) {
        self.dirty_range = None;
    }
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

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buf
    }

    /// Returns the camera world-space position as `[x, y, z]`.
    pub fn position(&self) -> [f32; 3] {
        let p = self.data.position_near;
        [p[0], p[1], p[2]]
    }

    /// Returns the camera forward direction as `[x, y, z]`.
    pub fn forward(&self) -> [f32; 3] {
        let f = self.data.forward_far;
        [f[0], f[1], f[2]]
    }

    /// Returns a reference to the raw GPU camera uniform data.
    pub fn data(&self) -> &GpuCameraUniforms {
        &self.data
    }

    pub fn update(&mut self, camera: GpuCameraUniforms) {
        self.data = camera;
        self.dirty = true;
    }

    pub fn flush(&mut self, queue: &wgpu::Queue) {
        if !self.dirty {
            return;
        }
        upload::write_buffer(queue, &self.buf, 0, bytemuck::bytes_of(&self.data));
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
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE,
            "Instance Buffer",
        ))
    }
}

impl std::ops::Deref for GpuInstanceBuffer {
    type Target = GrowableBuffer<GpuInstanceData>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuInstanceBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuAabbBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE,
            "AABB Buffer",
        ))
    }
}

impl std::ops::Deref for GpuAabbBuffer {
    type Target = GrowableBuffer<GpuInstanceAabb>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuAabbBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuDrawCallBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE,
            "DrawCall Buffer",
        ))
    }
}

impl std::ops::Deref for GpuDrawCallBuffer {
    type Target = GrowableBuffer<GpuDrawCall>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuDrawCallBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuLightBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            1024,
            wgpu::BufferUsages::STORAGE,
            "Light Buffer",
        ))
    }
}

impl std::ops::Deref for GpuLightBuffer {
    type Target = GrowableBuffer<GpuLight>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuLightBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuMaterialBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            2048,
            wgpu::BufferUsages::STORAGE,
            "Material Buffer",
        ))
    }
}

impl std::ops::Deref for GpuMaterialBuffer {
    type Target = GrowableBuffer<GpuMaterial>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuMaterialBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuShadowMatrixBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            256,
            wgpu::BufferUsages::STORAGE,
            "Shadow Matrix Buffer",
        ))
    }
}

impl std::ops::Deref for GpuShadowMatrixBuffer {
    type Target = GrowableBuffer<GpuShadowMatrix>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuShadowMatrixBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
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
}

impl std::ops::Deref for GpuIndirectBuffer {
    type Target = GrowableBuffer<DrawIndexedIndirectArgs>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuIndirectBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl GpuVisibilityBuffer {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self(GrowableBuffer::new(
            device,
            4096,
            wgpu::BufferUsages::STORAGE,
            "Visibility Buffer",
        ))
    }
}

impl std::ops::Deref for GpuVisibilityBuffer {
    type Target = GrowableBuffer<u32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl std::ops::DerefMut for GpuVisibilityBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
