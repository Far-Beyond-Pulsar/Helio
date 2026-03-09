//! Resource pooling for efficient memory reuse
//!
//! Pools allow transient resources (like bloom mip chains) to be reused
//! across frames without constant allocation/deallocation.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu;

/// Key for texture pool lookup
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct TextureKey {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: wgpu::TextureFormat,
    pub usage: wgpu::TextureUsages,
    pub dimension: wgpu::TextureDimension,
}

impl TextureKey {
    pub fn from_descriptor(desc: &wgpu::TextureDescriptor) -> Self {
        Self {
            width: desc.size.width,
            height: desc.size.height,
            depth: desc.size.depth_or_array_layers,
            format: desc.format,
            usage: desc.usage,
            dimension: desc.dimension,
        }
    }

    pub fn to_descriptor<'a>(&self, label: Option<&'a str>) -> wgpu::TextureDescriptor<'a> {
        wgpu::TextureDescriptor {
            label,
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: self.depth,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: self.dimension,
            format: self.format,
            usage: self.usage,
            view_formats: &[],
        }
    }
}

/// Pool for reusing textures
pub struct TexturePool {
    device: Arc<wgpu::Device>,
    available: HashMap<TextureKey, Vec<wgpu::Texture>>,
    frame_last_used: HashMap<u64, u64>,
    next_id: u64,
}

impl TexturePool {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            available: HashMap::new(),
            frame_last_used: HashMap::new(),
            next_id: 0,
        }
    }

    /// Acquire a texture from the pool (or create new one)
    pub fn acquire(&mut self, key: TextureKey) -> wgpu::Texture {
        // Try to reuse existing texture
        if let Some(textures) = self.available.get_mut(&key) {
            if let Some(texture) = textures.pop() {
                log::trace!("Reusing pooled texture {:?}", key);
                return texture;
            }
        }

        // Create new texture
        log::debug!("Creating new pooled texture {:?}", key);
        let desc = key.to_descriptor(Some("Pooled Texture"));
        self.device.create_texture(&desc)
    }

    /// Release a texture back to the pool
    pub fn release(&mut self, texture: wgpu::Texture, key: TextureKey, frame: u64) {
        self.available.entry(key).or_default().push(texture);
        let id = self.next_id;
        self.next_id += 1;
        self.frame_last_used.insert(id, frame);
    }

    /// Cleanup textures that haven't been used in a while
    pub fn cleanup_old(&mut self, current_frame: u64) {
        const MAX_AGE: u64 = 60; // Keep textures for 60 frames (1 second at 60fps)

        self.available.retain(|_key, textures| {
            textures.retain(|_| {
                // For now, keep all textures (can't easily track individual texture age)
                // In production, would track texture creation frame
                true
            });
            !textures.is_empty()
        });

        // Cleanup frame tracking
        self.frame_last_used
            .retain(|_, &mut last_frame| current_frame - last_frame < MAX_AGE);
    }
}

/// Key for buffer pool lookup
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct BufferKey {
    pub size: u64,
    pub usage: wgpu::BufferUsages,
}

impl BufferKey {
    pub fn from_descriptor(desc: &wgpu::BufferDescriptor) -> Self {
        Self {
            size: desc.size,
            usage: desc.usage,
        }
    }

    pub fn to_descriptor<'a>(&self, label: Option<&'a str>) -> wgpu::BufferDescriptor<'a> {
        wgpu::BufferDescriptor {
            label,
            size: self.size,
            usage: self.usage,
            mapped_at_creation: false,
        }
    }
}

/// Pool for reusing buffers
pub struct BufferPool {
    device: Arc<wgpu::Device>,
    available: HashMap<BufferKey, Vec<wgpu::Buffer>>,
    frame_last_used: HashMap<u64, u64>,
    next_id: u64,
}

impl BufferPool {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        Self {
            device,
            available: HashMap::new(),
            frame_last_used: HashMap::new(),
            next_id: 0,
        }
    }

    /// Acquire a buffer from the pool (or create new one)
    pub fn acquire(&mut self, key: BufferKey) -> wgpu::Buffer {
        // Try to reuse existing buffer
        if let Some(buffers) = self.available.get_mut(&key) {
            if let Some(buffer) = buffers.pop() {
                log::trace!("Reusing pooled buffer {:?}", key);
                return buffer;
            }
        }

        // Create new buffer
        log::debug!("Creating new pooled buffer {:?}", key);
        let desc = key.to_descriptor(Some("Pooled Buffer"));
        self.device.create_buffer(&desc)
    }

    /// Release a buffer back to the pool
    pub fn release(&mut self, buffer: wgpu::Buffer, key: BufferKey, frame: u64) {
        self.available.entry(key).or_default().push(buffer);
        let id = self.next_id;
        self.next_id += 1;
        self.frame_last_used.insert(id, frame);
    }

    /// Cleanup buffers that haven't been used in a while
    pub fn cleanup_old(&mut self, current_frame: u64) {
        const MAX_AGE: u64 = 60; // Keep buffers for 60 frames

        self.available.retain(|_key, buffers| {
            buffers.retain(|_| true); // Keep all for now
            !buffers.is_empty()
        });

        self.frame_last_used
            .retain(|_, &mut last_frame| current_frame - last_frame < MAX_AGE);
    }
}
