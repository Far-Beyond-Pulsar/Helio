//! Resource management system

mod bindgroup;
mod pool;

pub use bindgroup::{BindGroupLayouts, BindGroupBuilder};
pub use pool::{TexturePool, BufferPool, TextureKey, BufferKey};

use std::collections::HashMap;
use std::sync::Arc;
use wgpu;

/// Unique identifier for a resource
#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub struct ResourceId(u64);

impl ResourceId {
    fn new() -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

/// Central resource manager for the renderer
pub struct ResourceManager {
    device: Arc<wgpu::Device>,

    /// Standard bind group layouts shared by all pipelines
    pub bind_group_layouts: BindGroupLayouts,

    // Resource pools for transient resources
    texture_pool: TexturePool,
    buffer_pool: BufferPool,

    // Persistent resources
    persistent_textures: HashMap<ResourceId, wgpu::Texture>,
    persistent_buffers: HashMap<ResourceId, wgpu::Buffer>,

    // Bind group cache
    bind_group_cache: HashMap<BindGroupKey, Arc<wgpu::BindGroup>>,
}

#[derive(Hash, Eq, PartialEq, Clone)]
struct BindGroupKey {
    layout_id: usize,
    resource_ids: Vec<ResourceId>,
}

impl ResourceManager {
    pub fn new(device: Arc<wgpu::Device>) -> Self {
        let bind_group_layouts = BindGroupLayouts::new(&device);
        let texture_pool = TexturePool::new(device.clone());
        let buffer_pool = BufferPool::new(device.clone());

        Self {
            device,
            bind_group_layouts,
            texture_pool,
            buffer_pool,
            persistent_textures: HashMap::new(),
            persistent_buffers: HashMap::new(),
            bind_group_cache: HashMap::new(),
        }
    }

    /// Create a persistent texture
    pub fn create_texture(&mut self, label: &str, desc: &wgpu::TextureDescriptor) -> ResourceId {
        let texture = self.device.create_texture(desc);
        let id = ResourceId::new();
        self.persistent_textures.insert(id, texture);
        log::debug!("Created persistent texture '{}' with id {:?}", label, id);
        id
    }

    /// Create a persistent buffer
    pub fn create_buffer(&mut self, label: &str, desc: &wgpu::BufferDescriptor) -> ResourceId {
        let buffer = self.device.create_buffer(desc);
        let id = ResourceId::new();
        self.persistent_buffers.insert(id, buffer);
        log::debug!("Created persistent buffer '{}' with id {:?}", label, id);
        id
    }

    /// Get a persistent texture
    pub fn get_texture(&self, id: ResourceId) -> Option<&wgpu::Texture> {
        self.persistent_textures.get(&id)
    }

    /// Get a persistent buffer
    pub fn get_buffer(&self, id: ResourceId) -> Option<&wgpu::Buffer> {
        self.persistent_buffers.get(&id)
    }

    /// Acquire a transient texture from the pool
    pub fn acquire_transient_texture(&mut self, desc: &wgpu::TextureDescriptor) -> ResourceId {
        let key = TextureKey::from_descriptor(desc);
        let _texture = self.texture_pool.acquire(key);
        let id = ResourceId::new();
        // TODO: Track transient resources
        id
    }

    /// Release a transient texture back to the pool
    pub fn release_transient_texture(&mut self, _id: ResourceId) {
        // TODO: Return to pool
    }

    /// Cleanup old pooled resources
    pub fn cleanup_pools(&mut self, current_frame: u64) {
        self.texture_pool.cleanup_old(current_frame);
        self.buffer_pool.cleanup_old(current_frame);
    }

    /// Get the device
    pub fn device(&self) -> &wgpu::Device {
        &self.device
    }
}
