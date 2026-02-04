use crate::gpu;
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

pub type ResourceHandle = u64;

pub struct GpuBufferPool {
    buffers: HashMap<ResourceHandle, gpu::Buffer>,
    next_handle: ResourceHandle,
}

impl GpuBufferPool {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            next_handle: 1,
        }
    }

    pub fn allocate(&mut self, context: &gpu::Context, desc: gpu::BufferDesc) -> ResourceHandle {
        let handle = self.next_handle;
        self.next_handle += 1;
        let buffer = context.create_buffer(desc);
        self.buffers.insert(handle, buffer);
        handle
    }

    pub fn get(&self, handle: ResourceHandle) -> Option<gpu::Buffer> {
        self.buffers.get(&handle).copied()
    }

    pub fn free(&mut self, handle: ResourceHandle, context: &gpu::Context) {
        if let Some(buffer) = self.buffers.remove(&handle) {
            context.destroy_buffer(buffer);
        }
    }

    pub fn clear(&mut self, context: &gpu::Context) {
        for (_, buffer) in self.buffers.drain() {
            context.destroy_buffer(buffer);
        }
    }
}

impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GpuTexturePool {
    textures: HashMap<ResourceHandle, gpu::Texture>,
    views: HashMap<ResourceHandle, gpu::TextureView>,
    next_handle: ResourceHandle,
}

impl GpuTexturePool {
    pub fn new() -> Self {
        Self {
            textures: HashMap::new(),
            views: HashMap::new(),
            next_handle: 1,
        }
    }

    pub fn allocate(
        &mut self,
        context: &gpu::Context,
        desc: gpu::TextureDesc,
        view_desc: gpu::TextureViewDesc,
    ) -> ResourceHandle {
        let handle = self.next_handle;
        self.next_handle += 1;
        let texture = context.create_texture(desc);
        let view = context.create_texture_view(texture, view_desc);
        self.textures.insert(handle, texture);
        self.views.insert(handle, view);
        handle
    }

    pub fn get_texture(&self, handle: ResourceHandle) -> Option<gpu::Texture> {
        self.textures.get(&handle).copied()
    }

    pub fn get_view(&self, handle: ResourceHandle) -> Option<gpu::TextureView> {
        self.views.get(&handle).copied()
    }

    pub fn free(&mut self, handle: ResourceHandle, context: &gpu::Context) {
        if let Some(view) = self.views.remove(&handle) {
            context.destroy_texture_view(view);
        }
        if let Some(texture) = self.textures.remove(&handle) {
            context.destroy_texture(texture);
        }
    }

    pub fn clear(&mut self, context: &gpu::Context) {
        for (_, view) in self.views.drain() {
            context.destroy_texture_view(view);
        }
        for (_, texture) in self.textures.drain() {
            context.destroy_texture(texture);
        }
    }
}

impl Default for GpuTexturePool {
    fn default() -> Self {
        Self::new()
    }
}

pub struct GpuResourceManager {
    pub buffers: Arc<RwLock<GpuBufferPool>>,
    pub textures: Arc<RwLock<GpuTexturePool>>,
}

impl GpuResourceManager {
    pub fn new() -> Self {
        Self {
            buffers: Arc::new(RwLock::new(GpuBufferPool::new())),
            textures: Arc::new(RwLock::new(GpuTexturePool::new())),
        }
    }

    pub fn clear(&self, context: &gpu::Context) {
        self.buffers.write().clear(context);
        self.textures.write().clear(context);
    }
}

impl Default for GpuResourceManager {
    fn default() -> Self {
        Self::new()
    }
}
