use blade_graphics as gpu;
use std::sync::Arc;

pub struct GpuBuffer {
    pub buffer: gpu::Buffer,
    pub size: u64,
}

impl GpuBuffer {
    pub fn new(context: &gpu::Context, size: u64, name: &str) -> Self {
        let buffer = context.create_buffer(gpu::BufferDesc {
            name,
            size,
            memory: gpu::Memory::Device,
        });
        
        Self {
            buffer,
            size,
        }
    }
}

pub struct GpuTexture {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub size: gpu::Extent,
    pub format: gpu::TextureFormat,
    pub mip_levels: u32,
}

impl GpuTexture {
    pub fn new(
        context: &gpu::Context,
        size: gpu::Extent,
        format: gpu::TextureFormat,
        usage: gpu::TextureUsage,
        mip_levels: u32,
        name: &str,
    ) -> Self {
        let texture = context.create_texture(gpu::TextureDesc {
            name,
            format,
            size,
            array_layer_count: 1,
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: gpu::TextureDimension::D2,
            usage,
            external: None,
        });

        let view = context.create_texture_view(texture, gpu::TextureViewDesc {
            name,
            format,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });
        
        Self {
            texture,
            view,
            size,
            format,
            mip_levels,
        }
    }
}
