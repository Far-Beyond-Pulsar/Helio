use blade_graphics as gpu;
use glam::{UVec2, UVec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextureFormat {
    R8Unorm,
    R8Snorm,
    R16Float,
    R32Float,
    RG8Unorm,
    RG16Float,
    RG32Float,
    RGBA8Unorm,
    RGBA8Srgb,
    RGBA16Float,
    RGBA32Float,
    BGRA8Unorm,
    BGRA8Srgb,
    Depth32Float,
    Depth24Plus,
    Depth24PlusStencil8,
    BC1,
    BC3,
    BC4,
    BC5,
    BC6H,
    BC7,
}

impl TextureFormat {
    pub fn to_gpu_format(self) -> gpu::TextureFormat {
        match self {
            TextureFormat::R8Unorm => gpu::TextureFormat::R8Unorm,
            TextureFormat::R16Float => gpu::TextureFormat::R16Float,
            TextureFormat::R32Float => gpu::TextureFormat::R32Float,
            TextureFormat::RG8Unorm => gpu::TextureFormat::Rg8Unorm,
            TextureFormat::RG16Float => gpu::TextureFormat::R16Float, // Fallback
            TextureFormat::RG32Float => gpu::TextureFormat::R32Float, // Fallback
            TextureFormat::RGBA8Unorm => gpu::TextureFormat::Rgba8Unorm,
            TextureFormat::RGBA8Srgb => gpu::TextureFormat::Rgba8UnormSrgb,
            TextureFormat::RGBA16Float => gpu::TextureFormat::Rgba16Float,
            TextureFormat::RGBA32Float => gpu::TextureFormat::Rgba32Float,
            TextureFormat::BGRA8Unorm => gpu::TextureFormat::Bgra8Unorm,
            TextureFormat::BGRA8Srgb => gpu::TextureFormat::Bgra8UnormSrgb,
            TextureFormat::Depth32Float => gpu::TextureFormat::Depth32Float,
            _ => gpu::TextureFormat::Rgba8Unorm,
        }
    }
    
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            TextureFormat::R8Unorm | TextureFormat::R8Snorm => 1,
            TextureFormat::R16Float | TextureFormat::RG8Unorm => 2,
            TextureFormat::R32Float | TextureFormat::RG16Float => 4,
            TextureFormat::RG32Float => 8,
            TextureFormat::RGBA8Unorm | TextureFormat::RGBA8Srgb => 4,
            TextureFormat::BGRA8Unorm | TextureFormat::BGRA8Srgb => 4,
            TextureFormat::RGBA16Float => 8,
            TextureFormat::RGBA32Float => 16,
            TextureFormat::Depth32Float => 4,
            TextureFormat::Depth24Plus => 4,
            TextureFormat::Depth24PlusStencil8 => 4,
            _ => 4,
        }
    }
}

pub struct Texture2D {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub format: TextureFormat,
    pub mip_levels: u32,
}

impl Texture2D {
    pub fn new(
        context: &gpu::Context,
        width: u32,
        height: u32,
        format: TextureFormat,
        usage: gpu::TextureUsage,
        mip_levels: u32,
        name: &str,
    ) -> Self {
        let gpu_format = format.to_gpu_format();
        
        let texture = context.create_texture(gpu::TextureDesc {
            name,
            format: gpu_format,
            size: gpu::Extent {
                width,
                height,
                depth: 1,
            },
            array_layer_count: 1,
            mip_level_count: mip_levels,
            dimension: gpu::TextureDimension::D2,
            usage,
        });
        
        let view = context.create_texture_view(gpu::TextureViewDesc {
            texture,
            name,
            format: gpu_format,
            dimension: gpu::ViewDimension::D2,
            subresources: &gpu::TextureSubresources::default(),
        });
        
        Self {
            texture,
            view,
            width,
            height,
            format,
            mip_levels,
        }
    }
    
    pub fn size(&self) -> UVec2 {
        UVec2::new(self.width, self.height)
    }
}

pub struct Texture3D {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    pub format: TextureFormat,
    pub mip_levels: u32,
}

impl Texture3D {
    pub fn new(
        context: &gpu::Context,
        width: u32,
        height: u32,
        depth: u32,
        format: TextureFormat,
        usage: gpu::TextureUsage,
        mip_levels: u32,
        name: &str,
    ) -> Self {
        let gpu_format = format.to_gpu_format();
        
        let texture = context.create_texture(gpu::TextureDesc {
            name,
            format: gpu_format,
            size: gpu::Extent {
                width,
                height,
                depth,
            },
            array_layer_count: 1,
            mip_level_count: mip_levels,
            dimension: gpu::TextureDimension::D3,
            usage,
        });
        
        let view = context.create_texture_view(gpu::TextureViewDesc {
            texture,
            name,
            format: gpu_format,
            dimension: gpu::ViewDimension::D3,
            subresources: &gpu::TextureSubresources::default(),
        });
        
        Self {
            texture,
            view,
            width,
            height,
            depth,
            format,
            mip_levels,
        }
    }
    
    pub fn size(&self) -> UVec3 {
        UVec3::new(self.width, self.height, self.depth)
    }
}

pub struct TextureCube {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub size: u32,
    pub format: TextureFormat,
    pub mip_levels: u32,
}

impl TextureCube {
    pub fn new(
        context: &gpu::Context,
        size: u32,
        format: TextureFormat,
        usage: gpu::TextureUsage,
        mip_levels: u32,
        name: &str,
    ) -> Self {
        let gpu_format = format.to_gpu_format();
        
        let texture = context.create_texture(gpu::TextureDesc {
            name,
            format: gpu_format,
            size: gpu::Extent {
                width: size,
                height: size,
                depth: 1,
            },
            array_layer_count: 6,
            mip_level_count: mip_levels,
            dimension: gpu::TextureDimension::D2,
            usage,
        });
        
        let view = context.create_texture_view(gpu::TextureViewDesc {
            texture,
            name,
            format: gpu_format,
            dimension: gpu::ViewDimension::Cube,
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
