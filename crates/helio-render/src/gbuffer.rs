use blade_graphics as gpu;
use helio_core::{Texture2D, TextureFormat};

pub struct GBuffer {
    pub albedo: Texture2D,
    pub normal: Texture2D,
    pub metallic_roughness: Texture2D,
    pub emissive: Texture2D,
    pub velocity: Texture2D,
    pub depth: Texture2D,
    pub width: u32,
    pub height: u32,
}

impl GBuffer {
    pub fn new(context: &gpu::Context, width: u32, height: u32) -> Self {
        let usage = gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE;
        
        Self {
            albedo: Texture2D::new(
                context,
                width,
                height,
                TextureFormat::RGBA8Srgb,
                usage,
                1,
                "GBuffer_Albedo",
            ),
            normal: Texture2D::new(
                context,
                width,
                height,
                TextureFormat::RGBA16Float,
                usage,
                1,
                "GBuffer_Normal",
            ),
            metallic_roughness: Texture2D::new(
                context,
                width,
                height,
                TextureFormat::RGBA8Unorm,
                usage,
                1,
                "GBuffer_MetallicRoughness",
            ),
            emissive: Texture2D::new(
                context,
                width,
                height,
                TextureFormat::RGBA16Float,
                usage,
                1,
                "GBuffer_Emissive",
            ),
            velocity: Texture2D::new(
                context,
                width,
                height,
                TextureFormat::RG16Float,
                usage,
                1,
                "GBuffer_Velocity",
            ),
            depth: Texture2D::new(
                context,
                width,
                height,
                TextureFormat::Depth32Float,
                gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
                1,
                "GBuffer_Depth",
            ),
            width,
            height,
        }
    }
}
