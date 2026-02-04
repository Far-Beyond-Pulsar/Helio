use blade_graphics as gpu;
use helio_core::{Texture2D, TextureFormat};

pub struct RenderTarget {
    pub color: Texture2D,
    pub depth: Option<Texture2D>,
    pub width: u32,
    pub height: u32,
}

impl RenderTarget {
    pub fn new(
        context: &gpu::Context,
        width: u32,
        height: u32,
        format: TextureFormat,
        name: &str,
    ) -> Self {
        let color = Texture2D::new(
            context,
            width,
            height,
            format,
            gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            1,
            &format!("{}_Color", name),
        );
        
        let depth = Some(Texture2D::new(
            context,
            width,
            height,
            TextureFormat::Depth32Float,
            gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            1,
            &format!("{}_Depth", name),
        ));
        
        Self {
            color,
            depth,
            width,
            height,
        }
    }
    
    pub fn new_color_only(
        context: &gpu::Context,
        width: u32,
        height: u32,
        format: TextureFormat,
        name: &str,
    ) -> Self {
        let color = Texture2D::new(
            context,
            width,
            height,
            format,
            gpu::TextureUsage::TARGET | gpu::TextureUsage::RESOURCE,
            1,
            name,
        );
        
        Self {
            color,
            depth: None,
            width,
            height,
        }
    }
}
