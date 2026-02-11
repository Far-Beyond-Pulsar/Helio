use blade_graphics as gpu;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Unique identifier for a texture resource
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(u32);

impl TextureId {
    pub fn new(id: u32) -> Self {
        Self(id)
    }
    
    pub fn as_u32(&self) -> u32 {
        self.0
    }
}

/// Texture data loaded from an image file
pub struct TextureData {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub format: gpu::TextureFormat,
}

impl TextureData {
    /// Load texture data from a PNG file
    pub fn from_png(path: impl AsRef<Path>) -> Result<Self, String> {
        let img = image::open(path.as_ref())
            .map_err(|e| format!("Failed to load image {:?}: {}", path.as_ref(), e))?;
        
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        
        Ok(Self {
            width,
            height,
            data: rgba.into_raw(),
            format: gpu::TextureFormat::Rgba8Unorm,
        })
    }
    
    /// Create texture data from raw RGBA bytes
    pub fn from_rgba_bytes(width: u32, height: u32, data: Vec<u8>) -> Result<Self, String> {
        if data.len() != (width * height * 4) as usize {
            return Err(format!(
                "Invalid data size: expected {} bytes for {}x{} RGBA, got {}",
                width * height * 4,
                width,
                height,
                data.len()
            ));
        }
        
        Ok(Self {
            width,
            height,
            data,
            format: gpu::TextureFormat::Rgba8Unorm,
        })
    }
}

/// GPU texture resource with associated metadata
pub struct GpuTexture {
    pub texture: gpu::Texture,
    pub view: gpu::TextureView,
    pub sampler: gpu::Sampler,
    pub width: u32,
    pub height: u32,
}

/// Manages texture loading, GPU upload, and lifetime
/// 
/// This is designed to be shared across rendering features (billboards, materials, etc.)
pub struct TextureManager {
    context: Arc<gpu::Context>,
    textures: HashMap<TextureId, GpuTexture>,
    next_id: u32,
}

impl TextureManager {
    pub fn new(context: Arc<gpu::Context>) -> Self {
        Self {
            context,
            textures: HashMap::new(),
            next_id: 0,
        }
    }
    
    /// Load a PNG file and upload to GPU
    pub fn load_png(&mut self, path: impl AsRef<Path>) -> Result<TextureId, String> {
        let texture_data = TextureData::from_png(path)?;
        self.upload_texture(texture_data)
    }
    
    /// Upload raw texture data to GPU
    pub fn upload_texture(&mut self, data: TextureData) -> Result<TextureId, String> {
        let texture = self.context.create_texture(gpu::TextureDesc {
            name: "texture",
            format: data.format,
            size: gpu::Extent {
                width: data.width,
                height: data.height,
                depth: 1,
            },
            dimension: gpu::TextureDimension::D2,
            array_layer_count: 1,
            mip_level_count: 1,
            usage: gpu::TextureUsage::RESOURCE,
            sample_count: 1,
            external: None,
        });
        
        let view = self.context.create_texture_view(
            texture,
            gpu::TextureViewDesc {
                name: "texture_view",
                format: data.format,
                dimension: gpu::ViewDimension::D2,
                subresources: &Default::default(),
            },
        );
        
        let sampler = self.context.create_sampler(gpu::SamplerDesc {
            name: "texture_sampler",
            address_modes: [gpu::AddressMode::Repeat; 3],
            mag_filter: gpu::FilterMode::Linear,
            min_filter: gpu::FilterMode::Linear,
            mipmap_filter: gpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: Some(16.0),
            compare: None,
            border_color: None,
            anisotropy_clamp: 1,
        });
        
        // TODO: Upload texture data to GPU
        // blade-graphics might need a CommandEncoder or different API for texture upload
        // For now, we create the texture without data upload
        log::warn!("Texture data upload not yet implemented - texture will be uninitialized");
        
        let id = TextureId(self.next_id);
        self.next_id += 1;
        
        self.textures.insert(id, GpuTexture {
            texture,
            view,
            sampler,
            width: data.width,
            height: data.height,
        });
        
        Ok(id)
    }
    
    /// Get a GPU texture by ID
    pub fn get(&self, id: TextureId) -> Option<&GpuTexture> {
        self.textures.get(&id)
    }
    
    /// Remove a texture and free GPU resources
    pub fn remove(&mut self, id: TextureId) -> bool {
        self.textures.remove(&id).is_some()
    }
    
    /// Get the number of loaded textures
    pub fn count(&self) -> usize {
        self.textures.len()
    }
}
