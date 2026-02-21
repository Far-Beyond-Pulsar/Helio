use std::collections::HashMap;
use std::sync::Arc;

/// Opaque handle to a texture in the TextureManager.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(u32);

pub struct GpuTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub size: (u32, u32),
}

impl GpuTexture {
    pub fn from_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        width: u32,
        height: u32,
        label: Option<&str>,
    ) -> Self {
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0, bytes_per_row: Some(4 * width), rows_per_image: Some(height),
            },
            size,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view, size: (width, height) }
    }
}

pub struct TextureManager {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    textures: Vec<Arc<GpuTexture>>,
    id_map: HashMap<String, TextureId>,
    /// Shared linear sampler for textures.
    pub sampler: wgpu::Sampler,
}

impl TextureManager {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        Self { device, queue, textures: Vec::new(), id_map: HashMap::new(), sampler }
    }

    pub fn load_png(&mut self, path: &str) -> Result<TextureId, String> {
        if let Some(&id) = self.id_map.get(path) { return Ok(id); }
        let img = image::open(path)
            .map_err(|e| format!("Failed to open {}: {}", path, e))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        let tex = GpuTexture::from_rgba8(&self.device, &self.queue, &img, w, h, Some(path));
        let id = TextureId(self.textures.len() as u32);
        self.textures.push(Arc::new(tex));
        self.id_map.insert(path.to_string(), id);
        Ok(id)
    }

    pub fn load_from_bytes(&mut self, key: &str, data: &[u8]) -> Result<TextureId, String> {
        if let Some(&id) = self.id_map.get(key) { return Ok(id); }
        let img = image::load_from_memory(data)
            .map_err(|e| format!("Failed to decode texture {}: {}", key, e))?
            .to_rgba8();
        let (w, h) = img.dimensions();
        let tex = GpuTexture::from_rgba8(&self.device, &self.queue, &img, w, h, Some(key));
        let id = TextureId(self.textures.len() as u32);
        self.textures.push(Arc::new(tex));
        self.id_map.insert(key.to_string(), id);
        Ok(id)
    }

    pub fn get(&self, id: TextureId) -> Option<Arc<GpuTexture>> {
        self.textures.get(id.0 as usize).cloned()
    }
}
