use wgpu::util::DeviceExt;

// ── Texture format wrapper ─────────────────────────────────────────────────────

/// A commonly reused set of texture format aliases.
pub struct TextureFormat2D;
impl TextureFormat2D {
    pub const RGBA8:   wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    pub const RGBA16F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub const RGBA32F: wgpu::TextureFormat = wgpu::TextureFormat::Rgba32Float;
    pub const R32F:    wgpu::TextureFormat = wgpu::TextureFormat::R32Float;
    pub const RG32F:   wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;
}

// ── 2D bake texture ───────────────────────────────────────────────────────────

/// A GPU texture + view ready for use as a storage binding in compute shaders.
pub struct BakeTexture {
    pub texture: wgpu::Texture,
    pub view:    wgpu::TextureView,
    pub width:   u32,
    pub height:  u32,
    pub format:  wgpu::TextureFormat,
    pub mip_levels: u32,
}

impl BakeTexture {
    pub fn new(
        device: &wgpu::Device,
        label:  &str,
        width:  u32,
        height: u32,
        format: wgpu::TextureFormat,
        mip_levels: u32,
        extra_usage: wgpu::TextureUsages,
    ) -> Self {
        let usage = wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::COPY_DST
            | extra_usage;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some(label),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: mip_levels,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats:    &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label:           Some(&format!("{label}_view")),
            format:          Some(format),
            mip_level_count: Some(mip_levels),
            ..Default::default()
        });

        Self { texture, view, width, height, format, mip_levels }
    }

    pub fn read_back(
        &self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
    ) -> Vec<u8> {
        let bytes_per_texel = self.format.block_copy_size(None).unwrap_or(4) as u64;
        let padded_row = (self.width as u64 * bytes_per_texel + 255) & !255;
        let buf_size   = padded_row * self.height as u64;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("nebula_tex_readback"),
            size:               buf_size,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("nebula_tex_readback_enc"),
        });
        enc.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture:   &self.texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset:         0,
                    bytes_per_row:  Some(padded_row as u32),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d { width: self.width, height: self.height, depth_or_array_layers: 1 },
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        device.poll(wgpu::PollType::wait_indefinitely()).ok();
        let _ = rx.recv();

        let data: Vec<u8> = slice
            .get_mapped_range()
            .expect("Nebula texture readback buffer should be mapped")
            .to_vec();
        staging.unmap();

        // De-pad rows
        let row_bytes = (self.width as usize) * bytes_per_texel as usize;
        let padded    = padded_row as usize;
        let mut out   = Vec::with_capacity(row_bytes * self.height as usize);
        for row in 0..self.height as usize {
            out.extend_from_slice(&data[row * padded..row * padded + row_bytes]);
        }
        out
    }
}

// ── 2D texture array ──────────────────────────────────────────────────────────

/// A GPU texture array — used for lightmap atlases, cubemaps, etc.
pub struct BakeTextureArray {
    pub texture:     wgpu::Texture,
    pub view:        wgpu::TextureView,
    pub layer_views: Vec<wgpu::TextureView>,
    pub width:       u32,
    pub height:      u32,
    pub layers:      u32,
    pub format:      wgpu::TextureFormat,
}

impl BakeTextureArray {
    pub fn new(
        device: &wgpu::Device,
        label:  &str,
        width:  u32,
        height: u32,
        layers: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC
                | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let layer_views = (0..layers).map(|l| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension:           Some(wgpu::TextureViewDimension::D2),
                base_array_layer:    l,
                array_layer_count:   Some(1),
                label:               Some(&format!("{label}_layer_{l}")),
                ..Default::default()
            })
        }).collect();

        Self { texture, view, layer_views, width, height, layers, format }
    }
}
