//! GPU texture loading from SolidRS images

use crate::{Result, AssetError};
use solid_rs::scene::{Image, ImageSource};
use wgpu::util::DeviceExt;
use std::fs;

/// Upload a SolidRS image to a GPU texture
pub fn load_texture(
    image: &Image,
    srgb: bool,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> Result<wgpu::Texture> {
    // Get image data (either from embedded bytes or URI)
    let data_vec: Vec<u8>;
    let image_data = match &image.source {
        ImageSource::Embedded { data, .. } => data.as_slice(),
        ImageSource::Uri(path) => {
            // Load external image file
            data_vec = fs::read(path)
                .map_err(|e| AssetError::InvalidData(format!(
                    "Failed to read image file '{}': {}",
                    path, e
                )))?;
            data_vec.as_slice()
        }
    };

    // Decode image using the `image` crate
    let decoded = image::load_from_memory(image_data)
        .map_err(|e| AssetError::InvalidData(format!("Failed to decode image: {}", e)))?;

    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();

    // Determine texture format based on sRGB flag
    let format = if srgb {
        wgpu::TextureFormat::Rgba8UnormSrgb
    } else {
        wgpu::TextureFormat::Rgba8Unorm
    };

    // Create GPU texture
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };

    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some(&format!("Texture {}", if image.name.is_empty() { "unnamed" } else { &image.name })),
            size,
            mip_level_count: 1, // TODO: Generate mipmaps in Phase 3
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        rgba.as_raw(),
    );

    Ok(texture)
}

/// Create a wgpu sampler from SolidRS sampler settings
pub fn create_sampler(
    sampler: &solid_rs::scene::Sampler,
    device: &wgpu::Device,
) -> wgpu::Sampler {
    use solid_rs::scene::WrapMode;

    let address_mode_u = match sampler.wrap_s {
        WrapMode::Repeat => wgpu::AddressMode::Repeat,
        WrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        WrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
    };

    let address_mode_v = match sampler.wrap_t {
        WrapMode::Repeat => wgpu::AddressMode::Repeat,
        WrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        WrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
    };

    use solid_rs::scene::FilterMode as SolidFilter;

    // Mag filter doesn't use mipmaps (magnification = texture larger than source)
    let mag_filter = match sampler.mag_filter {
        SolidFilter::Nearest | SolidFilter::NearestMipmapNearest | SolidFilter::NearestMipmapLinear => {
            wgpu::FilterMode::Nearest
        }
        SolidFilter::Linear | SolidFilter::LinearMipmapNearest | SolidFilter::LinearMipmapLinear => {
            wgpu::FilterMode::Linear
        }
    };

    // Min filter and mipmap filter come from min_filter setting
    let (min_filter, mipmap_filter) = match sampler.min_filter {
        SolidFilter::Nearest => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Nearest),
        SolidFilter::Linear => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Linear),
        SolidFilter::NearestMipmapNearest => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Nearest),
        SolidFilter::LinearMipmapNearest => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Nearest),
        SolidFilter::NearestMipmapLinear => (wgpu::FilterMode::Nearest, wgpu::MipmapFilterMode::Linear),
        SolidFilter::LinearMipmapLinear => (wgpu::FilterMode::Linear, wgpu::MipmapFilterMode::Linear),
    };

    device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("SolidRS Sampler"),
        address_mode_u,
        address_mode_v,
        address_mode_w: address_mode_u, // Use U mode for W
        mag_filter,
        min_filter,
        mipmap_filter,
        lod_min_clamp: 0.0,
        lod_max_clamp: 32.0,
        compare: None,
        anisotropy_clamp: 1,
        border_color: None,
    })
}
