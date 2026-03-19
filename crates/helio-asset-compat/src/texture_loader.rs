//! Texture conversion from SolidRS assets to Helio uploads.

use std::fs;
use std::path::{Path, PathBuf};

use helio::{TextureSamplerDesc, TextureUpload};
use solid_rs::scene::{FilterMode as SolidFilter, Image, ImageSource, Sampler, Scene, Texture, WrapMode};

use crate::{AssetError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TextureSemantic {
    BaseColor,
    MetallicRoughness,
    Normal,
    Occlusion,
    Emissive,
    SpecularColor,
    SpecularWeight,
}

impl TextureSemantic {
    pub fn is_srgb(self) -> bool {
        matches!(
            self,
            TextureSemantic::BaseColor | TextureSemantic::Emissive | TextureSemantic::SpecularColor
        )
    }

    pub fn suffix(self) -> &'static str {
        match self {
            TextureSemantic::BaseColor => "base-color",
            TextureSemantic::MetallicRoughness => "metallic-roughness",
            TextureSemantic::Normal => "normal",
            TextureSemantic::Occlusion => "occlusion",
            TextureSemantic::Emissive => "emissive",
            TextureSemantic::SpecularColor => "specular-color",
            TextureSemantic::SpecularWeight => "specular-weight",
        }
    }
}

fn resolve_image_bytes(image: &Image, base_dir: &Path) -> Result<Vec<u8>> {
    match &image.source {
        ImageSource::Embedded { data, .. } => Ok(data.clone()),
        ImageSource::Uri(path) => {
            let resolved = {
                let candidate = PathBuf::from(path);
                if candidate.is_absolute() {
                    candidate
                } else {
                    base_dir.join(candidate)
                }
            };
            fs::read(&resolved).map_err(|e| {
                AssetError::InvalidData(format!(
                    "Failed to read image file '{}': {}",
                    resolved.display(),
                    e
                ))
            })
        }
    }
}

fn convert_sampler(sampler: &Sampler) -> TextureSamplerDesc {
    let address_mode = |wrap: WrapMode| match wrap {
        WrapMode::Repeat => wgpu::AddressMode::Repeat,
        WrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
        WrapMode::MirroredRepeat => wgpu::AddressMode::MirrorRepeat,
    };

    let mag_filter = match sampler.mag_filter {
        SolidFilter::Nearest | SolidFilter::NearestMipmapNearest | SolidFilter::NearestMipmapLinear => {
            wgpu::FilterMode::Nearest
        }
        SolidFilter::Linear | SolidFilter::LinearMipmapNearest | SolidFilter::LinearMipmapLinear => {
            wgpu::FilterMode::Linear
        }
    };

    let (min_filter, mipmap_filter) = match sampler.min_filter {
        SolidFilter::Nearest => (wgpu::FilterMode::Nearest, wgpu::FilterMode::Nearest),
        SolidFilter::Linear => (wgpu::FilterMode::Linear, wgpu::FilterMode::Linear),
        SolidFilter::NearestMipmapNearest => (wgpu::FilterMode::Nearest, wgpu::FilterMode::Nearest),
        SolidFilter::LinearMipmapNearest => (wgpu::FilterMode::Linear, wgpu::FilterMode::Nearest),
        SolidFilter::NearestMipmapLinear => (wgpu::FilterMode::Nearest, wgpu::FilterMode::Linear),
        SolidFilter::LinearMipmapLinear => (wgpu::FilterMode::Linear, wgpu::FilterMode::Linear),
    };

    TextureSamplerDesc {
        address_mode_u: address_mode(sampler.wrap_s),
        address_mode_v: address_mode(sampler.wrap_t),
        address_mode_w: address_mode(sampler.wrap_s),
        mag_filter,
        min_filter,
        mipmap_filter,
    }
}

pub fn load_texture_upload(
    scene: &Scene,
    texture_index: usize,
    semantic: TextureSemantic,
    base_dir: &Path,
) -> Result<TextureUpload> {
    let texture = scene.textures.get(texture_index).ok_or_else(|| {
        AssetError::InvalidData(format!("Texture index {} is out of bounds", texture_index))
    })?;
    let image = scene.images.get(texture.image_index).ok_or_else(|| {
        AssetError::InvalidData(format!(
            "Texture '{}' references missing image index {}",
            texture.name, texture.image_index
        ))
    })?;

    load_texture_upload_from_parts(texture, image, semantic, base_dir)
}

fn load_texture_upload_from_parts(
    texture: &Texture,
    image: &Image,
    semantic: TextureSemantic,
    base_dir: &Path,
) -> Result<TextureUpload> {
    let bytes = resolve_image_bytes(image, base_dir)?;
    let decoded = image::load_from_memory(&bytes)
        .map_err(|e| AssetError::InvalidData(format!("Failed to decode image: {}", e)))?;
    let rgba = decoded.to_rgba8();
    let (width, height) = rgba.dimensions();
    let label = if texture.name.is_empty() {
        format!(
            "{} ({})",
            if image.name.is_empty() { "texture" } else { &image.name },
            semantic.suffix()
        )
    } else {
        format!("{} ({})", texture.name, semantic.suffix())
    };

    Ok(TextureUpload::rgba8(
        label,
        width,
        height,
        semantic.is_srgb(),
        rgba.into_raw(),
        convert_sampler(&texture.sampler),
    ))
}
