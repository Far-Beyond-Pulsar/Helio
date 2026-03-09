//! Content-addressed GPU texture + sampler cache.
//!
//! Two goals:
//! 1. **No duplicate uploads** — identical pixel data (same hash) maps to one
//!    `wgpu::Texture` for the lifetime of the registry.  Critical for streaming
//!    engines (e.g. Stratum) that construct the same material for every chunk.
//! 2. **Sampler pooling** — the GPU has a hard cap on open sampler descriptors
//!    on some backends (D3D12: 2048 heap total).  All PBR materials share the
//!    same handful of samplers instead of allocating one each.
//!
//! Matches Unreal's `FTexture2DResource` content-addressing and
//! `FRHISamplerState` descriptor cache.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ── FNV-1a 64-bit ────────────────────────────────────────────────────────────

pub(crate) fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── Format tag (discriminant for content hash) ────────────────────────────────

fn format_tag(format: wgpu::TextureFormat) -> u8 {
    match format {
        wgpu::TextureFormat::Rgba8UnormSrgb      => 1,
        wgpu::TextureFormat::Rgba8Unorm          => 2,
        wgpu::TextureFormat::Bc1RgbaUnormSrgb    => 3,
        wgpu::TextureFormat::Bc3RgbaUnorm        => 4,
        wgpu::TextureFormat::Bc5RgUnorm          => 5,
        _                                         => 0,
    }
}

// ── TextureRegistry ───────────────────────────────────────────────────────────

/// Content-addressed GPU texture and sampler cache.
///
/// Call `get_or_upload` for every texture slot when building a material.
/// Identical pixel data returns the cached `Arc<wgpu::TextureView>` without
/// issuing any GPU work.
pub(crate) struct TextureRegistry {
    /// content hash → shared GPU texture view (wgpu keeps parent Texture alive)
    views:    HashMap<u64, Arc<wgpu::TextureView>>,
    /// sampler descriptor hash → shared sampler
    samplers: HashMap<u64, Arc<wgpu::Sampler>>,
}

impl TextureRegistry {
    pub fn new() -> Self {
        Self {
            views:    HashMap::new(),
            samplers: HashMap::new(),
        }
    }

    /// Return the cached view for this pixel data, or upload it and cache it.
    pub fn get_or_upload(
        &mut self,
        device: &wgpu::Device,
        queue:  &wgpu::Queue,
        data:   &[u8],
        width:  u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Arc<wgpu::TextureView> {
        let key = Self::content_hash(data, width, height, format);
        if let Some(v) = self.views.get(&key) {
            return Arc::clone(v);
        }

        let tex = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label:           Some("reg_texture"),
                size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format,
                usage:           wgpu::TextureUsages::TEXTURE_BINDING
                                 | wgpu::TextureUsages::COPY_DST,
                view_formats:    &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );
        let view = Arc::new(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.views.insert(key, Arc::clone(&view));
        view
    }

    /// Return a cached sampler for the given filter/address combination.
    ///
    /// All standard PBR materials use linear-repeat — after the first call they
    /// all share a single `wgpu::Sampler`.
    pub fn get_or_create_sampler(
        &mut self,
        device: &wgpu::Device,
        mag:    wgpu::FilterMode,
        min:    wgpu::FilterMode,
        mip:    wgpu::MipmapFilterMode,
        addr_u: wgpu::AddressMode,
        addr_v: wgpu::AddressMode,
    ) -> Arc<wgpu::Sampler> {
        let key_bytes: [u8; 5] = [
            mag    as u8,
            min    as u8,
            mip    as u8,
            addr_u as u8,
            addr_v as u8,
        ];
        let key = fnv1a(&key_bytes);
        if let Some(s) = self.samplers.get(&key) {
            return Arc::clone(s);
        }
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("reg_sampler"),
            address_mode_u: addr_u,
            address_mode_v: addr_v,
            mag_filter:     mag,
            min_filter:     min,
            mipmap_filter:  mip,
            ..Default::default()
        }));
        self.samplers.insert(key, Arc::clone(&sampler));
        sampler
    }

    /// 64-bit content hash — exposed so `MaterialRegistry` can fold it into the
    /// per-material hash without re-uploading or re-accessing the pixel data.
    pub fn content_hash(
        data:   &[u8],
        width:  u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> u64 {
        let mut h = fnv1a(&width.to_le_bytes())
            ^ fnv1a(&height.to_le_bytes()).wrapping_shl(32);
        h ^= format_tag(format) as u64;
        h = h.wrapping_mul(0x517cc1b727220a95);
        h ^= fnv1a(data);
        h
    }

    /// Number of unique textures currently resident.
    pub fn texture_count(&self) -> usize { self.views.len() }
    /// Number of unique samplers currently created.
    pub fn sampler_count(&self) -> usize { self.samplers.len() }
}
