use std::sync::Arc;

use libhelio::BakedPvsData;

use crate::bake::BakeError;
use crate::cache::{CachedAo, CachedLightmap, CachedProbes, CachedPvs};

/// GPU-resident baked data ready to be published into `FrameResources` each frame.
///
/// Owned by the [`BakeInjectPass`](crate::BakeInjectPass) inside the render graph.
/// `Arc`-wrapped fields allow `SsaoPass` to hold its own reference to the AO
/// texture without lifetime conflicts.
pub struct BakedData {
    // ── Ambient Occlusion ──────────────────────────────────────────────────────
    // R8Unorm texture matching the SSAO output format so it slots in directly.
    pub(crate) ao_texture: Option<wgpu::Texture>,
    pub(crate) ao_view: Option<Arc<wgpu::TextureView>>,
    pub(crate) ao_sampler: Option<Arc<wgpu::Sampler>>,

    // ── Lightmap ───────────────────────────────────────────────────────────────
    pub(crate) lightmap_texture: Option<wgpu::Texture>,
    pub(crate) lightmap_view: Option<Arc<wgpu::TextureView>>,
    pub(crate) lightmap_sampler: Option<Arc<wgpu::Sampler>>,
    /// Per-mesh UV atlas regions (for GBuffer shader lightmap lookup)
    pub(crate) lightmap_atlas_regions: Vec<crate::cache::CachedAtlasRegion>,
    /// Atlas pixel dimensions — needed to compute half-texel UV clamp bounds.
    pub(crate) lightmap_atlas_dims: Option<(u32, u32)>,

    // ── Reflection cubemap (pre-filtered specular IBL) ─────────────────────────
    // First probe only for now; multi-probe blending is future work.
    pub(crate) reflection_texture: Option<wgpu::Texture>,
    pub(crate) reflection_view: Option<Arc<wgpu::TextureView>>,
    pub(crate) reflection_sampler: Option<Arc<wgpu::Sampler>>,

    // ── Irradiance SH (diffuse IBL, GPU buffer of 9 RGB = 27 f32) ─────────────
    pub(crate) irradiance_sh_buf: Option<Arc<wgpu::Buffer>>,

    // ── PVS ───────────────────────────────────────────────────────────────────
    pub(crate) pvs: Option<BakedPvsData>,
}

impl BakedData {
    /// Returns the baked AO view (Arc clone — for SsaoPass to hold its own reference).
    pub fn ao_view(&self) -> Option<Arc<wgpu::TextureView>> {
        self.ao_view.clone()
    }

    /// Returns the baked AO sampler (Arc clone).
    pub fn ao_sampler(&self) -> Option<Arc<wgpu::Sampler>> {
        self.ao_sampler.clone()
    }

    /// Returns the baked irradiance SH buffer (Arc clone).
    pub fn irradiance_sh_buf(&self) -> Option<Arc<wgpu::Buffer>> {
        self.irradiance_sh_buf.clone()
    }

    /// Returns a reference to the CPU-side PVS data, if baked.
    pub fn pvs(&self) -> Option<&BakedPvsData> {
        self.pvs.as_ref()
    }

    // ── Zero-copy reference accessors for FrameResources population ────────────
    //
    // These borrow from the Arc contents (lifetime tied to &self).  The Renderer
    // stores `Arc<BakedData>` which lives for the entire render() call, so the
    // returned references are valid for the FrameResources<'_> lifetime.

    /// Borrowed AO texture view (zero-copy, no Arc clone).
    pub fn ao_view_ref(&self) -> Option<&wgpu::TextureView> {
        self.ao_view.as_deref()
    }

    /// Borrowed AO sampler (zero-copy).
    pub fn ao_sampler_ref(&self) -> Option<&wgpu::Sampler> {
        self.ao_sampler.as_deref()
    }

    /// Borrowed lightmap texture view (zero-copy).
    pub fn lightmap_view_ref(&self) -> Option<&wgpu::TextureView> {
        self.lightmap_view.as_deref()
    }

    /// Borrowed lightmap sampler (zero-copy).
    pub fn lightmap_sampler_ref(&self) -> Option<&wgpu::Sampler> {
        self.lightmap_sampler.as_deref()
    }

    /// Returns the lightmap atlas regions (UV offset/scale per-mesh).
    pub fn lightmap_atlas_regions(&self) -> &[crate::cache::CachedAtlasRegion] {
        &self.lightmap_atlas_regions
    }

    /// Convert lightmap atlas regions to GPU format:
    /// `[uv_offset.x, uv_offset.y, uv_scale.x, uv_scale.y,
    ///   uv_clamp_min.x, uv_clamp_min.y, uv_clamp_max.x, uv_clamp_max.y]`
    ///
    /// `uv_clamp_min/max` are half-texel-inset bounds that prevent bilinear filtering
    /// from sampling across adjacent atlas region boundaries (atlas bleed / light leak).
    /// They are precomputed here on the CPU so the vertex shader only needs a `clamp()`.
    pub fn lightmap_atlas_regions_gpu(&self) -> Vec<[f32; 8]> {
        let (atlas_w, atlas_h) = self.lightmap_atlas_dims.unwrap_or((1, 1));
        // Half-texel in normalised UV space.  Clamping by this amount ensures the
        // bilinear kernel never reaches beyond the region boundary.
        let half_u = 0.5 / atlas_w as f32;
        let half_v = 0.5 / atlas_h as f32;
        self.lightmap_atlas_regions.iter().map(|r| {
            [
                r.uv_offset[0], r.uv_offset[1],
                r.uv_scale[0],  r.uv_scale[1],
                r.uv_offset[0] + half_u,
                r.uv_offset[1] + half_v,
                r.uv_offset[0] + r.uv_scale[0] - half_u,
                r.uv_offset[1] + r.uv_scale[1] - half_v,
            ]
        }).collect()
    }

    /// Borrowed reflection cubemap view (zero-copy).
    pub fn reflection_view_ref(&self) -> Option<&wgpu::TextureView> {
        self.reflection_view.as_deref()
    }

    /// Borrowed reflection sampler (zero-copy).
    pub fn reflection_sampler_ref(&self) -> Option<&wgpu::Sampler> {
        self.reflection_sampler.as_deref()
    }

    /// Borrowed irradiance SH uniform buffer (zero-copy).
    pub fn irradiance_sh_buf_ref(&self) -> Option<&wgpu::Buffer> {
        self.irradiance_sh_buf.as_deref()
    }

    /// Constructs a zero-copy [`BakedPvsRef`] from the owned PVS data, if present.
    pub fn pvs_ref(&self) -> Option<libhelio::BakedPvsRef<'_>> {
        self.pvs.as_ref().map(|p| libhelio::BakedPvsRef {
            world_min: p.world_min,
            world_max: p.world_max,
            grid_dims: p.grid_dims,
            cell_size: p.cell_size,
            cell_count: p.cell_count,
            words_per_cell: p.words_per_cell,
            bits: &p.bits,
        })
    }

    // ── Internal: upload raw cache data to the Helio GPU device ───────────────

    pub(crate) fn upload_to_gpu(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ao: Option<CachedAo>,
        lm: Option<CachedLightmap>,
        probes: Option<CachedProbes>,
        pvs: Option<CachedPvs>,
    ) -> Result<Self, BakeError> {
        Ok(Self {
            ao_texture: None, // placeholder replaced below
            ao_view: None,
            ao_sampler: None,
            lightmap_texture: None,
            lightmap_view: None,
            lightmap_sampler: None,
            lightmap_atlas_regions: Vec::new(),
            lightmap_atlas_dims: None,
            reflection_texture: None,
            reflection_view: None,
            reflection_sampler: None,
            irradiance_sh_buf: None,
            pvs: None,
        }
        .with_ao(device, queue, ao)
        .with_lightmap(device, queue, lm)
        .with_probes(device, queue, probes)
        .with_pvs(pvs))
    }

    fn with_ao(mut self, device: &wgpu::Device, queue: &wgpu::Queue, ao: Option<CachedAo>) -> Self {
        let Some(ao) = ao else { return self };

        // Re-encode R32F AO texels → R8Unorm for SSAO slot compatibility.
        // Each f32 ∈ [0,1] → u8 ∈ [0, 255].
        let len = (ao.width * ao.height) as usize;
        let src: &[f32] = bytemuck::cast_slice(&ao.texels[..len * 4]);
        let r8: Vec<u8> = src.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Baked AO"),
            size: wgpu::Extent3d { width: ao.width, height: ao.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &r8,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(ao.width),
                rows_per_image: Some(ao.height),
            },
            wgpu::Extent3d { width: ao.width, height: ao.height, depth_or_array_layers: 1 },
        );
        let view = Arc::new(tex.create_view(&Default::default()));
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Baked AO Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
        log::info!("[helio-bake] Uploaded AO texture ({}×{}) to GPU.", ao.width, ao.height);
        self.ao_texture = Some(tex);
        self.ao_view = Some(view);
        self.ao_sampler = Some(sampler);
        self
    }

    fn with_lightmap(mut self, device: &wgpu::Device, queue: &wgpu::Queue, lm: Option<CachedLightmap>) -> Self {
        let Some(lm) = lm else { return self };

        // Always upload as Rgba16Float — universally filterable on all backends (Vulkan, DX12,
        // Metal, WebGPU) without requiring FLOAT32_FILTERABLE. Full HDR range up to ~65504
        // is more than sufficient for lightmap irradiance values in any practical scene.
        // This also halves VRAM usage vs Rgba32Float.
        let format = wgpu::TextureFormat::Rgba16Float;
        let bytes_per_row = lm.width * 8; // 4 channels × 2 bytes (f16)

        // If the bake cache stored f32 data, convert each channel to f16 before upload.
        // Otherwise the texels are already f16 bytes and can be used directly.
        let converted: Vec<u8>;
        let texels: &[u8] = if lm.is_f32 {
            let src: &[f32] = bytemuck::cast_slice(&lm.texels);
            let dst: Vec<u16> = src.iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect();
            converted = bytemuck::cast_slice::<u16, u8>(&dst).to_vec();
            &converted
        } else {
            &lm.texels
        };

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Baked Lightmap"),
            size: wgpu::Extent3d { width: lm.width, height: lm.height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            texels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(lm.height),
            },
            wgpu::Extent3d { width: lm.width, height: lm.height, depth_or_array_layers: 1 },
        );
        let view = Arc::new(tex.create_view(&Default::default()));
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Baked Lightmap Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }));
        log::info!("[helio-bake] Uploaded lightmap atlas ({}×{}, {:?}) to GPU.", lm.width, lm.height, format);
        self.lightmap_texture = Some(tex);
        self.lightmap_view = Some(view);
        self.lightmap_sampler = Some(sampler);
        self.lightmap_atlas_regions = lm.atlas_regions;
        self.lightmap_atlas_dims = Some((lm.width, lm.height));
        self
    }

    fn with_probes(mut self, device: &wgpu::Device, queue: &wgpu::Queue, probes: Option<CachedProbes>) -> Self {
        let Some(probes) = probes else { return self };
        if probes.positions.is_empty() { return self; }

        // Upload only the first probe for now (closest-probe selection is CPU-side in future work)
        let face_data = &probes.face_data[..probes.bytes_per_probe.min(probes.face_data.len())];

        let res = probes.face_resolution;
        let bytes_per_face = probes.bytes_per_probe / (6 * probes.mip_levels as usize).max(1);
        let format = if probes.is_rgbe {
            wgpu::TextureFormat::Rgba8Unorm
        } else {
            wgpu::TextureFormat::Rgba32Float
        };
        let bytes_per_pixel = if probes.is_rgbe { 4u32 } else { 16u32 };

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Baked Reflection Cubemap"),
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: 6 },
            mip_level_count: probes.mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload each mip × face
        let mut offset = 0usize;
        for mip in 0..probes.mip_levels {
            let mip_res = (res >> mip).max(1);
            let face_bytes = (mip_res * mip_res * bytes_per_pixel) as usize;
            for face in 0..6u32 {
                let end = (offset + face_bytes).min(face_data.len());
                if end <= offset { break; }
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: mip,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: face },
                        aspect: wgpu::TextureAspect::All,
                    },
                    &face_data[offset..end],
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(mip_res * bytes_per_pixel),
                        rows_per_image: Some(mip_res),
                    },
                    wgpu::Extent3d { width: mip_res, height: mip_res, depth_or_array_layers: 1 },
                );
                offset = end;
            }
        }

        let view = Arc::new(tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Baked Reflection View"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        }));
        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Baked Reflection Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        }));

        // Upload first probe's SH to a GPU buffer (9 RGB coefficients = 27 × f32 = 108 bytes)
        let sh_data: Vec<f32> = probes.irradiance_sh.first()
            .map(|sh| sh.iter().flat_map(|&[r, g, b]| [r, g, b]).collect())
            .unwrap_or_else(|| vec![0.0f32; 27]);
        let sh_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Baked Irradiance SH"),
            size: (sh_data.len() * 4) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        queue.write_buffer(&sh_buf, 0, bytemuck::cast_slice(&sh_data));

        log::info!(
            "[helio-bake] Uploaded reflection probe ({}×{}, {} mips) and {} SH coefficients to GPU.",
            res, res, probes.mip_levels, sh_data.len() / 3
        );
        self.reflection_texture = Some(tex);
        self.reflection_view = Some(view);
        self.reflection_sampler = Some(sampler);
        self.irradiance_sh_buf = Some(sh_buf);
        self
    }

    fn with_pvs(mut self, pvs: Option<CachedPvs>) -> Self {
        let Some(pvs) = pvs else { return self };
        log::info!(
            "[helio-bake] Loaded PVS ({} cells, {:.1}m cell size).",
            pvs.cell_count, pvs.cell_size
        );
        self.pvs = Some(BakedPvsData {
            world_min: pvs.world_min,
            world_max: pvs.world_max,
            grid_dims: pvs.grid_dims,
            cell_size: pvs.cell_size,
            cell_count: pvs.cell_count,
            words_per_cell: pvs.words_per_cell,
            bits: pvs.bits,
        });
        self
    }
}
