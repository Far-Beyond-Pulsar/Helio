//! PBR material types for geometry draw calls.
//!
//! # Texture conventions
//! - `base_color_texture` – sRGB RGBA albedo/alpha
//! - `normal_map`         – linear RGBA, tangent-space normals (packed 0..1 → -1..1)
//! - `orm_texture`        – linear RGBA; R = occlusion, G = roughness, B = metallic
//! - `emissive_texture`   – sRGB RGBA emissive color (multiplied by `emissive_color × emissive_factor`)

use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Raw CPU-side image data for uploading a texture to the GPU.
pub struct TextureData {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

impl TextureData {
    /// Create from raw RGBA bytes (8 bits per channel).
    pub fn new(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self { data, width, height }
    }
}

/// CPU-side description of a PBR material.
///
/// All texture fields are optional – omitting one falls back to a sensible
/// 1×1 default (white albedo, flat normal, no occlusion/roughness/metallic
/// override, no emission).
#[derive(Default)]
pub struct Material {
    // ── Scalar / tint factors ────────────────────────────────────────────────
    /// Base color multiplier (RGBA, linear).  Default white.
    pub base_color: [f32; 4],
    /// Metallic factor in [0, 1].  Default 0.
    pub metallic: f32,
    /// Roughness factor in [0, 1].  Default 0.5.
    pub roughness: f32,
    /// Ambient occlusion factor in [0, 1].  Default 1.
    pub ao: f32,
    /// Emissive color tint (linear RGB).  Combined with `emissive_factor` and
    /// the emissive texture.  Default black (no emission).
    pub emissive_color: [f32; 3],
    /// Emissive strength multiplier.  Default 0 (no emission).
    pub emissive_factor: f32,
    /// Alpha cutout threshold in [0, 1].
    ///
    /// Pixels with computed alpha below this value are discarded in the
    /// G-buffer/geometry shader paths. Use this for foliage, fences, decals,
    /// and other texture-mask transparency in deferred rendering.
    pub alpha_cutoff: f32,
    /// Force forward alpha blending path.
    ///
    /// When false, the material may still be auto-routed to transparent pass
    /// if the base-color texture contains alpha < 1 and `alpha_cutoff == 0`.
    pub transparent_blend: bool,

    // ── Optional textures ────────────────────────────────────────────────────
    /// sRGB RGBA albedo texture.
    pub base_color_texture: Option<TextureData>,
    /// Linear RGBA tangent-space normal map.
    pub normal_map: Option<TextureData>,
    /// Linear RGBA ORM texture (R=occlusion, G=roughness, B=metallic).
    pub orm_texture: Option<TextureData>,
    /// sRGB RGBA emissive texture.
    pub emissive_texture: Option<TextureData>,
}

impl Material {
    pub fn new() -> Self {
        Self {
            base_color: [1.0; 4],
            metallic: 0.0,
            roughness: 0.5,
            ao: 1.0,
            emissive_color: [0.0; 3],
            emissive_factor: 0.0,
            alpha_cutoff: 0.0,
            transparent_blend: false,
            ..Default::default()
        }
    }

    pub fn with_base_color(mut self, color: [f32; 4]) -> Self { self.base_color = color; self }
    pub fn with_metallic(mut self, v: f32) -> Self { self.metallic = v; self }
    pub fn with_roughness(mut self, v: f32) -> Self { self.roughness = v; self }
    pub fn with_ao(mut self, v: f32) -> Self { self.ao = v; self }
    pub fn with_emissive(mut self, color: [f32; 3], factor: f32) -> Self {
        self.emissive_color = color;
        self.emissive_factor = factor;
        self
    }
    pub fn with_alpha_cutoff(mut self, cutoff: f32) -> Self {
        self.alpha_cutoff = cutoff.clamp(0.0, 1.0);
        self
    }
    /// Force this material to use forward alpha blending.
    pub fn with_alpha_blend(mut self) -> Self {
        self.transparent_blend = true;
        self
    }
    pub fn with_base_color_texture(mut self, t: TextureData) -> Self { self.base_color_texture = Some(t); self }
    pub fn with_normal_map(mut self, t: TextureData) -> Self { self.normal_map = Some(t); self }
    pub fn with_orm_texture(mut self, t: TextureData) -> Self { self.orm_texture = Some(t); self }
    pub fn with_emissive_texture(mut self, t: TextureData) -> Self { self.emissive_texture = Some(t); self }
}

/// GPU-resident material ready to be used in draw calls.
#[derive(Clone)]
pub struct GpuMaterial {
    pub bind_group: Arc<wgpu::BindGroup>,
    pub transparent_blend: bool,
}

impl GpuMaterial {
    /// Build from an already-created wgpu bind group (advanced use).
    pub fn from_bind_group(bg: Arc<wgpu::BindGroup>) -> Self {
        Self { bind_group: bg, transparent_blend: false }
    }
}

// ── Internal uniform struct (must match geometry.wgsl `Material`) ────────────

/// Must match the WGSL `Material` struct exactly (48 bytes).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct MaterialUniform {
    pub base_color: [f32; 4],       // offset  0, 16 bytes
    pub metallic: f32,               // offset 16
    pub roughness: f32,              // offset 20
    pub emissive_factor: f32,        // offset 24
    pub ao: f32,                     // offset 28
    pub emissive_color: [f32; 3],   // offset 32, 12 bytes
    pub alpha_cutoff: f32,           // offset 44
    // total: 48 bytes (multiple of 16 — satisfies vec3 alignment)
}

impl From<&Material> for MaterialUniform {
    fn from(m: &Material) -> Self {
        Self {
            base_color: m.base_color,
            metallic: m.metallic,
            roughness: m.roughness,
            emissive_factor: m.emissive_factor,
            ao: m.ao,
            emissive_color: m.emissive_color,
            alpha_cutoff: m.alpha_cutoff,
        }
    }
}

// ── Helper: upload a 2D texture ───────────────────────────────────────────────

fn upload_texture_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> wgpu::TextureView {
    let tex = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        data,
    );
    tex.create_view(&wgpu::TextureViewDescriptor::default())
}

// ── Public factory (called from Renderer) ────────────────────────────────────

/// Upload a [`Material`] to the GPU and return a [`GpuMaterial`].
///
/// `default_views` should be the renderer's 1×1 fallback views in order:
/// `[white_srgb, flat_normal, white_orm, black_emissive]`.
pub(crate) fn build_gpu_material(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
    mat: &Material,
    defaults: &DefaultMaterialViews,
) -> GpuMaterial {
        let has_texture_alpha = mat.base_color_texture.as_ref().map(|t| {
            t.data.chunks_exact(4).any(|px| px[3] < 255)
        }).unwrap_or(false);
        let transparent_blend = mat.transparent_blend
            || (has_texture_alpha && mat.alpha_cutoff <= 0.0);

    let uniform: MaterialUniform = mat.into();
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Material Uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage: wgpu::BufferUsages::UNIFORM,
    });

    // Upload optional textures or fall back to shared defaults.
    let base_view = mat.base_color_texture.as_ref().map(|t| {
        upload_texture_2d(device, queue, &t.data, t.width, t.height,
            wgpu::TextureFormat::Rgba8UnormSrgb, "Material BaseColor")
    });
    let normal_view = mat.normal_map.as_ref().map(|t| {
        upload_texture_2d(device, queue, &t.data, t.width, t.height,
            wgpu::TextureFormat::Rgba8Unorm, "Material NormalMap")
    });
    let orm_view = mat.orm_texture.as_ref().map(|t| {
        upload_texture_2d(device, queue, &t.data, t.width, t.height,
            wgpu::TextureFormat::Rgba8Unorm, "Material ORM")
    });
    let emissive_view = mat.emissive_texture.as_ref().map(|t| {
        upload_texture_2d(device, queue, &t.data, t.width, t.height,
            wgpu::TextureFormat::Rgba8UnormSrgb, "Material Emissive")
    });

    let bv  = base_view    .as_ref().unwrap_or(&defaults.white_srgb);
    let nv  = normal_view  .as_ref().unwrap_or(&defaults.flat_normal);
    let ov  = orm_view     .as_ref().unwrap_or(&defaults.white_orm);
    let ev  = emissive_view.as_ref().unwrap_or(&defaults.black_emissive);

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Material Sampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        ..Default::default()
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Material Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(bv) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(nv) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(ov) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(ev) },
        ],
    });

    GpuMaterial {
        bind_group: Arc::new(bind_group),
        transparent_blend,
    }
}

/// Default 1×1 fallback texture views kept alive by the Renderer.
pub(crate) struct DefaultMaterialViews {
    pub white_srgb:    wgpu::TextureView,
    pub flat_normal:   wgpu::TextureView,
    pub white_orm:     wgpu::TextureView,   // R=1(AO), G=1(roughness), B=0(metallic) → treated as factors
    pub black_emissive: wgpu::TextureView,
    pub sampler:       wgpu::Sampler,
}

impl DefaultMaterialViews {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let white_srgb = upload_texture_2d(
            device, queue, &[255, 255, 255, 255], 1, 1,
            wgpu::TextureFormat::Rgba8UnormSrgb, "Default White sRGB");

        let flat_normal = upload_texture_2d(
            device, queue, &[128, 128, 255, 255], 1, 1,
            wgpu::TextureFormat::Rgba8Unorm, "Default Flat Normal");

        // ORM default: R=255 (AO=1), G=255 (roughness mult=1, uses material.roughness), B=0 (metallic mult=0)
        let white_orm = upload_texture_2d(
            device, queue, &[255, 255, 255, 255], 1, 1,
            wgpu::TextureFormat::Rgba8Unorm, "Default ORM");

        let black_emissive = upload_texture_2d(
            device, queue, &[0, 0, 0, 255], 1, 1,
            wgpu::TextureFormat::Rgba8UnormSrgb, "Default Black Emissive");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Default Material Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Self { white_srgb, flat_normal, white_orm, black_emissive, sampler }
    }
}
