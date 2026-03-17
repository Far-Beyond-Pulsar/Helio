//! PBR material types for geometry draw calls.
//!
//! # Texture conventions
//! - `base_color_texture` – sRGB RGBA albedo/alpha
//! - `normal_map`         – linear RGBA, tangent-space normals (packed 0..1 → -1..1)
//! - `orm_texture`        – linear RGBA; R = occlusion, G = roughness, B = metallic
//! - `emissive_texture`   – sRGB RGBA emissive color (multiplied by `emissive_color × emissive_factor`)
//! - `specular_color_texture`  – sRGB RGBA; RGB = explicit F0 tint, A unused
//! - `specular_weight_texture` – linear RGBA; A = explicit specular weight, RGB unused

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

/// Canonical CPU-side material workflow selection.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterialWorkflowKind {
    MetallicRoughness = 0,
    SpecularIor = 1,
}

/// Canonical metallic/roughness material workflow parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MetallicRoughnessWorkflow {
    pub metallic: f32,
    pub roughness: f32,
}

impl Default for MetallicRoughnessWorkflow {
    fn default() -> Self {
        Self { metallic: 0.0, roughness: 0.5 }
    }
}

/// Canonical explicit specular/IOR workflow parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecularIorWorkflow {
    /// Linear RGB F0 tint used by explicit specular workflows.
    pub specular_color: [f32; 3],
    /// Scalar multiplier for the specular term.
    pub specular_weight: f32,
    /// Index of refraction used to derive the dielectric Fresnel baseline.
    pub ior: f32,
    /// Perceptual roughness in [0, 1].
    pub roughness: f32,
}

impl SpecularIorWorkflow {
    pub fn dielectric_f0(self) -> f32 {
        let ior = self.ior.max(1.0);
        let f0 = (ior - 1.0) / (ior + 1.0);
        f0 * f0
    }
}

impl Default for SpecularIorWorkflow {
    fn default() -> Self {
        Self {
            specular_color: [1.0; 3],
            specular_weight: 1.0,
            ior: 1.5,
            roughness: 0.5,
        }
    }
}

/// Canonical CPU-side material workflow representation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaterialWorkflow {
    MetallicRoughness(MetallicRoughnessWorkflow),
    SpecularIor(SpecularIorWorkflow),
}

impl Default for MaterialWorkflow {
    fn default() -> Self {
        Self::MetallicRoughness(MetallicRoughnessWorkflow::default())
    }
}

impl MaterialWorkflow {
    pub fn kind(self) -> MaterialWorkflowKind {
        match self {
            Self::MetallicRoughness(_) => MaterialWorkflowKind::MetallicRoughness,
            Self::SpecularIor(_) => MaterialWorkflowKind::SpecularIor,
        }
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
    /// Canonical material workflow selection.
    ///
    /// Legacy `metallic`/`roughness` fields remain available for existing
    /// metallic-roughness call sites and are mirrored when that workflow is
    /// active.
    pub workflow: MaterialWorkflow,
    /// Metallic factor in [0, 1].  Default 0.
    ///
    /// This is preserved for backward compatibility with the existing
    /// metallic/roughness workflow API.
    pub metallic: f32,
    /// Roughness factor in [0, 1].  Default 0.5.
    ///
    /// This remains the compatibility surface for existing code and is also
    /// mirrored into explicit workflows when those are authored through the
    /// provided helpers.
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
    /// sRGB RGBA explicit specular-colour texture. RGB modulates F0 tint.
    pub specular_color_texture: Option<TextureData>,
    /// Linear RGBA explicit specular-weight texture. Alpha modulates weight.
    pub specular_weight_texture: Option<TextureData>,
}

impl Material {
    pub fn new() -> Self {
        Self {
            base_color: [1.0; 4],
            workflow: MaterialWorkflow::default(),
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
    pub fn with_metallic(mut self, v: f32) -> Self {
        self.metallic = v;
        self.workflow = MaterialWorkflow::MetallicRoughness(MetallicRoughnessWorkflow {
            metallic: self.metallic,
            roughness: self.roughness,
        });
        self
    }
    pub fn with_roughness(mut self, v: f32) -> Self {
        self.roughness = v;
        self.workflow = match self.workflow {
            MaterialWorkflow::MetallicRoughness(_) => {
                MaterialWorkflow::MetallicRoughness(MetallicRoughnessWorkflow {
                    metallic: self.metallic,
                    roughness: self.roughness,
                })
            }
            MaterialWorkflow::SpecularIor(specular) => {
                MaterialWorkflow::SpecularIor(SpecularIorWorkflow {
                    roughness: self.roughness,
                    ..specular
                })
            }
        };
        self
    }
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
    pub fn with_workflow(mut self, workflow: MaterialWorkflow) -> Self {
        self.set_workflow(workflow);
        self
    }
    pub fn with_metallic_roughness_workflow(mut self, metallic: f32, roughness: f32) -> Self {
        self.metallic = metallic;
        self.roughness = roughness;
        self.workflow = MaterialWorkflow::MetallicRoughness(MetallicRoughnessWorkflow {
            metallic,
            roughness,
        });
        self
    }
    pub fn with_specular_ior_workflow(
        mut self,
        specular_color: [f32; 3],
        specular_weight: f32,
        ior: f32,
        roughness: f32,
    ) -> Self {
        self.roughness = roughness;
        self.workflow = MaterialWorkflow::SpecularIor(SpecularIorWorkflow {
            specular_color,
            specular_weight,
            ior,
            roughness,
        });
        self
    }
    pub fn set_workflow(&mut self, workflow: MaterialWorkflow) {
        match workflow {
            MaterialWorkflow::MetallicRoughness(workflow) => {
                self.metallic = workflow.metallic;
                self.roughness = workflow.roughness;
                self.workflow = MaterialWorkflow::MetallicRoughness(workflow);
            }
            MaterialWorkflow::SpecularIor(workflow) => {
                self.roughness = workflow.roughness;
                self.workflow = MaterialWorkflow::SpecularIor(workflow);
            }
        }
    }
    pub fn workflow(&self) -> MaterialWorkflow {
        match self.workflow {
            MaterialWorkflow::MetallicRoughness(_) => {
                MaterialWorkflow::MetallicRoughness(MetallicRoughnessWorkflow {
                    metallic: self.metallic,
                    roughness: self.roughness,
                })
            }
            MaterialWorkflow::SpecularIor(workflow) => {
                MaterialWorkflow::SpecularIor(SpecularIorWorkflow {
                    roughness: self.roughness,
                    ..workflow
                })
            }
        }
    }
    pub fn workflow_kind(&self) -> MaterialWorkflowKind {
        self.workflow().kind()
    }
    pub fn with_base_color_texture(mut self, t: TextureData) -> Self { self.base_color_texture = Some(t); self }
    pub fn with_normal_map(mut self, t: TextureData) -> Self { self.normal_map = Some(t); self }
    pub fn with_orm_texture(mut self, t: TextureData) -> Self { self.orm_texture = Some(t); self }
    pub fn with_emissive_texture(mut self, t: TextureData) -> Self { self.emissive_texture = Some(t); self }
    pub fn with_specular_color_texture(mut self, t: TextureData) -> Self { self.specular_color_texture = Some(t); self }
    pub fn with_specular_weight_texture(mut self, t: TextureData) -> Self { self.specular_weight_texture = Some(t); self }
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

/// Must match the WGSL `Material` struct exactly (96 bytes).
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
    pub workflow: u32,              // offset 48
    pub workflow_flags: u32,        // offset 52
    pub _pad0: [u32; 2],            // offset 56, 8 bytes
    pub specular_color: [f32; 3],   // offset 64, 12 bytes
    pub specular_weight: f32,       // offset 76
    pub ior: f32,                   // offset 80
    pub dielectric_f0: f32,         // offset 84
    pub _reserved: [f32; 2],        // offset 88, 8 bytes
    // total: 96 bytes (multiple of 16 — satisfies uniform alignment)
}

impl From<&Material> for MaterialUniform {
    fn from(m: &Material) -> Self {
        let workflow = m.workflow();
        let (metallic, roughness, workflow_kind, specular_color, specular_weight, ior, dielectric_f0) =
            match workflow {
                MaterialWorkflow::MetallicRoughness(workflow) => {
                    let specular = SpecularIorWorkflow::default();
                    (
                        workflow.metallic,
                        workflow.roughness,
                        MaterialWorkflowKind::MetallicRoughness,
                        [specular.dielectric_f0(); 3],
                        specular.specular_weight,
                        specular.ior,
                        specular.dielectric_f0(),
                    )
                }
                MaterialWorkflow::SpecularIor(workflow) => (
                    0.0,
                    workflow.roughness,
                    MaterialWorkflowKind::SpecularIor,
                    workflow.specular_color,
                    workflow.specular_weight,
                    workflow.ior,
                    workflow.dielectric_f0(),
                ),
            };

        Self {
            base_color: m.base_color,
            metallic,
            roughness,
            emissive_factor: m.emissive_factor,
            ao: m.ao,
            emissive_color: m.emissive_color,
            alpha_cutoff: m.alpha_cutoff,
            workflow: workflow_kind as u32,
            workflow_flags: 0,
            _pad0: [0; 2],
            specular_color,
            specular_weight,
            ior,
            dielectric_f0,
            _reserved: [0.0; 2],
        }
    }
}

const fn specular_color_texture_format() -> wgpu::TextureFormat {
    wgpu::TextureFormat::Rgba8UnormSrgb
}

const fn specular_weight_texture_format() -> wgpu::TextureFormat {
    wgpu::TextureFormat::Rgba8Unorm
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
/// `[white_srgb, flat_normal, white_orm, black_emissive]` plus the shared
/// white defaults reused for explicit specular colour/weight slots.
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
    let specular_color_view = mat.specular_color_texture.as_ref().map(|t| {
        upload_texture_2d(device, queue, &t.data, t.width, t.height,
            specular_color_texture_format(), "Material Specular Color")
    });
    let specular_weight_view = mat.specular_weight_texture.as_ref().map(|t| {
        upload_texture_2d(device, queue, &t.data, t.width, t.height,
            specular_weight_texture_format(), "Material Specular Weight")
    });

    let bv  = base_view    .as_ref().unwrap_or(&defaults.white_srgb);
    let nv  = normal_view  .as_ref().unwrap_or(&defaults.flat_normal);
    let ov  = orm_view     .as_ref().unwrap_or(&defaults.white_orm);
    let ev  = emissive_view.as_ref().unwrap_or(&defaults.black_emissive);
    let scv = specular_color_view.as_ref().unwrap_or(&defaults.white_srgb);
    let swv = specular_weight_view.as_ref().unwrap_or(&defaults.white_orm);

    // Use ClampToEdge by default to avoid tiling artifacts on models with UVs at boundaries
    // This is more common for single-texture-per-primitive models like FBX exports
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Material Sampler"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
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
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(scv) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(swv) },
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
    pub white_orm:     wgpu::TextureView,   // R=1(AO), G=1(roughness), B=1(metallic mult) → material.metallic controls final value
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

        // ORM default: R=255 (AO=1), G=255 (roughness mult=1), B=255 (metallic mult=1, material.metallic=0 → final 0)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn material_uniform_size_matches_wgsl_layout() {
        assert_eq!(std::mem::size_of::<MaterialUniform>(), 96);
    }

    #[test]
    fn legacy_metallic_roughness_fields_still_drive_workflow() {
        let mut material = Material::new();
        material.metallic = 0.8;
        material.roughness = 0.3;

        assert_eq!(
            material.workflow(),
            MaterialWorkflow::MetallicRoughness(MetallicRoughnessWorkflow {
                metallic: 0.8,
                roughness: 0.3,
            })
        );

        let uniform = MaterialUniform::from(&material);
        assert_eq!(uniform.workflow, MaterialWorkflowKind::MetallicRoughness as u32);
        assert_eq!(uniform.metallic, 0.8);
        assert_eq!(uniform.roughness, 0.3);
        assert_eq!(uniform.ior, 1.5);
        assert!((uniform.dielectric_f0 - 0.04).abs() < 1e-6);
    }

    #[test]
    fn explicit_specular_ior_workflow_populates_uniform_fields() {
        let material = Material::new().with_specular_ior_workflow([0.9, 0.8, 0.7], 0.65, 1.33, 0.18);

        assert_eq!(material.workflow_kind(), MaterialWorkflowKind::SpecularIor);

        let uniform = MaterialUniform::from(&material);
        assert_eq!(uniform.workflow, MaterialWorkflowKind::SpecularIor as u32);
        assert_eq!(uniform.metallic, 0.0);
        assert_eq!(uniform.roughness, 0.18);
        assert_eq!(uniform.specular_color, [0.9, 0.8, 0.7]);
        assert_eq!(uniform.specular_weight, 0.65);
        assert!((uniform.ior - 1.33).abs() < 1e-6);
        assert!((uniform.dielectric_f0 - 0.020059314).abs() < 1e-6);
    }

    #[test]
    fn textured_specular_ior_workflow_populates_uniform_fields() {
        let material = Material::new()
            .with_specular_ior_workflow([0.76, 0.58, 0.42], 0.37, 1.61, 0.24)
            .with_specular_color_texture(TextureData::new(vec![194, 148, 107, 255], 1, 1))
            .with_specular_weight_texture(TextureData::new(vec![0, 0, 0, 94], 1, 1));

        assert_eq!(material.workflow_kind(), MaterialWorkflowKind::SpecularIor);
        assert!(material.specular_color_texture.is_some());
        assert!(material.specular_weight_texture.is_some());
        assert_eq!(
            material.workflow(),
            MaterialWorkflow::SpecularIor(SpecularIorWorkflow {
                specular_color: [0.76, 0.58, 0.42],
                specular_weight: 0.37,
                ior: 1.61,
                roughness: 0.24,
            })
        );

        let uniform = MaterialUniform::from(&material);
        let expected_f0 = ((1.61_f32 - 1.0_f32) / (1.61_f32 + 1.0_f32)).powi(2);
        assert_eq!(uniform.workflow, MaterialWorkflowKind::SpecularIor as u32);
        assert_eq!(uniform.metallic, 0.0);
        assert_eq!(uniform.roughness, 0.24);
        assert_eq!(uniform.specular_color, [0.76, 0.58, 0.42]);
        assert_eq!(uniform.specular_weight, 0.37);
        assert!((uniform.ior - 1.61).abs() < 1e-6);
        assert!((uniform.dielectric_f0 - expected_f0).abs() < 1e-6);
    }

    #[test]
    fn specular_texture_slots_use_expected_color_spaces() {
        assert_eq!(specular_color_texture_format(), wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(specular_weight_texture_format(), wgpu::TextureFormat::Rgba8Unorm);
    }
}
