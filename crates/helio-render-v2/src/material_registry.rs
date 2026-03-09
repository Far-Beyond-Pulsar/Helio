//! Content-addressed `GpuMaterial` cache.
//!
//! `MaterialRegistry::get_or_create` hashes the full material descriptor
//! (scalar PBR params + per-slot texture content).  On a hit the existing
//! `GpuMaterial` is returned as an O(1) `Clone` with no GPU work at all.
//! On a miss the textures are uploaded via [`TextureRegistry`] (which itself
//! deduplicates identical pixel data) and the bind group is created once.
//!
//! Matches Unreal's `FMaterialRenderProxy` — a persistent render-thread object
//! that is never rebuilt unless the material parameters actually change.

use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

use crate::material::{Material, GpuMaterial, MaterialUniform, DefaultMaterialViews};
use crate::texture_registry::{TextureRegistry, fnv1a};

// ── Material content hash ─────────────────────────────────────────────────────

fn material_hash(mat: &Material) -> u64 {
    let uniform = MaterialUniform::from(mat);
    // Mix POD scalar params.
    let mut h = fnv1a(bytemuck::bytes_of(&uniform));
    h ^= mat.transparent_blend as u64;

    // Mix per-slot texture fingerprints (or sentinel values for absent slots).
    let slot_hashes = [
        mat.base_color_texture.as_ref().map(|t| {
            TextureRegistry::content_hash(&t.data, t.width, t.height,
                wgpu::TextureFormat::Rgba8UnormSrgb)
        }).unwrap_or(0xdead_0001_dead_0001u64),

        mat.normal_map.as_ref().map(|t| {
            TextureRegistry::content_hash(&t.data, t.width, t.height,
                wgpu::TextureFormat::Rgba8Unorm)
        }).unwrap_or(0xdead_0002_dead_0002u64),

        mat.orm_texture.as_ref().map(|t| {
            TextureRegistry::content_hash(&t.data, t.width, t.height,
                wgpu::TextureFormat::Rgba8Unorm)
        }).unwrap_or(0xdead_0003_dead_0003u64),

        mat.emissive_texture.as_ref().map(|t| {
            TextureRegistry::content_hash(&t.data, t.width, t.height,
                wgpu::TextureFormat::Rgba8UnormSrgb)
        }).unwrap_or(0xdead_0004_dead_0004u64),
    ];

    for (i, &th) in slot_hashes.iter().enumerate() {
        h = h.wrapping_mul(0x517cc1b727220a95)
             .wrapping_add(th)
             .wrapping_add(i as u64 + 1);
    }
    h
}

// ── MaterialRegistry ──────────────────────────────────────────────────────────

/// Persistent material + texture cache owned by the `Renderer`.
///
/// There is no eviction — materials live as long as the `Renderer`.  For a
/// streaming engine where materials can be unloaded, add `release(GpuMaterial)`
/// that removes the entry only when the `Arc` refcount reaches 1 (i.e. the
/// registry holds the last reference).
pub(crate) struct MaterialRegistry {
    /// Shared texture + sampler pool.
    pub texture_registry: TextureRegistry,
    /// content hash → cached GpuMaterial (Arc-shared bind group).
    materials: HashMap<u64, GpuMaterial>,
}

impl MaterialRegistry {
    pub fn new() -> Self {
        Self {
            texture_registry: TextureRegistry::new(),
            materials:        HashMap::new(),
        }
    }

    /// Return a cached `GpuMaterial` (O(1) on hit) or build and cache one
    /// (uploading only textures not already in `texture_registry`).
    pub fn get_or_create(
        &mut self,
        device:   &wgpu::Device,
        queue:    &wgpu::Queue,
        layout:   &wgpu::BindGroupLayout,
        mat:      &Material,
        defaults: &DefaultMaterialViews,
    ) -> GpuMaterial {
        let hash = material_hash(mat);
        if let Some(cached) = self.materials.get(&hash) {
            return cached.clone();   // Arc bump — zero GPU work
        }
        let gpu_mat = build_registered(
            device, queue, layout, mat, defaults, &mut self.texture_registry,
        );
        self.materials.insert(hash, gpu_mat.clone());
        gpu_mat
    }

    /// Number of unique materials currently cached.
    pub fn material_count(&self) -> usize { self.materials.len() }
}

// ── Registry-aware builder ────────────────────────────────────────────────────

/// Internal factory called exactly once per unique material.
/// Textures are routed through `TextureRegistry` so duplicate pixel data
/// shares a single GPU upload across all materials.
fn build_registered(
    device:   &wgpu::Device,
    queue:    &wgpu::Queue,
    layout:   &wgpu::BindGroupLayout,
    mat:      &Material,
    defaults: &DefaultMaterialViews,
    reg:      &mut TextureRegistry,
) -> GpuMaterial {
    // Detect texture-sourced alpha for automatic transparent routing.
    let has_tex_alpha = mat.base_color_texture.as_ref()
        .map(|t| t.data.chunks_exact(4).any(|px| px[3] < 255))
        .unwrap_or(false);
    let transparent_blend = mat.transparent_blend
        || (has_tex_alpha && mat.alpha_cutoff <= 0.0);

    // Per-material uniform buffer (48 bytes — tiny fixed cost, not amortised).
    let uniform: MaterialUniform = mat.into();
    let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label:    Some("Material Uniform"),
        contents: bytemuck::bytes_of(&uniform),
        usage:    wgpu::BufferUsages::UNIFORM,
    });

    // Texture slots — all routed through the registry.
    // Identical pixel data across different materials shares one GPU upload.
    let base_view = mat.base_color_texture.as_ref().map(|t| {
        reg.get_or_upload(device, queue, &t.data, t.width, t.height,
            wgpu::TextureFormat::Rgba8UnormSrgb)
    });
    let normal_view = mat.normal_map.as_ref().map(|t| {
        reg.get_or_upload(device, queue, &t.data, t.width, t.height,
            wgpu::TextureFormat::Rgba8Unorm)
    });
    let orm_view = mat.orm_texture.as_ref().map(|t| {
        reg.get_or_upload(device, queue, &t.data, t.width, t.height,
            wgpu::TextureFormat::Rgba8Unorm)
    });
    let emissive_view = mat.emissive_texture.as_ref().map(|t| {
        reg.get_or_upload(device, queue, &t.data, t.width, t.height,
            wgpu::TextureFormat::Rgba8UnormSrgb)
    });

    // Shared linear-repeat sampler — after the first call all PBR materials
    // reuse the same wgpu::Sampler (one GPU object instead of one per material).
    let sampler = reg.get_or_create_sampler(
        device,
        wgpu::FilterMode::Linear,
        wgpu::FilterMode::Linear,
        wgpu::MipmapFilterMode::Linear,
        wgpu::AddressMode::Repeat,
        wgpu::AddressMode::Repeat,
    );

    let bv = base_view    .as_deref().unwrap_or(&defaults.white_srgb);
    let nv = normal_view  .as_deref().unwrap_or(&defaults.flat_normal);
    let ov = orm_view     .as_deref().unwrap_or(&defaults.white_orm);
    let ev = emissive_view.as_deref().unwrap_or(&defaults.black_emissive);

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label:  Some("Material Bind Group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(bv) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(nv) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&*sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(ov) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(ev) },
        ],
    });

    GpuMaterial { bind_group: Arc::new(bind_group), transparent_blend }
}
