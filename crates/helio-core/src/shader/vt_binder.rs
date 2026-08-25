//! Group-2 bind-group plumbing for every `//!use helio_vt` consumer
//! (Helio#238 §3/§4).
//!
//! The WGSL module OWNS its bindings (`@group(2)`: meta rows at 0, quarter-res
//! density target at 1), so every pass whose shaders opt in must bind exactly
//! this layout on every draw. This helper is the single implementation of
//! that:
//!
//! - one canonical [`wgpu::BindGroupLayout`] matching the module;
//! - lazy per-frame [`wgpu::BindGroup`] construction from the renderer's
//!   frame-transient meta buffer plus the graph-owned `vt_density` target
//!   (`VtDensity` resource);
//! - self-contained FALLBACKS (a one-row unmanaged meta table and a 1×1 zero
//!   density cell) so a pass stays bindable even in graphs that do not run
//!   the feedback pair — pipelines never fail validation merely because a
//!   host composed a custom graph without streaming.
//!
//! # Rebuild discipline (the part that must not regress)
//!
//! Bind groups are rebuilt ONLY when the inputs change: the frame's meta
//! buffer pointer, the density view pointer, or the material-texture version
//! advanced. That rides the same cadence as the §C material bind-group gate
//! in helio-pass-gbuffer — never rebuilt unconditionally, never rebuilt
//! mid-frame, and never read back from or blocked on GPU state during
//! recording (promote-before-bind: residency arrives as already-committed
//! rows inside the frame's meta buffer).

use std::sync::atomic::{AtomicU64, Ordering};
use wgpu::util::DeviceExt;

/// Label prefix for diagnostics.
const LABEL: &str = "VtGroup";

/// Canonical group-2 layout + fallback resources for one device.
pub struct VtGroupBinder {
    /// Total bind-group constructions ever issued through this binder — the
    /// churn counter BM3 measures (and any debug build can watch).
    rebuilds: AtomicU64,
    layout: wgpu::BindGroupLayout,
    /// One [`libhelio::GpuVtMetaRow::UNMANAGED`] row — used when the frame
    /// published no meta buffer (scene without streamed textures).
    fallback_meta: wgpu::Buffer,
    /// 1×1 zero cell — used when the graph declared no `VtDensity` target.
    fallback_density_view: wgpu::TextureView,
}

/// Identity of everything a built bind group captured. Cheap to compare;
/// drives the rebuild gate.
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct VtGroupKey {
    pub meta_ptr: usize,
    pub density_ptr: usize,
    pub version: u64,
}

impl VtGroupBinder {
    pub fn new(device: &wgpu::Device) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(&format!("{LABEL} BGL")),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let fallback_meta =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{LABEL} Fallback Meta")),
                contents: bytemuck::bytes_of(&libhelio::GpuVtMetaRow::UNMANAGED),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let fallback_density = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("{LABEL} Fallback Density")),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let fallback_density_view =
            fallback_density.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            rebuilds: AtomicU64::new(0),
            layout,
            fallback_meta,
            fallback_density_view,
        }
    }

    /// How many bind groups this binder has built (BM3's churn metric).
    pub fn rebuild_count(&self) -> u64 {
        self.rebuilds.load(Ordering::Relaxed)
    }

    /// The layout every VT-consuming pipeline must declare for group 2.
    pub fn layout(&self) -> &wgpu::BindGroupLayout {
        &self.layout
    }

    /// Builds (or falls back for) one frame's group-2 bind group.
    pub fn bind_group(
        &self,
        device: &wgpu::Device,
        meta: Option<&wgpu::Buffer>,
        density: Option<&wgpu::TextureView>,
    ) -> wgpu::BindGroup {
        let meta_binding = meta.unwrap_or(&self.fallback_meta);
        let density_binding = density.unwrap_or(&self.fallback_density_view);
        self.rebuilds.fetch_add(1, Ordering::Relaxed);
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{LABEL} BG")),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: meta_binding.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(density_binding),
                },
            ],
        })
    }
}
