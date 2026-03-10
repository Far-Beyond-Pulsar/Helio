//! Hi-Z (Hierarchical Z-Buffer) mipmap pyramid build pass.
//!
//! Runs two compute shaders each frame:
//!
//! 1. **Copy depth** — reads the existing `Depth32Float` texture from the depth
//!    prepass and writes its values into mip 0 of an `R32Float` Hi-Z pyramid.
//! 2. **Downsample loop** — for each subsequent mip level, reads the previous
//!    mip and writes the per-2×2-block maximum into the next mip.
//!
//! Storing the **maximum** depth per region is conservative: an object is only
//! culled by `OcclusionCullPass` when its nearest projected depth exceeds the
//! maximum depth stored in the Hi-Z region, guaranteeing no false culls.

use std::sync::Arc;
use crate::Result;

/// Maximum number of Hi-Z mip levels.
pub const HIZ_MAX_MIPS: u32 = 12; // enough for 4096² screens

/// GPU resources for the Hi-Z pyramid.
pub struct HiZResources {
    /// R32Float texture with mipmaps.  Mip 0 = copy of the depth buffer.
    pub texture: wgpu::Texture,
    /// Full-mip-chain view for sampling in the occlusion cull pass.
    pub full_view: Arc<wgpu::TextureView>,
    /// Per-mip views for compute storage writes (one per mip level).
    pub mip_views: Vec<wgpu::TextureView>,
    /// Per-mip views for sampled reads (used as source in the downsample bind groups).
    pub src_mip_views: Vec<wgpu::TextureView>,
    /// Sampler (nearest, no comparison) for Hi-Z reads.
    pub sampler: Arc<wgpu::Sampler>,
    /// Actual number of mip levels used.
    pub mip_count: u32,
    pub width:  u32,
    pub height: u32,
}

impl HiZResources {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mip_count = (width.max(height) as f32).log2().floor() as u32 + 1;
        let mip_count = mip_count.min(HIZ_MAX_MIPS);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("Hi-Z Pyramid"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: mip_count,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::R32Float,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING
                           | wgpu::TextureUsages::STORAGE_BINDING
                           | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });

        let full_view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label:           Some("Hi-Z Full View"),
            base_mip_level:  0,
            mip_level_count: Some(mip_count),
            ..Default::default()
        }));

        let mip_views: Vec<wgpu::TextureView> = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label:           Some(&format!("Hi-Z Mip {mip} Storage")),
                    base_mip_level:  mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // Separate views for use as sampled textures (TEXTURE_BINDING).
        // Same mip slices as mip_views but distinct view objects so the borrow
        // checker is happy when we later construct bind groups.
        let src_mip_views: Vec<wgpu::TextureView> = (0..mip_count)
            .map(|mip| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label:           Some(&format!("Hi-Z Mip {mip} Sample")),
                    base_mip_level:  mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label:        Some("Hi-Z Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter:   wgpu::FilterMode::Nearest,
            min_filter:   wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }));

        Self { texture, full_view, mip_views, src_mip_views, sampler, mip_count, width, height }
    }
}

/// Pass that builds the Hi-Z mipmap pyramid from the depth prepass output.
///
/// Must execute **after** the depth prepass and **before** `OcclusionCullPass`.
pub struct HiZBuildPass {
    /// Pipeline for copying depth → Hi-Z mip 0.
    copy_depth_pipeline: Arc<wgpu::ComputePipeline>,
    /// Pipeline for each downsample step (mip N-1 → mip N).
    downsample_pipeline: Arc<wgpu::ComputePipeline>,

    /// Bind group layout for the copy-depth pass.
    copy_bgl: wgpu::BindGroupLayout,
    /// Bind group layout for the downsample pass.
    down_bgl: wgpu::BindGroupLayout,

    /// Hi-Z GPU resources (texture + views + sampler).
    pub hiz: HiZResources,

    /// Per-frame bind groups (one for copy, one per downsample step).
    copy_bg: Option<wgpu::BindGroup>,
    down_bgs: Vec<wgpu::BindGroup>,
}

impl HiZBuildPass {
    pub fn new(
        device: &wgpu::Device,
        depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        let hiz = HiZResources::new(device, width, height);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Hi-Z Build Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/hiz_build.wgsl").into(),
            ),
        });

        // BGL for copy_depth: (depth_texture, hiz_mip0_storage)
        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("HiZ Copy BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // BGL for downsample: (src_mip_texture, dst_mip_storage)
        let down_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("HiZ Downsample BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Pipeline layouts
        let copy_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:               Some("HiZ Copy Layout"),
            bind_group_layouts:  &[Some(&copy_bgl)],
            immediate_size:      0,
        });
        let down_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:               Some("HiZ Down Layout"),
            bind_group_layouts:  &[Some(&down_bgl)],
            immediate_size:      0,
        });

        let copy_depth_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label:               Some("HiZ Copy Depth Pipeline"),
                layout:              Some(&copy_layout),
                module:              &shader,
                entry_point:         Some("copy_depth"),
                compilation_options: Default::default(),
                cache:               None,
            },
        ));

        let downsample_pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label:               Some("HiZ Downsample Pipeline"),
                layout:              Some(&down_layout),
                module:              &shader,
                entry_point:         Some("downsample"),
                compilation_options: Default::default(),
                cache:               None,
            },
        ));

        let mut pass = Self {
            copy_depth_pipeline,
            downsample_pipeline,
            copy_bgl,
            down_bgl,
            hiz,
            copy_bg: None,
            down_bgs: Vec::new(),
        };

        pass.rebuild_bind_groups(device, depth_view);
        Ok(pass)
    }

    /// Rebuild bind groups after a resize (new depth_view / new Hi-Z texture).
    pub fn rebuild_bind_groups(&mut self, device: &wgpu::Device, depth_view: &wgpu::TextureView) {
        // Copy bind group: (depth_texture, hiz_mip0)
        self.copy_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("HiZ Copy BG"),
            layout:  &self.copy_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::TextureView(&self.hiz.mip_views[0]),
                },
            ],
        }));

        // Downsample bind groups: one per pair (mip N-1 → mip N).
        // Use src_mip_views[mip-1] for the sampled read and mip_views[mip] for the storage write.
        self.down_bgs.clear();
        for mip in 1..self.hiz.mip_count {
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:   Some(&format!("HiZ Down BG mip {mip}")),
                layout:  &self.down_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: wgpu::BindingResource::TextureView(
                            &self.hiz.src_mip_views[(mip - 1) as usize],
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding:  1,
                        resource: wgpu::BindingResource::TextureView(
                            &self.hiz.mip_views[mip as usize],
                        ),
                    },
                ],
            });
            self.down_bgs.push(bg);
        }
    }

    /// Execute the Hi-Z build for a single frame.
    pub fn execute_hiz_build(&self, encoder: &mut wgpu::CommandEncoder) {
        let Some(copy_bg) = &self.copy_bg else { return; };

        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:             Some("Hi-Z Build"),
            timestamp_writes:  None,
        });

        // ── Step 1: copy depth → mip 0 ─────────────────────────────────────
        let (w0, h0) = (self.hiz.width, self.hiz.height);
        cp.set_pipeline(&self.copy_depth_pipeline);
        cp.set_bind_group(0, copy_bg, &[]);
        cp.dispatch_workgroups((w0 + 7) / 8, (h0 + 7) / 8, 1);

        // ── Step 2: downsample each mip level ──────────────────────────────
        cp.set_pipeline(&self.downsample_pipeline);
        for (i, bg) in self.down_bgs.iter().enumerate() {
            let mip = (i + 1) as u32;
            let mw = (self.hiz.width  >> mip).max(1);
            let mh = (self.hiz.height >> mip).max(1);
            cp.set_bind_group(0, bg, &[]);
            cp.dispatch_workgroups((mw + 7) / 8, (mh + 7) / 8, 1);
        }
    }
}
