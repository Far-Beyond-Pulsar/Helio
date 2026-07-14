//! Hi-Z (Hierarchical Z) pyramid builder.
//!
//! Two-phase build each frame — fully GPU-driven, O(1) CPU:
//!
//!  Phase 1 — Depth copy  (hiz_depth_copy.wgsl, one dispatch)
//!    Reads the `Depth32Float` render-attachment texture written by DepthPrepassPass
//!    and writes each depth value into mip-0 of the R32Float HiZ texture.
//!    This is necessary because Depth32Float cannot be bound as a storage texture.
//!
//!  Phase 2 — Mip chain  (hiz_build.wgsl, ~log2(max_dim) dispatches)
//!    Downsamples using MAX-reduction so each texel stores the farthest depth
//!    in its 2x2 footprint — "conservative Hi-Z".
//!
//! The finished pyramid is consumed NEXT FRAME by OcclusionCullPass (temporal
//! approach: frame N-1 depth tests visibility of frame N geometry).
//!
//! The HiZ texture is owned by the render graph and declared via `declare_resources`.
//! The pass recreates mip views and bind groups lazily during `execute()` from the
//! graph-owned texture accessed via `ctx.resource_pool`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_v3::graph::{ResourceBuilder, ResourceSize};
use helio_v3::{FrameResources, PassContext, PrepareContext, RenderPass, Result as HelioResult};
use wgpu::util::DeviceExt;

const WORKGROUP_SIZE: u32 = 8;
const MAX_MIP_LEVELS: u32 = 12;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HiZUniforms {
    src_size: [u32; 2],
    dst_size: [u32; 2],
}

pub struct HiZBuildPass {
    // Mip-chain downsampling pipeline
    mip_pipeline: wgpu::ComputePipeline,
    mip_bgl: wgpu::BindGroupLayout,
    mip_bind_groups: Vec<wgpu::BindGroup>,
    mip_uniforms: Vec<wgpu::Buffer>,
    mip_dispatch_groups: Vec<(u32, u32)>,

    // Depth-copy pipeline (Depth32Float -> R32Float mip-0)
    copy_pipeline: wgpu::ComputePipeline,
    copy_bgl: wgpu::BindGroupLayout,
    copy_bind_group: Option<wgpu::BindGroup>,
    copy_bind_group_key: Option<usize>,

    // HiZ sampler (always owned by this pass)
    pub hiz_sampler: Arc<wgpu::Sampler>,
    // Per-mip views created from graph-owned texture; rebuilt on resize
    mip_views: Vec<wgpu::TextureView>,
    width: u32,
    height: u32,

    // Camera tracking for HiZ reuse optimization (skip rebuild if camera static)
    prev_camera_generation: u64,
    /// Whether this is the first frame (forces a full rebuild regardless of generation).
    first_frame: bool,
}

impl HiZBuildPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let hiz_sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HiZ Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }));

        // Phase 2: mip-chain downsampling pipeline
        let mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HiZ Build Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hiz_build.wgsl").into()),
        });

        let mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HiZ Mip BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let mip_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HiZ Mip PL"),
            bind_group_layouts: &[Some(&mip_bgl)],
            immediate_size: 0,
        });

        let mip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HiZ Mip Pipeline"),
            layout: Some(&mip_pl),
            module: &mip_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Phase 1: depth-copy pipeline
        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HiZ Depth Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hiz_depth_copy.wgsl").into()),
        });

        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HiZ Copy BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let copy_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HiZ Copy PL"),
            bind_group_layouts: &[Some(&copy_bgl)],
            immediate_size: 0,
        });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HiZ Copy Pipeline"),
            layout: Some(&copy_pl),
            module: &copy_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Mip uniforms and bind groups are built lazily from the graph-owned texture.
        // Mip uniforms and dispatch groups are built once here (width/height-dependent).
        let mip_count = mip_levels(width, height).min(MAX_MIP_LEVELS);
        let mut mip_uniforms = Vec::with_capacity((mip_count.saturating_sub(1)) as usize);
        let mut mip_dispatch_groups = Vec::with_capacity((mip_count.saturating_sub(1)) as usize);
        for mip in 0..(mip_count.saturating_sub(1)) {
            let src_w = (width >> mip).max(1);
            let src_h = (height >> mip).max(1);
            let dst_w = (width >> (mip + 1)).max(1);
            let dst_h = (height >> (mip + 1)).max(1);
            let ub = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("HiZ Mip Uniform"),
                contents: bytemuck::bytes_of(&HiZUniforms {
                    src_size: [src_w, src_h],
                    dst_size: [dst_w, dst_h],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            mip_uniforms.push(ub);
            mip_dispatch_groups.push((
                dst_w.div_ceil(WORKGROUP_SIZE),
                dst_h.div_ceil(WORKGROUP_SIZE),
            ));
        }

        // mip_views and mip_bind_groups need the wgpu::Texture handle which is
        // available via ctx.resource_pool in execute().
        Self {
            mip_pipeline,
            mip_bgl,
            mip_bind_groups: Vec::new(),
            mip_uniforms,
            mip_dispatch_groups,
            copy_pipeline,
            copy_bgl,
            copy_bind_group: None,
            copy_bind_group_key: None,
            hiz_sampler,
            mip_views: Vec::new(),
            width,
            height,
            prev_camera_generation: 0,
            first_frame: true,
        }
    }
}

fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &'static str {
        "HiZBuild"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["depth"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["hiz", "hiz_sampler"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw(
            "hiz",
            wgpu::TextureFormat::R32Float,
            ResourceSize::MatchSurface,
        );
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // Graph textures are re-allocated by the pool. Clear lazy views/bind groups
        // so they are rebuilt from the new graph-owned texture in execute().
        self.mip_views.clear();
        self.mip_bind_groups.clear();
        self.copy_bind_group = None;
        self.copy_bind_group_key = None;
        self.first_frame = true;
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── Lazy init: build mip views and bind groups from graph-owned texture ──
        if self.mip_views.is_empty() {
            let hiz_texture = ctx
                .resource_pool
                .get_texture("hiz")
                .expect("HiZ texture 'hiz' must be declared as a graph resource");
            let mip_count = mip_levels(self.width, self.height).min(MAX_MIP_LEVELS);

            // Create per-mip single-level views
            let mut mip_views = Vec::with_capacity(mip_count as usize);
            for mip in 0..mip_count {
                mip_views.push(hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("HiZ Mip View"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                }));
            }
            self.mip_views = mip_views;

            // Build mip bind groups from the existing uniforms + new views
            let mut mip_bind_groups = Vec::with_capacity((mip_count.saturating_sub(1)) as usize);
            for mip in 0..(mip_count.saturating_sub(1)) {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HiZ Mip BG"),
                    layout: &self.mip_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.mip_uniforms[mip as usize].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.mip_views[mip as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &self.mip_views[(mip + 1) as usize],
                            ),
                        },
                    ],
                });
                mip_bind_groups.push(bg);
            }
            self.mip_bind_groups = mip_bind_groups;
        }

        // ── HiZ Reuse optimization: skip rebuild if camera static ─────────────
        let camera_gen = ctx.scene.camera_generation;
        let resolution_changed = false;

        if !self.first_frame
            && camera_gen == self.prev_camera_generation
            && self.copy_bind_group.is_some()
            && !resolution_changed
        {
            return Ok(());
        }

        self.first_frame = false;
        self.prev_camera_generation = camera_gen;

        // Rebuild depth-copy bind group if the depth texture view pointer changed
        let depth_key = ctx.depth as *const _ as usize;
        if self.copy_bind_group_key != Some(depth_key) {
            self.copy_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HiZ Copy BG"),
                layout: &self.copy_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.mip_views[0]),
                    },
                ],
            }));
            self.copy_bind_group_key = Some(depth_key);
        }

        // Phase 1: copy depth -> HiZ mip-0
        {
            let mut pass =
                unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("HiZ DepthCopy"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, self.copy_bind_group.as_ref().unwrap(), &[]);
            let wg_x = self.width.div_ceil(WORKGROUP_SIZE);
            let wg_y = self.height.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Phase 2: build the remaining mip levels via MAX-reduction
        {
            let mut pass =
                unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("HiZ MipChain"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.mip_pipeline);
            for (bg, &(wg_x, wg_y)) in self
                .mip_bind_groups
                .iter()
                .zip(self.mip_dispatch_groups.iter())
            {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
        Ok(())
    }

    fn publish<'a>(&'a self, frame: &mut FrameResources<'a>) {
        // The graph routes "hiz" texture view via pre_pass_actions before execute().
        // We only need to publish the sampler (not owned by the graph).
        frame.hiz_sampler.write(&*self.hiz_sampler, "HiZBuild");
    }
}
