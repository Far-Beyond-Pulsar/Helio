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
//! `hiz_view` and `hiz_sampler` are Arc-wrapped so OcclusionCullPass can hold
//! its own reference to the persistent texture without lifetime issues.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_v3::{FrameResources, PassContext, PrepareContext, RenderPass, Result as HelioResult};

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
    #[allow(dead_code)]
    mip_bgl: wgpu::BindGroupLayout,
    mip_bind_groups: Vec<wgpu::BindGroup>,
    mip_uniforms: Vec<wgpu::Buffer>,

    // Depth-copy pipeline (Depth32Float -> R32Float mip-0)
    copy_pipeline: wgpu::ComputePipeline,
    copy_bgl: wgpu::BindGroupLayout,
    copy_bind_group: Option<wgpu::BindGroup>,
    copy_bind_group_key: Option<usize>,

    // Shared HiZ texture resources (Arc so OcclusionCullPass can co-own)
    pub hiz_view: Arc<wgpu::TextureView>,
    pub hiz_sampler: Arc<wgpu::Sampler>,
    mip_views: Vec<wgpu::TextureView>,
    width: u32,
    height: u32,

    // Camera tracking for HiZ reuse optimization (skip rebuild if camera static)
    prev_camera_hash: u64,
}

impl HiZBuildPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mip_count = mip_levels(width, height).min(MAX_MIP_LEVELS);

        // HiZ texture
        let hiz_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HiZ Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let hiz_view = Arc::new(hiz_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("HiZ Full View"),
            ..Default::default()
        }));

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

        // Per-mip single-level views
        let mut mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            mip_views.push(hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("HiZ Mip View"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            }));
        }

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

        // Build per-mip bind groups for mip[i]->mip[i+1] downsampling
        let mut mip_bind_groups = Vec::with_capacity((mip_count - 1) as usize);
        let mut mip_uniforms = Vec::with_capacity((mip_count - 1) as usize);
        for mip in 0..(mip_count - 1) {
            let ub = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("HiZ Mip Uniform"),
                size: std::mem::size_of::<HiZUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HiZ Mip BG"),
                layout: &mip_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&mip_views[mip as usize]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &mip_views[(mip + 1) as usize],
                        ),
                    },
                ],
            });
            mip_uniforms.push(ub);
            mip_bind_groups.push(bg);
        }

        // Phase 1: depth-copy pipeline
        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HiZ Depth Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/hiz_depth_copy.wgsl").into(),
            ),
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

        Self {
            mip_pipeline,
            mip_bgl,
            mip_bind_groups,
            mip_uniforms,
            copy_pipeline,
            copy_bgl,
            copy_bind_group: None,
            copy_bind_group_key: None,
            hiz_view,
            hiz_sampler,
            mip_views,
            width,
            height,
            prev_camera_hash: 0, // Force rebuild on first frame
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

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let w = self.width;
        let h = self.height;
        for (mip, ub) in self.mip_uniforms.iter().enumerate() {
            let mip = mip as u32;
            let src_w = (w >> mip).max(1);
            let src_h = (h >> mip).max(1);
            let dst_w = (w >> (mip + 1)).max(1);
            let dst_h = (h >> (mip + 1)).max(1);
            ctx.write_buffer(
                ub,
                0,
                bytemuck::bytes_of(&HiZUniforms {
                    src_size: [src_w, src_h],
                    dst_size: [dst_w, dst_h],
                }),
            );
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
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
        // A separate compute pass provides an implicit GPU barrier so Phase 2
        // sees the freshly written mip-0 data.
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ DepthCopy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, self.copy_bind_group.as_ref().unwrap(), &[]);
            let wg_x = self.width.div_ceil(WORKGROUP_SIZE);
            let wg_y = self.height.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        } // pass drops here -> implicit GPU barrier

        // Phase 2: build the remaining mip levels via MAX-reduction
        // O(log resolution) dispatches, fixed at ~10 for <= 4K resolution.
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ MipChain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mip_pipeline);
            for (mip, bg) in self.mip_bind_groups.iter().enumerate() {
                let mip = mip as u32;
                let dst_w = (ctx.width >> (mip + 1)).max(1);
                let dst_h = (ctx.height >> (mip + 1)).max(1);
                let wg_x = dst_w.div_ceil(WORKGROUP_SIZE);
                let wg_y = dst_h.div_ceil(WORKGROUP_SIZE);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
        Ok(())
    }

    fn publish<'a>(&'a self, frame: &mut FrameResources<'a>) {
        // Expose the current frame's HiZ for any downstream pass that needs it.
        // OcclusionCullPass uses its own Arc ref so it always reads the
        // previous frame's data (temporal), not this freshly built pyramid.
        frame.hiz = Some(&*self.hiz_view);
        frame.hiz_sampler = Some(&*self.hiz_sampler);
    }
}
