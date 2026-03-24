//! Hi-Z (Hierarchical Z) pyramid builder.
//!
//! Reads the depth buffer and builds a full mip chain where each texel
//! stores the MAXIMUM depth in its 2×2 footprint (conservative depth).
//! Used by OcclusionCullPass to test occluded objects.
//!
//! O(1) CPU: fixed number of compute dispatches (one per mip level, ~10 total).

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const WORKGROUP_SIZE: u32 = 8;
const MAX_MIP_LEVELS: u32 = 12;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HiZUniforms {
    src_size: [u32; 2],
    dst_size: [u32; 2],
}

pub struct HiZBuildPass {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    /// Per-mip bind groups (one per mip transition: src_mip → dst_mip)
    mip_bind_groups: Vec<wgpu::BindGroup>,
    /// Per-mip uniform buffers
    mip_uniforms: Vec<wgpu::Buffer>,
    /// HiZ texture (owned by this pass, output for OcclusionCullPass)
    pub hiz_texture: wgpu::Texture,
    pub hiz_view: wgpu::TextureView,
    pub hiz_sampler: wgpu::Sampler,
    #[allow(dead_code)]
    mip_views: Vec<wgpu::TextureView>,
    #[allow(dead_code)]
    mip_count: u32,
    width: u32,
    height: u32,
}

impl HiZBuildPass {
    pub fn new(
        device: &wgpu::Device,
        _depth_view: &wgpu::TextureView,
        width: u32,
        height: u32,
    ) -> Self {
        let mip_count = mip_levels(width, height).min(MAX_MIP_LEVELS);
        let hiz_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HiZ Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let hiz_view = hiz_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("HiZ Full View"),
            ..Default::default()
        });

        let hiz_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HiZ Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HiZ Build Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hiz_build.wgsl").into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HiZ BGL"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HiZ PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HiZ Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Build per-mip views
        let mut mip_views = Vec::new();
        for mip in 0..mip_count {
            let view = hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("HiZ Mip View"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            });
            mip_views.push(view);
        }

        let mut mip_bind_groups = Vec::new();
        let mut mip_uniforms = Vec::new();

        for mip in 0..(mip_count - 1) {
            let ub = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("HiZ Mip Uniform"),
                size: std::mem::size_of::<HiZUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let src_view = &mip_views[mip as usize];
            let dst_view = &mip_views[(mip + 1) as usize];

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HiZ Mip BG"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(src_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(dst_view),
                    },
                ],
            });
            mip_uniforms.push(ub);
            mip_bind_groups.push(bg);
        }

        Self {
            pipeline,
            bind_group_layout,
            mip_bind_groups,
            mip_uniforms,
            hiz_texture,
            hiz_view,
            hiz_sampler,
            mip_views,
            mip_count,
            width,
            height,
        }
    }

    /// Returns the HiZ texture view suitable for binding in OcclusionCullPass.
    pub fn hiz_view(&self) -> &wgpu::TextureView {
        &self.hiz_view
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
        // Upload mip uniforms using stored dimensions (updated on resize via rebuild).
        // TODO: ctx.resize + ctx.width/ctx.height when PrepareContext exposes them.
        let w = self.width;
        let h = self.height;
        for (mip, ub) in self.mip_uniforms.iter().enumerate() {
            let mip = mip as u32;
            let src_w = (w >> mip).max(1);
            let src_h = (h >> mip).max(1);
            let dst_w = (w >> (mip + 1)).max(1);
            let dst_h = (h >> (mip + 1)).max(1);
            let uniforms = HiZUniforms {
                src_size: [src_w, src_h],
                dst_size: [dst_w, dst_h],
            };
            ctx.write_buffer(ub, 0, bytemuck::bytes_of(&uniforms));
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(log resolution) dispatches — fixed at ~10 for any 4K or below resolution
        let mut pass = ctx
            .encoder
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZBuild"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        for (mip, bg) in self.mip_bind_groups.iter().enumerate() {
            let mip = mip as u32;
            let dst_w = (ctx.width >> (mip + 1)).max(1);
            let dst_h = (ctx.height >> (mip + 1)).max(1);
            let wg_x = dst_w.div_ceil(WORKGROUP_SIZE);
            let wg_y = dst_h.div_ceil(WORKGROUP_SIZE);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }
        Ok(())
    }
}

