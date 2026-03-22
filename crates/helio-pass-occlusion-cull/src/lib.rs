//! Hi-Z occlusion culling pass.
//!
//! Reads the Hi-Z pyramid and per-instance AABB data, outputs a visibility bitmask.
//! The indirect dispatch pass uses this bitmask to zero-out instance_count for
//! occluded draws. O(1) CPU — single compute dispatch.

use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};
use bytemuck::{Pod, Zeroable};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OcclusionUniforms {
    instance_count: u32,
    hiz_width: u32,
    hiz_height: u32,
    hiz_mip_count: u32,
}

pub struct OcclusionCullPass {
    pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl OcclusionCullPass {
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        instances_buf: &wgpu::Buffer,
        hiz_view: &wgpu::TextureView,
        hiz_sampler: &wgpu::Sampler,
        indirect_buf: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("OcclusionCull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/occlusion_cull.wgsl").into(),
            ),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OcclusionCull Uniforms"),
            size: std::mem::size_of::<OcclusionUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("OcclusionCull BGL"),
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
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("OcclusionCull BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(hiz_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(hiz_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect_buf.as_entire_binding(),
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("OcclusionCull PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("OcclusionCull Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("occlusion_cull"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buf,
            bind_group,
        }
    }
}

impl RenderPass for OcclusionCullPass {
    fn name(&self) -> &'static str {
        "OcclusionCull"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let u = OcclusionUniforms {
            instance_count: ctx.scene.instances.len() as u32,
            hiz_width: ctx.width,
            hiz_height: ctx.height,
            hiz_mip_count: mip_count(ctx.width, ctx.height),
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let count = ctx.scene.instance_count;
        if count == 0 {
            return Ok(());
        }
        let wg = count.div_ceil(WORKGROUP_SIZE);
        let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("OcclusionCull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }
}

#[allow(dead_code)]
fn mip_count(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}
