//! Hi-Z occlusion-culling pass.
//!
//! Runs BEFORE DepthPrepassPass each frame, using the PREVIOUS frame's Hi-Z
//! pyramid (temporal approach).  For each draw slot the shader:
//!  1. Projects the bounding sphere centre into NDC.
//!  2. Samples the Hi-Z pyramid at the mip level matching the sphere's
//!     screen-space footprint.
//!  3. Writes `indirect[slot * 5 + 1]` = 1 (visible) or 0 (occluded).
//!
//! Frame 0 is skipped since no Hi-Z pyramid exists yet.
//! Bind-group is rebuilt lazily when buffer pointers change (e.g. scene grows).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullParams {
    screen_width:  u32,
    screen_height: u32,
    total_slots:   u32,
    hiz_mip_count: u32,
}

pub struct OcclusionCullPass {
    pipeline:        wgpu::ComputePipeline,
    bgl:             wgpu::BindGroupLayout,
    cull_params_buf: wgpu::Buffer,
    hiz_view:        Arc<wgpu::TextureView>,
    hiz_sampler:     Arc<wgpu::Sampler>,

    /// Cached bind group, invalidated when buffer pointers change.
    bind_group:     Option<wgpu::BindGroup>,
    /// (camera_ptr, instances_ptr, indirect_ptr)
    bind_group_key: Option<(usize, usize, usize)>,
}

impl OcclusionCullPass {
    /// Create the occlusion-cull pass.
    ///
    /// `hiz_view` and `hiz_sampler` are Arc-wrapped GPU resources owned by
    /// `HiZBuildPass`. The bind group for each frame's scene buffers is built
    /// lazily in `execute()`.
    pub fn new(
        device: &wgpu::Device,
        hiz_view: Arc<wgpu::TextureView>,
        hiz_sampler: Arc<wgpu::Sampler>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("OcclusionCull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/occlusion_cull.wgsl").into(),
            ),
        });

        let cull_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("OcclusionCull CullParams"),
            size:               std::mem::size_of::<CullParams>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Bind group layout must match occlusion_cull.wgsl binding declarations.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("OcclusionCull BGL"),
            entries: &[
                // 0: Camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 1: CullParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 2: GpuInstanceData[] (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 3: Hi-Z texture
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                // 4: Hi-Z sampler (non-filtering, nearest)
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // 5: indirect draw buffer (read + write, u32 raw view)
                wgpu::BindGroupLayoutEntry {
                    binding:    5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("OcclusionCull PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("OcclusionCull Pipeline"),
            layout:              Some(&pl),
            module:              &shader,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        Self {
            pipeline,
            bgl,
            cull_params_buf,
            hiz_view,
            hiz_sampler,
            bind_group:     None,
            bind_group_key: None,
        }
    }
}

impl RenderPass for OcclusionCullPass {
    fn name(&self) -> &'static str {
        "OcclusionCull"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let p = CullParams {
            screen_width:  ctx.width,
            screen_height: ctx.height,
            total_slots:   ctx.scene.instances.len() as u32,
            hiz_mip_count: mip_levels(ctx.width, ctx.height),
        };
        ctx.write_buffer(&self.cull_params_buf, 0, bytemuck::bytes_of(&p));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Temporal Hi-Z: frame 0 has no valid pyramid yet — skip culling.
        if ctx.frame_num == 0 {
            // Ensure all draws are visible on frame 0 (instance_count already 1
            // from CPU-side scene setup, so this is a no-op in most cases).
            return Ok(());
        }

        if ctx.scene.draw_count == 0 {
            // No draw commands this frame; skip expensive occlusion compute.
            return Ok(());
        }

        let count = ctx.scene.instance_count;
        if count == 0 {
            return Ok(());
        }

        // Lazy bind-group rebuild: rebuild whenever any buffer pointer changes
        // (e.g. scene grows and GrowableBuffer reallocates its wgpu::Buffer).
        let key = (
            ctx.scene.camera   as *const _ as usize,
            ctx.scene.instances as *const _ as usize,
            ctx.scene.indirect  as *const _ as usize,
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("OcclusionCull BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  1,
                        resource: self.cull_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  2,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  3,
                        resource: wgpu::BindingResource::TextureView(&self.hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding:  4,
                        resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding:  5,
                        resource: ctx.scene.indirect.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let wg = count.div_ceil(WORKGROUP_SIZE);
        let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("OcclusionCull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }
}

fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}
