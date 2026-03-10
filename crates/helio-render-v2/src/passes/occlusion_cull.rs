//! Hi-Z occlusion culling compute pass.
//!
//! Uses the Hi-Z pyramid built by `HiZBuildPass` to cull instance slots that
//! are entirely behind existing geometry.  Results are written to the
//! `visibility_buffer` bitmask in `GpuScene`.
//!
//! Must execute **after** `HiZBuildPass` and **before** `IndirectDispatchPass`
//! so the visibility bitmask is up-to-date when draw commands are built.

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use crate::Result;
use crate::passes::hiz_build::HiZResources;

// ── GPU uniform struct (must match WGSL `OcclusionCullInput`) ─────────────────
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct OcclusionCullInput {
    view_proj:     [[f32; 4]; 4],  // 64 bytes
    view_proj_inv: [[f32; 4]; 4],  // 64 bytes
    camera_pos:    [f32; 3],
    _pad0:         u32,
    screen_width:  u32,
    screen_height: u32,
    total_slots:   u32,
    hiz_mip_count: u32,
}

const UNIFORM_SIZE: u64 = std::mem::size_of::<OcclusionCullInput>() as u64;

pub struct OcclusionCullPass {
    pipeline:  Arc<wgpu::ComputePipeline>,
    bgl:       wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    uniform_buf: wgpu::Buffer,
}

impl OcclusionCullPass {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Occlusion Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/occlusion_cull.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Occlusion Cull BGL"),
            entries: &[
                // 0: OcclusionCullInput uniform
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 1: AABB buffer (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 2: Hi-Z texture (sampled)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                // 3: Hi-Z sampler (nearest)
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // 4: Visibility bitmask (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("Occlusion Cull Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label:               Some("Occlusion Cull Pipeline"),
                layout:              Some(&layout),
                module:              &shader,
                entry_point:         Some("occlusion_cull"),
                compilation_options: Default::default(),
                cache:               None,
            },
        ));

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Occlusion Cull Uniform"),
            size:               UNIFORM_SIZE,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self { pipeline, bgl, bind_group: None, uniform_buf })
    }

    /// Bind the per-frame GPU resources.  Call once after resources are created
    /// or after a resize (new Hi-Z texture / new aabb_buffer / visibility_buffer).
    pub fn bind_resources(
        &mut self,
        device:         &wgpu::Device,
        aabb_buf:       &wgpu::Buffer,
        hiz:            &HiZResources,
        visibility_buf: &wgpu::Buffer,
    ) {
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Occlusion Cull BG"),
            layout:  &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: aabb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: wgpu::BindingResource::TextureView(&hiz.full_view),
                },
                wgpu::BindGroupEntry {
                    binding:  3,
                    resource: wgpu::BindingResource::Sampler(&hiz.sampler),
                },
                wgpu::BindGroupEntry {
                    binding:  4,
                    resource: visibility_buf.as_entire_binding(),
                },
            ],
        }));
    }

    /// Upload camera uniforms and dispatch the occlusion cull compute.
    ///
    /// `total_slots` is the high-water-mark of the GpuScene slot allocator
    /// (i.e., `cpu_data.len()`).  One thread is dispatched per slot.
    pub fn execute(
        &self,
        encoder:       &mut wgpu::CommandEncoder,
        queue:         &wgpu::Queue,
        view_proj:     [[f32; 4]; 4],
        view_proj_inv: [[f32; 4]; 4],
        camera_pos:    [f32; 3],
        screen_width:  u32,
        screen_height: u32,
        total_slots:   u32,
        hiz_mip_count: u32,
    ) {
        let Some(bg) = &self.bind_group else { return; };
        if total_slots == 0 { return; }

        let uniforms = OcclusionCullInput {
            view_proj,
            view_proj_inv,
            camera_pos,
            _pad0: 0,
            screen_width,
            screen_height,
            total_slots,
            hiz_mip_count,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("Occlusion Cull"),
            timestamp_writes: None,
        });

        cp.set_pipeline(&self.pipeline);
        cp.set_bind_group(0, bg, &[]);
        // 64 threads per workgroup (matches shader @workgroup_size(64))
        cp.dispatch_workgroups((total_slots + 63) / 64, 1, 1);
    }
}
