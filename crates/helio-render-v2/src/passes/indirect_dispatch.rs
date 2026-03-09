//! GPU-driven indirect rendering: build indirect draw buffers via compute
//!
//! This pass runs a compute shader that:
//! 1. Iterates over visible draw calls
//! 2. Builds `DrawIndexedIndirect` commands in GPU buffers
//! 3. Separates opaque and transparent draws
//! 4. Writes draw counts to atomic counters
//!
//! The resulting indirect buffers are then used by GBufferPass and
//! TransparentPass to submit draws via `draw_indexed_indirect()` instead
//! of looping over draw calls on the CPU.
//!
//! This is the GPU-driven rendering pattern: move draw submission to GPU
//! (similar to Unreal's GPU Scene or Nanite scene).

use crate::graph::{RenderPass, PassContext, PassResourceBuilder};
use std::sync::Arc;
use crate::Result;

/// Indirect draw buffer (array of DrawIndexedIndirect commands)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// Pass that builds indirect draw buffers via compute shader
pub struct IndirectDispatchPass {
    /// Pipeline for building indirect buffers
    pipeline: Option<Arc<wgpu::ComputePipeline>>,
    
    /// Bind group for compute shader (camera, lights, input draw list)
    bind_group: Option<wgpu::BindGroup>,
    
    /// Max draws per category (opaque/transparent)
    max_draws_opaque: u32,
    max_draws_transparent: u32,
    
    /// Workgroup size (must match shader)
    workgroup_size: u32,
}

impl IndirectDispatchPass {
    pub fn new() -> Self {
        Self {
            pipeline: None,
            bind_group: None,
            max_draws_opaque: 512,      // Typically 50-200 real opaque draws per frame
            max_draws_transparent: 256,  // Typically 10-50 transparent draws per frame
            workgroup_size: 256,
        }
    }

    pub fn with_max_draws(mut self, max_opaque: u32, max_transparent: u32) -> Self {
        self.max_draws_opaque = max_opaque;
        self.max_draws_transparent = max_transparent;
        self
    }

    /// Initialize GPU pipeline and bind group
    pub fn initialize(
        &mut self,
        device: &wgpu::Device,
        global_bgl: &wgpu::BindGroupLayout,
        draw_list_buffer: &wgpu::Buffer,
        indirect_opaque_buffer: &wgpu::Buffer,
        indirect_transparent_buffer: &wgpu::Buffer,
        opaque_draw_count_buffer: &wgpu::Buffer,
        transparent_draw_count_buffer: &wgpu::Buffer,
    ) -> Result<()> {
        // Create compute pipeline
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Indirect Dispatch Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/indirect_dispatch.wgsl").into(),
            ),
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Indirect Dispatch Layout"),
            bind_group_layouts: &[
                Some(global_bgl),
            ],
            immediate_size: 0,
        });

        self.pipeline = Some(Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("Indirect Dispatch Pipeline"),
                layout: Some(&layout),
                module: &shader,
                entry_point: Some("build_indirect_buffers"),
                compilation_options: Default::default(),
                cache: None,
            },
        )));

        // Create bind group layout with all buffers: draw list input + indirect outputs + counters
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Indirect Dispatch BGL"),
            entries: &[
                // Binding 0: Input draw list (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: true,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 1: Opaque indirect buffer (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 2: Transparent indirect buffer (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 3: Opaque count (atomic storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Binding 4: Transparent count (atomic storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage {
                            read_only: false,
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Indirect Dispatch BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: draw_list_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: indirect_opaque_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indirect_transparent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: opaque_draw_count_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: transparent_draw_count_buffer.as_entire_binding(),
                },
            ],
        }));

        Ok(())
    }

    pub fn pipeline(&self) -> Option<&Arc<wgpu::ComputePipeline>> {
        self.pipeline.as_ref()
    }

    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    pub fn max_draws_opaque(&self) -> u32 {
        self.max_draws_opaque
    }

    pub fn max_draws_transparent(&self) -> u32 {
        self.max_draws_transparent
    }

    pub fn workgroup_size(&self) -> u32 {
        self.workgroup_size
    }
}

impl RenderPass for IndirectDispatchPass {
    fn name(&self) -> &str {
        "indirect_dispatch"
    }

    fn declare_resources(&self, _builder: &mut PassResourceBuilder) {
        // This pass doesn't read/write traditional render graph resources
        // It writes to GPU buffers directly via compute
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        //Skip if not initialized
        let Some(pipeline) = &self.pipeline else {
            return Ok(());
        };
        let Some(bind_group) = &self.bind_group else {
            return Ok(());
        };

        // Run compute shader to build indirect buffers from draw list
        // Each workgroup processes 256 draws in parallel
        let max_draws_total = self.max_draws_opaque + self.max_draws_transparent;
        let workgroups_needed = (max_draws_total + self.workgroup_size - 1) / self.workgroup_size;

        {
            let mut compute_pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Build Indirect Buffers"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, ctx.global_bind_group, &[]);
            compute_pass.set_bind_group(1, bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_needed, 1, 1);
        }

        // Barrier: ensure indirect buffers are written before GBuffer/Transparent passes read them
        ctx.encoder.insert_debug_marker("Indirect buffer ready");

        Ok(())
    }
}

impl Default for IndirectDispatchPass {
    fn default() -> Self {
        Self::new()
    }
}
