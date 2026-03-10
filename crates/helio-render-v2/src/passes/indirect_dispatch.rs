//! GPU-driven indirect rendering: frustum-cull all draw calls in a single
//! compute dispatch and write `DrawIndexedIndirect` commands to the indirect
//! buffer.
//!
//! Non-compacting design:
//!   - One DrawIndexedIndirect slot per input draw call (fixed index).
//!   - Culled draws get `instance_count = 0` — the GPU skips them for free.
//!   - `first_instance = dc.slot` so the vertex shader can index into
//!     `instance_data[slot]` for the per-object transform.
//!   - No atomics, no prefix sum, no CPU readback.  O(1) CPU cost.

use crate::graph::{RenderPass, PassContext, PassResourceBuilder};
use std::sync::Arc;
use crate::Result;

/// Non-compacting GPU culling and indirect-buffer-fill pass.
pub struct IndirectDispatchPass {
    pipeline:    Option<Arc<wgpu::ComputePipeline>>,
    bind_group:  Option<wgpu::BindGroup>,
    /// The output buffer: one `DrawIndexedIndirect` (20 bytes) per input draw.
    indirect_buffer: Option<Arc<wgpu::Buffer>>,
    /// Number of draw calls the buffers were sized for.
    capacity: u32,
}

impl IndirectDispatchPass {
    pub fn new() -> Self {
        Self {
            pipeline:        None,
            bind_group:      None,
            indirect_buffer: None,
            capacity:        0,
        }
    }

    /// (Re-)create GPU resources for `draw_count` draw calls.
    ///
    /// * `draw_call_buffer` — read-only storage, contains `GpuDrawCall[draw_count]`
    /// * `camera_buffer`    — uniform containing camera view_proj for frustum culling
    /// Update GPU resources for `draw_count` draw calls.
    /// Returns the (possibly newly allocated) indirect buffer, or `None` if draw_count == 0.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        draw_call_buffer: &wgpu::Buffer,
        camera_buffer: &wgpu::Buffer,
        draw_count: u32,
    ) -> Result<Option<Arc<wgpu::Buffer>>> {
        if draw_count == 0 {
            self.bind_group      = None;
            self.indirect_buffer = None;
            self.capacity        = 0;
            return Ok(None);
        }

        // (Re)allocate indirect buffer sized for draw_count 5-word indirect commands.
        let indirect_size = draw_count as u64 * 20; // 5 × u32 each
        let indirect_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Indirect Draw Buffer"),
            size: indirect_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Build pipeline on first call.
        if self.pipeline.is_none() {
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Indirect Dispatch Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    include_str!("../../shaders/passes/indirect_dispatch.wgsl").into(),
                ),
            });

            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Indirect Dispatch BGL"),
                entries: &[
                    // binding 0: GpuDrawCall array (read-only)
                    wgpu::BindGroupLayoutEntry {
                        binding:    0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: indirect output (read-write)
                    wgpu::BindGroupLayoutEntry {
                        binding:    1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 2: camera uniform (for frustum planes)
                    wgpu::BindGroupLayoutEntry {
                        binding:    2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Indirect Dispatch Pipeline Layout"),
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });

            self.pipeline = Some(Arc::new(device.create_compute_pipeline(
                &wgpu::ComputePipelineDescriptor {
                    label:    Some("Indirect Dispatch Pipeline"),
                    layout:   Some(&pl),
                    module:   &shader,
                    entry_point: Some("build_indirect_buffers"),
                    compilation_options: Default::default(),
                    cache:    None,
                },
            )));

            let pipeline = self.pipeline.as_ref().unwrap();
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("Indirect Dispatch BG"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: draw_call_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: indirect_buf.as_entire_binding()    },
                    wgpu::BindGroupEntry { binding: 2, resource: camera_buffer.as_entire_binding()   },
                ],
            });
            self.bind_group = Some(bg);
        } else {
            // Pipeline already exists — only recreate the bind group (buffers may have grown).
            let pipeline = self.pipeline.as_ref().unwrap();
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("Indirect Dispatch BG"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: draw_call_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: indirect_buf.as_entire_binding()    },
                    wgpu::BindGroupEntry { binding: 2, resource: camera_buffer.as_entire_binding()   },
                ],
            });
            self.bind_group = Some(bg);
        }

        self.indirect_buffer = Some(indirect_buf);
        self.capacity        = draw_count;
        Ok(self.indirect_buffer.clone())
    }

    /// Returns the indirect buffer produced by the last `update()`.
    pub fn indirect_buffer(&self) -> Option<&Arc<wgpu::Buffer>> {
        self.indirect_buffer.as_ref()
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

impl RenderPass for IndirectDispatchPass {
    fn name(&self) -> &str {
        "indirect_dispatch"
    }

    fn declare_resources(&self, _builder: &mut PassResourceBuilder) {}

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let Some(pipeline)    = &self.pipeline    else { return Ok(()); };
        let Some(bind_group)  = &self.bind_group  else { return Ok(()); };
        let Some(indirect_buf)= &self.indirect_buffer else { return Ok(()); };
        if self.capacity == 0 { return Ok(()); }

        // Zero the indirect buffer so culled draws have instance_count = 0.
        // (The compute shader only writes instance_count = 1 for visible draws.)
        ctx.encoder.clear_buffer(indirect_buf, 0, None);

        let workgroups = (self.capacity + 63) / 64;
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Indirect Dispatch"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        ctx.encoder.insert_debug_marker("indirect buffer ready");
        Ok(())
    }
}

impl Default for IndirectDispatchPass {
    fn default() -> Self { Self::new() }
}
