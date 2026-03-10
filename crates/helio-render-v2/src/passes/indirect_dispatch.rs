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

use std::sync::Arc;
use crate::Result;
use bytemuck;

/// Non-compacting GPU culling and indirect-buffer-fill pass.
pub struct IndirectDispatchPass {
    pipeline:    Option<Arc<wgpu::ComputePipeline>>,
    bind_group:  Option<wgpu::BindGroup>,
    /// The output buffer: one `DrawIndexedIndirect` (20 bytes) per input draw.
    /// Grow-only: never shrinks; the active region is [0 .. draw_count * 20].
    indirect_buffer: Option<Arc<wgpu::Buffer>>,
    /// Tiny uniform buffer holding draw_count so the shader knows how many
    /// entries in draw_call_buffer are actually valid (the rest is pre-allocated
    /// uninitialized VRAM and must NOT be read).
    params_buffer: Option<wgpu::Buffer>,
    /// Buffer allocation capacity (in draw slots), may be ≥ draw_count.
    buf_capacity: u32,
    /// Active draw-call count for this frame (set by update, read by dispatch).
    draw_count: u32,
    /// Last draw_call_buffer pointer seen — if it changes the bind group is stale.
    last_draw_call_buf_ptr: usize,
    /// Last camera_buffer pointer seen.
    last_camera_buf_ptr: usize,
    /// Last indirect_buffer pointer seen (changes on grow).
    last_indirect_buf_ptr: usize,
}

impl IndirectDispatchPass {
    pub fn new() -> Self {
        Self {
            pipeline:             None,
            bind_group:           None,
            indirect_buffer:      None,
            params_buffer:        None,
            buf_capacity:          0,
            draw_count:            0,
            last_draw_call_buf_ptr: 0,
            last_camera_buf_ptr:   0,
            last_indirect_buf_ptr: 0,
        }
    }

    /// (Re-)build GPU resources for `draw_count` draw calls.
    ///
    /// * `draw_call_buffer` — read-only storage, `GpuDrawCall[draw_count]`
    /// * `camera_buffer`    — uniform, camera view_proj for frustum culling
    ///
    /// Grow-only: the indirect buffer is never reallocated unless draw_count
    /// exceeds the current capacity.  The bind group is only recreated when
    /// one of the input buffers changes identity (pointer).
    ///
    /// Returns the (shared) indirect buffer, or `None` if draw_count == 0.
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        draw_call_buffer: &Arc<wgpu::Buffer>,
        camera_buffer: &wgpu::Buffer,
        draw_count: u32,
    ) -> Result<Option<Arc<wgpu::Buffer>>> {
        self.draw_count = draw_count;
        println!("[IndirectDispatch] update: draw_count={}, buf_capacity={}, pipeline={}, bg={}",
            draw_count, self.buf_capacity,
            self.pipeline.is_some(), self.bind_group.is_some());

        if draw_count == 0 {
            println!("[IndirectDispatch] update called with draw_count == 0, nothing to dispatch");
            return Ok(self.indirect_buffer.clone());
        }

        // Build the pipeline once.
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false, min_binding_size: None,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false, min_binding_size: None,
                        }, count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false, min_binding_size: None,
                        }, count: None,
                    },
                    // binding 3: draw_count uniform — the actual number of valid draw calls.
                    // The draw_call_buffer is pre-allocated to 16K; only [0..draw_count) is valid.
                    wgpu::BindGroupLayoutEntry {
                        binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false, min_binding_size: None,
                        }, count: None,
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
        }

        // Create params_buffer once (4 bytes, written every frame in dispatch()).
        if self.params_buffer.is_none() {
            self.params_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Indirect Dispatch Params"),
                size:  16, // min uniform buffer size; only first 4 bytes (draw_count u32) used
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            self.bind_group = None; // force rebuild
        }

        // Grow indirect buffer only when capacity is insufficient.
        let buf_dirty = draw_count > self.buf_capacity;
        if buf_dirty {
            let new_cap = draw_count.next_power_of_two().max(64);
            self.indirect_buffer = Some(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Indirect Draw Buffer"),
                size: new_cap as u64 * 20,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::INDIRECT
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })));
            self.buf_capacity = new_cap;
            self.bind_group = None;
        }

        // Rebuild bind group if any input buffer changed identity.
        let dc_ptr  = Arc::as_ptr(draw_call_buffer) as usize;
        let cam_ptr = camera_buffer as *const wgpu::Buffer as usize;
        let ind_ptr = self.indirect_buffer.as_ref().map_or(0, |b| Arc::as_ptr(b) as usize);
        let bg_stale = self.bind_group.is_none()
            || dc_ptr  != self.last_draw_call_buf_ptr
            || cam_ptr != self.last_camera_buf_ptr
            || ind_ptr != self.last_indirect_buf_ptr;

        if bg_stale {
            let pipeline = self.pipeline.as_ref().unwrap();
            let indirect = self.indirect_buffer.as_ref().unwrap();
            let params   = self.params_buffer.as_ref().unwrap();
            self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("Indirect Dispatch BG"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: draw_call_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: indirect.as_entire_binding()         },
                    wgpu::BindGroupEntry { binding: 2, resource: camera_buffer.as_entire_binding()    },
                    wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding()           },
                ],
            }));
            self.last_draw_call_buf_ptr = dc_ptr;
            self.last_camera_buf_ptr    = cam_ptr;
            self.last_indirect_buf_ptr  = ind_ptr;
        }

        Ok(self.indirect_buffer.clone())
    }

    /// Run the compute shader on the provided encoder.
    /// Must be called after `update()` and before any render pass reads the indirect buffer.
    pub fn dispatch(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        if self.pipeline.is_none() {
            println!("[IndirectDispatch] dispatch() called but pipeline is None — update() was not called this frame");
            return;
        }
        if self.bind_group.is_none() {
            println!("[IndirectDispatch] dispatch() called but bind_group is None — update() failed to build bind group");
            return;
        }
        if self.indirect_buffer.is_none() {
            println!("[IndirectDispatch] dispatch() called but indirect_buffer is None — update() was not called");
            return;
        }
        if self.params_buffer.is_none() {
            println!("[IndirectDispatch] dispatch() called but params_buffer is None");
            return;
        }
        if self.draw_count == 0 {
            println!("[IndirectDispatch] dispatch() called with draw_count == 0, skipping");
            return;
        }

        println!("[IndirectDispatch] dispatching {} draw calls, {} workgroups, indirect buf size {} bytes",
            self.draw_count,
            (self.draw_count + 63) / 64,
            self.draw_count as u64 * 20
        );

        // Upload draw_count to the params uniform so the shader knows the valid range.
        // The draw_call_buffer is 16K pre-allocated; only [0..draw_count) is valid.
        queue.write_buffer(
            self.params_buffer.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&self.draw_count),
        );

        let pipeline   = self.pipeline.as_ref().unwrap();
        let bind_group = self.bind_group.as_ref().unwrap();
        let indirect   = self.indirect_buffer.as_ref().unwrap();

        // Zero the active region so culled draws have instance_count = 0.
        encoder.clear_buffer(indirect, 0, Some(self.draw_count as u64 * 20));

        let workgroups = (self.draw_count + 63) / 64;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Indirect Dispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        println!("[IndirectDispatch] dispatch complete");
    }

    pub fn draw_count(&self) -> u32 { self.draw_count }
    pub fn capacity(&self)   -> u32 { self.buf_capacity }
}

impl Default for IndirectDispatchPass {
    fn default() -> Self { Self::new() }
}
