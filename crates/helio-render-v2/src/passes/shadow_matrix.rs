//! GPU shadow matrix computation pass.
//!
//! Computes shadow light-space matrices entirely on GPU, eliminating per-frame
//! CPU matrix math overhead (Mat4::perspective, look_at, CSM sphere fitting).
//!
//! One thread per light; each light can output 1-6 matrices:
//!   - Point lights: 6 cube-face matrices (±X, ±Y, ±Z)
//!   - Directional lights: 4 CSM cascade matrices + 2 identity
//!   - Spot lights: 1 perspective matrix + 5 identity
//!
//! Integrates with GPU indirect dispatch system — runs before shadow pass.
//! Zero CPU cost when shadows haven't changed (uses dirty flags + generation counter).

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use crate::Result;

// ── GPU uniform struct (must match WGSL `ShadowMatrixParams`) ────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowMatrixParams {
    light_count:   u32,
    camera_moved:  u32,  // Boolean: 1 if camera moved (triggers CSM recompute)
    _pad0:         u32,
    _pad1:         u32,
}

const UNIFORM_SIZE: u64 = std::mem::size_of::<ShadowMatrixParams>() as u64;

pub struct ShadowMatrixPass {
    pipeline:    Arc<wgpu::ComputePipeline>,
    bgl:         wgpu::BindGroupLayout,
    bind_group:  Option<wgpu::BindGroup>,
    uniform_buf: wgpu::Buffer,

    /// Last camera buffer pointer seen (for bind group invalidation)
    last_camera_ptr: usize,
    /// Last light buffer pointer seen
    last_light_ptr: usize,
    /// Last shadow matrix buffer pointer seen
    last_shadow_mat_ptr: usize,
    /// Last shadow dirty buffer pointer seen
    last_shadow_dirty_ptr: usize,
    /// Last shadow hash buffer pointer seen
    last_shadow_hash_ptr: usize,
}

impl ShadowMatrixPass {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Shadow Matrix Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/shadow_matrices.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Shadow Matrix BGL"),
            entries: &[
                // 0: lights (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 1: shadow_mats (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 2: camera (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 3: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 4: shadow_dirty (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 5: shadow_hashes (read-write storage)
                wgpu::BindGroupLayoutEntry {
                    binding:    5,
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
            label:              Some("Shadow Matrix Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let pipeline = Arc::new(device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label:               Some("Shadow Matrix Pipeline"),
                layout:              Some(&layout),
                module:              &shader,
                entry_point:         Some("compute_shadow_matrices"),
                compilation_options: Default::default(),
                cache:               None,
            },
        ));

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Shadow Matrix Params"),
            size:               UNIFORM_SIZE,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            pipeline,
            bgl,
            bind_group: None,
            uniform_buf,
            last_camera_ptr: 0,
            last_light_ptr: 0,
            last_shadow_mat_ptr: 0,
            last_shadow_dirty_ptr: 0,
            last_shadow_hash_ptr: 0,
        })
    }

    /// Bind GPU resources. Call when buffers are created or reallocated.
    pub fn bind_resources(
        &mut self,
        device:         &wgpu::Device,
        light_buf:      &wgpu::Buffer,
        shadow_mat_buf: &wgpu::Buffer,
        camera_buf:     &wgpu::Buffer,
        shadow_dirty_buf: &wgpu::Buffer,
        shadow_hash_buf:  &wgpu::Buffer,
    ) {
        // Check if bind group needs rebuild
        let light_ptr      = light_buf as *const wgpu::Buffer as usize;
        let shadow_mat_ptr = shadow_mat_buf as *const wgpu::Buffer as usize;
        let camera_ptr     = camera_buf as *const wgpu::Buffer as usize;
        let dirty_ptr      = shadow_dirty_buf as *const wgpu::Buffer as usize;
        let hash_ptr       = shadow_hash_buf as *const wgpu::Buffer as usize;

        let stale = self.bind_group.is_none()
            || light_ptr      != self.last_light_ptr
            || shadow_mat_ptr != self.last_shadow_mat_ptr
            || camera_ptr     != self.last_camera_ptr
            || dirty_ptr      != self.last_shadow_dirty_ptr
            || hash_ptr       != self.last_shadow_hash_ptr;

        if !stale { return; }

        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Shadow Matrix BG"),
            layout:  &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: light_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: shadow_mat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  2,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  3,
                    resource: self.uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  4,
                    resource: shadow_dirty_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding:  5,
                    resource: shadow_hash_buf.as_entire_binding(),
                },
            ],
        }));

        self.last_light_ptr         = light_ptr;
        self.last_shadow_mat_ptr    = shadow_mat_ptr;
        self.last_camera_ptr        = camera_ptr;
        self.last_shadow_dirty_ptr  = dirty_ptr;
        self.last_shadow_hash_ptr   = hash_ptr;
    }

    /// Execute the shadow matrix computation.
    ///
    /// `camera_moved`: true if camera moved this frame (triggers CSM recompute for directional lights)
    /// `light_count`: number of active lights
    pub fn execute(
        &self,
        encoder:      &mut wgpu::CommandEncoder,
        queue:        &wgpu::Queue,
        camera_moved: bool,
        light_count:  u32,
    ) {
        let Some(bg) = &self.bind_group else { return; };
        if light_count == 0 { return; }

        let uniforms = ShadowMatrixParams {
            light_count,
            camera_moved: if camera_moved { 1 } else { 0 },
            _pad0: 0,
            _pad1: 0,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("Shadow Matrix Compute"),
            timestamp_writes: None,
        });

        cp.set_pipeline(&self.pipeline);
        cp.set_bind_group(0, bg, &[]);
        // 64 threads per workgroup (matches shader @workgroup_size(64))
        cp.dispatch_workgroups((light_count + 63) / 64, 1, 1);
    }
}
