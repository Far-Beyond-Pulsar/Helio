//! GPU frustum culling and indirect draw command generation.
//!
//! This pass runs a compute shader that:
//! 1. Tests each instance's bounding sphere against the 6 frustum planes
//! 2. Writes DrawIndexedIndirect commands (instance_count=1 for visible, 0 for culled)
//! 3. Is O(1) CPU cost — single compute dispatch regardless of scene size
//!
//! Non-compacting design: culled draws get instance_count=0.
//! This means the indirect buffer stays the same size as the draw call list.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniforms {
    frustum_planes: [[f32; 4]; 6], // 6 planes × 4 floats = 96 bytes
    draw_count: u32,
    _pad: [u32; 3],
}

pub struct IndirectDispatchPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    cull_stats_buf: wgpu::Buffer,
    /// Lazy bind group — rebuilt whenever the underlying buffer pointers change
    /// (GrowableBuffers reallocate on resize, invalidating old bind groups).
    bind_group: Option<wgpu::BindGroup>,
    /// Tuple of raw buffer pointers used as a staleness key.
    bind_group_key: Option<(usize, usize, usize, usize, usize, usize)>,
    /// Draw count uploaded in `prepare()`, used in `execute()`.
    draw_count: u32,
}

impl IndirectDispatchPass {
    pub fn new(device: &wgpu::Device, cull_stats_buf: wgpu::Buffer) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("IndirectDispatch Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/indirect_dispatch.wgsl").into(),
            ),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CullUniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("IndirectDispatch BGL"),
            entries: &[
                // binding 0: camera uniforms
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
                // binding 1: cull uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: instances (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: draw calls (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: AABBs (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: indirect output (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 6: culling stats (read_write, atomic counters)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("IndirectDispatch PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("IndirectDispatch Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buf,
            cull_stats_buf: cull_stats_buf.clone(),
            bind_group: None,
            bind_group_key: None,
            draw_count: 0,
        }
    }
}

impl RenderPass for IndirectDispatchPass {
    fn name(&self) -> &'static str {
        "IndirectDispatch"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let draw_count = ctx.scene.draw_calls.len() as u32;
        self.draw_count = draw_count;
        let planes = extract_frustum_planes(ctx.scene.camera.data().view_proj);

        let uniforms = CullUniforms {
            frustum_planes: planes,
            draw_count,
            _pad: [0; 3],
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let draw_count = ctx.scene.draw_count;
        if draw_count == 0 {
            return Ok(());
        }

        // Rebuild bind group if any GrowableBuffer has reallocated (pointer changed).
        let key = (
            ctx.scene.camera as *const wgpu::Buffer as usize,
            ctx.scene.instances as *const wgpu::Buffer as usize,
            ctx.scene.draw_calls as *const wgpu::Buffer as usize,
            ctx.scene.aabbs as *const wgpu::Buffer as usize,
            ctx.scene.indirect as *const wgpu::Buffer as usize,
            &self.cull_stats_buf as *const wgpu::Buffer as usize,
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("IndirectDispatch BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.scene.draw_calls.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.scene.aabbs.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.scene.indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.cull_stats_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        // O(1) CPU: one dispatch, GPU culls all draw calls in parallel.
        let workgroups = draw_count.div_ceil(WORKGROUP_SIZE);
        let mut pass = unsafe { &mut *ctx.encoder_ptr }
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("IndirectDispatch"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}

/// Extract 6 frustum planes from a view-projection matrix (Gribb/Hartmann method).
///
/// `vp` is a flat column-major `[f32; 16]` (from `glam::Mat4::to_cols_array()`).
/// Returns **normalized** planes `(nx, ny, nz, d)` where a point P is *inside* the
/// frustum when `dot(n.xyz, P) + d >= 0`.
///
/// Normalization is essential because the GPU shader tests `dot(n.xyz, P) + d >= -r`
/// and if the normal is NOT unit-length, the sphere radius r is not comparable to
/// the raw signed distance — leading to both false-positive and false-negative culling.
///
/// Uses wgpu/DirectX depth conventions where NDC z ∈ [0, 1].
fn extract_frustum_planes(vp: [f32; 16]) -> [[f32; 4]; 6] {
    // Extract matrix rows: vp[col*4 + row], so row r has elements [vp[r], vp[4+r], vp[8+r], vp[12+r]].
    let row = |r: usize| -> [f32; 4] { [vp[r], vp[4 + r], vp[8 + r], vp[12 + r]] };
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    };
    let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    };
    let normalize = |p: [f32; 4]| -> [f32; 4] {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if len > 1e-10 {
            [p[0] / len, p[1] / len, p[2] / len, p[3] / len]
        } else {
            p
        }
    };
    [
        normalize(add(r3, r0)), // left:   -w ≤  x  →  x + w ≥ 0
        normalize(sub(r3, r0)), // right:   x ≤  w  → -x + w ≥ 0
        normalize(add(r3, r1)), // bottom: -w ≤  y  →  y + w ≥ 0
        normalize(sub(r3, r1)), // top:     y ≤  w  → -y + w ≥ 0
        normalize(r2),          // near:    z ≥  0  (wgpu NDC z ∈ [0,1], not OpenGL's -w ≤ z)
        normalize(sub(r3, r2)), // far:     z ≤  w  → -z + w ≥ 0
    ]
}

