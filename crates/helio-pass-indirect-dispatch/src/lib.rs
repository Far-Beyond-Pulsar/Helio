//! GPU frustum culling and indirect draw command generation.
//!
//! This pass runs a compute shader that:
//! 1. Tests each instance's bounding sphere against the 6 frustum planes
//! 2. Writes DrawIndexedIndirect commands (instance_count=1 for visible, 0 for culled)
//! 3. Is O(1) CPU cost — single compute dispatch regardless of scene size
//!
//! Non-compacting design: culled draws get instance_count=0.
//! This means the indirect buffer stays the same size as the draw call list.

use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};
use bytemuck::{Pod, Zeroable};

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
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl IndirectDispatchPass {
    pub fn new(
        device: &wgpu::Device,
        scene_instances: &wgpu::Buffer,
        scene_draw_calls: &wgpu::Buffer,
        indirect_buf: &wgpu::Buffer,
        camera_buf: &wgpu::Buffer,
    ) -> Self {
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

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    // binding 4: indirect output (write)
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
            label: Some("IndirectDispatch BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: scene_instances.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: scene_draw_calls.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: indirect_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("IndirectDispatch PL"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
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
            bind_group,
        }
    }
}

impl RenderPass for IndirectDispatchPass {
    fn name(&self) -> &'static str {
        "IndirectDispatch"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // TODO: Extract frustum planes from camera view-proj when PrepareContext exposes scene.
        // Frustum plane extraction is a fixed 6 dot products, O(1).
        let uniforms = CullUniforms {
            frustum_planes: [[0.0; 4]; 6], // TODO: extract_frustum_planes(ctx.scene.camera_view_proj())
            draw_count: 0,                 // TODO: ctx.scene.draw_calls.len() as u32
            _pad: [0; 3],
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single compute dispatch — GPU does all culling work
        let draw_count = 0u32; // TODO: ctx.scene.draw_count when SceneResources has this field
        if draw_count == 0 {
            return Ok(());
        }
        let workgroups = draw_count.div_ceil(WORKGROUP_SIZE);
        let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("IndirectDispatch"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.dispatch_workgroups(workgroups, 1, 1);
        Ok(())
    }
}

/// Extract 6 frustum planes from a view-projection matrix (Gribb/Hartmann method).
#[allow(dead_code)]
fn extract_frustum_planes(vp: [[f32; 4]; 4]) -> [[f32; 4]; 6] {
    let m = vp;
    let row = |i: usize| [m[0][i], m[1][i], m[2][i], m[3][i]];
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    let add = |a: [f32; 4], b: [f32; 4]| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]];
    let sub = |a: [f32; 4], b: [f32; 4]| [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]];
    [
        add(r3, r0), // left
        sub(r3, r0), // right
        add(r3, r1), // bottom
        sub(r3, r1), // top
        add(r3, r2), // near
        sub(r3, r2), // far
    ]
}
