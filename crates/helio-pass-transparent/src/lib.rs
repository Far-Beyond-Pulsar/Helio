//! Transparent geometry pass.
//!
//! Renders alpha-blended transparent geometry using `multi_draw_indexed_indirect`.
//! The pass shares the same Group 0 binding layout (camera / globals / instances) as the
//! opaque geometry pass, but enables `SrcAlpha / OneMinusSrcAlpha` blending and uses a
//! read-only depth attachment so transparent surfaces sort correctly against opaque ones.
//!
//! ## O(1) CPU cost
//! `execute()` issues a single `multi_draw_indexed_indirect` call regardless of scene size.
//!
//! ## Note on prepare()
//! `prepare()` uploads per-frame globals (frame counter, light count).  In a real renderer
//! the CPU-side depth sort of transparent instances would also happen here — that is an
//! intentional O(n) step documented as unavoidable for correct alpha-blending.
//! A future OIT (Order-Independent Transparency) implementation would eliminate this sort.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GBufferGlobals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],
    rc_world_min: [f32; 4],
    rc_world_max: [f32; 4],
    csm_splits: [f32; 4],
}

pub struct TransparentPass {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    globals_buf: wgpu::Buffer,
}

impl TransparentPass {
    /// Create the transparent pass.
    ///
    /// `camera_buf`    — the per-frame camera uniform buffer (shared with opaque passes).
    /// `instances_buf` — the GPU instance storage buffer (shared with the scene).
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        instances_buf: &wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Transparent Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/transparent.wgsl").into()),
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Transparent Globals"),
            size: std::mem::size_of::<GBufferGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Transparent BGL"),
            entries: &[
                // 0: camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: globals uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: instance_data storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Transparent BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instances_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Transparent PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        // Vertex layout must match `Vertex` struct in transparent.wgsl:
        //   location 0: position    (vec3<f32>)
        //   location 1: bitangent_sign (f32)
        //   location 2: tex_coords  (vec2<f32>)
        //   location 3: normal      (u32, packed snorm8x4)
        //   location 4: tangent     (u32, packed snorm8x4)
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: (3 + 1 + 2) * 4 + 2 * 4, // 32 bytes
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 16,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 24,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32,
                    offset: 28,
                    shader_location: 4,
                },
            ],
        };

        let alpha_blend = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::SrcAlpha,
                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::OVER,
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Transparent Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_buffer_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    // Caller's HDR or final colour target; Load to preserve opaque geometry.
                    format: wgpu::TextureFormat::Rgba16Float,
                    blend: Some(alpha_blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // Transparent objects may need both faces
                ..Default::default()
            },
            // Read-only depth: transparent objects test against opaque depth but don't write it.
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            globals_buf,
        }
    }
}

impl RenderPass for TransparentPass {
    fn name(&self) -> &'static str {
        "Transparent"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let globals = GBufferGlobals {
            frame: ctx.frame_num as u32,
            delta_time: 0.0,
            light_count: ctx.scene.lights.len() as u32,
            ambient_intensity: 0.1,
            ambient_color: [0.1, 0.1, 0.15, 1.0],
            rc_world_min: [0.0; 4],
            rc_world_max: [0.0; 4],
            csm_splits: [0.0; 4],
        };
        ctx.queue
            .write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let draw_count = ctx.scene.draw_count;
        if draw_count == 0 {
            return Ok(());
        }
        let main_scene = ctx.resources.main_scene.as_ref().ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig(
                "TransparentPass requires main_scene mesh buffers".to_string(),
            )
        })?;
        let indirect = ctx.scene.indirect;

        // Load existing colour (preserves opaque geometry rendered earlier).
        // Depth is read-only (depth_write_enabled = false in pipeline state).
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: ctx.target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        })];
        let depth_stencil = wgpu::RenderPassDepthStencilAttachment {
            view: ctx.depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        };
        let desc = wgpu::RenderPassDescriptor {
            label: Some("Transparent"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: Some(depth_stencil),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut pass = ctx.begin_render_pass(&desc);

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, main_scene.mesh_buffers.vertices.slice(..));
        pass.set_index_buffer(
            main_scene.mesh_buffers.indices.slice(..),
            wgpu::IndexFormat::Uint32,
        );
        #[cfg(not(target_arch = "wasm32"))]
        pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
        #[cfg(target_arch = "wasm32")]
        for i in 0..draw_count {
            pass.draw_indexed_indirect(indirect, i as u64 * 20);
        }

        Ok(())
    }
}
