//! Shadow depth pass - renders depth into the shadow atlas

use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};

/// Depth-only pass that fills shadow atlas layers
///
/// One render sub-pass per shadow-casting light. Each sub-pass clears and
/// re-draws all shadow casters into that light's atlas layer, using the
/// light-space view-proj matrix selected via `@builtin(instance_index)`.
pub struct ShadowPass {
    /// One view per shadow-casting light (D2, single array layer each)
    layer_views: Vec<Arc<wgpu::TextureView>>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    shadow_matrix_buffer: Arc<wgpu::Buffer>,
    /// Actual number of lights this frame (updated by Renderer before graph exec)
    light_count: Arc<AtomicU32>,
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
}

impl ShadowPass {
    pub fn new(
        layer_views: Vec<Arc<wgpu::TextureView>>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        shadow_matrix_buffer: Arc<wgpu::Buffer>,
        light_count: Arc<AtomicU32>,
        device: &wgpu::Device,
    ) -> Self {
        // Bind group layout: binding 0 = shadow matrices storage buffer (vertex stage)
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Matrix BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow Matrix Bind Group"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: shadow_matrix_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/shadow.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[wgpu::VertexBufferLayout {
                    // Matches PackedVertex stride; only position at offset 0 is used
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: None, // depth-only
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                // Peter-Pan trick: cull front faces to eliminate self-shadowing acne
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,      // constant units bias
                    slope_scale: 2.0, // slope-scale bias (eliminates acne at grazing angles)
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        Self {
            layer_views,
            draw_list,
            shadow_matrix_buffer,
            light_count,
            pipeline: Arc::new(pipeline),
            bind_group,
        }
    }
}

impl RenderPass for ShadowPass {
    fn name(&self) -> &str {
        "shadow"
    }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // This pass writes the shadow atlas; GeometryPass declares a read on it,
        // which forces shadow → geometry ordering via topological sort.
        builder.write(ResourceHandle::named("shadow_atlas"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let light_count = self.light_count.load(Ordering::Relaxed) as usize;
        // Each light occupies 6 consecutive layers
        let actual_count = light_count.min(self.layer_views.len() / 6);

        if actual_count == 0 {
            return Ok(());
        }

        let draw_calls: Vec<DrawCall> = self.draw_list.lock().unwrap().clone();

        for i in 0..actual_count {
            for face in 0u32..6u32 {
                let layer_idx = i as u32 * 6 + face;

                let mut pass = ctx.begin_render_pass(
                    &format!("Shadow Light {i} Face {face}"),
                    &[],
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.layer_views[layer_idx as usize],
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                );

                if draw_calls.is_empty() {
                    continue;
                }

                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);

                for dc in &draw_calls {
                    pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                    pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    // instance_index = layer_idx selects the correct face matrix in the vertex shader
                    pass.draw_indexed(0..dc.index_count, 0, layer_idx..(layer_idx + 1));
                }
            }
        }

        Ok(())
    }
}

