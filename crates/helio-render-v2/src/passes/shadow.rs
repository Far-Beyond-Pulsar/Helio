//! Shadow depth pass - renders depth into the shadow atlas

use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};

/// Per-light data written by the `Renderer` each frame and read by `ShadowPass::execute`
/// to skip draw calls that cannot contribute to a given shadow face.
///
/// Two cull stages are applied:
/// 1. **Range cull** – skip the mesh if its bounding sphere does not intersect
///    the light's influence sphere (`dist(center, light_pos) - radius > range`).
/// 2. **Hemisphere cull** (point lights only) – skip the mesh if it lies entirely
///    in the hemisphere opposite the cube face being rendered.
#[derive(Clone, Copy, Default)]
pub struct ShadowCullLight {
    /// World-space position (ignored for directional lights).
    pub position:       [f32; 3],
    /// Maximum influence radius in metres.
    pub range:          f32,
    /// Directional lights use ortho CSM covering the whole scene — skip all culling.
    pub is_directional: bool,
    /// Point lights get the per-face hemisphere cull in addition to the range cull.
    pub is_point:       bool,
}

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
    /// Active face count per light: 6=point, 4=directional (CSM), 1=spot.
    /// Skipping unused faces avoids rendering geometry into identity-matrix layers.
    light_face_counts: Arc<Mutex<Vec<u8>>>,
    /// Per-light position + range + type for CPU draw-call culling.
    cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>,
    pipeline: Arc<wgpu::RenderPipeline>,
    bind_group: wgpu::BindGroup,
}

impl ShadowPass {
    pub fn new(
        light_face_counts: Arc<Mutex<Vec<u8>>>,
        cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>,
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
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
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
                // Back-face culling: render front faces to shadow map (standard approach)
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                // Depth bias prevents self-shadowing (shadow acne)
                // slope_scale adds bias based on surface angle relative to light
                // constant offset adds fixed bias regardless of angle
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.5,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        Self {
            layer_views,
            draw_list,
            shadow_matrix_buffer,
            light_count,
            light_face_counts,
            cull_lights,
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
        let face_counts  = self.light_face_counts.lock().unwrap();
        let cull_lights  = self.cull_lights.lock().unwrap();

        for i in 0..actual_count {
            // Only render faces that have valid (non-identity) matrices:
            //   point light  → 6 cube faces
            //   directional  → 4 CSM cascades (slots 0-3)
            //   spot light   → 1 projection   (slot 0)
            let max_faces = face_counts.get(i).copied().unwrap_or(6) as u32;

            // ── Stage 1: range-sphere cull (once per light) ─────────────────────
            // Directional lights use ortho cascades covering the whole scene —
            // skip the test so we never accidentally cull a distant shadow caster.
            let cull = cull_lights.get(i).copied().unwrap_or_default();
            let range_filtered: Vec<&DrawCall> = if cull.is_directional {
                draw_calls.iter().collect()
            } else {
                draw_calls.iter().filter(|dc| {
                    let dx = dc.bounds_center[0] - cull.position[0];
                    let dy = dc.bounds_center[1] - cull.position[1];
                    let dz = dc.bounds_center[2] - cull.position[2];
                    let dist = (dx * dx + dy * dy + dz * dz).sqrt();
                    dist - dc.bounds_radius <= cull.range
                }).collect()
            };

            let t_light = ctx.scope_begin(&format!("shadow/light_{i}"));
            for face in 0u32..max_faces {
                let layer_idx = i as u32 * 6 + face;

                // ── Stage 2: per-face hemisphere cull (point lights only) ────────
                // Face ordering matches compute_point_light_matrices in renderer.rs:
                //   face 0 = +X,  face 1 = −1X
                //   face 2 = +Y,  face 3 = −1Y
                //   face 4 = +Z,  face 5 = −1Z
                // A mesh entirely behind the axis of this face can never appear
                // in its depth map, so skip it.
                let face_draws: Vec<&DrawCall> = if cull.is_point {
                    let axis = (face / 2) as usize; // 0=X, 1=Y, 2=Z
                    let sign = if face % 2 == 0 { 1.0f32 } else { -1.0 };
                    range_filtered.iter().copied().filter(|dc| {
                        let offset = (dc.bounds_center[axis] - cull.position[axis]) * sign;
                        offset + dc.bounds_radius >= 0.0
                    }).collect()
                } else {
                    range_filtered.iter().copied().collect()
                };

                let t_face = ctx.scope_begin(&format!("shadow/light_{i}/face_{face}"));
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

                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);

                for dc in face_draws {
                    pass.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                    pass.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    // instance_index = layer_idx selects the correct face matrix in the vertex shader
                    pass.draw_indexed(0..dc.index_count, 0, layer_idx..(layer_idx + 1));
                }
                drop(pass); // end render pass before writing end timestamp
                ctx.scope_end(t_face);
            }
            ctx.scope_end(t_light);
        }

        Ok(())
    }
}

