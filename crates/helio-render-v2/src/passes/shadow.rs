//! Shadow depth pass - renders depth into the shadow atlas

use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::DrawCall;
use crate::Result;
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};
use wgpu::util::DeviceExt;

/// Per-light data written by the `Renderer` each frame and read by `ShadowPass::execute`
/// to skip draw calls that cannot contribute to a given shadow face.
///
/// Culling strategy: **Aggressive shadow rendering**
/// 1. **Range cull** – extended to 5x for point/spot lights (instead of 1x) to fill
///    the atlas with quality shadows for visible objects.
/// 2. **Hemisphere cull** (point lights only) – skip the mesh if it lies entirely
///    in the hemisphere opposite the cube face being rendered.
#[derive(Clone, Copy, Default)]
pub struct ShadowCullLight {
    pub position:       [f32; 3],
    pub direction:      [f32; 3],
    pub range:          f32,
    pub is_directional: bool,
    pub is_point:       bool,
    /// FNV-1a hash of this light's computed shadow matrix/matrices.
    pub matrix_hash:    u64,
}

/// FNV-1a 64-bit hash over a slice of u64 values.
fn fnv64(vals: &[u64]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &v in vals {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

#[inline]
fn fnv64_push(mut h: u64, v: u64) -> u64 {
    h ^= v;
    h.wrapping_mul(0x100000001b3)
}

// ── Per-face frustum culling for point-light cubemaps ────────────────────────

const CUBE_FACE_PLANES: [[[f32; 3]; 4]; 6] = [
    [[ 1.0, 0.0,-1.0], [ 1.0, 0.0, 1.0], [ 1.0,-1.0, 0.0], [ 1.0, 1.0, 0.0]],
    [[-1.0, 0.0,-1.0], [-1.0, 0.0, 1.0], [-1.0,-1.0, 0.0], [-1.0, 1.0, 0.0]],
    [[ 0.0, 1.0,-1.0], [ 0.0, 1.0, 1.0], [-1.0, 1.0, 0.0], [ 1.0, 1.0, 0.0]],
    [[ 0.0,-1.0,-1.0], [ 0.0,-1.0, 1.0], [-1.0,-1.0, 0.0], [ 1.0,-1.0, 0.0]],
    [[-1.0, 0.0, 1.0], [ 1.0, 0.0, 1.0], [ 0.0,-1.0, 1.0], [ 0.0, 1.0, 1.0]],
    [[-1.0, 0.0,-1.0], [ 1.0, 0.0,-1.0], [ 0.0,-1.0,-1.0], [ 0.0, 1.0,-1.0]],
];

#[inline]
fn sphere_in_cube_face(delta: [f32; 3], radius: f32, face: u32) -> bool {
    let planes = &CUBE_FACE_PLANES[face as usize];
    let threshold = -radius * std::f32::consts::SQRT_2;
    for plane in planes {
        let d = delta[0] * plane[0] + delta[1] * plane[1] + delta[2] * plane[2];
        if d < threshold { return false; }
    }
    true
}

/// Per-light shadow cache entry.
#[derive(Clone, Copy, Default)]
struct ShadowLightCache {
    light_hash: u64,
    geom_hash:  u64,
    valid:      bool,
}

pub struct ShadowPass {
    device: Arc<wgpu::Device>,
    layer_views: Vec<Arc<wgpu::TextureView>>,
    draw_list: Arc<Mutex<Vec<DrawCall>>>,
    shadow_matrix_buffer: Arc<wgpu::Buffer>,
    light_count: Arc<AtomicU32>,
    light_face_counts: Arc<Mutex<Vec<u8>>>,
    cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>,
    pipeline: Arc<wgpu::RenderPipeline>,
    /// Per-face (light_idx*6+face) bind group.
    slot_bind_groups: Vec<wgpu::BindGroup>,
    slot_idx_buffers: Vec<wgpu::Buffer>,
    shadow_cache: Vec<ShadowLightCache>,
    filtered_indices: Vec<usize>,
    // Pool VB/IB for unified geometry (used when meshes are pool-allocated).
    pool_vertex_buffer: Arc<wgpu::Buffer>,
    pool_index_buffer:  Arc<wgpu::Buffer>,
}

impl ShadowPass {
    pub fn new(
        light_face_counts: Arc<Mutex<Vec<u8>>>,
        cull_lights: Arc<Mutex<Vec<ShadowCullLight>>>,
        layer_views: Vec<Arc<wgpu::TextureView>>,
        draw_list: Arc<Mutex<Vec<DrawCall>>>,
        shadow_matrix_buffer: Arc<wgpu::Buffer>,
        light_count: Arc<AtomicU32>,
        device: Arc<wgpu::Device>,
        material_layout: &wgpu::BindGroupLayout,
        instance_data_buffer: &wgpu::Buffer,
        pool_vertex_buffer: Arc<wgpu::Buffer>,
        pool_index_buffer: Arc<wgpu::Buffer>,
    ) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow Matrix BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/shadow.wgsl").into(),
            ),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl), Some(material_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            cache: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 16,
                                shader_location: 2,
                            },
                        ],
                    },
                ],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 1.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
        });

        let max_lights = layer_views.len() / 6;
        let total_slots = max_lights * 6;

        let mut slot_idx_buffers = Vec::with_capacity(total_slots);
        let mut slot_bind_groups = Vec::with_capacity(total_slots);
        for slot in 0..total_slots {
            let idx_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("shadow_layer_idx"),
                contents: &(slot as u32).to_le_bytes(),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow Slot BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: shadow_matrix_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: idx_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: instance_data_buffer.as_entire_binding(),
                    },
                ],
            });
            slot_idx_buffers.push(idx_buf);
            slot_bind_groups.push(bg);
        }

        Self {
            device,
            layer_views,
            draw_list,
            shadow_matrix_buffer,
            light_count,
            light_face_counts,
            cull_lights,
            pipeline: Arc::new(pipeline),
            slot_bind_groups,
            slot_idx_buffers,
            shadow_cache: vec![ShadowLightCache::default(); max_lights],
            filtered_indices: Vec::new(),
            pool_vertex_buffer,
            pool_index_buffer,
        }
    }
}

impl RenderPass for ShadowPass {
    fn name(&self) -> &str { "shadow" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        builder.write(ResourceHandle::named("shadow_atlas"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let light_count  = self.light_count.load(Ordering::Relaxed) as usize;
        let actual_count = light_count.min(self.layer_views.len() / 6);
        if actual_count == 0 { return Ok(()); }

        let draw_calls  = self.draw_list.lock().unwrap();
        let face_counts = self.light_face_counts.lock().unwrap();
        let cull_lights = self.cull_lights.lock().unwrap();

        const SHADOW_MAX_DISTANCE: f32 = 300.0;

        // ── Camera-distance filter ───────────────────────────────────────────
        self.filtered_indices.clear();
        self.filtered_indices.reserve(draw_calls.len());
        for (idx, dc) in draw_calls.iter().enumerate() {
            let dist = (glam::Vec3::from(dc.bounds_center) - ctx.camera_position).length();
            if dist <= SHADOW_MAX_DISTANCE {
                self.filtered_indices.push(idx);
            }
        }

        // ── Geometry hash (structural identity of visible caster set) ────────
        let geom_hash = {
            let mut h: u64 = 0xcbf29ce484222325;
            h = fnv64_push(h, self.filtered_indices.len() as u64);
            for &idx in &self.filtered_indices {
                h = fnv64_push(h, draw_calls[idx].slot as u64);
            }
            h
        };

        // ═══ Per-light render ════════════════════════════════════════════════
        for i in 0..actual_count {
            let max_faces = face_counts.get(i).copied().unwrap_or(6) as u32;
            let light = cull_lights.get(i).copied().unwrap_or_default();

            let light_hash = fnv64(&[
                light.matrix_hash,
                light.range.to_bits() as u64,
                light.is_directional as u64,
                light.is_point as u64,
                max_faces as u64,
            ]);

            // ── Cache check: skip GPU work if light + geometry unchanged ──────
            {
                let cached = &self.shadow_cache[i];
                if cached.valid
                    && cached.light_hash == light_hash
                    && cached.geom_hash  == geom_hash
                {
                    continue;
                }
            }

            let t_light = ctx.scope_begin(&format!("shadow/light_{i}"));

            for face in 0u32..max_faces {
                let layer_idx  = i as u32 * 6 + face;
                let bundle_slot = i * 6 + face as usize;

                let t_face = ctx.scope_begin(&format!("shadow/light_{i}/face_{face}"));

                // Per-face frustum filter for point lights.
                let face_indices: Vec<usize> = if light.is_point {
                    self.filtered_indices.iter().copied().filter(|&idx| {
                        let dc = &draw_calls[idx];
                        let delta = [
                            dc.bounds_center[0] - light.position[0],
                            dc.bounds_center[1] - light.position[1],
                            dc.bounds_center[2] - light.position[2],
                        ];
                        sphere_in_cube_face(delta, dc.bounds_radius, face)
                    }).collect()
                } else {
                    self.filtered_indices.clone()
                };

                let mut pass = ctx.begin_render_pass(
                    &format!("Shadow Light {i} Face {face}"),
                    &[],
                    Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &self.layer_views[layer_idx as usize],
                        depth_ops: Some(wgpu::Operations {
                            load:  wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                );

                if !face_indices.is_empty() {
                    pass.set_pipeline(&self.pipeline);
                    pass.set_bind_group(0, &self.slot_bind_groups[bundle_slot], &[]);
                    pass.set_vertex_buffer(0, self.pool_vertex_buffer.slice(..));
                    pass.set_index_buffer(self.pool_index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                    let mut last_mat: Option<usize> = None;
                    for &idx in &face_indices {
                        let dc = &draw_calls[idx];
                        let mat_ptr = Arc::as_ptr(&dc.material_bind_group) as usize;
                        if last_mat != Some(mat_ptr) {
                            pass.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                            last_mat = Some(mat_ptr);
                        }
                        pass.draw_indexed(
                            dc.pool_first_index..dc.pool_first_index + dc.index_count,
                            dc.pool_base_vertex,
                            dc.slot..dc.slot + 1,
                        );
                    }
                }

                drop(pass);
                ctx.scope_end(t_face);
            }

            ctx.scope_end(t_light);
            self.shadow_cache[i] = ShadowLightCache { light_hash, geom_hash, valid: true };
        }

        Ok(())
    }
}


