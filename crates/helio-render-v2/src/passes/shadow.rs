//! Shadow depth pass - renders depth into the shadow atlas

use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::mesh::{DrawCall, INSTANCE_STRIDE};
use crate::Result;
use std::sync::{Arc, Mutex, atomic::{AtomicU32, Ordering}};
use wgpu::util::DeviceExt;

/// Per-light data written by the `Renderer` each frame and read by `ShadowPass::execute`
/// to skip draw calls that cannot contribute to a given shadow face.
///
/// Culling strategy: **Aggressive shadow rendering**
/// 1. **Range cull** – extended to 5x for point/spot lights (instead of 1x) to fill
///    the atlas with quality shadows for visible objects. The wider range ensures
///    that objects casting shadows on camera-visible surfaces are included, even
///    if their light source has a modest range.
/// 2. **Hemisphere cull** (point lights only) – skip the mesh if it lies entirely
///    in the hemisphere opposite the cube face being rendered.
#[derive(Clone, Copy, Default)]
pub struct ShadowCullLight {
    /// World-space position (ignored for directional lights).
    pub position:       [f32; 3],
    /// World-space direction (used for directional/spot light cache key).
    pub direction:      [f32; 3],
    /// Maximum influence radius in metres (extended to 5x for point/spot lights).
    pub range:          f32,
    /// Directional lights use ortho CSM covering the whole scene — skip all culling.
    pub is_directional: bool,
    /// Point lights get the per-face hemisphere cull in addition to the range cull.
    pub is_point:       bool,
    /// FNV-1a hash of this light's computed shadow matrix/matrices.
    /// Using the post-snap matrix hash means the cache only invalidates when the
    /// shadow output actually changes (e.g. camera crosses a texel boundary for CSM).
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

/// Per-light shadow cache entry. Stores hashes of the light state and visible
/// geometry; if both match the previous frame the shadow faces are skipped.
#[derive(Clone, Copy, Default)]
struct ShadowLightCache {
    light_hash: u64,
    geom_hash:  u64,
    valid:      bool,
}

/// Depth-only pass that fills shadow atlas layers
///
/// One render sub-pass per shadow-casting light. Each sub-pass clears and
/// re-draws all shadow casters into that light's atlas layer, using the
/// light-space view-proj matrix selected via `@builtin(instance_index)`.
pub struct ShadowPass {
    device: Arc<wgpu::Device>,
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
    /// Per-face (light_idx*6+face) bind group. Each contains the shadow matrix storage buffer
    /// at binding 0 and a tiny uniform buffer holding the layer index at binding 1.
    /// This avoids needing push-constant (immediate) device support.
    slot_bind_groups: Vec<wgpu::BindGroup>,
    /// Backing u32 uniform buffers for `slot_bind_groups` — kept alive alongside the bind groups.
    slot_idx_buffers: Vec<wgpu::Buffer>,
    /// Per-light shadow cache — skip re-rendering when light and geometry are unchanged.
    shadow_cache: Vec<ShadowLightCache>,
    filtered_indices: Vec<usize>,
    /// Per-face (light_idx*6+face) cached RenderBundle.  Rebuilt only when shadow-caster
    /// geometry changes; light-matrix updates (CSM recalculation on camera rotation) are
    /// handled by the GPU shadow-matrix buffer written each frame in `renderer.rs`, so the
    /// bundle is replayed without re-encoding even when matrices change.
    bundle_cache: Vec<Option<wgpu::RenderBundle>>,
    /// Geometry hash when each light's bundles were last built.  `u64::MAX` = never built.
    bundle_geom_hashes: Vec<u64>,
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
    ) -> Self {
        // BGL binding 0: shadow matrices storage; binding 1: per-face layer index uniform.
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
                        // Matches PackedVertex stride
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &[
                            // position: vec3<f32>
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x3,
                                offset: 0,
                                shader_location: 0,
                            },
                            // tex_coords: vec2<f32>
                            wgpu::VertexAttribute {
                                format: wgpu::VertexFormat::Float32x2,
                                offset: 16,
                                shader_location: 2,
                            },
                        ],
                    },
                    // Instance model matrix (four vec4 columns, locations 5-8)
                    wgpu::VertexBufferLayout {
                        array_stride: 64,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset:  0, shader_location: 5 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 6 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 7 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 48, shader_location: 8 },
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
                // Two-sided shadow casters: avoids missing shadows on meshes with
                // mixed/inconsistent winding and matches a conservative "never miss
                // a caster" policy for indoor scenes.
                cull_mode: None,
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

        // Pre-bake one bind group per atlas layer so each face's RenderBundle can
        // embed the correct layer index without push-constant support.
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
            bundle_cache: (0..total_slots).map(|_| None).collect(),
            bundle_geom_hashes: vec![u64::MAX; max_lights],
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

        let draw_calls = self.draw_list.lock().unwrap();
        let face_counts  = self.light_face_counts.lock().unwrap();
        let cull_lights  = self.cull_lights.lock().unwrap();

        // Distance from camera at which shadows stop being rendered (in meters)
        const SHADOW_MAX_DISTANCE: f32 = 300.0;

        // ── Camera-distance filter (shared across all lights) ────────────────
        // Compute once outside the light loop; each light renders the same set.
        // Transparent objects now cast alpha-tested shadows.
        self.filtered_indices.clear();
        self.filtered_indices.reserve(draw_calls.len());
        for (idx, dc) in draw_calls.iter().enumerate() {
                let dist = (glam::Vec3::from(dc.bounds_center) - ctx.camera_position).length();
                if dist <= SHADOW_MAX_DISTANCE {
                    self.filtered_indices.push(idx);
                }
            }

        // ── Geometry hash (shared) ───────────────────────────────────────────
        // Hashes world-space identity + bounds of every camera-visible caster.
        // Changes when objects are added/removed/moved within shadow distance.
        let geom_hash = {
            let mut h: u64 = 0xcbf29ce484222325;
            h = fnv64_push(h, self.filtered_indices.len() as u64);
            for &idx in &self.filtered_indices {
                let dc = &draw_calls[idx];
                h = fnv64_push(h, dc.bounds_center[0].to_bits() as u64);
                h = fnv64_push(h, dc.bounds_center[1].to_bits() as u64);
                h = fnv64_push(h, dc.bounds_center[2].to_bits() as u64);
                h = fnv64_push(h, dc.bounds_radius.to_bits() as u64);
                h = fnv64_push(h, Arc::as_ptr(&dc.vertex_buffer) as u64);
            }
            h
        };

        for i in 0..actual_count {
            // Only render faces that have valid (non-identity) matrices:
            //   point light  → 6 cube faces
            //   directional  → 4 CSM cascades (slots 0-3)
            //   spot light   → 1 projection   (slot 0)
            let max_faces = face_counts.get(i).copied().unwrap_or(6) as u32;
            let light = cull_lights.get(i).copied().unwrap_or_default();

            // ── Light hash ───────────────────────────────────────────────────
            // Based on the pre-computed shadow matrix hash rather than raw camera
            // position. CSM matrices are texel-snapped, so this hash only changes
            // when the shadow output actually differs (camera crossed a texel
            // boundary), not on every sub-texel camera movement.
            let light_hash = fnv64(&[
                light.matrix_hash,
                light.range.to_bits() as u64,
                light.is_directional as u64,
                light.is_point as u64,
                max_faces as u64,
            ]);

            // ── Cache check ──────────────────────────────────────────────────
            // Skip all face renders if both the light parameters and the visible
            // geometry are identical to the previous rendered frame.
            {
                let cached = &self.shadow_cache[i];
                if cached.valid && cached.light_hash == light_hash && cached.geom_hash == geom_hash {
                    continue;
                }
            }

            // Geometry unchanged → replay cached bundles; shadow matrices are updated
            // via write_buffer in renderer.rs every frame so the GPU reads fresh values.
            // Only rebuild bundles when casters are added/removed/moved.
            let need_bundle_rebuild = geom_hash != self.bundle_geom_hashes[i];

            let t_light = ctx.scope_begin(&format!("shadow/light_{i}"));
            for face in 0u32..max_faces {
                let layer_idx = i as u32 * 6 + face;
                let bundle_slot = i * 6 + face as usize;

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

                if need_bundle_rebuild {
                    // Re-encode all draw calls into a new bundle for this face.
                    let mut enc = self.device.create_render_bundle_encoder(
                        &wgpu::RenderBundleEncoderDescriptor {
                            label: Some("shadow_bundle"),
                            color_formats: &[],
                            depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                                format: wgpu::TextureFormat::Depth32Float,
                                depth_read_only: false,
                                stencil_read_only: true,
                            }),
                            sample_count: 1,
                            multiview: None,
                        },
                    );
                    enc.set_pipeline(&self.pipeline);
                    // Each slot_bind_group has the layer index baked in at binding 1,
                    // so replaying this bundle always selects the correct shadow matrix.
                    enc.set_bind_group(0, &self.slot_bind_groups[bundle_slot], &[]);
                    for &idx in &self.filtered_indices {
                        let dc = &draw_calls[idx];
                        enc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                        enc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                        let inst_start = dc.instance_buffer_offset;
                        let inst_end   = inst_start + dc.instance_count as u64 * INSTANCE_STRIDE;
                        enc.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(inst_start..inst_end));
                        enc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        enc.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
                    }
                    let bundle = enc.finish(&wgpu::RenderBundleDescriptor { label: None });
                    pass.execute_bundles(std::iter::once(&bundle));
                    self.bundle_cache[bundle_slot] = Some(bundle);
                } else if let Some(bundle) = &self.bundle_cache[bundle_slot] {
                    // Geometry stable: replay the cached bundle.  The shadow-matrix GPU
                    // buffer was already updated this frame via write_buffer so the vertex
                    // shader reads the correct (post-rotation) matrices automatically.
                    pass.execute_bundles(std::iter::once(bundle));
                }

                drop(pass); // end render pass before writing end timestamp
                ctx.scope_end(t_face);
            }
            ctx.scope_end(t_light);

            if need_bundle_rebuild {
                self.bundle_geom_hashes[i] = geom_hash;
            }

            // Update cache now that this light's shadow maps are fresh.
            self.shadow_cache[i] = ShadowLightCache { light_hash, geom_hash, valid: true };
        }

        Ok(())
    }
}

