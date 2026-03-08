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
    /// Keeps `Arc<wgpu::Buffer>` clones for every instance buffer encoded into each slot's
    /// bundle.  Ensures buffers aren't freed while a stale bundle might still replay them
    /// (relevant during the amortised rebuild window when instance counts change).
    bundle_kept_arcs: Vec<Vec<Arc<wgpu::Buffer>>>,
    /// Per-light monotonic dirty counter.  Incremented whenever either the draw-list
    /// generation changes (new/removed batch) or the filtered caster set changes identity
    /// (a caster entered or left shadow range).
    bundle_dirty_gen: Vec<u64>,
    /// Per-light `draw_list_generation` value seen when its dirty counter was last bumped.
    /// Compared each frame to detect structural draw-list changes.
    bundle_seen_draw_gen: Vec<u64>,
    /// Per-light geom hash (vertex-buffer ptrs + instance counts) seen when dirty counter
    /// was last bumped.  Compared each frame to detect caster-set changes without relying
    /// on camera position or world transforms, which would fire every frame.
    bundle_geom_hashes: Vec<u64>,
    /// Per-slot (light*6+face) dirty counter value when this face was last rebuilt.
    /// Face is stale when `bundle_slot_built[slot] < bundle_dirty_gen[i]`.
    bundle_slot_built: Vec<u64>,
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
            bundle_kept_arcs: (0..total_slots).map(|_| Vec::new()).collect(),
            bundle_dirty_gen: vec![0; max_lights],
            bundle_seen_draw_gen: vec![0; max_lights],
            bundle_geom_hashes: vec![0; max_lights],
            bundle_slot_built: vec![0; total_slots],
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

        // Maximum face bundles re-encoded per execute() call.  Amortises the CPU cost of
        // bundle rebuilds across frames; stale faces replay their previous bundle, which is
        // safe because (a) draw calls in the shadow list reference per-batch stable instance
        // buffers, and (b) `bundle_kept_arcs` keeps those buffers alive until the face is
        // rebuilt, preventing use-after-free when a batch gains/loses instances.
        const MAX_FACE_REBUILDS: u32 = 2;

        // ── Camera-distance filter (shared across all lights) ────────────────
        self.filtered_indices.clear();
        self.filtered_indices.reserve(draw_calls.len());
        for (idx, dc) in draw_calls.iter().enumerate() {
            let dist = (glam::Vec3::from(dc.bounds_center) - ctx.camera_position).length();
            if dist <= SHADOW_MAX_DISTANCE {
                self.filtered_indices.push(idx);
            }
        }

        // ── Geometry hash (shared across all lights) ─────────────────────────
        // Hashes only the STRUCTURAL identity of the visible caster set:
        // which batches are present (vertex_buffer ptr) and how many instances
        // each has. Deliberately excludes camera position and world transforms so
        // it does NOT fire on every frame of object movement or camera pan —
        // only when a caster appears/disappears inside shadow range, or when
        // the batch structure changes (streaming, spawn, destroy).
        let geom_hash = {
            let mut h: u64 = 0xcbf29ce484222325;
            h = fnv64_push(h, self.filtered_indices.len() as u64);
            for &idx in &self.filtered_indices {
                let dc = &draw_calls[idx];
                h = fnv64_push(h, Arc::as_ptr(&dc.vertex_buffer) as u64);
                h = fnv64_push(h, dc.instance_count as u64);
            }
            h
        };

        // Budget of face rebuilds shared across all lights this frame.
        let mut faces_rebuilt_this_frame: u32 = 0;

        for i in 0..actual_count {
            let max_faces = face_counts.get(i).copied().unwrap_or(6) as u32;
            let light = cull_lights.get(i).copied().unwrap_or_default();

            // ── Light hash ───────────────────────────────────────────────────
            let light_hash = fnv64(&[
                light.matrix_hash,
                light.range.to_bits() as u64,
                light.is_directional as u64,
                light.is_point as u64,
                max_faces as u64,
            ]);

            // ── Invalidation check ───────────────────────────────────────────
            // Bump the per-light dirty counter when either:
            // (a) the draw-list structure changed (new/removed batch), or
            // (b) the filtered caster set changed identity (caster entered/left
            //     shadow range, or batch grew/shrank).
            // Using a separate dirty counter decouples the amortisation logic from
            // the draw_list_generation value, and handles both sources uniformly.
            let gen_changed  = ctx.draw_list_generation != self.bundle_seen_draw_gen[i];
            let geom_changed = geom_hash != self.bundle_geom_hashes[i];
            if gen_changed || geom_changed {
                self.bundle_dirty_gen[i] = self.bundle_dirty_gen[i].wrapping_add(1);
                self.bundle_seen_draw_gen[i] = ctx.draw_list_generation;
                self.bundle_geom_hashes[i]   = geom_hash;
                eprintln!(
                    "⚠️ [Shadow] {} changed: light {}, {} visible casters → {} faces pending",
                    if gen_changed { "Draw list gen" } else { "Caster set" },
                    i, self.filtered_indices.len(), max_faces,
                );
            }

            let any_face_pending = (0..max_faces as usize)
                .any(|f| self.bundle_slot_built[i * 6 + f] < self.bundle_dirty_gen[i]);

            // ── Cache check ──────────────────────────────────────────────────
            {
                let cached = &self.shadow_cache[i];
                if cached.valid && cached.light_hash == light_hash && !any_face_pending {
                    continue;
                }
            }

            let t_light = ctx.scope_begin(&format!("shadow/light_{i}"));
            for face in 0u32..max_faces {
                let layer_idx  = i as u32 * 6 + face;
                let bundle_slot = i * 6 + face as usize;
                let face_pending =
                    self.bundle_slot_built[bundle_slot] < self.bundle_dirty_gen[i];

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

                if face_pending && faces_rebuilt_this_frame < MAX_FACE_REBUILDS {
                    let _tb = std::time::Instant::now();
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
                    enc.set_bind_group(0, &self.slot_bind_groups[bundle_slot], &[]);

                    // Keep Arcs to every instance buffer encoded in this bundle so that
                    // replaying it while the next rebuild is still pending (amortisation
                    // window) cannot reference a dropped buffer.
                    let mut kept_arcs: Vec<Arc<wgpu::Buffer>> =
                        Vec::with_capacity(self.filtered_indices.len());

                    for &idx in &self.filtered_indices {
                        let dc = &draw_calls[idx];
                        enc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
                        enc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
                        let inst_start = dc.instance_buffer_offset;
                        let inst_end   = inst_start + dc.instance_count as u64 * INSTANCE_STRIDE;
                        enc.set_vertex_buffer(1, dc.instance_buffer.as_ref().unwrap().slice(inst_start..inst_end));
                        enc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                        enc.draw_indexed(0..dc.index_count, 0, 0..dc.instance_count);
                        if let Some(buf) = &dc.instance_buffer {
                            kept_arcs.push(Arc::clone(buf));
                        }
                    }

                    let bundle = enc.finish(&wgpu::RenderBundleDescriptor { label: None });
                    pass.execute_bundles(std::iter::once(&bundle));
                    self.bundle_cache[bundle_slot]     = Some(bundle);
                    self.bundle_kept_arcs[bundle_slot] = kept_arcs;
                    self.bundle_slot_built[bundle_slot] = self.bundle_dirty_gen[i];
                    faces_rebuilt_this_frame += 1;
                    eprintln!(
                        "⚠️ [Shadow]   face {}/{}: {} casters — {:.2}ms",
                        face, max_faces - 1, self.filtered_indices.len(),
                        _tb.elapsed().as_secs_f32() * 1000.0,
                    );
                } else if let Some(bundle) = &self.bundle_cache[bundle_slot] {
                    // Replay the cached bundle.  Shadow-matrix GPU buffer is already updated
                    // this frame; per-batch instance data is current (or kept alive via
                    // bundle_kept_arcs if this slot is still in the amortisation window).
                    pass.execute_bundles(std::iter::once(bundle));
                }

                drop(pass);
                ctx.scope_end(t_face);
            }
            ctx.scope_end(t_light);

            self.shadow_cache[i] = ShadowLightCache { light_hash, geom_hash: 0, valid: true };
        }

        Ok(())
    }
}

