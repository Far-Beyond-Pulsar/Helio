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

// ── Per-face frustum culling for point-light cubemaps ────────────────────────

/// Inward-pointing (unnormalized) plane normals for the 4 side planes of each
/// cubemap face's 90° frustum.  Normals have length √2; the sphere-vs-plane
/// test uses `dot(delta, normal) < -radius * √2` for the cull condition.
///
/// Face index mapping matches wgpu cubemap convention:
///   0 = +X, 1 = -X, 2 = +Y, 3 = -Y, 4 = +Z, 5 = -Z
const CUBE_FACE_PLANES: [[[f32; 3]; 4]; 6] = [
    // +X face — looking down +X
    [[ 1.0, 0.0,-1.0], [ 1.0, 0.0, 1.0], [ 1.0,-1.0, 0.0], [ 1.0, 1.0, 0.0]],
    // -X face — looking down -X
    [[-1.0, 0.0,-1.0], [-1.0, 0.0, 1.0], [-1.0,-1.0, 0.0], [-1.0, 1.0, 0.0]],
    // +Y face — looking down +Y
    [[ 0.0, 1.0,-1.0], [ 0.0, 1.0, 1.0], [-1.0, 1.0, 0.0], [ 1.0, 1.0, 0.0]],
    // -Y face — looking down -Y
    [[ 0.0,-1.0,-1.0], [ 0.0,-1.0, 1.0], [-1.0,-1.0, 0.0], [ 1.0,-1.0, 0.0]],
    // +Z face — looking down +Z
    [[-1.0, 0.0, 1.0], [ 1.0, 0.0, 1.0], [ 0.0,-1.0, 1.0], [ 0.0, 1.0, 1.0]],
    // -Z face — looking down -Z
    [[-1.0, 0.0,-1.0], [ 1.0, 0.0,-1.0], [ 0.0,-1.0,-1.0], [ 0.0, 1.0,-1.0]],
];

/// Test whether a bounding sphere (centre at `delta` from light, `radius`) is
/// potentially visible to the given cubemap `face`.  Uses conservative
/// sphere-vs-halfspace tests against the face's 4 frustum side planes.
///
/// Returns `true` if the sphere *may* contribute to shadows on this face.
/// Never produces false negatives — objects right on the boundary pass both
/// adjacent faces, avoiding cracks.
#[inline]
fn sphere_in_cube_face(delta: [f32; 3], radius: f32, face: u32) -> bool {
    let planes = &CUBE_FACE_PLANES[face as usize];
    let threshold = -radius * std::f32::consts::SQRT_2;
    for plane in planes {
        let d = delta[0] * plane[0] + delta[1] * plane[1] + delta[2] * plane[2];
        if d < threshold {
            return false;
        }
    }
    true
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
    /// Per-light monotonic dirty counter.  Incremented whenever the filtered shadow-caster
    /// set changes identity (a caster entered/left shadow range, or a batch grew/shrank).
    /// Does NOT increment on draw-list changes for out-of-range batches, which would cause
    /// spurious full rebuilds every time a distant chunk streams in.
    bundle_dirty_gen: Vec<u64>,
    /// Per-light geom hash (vertex-buffer ptrs + instance counts) seen when dirty counter
    /// was last bumped.  Compared each frame to detect in-range caster-set changes without
    /// relying on camera position or world transforms, which would fire every frame.
    bundle_geom_hashes: Vec<u64>,
    /// Per-slot (light*6+face) dirty counter value when this face was last rebuilt.
    /// Face is stale when `bundle_slot_built[slot] < bundle_dirty_gen[i]`.
    bundle_slot_built: Vec<u64>,
    /// Monotonic frame counter (incremented once per execute() call).  Used to
    /// rate-limit how often new dirty rounds start during chunk streaming.
    frame_counter: u64,
    /// Per-light frame_counter value when the most recent dirty round started.
    /// A new round cannot start until at least MIN_SHADOW_INTERVAL frames have
    /// elapsed since the last one, preventing the rebuild from chasing every
    /// incremental stream-in event at full frame rate.
    bundle_dirty_gen_start_frame: Vec<u64>,
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
    ) -> Self {
        // BGL: binding 0 = shadow matrices, binding 1 = per-face layer index,
        //      binding 2 = instance_data storage (transform read by @builtin(instance_index)).
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
                // binding 2: instance_data storage — vertex shader uses @builtin(instance_index)
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
            bundle_cache: (0..total_slots).map(|_| None).collect(),
            bundle_kept_arcs: (0..total_slots).map(|_| Vec::new()).collect(),
            bundle_dirty_gen: vec![0; max_lights],
            bundle_geom_hashes: vec![0; max_lights],
            bundle_slot_built: vec![0; total_slots],
            frame_counter: 0,
            bundle_dirty_gen_start_frame: vec![0; max_lights],
        }
    }
}

/// Standalone bundle encoder callable from any thread.  Produces a shadow
/// RenderBundle for a single cubemap face, encoding only the draw calls
/// specified by `face_indices`.
fn encode_shadow_bundle(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    slot_bind_group: &wgpu::BindGroup,
    draw_calls: &[DrawCall],
    face_indices: &[usize],
) -> (wgpu::RenderBundle, Vec<Arc<wgpu::Buffer>>) {
    let mut enc = device.create_render_bundle_encoder(
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
    enc.set_pipeline(pipeline);
    enc.set_bind_group(0, slot_bind_group, &[]);

    let mut kept_arcs: Vec<Arc<wgpu::Buffer>> = Vec::with_capacity(face_indices.len());
    for &idx in face_indices {
        let dc = &draw_calls[idx];
        enc.set_bind_group(1, Some(dc.material_bind_group.as_ref()), &[]);
        enc.set_vertex_buffer(0, dc.vertex_buffer.slice(..));
        enc.set_index_buffer(dc.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        enc.draw_indexed(
            dc.pool_first_index..dc.pool_first_index + dc.index_count,
            dc.pool_base_vertex,
            dc.slot..dc.slot + 1,
        );
        kept_arcs.push(Arc::clone(&dc.vertex_buffer));
    }

    let bundle = enc.finish(&wgpu::RenderBundleDescriptor { label: None });
    (bundle, kept_arcs)
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
        self.frame_counter += 1;
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

        // Maximum face bundles re-encoded per execute() call.  With per-face frustum
        // culling each face encodes only ~1/6th of the draws (point lights), so the per-face
        // cost drops from ~26 ms to ~4 ms.  Budget of 6 lets all faces of a point light
        // rebuild in a single frame at ~24 ms worst-case (vs ~52 ms before culling).
        const MAX_FACE_REBUILDS: u32 = 6;

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
                h = fnv64_push(h, dc.slot as u64);
            }
            h
        };

        // ═══ Phase 1: Collect dirty faces and lights needing replay ══════════
        struct FaceJob {
            light_idx: usize,
            face: u32,
            slot: usize,
        }
        let mut rebuild_jobs: Vec<FaceJob> = Vec::new();
        // (light_idx, max_faces, light_hash)
        let mut lights_needing_replay: Vec<(usize, u32, u64)> = Vec::new();

        for i in 0..actual_count {
            let max_faces = face_counts.get(i).copied().unwrap_or(6) as u32;

            // ── Light hash ───────────────────────────────────────────────────
            let light = cull_lights.get(i).copied().unwrap_or_default();
            let light_hash = fnv64(&[
                light.matrix_hash,
                light.range.to_bits() as u64,
                light.is_directional as u64,
                light.is_point as u64,
                max_faces as u64,
            ]);

            // ── Invalidation check ───────────────────────────────────────────
            // Only the geom_hash is used to detect changes.  draw_list_generation
            // is intentionally NOT checked here: it increments for every batch
            // addition anywhere in the scene, including chunks loading outside the
            // 300 m shadow sphere.  Those additions don't affect the shadow caster
            // set at all, but would cause a full bundle rebuild every frame during
            // streaming.  The geom_hash already covers every case that matters:
            //   • new batch enters shadow range         → new ptr in filtered set
            //   • batch removed / leaves shadow range   → ptr removed
            //   • batch gains / loses instances         → instance_count changes
            // Transforms are handled by per-batch stable instance buffers (no
            // re-encode needed; write_buffer keeps GPU data current).
            //
            // STREAMING ABSORPTION RULE: if any face from the current round is
            // still pending, do NOT start a new round.  The pending faces will be
            // encoded with the current frame's `filtered_indices`, so they already
            // incorporate the latest geometry.  Starting a new round while work is
            // in flight causes `dirty_gen` to race ahead of `slot_built`
            // indefinitely, meaning later faces are never reached.
            let any_face_pending = (0..max_faces as usize)
                .any(|f| self.bundle_slot_built[i * 6 + f] < self.bundle_dirty_gen[i]);

            // Minimum frames between shadow dirty rounds per light.
            const MIN_SHADOW_INTERVAL: u64 = 8;
            if geom_hash != self.bundle_geom_hashes[i] {
                let frames_since_last = self.frame_counter
                    .saturating_sub(self.bundle_dirty_gen_start_frame[i]);
                if !any_face_pending && frames_since_last >= MIN_SHADOW_INTERVAL {
                    self.bundle_dirty_gen[i] = self.bundle_dirty_gen[i].wrapping_add(1);
                    self.bundle_geom_hashes[i] = geom_hash;
                    self.bundle_dirty_gen_start_frame[i] = self.frame_counter;
                    eprintln!(
                        "⚠️ [Shadow] Caster set changed: light {}, {} visible casters → {} faces pending",
                        i, self.filtered_indices.len(), max_faces,
                    );
                }
            }

            // ── Cache check (uses pre-invalidation any_face_pending) ─────────
            {
                let cached = &self.shadow_cache[i];
                if cached.valid && cached.light_hash == light_hash && !any_face_pending {
                    continue;
                }
            }

            lights_needing_replay.push((i, max_faces, light_hash));

            // Collect pending faces for parallel encoding (up to budget).
            for face in 0u32..max_faces {
                let slot = i * 6 + face as usize;
                if self.bundle_slot_built[slot] < self.bundle_dirty_gen[i]
                    && rebuild_jobs.len() < MAX_FACE_REBUILDS as usize
                {
                    rebuild_jobs.push(FaceJob { light_idx: i, face, slot });
                }
            }
        }

        // ═══ Phase 2: Parallel bundle encoding ═══════════════════════════════
        if !rebuild_jobs.is_empty() {
            let _t = std::time::Instant::now();

            // Pre-compute per-face filtered draw indices with frustum culling.
            let per_face_indices: Vec<Vec<usize>> = rebuild_jobs.iter().map(|job| {
                let light = cull_lights.get(job.light_idx).copied().unwrap_or_default();
                self.filtered_indices.iter().copied().filter(|&idx| {
                    let dc = &draw_calls[idx];
                    if light.is_point {
                        let delta = [
                            dc.bounds_center[0] - light.position[0],
                            dc.bounds_center[1] - light.position[1],
                            dc.bounds_center[2] - light.position[2],
                        ];
                        sphere_in_cube_face(delta, dc.bounds_radius, job.face)
                    } else {
                        true
                    }
                }).collect()
            }).collect();

            // Encode bundles in parallel using scoped threads.
            // Extract shared references inside a block so they are dropped before
            // the mutable bundle_cache/kept_arcs writes that follow.
            let encoded: Vec<(wgpu::RenderBundle, Vec<Arc<wgpu::Buffer>>)> = {
                let device = &*self.device;
                let pipeline = &*self.pipeline;
                let slot_bgs = &self.slot_bind_groups;
                let dc_slice: &[DrawCall] = &draw_calls;

                std::thread::scope(|s| {
                    let handles: Vec<_> = rebuild_jobs
                        .iter()
                        .enumerate()
                        .map(|(j, job)| {
                            let bg = &slot_bgs[job.slot];
                            let fi = per_face_indices[j].as_slice();
                            s.spawn(move || {
                                encode_shadow_bundle(device, pipeline, bg, dc_slice, fi)
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap())
                        .collect()
                })
            };

            // Apply encoded bundles to cache.
            for (j, (bundle, kept)) in encoded.into_iter().enumerate() {
                let job = &rebuild_jobs[j];
                eprintln!(
                    "⚠️ [Shadow]   light {}/face {}: {} casters (of {} filtered, {:.0}% culled)",
                    job.light_idx, job.face, per_face_indices[j].len(),
                    self.filtered_indices.len(),
                    if !self.filtered_indices.is_empty() {
                        (1.0 - per_face_indices[j].len() as f64
                            / self.filtered_indices.len() as f64) * 100.0
                    } else { 0.0 },
                );
                self.bundle_cache[job.slot] = Some(bundle);
                self.bundle_kept_arcs[job.slot] = kept;
                self.bundle_slot_built[job.slot] = self.bundle_dirty_gen[job.light_idx];
            }

            eprintln!(
                "⚠️ [Shadow] Encoded {} face bundles in {:.2}ms (parallel)",
                rebuild_jobs.len(), _t.elapsed().as_secs_f32() * 1000.0,
            );
        }

        // ═══ Phase 3: Replay all active lights into render passes ════════════
        for &(i, max_faces, light_hash) in &lights_needing_replay {
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

                if let Some(bundle) = &self.bundle_cache[bundle_slot] {
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

