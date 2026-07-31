//! Shadow atlas pass.
//!
//! Renders scene geometry depth-only into a pre-allocated `Depth32Float` texture array
//! (one layer per shadow face).  Design is inspired by Unreal Engine 4's "Shadow Depth
//! Pass" and Unity HDRP's "Shadow Caster Pass":
//!
//! * **Depth-only pipeline** — no colour outputs, no fragment shader.
//! * **Front-face culled** — eliminates self-shadowing acne on lit surfaces,
//!   exactly matching the UE4/Unity convention.
//! * **GPU-driven dynamic atlas** — per-face dirty detection via `ShadowDirtyPass`;
//!   `multi_draw_indexed_indirect_count` suppresses draws on clean faces without
//!   CPU readback.  A companion depth-clear pipeline issues a GPU clear triangle
//!   before geometry draws so `LoadOp::Load` can be used on every face, preserving
//!   the cached atlas on clean faces.
//! * **Per-face granularity** — a moving object on the +X side of a point light
//!   does NOT trigger re-rendering of -X, ±Y, ±Z cube faces.
//! * **O(1) CPU per frame** — face loop bounded by `MAX_SHADOW_FACES`; the only
//!   CPU work per face is issuing wgpu commands (constant time).
//! * **Zero per-frame allocations** — all GPU and CPU resources pre-allocated.
//!
//! # Shadow Atlas
//!
//! | Property     | Value                                         |
//! |--------------|-----------------------------------------------|
//! | Format       | `Depth32Float`                                |
//! | Resolution   | `SHADOW_RES × SHADOW_RES` per face            |
//! | Array layers | `MAX_SHADOW_FACES` (256)                      |
//! | VRAM         | ~256 MB at 1024 px (constant, pre-allocated)  |
//!
//! # Dynamic Atlas — GPU-driven dirty detection
//!
//! Object movement is detected on GPU by `ShadowDirtyPass`, which writes two buffers:
//!
//! | Buffer           | Contents                                               |
//! |------------------|--------------------------------------------------------|
//! | `face_dirty_buf` | `array<u32, 256>` — 0 clean, 1 dirty per face          |
//! | `face_geom_count_buf` | `array<u32, 256>` — 0 or movable_draw_count per face |
//!
//! For each face:
//!   1. `multi_draw_indirect_count` with `face_dirty_buf[face]` as count (0 or 1)
//!      drives a full-screen depth-clear triangle (clears only dirty faces).
//!   2. `multi_draw_indexed_indirect_count` with `face_geom_count_buf[face]` as count
//!      (0 or movable_draw_count) drives shadow geometry draws.
//!   Both use `LoadOp::Load`, so clean faces preserve their cached shadow data.
//!
//! Light movement is still detected CPU-side via `per_caster_dirty_gen` (O(N_lights),
//! negligible).  Light-dirty faces use `LoadOp::Clear` + full movable geometry draws.

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::Arc;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum shadow atlas faces (42 point lights × 6 cube-faces = 252; 4 CSM cascades; ceiling = 256).
const MAX_SHADOW_FACES: usize = 256;

/// Byte stride between consecutive face-index entries in `face_idx_buf`.
///
/// Must satisfy `device.limits().min_uniform_buffer_offset_alignment`, which is
/// guaranteed to be ≤ 256 on every wgpu backend (Metal, Vulkan, DX12, WebGPU).
const FACE_BUF_STRIDE: u64 = 256;

/// Number of draws per face in the culled indirect buffer (written by ShadowCullPass).
/// Must match `MAX_DRAWS_PER_FACE` in helio-pass-shadow-cull.
const MAX_DRAWS_PER_FACE: u32 = 4096;

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct ShadowPass {
    /// Shadow geometry pipeline (depth-only, front-face culled, depth-bias = 2.0).
    pipeline: wgpu::RenderPipeline,

    /// Depth-clear pipeline — renders a full-screen triangle at z=1.0 with
    /// `DepthCompare::Always` to GPU-clear individual atlas faces before geometry.
    depth_clear_pipeline: wgpu::RenderPipeline,

    #[allow(dead_code)]
    bgl_0: wgpu::BindGroupLayout,

    /// 256 pre-populated non-indexed draw commands for the depth-clear triangle.
    /// All entries: `{ vertex_count: 3, instance_count: 1, first_vertex: 0, first_instance: 0 }`.
    /// `multi_draw_indirect_count` uses `face_dirty_buf[face]` (0 or 1) as the GPU count.
    clear_indirect_buf: wgpu::Buffer,

    /// Per-face face-index values, written once at construction and never touched again.
    face_idx_buf: wgpu::Buffer,

    // ── Dynamic shadow atlas (Movable objects only) ───────────────────────────
    face_views: Box<[wgpu::TextureView]>,
    bg_0: Option<wgpu::BindGroup>,
    bg_0_key: Option<(usize, usize)>,

    // ── Static shadow atlas (Static/Stationary objects only) ─────────────────
    static_face_views: Box<[wgpu::TextureView]>,
    /// Last `static_objects_generation` rendered.  `None` = never rendered.
    static_atlas_cache_gen: Option<u64>,

    pub compare_sampler: wgpu::Sampler,

    // ── GPU dirty buffers (shared with ShadowDirtyPass) ───────────────────────
    /// `array<u32, 256>` — 0 = clean, 1 = dirty (written by ShadowDirtyPass).
    /// Used as indirect draw count for the depth-clear triangle (0 = no clear, 1 = clear).
    face_dirty_buf: Arc<wgpu::Buffer>,
    /// `array<u32, 256>` — 0 = clean, movable_draw_count = dirty (written by ShadowDirtyPass).
    /// Used as indirect draw count for movable geometry (`multi_draw_indexed_indirect_count`).
    #[allow(dead_code)]
    face_geom_count_buf: Arc<wgpu::Buffer>,

    /// Per-face culled indirect commands (written by ShadowCullPass).
    /// Layout: `MAX_FACES × MAX_DRAWS_PER_FACE × 20` bytes — each face's range
    /// contains only objects whose bounding sphere intersects that face's frustum.
    face_cull_indirect: Arc<wgpu::Buffer>,

    /// Per-face culled draw counts (written by ShadowCullPass).
    /// `array<u32, 256>` — number of visible draws per face, written atomically
    /// by the compute shader.  Used with `multi_draw_indexed_indirect_count`.
    face_cull_counts: Arc<wgpu::Buffer>,

    /// Resolution of each atlas face (width × height).
    atlas_size: u32,

    /// Number of texture-array layers actually allocated by the graph.
    atlas_layers: u32,

    // ── Per-caster CPU dirty tracking (light movement only) ──────────────────
    /// Per-caster last-rendered generation, compared against `per_caster_dirty_gen`.
    /// Only updated when a light moves (object movement is now detected GPU-side).
    per_caster_last_gen: [u64; 42],

    /// Total shadow count at last render.  Detects caster topology changes.
    last_rendered_shadow_count: u32,

    /// `movable_objects_generation` at last render.  O(1) CPU check to gate the GPU path.
    last_movable_objects_gen: u64,

    /// True when the device supports MULTI_DRAW_INDIRECT_COUNT (Vulkan 1.2+, DX12 tier2).
    /// False on macOS Metal, WASM, and older Vulkan/DX12.  When false the ObjectDirty path
    /// falls back to a full LoadOp::Clear + multi_draw_indexed_indirect (no per-face GPU culling).
    supports_multi_draw_count: bool,
}

impl ShadowPass {
    /// Allocate all GPU resources.  Called once; zero allocations after this.
    ///
    /// `face_dirty_buf` and `face_geom_count_buf` are shared with `ShadowDirtyPass`
    /// which writes them each frame; they arrive via `Arc`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        face_dirty_buf: Arc<wgpu::Buffer>,
        face_geom_count_buf: Arc<wgpu::Buffer>,
        face_cull_indirect: Arc<wgpu::Buffer>,
        face_cull_counts: Arc<wgpu::Buffer>,
        atlas_size: u32,
        atlas_layers: u32,
    ) -> Self {
        let atlas_layers = atlas_layers.clamp(1, MAX_SHADOW_FACES as u32);
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
        });

        let clear_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow/DepthClear"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/depth_clear.wgsl").into()),
        });

        // ── Bind Group Layout 0 ───────────────────────────────────────────────
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow BGL 0"),
            entries: &[
                // binding 0: shadow_matrices — array of mat4x4 light-space transforms
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
                // binding 1: instances — per-instance world transforms
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: face index — 16-byte uniform, dynamic offset selects face
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow PL"),
            bind_group_layouts: &[Some(&bgl_0)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // Shared mesh vertex buffer layout (stride = 40 bytes, matches GBuffer pass).
                // Only position (Float32x3 at offset 0) is needed for depth projection.
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 40,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
            },
            // Depth-only: no colour outputs, no fragment shader.
            // The GPU writes depth from the vertex clip position automatically.
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Front-face culling: light "looks into" the scene; culling the faces
                // visible to the light prevents writing depth for lit-surface geometry
                // directly, eliminating shadow acne.  Identical convention to UE4/Unity.
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                // slope_scale compensates for FP depth precision on surfaces at
                // grazing angles to the light.  Without it the shadow map depth for
                // a surface can be equal-to or less-than the depth reconstructed in
                // the lighting shader for that same surface, causing self-shadowing
                // on every light independently (making each light appear to inherit
                // every other light's shadow geometry).
                // constant is left at 0 — that was the source of the visible offset.
                bias: wgpu::DepthBiasState {
                    constant: 0,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Depth-clear pipeline ───────────────────────────────────────────────
        // GPU-clear individual shadow atlas faces: renders a full-screen triangle
        // at depth=1.0 (far plane) using DepthCompare::Always to overwrite existing
        // depth values.  No vertex buffer, no fragment shader, no depth bias.
        let depth_clear_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Shadow/DepthClear PL"),
                bind_group_layouts: &[],
                immediate_size: 0,
            });

        let depth_clear_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow/DepthClear Pipeline"),
            layout: Some(&depth_clear_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &clear_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Clear indirect buffer ──────────────────────────────────────────────
        // 256 non-indexed draw commands, each drawing 3 vertices (the clear triangle).
        // Layout per command (16 bytes): { vertex_count: 3, instance_count: 1,
        //                                  first_vertex: 0, first_instance: 0 }
        // `multi_draw_indirect_count` uses `face_dirty_buf[face]` as the GPU draw count
        // (0 no clear, 1 clear), with indirect_offset = face * 16.
        let mut clear_indirect_data = vec![[0u32; 4]; MAX_SHADOW_FACES];
        for command in &mut clear_indirect_data {
            command[0] = 3;
            command[1] = 1;
        }
        // Avoid mappedAtCreation here and below: browser WebGPU may reject the
        // active mapping synchronously even for a small, otherwise valid buffer.
        let clear_indirect_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow/ClearIndirect"),
            size: MAX_SHADOW_FACES as u64 * 16,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &clear_indirect_buf,
            0,
            bytemuck::cast_slice(&clear_indirect_data),
        );
        // One u32 per face at FACE_BUF_STRIDE byte intervals.
        // The CPU never touches this buffer after construction.
        let mut face_idx_data = vec![0u8; MAX_SHADOW_FACES * FACE_BUF_STRIDE as usize];
        for i in 0..MAX_SHADOW_FACES {
            let offset = i * FACE_BUF_STRIDE as usize;
            face_idx_data[offset..offset + 4].copy_from_slice(&(i as u32).to_ne_bytes());
        }
        let face_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow/FaceIdx"),
            size: MAX_SHADOW_FACES as u64 * FACE_BUF_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&face_idx_buf, 0, &face_idx_data);

        // ── Face views (lazily initialized from graph-owned textures) ──────────
        let face_views = Box::default();
        let static_face_views = Box::default();

        // Comparison sampler for PCF shadow lookups in the lighting pass.
        let compare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Shadow/Compare"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        Self {
            pipeline,
            depth_clear_pipeline,
            bgl_0,
            bg_0: None,
            bg_0_key: None,
            static_atlas_cache_gen: None,
            face_idx_buf,
            clear_indirect_buf,
            face_views,
            static_face_views,
            compare_sampler,
            face_dirty_buf,
            face_geom_count_buf,
            face_cull_indirect,
            face_cull_counts,
            per_caster_last_gen: [0u64; 42],
            last_rendered_shadow_count: 0,
            last_movable_objects_gen: u64::MAX,
            supports_multi_draw_count: device
                .features()
                .contains(wgpu::Features::MULTI_DRAW_INDIRECT_COUNT),
            atlas_size,
            atlas_layers,
        }
    }

    fn create_face_views(
        texture: &wgpu::Texture,
        label: &str,
        layer_count: u32,
    ) -> Box<[wgpu::TextureView]> {
        (0..layer_count)
            .map(|i| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(label),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect()
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for ShadowPass {
    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        let sz = ResourceSize::Absolute {
            width: self.atlas_size,
            height: self.atlas_size,
        };
        builder.write_color_raw("shadow_atlas", wgpu::TextureFormat::Depth32Float, sz);
        builder.with_layers(self.atlas_layers);
        builder.write_color_raw("static_shadow_atlas", wgpu::TextureFormat::Depth32Float, sz);
        builder.with_layers(self.atlas_layers);
    }

    fn name(&self) -> &'static str {
        "Shadow"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["main_scene"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["shadow_atlas", "shadow_sampler", "static_shadow_atlas"]
    }

    fn publish<'a>(&'a self, _frame: &mut libhelio::FrameResources<'a>) {}

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let face_count = (ctx.scene.shadow_count as usize)
            .min(self.atlas_layers as usize)
            .min(MAX_SHADOW_FACES);
        let static_draw_count = ctx.scene.shadow_static_draw_count;
        let movable_draw_count = ctx.scene.shadow_movable_draw_count;

        // ── Lazily initialize per-face views from graph-owned textures ─────────
        if self.face_views.is_empty() {
            if let Some(tex) = ctx.resource_pool.get_texture("shadow_atlas") {
                self.face_views =
                    Self::create_face_views(tex, "Shadow/DynamicFace", self.atlas_layers);
            }
        }
        if self.static_face_views.is_empty() {
            if let Some(tex) = ctx.resource_pool.get_texture("static_shadow_atlas") {
                self.static_face_views =
                    Self::create_face_views(tex, "Shadow/StaticFace", self.atlas_layers);
            }
        }

        if face_count == 0 {
            self.per_caster_last_gen = [0u64; 42];
            self.last_rendered_shadow_count = 0;
            self.static_atlas_cache_gen = None;
            self.last_movable_objects_gen = u64::MAX;
            return Ok(());
        }

        let static_gen = ctx.scene.static_objects_generation;
        let shadow_count = ctx.scene.shadow_count;
        let caster_count = (face_count / 6).min(42);

        let need_static = self.static_atlas_cache_gen != Some(static_gen)
            || shadow_count != self.last_rendered_shadow_count;

        // Per-caster dirty check for LIGHT movement only.
        // Object-movement dirtiness is handled GPU-side via face_geom_count_buf.
        let mut dirty_casters = [false; 42];
        let mut any_dirty_caster = false;
        for slot in 0..caster_count {
            if ctx.scene.per_caster_dirty_gen[slot] != self.per_caster_last_gen[slot] {
                dirty_casters[slot] = true;
                any_dirty_caster = true;
            }
        }

        // O(1) CPU gate: did any movable object move this frame?
        let objects_moved = ctx.scene.movable_objects_generation != self.last_movable_objects_gen;

        if !need_static && !any_dirty_caster && !objects_moved {
            return Ok(());
        }

        let main_scene = ctx.resources.main_scene.read("Shadow").ok_or_else(|| {
            helio_core::Error::InvalidPassConfig("ShadowPass requires main_scene".into())
        })?;

        let vertices = main_scene.mesh_buffers.vertices;
        let indices = main_scene.mesh_buffers.indices;

        // ── Shared bind group (shadow_matrices + instances + face_idx) ──────────
        // Rebuilt only on GrowableBuffer reallocation (O(1) amortised).
        let sm_ptr = ctx.scene.shadow_matrices as *const _ as usize;
        let inst_ptr = ctx.scene.instances as *const _ as usize;
        let key = (sm_ptr, inst_ptr);
        if self.bg_0_key != Some(key) {
            self.bg_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Shadow BG 0"),
                layout: &self.bgl_0,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.shadow_matrices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.face_idx_buf,
                            offset: 0,
                            size: std::num::NonZeroU64::new(16),
                        }),
                    },
                ],
            }));
            self.bg_0_key = Some(key);
        }
        let bg = self.bg_0.as_ref().unwrap();

        let pipeline = &self.pipeline;

        // ── Static atlas render ────────────────────────────────────────────────
        if need_static || any_dirty_caster {
            let static_indirect = ctx.scene.shadow_static_indirect;
            if static_draw_count > 0 {
                for face in 0..face_count {
                    let caster_slot = face / 6;
                    if !need_static && (caster_slot >= 42 || !dirty_casters[caster_slot]) {
                        continue;
                    }
                    let face_view = &self.static_face_views[face];
                    let dyn_offset = (face as u64 * FACE_BUF_STRIDE) as u32;
                    let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Shadow/Static"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: face_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bg, &[dyn_offset]);
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                    #[cfg(not(target_arch = "wasm32"))]
                    pass.multi_draw_indexed_indirect(static_indirect, 0, static_draw_count);
                    #[cfg(target_arch = "wasm32")]
                    for i in 0..static_draw_count {
                        pass.draw_indexed_indirect(static_indirect, i as u64 * 20);
                    }
                }
            } else if need_static {
                for face in 0..face_count {
                    let face_view = &self.static_face_views[face];
                    let _pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Shadow/StaticClear"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: face_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                }
            }
            if need_static {
                self.static_atlas_cache_gen = Some(static_gen);
                self.last_rendered_shadow_count = shadow_count;
                log::debug!(
                    "Shadow: re-rendered static atlas ({} draws, {} faces)",
                    static_draw_count,
                    face_count
                );
            }
        }

        // ── Dynamic atlas render — GPU-driven per-face dirty ──────────────────
        //
        // Two dirty sources with different handling:
        //
        //   Light movement (any_dirty_caster = true):
        //     Full clear + all movable draws, CPU-driven.  Light movement is rare
        //     (typically < 5 lights) so this path is O(6) render passes per light.
        //
        //   Object movement (objects_moved = true):
        //     LoadOp::Load (preserve cached atlas) + GPU-clear triangle (only for dirty
        //     faces) + GPU-driven geometry draws.  ShadowDirtyPass has written
        //     face_dirty_buf[face] ∈ {0,1} and face_geom_count_buf[face] ∈ {0, N}
        //     so multi_draw_{indirect,indexed_indirect}_count suppresses all work on
        //     clean faces.  The loop runs for all active faces but clean faces produce
        //     a near-zero-cost render pass (LoadOp::Load with 0 GPU draws).
        if any_dirty_caster || objects_moved {
            let _movable_indirect = ctx.scene.shadow_movable_indirect;

            for face in 0..face_count {
                let caster_slot = face / 6;
                let light_dirty = caster_slot < 42 && dirty_casters[caster_slot];
                let face_view = &self.face_views[face];
                let dyn_offset = (face as u64 * FACE_BUF_STRIDE) as u32;

                if light_dirty {
                    // ── Light moved: full clear + culled draws ─────────────────
                    let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                        &wgpu::RenderPassDescriptor {
                            label: Some("Shadow/Dynamic/LightDirty"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(
                                wgpu::RenderPassDepthStencilAttachment {
                                    view: face_view,
                                    depth_ops: Some(wgpu::Operations {
                                        load: wgpu::LoadOp::Clear(1.0),
                                        store: wgpu::StoreOp::Store,
                                    }),
                                    stencil_ops: None,
                                },
                            ),
                            timestamp_writes: None,
                            occlusion_query_set: None,
                            multiview_mask: None,
                        },
                    );
                    if movable_draw_count > 0 {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, bg, &[dyn_offset]);
                        pass.set_vertex_buffer(0, vertices.slice(..));
                        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        let face_offset = face as u64 * MAX_DRAWS_PER_FACE as u64 * 20;
                        #[cfg(not(target_arch = "wasm32"))]
                        if self.supports_multi_draw_count {
                            pass.multi_draw_indexed_indirect_count(
                                &self.face_cull_indirect,
                                face_offset,
                                &self.face_cull_counts,
                                face as u64 * 4,
                                MAX_DRAWS_PER_FACE,
                            );
                        } else {
                            pass.multi_draw_indexed_indirect(
                                &self.face_cull_indirect,
                                face_offset,
                                MAX_DRAWS_PER_FACE,
                            );
                        }
                        #[cfg(target_arch = "wasm32")]
                        pass.multi_draw_indexed_indirect(
                            &self.face_cull_indirect,
                            face_offset,
                            MAX_DRAWS_PER_FACE,
                        );
                    }
                } else if objects_moved {
                    // ── Objects moved: GPU-driven clear + geometry ─────────────
                    // When MULTI_DRAW_INDIRECT_COUNT is available (Vulkan 1.2+, DX12):
                    //   LoadOp::Load preserves cached shadow data for clean faces.
                    //   The GPU-clear triangle (driven by face_dirty_buf count) clears
                    //   only faces that ShadowDirtyPass marked dirty.
                    // When the feature is unavailable (macOS Metal, older hardware):
                    //   Fall back to a full clear + draw all movable geometry,
                    //   equivalent to the LightDirty path but without per-face culling.
                    if self.supports_multi_draw_count {
                        let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                            &wgpu::RenderPassDescriptor {
                                label: Some("Shadow/Dynamic/ObjectDirty"),
                                color_attachments: &[],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: face_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Load,
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            },
                        );

                        if movable_draw_count > 0 {
                            // 1. Depth-clear triangle (GPU count 0 or 1 from face_dirty_buf).
                            pass.set_pipeline(&self.depth_clear_pipeline);
                            pass.multi_draw_indirect_count(
                                &self.clear_indirect_buf,
                                face as u64 * 16,
                                &self.face_dirty_buf,
                                face as u64 * 4,
                                1,
                            );

                            // 2. Shadow geometry (GPU count 0 or movable_draw_count from face_geom_count_buf).
                            pass.set_pipeline(pipeline);
                            pass.set_bind_group(0, bg, &[dyn_offset]);
                            pass.set_vertex_buffer(0, vertices.slice(..));
                            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                            let face_offset = face as u64 * MAX_DRAWS_PER_FACE as u64 * 20;
                            pass.multi_draw_indexed_indirect_count(
                                &self.face_cull_indirect,
                                face_offset,
                                &self.face_cull_counts,
                                face as u64 * 4,
                                MAX_DRAWS_PER_FACE,
                            );
                        }
                    } else {
                        // Fallback: full clear + draw all movable geometry (no per-face GPU culling).
                        let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(
                            &wgpu::RenderPassDescriptor {
                                label: Some("Shadow/Dynamic/ObjectDirty/Fallback"),
                                color_attachments: &[],
                                depth_stencil_attachment: Some(
                                    wgpu::RenderPassDepthStencilAttachment {
                                        view: face_view,
                                        depth_ops: Some(wgpu::Operations {
                                            load: wgpu::LoadOp::Clear(1.0),
                                            store: wgpu::StoreOp::Store,
                                        }),
                                        stencil_ops: None,
                                    },
                                ),
                                timestamp_writes: None,
                                occlusion_query_set: None,
                                multiview_mask: None,
                            },
                        );
                        if movable_draw_count > 0 {
                            pass.set_pipeline(pipeline);
                            pass.set_bind_group(0, bg, &[dyn_offset]);
                            pass.set_vertex_buffer(0, vertices.slice(..));
                            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                            let face_offset = face as u64 * MAX_DRAWS_PER_FACE as u64 * 20;
                            pass.multi_draw_indexed_indirect(
                                &self.face_cull_indirect,
                                face_offset,
                                MAX_DRAWS_PER_FACE,
                            );
                        }
                    }
                }
            }

            // Update per-caster gen tracking (light movement only).
            for slot in 0..caster_count {
                if dirty_casters[slot] {
                    self.per_caster_last_gen[slot] = ctx.scene.per_caster_dirty_gen[slot];
                }
            }

            self.last_movable_objects_gen = ctx.scene.movable_objects_generation;
        }

        Ok(())
    }
}
