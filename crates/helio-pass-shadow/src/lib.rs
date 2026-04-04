//! Shadow atlas pass.
//!
//! Renders scene geometry depth-only into a pre-allocated `Depth32Float` texture array
//! (one layer per shadow face).  Design is inspired by Unreal Engine 4's "Shadow Depth
//! Pass" and Unity HDRP's "Shadow Caster Pass":
//!
//! * **Depth-only pipeline** — no colour outputs, no fragment shader.
//! * **Front-face culled** — eliminates self-shadowing acne on lit surfaces,
//!   exactly matching the UE4/Unity convention.
//! * **GPU-driven** — one `multi_draw_indexed_indirect` per face; zero per-draw
//!   CPU work regardless of scene complexity.
//! * **O(1) CPU per frame** — face loop bounded by `MAX_SHADOW_FACES` (compile-time
//!   constant); the loop body issues only constant-time wgpu calls.
//! * **Zero per-frame allocations** — all GPU and CPU resources are pre-allocated
//!   in `new()` and never resized or recreated during rendering.
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
//! # Bind Group 0
//!
//! | Binding | Type                           | Contents                              |
//! |---------|--------------------------------|---------------------------------------|
//! | 0       | Storage (read-only), VERTEX    | Light-space matrices (one per face)   |
//! | 1       | Storage (read-only), VERTEX    | Per-instance world transforms         |
//! | 2       | Uniform (dynamic offset), VERTEX | Face index — selects the matrix     |
//!
//! The bind group is rebuilt **only** when the underlying `GrowableBuffer` backing
//! `shadow_matrices` or `instances` is reallocated (buffer pointer change).  In
//! steady-state this never happens, making bind group management O(1) amortised.
//!
//! The per-face face-index uniform is written **once** at construction time and
//! addressed via a `dynamic_offset` of `face × FACE_BUF_STRIDE` bytes.  There is
//! no per-frame CPU write to this buffer.

use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use std::sync::Arc;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum shadow atlas faces (42 point lights × 6 cube-faces = 252; 4 CSM cascades; ceiling = 256).
const MAX_SHADOW_FACES: usize = 256;

/// Texel resolution per atlas face.  1024² balances quality and VRAM (~256 MB).
const SHADOW_RES: u32 = 1024;

/// Byte stride between consecutive face-index entries in `face_idx_buf`.
///
/// Must satisfy `device.limits().min_uniform_buffer_offset_alignment`, which is
/// guaranteed to be ≤ 256 on every wgpu backend (Metal, Vulkan, DX12, WebGPU).
const FACE_BUF_STRIDE: u64 = 256;

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct ShadowPass {
    pipeline: wgpu::RenderPipeline,

    #[allow(dead_code)]
    bgl_0: wgpu::BindGroupLayout,

    /// Per-face face-index values, written once at construction and never touched again.
    /// Layout: `face_idx_buf[face * FACE_BUF_STRIDE]` = `u32(face)` followed by 252 zero bytes.
    face_idx_buf: wgpu::Buffer,

    // ── Dynamic shadow atlas (Movable objects only) ───────────────────────────
    /// Per-face render-target views into the dynamic atlas (movable objects).
    face_views: Box<[wgpu::TextureView]>,
    /// The dynamic atlas texture (owned).
    pub atlas_tex: wgpu::Texture,
    /// `D2Array` view — consumed by the deferred lighting pass.
    pub atlas_view: wgpu::TextureView,
    /// Bind group 0 (shared by both static and dynamic passes — both use `instances`).
    bg_0: Option<wgpu::BindGroup>,
    /// Detects GrowableBuffer reallocations; rebuilt on pointer change only.
    bg_0_key: Option<(usize, usize)>,

    // ── Static shadow atlas (Static/Stationary objects only) ─────────────────
    /// Per-face render-target views into the static atlas.
    static_face_views: Box<[wgpu::TextureView]>,
    /// The static atlas texture (owned, cached indefinitely until static topology changes).
    pub static_atlas_tex: wgpu::Texture,
    /// `D2Array` view of the static atlas — consumed by deferred lighting.
    pub static_atlas_view: wgpu::TextureView,
    /// Last `static_objects_generation` we rendered the static atlas for.
    /// `None` = never rendered yet → must render on first frame.
    static_atlas_cache_gen: Option<u64>,

    /// PCF comparison sampler (`LessEqual`) — consumed by the deferred lighting pass.
    pub compare_sampler: wgpu::Sampler,

    /// Dirty flag buffer reference (per-light dirty flags from shadow matrix compute)
    _shadow_dirty_buf: Arc<wgpu::Buffer>,

    /// Cache state for the dynamic atlas: (movable_objects_gen, movable_lights_gen, shadow_count).
    /// Skips re-render when nothing Movable has changed.
    shadow_cache_state: Option<(u64, u64, u32)>,
}

impl ShadowPass {
    /// Allocate all GPU resources.  Called once; zero allocations after this.
    pub fn new(device: &wgpu::Device, shadow_dirty_buf: Arc<wgpu::Buffer>) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow.wgsl").into()),
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
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 40,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
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

        // ── Face-index buffer (written once, immutable after unmap) ────────────
        // One u32 per face at FACE_BUF_STRIDE byte intervals.
        // The CPU never touches this buffer after construction.
        let face_idx_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Shadow/FaceIdx"),
            size: MAX_SHADOW_FACES as u64 * FACE_BUF_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: true,
        });
        {
            let mut map = face_idx_buf.slice(..).get_mapped_range_mut();
            for i in 0..MAX_SHADOW_FACES {
                let offset = i * FACE_BUF_STRIDE as usize;
                // Write the face index as a little-endian u32; the rest of the 256-byte
                // slot is zero-initialised by wgpu (mapped buffers are zeroed).
                map[offset..offset + 4].copy_from_slice(&(i as u32).to_ne_bytes());
            }
        }
        face_idx_buf.unmap();

        // ── Atlas texture ──────────────────────────────────────────────────────
        // Dynamic (Movable objects) atlas — re-rendered when Movable entities move.
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow/DynamicAtlas"),
            size: wgpu::Extent3d {
                width: SHADOW_RES,
                height: SHADOW_RES,
                depth_or_array_layers: MAX_SHADOW_FACES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let face_views: Box<[wgpu::TextureView]> = (0..MAX_SHADOW_FACES as u32)
            .map(|i| {
                atlas_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Shadow/DynamicFace"),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow/DynamicAtlasArray"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        // Static (Static/Stationary objects) atlas — re-rendered only when static topology changes.
        let static_atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow/StaticAtlas"),
            size: wgpu::Extent3d {
                width: SHADOW_RES,
                height: SHADOW_RES,
                depth_or_array_layers: MAX_SHADOW_FACES as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let static_face_views: Box<[wgpu::TextureView]> = (0..MAX_SHADOW_FACES as u32)
            .map(|i| {
                static_atlas_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Shadow/StaticFace"),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let static_atlas_view = static_atlas_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow/StaticAtlasArray"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

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
            bgl_0,
            bg_0: None,
            bg_0_key: None,
            static_atlas_cache_gen: None,
            face_idx_buf,
            face_views,
            atlas_tex,
            atlas_view,
            static_face_views,
            static_atlas_tex,
            static_atlas_view,
            compare_sampler,
            _shadow_dirty_buf: shadow_dirty_buf,
            shadow_cache_state: None,
        }
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for ShadowPass {
    fn name(&self) -> &'static str {
        "Shadow"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        // The dynamic atlas contains movable-object shadows.
        frame.shadow_atlas = Some(&self.atlas_view);
        frame.shadow_sampler = Some(&self.compare_sampler);
        // The static atlas contains static-object shadows (cached between frames).
        frame.static_shadow_atlas = Some(&self.static_atlas_view);
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        // TODO: Async dirty flag readback
        // For full shadow caching optimization, we would:
        // 1. Trigger async read of shadow_dirty_staging from previous frame
        // 2. Update self.light_dirty_flags when mapping completes
        // 3. Copy current dirty flags to staging for next frame
        //
        // Skipped for now due to buffer mapping complexity (see earlier GPU timestamp issues)
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let face_count = (ctx.scene.shadow_count as usize).min(MAX_SHADOW_FACES);
        let static_draw_count = ctx.scene.shadow_static_draw_count;
        let movable_draw_count = ctx.scene.shadow_movable_draw_count;

        if face_count == 0 {
            self.shadow_cache_state = None;
            self.static_atlas_cache_gen = None;
            return Ok(());
        }

        // ── Decide which atlas(es) need re-rendering ──────────────────────────
        //
        // Unreal-style static/dynamic shadow split:
        //
        //   Static atlas  — rendered ONCE when Static/Stationary objects change.
        //                   Cached indefinitely while static topology is stable.
        //
        //   Dynamic atlas — rendered only when Movable objects move or lights move.
        //                   For a scene where only the ball moves, 90%+ of shadow
        //                   render cost is eliminated by not re-rendering the
        //                   cathedral walls, floors, columns, etc.

        let static_gen = ctx.scene.static_objects_generation;
        let movable_objects_gen = ctx.scene.movable_objects_generation;
        let movable_lights_gen = ctx.scene.movable_lights_generation;
        let shadow_count = ctx.scene.shadow_count;

        let need_static = self.static_atlas_cache_gen != Some(static_gen)
            || self.shadow_cache_state.map_or(true, |(_, _, c)| c != shadow_count);

        let need_dynamic = {
            let current_state = (movable_objects_gen, movable_lights_gen, shadow_count);
            let changed = self.shadow_cache_state != Some(current_state);
            // Also re-render dynamic if static was just rebuilt (light count may differ)
            changed || need_static
        };

        if !need_static && !need_dynamic {
            return Ok(());
        }

        let main_scene = ctx.resources.main_scene.as_ref().ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig("ShadowPass requires main_scene".into())
        })?;

        let vertices = main_scene.mesh_buffers.vertices;
        let indices = main_scene.mesh_buffers.indices;

        // ── Bind group helpers ─────────────────────────────────────────────────
        // Each bind group uses: [shadow_matrices, <partition instances>, face_idx_buf].
        // Rebuilt only on GrowableBuffer pointer change (O(1) amortised).
        let sm_ptr = ctx.scene.shadow_matrices as *const _ as usize;

        // ── Single shared bind group ───────────────────────────────────────────
        // Both static and dynamic passes use the main `instances` buffer at binding 1.
        // Transforms are always up-to-date here because update_object_transform writes
        // directly into this buffer. Rebuilt only on GrowableBuffer reallocation.
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
        // Only re-rendered when Static/Stationary topology changes.
        // For a typical scene this renders ONCE at load time, never again.
        if need_static {
            let static_indirect = ctx.scene.shadow_static_indirect;
            if static_draw_count > 0 {
                for face in 0..face_count {
                    let face_view = &self.static_face_views[face];
                    let dyn_offset = (face as u64 * FACE_BUF_STRIDE) as u32;
                    let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Shadow/Static"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: face_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
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
            } else {
                // No static shadow casters — clear the atlas to 1.0 (fully lit / no shadow)
                for face in 0..face_count {
                    let face_view = &self.static_face_views[face];
                    let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Shadow/StaticClear"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: face_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
            }
            self.static_atlas_cache_gen = Some(static_gen);
            log::debug!("Shadow: re-rendered static atlas ({} draws, {} faces)", static_draw_count, face_count);
        }

        // ── Dynamic atlas render ───────────────────────────────────────────────
        // Re-rendered only when Movable objects or lights move.
        if need_dynamic {
            let movable_indirect = ctx.scene.shadow_movable_indirect;
            if movable_draw_count > 0 {
                for face in 0..face_count {
                    let face_view = &self.face_views[face];
                    let dyn_offset = (face as u64 * FACE_BUF_STRIDE) as u32;
                    let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Shadow/Dynamic"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: face_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, bg, &[dyn_offset]);
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                    #[cfg(not(target_arch = "wasm32"))]
                    pass.multi_draw_indexed_indirect(movable_indirect, 0, movable_draw_count);
                    #[cfg(target_arch = "wasm32")]
                    for i in 0..movable_draw_count {
                        pass.draw_indexed_indirect(movable_indirect, i as u64 * 20);
                    }
                }
            } else {
                // No movable shadow casters — clear to 1.0 (fully lit)
                for face in 0..face_count {
                    let face_view = &self.face_views[face];
                    let _pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("Shadow/DynamicClear"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: face_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
            }
            self.shadow_cache_state = Some((movable_objects_gen, movable_lights_gen, shadow_count));
        }

        Ok(())
    }
}

