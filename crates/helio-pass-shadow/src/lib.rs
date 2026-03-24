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

    /// Single bind group for the whole pass.  Rebuilt when buffer pointers change.
    bg_0: Option<wgpu::BindGroup>,
    /// (shadow_matrices ptr, instances ptr) — detects GrowableBuffer reallocations.
    bg_0_key: Option<(usize, usize)>,

    /// Per-face face-index values, written once at construction and never touched again.
    /// Layout: `face_idx_buf[face * FACE_BUF_STRIDE]` = `u32(face)` followed by 252 zero bytes.
    face_idx_buf: wgpu::Buffer,

    /// Per-face depth-only render-target views into the atlas array.
    /// `Box<[_]>` instead of `Vec` — capacity is fixed at MAX_SHADOW_FACES forever.
    face_views: Box<[wgpu::TextureView]>,

    /// The underlying atlas texture (owned).
    pub atlas_tex: wgpu::Texture,

    /// `D2Array` view over the full atlas — consumed by the deferred lighting pass.
    pub atlas_view: wgpu::TextureView,

    /// PCF comparison sampler (`LessEqual`) — consumed by the deferred lighting pass.
    pub compare_sampler: wgpu::Sampler,
}

impl ShadowPass {
    /// Allocate all GPU resources.  Called once; zero allocations after this.
    pub fn new(device: &wgpu::Device) -> Self {
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
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Shadow/Atlas"),
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

        // Per-face 2D render-target views — fixed capacity Box, never grows.
        let face_views: Box<[wgpu::TextureView]> = (0..MAX_SHADOW_FACES as u32)
            .map(|i| {
                atlas_tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("Shadow/Face"),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // Full D2Array view for downstream sampling.
        let atlas_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Shadow/AtlasArray"),
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
            face_idx_buf,
            face_views,
            atlas_tex,
            atlas_view,
            compare_sampler,
        }
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for ShadowPass {
    fn name(&self) -> &'static str {
        "Shadow"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.shadow_atlas = Some(&self.atlas_view);
        frame.shadow_sampler = Some(&self.compare_sampler);
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        // Zero per-frame CPU work: nothing to upload.
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let draw_count = ctx.scene.draw_count;
        let face_count = (ctx.scene.shadow_count as usize).min(MAX_SHADOW_FACES);

        if draw_count == 0 || face_count == 0 {
            return Ok(());
        }

        let main_scene = ctx.frame.main_scene.as_ref().ok_or_else(|| {
            helio_v3::Error::InvalidPassConfig("ShadowPass requires main_scene".into())
        })?;

        // ── Bind group — rebuilt only on GrowableBuffer reallocation ──────────
        // In steady state (no scene growth) this branch is never taken.
        // O(1) amortised: pointer changes happen at most O(log N) total across the
        // entire lifetime of the scene.
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
                        // Bind the first 16 bytes of the face-index buffer.
                        // The actual face is selected via the dynamic offset below.
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

        // ── Per-face render loop ───────────────────────────────────────────────
        // face_count ≤ MAX_SHADOW_FACES (compile-time constant = 256).
        // Each iteration: O(1) — only constant-size wgpu calls, no allocations.
        let bg = self.bg_0.as_ref().unwrap();
        let pipeline = &self.pipeline;
        let vertices = main_scene.mesh_buffers.vertices;
        let indices = main_scene.mesh_buffers.indices;
        let indirect = ctx.scene.indirect;

        for face in 0..face_count {
            let face_view = &self.face_views[face];
            // Byte offset into face_idx_buf that holds the u32 for this face.
            let dyn_offset = (face as u64 * FACE_BUF_STRIDE) as u32;

            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow"),
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
            pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
            #[cfg(target_arch = "wasm32")]
            for i in 0..draw_count {
                pass.draw_indexed_indirect(indirect, i as u64 * 20);
            }
        }

        Ok(())
    }
}

