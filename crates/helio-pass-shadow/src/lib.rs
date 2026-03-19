//! Shadow atlas pass.
//!
//! Renders shadow-casting geometry into a depth texture array (2048×2048×24).
//! One `multi_draw_indexed_indirect` call per shadow face (cascade/cube-face).
//!
//! # O(1) guarantee
//!
//! The execute loop runs at most `MAX_SHADOW_FACES` = 24 iterations, which is a
//! compile-time constant — it does **not** grow with scene complexity.
//!
//! # Vertex / Index Buffers
//!
//! This pass owns no mesh data.  The caller must bind the shared mesh vertex
//! buffer (slot 0) and index buffer before this pass executes.
//!
//! # Material Bind Group
//!
//! A placeholder material bind group (group 1) is used so the pipeline is fully
//! bound.  In production, the material system should supply per-material group 1
//! bind groups and this pass should be driven by a material-aware draw loop.

use bytemuck::{Pod, Zeroable};
use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};

/// Maximum number of shadow faces: 4 lights × 6 cube-faces.
const MAX_SHADOW_FACES: u32 = 24;
/// Resolution of each shadow atlas face.
const SHADOW_ATLAS_SIZE: u32 = 2048;

// ── Uniform types ─────────────────────────────────────────────────────────────

/// Per-face uniform that tells the vertex shader which light matrix to use.
/// Padded to 16 bytes (minimum wgpu uniform buffer size).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShadowLayerUniform {
    layer_idx: u32,
    _pad: [u32; 3],
}

/// Placeholder material — opaque white, no alpha cutoff.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaceholderMaterial {
    base_color:      [f32; 4],  // [1,1,1,1]
    metallic:        f32,
    roughness:       f32,
    emissive_factor: f32,
    ao:              f32,
    emissive_color:  [f32; 3],
    alpha_cutoff:    f32,       // 0.0 — disables alpha cutoff
    workflow:        u32,
    workflow_flags:  u32,
    _pad0:           [u32; 2],
    specular_color:  [f32; 3],
    specular_weight: f32,
    ior:             f32,
    dielectric_f0:   f32,
    _reserved:       [f32; 2],
}

// ── Pass struct ───────────────────────────────────────────────────────────────

pub struct ShadowPass {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bind_group_layout_0: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bind_group_layout_1: wgpu::BindGroupLayout,
    /// One uniform buffer per face, each holding its `layer_idx`.
    layer_uniform_bufs: Vec<wgpu::Buffer>,
    /// One bind group (group 0) per face: shadow_matrices + layer_idx + instance_data.
    /// Rebuilt lazily when shadow_matrices or instances buffer pointers change.
    bind_groups_0: Vec<wgpu::BindGroup>,
    bind_groups_0_key: Option<(usize, usize)>,
    /// Shared placeholder bind group for group 1 (material + textures).
    placeholder_bind_group_1: wgpu::BindGroup,
    /// Full 2D-array depth texture (owned; exposed for downstream passes).
    pub shadow_texture: wgpu::Texture,
    /// Per-layer 2D views (one per shadow face) used as render targets.
    pub shadow_views: Vec<wgpu::TextureView>,
    /// Full array view (used by lighting passes to sample the atlas).
    pub shadow_atlas_view: wgpu::TextureView,
    /// Comparison sampler for sampling the shadow atlas in lighting passes.
    shadow_compare_sampler: wgpu::Sampler,
}

impl ShadowPass {
    /// Create the shadow pass.
    pub fn new(device: &wgpu::Device) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/shadow.wgsl").into(),
            ),
        });

        // ── Bind Group Layout 0: scene bindings ───────────────────────────────
        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow BGL 0"),
                entries: &[
                    // binding 0: light_matrices (storage read, VERTEX)
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
                    // binding 1: shadow_layer_idx (uniform, VERTEX)
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
                    // binding 2: instance_data (storage read, VERTEX)
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

        // ── Bind Group Layout 1: material bindings ────────────────────────────
        let bind_group_layout_1 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Shadow BGL 1"),
                entries: &[
                    // binding 0: material uniform (FRAGMENT)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: base_color_texture (FRAGMENT)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: true,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // binding 3: material_sampler (FRAGMENT) — note: binding 2 unused in shader
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(
                            wgpu::SamplerBindingType::Filtering,
                        ),
                        count: None,
                    },
                ],
            });

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow PL"),
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // Vertex layout matches PackedVertex (stride = 40 bytes).
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 40, // PackedVertex: pos(12)+bitan(4)+uv0(8)+uv1(8)+normal(4)+tangent(4)
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0, // position
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 16,
                            shader_location: 2, // tex_coords
                        },
                    ],
                }],
            },
            // Fragment shader only runs discard for alpha-cutout geometry.
            // No colour outputs — only depth writes.
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Front), // front-face culling reduces shadow acne
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── Shadow atlas texture ───────────────────────────────────────────────
        let shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowAtlas"),
            size: wgpu::Extent3d {
                width: SHADOW_ATLAS_SIZE,
                height: SHADOW_ATLAS_SIZE,
                depth_or_array_layers: MAX_SHADOW_FACES,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        // Per-face 2D views (render targets).
        let shadow_views: Vec<wgpu::TextureView> = (0..MAX_SHADOW_FACES)
            .map(|i| {
                shadow_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some(&format!("ShadowFace_{i}")),
                    format: Some(wgpu::TextureFormat::Depth32Float),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: i,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        // Full-array view for downstream sampling.
        let shadow_atlas_view = shadow_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("ShadowAtlas"),
            format: Some(wgpu::TextureFormat::Depth32Float),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let shadow_compare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ShadowAtlas Compare Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        // ── Per-face layer uniform buffers (fixed; written once at creation) ────
        let mut layer_uniform_bufs = Vec::with_capacity(MAX_SHADOW_FACES as usize);

        for i in 0..MAX_SHADOW_FACES {
            let uniform = ShadowLayerUniform {
                layer_idx: i,
                _pad: [0; 3],
            };
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("ShadowLayerUniform_{i}")),
                size: std::mem::size_of::<ShadowLayerUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: true,
            });
            buf.slice(..)
                .get_mapped_range_mut()
                .copy_from_slice(bytemuck::bytes_of(&uniform));
            buf.unmap();
            layer_uniform_bufs.push(buf);
        }

        // ── Placeholder group 1 (white 1×1 texture + default material) ─────────
        let placeholder_bind_group_1 =
            create_placeholder_material_bg(device, &bind_group_layout_1);

        Self {
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            layer_uniform_bufs,
            bind_groups_0: Vec::new(),
            bind_groups_0_key: None,
            placeholder_bind_group_1,
            shadow_texture,
            shadow_views,
            shadow_atlas_view,
            shadow_compare_sampler,
        }
    }
}

impl RenderPass for ShadowPass {
    fn name(&self) -> &'static str {
        "Shadow"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.shadow_atlas = Some(&self.shadow_atlas_view);
        frame.shadow_sampler = Some(&self.shadow_compare_sampler);
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let draw_count = ctx.scene.draw_count;
        if draw_count == 0 {
            return Ok(());
        }
        let main_scene = ctx
            .frame
            .main_scene
            .as_ref()
            .ok_or_else(|| helio_v3::Error::InvalidPassConfig(
                "ShadowPass requires main_scene mesh buffers".to_string(),
            ))?;

        // O(1): bounded by MAX_SHADOW_FACES (compile-time constant), not scene size.
        let face_count = ctx.scene.shadow_count.min(MAX_SHADOW_FACES);
        let indirect = ctx.scene.indirect;

        // Rebuild all face bind groups when shadow_matrices or instances buffer pointers change.
        let sm_ptr = ctx.scene.shadow_matrices as *const _ as usize;
        let inst_ptr = ctx.scene.instances as *const _ as usize;
        let key = (sm_ptr, inst_ptr);
        if self.bind_groups_0_key != Some(key) {
            log::debug!("Shadow: rebuilding all {} face bind groups (buffer pointers changed)", MAX_SHADOW_FACES);
            self.bind_groups_0 = (0..MAX_SHADOW_FACES as usize)
                .map(|i| {
                    ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some(&format!("Shadow BG0 face {i}")),
                        layout: &self.bind_group_layout_0,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: ctx.scene.shadow_matrices.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: self.layer_uniform_bufs[i].as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: ctx.scene.instances.as_entire_binding(),
                            },
                        ],
                    })
                })
                .collect();
            self.bind_groups_0_key = Some(key);
        }

        let mesh_vertices = main_scene.mesh_buffers.vertices;
        let mesh_indices = main_scene.mesh_buffers.indices;

        for face in 0..face_count {
            let face_idx = face as usize;
            // Pre-borrow from self before the mutable encoder borrow begins.
            let shadow_view = &self.shadow_views[face_idx];
            let bg0 = &self.bind_groups_0[face_idx];
            let bg1 = &self.placeholder_bind_group_1;
            let pipeline = &self.pipeline;

            let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Shadow"),
                color_attachments: &[], // depth-only
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg0, &[]);
            pass.set_bind_group(1, bg1, &[]);
            pass.set_vertex_buffer(0, mesh_vertices.slice(..));
            pass.set_index_buffer(mesh_indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
        }

        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Create a 1×1 white `Rgba8Unorm` placeholder texture.
fn create_white_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Placeholder White 1x1"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

/// Create the placeholder material bind group for group 1.
fn create_placeholder_material_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    // Opaque white, alpha_cutoff = 0 (no cutoff), so no fragments are discarded.
    let material = PlaceholderMaterial {
        base_color:      [1.0, 1.0, 1.0, 1.0],
        metallic:        0.0,
        roughness:       0.5,
        emissive_factor: 0.0,
        ao:              1.0,
        emissive_color:  [0.0, 0.0, 0.0],
        alpha_cutoff:    0.0,
        workflow:        0,
        workflow_flags:  0,
        _pad0:           [0; 2],
        specular_color:  [0.04, 0.04, 0.04],
        specular_weight: 1.0,
        ior:             1.5,
        dielectric_f0:   0.04,
        _reserved:       [0.0; 2],
    };
    let material_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("PlaceholderMaterial"),
        size: std::mem::size_of::<PlaceholderMaterial>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    material_buf
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::bytes_of(&material));
    material_buf.unmap();

    let (_tex, tex_view) = create_white_texture(device);

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("PlaceholderSampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Shadow PlaceholderMaterial BG"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: material_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&tex_view),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    })
}
