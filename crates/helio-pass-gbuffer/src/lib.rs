//! G-Buffer pass.
//!
//! Renders opaque geometry to 4 render targets (albedo, normal, ORM, emissive) + depth.
//! O(1) CPU: single `multi_draw_indexed_indirect` call regardless of scene size.
//!
//! # Render Targets (owned by this pass)
//!
//! | Slot | Name     | Format        | Contents                          |
//! |------|----------|---------------|-----------------------------------|
//! | 0    | albedo   | Rgba8Unorm    | albedo.rgb + alpha                |
//! | 1    | normal   | Rgba16Float   | world normal.xyz + F0.r           |
//! | 2    | orm      | Rgba8Unorm    | AO, roughness, metallic, F0.g     |
//! | 3    | emissive | Rgba16Float   | emissive.rgb + F0.b               |
//!
//! # Material Bind Group
//!
//! Group 1 (textures) uses a placeholder white 1×1 for all texture slots so the
//! pipeline is fully bound at creation time.  In production, the material system
//! should supply per-draw group 1 bind groups.
//!
//! # Vertex / Index Buffers
//!
//! This pass owns no mesh data.  The caller must bind the shared mesh vertex
//! buffer (slot 0) and index buffer before this pass executes.

use bytemuck::{Pod, Zeroable};
use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};

// ── Uniform types ─────────────────────────────────────────────────────────────

/// Per-frame globals uploaded to the GPU each frame (matches `Globals` in gbuffer.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GBufferGlobals {
    pub frame:             u32,
    pub delta_time:        f32,
    pub light_count:       u32,
    pub ambient_intensity: f32,
    pub ambient_color:     [f32; 4],
    pub rc_world_min:      [f32; 4],
    pub rc_world_max:      [f32; 4],
    pub csm_splits:        [f32; 4],
    pub debug_mode:        u32,
    pub _pad0:             u32,
    pub _pad1:             u32,
    pub _pad2:             u32,
}

/// Placeholder material — opaque white, no alpha cutoff, metallic-roughness workflow.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PlaceholderMaterial {
    base_color:      [f32; 4],
    metallic:        f32,
    roughness:       f32,
    emissive_factor: f32,
    ao:              f32,
    emissive_color:  [f32; 3],
    alpha_cutoff:    f32,
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

pub struct GBufferPass {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bind_group_layout_0: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bind_group_layout_1: wgpu::BindGroupLayout,
    /// Group 0: camera + globals + instance_data.
    bind_group_0: wgpu::BindGroup,
    /// Placeholder group 1 (all textures = white 1×1).
    placeholder_bind_group_1: wgpu::BindGroup,
    /// Per-frame globals uploaded in `prepare()`.
    globals_buf: wgpu::Buffer,
    // ── GBuffer textures (owned; exposed for downstream passes) ───────────────
    pub albedo_tex:   wgpu::Texture,
    pub albedo_view:  wgpu::TextureView,
    pub normal_tex:   wgpu::Texture,
    pub normal_view:  wgpu::TextureView,
    pub orm_tex:      wgpu::Texture,
    pub orm_view:     wgpu::TextureView,
    pub emissive_tex: wgpu::Texture,
    pub emissive_view: wgpu::TextureView,
}

impl GBufferPass {
    /// Create the GBuffer pass.
    ///
    /// * `camera_buf`    – scene camera uniform buffer
    /// * `instances_buf` – per-instance transform storage buffer
    /// * `width/height`  – initial render resolution
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        instances_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> Self {
        // ── Shader ────────────────────────────────────────────────────────────
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GBuffer Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/gbuffer.wgsl").into(),
            ),
        });

        // ── Globals buffer ────────────────────────────────────────────────────
        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GBufferGlobals"),
            size: std::mem::size_of::<GBufferGlobals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Bind Group Layout 0 ───────────────────────────────────────────────
        let bind_group_layout_0 =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GBuffer BGL 0"),
                entries: &[
                    // binding 0: camera (uniform, VERTEX | FRAGMENT)
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // binding 1: globals (uniform, FRAGMENT)
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
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

        // ── Bind Group Layout 1: material + textures ──────────────────────────
        let bind_group_layout_1 = create_gbuffer_material_bgl(device);

        // ── Pipeline ──────────────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GBuffer PL"),
            bind_group_layouts: &[&bind_group_layout_0, &bind_group_layout_1],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("GBuffer Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // Full vertex layout (stride = 32 bytes, matching shared mesh buffer).
                //   offset  0 — position       Float32x3  location 0
                //   offset 12 — bitangent_sign Float32    location 1
                //   offset 16 — tex_coords     Float32x2  location 2
                //   offset 24 — normal         Uint32     location 3
                //   offset 28 — tangent        Uint32     location 4
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 32,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: 12,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: 16,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 24,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: 28,
                            shader_location: 4,
                        },
                    ],
                }],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                    Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba16Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Equal, // depth prepass already ran
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ── GBuffer textures ──────────────────────────────────────────────────
        let (albedo_tex, albedo_view) = gbuffer_texture(
            device, width, height, wgpu::TextureFormat::Rgba8Unorm, "GBuffer/Albedo",
        );
        let (normal_tex, normal_view) = gbuffer_texture(
            device, width, height, wgpu::TextureFormat::Rgba16Float, "GBuffer/Normal",
        );
        let (orm_tex, orm_view) = gbuffer_texture(
            device, width, height, wgpu::TextureFormat::Rgba8Unorm, "GBuffer/ORM",
        );
        let (emissive_tex, emissive_view) = gbuffer_texture(
            device, width, height, wgpu::TextureFormat::Rgba16Float, "GBuffer/Emissive",
        );

        // ── Bind groups ───────────────────────────────────────────────────────
        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GBuffer BG 0"),
            layout: &bind_group_layout_0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: globals_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: instances_buf.as_entire_binding(),
                },
            ],
        });

        let placeholder_bind_group_1 =
            create_placeholder_material_bg(device, &bind_group_layout_1);

        Self {
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            bind_group_0,
            placeholder_bind_group_1,
            globals_buf,
            albedo_tex,
            albedo_view,
            normal_tex,
            normal_view,
            orm_tex,
            orm_view,
            emissive_tex,
            emissive_view,
        }
    }
}

impl RenderPass for GBufferPass {
    fn name(&self) -> &'static str {
        "GBuffer"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Upload per-frame globals (O(1) — fixed-size struct).
        let globals = GBufferGlobals {
            frame:             ctx.frame as u32,
            delta_time:        0.016,   // TODO: expose delta_time in PrepareContext
            light_count:       ctx.scene.lights.len() as u32,
            ambient_intensity: 0.1,
            ambient_color:     [0.1, 0.1, 0.15, 1.0],
            rc_world_min:      [-100.0, -100.0, -100.0, 0.0],
            rc_world_max:      [100.0, 100.0, 100.0, 0.0],
            csm_splits:        [5.0, 20.0, 60.0, 200.0],
            debug_mode:        0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        ctx.queue.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single multi_draw_indexed_indirect — no CPU loop over draw calls.
        let draw_count = ctx.scene.draw_count;
        if draw_count == 0 {
            return Ok(());
        }

        let indirect = ctx.scene.indirect;

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GBuffer"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.albedo_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.normal_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.orm_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &self.emissive_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: ctx.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, // depth prepass already wrote depth
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group_0, &[]);
        // Group 1: placeholder material (real material system sets this per-draw).
        pass.set_bind_group(1, &self.placeholder_bind_group_1, &[]);
        // TODO: Caller must set_vertex_buffer(0, mesh_vb) and set_index_buffer
        //       before this pass, matching the 32-byte stride vertex layout.
        pass.multi_draw_indexed_indirect(indirect, 0, draw_count);
        Ok(())
    }
}

impl GBufferPass {
    /// Recreates GBuffer textures at a new resolution (call on window resize).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (albedo_tex, albedo_view) = gbuffer_texture(
            device, width, height, wgpu::TextureFormat::Rgba8Unorm, "GBuffer/Albedo",
        );
        let (normal_tex, normal_view) = gbuffer_texture(
            device, width, height, wgpu::TextureFormat::Rgba16Float, "GBuffer/Normal",
        );
        let (orm_tex, orm_view) = gbuffer_texture(
            device, width, height, wgpu::TextureFormat::Rgba8Unorm, "GBuffer/ORM",
        );
        let (emissive_tex, emissive_view) = gbuffer_texture(
            device, width, height, wgpu::TextureFormat::Rgba16Float, "GBuffer/Emissive",
        );
        self.albedo_tex = albedo_tex;
        self.albedo_view = albedo_view;
        self.normal_tex = normal_tex;
        self.normal_view = normal_view;
        self.orm_tex = orm_tex;
        self.orm_view = orm_view;
        self.emissive_tex = emissive_tex;
        self.emissive_view = emissive_view;
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Create a screen-sized GBuffer texture and its default view.
fn gbuffer_texture(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    format: wgpu::TextureFormat,
    label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Create a 1×1 white `Rgba8Unorm` placeholder texture.
fn create_white_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("GBuffer Placeholder White 1x1"),
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
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

/// Build the BGL for group 1 (material + 6 textures + 1 sampler).
fn create_gbuffer_material_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };

    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("GBuffer BGL 1"),
        entries: &[
            // binding 0: material uniform
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
            tex_entry(1), // base_color_texture
            tex_entry(2), // normal_map
            // binding 3: sampler
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            tex_entry(4), // orm_texture
            tex_entry(5), // emissive_texture
            tex_entry(6), // specular_color_texture
            tex_entry(7), // specular_weight_texture
        ],
    })
}

/// Create the placeholder material bind group for group 1.
fn create_placeholder_material_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
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
        label: Some("GBuffer PlaceholderMaterial"),
        size: std::mem::size_of::<PlaceholderMaterial>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    material_buf
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::bytes_of(&material));
    material_buf.unmap();

    let (_tex1, white1) = create_white_texture(device);
    let (_tex2, white2) = create_white_texture(device);
    let (_tex3, white3) = create_white_texture(device);
    let (_tex4, white4) = create_white_texture(device);
    let (_tex5, white5) = create_white_texture(device);
    let (_tex6, white6) = create_white_texture(device);

    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("GBuffer PlaceholderSampler"),
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("GBuffer PlaceholderMaterial BG"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: material_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&white1) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&white2) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&sampler) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&white3) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&white4) },
            wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&white5) },
            wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&white6) },
        ],
    })
}
