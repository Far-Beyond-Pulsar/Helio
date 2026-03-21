//! Billboard pass — camera-facing instanced quads.
//!
//! Each billboard is a 6-vertex quad (2 triangles) rendered with alpha blending.
//! Instance buffer is GPU-side; CPU uploads billboard data once per frame and issues
//! a single instanced draw call. O(1) CPU.

use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};
use bytemuck::{Pod, Zeroable};

const MAX_BILLBOARDS: u32 = 65536;

/// Per-billboard instance data uploaded to the GPU.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BillboardInstance {
    /// World-space position (xyz) + unused pad (w).
    pub world_pos:   [f32; 4],
    /// Scale (xy), screen_scale flag as f32 (z), unused (w).
    pub scale_flags: [f32; 4],
    /// RGBA tint color.
    pub color:       [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BillboardGlobals {
    frame:             u32,
    delta_time:        f32,
    ambient_intensity: f32,
    _pad:              f32,
}

pub struct BillboardPass {
    pipeline:       wgpu::RenderPipeline,
    #[allow(dead_code)]
    bgl_0:          wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bgl_1:          wgpu::BindGroupLayout,
    bind_group_0:   wgpu::BindGroup,
    bind_group_1:   wgpu::BindGroup,
    globals_buf:    wgpu::Buffer,
    /// Billboard instance data — caller writes via `update_instances()`.
    pub instance_buf:   wgpu::Buffer,
    quad_vertex_buf:    wgpu::Buffer,
    pub instance_count: u32,
    #[allow(dead_code)]
    white_texture:  wgpu::Texture,
    #[allow(dead_code)]
    white_view:     wgpu::TextureView,
    #[allow(dead_code)]
    sampler:        wgpu::Sampler,
}

impl BillboardPass {
    /// Create the billboard pass.
    ///
    /// - `camera_buf`    — camera uniform (must match `Camera` struct in billboard.wgsl)
    /// - `target_format` — colour attachment format (e.g. `Rgba16Float`)
    pub fn new(
        device:        &wgpu::Device,
        queue:         &wgpu::Queue,
        camera_buf:    &wgpu::Buffer,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Billboard Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/billboard.wgsl").into()),
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Billboard Globals"),
            size:               std::mem::size_of::<BillboardGlobals>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Group 0: camera (b0) + globals (b1) ──────────────────────────────
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Billboard BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        // ── Group 1: sprite_tex (b0) + sprite_sampler (b1) ───────────────────
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("Billboard BGL1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count:      None,
                },
            ],
        });

        // ── 1×1 opaque-white default sprite ──────────────────────────────────
        let white_texture = device.create_texture(&wgpu::TextureDescriptor {
            label:               Some("Billboard White Texture"),
            size:                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           wgpu::TextureDimension::D2,
            format:              wgpu::TextureFormat::Rgba8UnormSrgb,
            usage:               wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:        &[],
        });
        helio_v3::upload::write_texture(
            queue,
            wgpu::ImageCopyTexture {
                texture:   &white_texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::ImageDataLayout {
                offset:         0,
                bytes_per_row:  Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        );
        let white_view = white_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:           Some("Billboard Sampler"),
            address_mode_u:  wgpu::AddressMode::ClampToEdge,
            address_mode_v:  wgpu::AddressMode::ClampToEdge,
            address_mode_w:  wgpu::AddressMode::ClampToEdge,
            mag_filter:      wgpu::FilterMode::Linear,
            min_filter:      wgpu::FilterMode::Linear,
            mipmap_filter:   wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Billboard BG0"),
            layout:  &bgl_0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buf.as_entire_binding() },
            ],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("Billboard BG1"),
            layout:  &bgl_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding:  0,
                    resource: wgpu::BindingResource::TextureView(&white_view),
                },
                wgpu::BindGroupEntry {
                    binding:  1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("Billboard PL"),
            bind_group_layouts:   &[&bgl_0, &bgl_1],
            push_constant_ranges: &[],
        });

        // ── Quad vertex buffer (6 vertices: 2 triangles CCW) ─────────────────
        // Each vertex: [pos.x, pos.y, uv.x, uv.y]
        let quad_verts: [[f32; 4]; 6] = [
            [-0.5, -0.5, 0.0, 0.0],
            [ 0.5, -0.5, 1.0, 0.0],
            [-0.5,  0.5, 0.0, 1.0],
            [-0.5,  0.5, 0.0, 1.0],
            [ 0.5, -0.5, 1.0, 0.0],
            [ 0.5,  0.5, 1.0, 1.0],
        ];
        let quad_vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Billboard Quad VB"),
            size:               std::mem::size_of_val(&quad_verts) as u64,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        helio_v3::upload::write_buffer(queue, &quad_vertex_buf, 0, bytemuck::cast_slice(&quad_verts));

        // ── Instance buffer ───────────────────────────────────────────────────
        let instance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("Billboard Instances"),
            size:               (MAX_BILLBOARDS as usize * std::mem::size_of::<BillboardInstance>()) as u64,
            usage:              wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Render pipeline ───────────────────────────────────────────────────
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Billboard Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    // Slot 0: per-vertex quad data  (stride 16)
                    wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode:    wgpu::VertexStepMode::Vertex,
                        attributes:   &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 0, shader_location: 0 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 8, shader_location: 1 },
                        ],
                    },
                    // Slot 1: per-instance billboard data  (stride 48)
                    wgpu::VertexBufferLayout {
                        array_stride: 48,
                        step_mode:    wgpu::VertexStepMode::Instance,
                        attributes:   &[
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset:  0, shader_location: 2 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 16, shader_location: 3 },
                            wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x4, offset: 32, shader_location: 4 },
                        ],
                    },
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend:  Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology:  wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, // billboards are always camera-facing
                ..Default::default()
            },
            // Depth read-only: billboards depth-test against scene geometry but don't write depth.
            depth_stencil: Some(wgpu::DepthStencilState {
                format:                wgpu::TextureFormat::Depth32Float,
                depth_write_enabled:   false,
                depth_compare:         wgpu::CompareFunction::LessEqual,
                stencil:               wgpu::StencilState::default(),
                bias:                  wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview:   None,
            cache:       None,
        });

        Self {
            pipeline,
            bgl_0,
            bgl_1,
            bind_group_0,
            bind_group_1,
            globals_buf,
            instance_buf,
            quad_vertex_buf,
            instance_count: 0,
            white_texture,
            white_view,
            sampler,
        }
    }

    /// Upload billboard instances. Call once per frame (or when the set changes).
    pub fn update_instances(&mut self, queue: &wgpu::Queue, instances: &[BillboardInstance]) {
        let count = instances.len().min(MAX_BILLBOARDS as usize);
        if count > 0 {
            helio_v3::upload::write_buffer(queue, &self.instance_buf, 0, bytemuck::cast_slice(&instances[..count]));
        }
        self.instance_count = count as u32;
    }
}

impl RenderPass for BillboardPass {
    fn name(&self) -> &'static str { "Billboard" }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Upload billboard instances from the high-level renderer's frame data.
        if let Some(data) = ctx.frame_resources.billboards {
            let max_bytes = MAX_BILLBOARDS as usize * std::mem::size_of::<BillboardInstance>();
            let upload_bytes = data.instances.len().min(max_bytes);
            if upload_bytes > 0 {
                ctx.write_buffer(&self.instance_buf, 0, &data.instances[..upload_bytes]);
            }
            self.instance_count = data.count.min(MAX_BILLBOARDS);
        } else {
            self.instance_count = 0;
        }
        let globals = BillboardGlobals {
            frame:             ctx.frame as u32,
            delta_time:        0.0,
            ambient_intensity: 1.0,
            _pad:              0.0,
        };
        ctx.write_buffer(&self.globals_buf, 0, bytemuck::bytes_of(&globals));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single instanced draw — GPU expands billboards into screen quads.
        if self.instance_count == 0 {
            return Ok(());
        }

        let color_attachment = wgpu::RenderPassColorAttachment {
            view:           ctx.target,
            resolve_target: None,
            ops: wgpu::Operations {
                load:  wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        };
        let depth_attachment = wgpu::RenderPassDepthStencilAttachment {
            view: ctx.depth,
            // Read-only depth: test against opaque scene but do not write.
            depth_ops: Some(wgpu::Operations {
                load:  wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Discard,
            }),
            stencil_ops: None,
        };
        let color_attachments = [Some(color_attachment)];
        let desc = wgpu::RenderPassDescriptor {
            label:                    Some("Billboard"),
            color_attachments:        &color_attachments,
            depth_stencil_attachment: Some(depth_attachment),
            timestamp_writes:         None,
            occlusion_query_set:      None,
        };

        let mut pass = ctx.encoder.begin_render_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group_0, &[]);
        pass.set_bind_group(1, &self.bind_group_1, &[]);
        pass.set_vertex_buffer(0, self.quad_vertex_buf.slice(..));
        pass.set_vertex_buffer(1, self.instance_buf.slice(..));
        pass.draw(0..6, 0..self.instance_count); // O(1) — single GPU draw call
        Ok(())
    }
}
