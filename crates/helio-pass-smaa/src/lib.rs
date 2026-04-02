//! Subpixel Morphological Anti-Aliasing (SMAA) pass.
//!
//! Implements SMAA as three sequential fullscreen passes:
//!
//! 1. **Edge detection** (`smaa_edge.wgsl`)  — luma-based edge map → `edge_view` (Rg16Float)
//! 2. **Blend weights** (`smaa_blend.wgsl`)  — computes blend weights → `blend_view` (Rgba8Unorm)
//! 3. **Neighborhood blending** (`smaa_neighbor.wgsl`) — applies AA → `ctx.target`
//!
//! ## O(1) guarantee
//! `execute()` records exactly **3** `draw(0..3, 0..1)` calls regardless of scene size.

use helio_v3::{PassContext, RenderPass, Result as HelioResult};

/// SMAA pass (3 sequential fullscreen draws).
pub struct SmaaPass {
    edge_pipeline: wgpu::RenderPipeline,
    edge_bind_group: wgpu::BindGroup,

    blend_pipeline: wgpu::RenderPipeline,
    blend_bind_group: wgpu::BindGroup,

    neighbor_pipeline: wgpu::RenderPipeline,
    neighbor_bind_group: wgpu::BindGroup,

    pub edge_texture: wgpu::Texture,
    pub edge_view: wgpu::TextureView,
    pub blend_texture: wgpu::Texture,
    pub blend_view: wgpu::TextureView,

    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,
}

impl SmaaPass {
    /// Create the SMAA pass.
    ///
    /// `input_view` — the pre-AA colour buffer (same view used by both edge and neighbor passes).
    /// `target_format` — format of `ctx.target`.
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        input_view: &wgpu::TextureView,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let edge_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SMAA Edge Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/smaa_edge.wgsl").into()),
        });
        let blend_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SMAA Blend Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/smaa_blend.wgsl").into()),
        });
        let neighbor_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SMAA Neighbor Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/smaa_neighbor.wgsl").into()),
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMAA Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("SMAA Point Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Intermediate textures
        let edge_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMAA Edge Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            // Rg16Float: 2-channel float, sufficient for vec2<f32> edge output
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let edge_view = edge_texture.create_view(&Default::default());

        let blend_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SMAA Blend Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let blend_view = blend_texture.create_view(&Default::default());

        // All three shaders share the same BGL layout (tex + linear + point)
        let bgl = |label: &'static str, tex_filterable: bool| -> wgpu::BindGroupLayout {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float {
                                filterable: tex_filterable,
                            },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            })
        };

        let make_bg = |layout: &wgpu::BindGroupLayout,
                       label: &'static str,
                       tex: &wgpu::TextureView|
         -> wgpu::BindGroup {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(tex),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&linear_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&point_sampler),
                    },
                ],
            })
        };

        let make_pipeline = |label: &'static str,
                             shader: &wgpu::ShaderModule,
                             layout: &wgpu::BindGroupLayout,
                             fmt: wgpu::TextureFormat|
         -> wgpu::RenderPipeline {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: fmt,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            })
        };

        // Edge detection: input (filterable) → Rg16Float
        let edge_bgl = bgl("SMAA Edge BGL", true);
        let edge_bind_group = make_bg(&edge_bgl, "SMAA Edge BG", input_view);
        let edge_pipeline = make_pipeline(
            "SMAA Edge Pipeline",
            &edge_shader,
            &edge_bgl,
            wgpu::TextureFormat::Rg16Float,
        );

        // Blend weight: edge_view → Rgba8Unorm
        let blend_bgl = bgl("SMAA Blend BGL", true);
        let blend_bind_group = make_bg(&blend_bgl, "SMAA Blend BG", &edge_view);
        let blend_pipeline = make_pipeline(
            "SMAA Blend Pipeline",
            &blend_shader,
            &blend_bgl,
            wgpu::TextureFormat::Rgba8Unorm,
        );

        // Neighborhood blending: input → ctx.target format
        let neighbor_bgl = bgl("SMAA Neighbor BGL", true);
        let neighbor_bind_group = make_bg(&neighbor_bgl, "SMAA Neighbor BG", input_view);
        let neighbor_pipeline = make_pipeline(
            "SMAA Neighbor Pipeline",
            &neighbor_shader,
            &neighbor_bgl,
            target_format,
        );

        Self {
            edge_pipeline,
            edge_bind_group,
            blend_pipeline,
            blend_bind_group,
            neighbor_pipeline,
            neighbor_bind_group,
            edge_texture,
            edge_view,
            blend_texture,
            blend_view,
            linear_sampler,
            point_sampler,
        }
    }
}

impl RenderPass for SmaaPass {
    fn name(&self) -> &'static str {
        "SMAA"
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): exactly 3 fullscreen draws regardless of scene size.

        // Pass 1 — edge detection → edge_view
        {
            let color = [Some(wgpu::RenderPassColorAttachment {
                view: &self.edge_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("SMAA Edge"),
                color_attachments: &color,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.encoder.begin_render_pass(&desc);
            pass.set_pipeline(&self.edge_pipeline);
            pass.set_bind_group(0, &self.edge_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 2 — blend weight calculation → blend_view
        {
            let color = [Some(wgpu::RenderPassColorAttachment {
                view: &self.blend_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("SMAA Blend"),
                color_attachments: &color,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.encoder.begin_render_pass(&desc);
            pass.set_pipeline(&self.blend_pipeline);
            pass.set_bind_group(0, &self.blend_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Pass 3 — neighborhood blending → ctx.target
        {
            let target = ctx.target;
            let color = [Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("SMAA Neighbor"),
                color_attachments: &color,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.encoder.begin_render_pass(&desc);
            pass.set_pipeline(&self.neighbor_pipeline);
            pass.set_bind_group(0, &self.neighbor_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

