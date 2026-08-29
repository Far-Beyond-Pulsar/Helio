//! Editor-only checkerboard indicator over each portal's opening. See
//! `shaders/portal_editor_overlay.wgsl` for the full rationale. Disabled
//! (zero draws, zero cost) unless [`PortalEditorOverlayPass::set_editor_mode`]
//! has been called with `true` — call it from wherever your application
//! already tracks editor vs. game mode (mirrors `helio::Renderer`'s own
//! `is_editor_mode()` / `set_editor_mode()`, which this pass does not read
//! automatically since plain render passes have no access to the `Renderer`
//! that owns them).

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub struct PortalEditorOverlayPass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(usize, usize)>,
    portal_count: u32,
    editor_mode: bool,
}

impl PortalEditorOverlayPass {
    pub fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PortalEditorOverlay Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/portal_editor_overlay.wgsl").into(),
            ),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PortalEditorOverlay BGL"),
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
                }, // cameras
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // portal_views
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PortalEditorOverlay PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PortalEditorOverlay Pipeline"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        // Leave pre_aa's alpha channel alone — later passes
                        // in the chain (LensFlare/PostProcess) use it for
                        // more than plain transparency, and blending our own
                        // alpha into it (as helio-pass-corona's particles
                        // legitimately do, being the actual scene content)
                        // corrupted whatever they read from it into visible
                        // ray/streak artifacts. This overlay only ever wants
                        // to darken RGB, so just keep destination alpha.
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Zero,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Read-only: respect real occluders in front of the portal
                // (same reasoning as helio-pass-portal-mask's stamp draw) —
                // don't paint the indicator through a wall.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            bind_group: None,
            bind_group_key: None,
            portal_count: 0,
            editor_mode: false,
        }
    }

    /// Enable or disable the checkerboard indicator. Mirrors whatever
    /// game/editor-mode toggle the host application already has — call this
    /// whenever that flips (e.g. alongside `Renderer::set_editor_mode`).
    pub fn set_editor_mode(&mut self, enabled: bool) {
        self.editor_mode = enabled;
    }

    pub fn is_editor_mode(&self) -> bool {
        self.editor_mode
    }
}

impl RenderPass for PortalEditorOverlayPass {
    fn name(&self) -> &'static str {
        "PortalEditorOverlay"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["pre_aa", "depth"]
    }

    fn writes(&self) -> &'static [&'static str] {
        // Draws (LoadOp::Load) directly onto pre_aa — see render_pass_descriptor.
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
    }

    fn set_editor_mode(&mut self, enabled: bool) {
        self.editor_mode = enabled;
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // Always structurally participates in the pre_aa fusion chain,
        // regardless of `editor_mode` — that flag is runtime, mutable state,
        // while chain fusion is decided from attachment identity once (see
        // PortalInstancePass's own doc comment on why). Gating on it here
        // would make this pass flicker in and out of the chain and could
        // break fusion for passes chained through it. `execute()` is where
        // editor_mode actually matters: no draw call, zero cost, when off.
        let target_view = resources.pre_aa.get().unwrap_or(target);
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("PortalEditorOverlay"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.portal_count = ctx.scene.portal_views.len() as u32;
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.editor_mode || self.portal_count == 0 {
            return Ok(());
        }
        let Some(pass_ptr) = ctx.active_render_pass_ptr() else {
            return Ok(());
        };

        let key = (
            ctx.scene.camera as *const _ as usize,
            ctx.scene.portal_views as *const _ as usize,
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PortalEditorOverlay BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: ctx.scene.portal_views.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let pass = unsafe { &mut *pass_ptr };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.draw(0..6, 0..self.portal_count);

        Ok(())
    }
}
