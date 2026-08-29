//! Screen-space portal-opening mask pass. See `shaders/portal_mask.wgsl` for
//! the full design rationale; in short, this pass runs two tiny sub-passes
//! immediately before `PortalInstancePass` every frame:
//!
//! 1. Stamp each active portal's real opening quad into `portal_mask`,
//!    depth-tested (read-only) against the real scene so occluded portals
//!    correctly stay unmasked.
//! 2. Reset the real depth buffer to the far plane wherever that mask
//!    landed, so the duplicate-content pass's own depth test self-occludes
//!    correctly among the many duplicated copies instead of comparing
//!    against unrelated nearby real geometry.

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub struct PortalMaskPass {
    stamp_pipeline: wgpu::RenderPipeline,
    stamp_bgl: wgpu::BindGroupLayout,
    stamp_bind_group: Option<wgpu::BindGroup>,
    stamp_bind_group_key: Option<(usize, usize)>,

    reset_pipeline: wgpu::RenderPipeline,
    reset_bgl: wgpu::BindGroupLayout,
    reset_bind_group: Option<wgpu::BindGroup>,
    reset_bind_group_key: Option<usize>,

    portal_count: u32,
}

impl PortalMaskPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PortalMask Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/portal_mask.wgsl").into()),
        });

        // ── Stamp pipeline: draw each portal's real opening quad, testing
        // (read-only) against real depth, writing portal_index+1 to the mask.
        let stamp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PortalMask Stamp BGL"),
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
        let stamp_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PortalMask Stamp PL"),
            bind_group_layouts: &[Some(&stamp_bgl)],
            immediate_size: 0,
        });
        let stamp_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PortalMask Stamp Pipeline"),
            layout: Some(&stamp_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_stamp"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_stamp"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R32Uint,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // Both winding directions valid — sign of the composed
                // transform's determinant isn't guaranteed either way.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Writes ARE enabled here, even though this same depth
                // buffer holds GBufferPass's real scene depth. That real
                // depth still isn't disturbed for anything downstream: every
                // pixel this stamp can possibly touch is, by construction,
                // unconditionally overwritten again a moment later by
                // PortalMaskReset (below) to a single canonical far-plane
                // value, so nothing after this pass ever observes whatever
                // depth a portal quad happened to write. What the write
                // *does* do is give multiple portals' quads correct
                // depth-ordering against each other within this one draw
                // call — e.g. two faces of a portal cube both filling the
                // screen from an oblique angle, with no real wall between
                // them to occlude the farther one via the real-depth test
                // alone. Without this, whichever portal happened to
                // rasterize last simply overwrote the mask at every
                // overlapping pixel regardless of which one was actually
                // nearer — the "z-fighting"/wrong-layering symptom. With it,
                // ordinary depth buffering resolves overlaps correctly
                // regardless of instance submission order: nearer-portal
                // fragments always win, the same as any other opaque
                // geometry.
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                // The quad is meant to be flush with the portal's real
                // surrounding surface (so content lines up exactly with no
                // seam) — e.g. a portal placed right at the end of a
                // corridor is coplanar with that corridor's own walls right
                // at the boundary. Without a nudge, that coincident geometry
                // is a coin-flip depth tie at the portal's edge, so the
                // stamp randomly loses a thin border of pixels there and
                // leaves them unmasked — a visible gap between the portal's
                // content and the real scene around it. Push the quad
                // slightly toward the camera so it reliably wins.
                bias: wgpu::DepthBiasState {
                    constant: -2,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        // ── Reset pipeline: full-screen triangle, writes far depth wherever
        // the mask just stamped is non-zero.
        let reset_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PortalMask Reset BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Uint,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            }],
        });
        let reset_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PortalMask Reset PL"),
            bind_group_layouts: &[Some(&reset_bgl)],
            immediate_size: 0,
        });
        let reset_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PortalMask Reset Pipeline"),
            layout: Some(&reset_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_reset"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_reset"),
                compilation_options: Default::default(),
                targets: &[],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Unconditional write for surviving (non-discarded) fragments
                // — the mask lookup in fs_reset is the real gate, not depth.
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            stamp_pipeline,
            stamp_bgl,
            stamp_bind_group: None,
            stamp_bind_group_key: None,
            reset_pipeline,
            reset_bgl,
            reset_bind_group: None,
            reset_bind_group_key: None,
            portal_count: 0,
        }
    }
}

impl RenderPass for PortalMaskPass {
    fn name(&self) -> &'static str {
        "PortalMask"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw(
            "portal_mask",
            wgpu::TextureFormat::R32Uint,
            ResourceSize::MatchSurface,
        );
        builder.with_extra_usage(wgpu::TextureUsages::TEXTURE_BINDING);
    }

    fn reads(&self) -> &'static [&'static str] {
        &["depth"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["portal_mask"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // Standalone — see `execute`, which opens its own two render passes
        // directly via `ctx.begin_render_pass`.
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        self.portal_count = ctx.scene.portal_views.len() as u32;
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.portal_count == 0 {
            return Ok(());
        }
        let Some(mask_view) = ctx.resource_pool.get_view("portal_mask") else {
            log::warn!(
                "[PortalMask] frame={} portal_mask resource not allocated",
                ctx.frame_num
            );
            return Ok(());
        };

        // ── Sub-pass 1: stamp ────────────────────────────────────────────
        let stamp_key = (
            ctx.scene.camera as *const _ as usize,
            ctx.scene.portal_views as *const _ as usize,
        );
        if self.stamp_bind_group_key != Some(stamp_key) {
            self.stamp_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PortalMask Stamp BG"),
                    layout: &self.stamp_bgl,
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
            self.stamp_bind_group_key = Some(stamp_key);
        }

        {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: mask_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.0,
                        g: 0.0,
                        b: 0.0,
                        a: 0.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("PortalMask Stamp"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.stamp_pipeline);
            pass.set_bind_group(0, self.stamp_bind_group.as_ref().unwrap(), &[]);
            pass.draw(0..6, 0..self.portal_count);
        }

        // ── Sub-pass 2: reset ────────────────────────────────────────────
        let reset_key = mask_view as *const _ as usize;
        if self.reset_bind_group_key != Some(reset_key) {
            self.reset_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PortalMask Reset BG"),
                    layout: &self.reset_bgl,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(mask_view),
                    }],
                }));
            self.reset_bind_group_key = Some(reset_key);
        }

        {
            let desc = wgpu::RenderPassDescriptor {
                label: Some("PortalMask Reset"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.reset_pipeline);
            pass.set_bind_group(0, self.reset_bind_group.as_ref().unwrap(), &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
