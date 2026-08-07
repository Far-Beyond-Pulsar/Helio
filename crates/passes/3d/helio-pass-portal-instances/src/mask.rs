//! Screen-space portal-opening mask pass for one recursion *level*. See
//! `shaders/portal_mask.wgsl` for the full design rationale; in short, one
//! instance of this pass per level `1..=MAX_CHAIN_DEPTH` runs immediately
//! before its matching `PortalInstancePass` level, two tiny sub-passes each:
//!
//! 1. Stamp every depth-`level` chain's real opening quad (its final
//!    portal's own geometry, mapped through the chain's parent-prefix
//!    transform) into `portal_mask`, depth-tested (read-only) against
//!    whatever's already been drawn — which for level > 1 includes the
//!    previous level's own duplicate content, so occlusion between
//!    recursion levels composes correctly.
//! 2. Reset the real depth buffer to the far plane wherever that mask
//!    landed, so this level's duplicate-content pass self-occludes
//!    correctly among its own copies instead of comparing against whatever
//!    was there before.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct StampUniform {
    level: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct PortalMaskPass {
    level: u32,

    stamp_pipeline: wgpu::RenderPipeline,
    stamp_bgl: wgpu::BindGroupLayout,
    stamp_uniform_buf: wgpu::Buffer,
    stamp_uniform_written: bool,
    stamp_bind_group: Option<wgpu::BindGroup>,
    stamp_bind_group_key: Option<(usize, usize, usize)>,

    reset_pipeline: wgpu::RenderPipeline,
    reset_bgl: wgpu::BindGroupLayout,
    reset_bind_group: Option<wgpu::BindGroup>,
    reset_bind_group_key: Option<usize>,

    chain_count: u32,
}

impl PortalMaskPass {
    /// `level`: which recursion depth this pass instance stamps —
    /// `1..=libhelio::MAX_CHAIN_DEPTH`. See the module doc.
    pub fn new(device: &wgpu::Device, level: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PortalMask Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/portal_mask.wgsl").into()),
        });

        let stamp_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalMask/StampUniform"),
            size: std::mem::size_of::<StampUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Stamp pipeline: draw every depth-`level` chain's real opening
        // quad (mapped through its parent prefix), testing (read-only)
        // against whatever's currently in depth, writing chain_index+1.
        let stamp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PortalMask Stamp BGL"),
            entries: &[
                storage_entry(0, wgpu::ShaderStages::VERTEX), // cameras
                storage_entry(1, wgpu::ShaderStages::VERTEX), // portal_views
                storage_entry(2, wgpu::ShaderStages::VERTEX), // portal_chains
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, // stamp uniform (level)
                storage_entry(4, wgpu::ShaderStages::VERTEX), // coordinate_spaces
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
                // Read-only: must not disturb real/prior-level depth — the
                // reset sub-pass (below) is what edits it, and only where
                // this stamp actually lands.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                // The quad is meant to be flush with its real surrounding
                // surface (so content lines up exactly with no seam) — e.g.
                // a portal placed right at the end of a corridor is
                // coplanar with that corridor's own walls right at the
                // boundary. Without a nudge, that coincident geometry is a
                // coin-flip depth tie at the portal's edge, so the stamp
                // randomly loses a thin border of pixels there. Push the
                // quad slightly toward the camera so it reliably wins.
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
            level,
            stamp_pipeline,
            stamp_bgl,
            stamp_uniform_buf,
            stamp_uniform_written: false,
            stamp_bind_group: None,
            stamp_bind_group_key: None,
            reset_pipeline,
            reset_bgl,
            reset_bind_group: None,
            reset_bind_group_key: None,
            chain_count: 0,
        }
    }
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl RenderPass for PortalMaskPass {
    fn name(&self) -> &'static str {
        "PortalMask"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // Safe to declare from every level's instance — the graph keeps
        // only the first declaration for a given resource name and every
        // later `read`/`write` of the same name just extends its lifetime,
        // so this doesn't allocate a separate texture per level.
        builder.write_color_raw("portal_mask", wgpu::TextureFormat::R32Uint, ResourceSize::MatchSurface);
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
        self.chain_count = ctx.scene.portal_chains.len() as u32;
        if !self.stamp_uniform_written {
            let data = StampUniform { level: self.level, _pad0: 0, _pad1: 0, _pad2: 0 };
            ctx.write_buffer(&self.stamp_uniform_buf, 0, bytemuck::bytes_of(&data));
            self.stamp_uniform_written = true;
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.chain_count == 0 {
            return Ok(());
        }
        let Some(mask_view) = ctx.resource_pool.get_view("portal_mask") else {
            log::warn!("[PortalMask] frame={} level={} portal_mask resource not allocated", ctx.frame_num, self.level);
            return Ok(());
        };

        // ── Sub-pass 1: stamp ────────────────────────────────────────────
        let stamp_key = (
            ctx.scene.camera as *const _ as usize,
            ctx.scene.portal_views as *const _ as usize,
            ctx.scene.portal_chains as *const _ as usize,
        );
        if self.stamp_bind_group_key != Some(stamp_key) {
            self.stamp_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PortalMask Stamp BG"),
                layout: &self.stamp_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.camera.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: ctx.scene.portal_views.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.scene.portal_chains.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.stamp_uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ctx.scene.coordinate_spaces.as_entire_binding() },
                ],
            }));
            self.stamp_bind_group_key = Some(stamp_key);
        }

        {
            // Every level clears the mask fresh: by the time this stamp
            // runs, this level's matching `PortalInstancePass` instance
            // hasn't drawn yet (it runs right after this pass) and the
            // *previous* level's instance pass already consumed whatever
            // was here before — nothing still needs the old values.
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: mask_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 }),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("PortalMask Stamp"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            };
            let mut pass = ctx.begin_render_pass(&desc);
            pass.set_pipeline(&self.stamp_pipeline);
            pass.set_bind_group(0, self.stamp_bind_group.as_ref().unwrap(), &[]);
            // Every chain, every level — non-matching-depth chains
            // degenerate to nothing in vs_stamp (see that shader).
            pass.draw(0..6, 0..self.chain_count);
        }

        // ── Sub-pass 2: reset ────────────────────────────────────────────
        let reset_key = mask_view as *const _ as usize;
        if self.reset_bind_group_key != Some(reset_key) {
            self.reset_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PortalMask Reset BG"),
                layout: &self.reset_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(mask_view) }],
            }));
            self.reset_bind_group_key = Some(reset_key);
        }

        {
            let desc = wgpu::RenderPassDescriptor {
                label: Some("PortalMask Reset"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: ctx.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
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
