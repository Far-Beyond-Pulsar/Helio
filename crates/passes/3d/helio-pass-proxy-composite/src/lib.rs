//! Proxy composite — merge secondary G-buffers (portal eyes, sublevel
//! cameras) into their composite destination.
//!
//! Self-managed (`render_pass_descriptor` returns `None`, `ShadowPass`'s
//! pattern): every active view opens its **own** render pass directly on
//! `ctx.encoder_ptr`. This is what makes recursive portal composites
//! possible — a depth-1 view's destination is the main G-buffer's real
//! textures (`ctx.resources.gbuffer` etc., fetched directly rather than
//! through the executor's chain-fusion `render_pass_descriptor` path), while
//! a depth-2+ view's destination is its *parent* view's own pooled secondary
//! G-buffer slot (`GpuSecondaryView::parent_index`). Both destinations are
//! `LoadOp::Load` — the same reproject/depth-test/write-8-channels shader and
//! pipeline serve either one, since a secondary G-buffer slot and the main
//! G-buffer share byte-identical target formats.
//!
//! (Historical note: this pass originally tried to fuse into the main
//! G-buffer's executor-managed render-pass chain, matching `FoliageGBufferPass`.
//! That fusion was never actually happening — `SecondaryGBufferPass` sits
//! between `FoliageGBufferPass` and this pass and is itself self-managed
//! (`render_pass_descriptor` returns `None`), which breaks any chain before
//! it reaches here — confirmed by the graph debug trace
//! (`RenderGraph`'s "NOT CHAINED" log). Going fully self-managed costs nothing
//! that wasn't already lost, and is what makes nested recursion possible.)
//!
//! Views are processed in descending index order: `Scene::refresh_secondary_views`
//! allocates portal recursion depth-first, so a nested view's index is always
//! greater than its parent's, and descending order is therefore always
//! innermost-first — a depth-3 view merges into depth-2's buffer before
//! depth-2 merges into depth-1's before depth-1 merges into the main frame.

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_pass_secondary_gbuffer::{create_layer_views, SECONDARY_GBUFFER_DEPTH, SECONDARY_GBUFFER_TARGETS};
use helio_secondary_core::{GpuSecondaryView, MAX_SECONDARY_VIEWS, NO_PARENT, SECONDARY_RESOLUTION_DIVISOR};

const SHADER: &str = include_str!("../shaders/proxy_composite.wgsl");

/// CPU-side transcode of `GpuSecondaryView` into a `<uniform>`-address-space
/// safe layout.
///
/// `GpuSecondaryView` is tightly packed (`#[repr(C)]`, plain `[f32; N]`
/// arrays, no padding — pinned at 120 bytes) so it round-trips through a
/// WGSL `storage` binding byte-for-byte, but WGSL's `uniform` address space
/// requires `vec4`/`mat4x4` members to start at 16-byte-aligned offsets,
/// which `GpuSecondaryView`'s own field order doesn't guarantee (`clip_plane`
/// sits at byte offset 8). Rather than bind it as `storage` (needlessly wide
/// for what this shader actually reads), this small 80-byte struct carries
/// only `camera_slot` + `space_transform`, plus the viewport size the
/// fragment shader needs to turn `@builtin(position)` into a UV — laid out so
/// `space_transform` (a `mat4x4<f32>`) starts at offset 16.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeParams {
    camera_slot: u32,
    viewport_width: f32,
    viewport_height: f32,
    _pad0: u32,
    space_transform: [f32; 16],
}

struct PerView {
    uniform_buf: wgpu::Buffer,
    active: bool,
    /// Scissor rect in **destination** pixels: `[x, y, width, height]`. The
    /// destination is the main frame when `parent_index == NO_PARENT`, or
    /// the parent view's own pooled slot otherwise — both fixed resolutions,
    /// computed by `Scene::refresh_secondary_views`.
    region_rect: [u32; 4],
    /// `NO_PARENT`, or another active view's index in this same frame's
    /// array — see `GpuSecondaryView::parent_index`.
    parent_index: u32,
}

fn create_per_view(device: &wgpu::Device) -> PerView {
    PerView {
        uniform_buf: device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ProxyComposite Params"),
            size: std::mem::size_of::<CompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }),
        active: false,
        region_rect: [0, 0, 0, 0],
        parent_index: NO_PARENT,
    }
}

pub struct ProxyCompositePass {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    views: Vec<PerView>,
    active_view_count: usize,
}

impl ProxyCompositePass {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = helio_core::shader::module(device, "ProxyComposite", SHADER);
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ProxyComposite BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                texture_entry(5),
                texture_entry(6),
                texture_entry(7),
                texture_entry(8),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ProxyComposite Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let color_targets: Vec<Option<wgpu::ColorTargetState>> = SECONDARY_GBUFFER_TARGETS
            .iter()
            .map(|&(_, format)| {
                Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })
            })
            .collect();
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ProxyComposite Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &color_targets,
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                // Strictly less: a composited surface exactly coincident with
                // existing main-scene depth (e.g. a sublevel's placement
                // lining up flush with a wall) should not flicker-overwrite
                // it every frame.
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ProxyComposite Sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let views = (0..MAX_SECONDARY_VIEWS as usize).map(|_| create_per_view(device)).collect();

        Self { pipeline, bgl, sampler, views, active_view_count: 0 }
    }
}

fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl RenderPass for ProxyCompositePass {
    fn name(&self) -> &'static str {
        "ProxyComposite"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["secondary_gbuffers", "gbuffer"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["gbuffer"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("gbuffer");
        for &(name, _) in SECONDARY_GBUFFER_TARGETS.iter() {
            builder.read(name);
        }
        builder.read(SECONDARY_GBUFFER_DEPTH);
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // Self-managed — see this module's doc comment for why (recursive
        // nested composites need to open more than one render pass per
        // frame, with different attachment sets per view).
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let Some(secondary) = ctx.frame_resources.secondary.get() else {
            println!("ProxyComposite: no secondary frame data");
            self.active_view_count = 0;
            for view in &mut self.views {
                view.active = false;
            }
            return Ok(());
        };

        let gpu_views: &[GpuSecondaryView] = bytemuck::cast_slice(secondary.view_bytes);
        let view_count = (secondary.view_count as usize).min(gpu_views.len()).min(self.views.len());
        self.active_view_count = view_count;
        println!("ProxyComposite: {} active views (published {})", view_count, secondary.view_count);
        for vi in 0..view_count.min(5) {
            let v = &gpu_views[vi];
            println!("  view[{}]: camera_slot={}, region_rect=({:.0},{:.0},{:.0},{:.0}), parent={}", vi, v.camera_slot, v.region_rect[0], v.region_rect[1], v.region_rect[2], v.region_rect[3], v.parent_index);
        }

        // Every pooled secondary-view slot shares one fixed resolution
        // regardless of recursion depth (see `SECONDARY_RESOLUTION_DIVISOR`'s
        // doc comment) — a nested view's `region_rect` (published against
        // that resolution by `Scene::refresh_secondary_views`) scissors into
        // that resolution too, not the main output's.
        let pool_viewport = [
            (ctx.width / SECONDARY_RESOLUTION_DIVISOR).max(1),
            (ctx.height / SECONDARY_RESOLUTION_DIVISOR).max(1),
        ];

        for i in 0..self.views.len() {
            if i >= view_count {
                self.views[i].active = false;
                continue;
            }
            let gv = gpu_views[i];
            let dest = if gv.parent_index == NO_PARENT { [ctx.width, ctx.height] } else { pool_viewport };
            let params = CompositeParams {
                camera_slot: gv.camera_slot,
                viewport_width: dest[0] as f32,
                viewport_height: dest[1] as f32,
                _pad0: 0,
                space_transform: gv.space_transform,
            };
            ctx.queue.write_buffer(&self.views[i].uniform_buf, 0, bytemuck::bytes_of(&params));

            let rect = gv.region_rect;
            let x = rect[0].max(0.0) as u32;
            let y = rect[1].max(0.0) as u32;
            let w = (rect[2].max(0.0) as u32).min(dest[0].saturating_sub(x));
            let h = (rect[3].max(0.0) as u32).min(dest[1].saturating_sub(y));
            self.views[i].region_rect = [x, y, w, h];
            self.views[i].parent_index = gv.parent_index;
            self.views[i].active = w > 0 && h > 0;
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.active_view_count == 0 {
            return Ok(());
        }

        // Sample-usage views over the pool `SecondaryGBufferPass` rendered
        // into this frame — used both as this pass's texture-sample *source*
        // (every view samples its own slot) and, for nested views, as the
        // write *destination* (a nested view's parent's own slot). Rebuilt
        // every frame rather than cached, matching `SecondaryGBufferPass`'s
        // identical "correctness over micro-optimisation" choice.
        let mut pool_color: [Vec<wgpu::TextureView>; 8] = Default::default();
        for (i, &(name, format)) in SECONDARY_GBUFFER_TARGETS.iter().enumerate() {
            let Some(tex) = ctx.resource_pool.get_texture(name) else { return Ok(()) };
            pool_color[i] = create_layer_views(tex, format, name);
        }
        let Some(depth_tex) = ctx.resource_pool.get_texture(SECONDARY_GBUFFER_DEPTH) else { return Ok(()) };
        let pool_depth = create_layer_views(depth_tex, wgpu::TextureFormat::Depth32Float, SECONDARY_GBUFFER_DEPTH);

        // Main G-buffer's real textures — fetched directly (not through
        // `render_pass_descriptor`'s parameters, since this pass is
        // self-managed) for the `parent_index == NO_PARENT` destination.
        let Some(main_gbuffer) = ctx.resources.gbuffer.get() else { return Ok(()) };
        let Some(main_lightmap_uv) = ctx.resources.gbuffer_lightmap_uv.get() else { return Ok(()) };
        let Some(main_sss) = ctx.resources.gbuffer_sss.get() else { return Ok(()) };
        let Some(main_extra) = ctx.resources.gbuffer_extra.get() else { return Ok(()) };
        let Some(main_velocity) = ctx.resources.gbuffer_velocity.get() else { return Ok(()) };
        let Some(main_depth_texture) = ctx.resources.depth_texture.get() else { return Ok(()) };
        let main_depth_view = main_depth_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let main_color: [&wgpu::TextureView; 8] = [
            main_gbuffer.albedo,
            main_gbuffer.normal,
            main_gbuffer.orm,
            main_gbuffer.emissive,
            main_lightmap_uv,
            main_sss,
            main_extra,
            main_velocity,
        ];

        let cameras_buf = ctx.scene.camera;

        // Descending index order = innermost-first (see this module's doc
        // comment for why DFS allocation guarantees this).
        let mut order: Vec<usize> = (0..self.views.len()).filter(|&i| self.views[i].active).collect();
        order.sort_unstable_by(|a, b| b.cmp(a));

        for i in order {
            let [x, y, w, h] = self.views[i].region_rect;
            if w == 0 || h == 0 {
                continue;
            }

            let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ProxyComposite BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: cameras_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.views[i].uniform_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&pool_color[0][i]) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&pool_color[1][i]) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&pool_color[2][i]) },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&pool_color[3][i]) },
                    wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&pool_color[4][i]) },
                    wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&pool_color[5][i]) },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&pool_color[6][i]) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&pool_depth[i]) },
                    wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            });

            let parent = self.views[i].parent_index;
            let (color_attachments, depth_view): (Vec<Option<wgpu::RenderPassColorAttachment>>, &wgpu::TextureView) =
                if parent == NO_PARENT {
                    (
                        main_color
                            .iter()
                            .map(|&view| {
                                Some(wgpu::RenderPassColorAttachment {
                                    view,
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                                })
                            })
                            .collect(),
                        &main_depth_view,
                    )
                } else {
                    let p = parent as usize;
                    (
                        pool_color
                            .iter()
                            .map(|views| {
                                Some(wgpu::RenderPassColorAttachment {
                                    view: &views[p],
                                    resolve_target: None,
                                    depth_slice: None,
                                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                                })
                            })
                            .collect(),
                        &pool_depth[p],
                    )
                };

            let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(if parent == NO_PARENT { "ProxyComposite (main)" } else { "ProxyComposite (nested)" }),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            pass.set_scissor_rect(x, y, w, h);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
