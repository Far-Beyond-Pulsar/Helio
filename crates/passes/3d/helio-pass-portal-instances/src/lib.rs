//! Draws the portal-duplicated instances `helio-pass-portal-cull` selected,
//! into the G-buffer, clipped to each portal's opening.
//!
//! Fused into the same physical render pass `helio-pass-gbuffer` opened
//! (`LoadOp::Load` on all 8 attachments), following `helio-pass-foliage-gbuffer`'s
//! precedent exactly: real depth buffer, real materials, no separate camera,
//! no compositing. One `multi_draw_indexed_indirect` call per active portal.
//!
//! # Integration
//!
//! ```ignore
//! let cull_pass = PortalCullPass::new(device);
//! let indirect_buf = Arc::clone(&cull_pass.portal_indirect_buf);
//! let compacted_buf = Arc::clone(&cull_pass.portal_compacted_indices_buf);
//! graph.add_pass(Box::new(cull_pass));           // before GBufferPass
//! // ... GBufferPass added here ...
//! graph.add_pass(Box::new(
//!     PortalInstancePass::new(device, indirect_buf, compacted_buf),
//! )); // immediately after GBufferPass, before FoliageGBufferPass/VirtualGeometryPass
//! ```

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_pass_portal_cull::{MAX_PORTAL_SLOTS, PORTAL_DRAW_CAPACITY};

/// Byte stride between consecutive per-portal dynamic-uniform entries.
/// Matches `helio-pass-shadow`'s own `FACE_BUF_STRIDE` — 256 is guaranteed to
/// satisfy `min_uniform_buffer_offset_alignment` on every wgpu backend.
const PORTAL_UNIFORM_STRIDE: u64 = 256;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScreenSize {
    width: f32,
    height: f32,
    _pad0: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PortalDrawUniform {
    portal_view_index: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct PortalInstancePass {
    material_binding: libhelio::MaterialBindingConfig,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,

    screen_buf: wgpu::Buffer,
    /// Written once at construction: slot N always selects `portal_views[N]`.
    portal_draw_buf: wgpu::Buffer,

    /// Shared with `PortalCullPass` — this pass only reads them.
    portal_indirect_buf: Arc<wgpu::Buffer>,
    portal_compacted_indices_buf: Arc<wgpu::Buffer>,

    bind_group_0: Option<wgpu::BindGroup>,
    bind_group_0_key: Option<(usize, usize, usize, usize, usize, usize)>,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_1_version: Option<u64>,

    portal_count: u32,
    draw_count: u32,
    portal_draw_written: bool,
}

impl PortalInstancePass {
    pub fn new(
        device: &wgpu::Device,
        portal_indirect_buf: Arc<wgpu::Buffer>,
        portal_compacted_indices_buf: Arc<wgpu::Buffer>,
    ) -> Self {
        let material_binding = libhelio::MaterialBindingConfig::for_device(device);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PortalInstance Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gbuffer_portal.wgsl").into()),
        });

        let screen_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalInstance/ScreenSize"),
            size: std::mem::size_of::<ScreenSize>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Write-once: slot N's `portal_view_index` is always N — populated
        // lazily on the first `prepare()` call (this constructor only
        // receives `&wgpu::Device`, no queue).
        let portal_draw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalInstance/PortalDrawUniform"),
            size: (MAX_PORTAL_SLOTS as u64) * PORTAL_UNIFORM_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group_layout_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PortalInstance BGL 0"),
            entries: &[
                storage_entry(0, wgpu::ShaderStages::VERTEX, true),   // cameras
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT, false), // screen size
                storage_entry(2, wgpu::ShaderStages::VERTEX, true),   // instance_data
                storage_entry(3, wgpu::ShaderStages::VERTEX, true),   // coordinate_spaces
                storage_entry(4, wgpu::ShaderStages::VERTEX, true),   // coordinate_spaces_prev
                storage_entry(5, wgpu::ShaderStages::VERTEX, true),   // portal_compacted_indices
                storage_entry(
                    6,
                    wgpu::ShaderStages::VERTEX_FRAGMENT,
                    true,
                ), // portal_views
                uniform_entry(7, wgpu::ShaderStages::VERTEX_FRAGMENT, true), // portal_draw (dynamic)
            ],
        });
        let bind_group_layout_1 = helio_pass_gbuffer::create_material_bgl(device, material_binding);

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PortalInstance PL"),
            bind_group_layouts: &[Some(&bind_group_layout_0), Some(&bind_group_layout_1)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PortalInstance Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                // Shared mesh vertex buffer layout — identical to GBufferPass
                // (stride = 40 bytes; see there for the per-field breakdown).
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: 40,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x3, offset: 0, shader_location: 0 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32, offset: 12, shader_location: 1 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 16, shader_location: 2 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Float32x2, offset: 24, shader_location: 5 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 32, shader_location: 3 },
                        wgpu::VertexAttribute { format: wgpu::VertexFormat::Uint32, offset: 36, shader_location: 4 },
                    ],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba8Unorm, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rg16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rgba16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                    Some(wgpu::ColorTargetState { format: wgpu::TextureFormat::Rg16Float, blend: None, write_mask: wgpu::ColorWrites::ALL }),
                ],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                // Test against, but do not write, depth: this is a *duplicate*
                // of geometry that already occupies real depth somewhere else
                // in the scene. Writing depth here would let the duplicate
                // incorrectly occlude real geometry drawn afterward in the
                // same pass (foliage, virtual geometry) at the portal's
                // screen position.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: -1,
                    slope_scale: 0.0,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        Self {
            material_binding,
            pipeline,
            bind_group_layout_0,
            bind_group_layout_1,
            screen_buf,
            portal_draw_buf,
            portal_indirect_buf,
            portal_compacted_indices_buf,
            bind_group_0: None,
            bind_group_0_key: None,
            bind_group_1: None,
            bind_group_1_version: None,
            portal_count: 0,
            draw_count: 0,
            portal_draw_written: false,
        }
    }
}

fn storage_entry(binding: u32, visibility: wgpu::ShaderStages, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32, visibility: wgpu::ShaderStages, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: dynamic,
            min_binding_size: None,
        },
        count: None,
    }
}

impl RenderPass for PortalInstancePass {
    fn name(&self) -> &'static str {
        "PortalInstance"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // Read-only, exactly like FoliageGBufferPass — the eight G-buffer
        // targets are declared and owned by GBufferPass; this pass loads and
        // draws into them, which is what makes fusion into the same physical
        // pass possible.
        builder.read("gbuffer");
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        // Always `Some` when the G-buffer exists (chain fusion is decided by
        // attachment identity at lock time, not per-frame content — see
        // helio-pass-foliage-gbuffer's docs for why returning `None` here
        // conditionally would break fusion).
        let gbuffer = resources.gbuffer.read("PortalInstance")?;
        let lightmap_uv = resources.gbuffer_lightmap_uv.read("PortalInstance")?;
        let sss_target = resources.gbuffer_sss.read("PortalInstance")?;
        let extra_target = resources.gbuffer_extra.read("PortalInstance")?;
        let velocity_target = resources.gbuffer_velocity.read("PortalInstance")?;

        const LOAD: wgpu::Operations<wgpu::Color> = wgpu::Operations {
            load: wgpu::LoadOp::Load,
            store: wgpu::StoreOp::Store,
        };
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([
                Some(wgpu::RenderPassColorAttachment { view: gbuffer.albedo, resolve_target: None, depth_slice: None, ops: LOAD }),
                Some(wgpu::RenderPassColorAttachment { view: gbuffer.normal, resolve_target: None, depth_slice: None, ops: LOAD }),
                Some(wgpu::RenderPassColorAttachment { view: gbuffer.orm, resolve_target: None, depth_slice: None, ops: LOAD }),
                Some(wgpu::RenderPassColorAttachment { view: gbuffer.emissive, resolve_target: None, depth_slice: None, ops: LOAD }),
                Some(wgpu::RenderPassColorAttachment { view: lightmap_uv, resolve_target: None, depth_slice: None, ops: LOAD }),
                Some(wgpu::RenderPassColorAttachment { view: sss_target, resolve_target: None, depth_slice: None, ops: LOAD }),
                Some(wgpu::RenderPassColorAttachment { view: extra_target, resolve_target: None, depth_slice: None, ops: LOAD }),
                Some(wgpu::RenderPassColorAttachment { view: velocity_target, resolve_target: None, depth_slice: None, ops: LOAD }),
            ]));

        Some(wgpu::RenderPassDescriptor {
            label: Some("PortalInstance"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if !self.portal_draw_written {
            for slot in 0..MAX_PORTAL_SLOTS {
                let data = PortalDrawUniform {
                    portal_view_index: slot,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                ctx.write_buffer(
                    &self.portal_draw_buf,
                    slot as u64 * PORTAL_UNIFORM_STRIDE,
                    bytemuck::bytes_of(&data),
                );
            }
            self.portal_draw_written = true;
        }

        self.portal_count = ctx.scene.portal_views.len() as u32;
        self.draw_count = ctx.scene.draw_calls.len() as u32;
        let screen = ScreenSize {
            width: ctx.width as f32,
            height: ctx.height as f32,
            _pad0: 0.0,
            _pad1: 0.0,
        };
        ctx.write_buffer(&self.screen_buf, 0, bytemuck::bytes_of(&screen));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if ctx.frame_num < 3 || ctx.frame_num % 120 == 0 {
            log::info!(
                "[PortalInstance] frame={} portal_count={} draw_count={} render_pass_open={}",
                ctx.frame_num, self.portal_count, self.draw_count, ctx.active_render_pass_ptr().is_some(),
            );
        }
        if self.portal_count == 0 {
            return Ok(());
        }
        let Some(pass_ptr) = ctx.active_render_pass_ptr() else {
            log::warn!("[PortalInstance] frame={} no active render pass — G-buffer chain not fused/opened", ctx.frame_num);
            return Ok(());
        };
        let main_scene = ctx.resources.main_scene.read("PortalInstance");
        if ctx.frame_num < 3 {
            log::info!("[PortalInstance] frame={} main_scene_available={}", ctx.frame_num, main_scene.is_some());
        }

        // ── Bind group 0 ──────────────────────────────────────────────────
        let key = (
            ctx.scene.camera as *const _ as usize,
            ctx.scene.instances as *const _ as usize,
            ctx.scene.coordinate_spaces as *const _ as usize,
            ctx.scene.portal_views as *const _ as usize,
            &*self.portal_compacted_indices_buf as *const _ as usize,
            &*self.portal_indirect_buf as *const _ as usize,
        );
        if self.bind_group_0_key != Some(key) {
            self.bind_group_0 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PortalInstance BG 0"),
                layout: &self.bind_group_layout_0,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.camera.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.screen_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: ctx.scene.instances.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: ctx.scene.coordinate_spaces.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: ctx.scene.coordinate_spaces_prev.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: self.portal_compacted_indices_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: ctx.scene.portal_views.as_entire_binding() },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &self.portal_draw_buf,
                            offset: 0,
                            size: std::num::NonZeroU64::new(std::mem::size_of::<PortalDrawUniform>() as u64),
                        }),
                    },
                ],
            }));
            self.bind_group_0_key = Some(key);
        }

        // ── Bind group 1 (materials) — rebuilt when texture set changes ────
        let Some(main_scene) = main_scene else {
            return Ok(());
        };
        let needs_rebuild = self.bind_group_1_version != Some(main_scene.material_textures.version)
            || self.bind_group_1.is_none();
        if needs_rebuild {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: ctx.scene.materials.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: main_scene.material_textures.material_textures.as_entire_binding(),
                },
            ];
            self.material_binding.append_bind_group_entries(
                &mut entries,
                2,
                main_scene.material_textures.texture_views,
                main_scene.material_textures.samplers,
            );
            self.bind_group_1 = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PortalInstance BG 1"),
                layout: &self.bind_group_layout_1,
                entries: &entries,
            }));
            self.bind_group_1_version = Some(main_scene.material_textures.version);
        }

        let vertices = main_scene.mesh_buffers.vertices;
        let indices = main_scene.mesh_buffers.indices;

        let pass = unsafe { &mut *pass_ptr };
        pass.set_pipeline(&self.pipeline);
        pass.set_vertex_buffer(0, vertices.slice(..));
        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);

        // Only the first `draw_count` entries of each portal's reserved slice
        // were written by PortalCullPass this frame (see portal_cull.wgsl) —
        // clamped to the fixed allocation ceiling PORTAL_DRAW_CAPACITY.
        // Issuing exactly this many, not the full capacity, avoids submitting
        // thousands of guaranteed-empty indirect commands per portal.
        let indirect_draw_count = self.draw_count.min(PORTAL_DRAW_CAPACITY);
        let active_portals = self.portal_count.min(MAX_PORTAL_SLOTS);
        for portal_idx in 0..active_portals {
            pass.set_bind_group(
                0,
                self.bind_group_0.as_ref().unwrap(),
                &[(portal_idx as u64 * PORTAL_UNIFORM_STRIDE) as u32],
            );
            let indirect_offset = (portal_idx as u64) * (PORTAL_DRAW_CAPACITY as u64) * 20;
            pass.multi_draw_indexed_indirect(&self.portal_indirect_buf, indirect_offset, indirect_draw_count);
        }
        Ok(())
    }
}
