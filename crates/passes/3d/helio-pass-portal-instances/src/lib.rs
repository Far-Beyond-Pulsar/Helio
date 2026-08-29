//! Draws the portal-duplicated instances `helio-pass-portal-cull` selected,
//! into the G-buffer, clipped to every portal along each survivor's chain.
//!
//! Fused into the same physical render pass `helio-pass-gbuffer` opened
//! (`LoadOp::Load` on all 8 attachments), following `helio-pass-foliage-gbuffer`'s
//! precedent exactly: real depth buffer, real materials, no separate camera,
//! no compositing. One `multi_draw_indexed_indirect` call, same shape as the
//! plain non-portal G-buffer pass — one draw per mesh/material draw group,
//! not one per chain; each instance's chain is looked up per-instance from a
//! buffer `helio-pass-portal-cull` wrote, not chosen by a per-draw uniform.
//!
//! # Integration
//!
//! ```ignore
//! let cull_pass = PortalCullPass::new(device);
//! let indirect_buf = Arc::clone(&cull_pass.portal_indirect_buf);
//! let compacted_buf = Arc::clone(&cull_pass.portal_compacted_indices_buf);
//! let compacted_chains_buf = Arc::clone(&cull_pass.portal_compacted_chains_buf);
//! graph.add_pass(Box::new(cull_pass));           // before GBufferPass
//! // ... GBufferPass added here ...
//! graph.add_pass(Box::new(PortalMaskPass::new(device))); // after GBufferPass, before PortalInstancePass
//! graph.add_pass(Box::new(
//!     PortalInstancePass::new(device, indirect_buf, compacted_buf, compacted_chains_buf),
//! )); // immediately after PortalMaskPass, before FoliageGBufferPass/VirtualGeometryPass
//! ```

use std::{borrow::Cow, sync::Arc};

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_pass_portal_cull::PORTAL_DRAW_CAPACITY;

mod mask;
pub use mask::PortalMaskPass;

mod editor_overlay;
pub use editor_overlay::PortalEditorOverlayPass;

use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ScreenSize {
    width: f32,
    height: f32,
    _pad0: f32,
    _pad1: f32,
}

type PortalBindGroupKey = (usize, usize, usize, usize, usize, usize, usize, usize);

pub struct PortalInstancePass {
    material_binding: libhelio::MaterialBindingConfig,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout_0: wgpu::BindGroupLayout,
    bind_group_layout_1: wgpu::BindGroupLayout,

    screen_buf: wgpu::Buffer,

    /// Shared with `PortalCullPass` — this pass only reads them.
    portal_indirect_buf: Arc<wgpu::Buffer>,
    portal_compacted_indices_buf: Arc<wgpu::Buffer>,
    portal_compacted_chains_buf: Arc<wgpu::Buffer>,

    bind_group_0: Option<wgpu::BindGroup>,
    bind_group_0_key: Option<PortalBindGroupKey>,
    bind_group_1: Option<wgpu::BindGroup>,
    bind_group_1_version: Option<u64>,

    draw_count: u32,
}

impl PortalInstancePass {
    pub fn new(
        device: &wgpu::Device,
        portal_indirect_buf: Arc<wgpu::Buffer>,
        portal_compacted_indices_buf: Arc<wgpu::Buffer>,
        portal_compacted_chains_buf: Arc<wgpu::Buffer>,
    ) -> Self {
        let material_binding = libhelio::MaterialBindingConfig::for_device(device);

        let shader_source = portal_shader_source(material_binding);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PortalInstance Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source),
        });

        let screen_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalInstance/ScreenSize"),
            size: std::mem::size_of::<ScreenSize>() as u64,
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
                storage_entry(6, wgpu::ShaderStages::VERTEX_FRAGMENT, true), // portal_views
                storage_entry(7, wgpu::ShaderStages::VERTEX_FRAGMENT, true), // portal_chains
                storage_entry(8, wgpu::ShaderStages::VERTEX_FRAGMENT, true), // portal_compacted_chains
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }, // portal_mask — written by helio-pass-portal-mask
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
                // `helio-pass-portal-mask` (run immediately before this pass)
                // already reset depth to the far plane everywhere a portal's
                // opening is actually visible on screen, and left real depth
                // untouched everywhere else. So a normal LessEqual+write test
                // here does the right thing on both sides of that boundary:
                // duplicate content behind a visible opening self-occludes
                // correctly (nearer copies win), while any pixel the mask
                // pass *didn't* reset (portal not visible / occluded from
                // here) keeps real depth and would fail this test too — belt
                // and suspenders alongside the mask discard in the shader.
                // Writing depth lets downstream passes in the same fused
                // pass (foliage, virtual geometry) occlude correctly against
                // the illusion content instead of the stale real depth.
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    // Small nudge so adjacent duplicate copies' coplanar
                    // seams (e.g. two segments' floors meeting end to end)
                    // don't z-fight.
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
            portal_indirect_buf,
            portal_compacted_indices_buf,
            portal_compacted_chains_buf,
            bind_group_0: None,
            bind_group_0_key: None,
            bind_group_1: None,
            bind_group_1_version: None,
            draw_count: 0,
        }
    }
}

fn portal_shader_source(material_binding: libhelio::MaterialBindingConfig) -> Cow<'static, str> {
    let source = include_str!("../shaders/gbuffer_portal.wgsl");
    if material_binding.uses_binding_arrays() {
        Cow::Borrowed(source)
    } else {
        Cow::Owned(libhelio::shader::apply_webgpu_material_bindings(
            source,
            material_binding.max_textures,
        ))
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
        // Written by helio-pass-portal-mask, which must run before this pass.
        builder.read("portal_mask");
    }

    fn reads(&self) -> &'static [&'static str] {
        &["gbuffer", "portal_mask"]
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
        if ctx.frame_num < 3 || ctx.frame_num.is_multiple_of(120) {
            log::info!(
                "[PortalInstance] frame={} draw_count={} render_pass_open={}",
                ctx.frame_num, self.draw_count, ctx.active_render_pass_ptr().is_some(),
            );
        }
        if self.draw_count == 0 {
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

        // helio-pass-portal-mask must run before this pass every frame (see
        // helio-default-graphs' add_pass order) — its output gates every
        // fragment below on actual on-screen portal visibility.
        let Some(portal_mask_view) = ctx.resource_pool.get_view("portal_mask") else {
            log::warn!("[PortalInstance] frame={} portal_mask not available — PortalMaskPass not wired before this pass?", ctx.frame_num);
            return Ok(());
        };

        // ── Bind group 0 ──────────────────────────────────────────────────
        let key = (
            ctx.scene.camera as *const _ as usize,
            ctx.scene.instances as *const _ as usize,
            ctx.scene.coordinate_spaces as *const _ as usize,
            ctx.scene.portal_views as *const _ as usize,
            ctx.scene.portal_chains as *const _ as usize,
            &*self.portal_compacted_indices_buf as *const _ as usize,
            &*self.portal_compacted_chains_buf as *const _ as usize,
            portal_mask_view as *const _ as usize,
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
                    wgpu::BindGroupEntry { binding: 7, resource: ctx.scene.portal_chains.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: self.portal_compacted_chains_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(portal_mask_view) },
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
        pass.set_bind_group(0, self.bind_group_0.as_ref().unwrap(), &[]);
        pass.set_bind_group(1, self.bind_group_1.as_ref().unwrap(), &[]);

        // One indirect command per draw group (mesh+material), same shape
        // as the plain non-portal G-buffer pass — every chain's surviving
        // instances for a given group are already merged into that group's
        // single indirect command by PortalCullPass's `finalize` step.
        let indirect_draw_count = self.draw_count.min(PORTAL_DRAW_CAPACITY);
        pass.multi_draw_indexed_indirect(&self.portal_indirect_buf, 0, indirect_draw_count);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::portal_shader_source;
    use libhelio::{MaterialBindingConfig, MaterialBindingMode};

    #[test]
    fn expanded_tier_rewrites_the_portal_material_bindings() {
        let source = portal_shader_source(MaterialBindingConfig {
            mode: MaterialBindingMode::Expanded,
            max_textures: 2,
        });

        assert!(!source.contains("wgpu_binding_array"));
        assert!(!source.contains("binding_array<"));
        assert!(source.contains("@group(1) @binding(2) var scene_texture_0"));
        assert!(source.contains("@group(1) @binding(5) var scene_sampler_1"));
        assert!(source.contains("case 1u: { return textureSampleLevel"));
    }

    #[test]
    fn binding_array_tier_keeps_the_native_portal_shader() {
        let source = portal_shader_source(MaterialBindingConfig {
            mode: MaterialBindingMode::BindingArray,
            max_textures: 256,
        });

        assert!(matches!(source, std::borrow::Cow::Borrowed(_)));
        assert!(source.contains("enable wgpu_binding_array;"));
        assert!(source.contains("binding_array<texture_2d<f32>, 256>"));
    }
}
