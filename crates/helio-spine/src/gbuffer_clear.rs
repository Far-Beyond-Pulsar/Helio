//! The G-buffer phase's opener of last resort: clears the frame's G-buffer
//! (`helio_frame_api::gbuffer`) and depth when no kind installed opens the
//! phase.
//!
//! The attachments are the frame's, not a kind's: whatever reads the
//! G-buffer (the lighting, reflections, a kind's drawers that load it) must
//! find it written every frame, whether an object kind that clears it while
//! it draws (the meshes plugin's G-buffer pass) is installed or not. Bevy's
//! opaque pass node does the same: it begins its render pass (and so clears)
//! with no phase items queued. A preset places this pass only when the
//! G-buffer phase has no opener, so with one installed nothing changes.

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, RenderPass, ResourceKey, Result as HelioResult};
use helio_frame_api::gbuffer;

/// Clears every G-buffer attachment and depth; records no draws.
pub struct GBufferClearPass;

impl RenderPass for GBufferClearPass {
    fn name(&self) -> &'static str {
        "GBufferClear"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // Declared as the frame's layout with the usages every G-buffer
        // writer's attachments carry, so a drawer that loads them after this
        // pass sees the same textures it would after a kind's opener.
        builder.write_group(gbuffer::GROUP, gbuffer::GROUP_MEMBERS, ResourceSize::MatchSurface);
        builder.with_group_extra_usage(gbuffer::GROUP, wgpu::TextureUsages::STORAGE_BINDING);
        builder.write_color_raw(
            gbuffer::LIGHTMAP_UV,
            wgpu::TextureFormat::Rg16Float,
            ResourceSize::MatchSurface,
        );
        for name in [gbuffer::SSS, gbuffer::EXTRA, gbuffer::VELOCITY] {
            builder.write_color_raw(name, wgpu::TextureFormat::Rgba16Float, ResourceSize::MatchSurface);
            builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
        }
    }

    fn publish_group<'a>(
        &self,
        group_name: &'static str,
        views: &[&'a wgpu::TextureView],
        frame: &mut helio_core::ResourceRegistry<'a>,
    ) {
        if group_name == gbuffer::GROUP {
            if let [albedo, normal, orm, emissive] = *views {
                frame.write(
                    ResourceKey::new(gbuffer::GROUP),
                    helio_core::ViewGroup::<4> {
                        views: [albedo, normal, orm, emissive],
                    },
                    "GBufferClear",
                );
            }
        }
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let group = resources.read::<helio_core::ViewGroup<'_, 4>>(
            ResourceKey::new(gbuffer::GROUP),
            "GBufferClear",
        )?;
        let mut views: Vec<&'a wgpu::TextureView> = group.views.to_vec();
        for name in [gbuffer::LIGHTMAP_UV, gbuffer::SSS, gbuffer::EXTRA, gbuffer::VELOCITY] {
            views.push(resources.read(ResourceKey::new(name), "GBufferClear")?);
        }
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = storage
            .retain_boxed_slice(
                views
                    .into_iter()
                    .map(|view| {
                        Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                store: wgpu::StoreOp::Store,
                            },
                        })
                    })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            );
        Some(wgpu::RenderPassDescriptor {
            label: Some("GBufferClear"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(helio_core::depth::CLEAR),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}
