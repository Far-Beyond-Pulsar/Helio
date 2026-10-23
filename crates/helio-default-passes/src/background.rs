//! The scene's background: `pre_aa` cleared to the configured colour before anything draws
//! into it.
//!
//! The sky is the atmosphere (`helio_pass_sky::AtmosphereCompositePass`),
//! drawn over this background where no geometry is. Without an atmosphere
//! the background stays black. This pass took over the clear the retired
//! procedural `SkyPass` did (Pulsar-Native #1057), so every graph's lighting
//! and geometry passes load a defined `pre_aa`: forward lighting draws over
//! it, HLFS reads it at uncovered pixels, and SSR and planar reflections
//! sample it before deferred lighting.

use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, RenderPass, ResourceKey, Result as HelioResult};

/// Clears `pre_aa` to the caller's colour; records no draws.
pub(crate) struct BackgroundPass {
    format: wgpu::TextureFormat,
    color: wgpu::Color,
}

impl BackgroundPass {
    pub(crate) fn new(format: wgpu::TextureFormat, color: [f32; 4]) -> Self {
        Self {
            format,
            color: wgpu::Color {
                r: f64::from(color[0]),
                g: f64::from(color[1]),
                b: f64::from(color[2]),
                a: f64::from(color[3]),
            },
        }
    }
}

impl RenderPass for BackgroundPass {
    fn name(&self) -> &'static str {
        "Background"
    }

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.write_color_raw("pre_aa", self.format, ResourceSize::MatchSurface);
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let target = resources.texture_view(ResourceKey::new("pre_aa"))?;
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            storage.retain_boxed_slice(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(self.color),
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Background"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, _ctx: &mut PassContext) -> HelioResult<()> {
        Ok(())
    }
}
