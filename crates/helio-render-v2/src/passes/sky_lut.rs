//! Sky-View LUT pass (Hillaire 2020 technique, used by UE5 / Frostbite)
//!
//! Renders the Nishita atmosphere into a small panoramic texture (192×108)
//! once per frame.  The main SkyPass then samples this LUT instead of running
//! the expensive ray-march for every screen pixel.
//!
//! Cost reduction: atmosphere goes from O(W×H) pixels → O(192×108) = ~20k
//! pixels, a ~46× reduction at 1280×720 with no visible quality change.
//!
//! The LUT is encoded in a latitude-longitude panoramic layout:
//!   u = azimuth  / (2π) + 0.5          ∈ [0, 1]
//!   v = sin(elevation) * 0.5 + 0.5     ∈ [0, 1]  (sin-mapping preserves
//!                                                   horizon detail better than
//!                                                   a linear elevation remap)

use crate::graph::{RenderPass, PassContext, PassResourceBuilder, ResourceHandle};
use crate::Result;
use std::sync::Arc;

/// Width of the sky-view LUT in texels.
pub const SKY_LUT_W: u32 = 192;
/// Height of the sky-view LUT in texels.
pub const SKY_LUT_H: u32 = 108;
/// Texture format: Rgba16Float gives enough HDR range for the sky radiance.
pub const SKY_LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct SkyLutPass {
    pipeline:       Arc<wgpu::RenderPipeline>,
    sky_bind_group: Arc<wgpu::BindGroup>,
    lut_view:       Arc<wgpu::TextureView>,
}

impl SkyLutPass {
    pub fn new(
        device:          &wgpu::Device,
        sky_uniform_bgl: &wgpu::BindGroupLayout,
        global_bgl:      &wgpu::BindGroupLayout,
        sky_bind_group:  Arc<wgpu::BindGroup>,
        lut_view:        Arc<wgpu::TextureView>,
    ) -> Self {
        // The LUT pass only needs group(0)=Camera and group(1)=SkyUniforms-only.
        // It produces the LUT texture so it cannot bind it.
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky LUT Pipeline Layout"),
            bind_group_layouts: &[
                Some(global_bgl),
                Some(sky_uniform_bgl),
            ],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Sky LUT Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/sky_lut.wgsl").into(),
            ),
        });

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Sky LUT Pipeline"),
            layout: Some(&layout),
            cache:  None,
            vertex: wgpu::VertexState {
                module:               &shader,
                entry_point:          Some("vs_main"),
                buffers:              &[],
                compilation_options:  Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                targets:             &[Some(wgpu::ColorTargetState {
                    format:     SKY_LUT_FORMAT,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:    wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None,
            multisample:  wgpu::MultisampleState::default(),
            multiview_mask: None,
        }));

        Self { pipeline, sky_bind_group, lut_view }
    }
}

impl RenderPass for SkyLutPass {
    fn name(&self) -> &str { "sky_lut" }

    fn declare_resources(&self, builder: &mut PassResourceBuilder) {
        // Writes the sky_lut resource; SkyPass declares a read on it to force ordering.
        builder.write(ResourceHandle::named("sky_lut"));
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        if !ctx.has_sky { return Ok(()); }
        // Skip LUT re-render when sky parameters haven't changed since last frame.
        // The LUT texture retains its previous contents and remains valid.
        if !ctx.sky_state_changed { return Ok(()); }

        let mut pass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Sky LUT Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view:           &self.lut_view,
                resolve_target: None,
                depth_slice:    None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
            multiview_mask:           None,
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, ctx.global_bind_group, &[]);
        pass.set_bind_group(1, Some(self.sky_bind_group.as_ref()), &[]);
        pass.draw(0..3, 0..1);

        Ok(())
    }
}
