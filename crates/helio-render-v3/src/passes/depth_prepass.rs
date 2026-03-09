/// Depth pre-pass.
///
/// Bundle-cached: the RenderBundle is rebuilt only when the opaque draw list
/// structural hash changes. Hash = FNV-1a over (hism_handle, instance_count) pairs.
/// Skips rebuilds on camera-only changes — same objects, same bundles.
use std::sync::Arc;
use crate::{
    graph::pass::{RenderPass, PassContext},
    mesh::{PackedVertex, DrawCall},
    hism::draw_list_hash,
};

pub struct DepthPrepass {
    pipeline:          Arc<wgpu::RenderPipeline>,
    cached_bundle:     Option<wgpu::RenderBundle>,
    cached_hash:       u64,
}

impl DepthPrepass {
    pub fn new(device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout) -> Self {
        let pipeline = build_pipeline(device, camera_bgl);
        DepthPrepass { pipeline: Arc::new(pipeline), cached_bundle: None, cached_hash: 0 }
    }

    fn rebuild_bundle(&mut self, device: &wgpu::Device, camera_bg: &wgpu::BindGroup, draws: &[DrawCall]) {
        let mut encoder = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label:            Some("depth_prepass_bundle"),
            color_formats:    &[],
            depth_stencil:    Some(wgpu::RenderBundleDepthStencil {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_read_only:     false,
                stencil_read_only:   true,
            }),
            sample_count:     1,
            multiview:        None,
        });

        encoder.set_pipeline(&self.pipeline);
        encoder.set_bind_group(0, camera_bg, &[]);

        for draw in draws {
            if draw.transparent_blend { continue; }
            encoder.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            encoder.set_vertex_buffer(1, draw.instance_buffer.slice(draw.instance_buffer_offset..));
            encoder.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            encoder.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
        }

        self.cached_bundle = Some(encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("depth_prepass_bundle"),
        }));
    }
}

impl RenderPass for DepthPrepass {
    fn execute(&mut self, ctx: &mut PassContext) {
        let hash = draw_list_hash(ctx.opaque_draws);
        if hash != self.cached_hash || self.cached_bundle.is_none() {
            self.rebuild_bundle(ctx.device, ctx.camera_bg, ctx.opaque_draws);
            self.cached_hash = hash;
        }

        let bundle = match &self.cached_bundle { Some(b) => b, None => return };

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("depth_prepass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.frame_tex.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        rpass.set_bind_group(0, ctx.camera_bg, &[]);
        rpass.execute_bundles(std::iter::once(bundle));
    }
}

fn build_pipeline(device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("depth_prepass_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/depth_prepass.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("depth_prepass_layout"),
        bind_group_layouts:   &[Some(camera_bgl)],
        immediate_size:       0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("depth_prepass"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module:      &shader,
            entry_point: Some("vs_main"),
            buffers:     &[PackedVertex::vertex_buffer_layout(), PackedVertex::instance_buffer_layout()],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology:          wgpu::PrimitiveTopology::TriangleList,
            front_face:        wgpu::FrontFace::Ccw,
            cull_mode:         Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format:              wgpu::TextureFormat::Depth32Float,
            depth_write_enabled: Some(true),
            depth_compare:       Some(wgpu::CompareFunction::Less),
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment:    None,
        multiview_mask:   None,
        cache:       None,
    })
}
