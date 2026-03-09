/// GBuffer pass — deferred geometry.
///
/// 4 MRT targets: albedo (Rgba8Unorm), normal (Rgba16Float), ORM (Rgba8Unorm), emissive (Rgba16Float).
/// Depth is loaded from the prepass result (`LoadOp::Load`), depth writes DISABLED,
/// compare = LessEqual (pixels at exactly the prepass depth pass).
///
/// Bundle-cached: same HISM-based hash scheme as depth prepass.
use std::sync::Arc;
use crate::{
    graph::pass::{RenderPass, PassContext},
    mesh::{PackedVertex, DrawCall},
    hism::draw_list_hash,
};

pub struct GBufferPass {
    opaque_pipeline:  Arc<wgpu::RenderPipeline>,
    masked_pipeline:  Arc<wgpu::RenderPipeline>,
    cached_bundle:    Option<wgpu::RenderBundle>,
    cached_hash:      u64,
}

impl GBufferPass {
    pub fn new(
        device:     &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        mat_bgl:    &wgpu::BindGroupLayout,
    ) -> Self {
        let opaque = build_pipeline(device, camera_bgl, mat_bgl, false);
        let masked = build_pipeline(device, camera_bgl, mat_bgl, true);
        GBufferPass {
            opaque_pipeline: Arc::new(opaque),
            masked_pipeline: Arc::new(masked),
            cached_bundle: None,
            cached_hash:   0,
        }
    }

    fn rebuild_bundle(&mut self, device: &wgpu::Device, camera_bg: &wgpu::BindGroup, draws: &[DrawCall]) {
        let color_formats: &[Option<wgpu::TextureFormat>] = &[
            Some(wgpu::TextureFormat::Rgba8Unorm),
            Some(wgpu::TextureFormat::Rgba16Float),
            Some(wgpu::TextureFormat::Rgba8Unorm),
            Some(wgpu::TextureFormat::Rgba16Float),
        ];
        let mut enc = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label:            Some("gbuffer_bundle"),
            color_formats,
            depth_stencil:    Some(wgpu::RenderBundleDepthStencil {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_read_only:     true,
                stencil_read_only:   true,
            }),
            sample_count: 1,
            multiview:    None,
        });

        enc.set_bind_group(0, camera_bg, &[]);

        let mut last_masked = None::<bool>;
        for draw in draws {
            if draw.transparent_blend { continue; }

            let is_masked = draw.material_id == u32::MAX; // convention: MAX = masked
            if last_masked != Some(is_masked) {
                enc.set_pipeline(if is_masked { &self.masked_pipeline } else { &self.opaque_pipeline });
                last_masked = Some(is_masked);
            }

            enc.set_bind_group(1, &*draw.material_bind_group, &[]);
            enc.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            enc.set_vertex_buffer(1, draw.instance_buffer.slice(draw.instance_buffer_offset..));
            enc.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            enc.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
        }

        self.cached_bundle = Some(enc.finish(&wgpu::RenderBundleDescriptor {
            label: Some("gbuffer_bundle"),
        }));
    }
}

impl RenderPass for GBufferPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        let hash = draw_list_hash(ctx.opaque_draws);
        if hash != self.cached_hash || self.cached_bundle.is_none() {
            self.rebuild_bundle(ctx.device, ctx.camera_bg, ctx.opaque_draws);
            self.cached_hash = hash;
        }

        let bundle = match &self.cached_bundle { Some(b) => b, None => return };

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("gbuffer"),
            color_attachments: &[
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.frame_tex.gbuf_albedo_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.frame_tex.gbuf_normal_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.frame_tex.gbuf_orm_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &ctx.frame_tex.gbuf_emissive_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                    depth_slice: None,
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.frame_tex.depth_view,
                // Load from prepass — NEVER clear depth in gbuffer pass
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
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

fn build_pipeline(
    device:     &wgpu::Device,
    camera_bgl: &wgpu::BindGroupLayout,
    mat_bgl:    &wgpu::BindGroupLayout,
    masked:     bool,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("gbuffer_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/gbuffer.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("gbuffer_layout"),
        bind_group_layouts:   &[Some(camera_bgl), Some(mat_bgl)],
        immediate_size:       0,
    });

    let color_targets: Vec<Option<wgpu::ColorTargetState>> = [
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba16Float,
        wgpu::TextureFormat::Rgba8Unorm,
        wgpu::TextureFormat::Rgba16Float,
    ].iter().map(|&fmt| Some(wgpu::ColorTargetState {
        format:     fmt,
        blend:      None,
        write_mask: wgpu::ColorWrites::ALL,
    })).collect();

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some(if masked { "gbuffer_masked" } else { "gbuffer_opaque" }),
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
            depth_write_enabled: Some(false),
            depth_compare:       Some(wgpu::CompareFunction::LessEqual),
            stencil:             wgpu::StencilState::default(),
            bias:                wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module:      &shader,
            entry_point: Some(if masked { "fs_masked" } else { "fs_opaque" }),
            targets:     &color_targets,
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}
