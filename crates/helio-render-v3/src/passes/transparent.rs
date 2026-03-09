/// Transparent pass — NO RenderBundle.
///
/// Transparent draw order changes every frame (back-to-front by camera depth).
/// Bundles are incompatible with per-frame sort order — recording inline is correct here.
/// The overhead is proportional to transparent draw count which is typically small.
///
/// Optimisation: skip `set_bind_group` when the pointer to the bind group Arc hasn't changed.
use std::sync::Arc;
use crate::{
    graph::pass::{RenderPass, PassContext},
    mesh::{PackedVertex, DrawCall},
};

pub struct TransparentPass {
    pipeline:    Arc<wgpu::RenderPipeline>,
    lights_bgl:  wgpu::BindGroupLayout,
    /// Track last bind group pointer to skip redundant set_bind_group calls.
    last_bg_ptr: u64,
}

impl TransparentPass {
    pub fn new(
        device:         &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        camera_bgl:     &wgpu::BindGroupLayout,
        mat_bgl:        &wgpu::BindGroupLayout,
    ) -> Self {
        let lights_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("transparent_lights_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false, min_binding_size: None,
                }, count: None,
            }],
        });
        let pipeline = build_pipeline(device, surface_format, camera_bgl, mat_bgl, &lights_bgl);
        TransparentPass { pipeline: Arc::new(pipeline), lights_bgl, last_bg_ptr: 0 }
    }
}

impl RenderPass for TransparentPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        let draws = ctx.transparent_draws;
        if draws.is_empty() { return; }

        self.last_bg_ptr = 0;

        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("transparent"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.frame_tex.pre_aa_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &ctx.frame_tex.depth_view,
                depth_ops: Some(wgpu::Operations {
                    load:  wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });

        let lights_bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   None,
            layout:  &self.lights_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ctx.light_buffer.as_entire_binding() }],
        });

        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, ctx.camera_bg, &[]);
        rpass.set_bind_group(2, &lights_bg, &[]);

        for draw in draws {
            // Skip set_bind_group if same bind group as previous draw (pointer identity check).
            let bg_ptr = Arc::as_ptr(&draw.material_bind_group) as u64;
            if bg_ptr != self.last_bg_ptr {
                rpass.set_bind_group(1, &*draw.material_bind_group, &[]);
                self.last_bg_ptr = bg_ptr;
            }

            rpass.set_vertex_buffer(0, draw.vertex_buffer.slice(..));
            rpass.set_vertex_buffer(1, draw.instance_buffer.slice(draw.instance_buffer_offset..));
            rpass.set_index_buffer(draw.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            rpass.draw_indexed(0..draw.index_count, 0, 0..draw.instance_count);
        }
    }
}

fn build_pipeline(
    device:         &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    camera_bgl:     &wgpu::BindGroupLayout,
    mat_bgl:        &wgpu::BindGroupLayout,
    lights_bgl:     &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label:  Some("transparent_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/forward_lit.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label:                Some("transparent_layout"),
        bind_group_layouts:   &[Some(camera_bgl), Some(mat_bgl), Some(lights_bgl)],
        immediate_size:       0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some("transparent"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader, entry_point: Some("vs_main"),
            buffers: &[PackedVertex::vertex_buffer_layout(), PackedVertex::instance_buffer_layout()],
            compilation_options: Default::default(),
        },
        primitive: wgpu::PrimitiveState {
            topology:   wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode:  None,
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
            module: &shader, entry_point: Some("fs_transparent"),
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend:  Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        multiview_mask: None, cache: None,
    })
}
