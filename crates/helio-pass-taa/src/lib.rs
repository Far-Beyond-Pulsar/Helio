//! Temporal Anti-Aliasing (TAA) pass.
//!
//! Blends the current frame with a history buffer using velocity-based reprojection
//! and YCoCg variance-clipped neighbourhood clamping.
//!
//! ## O(1) guarantee
//! `execute()` records exactly one fullscreen `draw(0..3, 0..1)` plus one
//! `copy_texture_to_texture` call (both are constant-time GPU operations).
//!
//! ## Jitter
//! A Halton(2, 3) sequence of 16 entries is indexed by `frame_num % 16`.
//! The jitter offset is uploaded to the GPU uniform every frame in `prepare()`.
//!
//! ## History ping-pong
//! The pass owns two textures: `output_texture` (render target each frame) and
//! `history_texture` (sampled as temporal history).  After each frame, the output
//! is GPU-copied into the history so the next frame sees the updated accumulation.

use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};
use bytemuck::{Pod, Zeroable};

/// Halton(2, 3) sequence, 16 entries, values in (0, 1).
/// Offset by -0.5 at upload time so jitter is centred on the pixel.
const HALTON_JITTER: [[f32; 2]; 16] = [
    [0.500000, 0.333333],
    [0.250000, 0.666667],
    [0.750000, 0.111111],
    [0.125000, 0.444444],
    [0.625000, 0.777778],
    [0.375000, 0.222222],
    [0.875000, 0.555556],
    [0.062500, 0.888889],
    [0.562500, 0.037037],
    [0.312500, 0.370370],
    [0.812500, 0.703704],
    [0.187500, 0.148148],
    [0.687500, 0.481481],
    [0.437500, 0.814815],
    [0.937500, 0.259259],
    [0.031250, 0.592593],
];

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TaaUniform {
    feedback_min: f32,
    feedback_max: f32,
    jitter: [f32; 2],
}

/// TAA pass.
///
/// `current_view` — the pre-TAA HDR colour buffer (e.g. `FrameResources::pre_aa`).
/// `velocity_view` — screen-space velocity (GBuffer `velocity` channel).
/// `depth_view`    — depth-only view of the depth buffer (`TextureAspect::DepthOnly`).
pub struct TaaPass {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    taa_uniform_buf: wgpu::Buffer,
    pub history_texture: wgpu::Texture,
    pub history_view: wgpu::TextureView,
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,
}

impl TaaPass {
    pub fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        current_view: &wgpu::TextureView,
        velocity_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TAA Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/taa.wgsl").into()),
        });

        let taa_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TAA Uniform"),
            size: std::mem::size_of::<TaaUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA Point Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let tex_desc = |label: &'static str, extra_usage: wgpu::TextureUsages| {
            wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | extra_usage,
                view_formats: &[],
            }
        };

        let history_texture = device.create_texture(&tex_desc(
            "TAA History",
            wgpu::TextureUsages::COPY_DST,
        ));
        let history_view = history_texture.create_view(&Default::default());

        let output_texture = device.create_texture(&tex_desc(
            "TAA Output",
            wgpu::TextureUsages::COPY_SRC,
        ));
        let output_view = output_texture.create_view(&Default::default());

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("TAA BGL"),
                entries: &[
                    // 0: current_frame (filterable float — sampled with linear_sampler)
                    tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                    // 1: history_frame (filterable float)
                    tex_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                    // 2: velocity_tex (non-filterable, sampled with point_sampler)
                    tex_entry(2, wgpu::TextureSampleType::Float { filterable: false }),
                    // 3: depth_tex (depth texture)
                    tex_entry(3, wgpu::TextureSampleType::Depth),
                    // 4: linear_sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    // 5: point_sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 5,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                    // 6: taa uniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 6,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&history_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(velocity_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Sampler(&linear_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&point_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: taa_uniform_buf.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("TAA PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TAA Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: 0,
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
            taa_uniform_buf,
            history_texture,
            history_view,
            output_texture,
            output_view,
            linear_sampler,
            point_sampler,
        }
    }
}

/// Helper: 2-D texture BGL entry for group 0, fragment-visible.
fn tex_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

impl RenderPass for TaaPass {
    fn name(&self) -> &'static str {
        "TAA"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let jitter_idx = (ctx.frame % 16) as usize;
        let raw = HALTON_JITTER[jitter_idx];
        // Centre jitter around zero so the sub-pixel offset is symmetric
        let jitter = [raw[0] - 0.5, raw[1] - 0.5];
        let uniforms = TaaUniform {
            feedback_min: 0.88,
            feedback_max: 0.97,
            jitter,
        };
        ctx.queue
            .write_buffer(&self.taa_uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): one fullscreen draw to output_view
        {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &self.output_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })];
            let desc = wgpu::RenderPassDescriptor {
                label: Some("TAA"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: 0,
            };
            let mut pass = ctx.encoder.begin_render_pass(&desc);
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..3, 0..1);
        }

        // Copy output → history so the next frame can sample it.
        // This is a single O(1) GPU blit; no CPU work involved.
        ctx.encoder.copy_texture_to_texture(
            self.output_texture.as_image_copy(),
            self.history_texture.as_image_copy(),
            wgpu::Extent3d {
                width: ctx.width,
                height: ctx.height,
                depth_or_array_layers: 1,
            },
        );

        Ok(())
    }
}
