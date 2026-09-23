//! Add current-frame SSR to an existing linear HDR lighting target.
use helio_core::{graph::ResourceBuilder, PassContext, RenderPass, ResourceKey, Result};

pub struct SsrCompositePass {
    pipeline: wgpu::RenderPipeline,
    camera: wgpu::BindGroup,
    layout: wgpu::BindGroupLayout,
    textures: Option<([wgpu::TextureView; 5], wgpu::BindGroup)>,
}
impl SsrCompositePass {
    pub fn new(device: &wgpu::Device, camera: &wgpu::Buffer, format: wgpu::TextureFormat) -> Self {
        let mut camera_entry = super::buffer_uniform_entry(0);
        camera_entry.visibility = wgpu::ShaderStages::FRAGMENT;
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR composition camera"),
            entries: &[camera_entry],
        });
        let camera = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSR composition camera"),
            layout: &camera_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera.as_entire_binding(),
            }],
        });
        let entries: Vec<_> = (0..5)
            .map(|i| {
                let mut entry = if i == 3 {
                    super::texture_depth_entry(i)
                } else {
                    super::texture_unfiltered_entry(i)
                };
                entry.visibility = wgpu::ShaderStages::FRAGMENT;
                entry
            })
            .collect();
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSR composition textures"),
            entries: &entries,
        });
        let shader = helio_core::shader::module(
            device,
            "SSR composition",
            include_str!("../shaders/ssr_compose.wgsl"),
        );
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SSR composition"),
            bind_group_layouts: &[Some(&camera_layout), Some(&layout)],
            immediate_size: 0,
        });
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SSR additive composition"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState {
                        color: additive,
                        alpha: additive,
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            camera,
            layout,
            textures: None,
        }
    }
}
impl RenderPass for SsrCompositePass {
    fn name(&self) -> &'static str {
        "SsrCompositePass"
    }
    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }
    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("gbuffer");
        builder.read("depth");
        builder.read("ssr_trace");
        builder.read("pre_aa");
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        let Some(gbuffer) = ctx
            .registry
            .read::<helio_core::ViewGroup<'_, 4>>(ResourceKey::new("gbuffer"), self.name())
        else {
            return Ok(());
        };
        let Some(output) = ctx.registry.get(ResourceKey::new("pre_aa")) else {
            return Ok(());
        };
        let Some(reflection) = ctx.resource_pool.get_view("ssr_trace") else {
            return Ok(());
        };
        let views = [
            gbuffer.views[1],
            gbuffer.views[2],
            gbuffer.views[3],
            ctx.depth,
            reflection,
        ];
        let key = views.map(Clone::clone);
        if !self.textures.as_ref().is_some_and(|(old, _)| old == &key) {
            let entries: Vec<_> = views
                .iter()
                .enumerate()
                .map(|(i, v)| super::texture_view_entry(i as u32, v))
                .collect();
            let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SSR composition inputs"),
                layout: &self.layout,
                entries: &entries,
            });
            self.textures = Some((key, group));
        }
        // The traced image is separate from this load/add attachment: no sampled
        // feedback loop with the current lighting target.
        let mut pass =
            unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("SSR additive composition"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.camera, &[]);
        pass.set_bind_group(1, &self.textures.as_ref().unwrap().1, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;
    use wgpu::util::DeviceExt;

    #[test]
    #[ignore = "requires a GPU"]
    fn additive_composition_respects_fresnel_confidence_ao_and_background() {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(
                wgpu::InstanceDescriptor::new_without_display_handle_from_env(),
            );
            let adapter = instance.request_adapter(&Default::default()).await.unwrap();
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    required_limits: adapter.limits(),
                    ..Default::default()
                })
                .await
                .unwrap();
            let mut camera = helio_core::GpuCameraUniforms::zeroed();
            camera.inv_view_proj = [
                1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
            ];
            camera.position_near = [0., 0., 3., 0.1];
            let camera = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[camera, camera]),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let pass = SsrCompositePass::new(&device, &camera, wgpu::TextureFormat::Rgba8Unorm);
            let size = wgpu::Extent3d {
                width: 3,
                height: 1,
                depth_or_array_layers: 1,
            };
            let texture = |format, usage| {
                device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
            };
            let upload = |data: &[[f32; 4]; 3]| {
                let t = texture(
                    wgpu::TextureFormat::Rgba32Float,
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                );
                queue.write_texture(
                    t.as_image_copy(),
                    bytemuck::cast_slice(data),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(48),
                        rows_per_image: Some(1),
                    },
                    size,
                );
                t.create_view(&Default::default())
            };
            let normal = upload(&[[0., 0., 1., 0.5]; 3]);
            let orm = upload(&[[0.8, 1., 0., 0.25], [0.8, 1., 0., 0.25], [0., 1., 0., 0.25]]);
            let emissive = upload(&[[0., 0., 0., 0.125]; 3]);
            let reflected = upload(&[[2., 4., 8., 0.5], [2., 4., 8., 0.], [2., 4., 8., 1.]]);
            let depth = texture(
                wgpu::TextureFormat::Depth32Float,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            )
            .create_view(&Default::default());
            let views = [&normal, &orm, &emissive, &depth, &reflected];
            let entries: Vec<_> = views
                .iter()
                .enumerate()
                .map(|(i, v)| crate::texture_view_entry(i as u32, v))
                .collect();
            let inputs = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pass.layout,
                entries: &entries,
            });
            let output = texture(
                wgpu::TextureFormat::Rgba8Unorm,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            );
            let output_view = output.create_view(&Default::default());
            for background in [false, true] {
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let _depth = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(if background { 1. } else { 0.5 }),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
                {
                    let mut draw = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &output_view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color {
                                    r: 0.1,
                                    g: 0.2,
                                    b: 0.3,
                                    a: 1.,
                                }),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    draw.set_pipeline(&pass.pipeline);
                    draw.set_bind_group(0, &pass.camera, &[]);
                    draw.set_bind_group(1, &inputs, &[]);
                    draw.draw(0..3, 0..1);
                }
                let readback = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 256,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                });
                encoder.copy_texture_to_buffer(
                    output.as_image_copy(),
                    wgpu::TexelCopyBufferInfo {
                        buffer: &readback,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(256),
                            rows_per_image: Some(1),
                        },
                    },
                    size,
                );
                queue.submit([encoder.finish()]);
                let (tx, rx) = std::sync::mpsc::channel();
                readback
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                rx.recv().unwrap().unwrap();
                let data = readback.slice(..).get_mapped_range().unwrap();
                for pixel in 0..3 {
                    let expected = if pixel == 0 && !background {
                        [0.5, 0.6, 0.7]
                    } else {
                        [0.1, 0.2, 0.3]
                    };
                    for channel in 0..3 {
                        assert!(
                            (data[pixel * 4 + channel] as f32 / 255. - expected[channel]).abs()
                                < 0.01,
                            "pixel {pixel} channel {channel} background {background}"
                        );
                    }
                    assert_eq!(
                        data[pixel * 4 + 3],
                        255,
                        "composition must preserve destination alpha"
                    );
                }
            }
        });
    }
}
