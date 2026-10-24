//! Side-by-side spectator view. The headset never waits for window presentation.

pub struct Mirror {
    pipeline: wgpu::RenderPipeline,
    bindings: Vec<wgpu::BindGroup>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exercise the actual shader and external-view binding shape without a headset.
    #[test]
    fn stereo_mirror_keeps_eye_order_and_letterboxes() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("GPU adapter for mirror test");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("GPU device");
        let eyes = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test eyes"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 2,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        for (layer, colour) in [[255, 0, 0, 255], [0, 0, 255, 255]].into_iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &eyes,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: 0,
                        y: 0,
                        z: layer as u32,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                &colour.repeat(16 * 16),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(64),
                    rows_per_image: Some(16),
                },
                wgpu::Extent3d {
                    width: 16,
                    height: 16,
                    depth_or_array_layers: 1,
                },
            );
        }
        let mirror = Mirror::new(
            &device,
            wgpu::TextureFormat::Rgba8Unorm,
            &[eyes.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })],
        );
        let output = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("test mirror"),
            size: wgpu::Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        mirror.draw(
            &device,
            &queue,
            0,
            &output.create_view(&Default::default()),
            [64, 64],
            [16, 16],
        );
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 64 * 256,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(256),
                    rows_per_image: Some(64),
                },
            },
            output.size(),
        );
        queue.submit([encoder.finish()]);
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, |result| result.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let pixels = buffer.slice(..).get_mapped_range().unwrap();
        let pixel = |x: usize, y: usize| &pixels[y * 256 + x * 4..y * 256 + x * 4 + 4];
        assert_eq!(pixel(16, 32), [255, 0, 0, 255]);
        assert_eq!(pixel(48, 32), [0, 0, 255, 255]);
        assert_eq!(pixel(32, 2), [0, 0, 0, 255]);
    }
}

const SHADER: &str = r#"
struct Vertex { @builtin(position) position: vec4<f32>, @location(0) uv: vec2<f32> }
@vertex fn vs(@builtin(vertex_index) i: u32) -> Vertex {
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return Vertex(vec4<f32>(uv * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0), uv);
}
@group(0) @binding(0) var eyes: texture_2d_array<f32>;
@group(0) @binding(1) var eye_sampler: sampler;
@fragment fn fs(in: Vertex) -> @location(0) vec4<f32> {
    let eye = min(u32(in.uv.x * 2.0), textureNumLayers(eyes) - 1u);
    return textureSample(eyes, eye_sampler, vec2<f32>(fract(in.uv.x * 2.0), in.uv.y), eye);
}
"#;

impl Mirror {
    pub fn new(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        views: &[wgpu::TextureView],
    ) -> Self {
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Voxel Craft VR mirror"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bindings = views
            .iter()
            .map(|view| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Voxel Craft VR mirror image"),
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                    ],
                })
            })
            .collect();
        let shader = helio_core::shader::module(device, "Voxel Craft VR mirror", SHADER);
        let pipeline = helio_core::shader::create_render_pipeline(
            device,
            wgpu::RenderPipelineDescriptor {
                label: Some("Voxel Craft VR mirror"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[Some(&layout)],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                primitive: Default::default(),
                depth_stencil: None,
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            },
        );
        Self { pipeline, bindings }
    }

    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: u32,
        target: &wgpu::TextureView,
        window: [u32; 2],
        eye: [u32; 2],
    ) {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Voxel Craft VR mirror"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // Preserve the two eyes' aspect ratio with letterboxing on resize.
            let scale =
                (window[0] as f32 / (eye[0] * 2) as f32).min(window[1] as f32 / eye[1] as f32);
            let (w, h) = ((eye[0] * 2) as f32 * scale, eye[1] as f32 * scale);
            pass.set_viewport(
                (window[0] as f32 - w) * 0.5,
                (window[1] as f32 - h) * 0.5,
                w,
                h,
                0.0,
                1.0,
            );
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings[image as usize], &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}
