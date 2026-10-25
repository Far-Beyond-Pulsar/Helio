//! Side-by-side spectator view. The headset never waits for window presentation.

pub struct StereoMirror {
    pipeline: wgpu::RenderPipeline,
    bindings: Vec<wgpu::BindGroup>,
    overlay_texture: wgpu::Texture,
    overlay_pipeline: wgpu::RenderPipeline,
    overlay_bind_group: wgpu::BindGroup,
    overlay_params: wgpu::Buffer,
    overlay_pixels: Vec<u8>,
}

/// Runtime and renderer stats shown on the PC-side XR mirror.
#[derive(Clone, Copy, Debug, Default)]
pub struct XrMirrorStats {
    /// App frames submitted per second, estimated over recent renderer frames.
    pub fps: Option<f32>,
    /// Headset display refresh rate reported or inferred by the OpenXR runtime.
    pub display_hz: Option<f32>,
    /// Most recently completed GPU frame time.
    pub gpu_ms: Option<f32>,
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
        let mut mirror = StereoMirror::new(
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
            XrMirrorStats::default(),
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
        assert_eq!(pixel(16, 20), [255, 0, 0, 255]);
        assert_eq!(pixel(48, 20), [0, 0, 255, 255]);
        assert_ne!(pixel(16, 32), [255, 0, 0, 255]);
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

const OVERLAY_SHADER: &str = r#"
struct OverlayParams { origin: vec2<f32> }
struct Vertex { @builtin(position) position: vec4<f32> }
@group(0) @binding(0) var overlay: texture_2d<f32>;
@group(0) @binding(1) var<uniform> params: OverlayParams;
@vertex fn vs(@builtin(vertex_index) i: u32) -> Vertex {
    let p = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    return Vertex(vec4<f32>(p * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0));
}
@fragment fn fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    return textureLoad(overlay, vec2<i32>(position.xy - params.origin), 0);
}
"#;

impl StereoMirror {
    const OVERLAY_WIDTH: u32 = 512;
    const OVERLAY_HEIGHT: u32 = 32;

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
        let overlay_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("XR mirror stats overlay"),
            size: wgpu::Extent3d {
                width: Self::OVERLAY_WIDTH,
                height: Self::OVERLAY_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let overlay_view = overlay_texture.create_view(&Default::default());
        let overlay_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("XR mirror stats overlay position"),
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("XR mirror stats overlay layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(8),
                    },
                    count: None,
                },
            ],
        });
        let overlay_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("XR mirror stats overlay bindings"),
            layout: &overlay_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&overlay_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: overlay_params.as_entire_binding(),
                },
            ],
        });
        let overlay_shader =
            helio_core::shader::module(device, "XR mirror stats overlay shader", OVERLAY_SHADER);
        let overlay_pipeline = helio_core::shader::create_render_pipeline(
            device,
            wgpu::RenderPipelineDescriptor {
                label: Some("XR mirror stats overlay pipeline"),
                layout: Some(
                    &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("XR mirror stats overlay pipeline layout"),
                        bind_group_layouts: &[Some(&overlay_layout)],
                        immediate_size: 0,
                    }),
                ),
                vertex: wgpu::VertexState {
                    module: &overlay_shader,
                    entry_point: Some("vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &overlay_shader,
                    entry_point: Some("fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        Self {
            pipeline,
            bindings,
            overlay_texture,
            overlay_pipeline,
            overlay_bind_group,
            overlay_params,
            overlay_pixels: vec![0; (Self::OVERLAY_WIDTH * Self::OVERLAY_HEIGHT * 4) as usize],
        }
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: u32,
        target: &wgpu::TextureView,
        window: [u32; 2],
        eye: [u32; 2],
        stats: XrMirrorStats,
    ) {
        self.update_overlay(queue, stats);
        let mut encoder = device.create_command_encoder(&Default::default());
        // Preserve the two eyes' aspect ratio with letterboxing on resize.
        let scale = (window[0] as f32 / (eye[0] * 2) as f32).min(window[1] as f32 / eye[1] as f32);
        let (w, h) = ((eye[0] * 2) as f32 * scale, eye[1] as f32 * scale);
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
        let viewport_x = ((window[0] as f32 - w) * 0.5).round() as u32;
        let viewport_y = ((window[1] as f32 - h) * 0.5).round() as u32;
        let overlay_x = viewport_x
            .saturating_add(12)
            .min(window[0].saturating_sub(1));
        let overlay_y = viewport_y
            .saturating_add(12)
            .min(window[1].saturating_sub(1));
        let overlay_w = Self::OVERLAY_WIDTH.min(window[0] - overlay_x);
        let overlay_h = Self::OVERLAY_HEIGHT.min(window[1] - overlay_y);
        let origin = [overlay_x as f32, overlay_y as f32];
        queue.write_buffer(&self.overlay_params, 0, bytemuck::cast_slice(&origin));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("XR mirror stats overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
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
            pass.set_viewport(
                overlay_x as f32,
                overlay_y as f32,
                overlay_w as f32,
                overlay_h as f32,
                0.0,
                1.0,
            );
            pass.set_pipeline(&self.overlay_pipeline);
            pass.set_bind_group(0, &self.overlay_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }

    fn update_overlay(&mut self, queue: &wgpu::Queue, stats: XrMirrorStats) {
        self.overlay_pixels.fill(0);
        let fps = stats
            .fps
            .map_or_else(|| "--".to_owned(), |value| format!("{value:.0}"));
        let hz = stats
            .display_hz
            .map_or_else(|| "--".to_owned(), |value| format!("{value:.0}"));
        let gpu = stats
            .gpu_ms
            .map_or_else(|| "--".to_owned(), |value| format!("{value:.1}"));
        let text = format!("XR {hz}HZ | FPS {fps} | GPU {gpu}MS");
        let text_width = (text.len() as u32 * 12 + 16).min(Self::OVERLAY_WIDTH);
        for y in 4..28 {
            for x in 4..text_width {
                set_pixel(&mut self.overlay_pixels, x, y, [8, 13, 22, 220]);
            }
        }
        let smooth = stats
            .fps
            .zip(stats.display_hz)
            .map(|(fps, hz)| fps >= hz * 0.95);
        let color = match smooth {
            Some(true) => [132, 255, 177, 255],
            Some(false) => [255, 204, 105, 255],
            None => [235, 243, 255, 255],
        };
        for (index, character) in text.bytes().enumerate() {
            for (row, bits) in glyph_rows(character).into_iter().enumerate() {
                let left = 8 + index as u32 * 12;
                for column in 0..5 {
                    if bits & (1 << (4 - column)) != 0 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                set_pixel(
                                    &mut self.overlay_pixels,
                                    left + column * 2 + dx,
                                    8 + row as u32 * 2 + dy,
                                    color,
                                );
                            }
                        }
                    }
                }
            }
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.overlay_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.overlay_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(Self::OVERLAY_WIDTH * 4),
                rows_per_image: Some(Self::OVERLAY_HEIGHT),
            },
            wgpu::Extent3d {
                width: Self::OVERLAY_WIDTH,
                height: Self::OVERLAY_HEIGHT,
                depth_or_array_layers: 1,
            },
        );
    }
}

fn set_pixel(pixels: &mut [u8], x: u32, y: u32, color: [u8; 4]) {
    if x >= StereoMirror::OVERLAY_WIDTH || y >= StereoMirror::OVERLAY_HEIGHT {
        return;
    }
    let offset = ((y * StereoMirror::OVERLAY_WIDTH + x) * 4) as usize;
    pixels[offset..offset + 4].copy_from_slice(&color);
}

fn glyph_rows(character: u8) -> [u8; 7] {
    match character {
        b'A' => [14, 17, 17, 31, 17, 17, 17],
        b'F' => [31, 16, 16, 30, 16, 16, 16],
        b'G' => [14, 17, 16, 23, 17, 17, 15],
        b'H' => [17, 17, 17, 31, 17, 17, 17],
        b'I' => [14, 4, 4, 4, 4, 4, 14],
        b'M' => [17, 27, 21, 21, 17, 17, 17],
        b'P' => [30, 17, 17, 30, 16, 16, 16],
        b'R' => [30, 17, 17, 30, 20, 18, 17],
        b'S' => [15, 16, 16, 14, 1, 1, 30],
        b'U' => [17, 17, 17, 17, 17, 17, 14],
        b'X' => [17, 17, 10, 4, 10, 17, 17],
        b'Z' => [31, 1, 2, 4, 8, 16, 31],
        b'0' => [14, 17, 19, 21, 25, 17, 14],
        b'1' => [4, 12, 4, 4, 4, 4, 14],
        b'2' => [14, 17, 1, 2, 4, 8, 31],
        b'3' => [30, 1, 1, 14, 1, 1, 30],
        b'4' => [2, 6, 10, 18, 31, 2, 2],
        b'5' => [31, 16, 16, 30, 1, 1, 30],
        b'6' => [14, 16, 16, 30, 17, 17, 14],
        b'7' => [31, 1, 2, 4, 8, 8, 8],
        b'8' => [14, 17, 17, 14, 17, 17, 14],
        b'9' => [14, 17, 17, 15, 1, 1, 14],
        b'.' => [0, 0, 0, 0, 0, 6, 6],
        b'|' => [4, 4, 4, 4, 4, 4, 4],
        _ => [0; 7],
    }
}
