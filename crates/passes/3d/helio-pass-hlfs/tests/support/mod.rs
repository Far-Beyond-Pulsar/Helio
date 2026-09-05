use glam::{Mat4, Vec3};
use helio_core::{GpuScene, RenderGraph};
use helio_pass_hlfs::{HlfsConfig, HlfsPass};
use std::sync::Arc;

pub struct Fixture {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub scene: GpuScene,
    pub graph: RenderGraph,
    pub width: u32,
    pub height: u32,
    views: Vec<wgpu::TextureView>,
    depth: wgpu::TextureView,
    target: wgpu::TextureView,
    timestamps: Option<wgpu::QuerySet>,
    shadow: Option<wgpu::TextureView>,
}

impl Fixture {
    pub async fn new(width: u32, height: u32) -> Self {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .expect("GPU adapter required");
        eprintln!("HLFS adapter: {:?}", adapter.get_info());
        let timing_features =
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        let timing = adapter.features().contains(timing_features);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                required_features: if timing {
                    timing_features
                } else {
                    wgpu::Features::empty()
                },
                ..Default::default()
            })
            .await
            .unwrap();
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let mut scene = GpuScene::new(device.clone(), queue.clone());
        scene.width = width;
        scene.height = height;
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 3.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::orthographic_rh(-2.0, 2.0, -2.0, 2.0, 0.1, 10.0);
        scene.camera.update(libhelio::GpuCameraUniforms::new(
            view,
            proj,
            Vec3::new(0.0, 0.0, 3.0),
            0.1,
            10.0,
            0,
            [0.0; 2],
            proj * view,
        ));
        let make = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("HLFS fixture"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let mut views = Vec::new();
        for color in [
            [0.7f32, 0.4, 0.2, 1.0],
            [0.0, 0.0, 1.0, 1.0],
            [1.0, 0.6, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ] {
            let texture = make(
                wgpu::TextureFormat::Rgba16Float,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            );
            let values: Vec<u16> = color
                .iter()
                .map(|v| half::f16::from_f32(*v).to_bits())
                .cycle()
                .take((width * height * 4) as usize)
                .collect();
            queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(&values),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 8),
                    rows_per_image: Some(height),
                },
                texture.size(),
            );
            views.push(texture.create_view(&Default::default()));
        }
        let depth_texture = make(
            wgpu::TextureFormat::Depth32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        );
        let depth = depth_texture.create_view(&Default::default());
        let target = make(
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        )
        .create_view(&Default::default());
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Fixture plane depth"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear((3.0 - 0.1) / (10.0 - 0.1)),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }
        queue.submit([encoder.finish()]);
        let mut graph = RenderGraph::new(&device, &queue);
        let timestamps = timing.then(|| {
            device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("HLFS frame timing"),
                ty: wgpu::QueryType::Timestamp,
                count: 2,
            })
        });
        if let Some(q) = &timestamps {
            graph.add_pass(Box::new(Timestamp {
                query: q.clone(),
                index: 0,
            }));
        }
        graph.add_pass(Box::new(HlfsPass::new(
            &device,
            &queue,
            width,
            height,
            wgpu::TextureFormat::Rgba16Float,
        )));
        if let Some(q) = &timestamps {
            graph.add_pass(Box::new(Timestamp {
                query: q.clone(),
                index: 1,
            }));
        }
        graph.lock(width, height);
        Self {
            device,
            queue,
            scene,
            graph,
            width,
            height,
            views,
            depth,
            target,
            timestamps,
            shadow: None,
        }
    }
    pub fn config(&mut self, config: HlfsConfig) {
        self.graph
            .find_pass_mut::<HlfsPass>()
            .unwrap()
            .set_config(&self.device, config);
    }
    pub fn lights(&mut self, lights: Vec<libhelio::GpuLight>) {
        self.scene.movable_light_count = lights.len() as u32;
        self.scene.movable_lights_generation += 1;
        self.scene.lights.set_data(lights);
    }
    pub fn frame(&mut self) {
        self.scene.flush();
        let mut resources = libhelio::FrameResources::empty();
        resources.gbuffer.write(
            libhelio::GBufferViews {
                albedo: &self.views[0],
                normal: &self.views[1],
                orm: &self.views[2],
                emissive: &self.views[3],
            },
            "Fixture",
        );
        resources.pre_aa.write(&self.views[4], "Fixture");
        if let Some(shadow) = &self.shadow {
            resources.shadow_atlas.write(shadow, "Fixture");
        }
        self.graph
            .execute_with_frame_resources(&self.scene, &self.target, &self.depth, &resources)
            .unwrap();
        self.scene.frame_count += 1;
    }
    pub fn read(&self) -> Vec<[f32; 3]> {
        let texture = self.graph.find_pass::<HlfsPass>().unwrap().output_texture();
        let stride = (self.width * 8).div_ceil(256) * 256;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HLFS capture"),
            size: u64::from(stride * self.height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: Some(self.height),
                },
            },
            texture.size(),
        );
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let data = buffer.slice(..).get_mapped_range().unwrap();
        let mut pixels = Vec::new();
        for row in data.chunks(stride as usize) {
            for pixel in row[..(self.width * 8) as usize].chunks(8) {
                pixels.push(std::array::from_fn(|i| {
                    half::f16::from_bits(u16::from_le_bytes([pixel[2 * i], pixel[2 * i + 1]]))
                        .to_f32()
                }));
            }
        }
        pixels
    }
    pub fn milliseconds(&self) -> f64 {
        let query = self
            .timestamps
            .as_ref()
            .expect("GPU timestamps required for benchmark");
        let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 16,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.resolve_query_set(query, 0..2, &resolve, 0);
        encoder.copy_buffer_to_buffer(&resolve, 0, &read, 0, 16);
        self.queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        read.slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = read.slice(..).get_mapped_range().unwrap();
        let start = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let end = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        (end - start) as f64 * self.queue.get_timestamp_period() as f64 / 1e6
    }
    pub fn constant_shadow(&mut self, depth: f32) {
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fixture shadow atlas"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 6,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        for layer in 0..6 {
            let view = texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            });
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(depth),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }
        self.queue.submit([encoder.finish()]);
        self.shadow = Some(texture.create_view(&Default::default()));
        let matrix = Mat4::from_cols_array(&[
            0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0,
        ]);
        self.scene.shadow_matrices.set_data(vec![
            libhelio::GpuShadowMatrix {
                light_view_proj: matrix.to_cols_array()
            };
            6
        ]);
    }
    pub fn depth_values(&mut self, values: &[f32]) {
        assert_eq!(values.len(), (self.width * self.height) as usize);
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Fixture depth source"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            texture.as_image_copy(),
            bytemuck::cast_slice(values),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(self.width * 4),
                rows_per_image: Some(self.height),
            },
            texture.size(),
        );
        let shader=self.device.create_shader_module(wgpu::ShaderModuleDescriptor{label:Some("Fixture depth write"),source:wgpu::ShaderSource::Wgsl(r#"
            @group(0) @binding(0) var depths:texture_2d<f32>;
            @vertex fn vs(@builtin(vertex_index) i:u32)->@builtin(position) vec4<f32> {
                let p=array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0),vec2<f32>(3.0,-1.0),vec2<f32>(-1.0,3.0));
                return vec4<f32>(p[i],0.0,1.0);
            }
            @fragment fn fs(@builtin(position) p:vec4<f32>)->@builtin(frag_depth) f32 { return textureLoad(depths,vec2<i32>(p.xy),0).r; }
        "#.into())});
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[Some(&bgl)],
                immediate_size: 0,
            });
        let pipeline = self
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: Some(&layout),
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
                    targets: &[],
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
        let view = texture.create_view(&Default::default());
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit([encoder.finish()]);
    }
}

struct Timestamp {
    query: wgpu::QuerySet,
    index: u32,
}
impl helio_core::RenderPass for Timestamp {
    fn name(&self) -> &'static str {
        if self.index == 0 {
            "HLFS timestamp start"
        } else {
            "HLFS timestamp end"
        }
    }
    fn render_pass_descriptor<'a>(
        &'a self,
        _: &'a wgpu::TextureView,
        _: &'a wgpu::TextureView,
        _: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }
    fn execute(&mut self, ctx: &mut helio_core::PassContext) -> helio_core::Result<()> {
        // The lighting dispatches run on the render encoder after the GBuffer.
        unsafe {
            (&mut *ctx.encoder_ptr).write_timestamp(&self.query, self.index);
        }
        Ok(())
    }
}

pub fn point(position: [f32; 3], color: [f32; 3], intensity: f32) -> libhelio::GpuLight {
    libhelio::GpuLight {
        position_range: [position[0], position[1], position[2], 20.0],
        color_intensity: [color[0], color[1], color[2], intensity],
        ..Default::default()
    }
}
pub fn mean(pixels: &[[f32; 3]]) -> f32 {
    pixels.iter().flatten().sum::<f32>() / (pixels.len() * 3) as f32
}
