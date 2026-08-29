use helio_core::Result as HelioResult;

pub struct LutBuilder {
    pipeline: wgpu::ComputePipeline,
    pipeline_layout: wgpu::PipelineLayout,
    bind_group_layout: wgpu::BindGroupLayout,
    lut_texture: wgpu::Texture,
    lut_view_3d: wgpu::TextureView,
    size: u32,
    last_generation: u32,
}

impl LutBuilder {
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LUT Build Shader"),
            source: wgpu::ShaderSource::Wgsl(
                helio_core::shader::resolve(include_str!("../shaders/lut_build.wgsl"))
                    .into_owned()
                    .into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LUT Build BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LUT Build PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LUT Build Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (tex, view) = Self::create_texture(device, size);

        Self {
            pipeline,
            pipeline_layout,
            bind_group_layout,
            lut_texture: tex,
            lut_view_3d: view,
            size,
            last_generation: u32::MAX,
        }
    }

    fn create_texture(device: &wgpu::Device, size: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Color Grading LUT"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: size,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        (tex, view)
    }

    pub fn lut_view(&self) -> &wgpu::TextureView {
        &self.lut_view_3d
    }

    pub fn build_if_needed(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        postprocess_buf: &wgpu::Buffer,
        generation: u32,
    ) -> bool {
        if generation == self.last_generation {
            return false;
        }

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LUT Build BG"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: postprocess_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.lut_view_3d),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("LUT Build Encoder"),
        });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LUT Build"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let wgs = (self.size + 3) / 4;
            cpass.dispatch_workgroups(wgs, wgs, wgs);
        }
        queue.submit(std::iter::once(encoder.finish()));

        self.last_generation = generation;
        true
    }

    pub fn on_resize(&mut self, device: &wgpu::Device, size: u32) {
        if size == self.size {
            return;
        }
        self.size = size;
        let (tex, view) = Self::create_texture(device, size);
        self.lut_texture = tex;
        self.lut_view_3d = view;
        self.last_generation = u32::MAX;
    }
}
