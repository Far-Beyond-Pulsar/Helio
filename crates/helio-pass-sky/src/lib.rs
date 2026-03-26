//! Sky rendering pass.
//!
//! Renders the sky dome to the HDR target by sampling the pre-baked sky LUT.
//! O(1) CPU: single fullscreen draw.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Sky uniforms matching the WGSL shader layout (112 bytes, 16-byte aligned).
/// Must match the layout used in sky.wgsl and sky_lut.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaderSkyUniforms {
    sun_direction: [f32; 3],
    sun_intensity: f32,
    rayleigh_scatter: [f32; 3],
    rayleigh_h_scale: f32,
    mie_scatter: f32,
    mie_h_scale: f32,
    mie_g: f32,
    sun_disk_cos: f32,
    earth_radius: f32,
    atm_radius: f32,
    exposure: f32,
    clouds_enabled: u32,
    cloud_coverage: f32,
    cloud_density: f32,
    cloud_base: f32,
    cloud_top: f32,
    cloud_wind_x: f32,
    cloud_wind_z: f32,
    cloud_speed: f32,
    time_sky: f32,
    skylight_intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

impl ShaderSkyUniforms {
    fn earth_like() -> Self {
        let d = [0.0f32, 0.9, 0.4];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        Self {
            sun_direction: [d[0] / len, d[1] / len, d[2] / len],
            sun_intensity: 22.0,
            rayleigh_scatter: [5.8e-3, 1.35e-2, 3.31e-2],
            rayleigh_h_scale: 0.1,
            mie_scatter: 2.1e-3,
            mie_h_scale: 0.075,
            mie_g: 0.76,
            sun_disk_cos: 0.9998,
            earth_radius: 6360.0,
            atm_radius: 6420.0,
            exposure: 0.1,
            clouds_enabled: 0,
            cloud_coverage: 0.0,
            cloud_density: 0.0,
            cloud_base: 0.0,
            cloud_top: 0.0,
            cloud_wind_x: 0.0,
            cloud_wind_z: 0.0,
            cloud_speed: 0.0,
            time_sky: 0.0,
            skylight_intensity: 0.0,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        }
    }
}

pub struct SkyPass {
    pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    bgl_0: wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bgl_1: wgpu::BindGroupLayout,
    bind_group_0: wgpu::BindGroup,
    bind_group_1: wgpu::BindGroup,
    sky_uniform_buf: wgpu::Buffer,
    width: u32,
    height: u32,
    target_format: wgpu::TextureFormat,
}

impl SkyPass {
    /// Creates the sky pass.
    ///
    /// - `camera_buf`: buffer whose first bytes match the sky.wgsl Camera struct
    /// - `sky_lut_view`: the Rgba16Float sky LUT produced by `SkyLutPass`
    /// - `target_format`: format of the HDR render target
    /// - `width/height`: render target size (internal scaled resolution)
    pub fn new(
        device: &wgpu::Device,
        camera_buf: &wgpu::Buffer,
        sky_lut_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        target_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sky Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky.wgsl").into()),
        });

        let sky_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sky Uniforms"),
            size: std::mem::size_of::<ShaderSkyUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Linear-clamp sampler for the sky LUT
        let sky_lut_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Sky LUT Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Group 0: camera uniform
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky BGL0"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // Group 1: sky uniforms + LUT texture + LUT sampler
        // sky.wgsl: @group(1) @binding(0) sky uniforms
        //           @group(1) @binding(1) sky_lut texture_2d<f32>
        //           @group(1) @binding(2) sky_sampler sampler
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sky BGL1"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky BG0"),
            layout: &bgl_0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sky BG1"),
            layout: &bgl_1,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sky_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(sky_lut_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sky_lut_sampler),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sky PL"),
            bind_group_layouts: &[Some(&bgl_0), Some(&bgl_1)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Sky Pipeline"),
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
                    format: target_format,
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
            multiview_mask: None,
            cache: None,
        });

        Self {
            pipeline,
            bgl_0,
            bgl_1,
            bind_group_0,
            bind_group_1,
            sky_uniform_buf,
            width,
            height,
            target_format,
        }
    }

    fn format_to_resource(format: wgpu::TextureFormat) -> helio_v3::graph::ResourceFormat {
        match format {
            wgpu::TextureFormat::Rgba16Float => helio_v3::graph::ResourceFormat::Rgba16Float,
            wgpu::TextureFormat::Bgra8UnormSrgb => helio_v3::graph::ResourceFormat::Bgra8UnormSrgb,
            wgpu::TextureFormat::Rgba8UnormSrgb => helio_v3::graph::ResourceFormat::Rgba8UnormSrgb,
            wgpu::TextureFormat::R16Float => helio_v3::graph::ResourceFormat::R16Float,
            wgpu::TextureFormat::R8Unorm => helio_v3::graph::ResourceFormat::R8Unorm,
            wgpu::TextureFormat::Depth32Float => helio_v3::graph::ResourceFormat::Depth32Float,
            _ => panic!("Unsupported SkyPass target format: {:?}", format),
        }
    }
}

impl RenderPass for SkyPass {
    fn name(&self) -> &'static str {
        "Sky"
    }

    fn declare_resources(&self, builder: &mut helio_v3::graph::ResourceBuilder) {
        builder.read("sky_lut");
        builder.write_color(
            "pre_aa",
            Self::format_to_resource(self.target_format),
            helio_v3::graph::ResourceSize::Absolute {
                width: self.width,
                height: self.height,
            },
        );
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let mut uniforms = ShaderSkyUniforms::earth_like();

        if let Some(clouds) = ctx.frame_resources.sky.clouds {
            uniforms.clouds_enabled = 1;
            uniforms.cloud_coverage = clouds.coverage;
            uniforms.cloud_density = clouds.density;
            uniforms.cloud_base = clouds.base;
            uniforms.cloud_top = clouds.top;
            uniforms.cloud_wind_x = clouds.wind_x;
            uniforms.cloud_wind_z = clouds.wind_z;
            uniforms.cloud_speed = clouds.speed;
            uniforms.skylight_intensity = clouds.skylight_intensity;
        }

        uniforms.time_sky = (ctx.frame as f32) * 0.03;
        ctx.write_buffer(&self.sky_uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single fullscreen draw — GPU samples LUT and composites sky.
        let target_view = if let Some(pre_aa) = ctx.frame.pre_aa {
            pre_aa
        } else {
            ctx.target
        };

        let color_attachment = wgpu::RenderPassColorAttachment {
            view: target_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        };
        let color_attachments = [Some(color_attachment)];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("Sky"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let mut pass = ctx.encoder.begin_render_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group_0, &[]);
        pass.set_bind_group(1, &self.bind_group_1, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

