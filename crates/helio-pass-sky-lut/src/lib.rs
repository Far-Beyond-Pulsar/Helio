//! Sky-view LUT pass (Hillaire 2020).
//!
//! Bakes a 192×108 panoramic sky LUT once per changed sky state.
//! O(1) CPU: single fullscreen draw.

use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};
use bytemuck::{Pod, Zeroable};

const LUT_WIDTH: u32 = 192;
const LUT_HEIGHT: u32 = 108;

/// Sky uniforms matching the WGSL shader layout (112 bytes, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaderSkyUniforms {
    sun_direction:      [f32; 3],
    sun_intensity:      f32,
    rayleigh_scatter:   [f32; 3],
    rayleigh_h_scale:   f32,
    mie_scatter:        f32,
    mie_h_scale:        f32,
    mie_g:              f32,
    sun_disk_cos:       f32,
    earth_radius:       f32,
    atm_radius:         f32,
    exposure:           f32,
    clouds_enabled:     u32,
    cloud_coverage:     f32,
    cloud_density:      f32,
    cloud_base:         f32,
    cloud_top:          f32,
    cloud_wind_x:       f32,
    cloud_wind_z:       f32,
    cloud_speed:        f32,
    time_sky:           f32,
    skylight_intensity: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
}

impl ShaderSkyUniforms {
    fn earth_like() -> Self {
        let d = [0.0f32, 0.9, 0.4];
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        Self {
            sun_direction:    [d[0] / len, d[1] / len, d[2] / len],
            sun_intensity:    22.0,
            rayleigh_scatter: [5.8e-3, 1.35e-2, 3.31e-2],
            rayleigh_h_scale: 0.1,
            mie_scatter:      2.1e-3,
            mie_h_scale:      0.075,
            mie_g:            0.76,
            sun_disk_cos:     0.9998,
            earth_radius:     6360.0,
            atm_radius:       6420.0,
            exposure:         0.1,
            clouds_enabled:   0,
            cloud_coverage:   0.0,
            cloud_density:    0.0,
            cloud_base:       0.0,
            cloud_top:        0.0,
            cloud_wind_x:     0.0,
            cloud_wind_z:     0.0,
            cloud_speed:      0.0,
            time_sky:         0.0,
            skylight_intensity: 0.0,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
        }
    }
}

pub struct SkyLutPass {
    pipeline:          wgpu::RenderPipeline,
    #[allow(dead_code)]
    bgl_0:             wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bgl_1:             wgpu::BindGroupLayout,
    bind_group_0:      wgpu::BindGroup,
    bind_group_1:      wgpu::BindGroup,
    sky_uniform_buf:   wgpu::Buffer,
    pub sky_lut_texture: wgpu::Texture,
    pub sky_lut_view:    wgpu::TextureView,
}

impl SkyLutPass {
    pub fn new(device: &wgpu::Device, camera_buf: &wgpu::Buffer) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("SkyLUT Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sky_lut.wgsl").into()),
        });

        let sky_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("SkyLUT Uniforms"),
            size:               std::mem::size_of::<ShaderSkyUniforms>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sky_lut_texture = device.create_texture(&wgpu::TextureDescriptor {
            label:               Some("Sky LUT"),
            size:                wgpu::Extent3d { width: LUT_WIDTH, height: LUT_HEIGHT, depth_or_array_layers: 1 },
            mip_level_count:     1,
            sample_count:        1,
            dimension:           wgpu::TextureDimension::D2,
            format:              wgpu::TextureFormat::Rgba16Float,
            usage:               wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:        &[],
        });
        let sky_lut_view = sky_lut_texture.create_view(&Default::default());

        // Group 0: camera uniform
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("SkyLUT BGL0"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        // Group 1: sky uniforms
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("SkyLUT BGL1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding:    0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty:                 wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size:   None,
                },
                count: None,
            }],
        });

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("SkyLUT BG0"),
            layout:  &bgl_0,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: camera_buf.as_entire_binding(),
            }],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("SkyLUT BG1"),
            layout:  &bgl_1,
            entries: &[wgpu::BindGroupEntry {
                binding:  0,
                resource: sky_uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("SkyLUT PL"),
            bind_group_layouts:   &[Some(&bgl_0), Some(&bgl_1)],
            immediate_size:       0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("SkyLUT Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:               &shader,
                entry_point:          Some("vs_main"),
                compilation_options:  Default::default(),
                buffers:              &[],
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample:   wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache:         None,
        });

        Self {
            pipeline,
            bgl_0,
            bgl_1,
            bind_group_0,
            bind_group_1,
            sky_uniform_buf,
            sky_lut_texture,
            sky_lut_view,
        }
    }
}

impl RenderPass for SkyLutPass {
    fn name(&self) -> &'static str {
        "SkyLUT"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.sky_lut = Some(&self.sky_lut_view);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Upload default Nishita atmosphere parameters.
        // A real engine would derive these from a SkySystem component.
        let uniforms = ShaderSkyUniforms::earth_like();
        ctx.write_buffer(&self.sky_uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single fullscreen draw — GPU integrates atmosphere per LUT texel.
        let color_attachment = wgpu::RenderPassColorAttachment {
            view:           &self.sky_lut_view,
            resolve_target: None,
            depth_slice:    None,
            ops: wgpu::Operations {
                load:  wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        };
        let color_attachments = [Some(color_attachment)];
        let desc = wgpu::RenderPassDescriptor {
            label:                    Some("SkyLUT"),
            color_attachments:        &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
            multiview_mask:           None,
        };

        let mut pass = ctx.encoder.begin_render_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group_0, &[]);
        pass.set_bind_group(1, &self.bind_group_1, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}
