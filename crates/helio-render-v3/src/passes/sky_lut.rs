/// Sky LUT pre-computation pass.
///
/// Computes the transmittance + multi-scatter LUT into a small texture.
/// Only re-runs when `sky_state_changed` is true — the result is stable for
/// many thousands of frames as long as the sun direction doesn't move.
use std::sync::Arc;
use crate::{
    graph::pass::{RenderPass, PassContext},
    pipeline::{PipelineCache, PipelineKey, PipelineVariant, fnv1a_str},
};

const LUT_W: u32 = 256;
const LUT_H: u32 = 64;

pub struct SkyLutPass {
    pipeline:    Arc<wgpu::ComputePipeline>,
    bind_group:  wgpu::BindGroup,
    pub lut:     Arc<wgpu::Texture>,
    pub lut_view: wgpu::TextureView,
    sky_hash:    u64,
}

impl SkyLutPass {
    pub fn new(device: &wgpu::Device, globals_buffer: &wgpu::Buffer) -> Self {
        let lut = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("sky_lut"),
            size:            wgpu::Extent3d { width: LUT_W, height: LUT_H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba16Float,
            usage:           wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats:    &[],
        }));
        let lut_view = lut.create_view(&wgpu::TextureViewDescriptor::default());

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sky_lut_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sky_lut_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&lut_view) },
            ],
        });

        let shader_src = include_str!("../shaders/sky_lut.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("sky_lut_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });
        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("sky_lut_layout"),
            bind_group_layouts:   &[Some(&bgl)],
            immediate_size:       0,
        });
        let pipeline = Arc::new(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:       Some("sky_lut_pipeline"),
            layout:      Some(&pl_layout),
            module:      &shader,
            entry_point: Some("cs_main"),
            compilation_options: Default::default(),
            cache: None,
        }));

        SkyLutPass { pipeline, bind_group, lut, lut_view, sky_hash: u64::MAX }
    }

    fn sky_hash(ctx: &PassContext) -> u64 {
        match ctx.sky_atmosphere {
            None => 0,
            Some(sky) => {
                let mut h = 0xcbf29ce484222325u64;
                h ^= sky.sun_direction.x.to_bits() as u64; h = h.wrapping_mul(0x100000001b3);
                h ^= sky.sun_direction.y.to_bits() as u64; h = h.wrapping_mul(0x100000001b3);
                h ^= sky.sun_direction.z.to_bits() as u64; h = h.wrapping_mul(0x100000001b3);
                h ^= sky.sun_intensity   .to_bits() as u64; h = h.wrapping_mul(0x100000001b3);
                h ^= sky.rayleigh_scale  .to_bits() as u64; h = h.wrapping_mul(0x100000001b3);
                h ^= sky.mie_scale       .to_bits() as u64; h = h.wrapping_mul(0x100000001b3);
                h
            }
        }
    }
}

impl RenderPass for SkyLutPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        if ctx.sky_atmosphere.is_none() { return; }

        let new_hash = Self::sky_hash(ctx);
        if new_hash == self.sky_hash { return; }   // sky unchanged — skip entirely
        self.sky_hash = new_hash;

        let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("sky_lut"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.dispatch_workgroups(LUT_W / 8, LUT_H / 8, 1);
    }
}
