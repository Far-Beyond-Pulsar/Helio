//! Optional measurements of the production history decisions. No filter tuning.
use super::{camera_storage_entry, sampler_entry, tex_entry, uniform_entry};
use std::borrow::Cow;

#[cfg(test)]
mod tests;

pub struct Diagnostics {
    pub buffer: wgpu::Buffer,
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    group: Option<wgpu::BindGroup>,
    target: wgpu::TextureView,
}

impl Diagnostics {
    pub fn new(device: &wgpu::Device, original: &str, width: u32, height: u32) -> Self {
        let mut shader = source(original, true).into_owned();
        assert_eq!(shader.matches("@fragment").count(), 1);
        shader = shader.replace("@fragment", "");
        shader.push_str("\n@fragment fn fs_diagnostics(in:VertexOutput)->@location(0)vec4<f32>{let ignored=fs_main(in);return vec4<f32>(0.0);}\n");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TSR separate diagnostic shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TSR separate diagnostic layout"),
            entries: &layout(
                &[
                    tex_entry(0, wgpu::TextureSampleType::Float { filterable: true }),
                    tex_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                    tex_entry(2, wgpu::TextureSampleType::Depth),
                    sampler_entry(3, wgpu::SamplerBindingType::Filtering),
                    sampler_entry(4, wgpu::SamplerBindingType::NonFiltering),
                    camera_storage_entry(5),
                    uniform_entry(6),
                    tex_entry(7, wgpu::TextureSampleType::Float { filterable: false }),
                    tex_entry(8, wgpu::TextureSampleType::Float { filterable: false }),
                    tex_entry(9, wgpu::TextureSampleType::Float { filterable: false }),
                ],
                true,
            ),
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("TSR separate diagnostic draw"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_diagnostics"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::R8Unorm,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            buffer: buffer(device, width, height),
            pipeline,
            bgl,
            group: None,
            target: target(device, width, height),
        }
    }
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.buffer = buffer(device, width, height);
        self.target = target(device, width, height);
        self.group = None;
    }
    pub fn bind(&mut self, device: &wgpu::Device, entries: &[wgpu::BindGroupEntry<'_>]) {
        self.group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TSR separate diagnostic inputs"),
            layout: &self.bgl,
            entries: &bindings(entries, Some(&self.buffer)),
        }));
    }
    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("TSR history decisions diagnostic"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Discard,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.group.as_ref().unwrap(), &[]);
        pass.draw(0..3, 0..1);
    }
}

fn target(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("TSR diagnostic raster target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

pub fn source(shader: &str, enabled: bool) -> Cow<'_, str> {
    if !enabled {
        return Cow::Borrowed(shader);
    }
    let mut source = shader.to_owned();
    for (old,new) in [
        ("fn fs_main(in: VertexOutput) -> ResolveOutput {", "fn fs_main(in: VertexOutput) -> ResolveOutput {\n    let diagnostic_index=(u32(in.position.x)+u32(in.position.y)*textureDimensions(history_frame).x)*2u;\n    history_diagnostics[diagnostic_index]=vec4<f32>(0.0,0.0,1.0,select(64.0,32.0,tsr.reset!=0u));\n    history_diagnostics[diagnostic_index+1u]=vec4<f32>(0.0);"),
        ("    return ResolveOutput(vec4<f32>(result_linear, 1.0), depth_range,", "    history_diagnostics[diagnostic_index]=vec4<f32>(history.weight,history.count,blend,f32(flags));\n    history_diagnostics[diagnostic_index+1u]=vec4<f32>(length(history_tm-clamped_history),length(velocity*out_dims),current_tm.x,history_tm.x);\n    return ResolveOutput(vec4<f32>(result_linear, 1.0), depth_range,")
    ] {
        assert_eq!(source.matches(old).count(),1,"TSR diagnostic anchor changed");
        source=source.replace(old,new);
    }
    source.push_str(
        "\n@group(0) @binding(10) var<storage,read_write> history_diagnostics:array<vec4<f32>>;\n",
    );
    Cow::Owned(source)
}

pub fn buffer(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("TSR history diagnostic records"),
        size: u64::from(width) * u64::from(height) * 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}
pub fn layout(
    entries: &[wgpu::BindGroupLayoutEntry],
    enabled: bool,
) -> Vec<wgpu::BindGroupLayoutEntry> {
    let mut entries = entries.to_vec();
    if enabled {
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 10,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
    }
    entries
}
pub fn bindings<'a>(
    entries: &[wgpu::BindGroupEntry<'a>],
    buffer: Option<&'a wgpu::Buffer>,
) -> Vec<wgpu::BindGroupEntry<'a>> {
    let mut entries = entries.to_vec();
    if let Some(buffer) = buffer {
        entries.push(wgpu::BindGroupEntry {
            binding: 10,
            resource: buffer.as_entire_binding(),
        });
    }
    entries
}
