/// Deferred lighting pass — fullscreen triangle that reads GBuffer and writes pre_aa.
///
/// Cook-Torrance BRDF + shadow atlas PCF + RC GI probe lookup + ACES tonemapping.
/// Bloom is done inline in this shader via WGSL `override` constants — no separate pass.
use std::sync::Arc;
use crate::graph::pass::{RenderPass, PassContext};

pub struct DeferredLightingPass {
    pipeline:     Arc<wgpu::RenderPipeline>,
    bind_group:   wgpu::BindGroup,
}

impl DeferredLightingPass {
    pub fn new(
        device:           &wgpu::Device,
        surface_format:   wgpu::TextureFormat,
        camera_buffer:    &wgpu::Buffer,
        globals_buffer:   &wgpu::Buffer,
        light_buffer:     &wgpu::Buffer,
        gbuf_albedo_view: &wgpu::TextureView,
        gbuf_normal_view: &wgpu::TextureView,
        gbuf_orm_view:    &wgpu::TextureView,
        gbuf_emissive_view: &wgpu::TextureView,
        depth_view:       &wgpu::TextureView,
        shadow_atlas_view: &wgpu::TextureView,   // may be stub 1×1
        shadow_matrix_buffer: &wgpu::Buffer,
        rc_cascade0_view: &wgpu::TextureView,    // may be stub 1×1
        env_cube_view:    &wgpu::TextureView,    // may be stub 1×1 cube
        linear_sampler:   &wgpu::Sampler,
        shadow_sampler:   &wgpu::Sampler,
    ) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("deferred_bgl"),
            entries: &[
                uniform_entry(0, wgpu::ShaderStages::FRAGMENT),
                uniform_entry(1, wgpu::ShaderStages::FRAGMENT),
                storage_ro_entry(2, wgpu::ShaderStages::FRAGMENT),
                // gbuffer
                tex2d_entry(3, wgpu::ShaderStages::FRAGMENT),
                tex2d_entry(4, wgpu::ShaderStages::FRAGMENT),
                tex2d_entry(5, wgpu::ShaderStages::FRAGMENT),
                tex2d_entry(6, wgpu::ShaderStages::FRAGMENT),
                // depth
                tex2d_depth_entry(7, wgpu::ShaderStages::FRAGMENT),
                // shadow
                tex2d_array_depth_entry(8, wgpu::ShaderStages::FRAGMENT),
                shadow_sampler_entry(9, wgpu::ShaderStages::FRAGMENT),
                storage_ro_entry(10, wgpu::ShaderStages::FRAGMENT), // shadow matrices
                // RC GI
                tex2d_entry(11, wgpu::ShaderStages::FRAGMENT),
                // env cube
                tex_cube_entry(12, wgpu::ShaderStages::FRAGMENT),
                sampler_entry(13, wgpu::ShaderStages::FRAGMENT),
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("deferred_bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0,  resource: camera_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1,  resource: globals_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2,  resource: light_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3,  resource: wgpu::BindingResource::TextureView(gbuf_albedo_view) },
                wgpu::BindGroupEntry { binding: 4,  resource: wgpu::BindingResource::TextureView(gbuf_normal_view) },
                wgpu::BindGroupEntry { binding: 5,  resource: wgpu::BindingResource::TextureView(gbuf_orm_view) },
                wgpu::BindGroupEntry { binding: 6,  resource: wgpu::BindingResource::TextureView(gbuf_emissive_view) },
                wgpu::BindGroupEntry { binding: 7,  resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 8,  resource: wgpu::BindingResource::TextureView(shadow_atlas_view) },
                wgpu::BindGroupEntry { binding: 9,  resource: wgpu::BindingResource::Sampler(shadow_sampler) },
                wgpu::BindGroupEntry { binding: 10, resource: shadow_matrix_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::TextureView(rc_cascade0_view) },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(env_cube_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(linear_sampler) },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("deferred_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/deferred_lighting.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("deferred_layout"),
            bind_group_layouts:   &[Some(&bgl)],
            immediate_size:       0,
        });
        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("deferred_lighting"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format, blend: None, write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview_mask: None, cache: None,
        }));

        DeferredLightingPass { pipeline, bind_group }
    }
}

impl RenderPass for DeferredLightingPass {
    fn execute(&mut self, ctx: &mut PassContext) {
        let mut rpass = ctx.encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("deferred_lighting"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &ctx.frame_tex.pre_aa_view,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                depth_slice: None,
            })],
            ..Default::default()
        });
        rpass.set_pipeline(&self.pipeline);
        rpass.set_bind_group(0, &self.bind_group, &[]);
        rpass.draw(0..3, 0..1);
    }
}

fn uniform_entry(b: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn storage_ro_entry(b: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: vis,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn tex2d_entry(b: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: vis,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None }
}
fn tex2d_depth_entry(b: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: vis,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None }
}
fn tex2d_array_depth_entry(b: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: vis,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false }, count: None }
}
fn shadow_sampler_entry(b: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: vis,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None }
}
fn sampler_entry(b: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: vis,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None }
}
fn tex_cube_entry(b: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding: b, visibility: vis,
        ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::Cube, multisampled: false }, count: None }
}
