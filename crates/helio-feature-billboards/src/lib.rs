use bytemuck::{Pod, Zeroable};
use helio_core::{BillboardVertex, TextureId, TextureManager, create_billboard_quad};
use helio_features::{Feature, FeatureContext, ShaderInjection};
use std::sync::Arc;
use wgpu::util::DeviceExt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode { Opaque, Transparent }

#[derive(Clone)]
pub struct BillboardData {
    pub position: [f32; 3],
    pub scale: [f32; 2],
    pub texture: TextureId,
    pub blend_mode: BlendMode,
    pub screen_scale: bool,
}

impl BillboardData {
    pub fn new(position: [f32; 3], scale: [f32; 2], texture: TextureId) -> Self {
        Self { position, scale, texture, blend_mode: BlendMode::Transparent, screen_scale: false }
    }
    pub fn with_blend_mode(mut self, mode: BlendMode) -> Self { self.blend_mode = mode; self }
    pub fn with_screen_scale(mut self, v: bool) -> Self { self.screen_scale = v; self }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BillboardCameraUniforms {
    view_proj: [[f32; 4]; 4],
    position: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BillboardInstanceUniforms {
    world_position: [f32; 3],
    _pad1: f32,
    scale: [f32; 2],
    screen_scale: u32,
    _pad2: u32,
}

pub struct BillboardFeature {
    enabled: bool,
    texture_manager: Option<Arc<TextureManager>>,
    pipeline: Option<wgpu::RenderPipeline>,
    quad_vertex_buffer: Option<wgpu::Buffer>,
    quad_index_buffer: Option<wgpu::Buffer>,
    camera_bgl: Option<wgpu::BindGroupLayout>,
    instance_bgl: Option<wgpu::BindGroupLayout>,
    texture_bgl: Option<wgpu::BindGroupLayout>,
    camera_buffer: Option<wgpu::Buffer>,
    device: Option<Arc<wgpu::Device>>,
    queue: Option<Arc<wgpu::Queue>>,
    depth_format: Option<wgpu::TextureFormat>,
    color_format: Option<wgpu::TextureFormat>,
}

impl BillboardFeature {
    pub fn new() -> Self {
        Self { enabled: true, texture_manager: None, pipeline: None, quad_vertex_buffer: None, quad_index_buffer: None, camera_bgl: None, instance_bgl: None, texture_bgl: None, camera_buffer: None, device: None, queue: None, depth_format: None, color_format: None }
    }
    pub fn set_texture_manager(&mut self, manager: Arc<TextureManager>) { self.texture_manager = Some(manager); }
    pub fn texture_manager(&self) -> Option<&Arc<TextureManager>> { self.texture_manager.as_ref() }

    pub fn render_billboard_overlay(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        view_proj: [[f32; 4]; 4],
        camera_pos: [f32; 3],
        billboards: &[BillboardData],
    ) {
        if billboards.is_empty() { return; }
        let (pipeline, qvb, qib, cam_bgl, inst_bgl, tex_bgl, cam_buf, device, queue, tm) = match (
            &self.pipeline, &self.quad_vertex_buffer, &self.quad_index_buffer,
            &self.camera_bgl, &self.instance_bgl, &self.texture_bgl,
            &self.camera_buffer, &self.device, &self.queue, &self.texture_manager,
        ) {
            (Some(a),Some(b),Some(c),Some(d),Some(e),Some(f),Some(g),Some(h),Some(i),Some(j)) => (a,b,c,d,e,f,g,h,i,j),
            _ => { log::warn!("Billboard feature not fully initialized"); return; }
        };

        let cam = BillboardCameraUniforms { view_proj, position: camera_pos, _pad: 0.0 };
        queue.write_buffer(cam_buf, 0, bytemuck::bytes_of(&cam));

        let cam_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: cam_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: cam_buf.as_entire_binding() }],
        });

        // Pre-create all per-billboard bind groups (must outlive render pass)
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default()
        });
        let mut per_billboard: Vec<(wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroup)> = Vec::new();
        for billboard in billboards {
            let gpu_tex = match tm.get(billboard.texture) {
                Some(t) => t, None => { log::warn!("Unknown billboard texture"); continue; }
            };
            let inst = BillboardInstanceUniforms {
                world_position: billboard.position, _pad1: 0.0,
                scale: billboard.scale, screen_scale: billboard.screen_scale as u32, _pad2: 0,
            };
            let inst_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None, contents: bytemuck::bytes_of(&inst), usage: wgpu::BufferUsages::UNIFORM,
            });
            let inst_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: inst_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: inst_buf.as_entire_binding() }],
            });
            let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: tex_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&gpu_tex.view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&linear_sampler) },
                ],
            });
            per_billboard.push((inst_buf, inst_bg, tex_bg));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("billboard_overlay"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view, resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None, occlusion_query_set: None,
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &cam_bg, &[]);
        pass.set_vertex_buffer(0, qvb.slice(..));
        pass.set_index_buffer(qib.slice(..), wgpu::IndexFormat::Uint32);

        for (_, inst_bg, tex_bg) in &per_billboard {
            pass.set_bind_group(1, inst_bg, &[]);
            pass.set_bind_group(2, tex_bg, &[]);
            pass.draw_indexed(0..6, 0, 0..1);
        }
    }
}

impl Feature for BillboardFeature {
    fn name(&self) -> &str { "billboards" }
    fn is_enabled(&self) -> bool { self.enabled }
    fn set_enabled(&mut self, e: bool) { self.enabled = e; }
    fn shader_injections(&self) -> Vec<ShaderInjection> { Vec::new() }

    fn init(&mut self, ctx: &FeatureContext) {
        log::info!("Initializing billboard feature");
        let device = &ctx.device;
        self.device = Some(ctx.device.clone());
        self.queue = Some(ctx.queue.clone());
        self.depth_format = Some(ctx.depth_format);
        self.color_format = Some(ctx.color_format);

        let quad = create_billboard_quad(1.0);
        self.quad_vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("billboard_vb"), contents: bytemuck::cast_slice(&quad.vertices), usage: wgpu::BufferUsages::VERTEX,
        }));
        self.quad_index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("billboard_ib"), contents: bytemuck::cast_slice(&quad.indices), usage: wgpu::BufferUsages::INDEX,
        }));
        self.camera_buffer = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("billboard_camera"), size: std::mem::size_of::<BillboardCameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false,
        }));

        let cam_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bb_cam_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }],
        });
        let inst_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bb_inst_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::VERTEX, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None }],
        });
        let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bb_tex_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::FRAGMENT, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&cam_bgl, &inst_bgl, &tex_bgl], push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("billboard_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/billboard.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("billboard_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: "vs_main", buffers: &[BillboardVertex::desc()], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState { format: ctx.color_format, blend: Some(wgpu::BlendState::ALPHA_BLENDING), write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, cull_mode: None, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState { format: ctx.depth_format, depth_write_enabled: false, depth_compare: wgpu::CompareFunction::Less, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(), multiview: None,
        });

        self.camera_bgl = Some(cam_bgl);
        self.instance_bgl = Some(inst_bgl);
        self.texture_bgl = Some(tex_bgl);
        self.pipeline = Some(pipeline);
        log::info!("Billboard pipeline created");
    }
}

impl Default for BillboardFeature {
    fn default() -> Self { Self::new() }
}
