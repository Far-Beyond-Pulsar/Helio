//! Depth of Field pass.
//!
//! Two-pass (horizontal + vertical) separable Gaussian blur with per-pixel
//! Circle of Confusion radius derived from depth and focus distance.
//!
//! Both passes write to the HDR buffer (ping-pong via an internal temporary
//! Rgba16Float texture so the horizontal result is not overwritten by the
//! vertical pass read).

use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::Result;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DofUniforms {
    /// World-space distance to the sharpest point.
    pub focus_dist:   f32,
    /// Aperture multiplier (enlarges CoC).  Set to 0 to disable DoF entirely.
    pub aperture:     f32,
    /// Maximum blur radius in screen pixels.
    pub max_blur_px:  f32,
    /// 0 = horizontal pass, 1 = vertical pass (written by execute()).
    pub direction:    u32,
}

impl Default for DofUniforms {
    fn default() -> Self {
        Self { focus_dist: 10.0, aperture: 0.0, max_blur_px: 20.0, direction: 0 }
    }
}

pub struct DofPass {
    pipeline:           Arc<wgpu::RenderPipeline>,
    bind_group_layout:  wgpu::BindGroupLayout,
    sampler:            wgpu::Sampler,
    pub uniform_buffer: wgpu::Buffer,

    // Internal ping-pong texture (same format + size as HDR buffer)
    temp_texture: Option<wgpu::Texture>,
    temp_view:    Option<wgpu::TextureView>,

    // H-pass: reads hdr_view → writes temp
    bind_group_h: Option<wgpu::BindGroup>,
    // V-pass: reads temp    → writes hdr_view
    bind_group_v: Option<wgpu::BindGroup>,

    pub config: DofUniforms,
    width:  u32,
    height: u32,
}

impl DofPass {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DoF BGL"),
            entries: &[
                // 0: scene_color
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // 1: scene_depth
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    }, count: None,
                },
                // 2: linear sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // 3: DofUniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }, count: None,
                },
                // 4: camera uniform (view_proj_inv for depth linearisation)
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("DoF Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("DoF Uniform Buffer"),
            contents: bytemuck::bytes_of(&DofUniforms::default()),
            usage:    wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("DoF Pipeline Layout"),
            bind_group_layouts: &[Some(&bgl as &wgpu::BindGroupLayout)],
            immediate_size:     0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("DoF Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/dof.wgsl").into()
            ),
        });

        let pipeline = Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("DoF Pipeline"),
            layout: Some(&pipeline_layout),
            cache:  None,
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"),
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::Rgba16Float,
                    blend:      Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive:      wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil:  None,
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None,
        }));

        Self {
            pipeline, bind_group_layout: bgl, sampler, uniform_buffer,
            temp_texture: None, temp_view: None,
            bind_group_h: None, bind_group_v: None,
            config: DofUniforms::default(), width, height,
        }
    }

    fn make_bg(
        device:      &wgpu::Device,
        bgl:         &wgpu::BindGroupLayout,
        color_view:  &wgpu::TextureView,
        depth_view:  &wgpu::TextureView,
        samp:        &wgpu::Sampler,
        ubuf:        &wgpu::Buffer,
        camera_buf:  &wgpu::Buffer,
        label:       &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(samp) },
                wgpu::BindGroupEntry { binding: 3, resource: ubuf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: camera_buf.as_entire_binding() },
            ],
        })
    }

    /// Build / rebuild bind groups and the internal ping-pong texture.
    /// Call after construction and on resize.
    pub fn create_bind_groups(
        &mut self,
        device:     &wgpu::Device,
        hdr_view:   &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
    ) {
        self.create_bind_groups_sized(device, hdr_view, depth_view, camera_buf, self.width, self.height)
    }

    /// Like `create_bind_groups` but also updates the stored width/height (call on resize).
    pub fn create_bind_groups_resized(
        &mut self,
        device:     &wgpu::Device,
        hdr_view:   &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
        width:      u32,
        height:     u32,
    ) {
        self.width  = width;
        self.height = height;
        self.create_bind_groups_sized(device, hdr_view, depth_view, camera_buf, width, height);
    }

    fn create_bind_groups_sized(
        &mut self,
        device:     &wgpu::Device,
        hdr_view:   &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
        width:      u32,
        height:     u32,
    ) {
        // (Re)create the ping-pong temp texture.
        let temp_tex = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("DoF Temp"),
            size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let temp_view = temp_tex.create_view(&Default::default());

        self.bind_group_h = Some(Self::make_bg(
            device, &self.bind_group_layout, hdr_view,   depth_view,
            &self.sampler, &self.uniform_buffer, camera_buf, "DoF H BG",
        ));
        self.bind_group_v = Some(Self::make_bg(
            device, &self.bind_group_layout, &temp_view, depth_view,
            &self.sampler, &self.uniform_buffer, camera_buf, "DoF V BG",
        ));

        self.temp_texture = Some(temp_tex);
        self.temp_view    = Some(temp_view);
    }

    pub fn execute(
        &self,
        encoder:  &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
        queue:    &wgpu::Queue,
    ) -> Result<()> {
        if self.config.aperture <= 0.0 { return Ok(()); }
        let (Some(bg_h), Some(bg_v), Some(temp_view)) =
            (&self.bind_group_h, &self.bind_group_v, &self.temp_view)
        else { return Ok(()); };

        // Horizontal pass → temp
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&DofUniforms { direction: 0, ..self.config }));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF H Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: temp_view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg_h, &[]);
            pass.draw(0..3, 0..1);
        }

        // Vertical pass → HDR
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&DofUniforms { direction: 1, ..self.config }));
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("DoF V Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: hdr_view, resolve_target: None, depth_slice: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None, timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, bg_v, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
