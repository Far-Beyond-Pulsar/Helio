//! Buffer debug views: the scene depth or the G-buffer's world normals,
//! drawn over the final image.
//!
//! Debug modes inside a lighting pass still go through fog, anti-aliasing,
//! the lens and tone mapping, which blur and remap them. This pass runs after
//! post-processing and overwrites `ctx.target` with the raw buffer instead,
//! so what shows is what the passes read. With any other debug mode it
//! records nothing.
//!
//! - **Scene Depth** (`DEPTH_VIEW`): distance along the view axis, log-scaled
//!   between the near and far planes (white near, black far, sky black).
//! - **Scene Normals** (`NORMALS_VIEW`): world-space normal as RGB
//!   (R = +X, G = +Y, B = +Z).

use helio_core::graph::ResourceBuilder;
use helio_core::{DebugViewDescriptor, PassContext, RenderPass, Result as HelioResult};

/// Debug mode showing the scene depth.
pub const DEPTH_VIEW: u32 = 40;
/// Debug mode showing the G-buffer normals.
pub const NORMALS_VIEW: u32 = 41;

pub struct BufferViewPass {
    mode: u32,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    view_buf: wgpu::Buffer,
    bind_group: Option<wgpu::BindGroup>,
    bind_group_key: Option<(wgpu::TextureView, wgpu::TextureView, wgpu::Buffer)>,
}

impl BufferViewPass {
    /// `target_format` — the format of `ctx.target`.
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = helio_core::shader::module(
            device,
            "Buffer View Shader",
            helio_core::include_wgsl!("../shaders/buffer_view.wgsl"),
        );
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Buffer View BGL"),
            entries: &[
                // 0: scene depth (read as a float texture), 1: G-buffer normal
                texture(0),
                texture(1),
                // 2: cameras
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: which view
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(16),
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Buffer View PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Buffer View Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
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
        let view_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buffer View Mode"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Self {
            mode: 0,
            pipeline,
            bind_group_layout,
            view_buf,
            bind_group: None,
            bind_group_key: None,
        }
    }

    fn active(&self) -> bool {
        matches!(self.mode, DEPTH_VIEW | NORMALS_VIEW)
    }
}

impl RenderPass for BufferViewPass {
    fn name(&self) -> &'static str {
        "BufferView"
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("gbuffer");
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
        storage: &'a mut helio_core::RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        if !self.active() {
            return None;
        }
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            storage.retain_boxed_slice(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Buffer View"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if !self.active() {
            return Ok(());
        }
        let Some(gbuffer) = ctx
            .registry
            .read::<helio_core::ViewGroup<'_, 4>>(helio_core::ResourceKey::new("gbuffer"), "BufferView")
        else {
            return Ok(());
        };
        let key = (ctx.depth.clone(), gbuffer.views[1].clone(), ctx.camera.clone());
        if self.bind_group_key.as_ref() != Some(&key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Buffer View BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&key.0) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&key.1) },
                    wgpu::BindGroupEntry { binding: 2, resource: key.2.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: self.view_buf.as_entire_binding() },
                ],
            }));
            self.bind_group_key = Some(key);
        }
        ctx.queue.write_buffer(&self.view_buf, 0, &[self.mode.to_le_bytes(), [0; 4], [0; 4], [0; 4]].concat());
        let Some(mut rp) = ctx.render_cmds() else { return Ok(()) };
        rp.set_pipeline(&self.pipeline);
        rp.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        rp.draw(0..3, 0..1);
        Ok(())
    }

    fn set_debug_mode(&mut self, mode: u32) {
        self.mode = mode;
    }

    fn debug_views(&self) -> &'static [DebugViewDescriptor] {
        static VIEWS: &[DebugViewDescriptor] = &[
            DebugViewDescriptor {
                name: "Scene Depth",
                debug_mode: DEPTH_VIEW,
                description: "Distance from the camera, log-scaled: white near, black far",
            },
            DebugViewDescriptor {
                name: "Scene Normals",
                debug_mode: NORMALS_VIEW,
                description: "G-buffer world-space normals as RGB (R=+X, G=+Y, B=+Z)",
            },
        ];
        VIEWS
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        self.bind_group = None;
        self.bind_group_key = None;
    }
}
