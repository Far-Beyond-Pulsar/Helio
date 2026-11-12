//! The status screen: what a renderer draws when it has no pipeline, or its
//! pipeline is invalid.
//!
//! One fullscreen pass over the output frame. A fragment shader draws a
//! near-black field and one red dialog from a rounded-rectangle signed
//! distance field in physical pixels (resolution and DPI independent,
//! anti-aliased analytically), then the text, from the engine's own 8x8 glyph
//! atlas (`helio_pass_debug_overlay`), and grain. There is no glow, no bloom
//! and no image.
//!
//! The text is laid out on the CPU once per frame size ([`layout::compute`])
//! and uploaded as a handful of runs and rectangles; a frame costs one draw.

pub mod abi;
pub mod content;
pub mod layout;
pub mod theme;

pub use content::{Detail, Severity, StatusContent, Tone};
pub use layout::Layout;

use helio_core::{PassContext, PrepareContext, RenderPass, RenderSize, Result as HelioResult};

/// The pass: draws the status screen over the whole output.
pub struct StatusScreenPass {
    content: StatusContent,
    size: [u32; 2],
    srgb_target: bool,
    layout: Option<Layout>,
    /// The layout is not on the GPU yet (first frame, or the size changed).
    dirty: bool,
    pipeline: wgpu::RenderPipeline,
    params: wgpu::Buffer,
    lines: wgpu::Buffer,
    rects: wgpu::Buffer,
    chars: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    _atlas: wgpu::Texture,
}

impl StatusScreenPass {
    /// A pass for `content` on an output of `size` pixels in `target_format`.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target_format: wgpu::TextureFormat,
        size: [u32; 2],
        content: StatusContent,
    ) -> Self {
        let shader = helio_core::shader::module(
            device,
            "StatusScreen Shader",
            helio_core::include_wgsl!("../shaders/status_screen.wgsl"),
        );
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("StatusScreen BGL"),
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
                storage(1),
                storage(2),
                storage(3),
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("StatusScreen PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = helio_core::shader::create_render_pipeline(
            device,
            wgpu::RenderPipelineDescriptor {
                label: Some("StatusScreen Pipeline"),
                layout: Some(&layout),
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
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            },
        );

        let buffer = |label, size: usize, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let params = buffer(
            "StatusScreen Params",
            std::mem::size_of::<abi::Params>(),
            wgpu::BufferUsages::UNIFORM,
        );
        let lines = buffer(
            "StatusScreen Lines",
            abi::MAX_LINES * std::mem::size_of::<abi::Line>(),
            wgpu::BufferUsages::STORAGE,
        );
        let rects = buffer(
            "StatusScreen Rects",
            abi::MAX_RECTS * std::mem::size_of::<abi::Rect>(),
            wgpu::BufferUsages::STORAGE,
        );
        let chars = buffer(
            "StatusScreen Chars",
            abi::MAX_CHARS * std::mem::size_of::<u32>(),
            wgpu::BufferUsages::STORAGE,
        );
        let atlas = helio_pass_debug_overlay::create_font_texture(device, queue);
        let atlas_view = atlas.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("StatusScreen Font Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("StatusScreen BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lines.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rects.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: chars.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        Self {
            content,
            size: [size[0].max(1), size[1].max(1)],
            srgb_target: target_format.is_srgb(),
            layout: None,
            dirty: true,
            pipeline,
            params,
            lines,
            rects,
            chars,
            bind_group,
            _atlas: atlas,
        }
    }

    /// The layout the pass drew last (`None` before its first frame).
    pub fn layout(&self) -> Option<&Layout> {
        self.layout.as_ref()
    }

    /// The text the pass shows.
    pub fn content(&self) -> &StatusContent {
        &self.content
    }
}

impl RenderPass for StatusScreenPass {
    fn name(&self) -> &'static str {
        "StatusScreen"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if !self.dirty {
            return Ok(());
        }
        let layout = layout::compute(&self.content, self.size[0], self.size[1]);
        ctx.write_buffer(
            &self.params,
            0,
            bytemuck::bytes_of(&layout.params(self.srgb_target)),
        );
        ctx.write_buffer(&self.lines, 0, bytemuck::cast_slice(&layout.lines));
        ctx.write_buffer(&self.rects, 0, bytemuck::cast_slice(&layout.rects));
        ctx.write_buffer(&self.chars, 0, bytemuck::cast_slice(&layout.chars));
        self.layout = Some(layout);
        self.dirty = false;
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let attachments = [Some(wgpu::RenderPassColorAttachment {
            view: ctx.target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                // Every pixel is written: the previous frame is not read.
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let desc = wgpu::RenderPassDescriptor {
            label: Some("StatusScreen"),
            color_attachments: &attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_render_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn on_resize(&mut self, _device: &wgpu::Device, size: RenderSize) {
        let size = size.output;
        if size != self.size {
            self.size = size;
            self.dirty = true;
        }
    }

    /// Every pixel of the output is written, so the host need not clear it.
    fn initializes_target(&self) -> bool {
        true
    }

    /// This pass is the frame's image.
    fn writes_target(&self) -> bool {
        true
    }
}
