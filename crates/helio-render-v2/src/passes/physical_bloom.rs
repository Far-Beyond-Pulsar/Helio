//! Physical bloom pass using UE5-style dual-Kawase downsample + tent-filter upsample.
//!
//! Pipeline:
//!   1. First downsample from HDR source with soft-knee threshold extraction.
//!   2. 5 more downsample passes halving resolution each time (6 mip levels total).
//!   3. 6 upsample passes doubling resolution, accumulating with blend_factor.
//!   4. Final upsample composites additively back onto the HDR source texture.

use std::sync::Arc;
use wgpu::util::DeviceExt;
use crate::Result;

const MIP_COUNT: u32 = 6;

/// Bloom pass controls
#[derive(Copy, Clone)]
pub struct BloomConfig {
    pub threshold:    f32,   // luminance threshold (pre-knee)
    pub knee:         f32,   // soft-knee width
    pub intensity:    f32,   // final bloom multiplier
    pub filter_radius: f32,  // upsample tent filter radius (texels)
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self { threshold: 1.0, knee: 0.5, intensity: 0.04, filter_radius: 0.005 }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DownUniforms { threshold: f32, knee: f32, intensity: f32, is_first: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct UpUniforms { filter_radius: f32, blend_factor: f32, _pad: [f32; 2] }

pub struct PhysicalBloomPass {
    downsample_pipeline: Arc<wgpu::RenderPipeline>,
    upsample_pipeline:   Arc<wgpu::RenderPipeline>,

    // Mip chain:  6 textures each half the previous resolution.
    mip_textures: Vec<wgpu::Texture>,
    mip_views:    Vec<wgpu::TextureView>,

    // Bind groups for each downsample step: reads from [hdr_source, mip0, mip1, …]
    down_bgs: Vec<wgpu::BindGroup>,
    // Bind groups for each upsample step: reads from [mip5, mip4, …, mip0]
    up_bgs:   Vec<wgpu::BindGroup>,
    // Final additive-composite bind group: reads mip0 → writes to HDR source
    composite_bg: Option<wgpu::BindGroup>,

    // Re-usable sampler
    sampler: wgpu::Sampler,

    bgl_down: wgpu::BindGroupLayout,
    bgl_up:   wgpu::BindGroupLayout,

    down_uniform_buf: wgpu::Buffer,
    up_uniform_buf:   wgpu::Buffer,

    pub config: BloomConfig,
    width:  u32,
    height: u32,
}

fn fullscreen_pipeline(
    device:  &wgpu::Device,
    layout:  &wgpu::PipelineLayout,
    shader:  &wgpu::ShaderModule,
    format:  wgpu::TextureFormat,
    blend:   Option<wgpu::BlendState>,
    label:   &str,
) -> Arc<wgpu::RenderPipeline> {
    Arc::new(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label:  Some(label),
        layout: Some(layout),
        cache:  None,
        vertex: wgpu::VertexState {
            module: shader, entry_point: Some("vs_main"),
            buffers: &[], compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader, entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })],
            compilation_options: Default::default(),
        }),
        primitive:      wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleList, ..Default::default() },
        depth_stencil:  None,
        multisample:    wgpu::MultisampleState::default(),
        multiview_mask: None,
    }))
}

impl PhysicalBloomPass {
    pub fn new(
        device: &wgpu::Device,
        width:  u32,
        height: u32,
        config: BloomConfig,
    ) -> Result<Self> {
        let format = wgpu::TextureFormat::Rgba16Float;

        // ── Bind group layouts ───────────────────────────────────────────────
        let tex_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let samp_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let uniform_entry = |binding: u32, size: u64| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: wgpu::BufferSize::new(size),
            },
            count: None,
        };

        let bgl_down = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Down BGL"),
            entries: &[
                tex_entry(0), samp_entry(1),
                uniform_entry(2, std::mem::size_of::<DownUniforms>() as u64),
            ],
        });
        let bgl_up = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom Up BGL"),
            entries: &[
                tex_entry(0), samp_entry(1),
                uniform_entry(2, std::mem::size_of::<UpUniforms>() as u64),
            ],
        });

        // ── Uniform buffers ───────────────────────────────────────────────────
        let down_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Bloom Down Uniforms"),
            contents: bytemuck::bytes_of(&DownUniforms {
                threshold: config.threshold, knee: config.knee,
                intensity: config.intensity, is_first: 1,
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let up_uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label:    Some("Bloom Up Uniforms"),
            contents: bytemuck::bytes_of(&UpUniforms {
                filter_radius: config.filter_radius, blend_factor: 0.5, _pad: [0.0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ── Pipelines ─────────────────────────────────────────────────────────
        let layout_down = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("Bloom Down Layout"),
            bind_group_layouts: &[Some(&bgl_down as &wgpu::BindGroupLayout)],
            immediate_size:     0,
        });
        let layout_up = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("Bloom Up Layout"),
            bind_group_layouts: &[Some(&bgl_up as &wgpu::BindGroupLayout)],
            immediate_size:     0,
        });

        let shader_down = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Bloom Downsample Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/bloom_downsample.wgsl").into()
            ),
        });
        let shader_up = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Bloom Upsample Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/passes/bloom_upsample.wgsl").into()
            ),
        });

        let downsample_pipeline = fullscreen_pipeline(
            device, &layout_down, &shader_down, format, None, "Bloom Downsample Pipeline",
        );
        // Upsample with additive blend so each mip accumulates on the one above it.
        let additive = Some(wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation:  wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        });
        let upsample_pipeline = fullscreen_pipeline(
            device, &layout_up, &shader_up, format, additive, "Bloom Upsample Pipeline",
        );

        // ── Mip chain textures ────────────────────────────────────────────────
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("Bloom Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let mut mip_textures = Vec::with_capacity(MIP_COUNT as usize);
        let mut mip_views    = Vec::with_capacity(MIP_COUNT as usize);
        let mut mw = (width  / 2).max(1);
        let mut mh = (height / 2).max(1);
        for i in 0..MIP_COUNT {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label:           Some(&format!("Bloom Mip {}", i)),
                size:            wgpu::Extent3d { width: mw, height: mh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count:    1,
                dimension:       wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            mip_textures.push(tex);
            mip_views.push(view);
            mw = (mw / 2).max(1);
            mh = (mh / 2).max(1);
        }

        // Bind groups are created lazily via `create_bind_groups`.
        // (We don't have the HDR source view yet at construction time.)
        Ok(Self {
            downsample_pipeline,
            upsample_pipeline,
            mip_textures, mip_views,
            down_bgs: Vec::new(),
            up_bgs:   Vec::new(),
            composite_bg: None,
            sampler, bgl_down, bgl_up,
            down_uniform_buf, up_uniform_buf,
            config, width, height,
        })
    }

    fn make_down_bg(
        device: &wgpu::Device,
        bgl:    &wgpu::BindGroupLayout,
        src:    &wgpu::TextureView,
        samp:   &wgpu::Sampler,
        ubuf:   &wgpu::Buffer,
        label:  &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some(label),
            layout:  bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
                wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
            ],
        })
    }

    fn make_up_bg(
        device: &wgpu::Device,
        bgl:    &wgpu::BindGroupLayout,
        src:    &wgpu::TextureView,
        samp:   &wgpu::Sampler,
        ubuf:   &wgpu::Buffer,
        label:  &str,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some(label),
            layout:  bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(samp) },
                wgpu::BindGroupEntry { binding: 2, resource: ubuf.as_entire_binding() },
            ],
        })
    }

    /// Build all bind groups referencing the HDR source view.
    /// Must be called after construction and on every resize.
    pub fn create_bind_groups(&mut self, device: &wgpu::Device, hdr_view: &wgpu::TextureView) {
        self.down_bgs.clear();
        self.up_bgs.clear();

        // Downsample: hdr → mip0, mip0 → mip1, …, mip4 → mip5
        let srcs: Vec<&wgpu::TextureView> = std::iter::once(hdr_view)
            .chain(self.mip_views.iter().take(MIP_COUNT as usize - 1))
            .collect();
        for (i, src) in srcs.iter().enumerate() {
            self.down_bgs.push(Self::make_down_bg(
                device, &self.bgl_down, src, &self.sampler, &self.down_uniform_buf,
                &format!("Bloom Down BG {}", i),
            ));
        }

        // Upsample: mip5 → mip4, …, mip1 → mip0
        // (outputs to mip[i-1], reads from mip[i])
        for i in (1..MIP_COUNT as usize).rev() {
            self.up_bgs.push(Self::make_up_bg(
                device, &self.bgl_up, &self.mip_views[i], &self.sampler, &self.up_uniform_buf,
                &format!("Bloom Up BG {}", i),
            ));
        }

        // Final composite: mip0 additively onto HDR source.
        self.composite_bg = Some(Self::make_up_bg(
            device, &self.bgl_up, &self.mip_views[0], &self.sampler, &self.up_uniform_buf,
            "Bloom Composite BG",
        ));
    }

    /// Upload latest config values to the GPU uniform buffers.
    pub fn upload_uniforms(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.down_uniform_buf, 0, bytemuck::bytes_of(&DownUniforms {
            threshold: self.config.threshold,
            knee:      self.config.knee,
            intensity: self.config.intensity,
            is_first:  1,
        }));
        queue.write_buffer(&self.up_uniform_buf, 0, bytemuck::bytes_of(&UpUniforms {
            filter_radius: self.config.filter_radius,
            blend_factor:  0.5,
            _pad:          [0.0; 2],
        }));
    }

    fn render_fullscreen(
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::RenderPipeline,
        bg:       &wgpu::BindGroup,
        target:   &wgpu::TextureView,
        label:    &str,
        load:     wgpu::LoadOp<wgpu::Color>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some(label),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
            multiview_mask:           None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Execute all bloom passes.
    /// On return the result is additively composited back onto `hdr_view`.
    pub fn execute(
        &self,
        encoder:  &mut wgpu::CommandEncoder,
        hdr_view: &wgpu::TextureView,
    ) -> Result<()> {
        if self.down_bgs.len() < MIP_COUNT as usize { return Ok(()); }

        // Downsample pass 0: threshold extraction from HDR source → mip0
        Self::render_fullscreen(
            encoder, &self.downsample_pipeline, &self.down_bgs[0],
            &self.mip_views[0], "Bloom Down 0", wgpu::LoadOp::Clear(wgpu::Color::BLACK),
        );
        // Downsample passes 1-5: each mip → next
        for i in 1..MIP_COUNT as usize {
            Self::render_fullscreen(
                encoder, &self.downsample_pipeline, &self.down_bgs[i],
                &self.mip_views[i], &format!("Bloom Down {}", i),
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            );
        }

        // Upsample passes: mip5 → mip4, …, mip1 → mip0  (additive blend)
        for (k, bg) in self.up_bgs.iter().enumerate() {
            let target_idx = MIP_COUNT as usize - 2 - k;
            Self::render_fullscreen(
                encoder, &self.upsample_pipeline, bg,
                &self.mip_views[target_idx], &format!("Bloom Up {}", k),
                // LoadOp::Load so additive blend accumulates correctly.
                wgpu::LoadOp::Load,
            );
        }

        // Final additive composite: mip0 → HDR source (LoadOp::Load preserves scene)
        if let Some(composite_bg) = &self.composite_bg {
            Self::render_fullscreen(
                encoder, &self.upsample_pipeline, composite_bg,
                hdr_view, "Bloom Composite", wgpu::LoadOp::Load,
            );
        }

        Ok(())
    }
}
