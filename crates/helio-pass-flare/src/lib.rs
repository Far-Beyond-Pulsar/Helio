//! Lens flare / glare pass for Helio.
//!
//! GPU pipeline (both stages run in a single `execute()` call):
//!   1. **Flare Query** (compute) — projects each flare-enabled light to screen
//!      space, checks occlusion against the depth buffer, writes a compacted
//!      list of visible flares.
//!   2. **Flare Render** (fullscreen tri) — reads the compacted flare list and
//!      renders ghost reflections + halation over the scene with additive
//!      blending.
//!
//! The pass is a no-op when no lights have `flare_enabled != 0`.

use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const MAX_FLARES: u32 = 64;
const WG: u32 = 64;

/// Matches FlareUniforms in both shaders (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FlareUniforms {
    light_count: u32,
    max_flares: u32,
    screen_width: f32,
    screen_height: f32,
}

pub struct LensFlarePass {
    query_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,

    query_bgl: wgpu::BindGroupLayout,
    render_bgl: wgpu::BindGroupLayout,

    flare_query_buf: wgpu::Buffer,
    flare_count_buf: wgpu::Buffer,
    uniform_buf: wgpu::Buffer,

    // Procedural flare atlas
    _flare_tex: wgpu::Texture,
    flare_view: wgpu::TextureView,
    flare_sampler: wgpu::Sampler,

    // Bind groups
    query_bg: Option<wgpu::BindGroup>,
    render_bg: Option<wgpu::BindGroup>,
    bg_key: Option<(usize, usize, usize, usize)>,

    width: u32,
    height: u32,
    format: wgpu::TextureFormat,

    active_flare_count: u32,
}

impl LensFlarePass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        lights_buf: &wgpu::Buffer,
        width: u32,
        height: u32,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let query_src = include_str!("../shaders/flare_query.wgsl");
        let render_src = include_str!("../shaders/flare_render.wgsl");
        let query_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LensFlare Query"),
            source: wgpu::ShaderSource::Wgsl(query_src.into()),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("LensFlare Render"),
            source: wgpu::ShaderSource::Wgsl(render_src.into()),
        });

        // ── Buffers ──

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LensFlare Uniforms"),
            size: std::mem::size_of::<FlareUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let flare_query_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LensFlare Query Buffer"),
            size: MAX_FLARES as u64 * std::mem::size_of::<libhelio::GpuFlareQuery>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let flare_count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LensFlare Count Buffer"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Procedural flare atlas (4×4 grid, 128×128) ──

        let atlas_size = 128u32;
        let atlas_cells = 4u32;
        let tex_data = Self::make_atlas(atlas_size, atlas_cells);
        let flare_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("LensFlare Atlas"),
            size: wgpu::Extent3d { width: atlas_size, height: atlas_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &flare_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &tex_data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(atlas_size * 4), rows_per_image: Some(atlas_size) },
            wgpu::Extent3d { width: atlas_size, height: atlas_size, depth_or_array_layers: 1 },
        );
        let flare_view = flare_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let flare_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("LensFlare Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ── BGLs ──

        let query_bgl = Self::create_query_bgl(device);
        let render_bgl = Self::create_render_bgl(device);

        // ── Pipelines ──

        let query_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LensFlare Query PL"),
            bind_group_layouts: &[Some(&query_bgl)],
            immediate_size: 0,
        });
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("LensFlare Render PL"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        let query_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LensFlare Query"),
            layout: Some(&query_pl),
            module: &query_shader,
            entry_point: Some("cs_flare_query"),
            compilation_options: Default::default(),
            cache: None,
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("LensFlare Render"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_fullscreen"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_flare"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
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

        let bg_key = (lights_buf as *const _ as usize, 0, 0, 0);
        Self {
            query_pipeline,
            render_pipeline,
            query_bgl,
            render_bgl,
            flare_query_buf,
            flare_count_buf,
            uniform_buf,
            _flare_tex: flare_tex,
            flare_view,
            flare_sampler,
            query_bg: None,
            render_bg: None,
            bg_key: Some(bg_key),
            width,
            height,
            format: surface_format,
            active_flare_count: 0,
        }
    }

    // ── BGL helpers ──

    fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding, visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_entry(binding: u32, vis: wgpu::ShaderStages, ro: bool) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding, visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false, min_binding_size: None,
            },
            count: None,
        }
    }

    fn create_query_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use wgpu::ShaderStages as SS;
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LensFlare Query BGL"),
            entries: &[
                Self::storage_entry(0, SS::COMPUTE, true),   // lights
                Self::storage_entry(1, SS::COMPUTE, false),  // flare_queries
                Self::storage_entry(2, SS::COMPUTE, false),  // flare_count
                Self::storage_entry(3, SS::COMPUTE, true),   // camera
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: SS::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                Self::uniform_entry(5, SS::COMPUTE),          // flare_uniforms
            ],
        })
    }

    fn create_render_bgl(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        use wgpu::ShaderStages as SS;
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LensFlare Render BGL"),
            entries: &[
                Self::storage_entry(0, SS::FRAGMENT, true),   // flare_queries
                Self::storage_entry(1, SS::FRAGMENT, true),   // flare_count
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: SS::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: SS::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                Self::uniform_entry(4, SS::FRAGMENT),          // flare_uniforms
            ],
        })
    }

    fn build_query_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        lights: &wgpu::Buffer,
        queries: &wgpu::Buffer,
        count: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        depth: &wgpu::TextureView,
        uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LensFlare Query BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: queries.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 5, resource: uniforms.as_entire_binding() },
            ],
        })
    }

    fn build_render_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        queries: &wgpu::Buffer,
        count: &wgpu::Buffer,
        atlas_view: &wgpu::TextureView,
        atlas_sampler: &wgpu::Sampler,
        uniforms: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LensFlare Render BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: queries.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: count.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(atlas_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(atlas_sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: uniforms.as_entire_binding() },
            ],
        })
    }

    // ── Procedural atlas generation ──
    // 4×4 grid of flare sprites:
    //   Row 0: Soft blobs (ghost sprites)
    //   Row 1: Ghosts with ring falloff
    //   Row 2: Streaks (anamorphic-style)
    //   Row 3: Halos / rings

    fn make_atlas(atlas_size: u32, cells: u32) -> Vec<u8> {
        let cell = atlas_size / cells;
        let half = cell as f32 * 0.5;
        let mut data = vec![0u8; (atlas_size * atlas_size * 4) as usize];

        // Per-cell tint colours (warm → cool variation)
        let tints: [[f32; 3]; 16] = [
            [1.0, 0.95, 0.90], [0.95, 0.92, 1.0], [1.0, 0.85, 0.80], [0.85, 0.90, 1.0],
            [1.0, 1.0, 0.95], [0.90, 0.85, 1.0], [1.0, 0.80, 0.85], [0.80, 0.95, 1.0],
            [1.0, 0.90, 0.85], [0.85, 0.95, 1.0], [1.0, 0.85, 0.75], [0.80, 0.85, 1.0],
            [0.95, 0.95, 1.0], [1.0, 0.90, 0.95], [1.0, 1.0, 1.0],  [0.90, 0.85, 0.95],
        ];

        for sprite in 0..16u32 {
            let col = sprite % cells;
            let row = sprite / cells;
            let ox = col * cell;
            let oy = row * cell;
            let (_tr, _tg, _tb) = (tints[sprite as usize][0], tints[sprite as usize][1], tints[sprite as usize][2]);

            for py in 0..cell {
                for px in 0..cell {
                    let dx = (px as f32 + 0.5 - half) / half;
                    let dy = (py as f32 + 0.5 - half) / half;
                    let r = (dx * dx + dy * dy).sqrt().min(1.0);
                    let _a = r.max(0.001);

                    let (red, green, blue, mut alpha) = match (row, col) {
                        // Row 0: Gaussian blobs with varying softness
                        (0, 0) => {
                            let s = 2.0; // sigma
                            let g = (-r * r * s).exp();
                            (g, g * 0.97, g * 0.93, g)
                        }
                        (0, 1) => {
                            let s = 4.0;
                            let g = (-r * r * s).exp();
                            (g * 0.95, g, g * 0.98, g)
                        }
                        (0, 2) => {
                            let s = 10.0;
                            let g = (-r * r * s).exp();
                            (g, g * 0.92, g * 0.85, g)
                        }
                        (0, 3) => {
                            let s = 1.5;
                            let g = (-r * r * s).exp();
                            (g * 0.9, g * 0.95, g, g)
                        }

                        // Row 1: Rings and interference patterns
                        (1, 0) => {
                            let rings = ((r * 20.0).sin() * 0.5 + 0.5) * (1.0 - r).max(0.0);
                            let core = (-r * r * 6.0).exp();
                            (core + rings * 0.4, core * 0.9 + rings * 0.3, core * 0.85 + rings * 0.2, (core + rings * 0.4).min(1.0))
                        }
                        (1, 1) => {
                            let rings = ((r * 12.0 - 1.5).sin() * 0.5 + 0.5) * (1.0 - r * r).max(0.0);
                            let a = rings * 0.8;
                            (a, a * 0.9, a * 0.7, a)
                        }
                        (1, 2) => {
                            let ring1 = ((r * 15.0).sin().abs()) * (1.0 - r).max(0.0);
                            let ring2 = ((r * 25.0).cos().abs() * 0.3) * (1.0 - r).max(0.0);
                            let sum = (ring1 + ring2).min(1.0);
                            (sum * 0.9, sum, sum * 0.95, sum)
                        }
                        (1, 3) => {
                            let inner = 0.1;
                            let outer = 0.6;
                            let ring = 1.0 - ((r - (inner + outer) * 0.5).abs() / ((outer - inner) * 0.5)).clamp(0.0, 1.0);
                            let glow = (-r * r * 3.0).exp() * 0.5;
                            let v = (ring * ring + glow).min(1.0);
                            (v, v * 0.85, v * 0.7, v)
                        }

                        // Row 2: Aperture/bokeh shapes (hexagonal and polygonal)
                        (2, 0) => {
                            // Hexagonal aperture
                            let sides = 6.0;
                            let angle = dy.atan2(dx);
                            let closest = (angle % (6.2832 / sides) - 3.1416 / sides).abs();
                            let hex_r = r / (0.9 / (closest * sides * 0.5).cos().max(0.01));
                            let falloff = 1.0 - hex_r;
                            let v = falloff.clamp(0.0, 1.0);
                            let glow = (-r * r * 8.0).exp() * 0.3;
                            ((v + glow).min(1.0), (v * 0.85 + glow).min(1.0), (v * 0.8 + glow).min(1.0), (v + glow).min(1.0))
                        }
                        (2, 1) => {
                            // Octagonal aperture with bright edges
                            let sides = 8.0;
                            let angle = dy.atan2(dx);
                            let closest = (angle % (6.2832 / sides) - 3.1416 / sides).abs();
                            let oct_r = r / (0.92 / (closest * sides * 0.5).cos().max(0.01));
                            let falloff = 1.0 - oct_r;
                            let v = falloff.clamp(0.0, 1.0);
                            let bright_edge = (1.0 - (r - 0.7).abs() / 0.15).clamp(0.0, 1.0) * 0.5;
                            ((v + bright_edge).min(1.0), (v * 0.9).min(1.0), (v * 0.85).min(1.0), (v + bright_edge).min(1.0))
                        }
                        (2, 2) => {
                            // Hexagonal with soft glow
                            let sides = 6.0;
                            let angle = dy.atan2(dx);
                            let closest = (angle % (6.2832 / sides) - 3.1416 / sides).abs();
                            let hex_r = r / (0.85 / (closest * sides * 0.5).cos().max(0.01));
                            let soft = 1.0 / (1.0 + hex_r * hex_r * 4.0);
                            (soft, soft * 0.92, soft * 0.88, soft)
                        }
                        (2, 3) => {
                            // Soft circular bokeh
                            let soft = 1.0 / (1.0 + r * r * 6.0);
                            let rim = (1.0 - (r - 0.4).abs() / 0.3).clamp(0.0, 1.0) * 0.2;
                            ((soft + rim).min(1.0), soft * 0.95, soft * 0.9, (soft + rim).min(1.0))
                        }

                        // Row 3: Streaks and star patterns
                        (3, 0) => {
                            // Horizontal streak
                            let sx = dx.abs() * 0.15;
                            let sy = dy * dy * 6.0;
                            let s = (-sx - sy).exp() * 0.8;
                            (s, s * 0.95, s * 0.9, s)
                        }
                        (3, 1) => {
                            // Vertical streak
                            let sx = dx * dx * 6.0;
                            let sy = dy.abs() * 0.15;
                            let s = (-sx - sy).exp() * 0.8;
                            (s * 0.9, s * 0.95, s, s)
                        }
                        (3, 2) => {
                            // 4-point star cross
                            let cross = (-dx * dx * 20.0).exp() * (-dy * dy * 20.0).exp();
                            let sc = (-r * r * 2.0).exp();
                            (cross + sc * 0.3, cross * 0.9 + sc * 0.25, cross * 0.85 + sc * 0.2, (cross + sc * 0.3).min(1.0))
                        }
                        (3, 3) => {
                            // 6-point star / diffraction spikes
                            let angle = dy.atan2(dx);
                            let spike = (0..6).fold(0.0, |acc, i| {
                                let a = i as f32 * 1.0472;
                                let d = (angle - a).abs();
                                let w = d.min(3.1416 - d);
                                acc + (-w * w * 200.0).exp() * (-r * 8.0).exp()
                            });
                            let core = (-r * r * 4.0).exp();
                            let v = (spike + core).min(1.0);
                            (v, v * 0.92, v * 0.88, v)
                        }
                        _ => (0.0, 0.0, 0.0, 0.0),
                    };

                    alpha = alpha.clamp(0.0, 1.0);
                    let base = ((oy + py) * atlas_size + ox + px) as usize * 4;
                    data[base] = (red.clamp(0.0, 1.0) * 255.0) as u8;
                    data[base + 1] = (green.clamp(0.0, 1.0) * 255.0) as u8;
                    data[base + 2] = (blue.clamp(0.0, 1.0) * 255.0) as u8;
                    data[base + 3] = (alpha * 255.0) as u8;
                }
            }
        }
        data
    }
}

impl RenderPass for LensFlarePass {
    fn name(&self) -> &'static str {
        "LensFlare"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["depth"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("depth");
    }

    fn on_resize(&mut self, _device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let light_count = ctx.scene.lights.len() as u32;
        self.active_flare_count = light_count;

        let uniforms = FlareUniforms {
            light_count,
            max_flares: MAX_FLARES,
            screen_width: self.width as f32,
            screen_height: self.height as f32,
        };
        ctx.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        // Reset atomic flare count to 0
        ctx.write_buffer(&self.flare_count_buf, 0, &[0u8; 4]);

        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        if self.active_flare_count == 0 {
            return None;
        }
        let target_view = resources.pre_aa.get().unwrap_or(target);
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] =
            Box::leak(Box::new([Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("LensFlare"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.active_flare_count == 0 {
            return Ok(());
        }

        // Rebuild bind groups when buffer/depth pointers change
        let lights_ptr = ctx.scene.lights as *const _ as usize;
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let depth_ptr = ctx.depth as *const _ as usize;
        let uniform_ptr = &self.uniform_buf as *const _ as usize;
        let key = (lights_ptr, camera_ptr, depth_ptr, uniform_ptr);

        if self.bg_key != Some(key) {
            self.query_bg = None;
            self.render_bg = None;
        }

        if self.query_bg.is_none() {
            let qbg = Self::build_query_bg(
                ctx.device, &self.query_bgl,
                ctx.scene.lights, &self.flare_query_buf, &self.flare_count_buf,
                ctx.scene.camera, ctx.depth, &self.uniform_buf,
            );
            let rbg = Self::build_render_bg(
                ctx.device, &self.render_bgl,
                &self.flare_query_buf, &self.flare_count_buf,
                &self.flare_view, &self.flare_sampler, &self.uniform_buf,
            );
            self.query_bg = Some(qbg);
            self.render_bg = Some(rbg);
            self.bg_key = Some(key);
        }

        let qbg = self.query_bg.as_ref().unwrap();
        let rbg = self.render_bg.as_ref().unwrap();

        // Pass 1: Flare query compute
        {
            let mut cpass = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("LensFlare Query"),
                    timestamp_writes: None,
                },
            );
            cpass.set_pipeline(&self.query_pipeline);
            cpass.set_bind_group(0, qbg, &[]);
            let wg_count = (self.active_flare_count + WG - 1) / WG;
            cpass.dispatch_workgroups(wg_count.max(1), 1, 1);
        }

        // Pass 2: Flare render — draws into the active render pass
        if let Some(rp_ptr) = ctx.active_render_pass_ptr() {
            let rp = unsafe { &mut *rp_ptr };
            rp.set_pipeline(&self.render_pipeline);
            rp.set_bind_group(0, rbg, &[]);
            rp.draw(0..3, 0..1);
        }

        Ok(())
    }
}
