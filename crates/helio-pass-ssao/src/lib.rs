//! Screen-space ambient occlusion pass.
//!
//! Reads GBuffer depth + normals, outputs a full-screen R8Unorm AO texture.
//! O(1) CPU: single fullscreen draw.

use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};
use bytemuck::{Pod, Zeroable};

const KERNEL_SIZE: usize = 64;
const NOISE_DIM: u32 = 4;

/// Camera uniform matching ssao.wgsl CameraUniform (272 bytes, 4 × mat4 + vec3 + pad).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsaoCameraUniform {
    view:          [[f32; 4]; 4],
    proj:          [[f32; 4]; 4],
    view_proj:     [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    position:      [f32; 3],
    _pad0:         f32,
}

/// Globals matching ssao.wgsl Globals (80 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GpuGlobals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     [f32; 4],
    rc_world_min:      [f32; 4],
    rc_world_max:      [f32; 4],
    csm_splits:        [f32; 4],
}

/// SSAO parameters matching ssao.wgsl SsaoUniform (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsaoUniform {
    radius:      f32,
    bias:        f32,
    power:       f32,
    samples:     u32,
    noise_scale: [f32; 2],
    _pad:        [f32; 2],
}

pub struct SsaoPass {
    pipeline:          wgpu::RenderPipeline,
    #[allow(dead_code)]
    bgl_0:             wgpu::BindGroupLayout,
    #[allow(dead_code)]
    bgl_1:             wgpu::BindGroupLayout,
    bgl_2:             wgpu::BindGroupLayout,
    bind_group_0:      wgpu::BindGroup,
    bind_group_1:      wgpu::BindGroup,
    bind_group_2:      wgpu::BindGroup,
    ssao_camera_buf:   wgpu::Buffer,
    globals_buf:       wgpu::Buffer,
    ssao_uniform_buf:  wgpu::Buffer,
    sample_kernel_buf: wgpu::Buffer,
    noise_texture:     wgpu::Texture,
    noise_sampler:     wgpu::Sampler,
    pub ssao_texture:  wgpu::Texture,
    pub ssao_view:     wgpu::TextureView,
}

impl SsaoPass {
    pub fn new(
        device:        &wgpu::Device,
        queue:         &wgpu::Queue,
        width:         u32,
        height:        u32,
        gbuf_albedo:   &wgpu::TextureView,
        gbuf_normal:   &wgpu::TextureView,
        gbuf_orm:      &wgpu::TextureView,
        gbuf_emissive: &wgpu::TextureView,
        gbuf_depth:    &wgpu::TextureView,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("SSAO Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/ssao.wgsl").into()),
        });

        // ── Buffers ────────────────────────────────────────────────────────────

        let ssao_camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("SSAO Camera"),
            size:               std::mem::size_of::<SsaoCameraUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("SSAO Globals"),
            size:               std::mem::size_of::<GpuGlobals>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let ssao_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("SSAO Uniform"),
            size:               std::mem::size_of::<SsaoUniform>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 64-sample hemisphere kernel — fixed size, O(1)
        let kernel = generate_kernel();
        let sample_kernel_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("SSAO Kernel"),
            size:               (KERNEL_SIZE * std::mem::size_of::<[f32; 4]>()) as u64,
            usage:              wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        helio_v3::upload::write_buffer(queue, &sample_kernel_buf, 0, bytemuck::cast_slice(&kernel));

        // ── Noise texture (4×4 Rgba8Unorm, random rotation vectors) ───────────
        let noise_data = generate_noise();
        let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label:           Some("SSAO Noise"),
            size:            wgpu::Extent3d { width: NOISE_DIM, height: NOISE_DIM, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count:    1,
            dimension:       wgpu::TextureDimension::D2,
            format:          wgpu::TextureFormat::Rgba8Unorm,
            usage:           wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats:    &[],
        });
        helio_v3::upload::write_texture(
            queue,
            wgpu::ImageCopyTexture {
                texture:   &noise_texture,
                mip_level: 0,
                origin:    wgpu::Origin3d::ZERO,
                aspect:    wgpu::TextureAspect::All,
            },
            &noise_data,
            wgpu::ImageDataLayout {
                offset:         0,
                bytes_per_row:  Some(NOISE_DIM * 4),
                rows_per_image: Some(NOISE_DIM),
            },
            wgpu::Extent3d { width: NOISE_DIM, height: NOISE_DIM, depth_or_array_layers: 1 },
        );
        let noise_view = noise_texture.create_view(&Default::default());

        // Non-filtering repeat sampler for the noise tile (also used for depth reads)
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:          Some("SSAO Noise Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter:     wgpu::FilterMode::Nearest,
            min_filter:     wgpu::FilterMode::Nearest,
            mipmap_filter:  wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ── SSAO output texture ────────────────────────────────────────────────
        let (ssao_texture, ssao_view) = make_ssao_texture(device, width, height);

        // ── Bind group layouts ─────────────────────────────────────────────────

        // Group 0: camera + globals
        let bgl_0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("SSAO BGL0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        // Group 1: GBuffer textures (albedo, normal, orm, emissive — float; depth)
        let bgl_1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("SSAO BGL1"),
            entries: &[
                gbuf_float_entry(0),
                gbuf_float_entry(1),
                gbuf_float_entry(2),
                gbuf_float_entry(3),
                // binding 4: depth
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
            ],
        });

        // Group 2: ssao uniform, kernel (storage read), noise tex, noise sampler
        let bgl_2 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("SSAO BGL2"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty:         wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count:      None,
                },
            ],
        });

        // ── Bind groups ────────────────────────────────────────────────────────

        let bind_group_0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("SSAO BG0"),
            layout:  &bgl_0,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ssao_camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: globals_buf.as_entire_binding() },
            ],
        });

        let bind_group_1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("SSAO BG1"),
            layout:  &bgl_1,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(gbuf_albedo) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(gbuf_normal) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(gbuf_orm) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(gbuf_emissive) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(gbuf_depth) },
            ],
        });

        let bind_group_2 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label:   Some("SSAO BG2"),
            layout:  &bgl_2,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ssao_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: sample_kernel_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&noise_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&noise_sampler) },
            ],
        });

        // ── Pipeline ───────────────────────────────────────────────────────────

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("SSAO PL"),
            bind_group_layouts:   &[&bgl_0, &bgl_1, &bgl_2],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("SSAO Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module:              &shader,
                entry_point:         Some("vs_main"),
                compilation_options: Default::default(),
                buffers:             &[],
            },
            fragment: Some(wgpu::FragmentState {
                module:              &shader,
                entry_point:         Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format:     wgpu::TextureFormat::R8Unorm,
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
            multiview:     None,
            cache:         None,
        });

        Self {
            pipeline,
            bgl_0,
            bgl_1,
            bgl_2,
            bind_group_0,
            bind_group_1,
            bind_group_2,
            ssao_camera_buf,
            globals_buf,
            ssao_uniform_buf,
            sample_kernel_buf,
            noise_texture,
            noise_sampler,
            ssao_texture,
            ssao_view,
        }
    }
}

impl RenderPass for SsaoPass {
    fn name(&self) -> &'static str {
        "SSAO"
    }

    fn publish<'a>(&'a self, frame: &mut libhelio::FrameResources<'a>) {
        frame.ssao = Some(&self.ssao_view);
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // TODO: Derive view, proj, inv_view_proj from scene camera for accurate SSAO.
        // Currently zeroed — GPU will return 1.0 (no occlusion) for sky pixels.
        let camera = SsaoCameraUniform::zeroed();
        ctx.write_buffer(&self.ssao_camera_buf, 0, bytemuck::bytes_of(&camera));

        let ssao = SsaoUniform {
            radius:      0.5,
            bias:        0.025,
            power:       2.0,
            samples:     KERNEL_SIZE as u32,
            noise_scale: [
                ctx.width  as f32 / NOISE_DIM as f32,
                ctx.height as f32 / NOISE_DIM as f32,
            ],
            _pad: [0.0; 2],
        };
        ctx.write_buffer(&self.ssao_uniform_buf, 0, bytemuck::bytes_of(&ssao));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // O(1): single fullscreen draw — GPU samples GBuffer and accumulates AO.
        let color_attachment = wgpu::RenderPassColorAttachment {
            view:           &self.ssao_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load:  wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        };
        let color_attachments = [Some(color_attachment)];
        let desc = wgpu::RenderPassDescriptor {
            label:                    Some("SSAO"),
            color_attachments:        &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes:         None,
            occlusion_query_set:      None,
        };

        let mut pass = ctx.encoder.begin_render_pass(&desc);
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group_0, &[]);
        pass.set_bind_group(1, &self.bind_group_1, &[]);
        pass.set_bind_group(2, &self.bind_group_2, &[]);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

impl SsaoPass {
    /// Recreates the SSAO output texture at a new resolution (call on window resize).
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (tex, view) = make_ssao_texture(device, width, height);
        self.ssao_texture = tex;
        self.ssao_view    = view;
    }
}

// ── Private helpers ────────────────────────────────────────────────────────────

fn make_ssao_texture(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("SSAO"),
        size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::R8Unorm,
        usage:           wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn gbuf_float_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type:    wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled:   false,
        },
        count: None,
    }
}

/// Deterministic 64-sample hemisphere kernel in tangent space (z ≥ 0).
fn generate_kernel() -> [[f32; 4]; KERNEL_SIZE] {
    let mut result = [[0f32; 4]; KERNEL_SIZE];
    let mut state: u32 = 1_234_567;
    let mut rng = move || -> f32 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 16) as f32 / 65_535.0
    };
    for i in 0..KERNEL_SIZE {
        let x = rng() * 2.0 - 1.0;
        let y = rng() * 2.0 - 1.0;
        let z = rng(); // [0, 1] → +z hemisphere in tangent space
        let len = (x * x + y * y + z * z).sqrt().max(1e-6);
        // Accelerating scale: more samples near origin for better near-field AO
        let t = (i as f32) / (KERNEL_SIZE as f32);
        let scale = 0.1 + 0.9 * t * t;
        result[i] = [x / len * scale, y / len * scale, z / len * scale, 0.0];
    }
    result
}

/// 4×4 Rgba8Unorm noise texture.
/// R/G store packed random XY rotation components; B is fixed at 128 (z=0).
fn generate_noise() -> [u8; (NOISE_DIM * NOISE_DIM * 4) as usize] {
    let mut data = [0u8; (NOISE_DIM * NOISE_DIM * 4) as usize];
    let mut state: u32 = 9_876_543;
    let mut rng = move || -> u8 {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (state >> 24) as u8
    };
    let mut i = 0;
    while i < data.len() {
        data[i]     = rng(); // R
        data[i + 1] = rng(); // G
        data[i + 2] = 128;   // B — z component = 0 in [-1,1], keeps rotations in XY
        data[i + 3] = 255;   // A (unused)
        i += 4;
    }
    data
}
