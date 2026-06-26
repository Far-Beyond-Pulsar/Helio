//! Corona — fully GPU-native particle system.
//!
//! Designed as a Helio render pass with four compute stages:
//!   1. Simulate  — advance age, apply physics, kill expired particles
//!   2. Emit      — spawn new particles (ring-buffer per emitter)
//!   3. Count     — atomic count of living particles
//!   4. Build     — write alive count into indirect draw args
//! Then a single instanced render pass draws camera-facing billboards.

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

// ── Constants ────────────────────────────────────────────────────────────────

/// Default total particles (= 512K; 64 MiB VRAM for particles alone).
const DEFAULT_MAX_PARTICLES: u32 = 524_288;

/// Maximum emitters we can have in the GPU buffer.
const MAX_EMITTERS: u32 = 64;

/// Workgroup size used in most compute shaders.
const WG: u32 = 256;

// ── CPU-side uniform struct ──────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CoronaUniforms {
    delta_time: f32,
    total_particles: u32,
    emitter_count: u32,
    frame_count: u32,
}

// ── Pass ─────────────────────────────────────────────────────────────────────

pub struct CoronaPass {
    // Pipelines
    simulate_pipeline: wgpu::ComputePipeline,
    emit_pipeline: wgpu::ComputePipeline,
    reset_pipeline: wgpu::ComputePipeline,
    count_pipeline: wgpu::ComputePipeline,
    build_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,

    // Bind group layout & pipeline layouts
    compute_bgl: wgpu::BindGroupLayout,
    compute_pl: wgpu::PipelineLayout,
    render_bgl: wgpu::BindGroupLayout,
    render_pl: wgpu::PipelineLayout,

    // GPU buffers
    uniform_buf: wgpu::Buffer,
    particle_buf: wgpu::Buffer,
    emitter_buf: wgpu::Buffer,
    live_counter_buf: wgpu::Buffer,
    /// Written by cs_build_indirect (STORAGE). Bound at b4 in both BGLs.
    indirect_staging_buf: wgpu::Buffer,
    /// Used only by draw_indirect (INDIRECT). Never bound in any bind group.
    indirect_draw_buf: wgpu::Buffer,

    // Bind groups (rebuilt when emitter/camera pointers change)
    compute_bg: Option<wgpu::BindGroup>,
    compute_bg_key: Option<(usize, usize)>,
    render_bg: Option<wgpu::BindGroup>,
    render_bg_key: Option<(usize, usize)>,

    // Camera buffer pointer (track changes for BG rebuild)
    camera_buf_ptr: Option<usize>,

    // Emitter generation tracking (skip re-upload when unchanged)
    uploaded_generation: u64,

    // Particle texture (soft circle procedural)
    particle_tex: wgpu::Texture,
    particle_view: wgpu::TextureView,
    particle_sampler: wgpu::Sampler,

    // Cached counts
    max_particles: u32,
    emitter_count: u32,
    surface_format: wgpu::TextureFormat,
}

impl CoronaPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Corona Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/corona.wgsl").into()),
        });

        // ── Buffers ──────────────────────────────────────────────────────────
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Uniforms"),
            size: std::mem::size_of::<CoronaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_size = std::mem::size_of::<libhelio::GpuCoronaParticle>() as u64;
        let particle_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Particles"),
            size: DEFAULT_MAX_PARTICLES as u64 * particle_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let emitter_size = std::mem::size_of::<libhelio::GpuCoronaEmitter>() as u64;
        let emitter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Emitters"),
            size: MAX_EMITTERS as u64 * emitter_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let live_counter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Live Counter"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: true,
        });
        live_counter_buf
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(&0u32.to_ne_bytes());
        live_counter_buf.unmap();

        let indirect_size = std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        // Compute writes here; also bound as STORAGE at b4 in both BGLs.
        let indirect_staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Indirect Staging"),
            size: indirect_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Receives a copy from staging each frame; only used by draw_indirect.
        let indirect_draw_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Indirect Draw"),
            size: indirect_size,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Particle texture (soft circle) ───────────────────────────────────
        let tex_size = 32u32;
        let tex_data = Self::make_soft_circle_tex(tex_size);
        let particle_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Corona Particle Tex"),
            size: wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &particle_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &tex_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(tex_size * 4),
                rows_per_image: Some(tex_size),
            },
            wgpu::Extent3d { width: tex_size, height: tex_size, depth_or_array_layers: 1 },
        );
        let particle_view = particle_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let particle_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Corona Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // ── Compute bind group layout ────────────────────────────────────────
        let compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Corona Compute BGL"),
            entries: &[
                // b0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b1: particles storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b2: emitters storage
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // b3: live_counter atomic
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b4: indirect_buf storage
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Corona Compute PL"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });

        // ── Compute pipelines ────────────────────────────────────────────────
        let simulate_pipeline = Self::make_compute_pipeline(device, &shader, &compute_pl, "cs_simulate");
        let emit_pipeline = Self::make_compute_pipeline(device, &shader, &compute_pl, "cs_emit");
        let reset_pipeline = Self::make_compute_pipeline(device, &shader, &compute_pl, "cs_reset_counter");
        let count_pipeline = Self::make_compute_pipeline(device, &shader, &compute_pl, "cs_count_alive");
        let build_pipeline = Self::make_compute_pipeline(device, &shader, &compute_pl, "cs_build_indirect");

        // ── Render bind group layout ────────────────────────────────────────
        // Mirrors the compute BGL so all shader globals are satisfied, but b4
        // binds indirect_staging_buf (STORAGE only — no INDIRECT flag).
        // draw_indirect uses indirect_draw_buf which is INDIRECT only and is
        // never in any bind group, so there is no exclusive-usage conflict.
        let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Corona Render BGL"),
            entries: &[
                // b0: uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b1: particles storage (vertex reads instance data)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b2: emitters
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b3: live counter
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b4: indirect staging (STORAGE only — no INDIRECT usage conflict)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b5: camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // b6: particle texture
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // b7: particle sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Corona Render PL"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        // ── Render pipeline ──────────────────────────────────────────────────
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Corona Render Pipeline"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[], // no vertex buffers; reads from storage
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let camera_ptr = camera_buf as *const _ as usize;
        let p_ptr = &particle_buf as *const _ as usize;
        let e_ptr = &emitter_buf as *const _ as usize;
        let compute_bg = Some(Self::build_compute_bg(device, &compute_bgl, &uniform_buf, &particle_buf, &emitter_buf, &live_counter_buf, &indirect_staging_buf));
        let render_bg = Some(Self::build_render_bg(device, &render_bgl, &uniform_buf, &particle_buf, &emitter_buf, &live_counter_buf, &indirect_staging_buf, camera_buf, &particle_view, &particle_sampler));

        Self {
            simulate_pipeline,
            emit_pipeline,
            reset_pipeline,
            count_pipeline,
            build_pipeline,
            render_pipeline,
            compute_bgl,
            compute_pl,
            render_bgl,
            render_pl,
            uniform_buf,
            particle_buf,
            emitter_buf,
            live_counter_buf,
            indirect_staging_buf,
            indirect_draw_buf,
            compute_bg,
            compute_bg_key: Some((p_ptr, e_ptr)),
            render_bg,
            render_bg_key: Some((camera_ptr, p_ptr)),
            camera_buf_ptr: Some(camera_ptr),
            uploaded_generation: u64::MAX,
            particle_tex,
            particle_view,
            particle_sampler,
            max_particles: DEFAULT_MAX_PARTICLES,
            emitter_count: 0,
            surface_format,
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn make_compute_pipeline(
        device: &wgpu::Device,
        module: &wgpu::ShaderModule,
        layout: &wgpu::PipelineLayout,
        entry: &str,
    ) -> wgpu::ComputePipeline {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(&format!("Corona {}", entry)),
            layout: Some(layout),
            module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    }

    fn build_compute_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        particles: &wgpu::Buffer,
        emitters: &wgpu::Buffer,
        counter: &wgpu::Buffer,
        indirect: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Corona Compute BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: emitters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counter.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: indirect.as_entire_binding() },
            ],
        })
    }

    fn build_render_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniform: &wgpu::Buffer,
        particles: &wgpu::Buffer,
        emitters: &wgpu::Buffer,
        counter: &wgpu::Buffer,
        indirect_staging: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        tex_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Corona Render BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: particles.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: emitters.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: counter.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: indirect_staging.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(tex_view) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    fn make_soft_circle_tex(size: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((size * size * 4) as usize);
        let half = size as f32 * 0.5;
        for y in 0..size {
            for x in 0..size {
                let dx = (x as f32 + 0.5 - half) / half;
                let dy = (y as f32 + 0.5 - half) / half;
                let dist = (dx * dx + dy * dy).sqrt();
                let alpha = (1.0 - dist).clamp(0.0, 1.0);
                let smooth = alpha * alpha * (3.0 - 2.0 * alpha); // smoothstep
                let a = (smooth * 255.0) as u8;
                data.push(255); // R
                data.push(255); // G
                data.push(255); // B
                data.push(a);   // A
            }
        }
        data
    }

    /// Set camera buffer pointer so we can rebuild bind groups when it changes.
    pub fn set_camera_buffer(&mut self, camera_buf: &wgpu::Buffer) {
        self.camera_buf_ptr = Some(camera_buf as *const _ as usize);
        self.render_bg_key = None; // force rebuild
    }
}

impl RenderPass for CoronaPass {
    fn name(&self) -> &'static str {
        "Corona"
    }

    fn reads(&self) -> &'static [helio_v3::graph::ResourceSlot] {
        &[
            helio_v3::graph::ResourceSlot::PreAa,
            helio_v3::graph::ResourceSlot::FullResDepth,
            helio_v3::graph::ResourceSlot::CoronaEmitters,
        ]
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // ── Upload emitter definitions if changed ────────────────────────────
        if let Some(data) = ctx.frame_resources.corona_emitters.get() {
            if data.generation != self.uploaded_generation {
                let max_bytes = (MAX_EMITTERS as usize * std::mem::size_of::<libhelio::GpuCoronaEmitter>())
                    .min(data.emitters.len());
                if max_bytes > 0 {
                    ctx.write_buffer(&self.emitter_buf, 0, &data.emitters[..max_bytes]);
                }
                self.emitter_count = data.count.min(MAX_EMITTERS);
                self.uploaded_generation = data.generation;
                self.max_particles = data.max_particles.clamp(1024, 4_194_304);
            }
        } else {
            self.emitter_count = 0;
        }

        // ── Upload uniforms ──────────────────────────────────────────────────
        let uniforms = CoronaUniforms {
            delta_time: ctx.delta_time,
            total_particles: self.max_particles,
            emitter_count: self.emitter_count,
            frame_count: ctx.frame_num as u32,
        };
        ctx.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.emitter_count == 0 {
            return Ok(());
        }

        // ── Check for bind group rebuild ────────────────────────────────────
        let particle_ptr = &self.particle_buf as *const _ as usize;
        let emitter_ptr = &self.emitter_buf as *const _ as usize;
        let camera_ptr = ctx.scene.camera as *const _ as usize;

        let comp_key = (particle_ptr, emitter_ptr);
        if self.compute_bg_key != Some(comp_key) {
            self.compute_bg = Some(Self::build_compute_bg(
                ctx.device, &self.compute_bgl,
                &self.uniform_buf, &self.particle_buf, &self.emitter_buf,
                &self.live_counter_buf, &self.indirect_staging_buf,
            ));
            self.compute_bg_key = Some(comp_key);
        }

        let render_key = (camera_ptr, particle_ptr);
        if self.render_bg_key != Some(render_key) {
            self.render_bg = Some(Self::build_render_bg(
                ctx.device, &self.render_bgl,
                &self.uniform_buf, &self.particle_buf, &self.emitter_buf,
                &self.live_counter_buf, &self.indirect_staging_buf,
                &ctx.scene.camera, &self.particle_view, &self.particle_sampler,
            ));
            self.render_bg_key = Some(render_key);
        }

        // ── Compute pass 1: Simulate ─────────────────────────────────────────
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Simulate"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.simulate_pipeline);
            pass.set_bind_group(0, self.compute_bg.as_ref().unwrap(), &[]);
            let wg_count = (self.max_particles + WG - 1) / WG;
            pass.dispatch_workgroups(wg_count, 1, 1);
        }

        // ── Compute pass 2: Emit ─────────────────────────────────────────────
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Emit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.emit_pipeline);
            pass.set_bind_group(0, self.compute_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(self.emitter_count.max(1), 1, 1);
        }

        // ── Compute pass 3a: Reset live counter ──────────────────────────────
        // Must be a separate dispatch so the reset is globally visible before
        // any counting threads run (GPU dispatches within an encoder are ordered).
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona ResetCounter"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.reset_pipeline);
            pass.set_bind_group(0, self.compute_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // ── Compute pass 3b: Count alive ─────────────────────────────────────
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Count"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.count_pipeline);
            pass.set_bind_group(0, self.compute_bg.as_ref().unwrap(), &[]);
            let wg_count = (self.max_particles + WG - 1) / WG;
            pass.dispatch_workgroups(wg_count, 1, 1);
        }

        // ── Compute pass 4: Build indirect ───────────────────────────────────
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona BuildIndirect"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.build_pipeline);
            pass.set_bind_group(0, self.compute_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Copy the indirect args from the staging buffer (STORAGE) to the draw
        // buffer (INDIRECT). This separates the two usages so wgpu never sees
        // the same buffer as both STORAGE and INDIRECT in the same scope.
        let indirect_size = std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        ctx.encoder.copy_buffer_to_buffer(
            &self.indirect_staging_buf, 0,
            &self.indirect_draw_buf, 0,
            indirect_size,
        );

        // ── Render pass: Draw particles ──────────────────────────────────────
        let target_view = ctx.resources.pre_aa.get().unwrap_or(ctx.target);

        let color_attachment = wgpu::RenderPassColorAttachment {
            view: target_view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            },
        };
        let depth_attachment = wgpu::RenderPassDepthStencilAttachment {
            view: ctx.depth,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Load,
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        };

        let desc = wgpu::RenderPassDescriptor {
            label: Some("Corona Render"),
            color_attachments: &[Some(color_attachment)],
            depth_stencil_attachment: Some(depth_attachment),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        };

        let mut pass = ctx.encoder.begin_render_pass(&desc);
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, self.render_bg.as_ref().unwrap(), &[]);
        pass.draw_indirect(&self.indirect_draw_buf, 0);

        Ok(())
    }
}
