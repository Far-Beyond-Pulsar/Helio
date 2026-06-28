//! Corona — fully GPU-native particle system.
//!
//! Per-frame GPU pipeline:
//!   1. Simulate     — physics + aging
//!   2. Emit         — ring-buffer spawn (stores emitter_idx in particle.velocity.w)
//!   3. ScanLocal    — prefix scan per 256-block (Hillis-Steele) + sort-key reset
//!   4. ScanBlocks   — sequential cumulative sum per emitter; writes emitter_alive
//!   5. Scatter      — scatter alive indices into compact_buf + depth to sort_key_buf
//!   6. BuildMulti   — write one DrawArgs per emitter
//!   copy_buffer_to_buffer: draw_args_staging → draw_args_buf
//!   7+. Sort        — bitonic sort (descending) per emitter for back-to-front order
//!   8.  Render      — one draw_indirect per emitter; atlas sprite from emitter.texture_index

use bytemuck::{Pod, Zeroable};
use helio_v3::graph::ResourceBuilder;
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

// ── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_MAX_PARTICLES: u32 = libhelio::CORONA_MAX_PARTICLES;
const MAX_EMITTERS:          u32 = 64;
const WG:                    u32 = 256;
const ATLAS_SIZE:            u32 = 128;   // 128×128 atlas, 4×4 cells of 32×32 each
const ATLAS_CELLS:           u32 = 4;     // cells per row/column
const _CELL_SIZE:            u32 = ATLAS_SIZE / ATLAS_CELLS;  // 32

// ── CPU structs ──────────────────────────────────────────────────────────────

/// Matches GpuCoronaUniforms in corona.wgsl (8 × u32/f32 = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CoronaUniforms {
    delta_time:      f32,
    total_particles: u32,
    emitter_count:   u32,
    frame_count:     u32,
    // Written per sort-dispatch via copy_buffer_to_buffer from sort_steps_buf.
    sort_k:          u32,
    sort_j:          u32,
    sort_lo:         u32,
    sort_n:          u32,
}

/// One entry in sort_steps_buf (16 bytes). Pre-built at prepare() time.
/// Copied into uniforms[16..32] before each sort dispatch.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortStep {
    k:  u32,
    j:  u32,
    lo: u32,
    n:  u32,
}

// ── Pass ─────────────────────────────────────────────────────────────────────

pub struct CoronaPass {
    // ── Pipelines ────────────────────────────────────────────────────────────
    simulate_pipeline:        wgpu::ComputePipeline,
    emit_pipeline:            wgpu::ComputePipeline,
    scan_local_pipeline:      wgpu::ComputePipeline,
    scan_blocks_pipeline:     wgpu::ComputePipeline,
    scatter_pipeline:         wgpu::ComputePipeline,
    build_multi_pipeline:     wgpu::ComputePipeline,
    sort_local_pipeline:      wgpu::ComputePipeline,
    sort_global_pipeline:     wgpu::ComputePipeline,
    sort_local_merge_pipeline: wgpu::ComputePipeline,
    render_pipeline:          wgpu::RenderPipeline,

    // ── Single unified BGL / pipeline layout ─────────────────────────────────
    bgl: wgpu::BindGroupLayout,

    // ── GPU buffers ──────────────────────────────────────────────────────────
    uniform_buf:       wgpu::Buffer,   // CoronaUniforms (32 bytes, UNIFORM | COPY_DST)
    particle_buf:      wgpu::Buffer,
    emitter_buf:       wgpu::Buffer,
    compact_buf:       wgpu::Buffer,
    emitter_alive_buf: wgpu::Buffer,   // non-atomic u32 per emitter
    draw_args_staging: wgpu::Buffer,   // STORAGE | COPY_SRC
    draw_args_buf:     wgpu::Buffer,   // INDIRECT | COPY_DST
    prefix_buf:        wgpu::Buffer,   // u32 per particle slot
    block_sums_buf:    wgpu::Buffer,   // u32 per 256-block
    sort_key_buf:      wgpu::Buffer,   // f32 per particle slot
    // Pre-built sort steps; 16 bytes per step, STORAGE | COPY_SRC.
    // Entries are copied into uniform_buf[16..32] before each sort dispatch.
    sort_steps_buf:    wgpu::Buffer,

    // ── Particle texture (4×4 atlas, 128×128) ────────────────────────────────
    _particle_tex:    wgpu::Texture,
    particle_view:    wgpu::TextureView,
    particle_sampler: wgpu::Sampler,

    // ── Bind group (rebuilt when camera or particle buffer pointer changes) ───
    bg:     Option<wgpu::BindGroup>,
    bg_key: Option<(usize, usize)>,  // (particle_buf ptr, camera_buf ptr)

    // ── State ────────────────────────────────────────────────────────────────
    uploaded_generation: u64,
    max_particles:       u32,
    emitter_count:       u32,
    max_sort_steps:      u32,   // capacity of sort_steps_buf in step count

    // CPU emitter list: preserves spawn_cursor and non-overlapping offsets.
    cpu_emitters: Vec<libhelio::GpuCoronaEmitter>,
    // Pre-computed sort steps for the current emitter set.
    sort_steps: Vec<SortStep>,
    /// Enable per-emitter back-to-front depth sort before rendering.
    /// Costs ~50–200 compute dispatches per frame depending on emitter sizes.
    /// Leave false for additive effects (fire, sparks) — order-independent.
    /// Set true only for alpha-blended volumetric effects (smoke, clouds).
    pub depth_sort_enabled: bool,
}

impl CoronaPass {
    pub fn new(
        device:         &wgpu::Device,
        queue:          &wgpu::Queue,
        camera_buf:     &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("Corona Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/corona.wgsl").into()),
        });

        // ── Buffers ──────────────────────────────────────────────────────────

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Uniforms"),
            size:  std::mem::size_of::<CoronaUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let particle_size = std::mem::size_of::<libhelio::GpuCoronaParticle>() as u64;
        let particle_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Particles"),
            size:  DEFAULT_MAX_PARTICLES as u64 * particle_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let emitter_size = std::mem::size_of::<libhelio::GpuCoronaEmitter>() as u64;
        let emitter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Emitters"),
            size:  MAX_EMITTERS as u64 * emitter_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compact_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Compact"),
            size:  DEFAULT_MAX_PARTICLES as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let emitter_alive_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Emitter Alive"),
            size:  MAX_EMITTERS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let draw_args_size = MAX_EMITTERS as u64
            * std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        let draw_args_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona DrawArgs Staging"),
            size:  draw_args_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let draw_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona DrawArgs"),
            size:  draw_args_size,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prefix_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Prefix"),
            size:  DEFAULT_MAX_PARTICLES as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let max_blocks = DEFAULT_MAX_PARTICLES.div_ceil(WG);
        let block_sums_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona BlockSums"),
            size:  max_blocks as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_key_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona SortKeys"),
            size:  DEFAULT_MAX_PARTICLES as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Initial sort_steps_buf capacity: enough for 4 emitters of max size.
        // Grows on first prepare() call with real emitter data.
        let initial_sort_cap = 256u32;
        let sort_steps_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona SortSteps"),
            size:  initial_sort_cap as u64 * std::mem::size_of::<SortStep>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                 | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── 4×4 sprite atlas ─────────────────────────────────────────────────

        let tex_data = Self::make_atlas(ATLAS_SIZE, ATLAS_CELLS);
        let particle_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Corona Atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE, height: ATLAS_SIZE, depth_or_array_layers: 1,
            },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &particle_tex, mip_level: 0,
                origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All,
            },
            &tex_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row:  Some(ATLAS_SIZE * 4),
                rows_per_image: Some(ATLAS_SIZE),
            },
            wgpu::Extent3d { width: ATLAS_SIZE, height: ATLAS_SIZE, depth_or_array_layers: 1 },
        );
        let particle_view = particle_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let particle_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Corona Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        // ── Unified bind group layout (b0-b11) ────────────────────────────────

        use wgpu::ShaderStages as SS;
        let cv  = SS::COMPUTE | SS::VERTEX;
        let cvf = SS::COMPUTE | SS::VERTEX | SS::FRAGMENT;
        let c   = SS::COMPUTE;
        let f   = SS::FRAGMENT;

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Corona BGL"),
            entries: &[
                Self::uniform_entry(0,  cvf),                          // uniforms
                Self::storage_entry(1,  cv,  false),                   // particles
                Self::storage_entry(2,  cv,  false),                   // emitters (VERTEX for atlas)
                Self::storage_entry(3,  cv,  false),                   // compact_buf
                Self::storage_entry(4,  c,   false),                   // emitter_alive
                Self::storage_entry(5,  c,   false),                   // draw_args_staging
                Self::uniform_entry(6,  cv),                           // camera
                Self::storage_entry(7,  c,   false),                   // prefix_buf
                Self::storage_entry(8,  c,   false),                   // block_sums_buf
                Self::storage_entry(9,  c,   false),                   // sort_key_buf
                wgpu::BindGroupLayoutEntry {                            // particle_tex
                    binding: 10, visibility: f,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {                            // particle_sampler
                    binding: 11, visibility: f,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Corona PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        // ── Compute pipelines ────────────────────────────────────────────────

        let mk_compute = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label:  Some(&format!("Corona {entry}")),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let simulate_pipeline         = mk_compute("cs_simulate");
        let emit_pipeline             = mk_compute("cs_emit");
        let scan_local_pipeline       = mk_compute("cs_scan_local");
        let scan_blocks_pipeline      = mk_compute("cs_scan_blocks");
        let scatter_pipeline          = mk_compute("cs_scatter");
        let build_multi_pipeline      = mk_compute("cs_build_multi");
        let sort_local_pipeline       = mk_compute("cs_sort_local");
        let sort_global_pipeline      = mk_compute("cs_sort_global");
        let sort_local_merge_pipeline = mk_compute("cs_sort_local_merge");

        // ── Render pipeline ──────────────────────────────────────────────────

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label:  Some("Corona Render"),
            layout: Some(&pl),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: Some("vs_main"),
                compilation_options: Default::default(), buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::SrcAlpha,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent::OVER,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None, ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format:              wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(false),
                depth_compare:       Some(wgpu::CompareFunction::LessEqual),
                stencil: wgpu::StencilState::default(),
                bias:    wgpu::DepthBiasState::default(),
            }),
            multisample:    wgpu::MultisampleState::default(),
            multiview_mask: None, cache: None,
        });

        // ── Initial bind group ───────────────────────────────────────────────

        let camera_ptr = camera_buf as *const _ as usize;
        let part_ptr   = &particle_buf as *const _ as usize;

        let bg = Some(Self::build_bg(
            device, &bgl,
            &uniform_buf, &particle_buf, &emitter_buf, &compact_buf,
            &emitter_alive_buf, &draw_args_staging,
            camera_buf,
            &prefix_buf, &block_sums_buf, &sort_key_buf,
            &particle_view, &particle_sampler,
        ));

        Self {
            simulate_pipeline, emit_pipeline,
            scan_local_pipeline, scan_blocks_pipeline, scatter_pipeline, build_multi_pipeline,
            sort_local_pipeline, sort_global_pipeline, sort_local_merge_pipeline,
            render_pipeline,
            bgl,
            uniform_buf, particle_buf, emitter_buf, compact_buf,
            emitter_alive_buf, draw_args_staging, draw_args_buf,
            prefix_buf, block_sums_buf, sort_key_buf,
            sort_steps_buf,
            _particle_tex: particle_tex, particle_view, particle_sampler,
            bg, bg_key: Some((part_ptr, camera_ptr)),
            uploaded_generation: u64::MAX,
            max_particles: DEFAULT_MAX_PARTICLES,
            emitter_count: 0,
            max_sort_steps: initial_sort_cap,
            cpu_emitters: Vec::new(),
            sort_steps: Vec::new(),
            depth_sort_enabled: false,
        }
    }

    // ── BGL helpers ──────────────────────────────────────────────────────────

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

    // ── Bind group builder ────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn build_bg(
        device: &wgpu::Device, bgl: &wgpu::BindGroupLayout,
        uniforms: &wgpu::Buffer, particles: &wgpu::Buffer, emitters: &wgpu::Buffer,
        compact: &wgpu::Buffer, alive: &wgpu::Buffer, draw_staging: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        prefix: &wgpu::Buffer, block_sums: &wgpu::Buffer, sort_keys: &wgpu::Buffer,
        tex_view: &wgpu::TextureView, sampler: &wgpu::Sampler,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Corona BG"), layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding:  0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  1, resource: particles.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  2, resource: emitters.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  3, resource: compact.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  4, resource: alive.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  5, resource: draw_staging.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  6, resource: camera.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  7, resource: prefix.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  8, resource: block_sums.as_entire_binding() },
                wgpu::BindGroupEntry { binding:  9, resource: sort_keys.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(tex_view) },
                wgpu::BindGroupEntry { binding: 11, resource: wgpu::BindingResource::Sampler(sampler) },
            ],
        })
    }

    // ── Sort step pre-computation ─────────────────────────────────────────────

    /// Compute all bitonic sort dispatches for one emitter (particle_offset=lo, particle_count=n).
    /// Appends to `out`. Steps are: initial local sort, then for each k-stage:
    ///   global steps (j ≥ 256), then one local-merge step (j = 128..1).
    fn push_sort_steps(lo: u32, n: u32, out: &mut Vec<SortStep>) {
        if n == 0 { return; }

        // Initial block sort (k = 2..256, all in shared memory).
        // j == 0 signals cs_sort_local (not cs_sort_global).
        out.push(SortStep { k: 256, j: 0, lo, n });

        // Global stages for k = 512, 1024, ..., n.
        let mut k = 512u32;
        while k <= n {
            // Global steps: j = k/2 down to 256 (inclusive).
            let mut j = k >> 1;
            while j >= 256 {
                out.push(SortStep { k, j, lo, n });
                j >>= 1;
            }
            // Local merge step: j = 128..1 in shared memory.
            // j == 0xFFFF_FFFF signals cs_sort_local_merge.
            out.push(SortStep { k, j: u32::MAX, lo, n });
            k <<= 1;
        }
    }

    /// Rebuild sort_steps from the current emitter configuration.
    fn rebuild_sort_steps(
        device:     &wgpu::Device,
        queue:      &wgpu::Queue,
        cpu_emitters: &[libhelio::GpuCoronaEmitter],
        sort_steps_buf: &mut wgpu::Buffer,
        max_sort_steps: &mut u32,
        sort_steps: &mut Vec<SortStep>,
    ) {
        sort_steps.clear();
        for em in cpu_emitters {
            Self::push_sort_steps(em.particle_offset, em.particle_count, sort_steps);
        }

        let needed = sort_steps.len() as u32;
        if needed > *max_sort_steps {
            *max_sort_steps = needed.next_power_of_two().max(256);
            *sort_steps_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Corona SortSteps"),
                size:  *max_sort_steps as u64 * std::mem::size_of::<SortStep>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC
                     | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if !sort_steps.is_empty() {
            queue.write_buffer(sort_steps_buf, 0, bytemuck::cast_slice(sort_steps));
        }
    }

    // ── 4×4 Sprite atlas generation ───────────────────────────────────────────
    //
    // 16 procedural sprites arranged in a 4×4 grid:
    //   Row 0 (0-3):  Soft blobs — varying softness / core sharpness
    //   Row 1 (4-7):  Rings — varying thickness
    //   Row 2 (8-11): Stars — varying point count / spike sharpness
    //   Row 3 (12-15):Sparkles — elongated cross / streaks

    fn make_atlas(atlas_size: u32, cells: u32) -> Vec<u8> {
        let cell = atlas_size / cells;  // 32
        let mut data = vec![0u8; (atlas_size * atlas_size * 4) as usize];

        for sprite in 0..16u32 {
            let col = sprite % cells;
            let row = sprite / cells;
            let ox  = col * cell;
            let oy  = row * cell;

            for py in 0..cell {
                for px in 0..cell {
                    let half = cell as f32 * 0.5;
                    let dx = (px as f32 + 0.5 - half) / half;
                    let dy = (py as f32 + 0.5 - half) / half;
                    let r  = (dx * dx + dy * dy).sqrt();
                    let a = match row {
                        0 => {
                            // Soft blobs: vary from very soft to hard-edged
                            let sharpness = 1.0 + col as f32 * 2.0;
                            (1.0 - r.powf(sharpness)).clamp(0.0, 1.0)
                        }
                        1 => {
                            // Rings: vary inner radius
                            let inner = 0.3 + col as f32 * 0.12;
                            let outer = 0.85;
                            let ring  = 1.0 - ((r - (inner + outer) * 0.5).abs()
                                              / ((outer - inner) * 0.5)).clamp(0.0, 1.0);
                            ring * ring
                        }
                        2 => {
                            // Stars: 4-8 points
                            let points = 4.0 + col as f32 * 1.5;
                            let angle  = dy.atan2(dx);
                            let star_r = r / (0.5 + 0.5 * (angle * points).cos().abs());
                            (1.0 - star_r * 1.5).clamp(0.0, 1.0)
                        }
                        _ => {
                            // Sparkles: soft cross/streak with varying elongation
                            let elongation = 1.0 + col as f32 * 1.5;
                            let rx = dx / elongation;
                            let ry = dy;
                            let dr = (rx * rx + ry * ry).sqrt();
                            let cx = dx;
                            let cy = dy * elongation;
                            let dc = (cx * cx + cy * cy).sqrt();
                            let combined = (1.0 - dr).max(1.0 - dc).clamp(0.0, 1.0);
                            combined * combined
                        }
                    };
                    let alpha = (a.clamp(0.0, 1.0) * 255.0) as u8;
                    let base = ((oy + py) * atlas_size + ox + px) as usize * 4;
                    data[base]     = 255;
                    data[base + 1] = 255;
                    data[base + 2] = 255;
                    data[base + 3] = alpha;
                }
            }
        }
        data
    }
}

// ── RenderPass impl ──────────────────────────────────────────────────────────

impl RenderPass for CoronaPass {
    fn name(&self) -> &'static str { "Corona" }

    fn reads(&self) -> &'static [&'static str] {
        &["pre_aa", "full_res_depth", "corona_emitters", "depth", "main_scene"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
        builder.read("full_res_depth");
        builder.read("corona_emitters");
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if let Some(data) = ctx.frame_resources.corona_emitters.get() {
            if data.generation != self.uploaded_generation {
                let em_size = std::mem::size_of::<libhelio::GpuCoronaEmitter>();
                let count   = (data.count as usize).min(MAX_EMITTERS as usize);
                let src: &[libhelio::GpuCoronaEmitter] =
                    bytemuck::cast_slice(&data.emitters[..count * em_size]);

                self.cpu_emitters.resize(count, libhelio::GpuCoronaEmitter::zeroed());

                let mut upload = self.cpu_emitters[..count].to_vec();
                let dt = ctx.delta_time;
                let mut offset = 0u32;
                for (i, src_em) in src.iter().enumerate() {
                    let this_cursor = self.cpu_emitters[i].spawn_cursor;
                    let emit_count  = (src_em.emit_params[0] * dt) as u32;
                    // Align particle_count to WG so block boundaries never cross emitters.
                    let aligned_count = (src_em.particle_count + WG - 1) / WG * WG;
                    let range = aligned_count.max(1);

                    upload[i] = *src_em;
                    upload[i].particle_count  = aligned_count;
                    upload[i].particle_offset = offset;
                    upload[i].spawn_cursor    = this_cursor;

                    self.cpu_emitters[i].spawn_cursor = (this_cursor + emit_count) % range;

                    offset = offset.saturating_add(aligned_count);
                }

                let bytes: &[u8] = bytemuck::cast_slice(&upload);
                let cap = (MAX_EMITTERS as usize * em_size).min(bytes.len());
                ctx.write_buffer(&self.emitter_buf, 0, &bytes[..cap]);

                // Rebuild sort step sequence (also updates self.cpu_emitters offsets).
                for (i, src_em) in src.iter().enumerate() {
                    let aligned_count = (src_em.particle_count + WG - 1) / WG * WG;
                    self.cpu_emitters[i].particle_count  = aligned_count;
                    self.cpu_emitters[i].particle_offset = upload[i].particle_offset;
                }
                Self::rebuild_sort_steps(
                    ctx.device, ctx.queue,
                    &self.cpu_emitters[..count],
                    &mut self.sort_steps_buf,
                    &mut self.max_sort_steps,
                    &mut self.sort_steps,
                );

                self.emitter_count       = count as u32;
                self.uploaded_generation = data.generation;
                self.max_particles       = data.max_particles.clamp(1024, DEFAULT_MAX_PARTICLES);
            }
        } else {
            self.emitter_count = 0;
        }

        let uniforms = CoronaUniforms {
            delta_time:      ctx.delta_time,
            total_particles: self.max_particles,
            emitter_count:   self.emitter_count,
            frame_count:     ctx.frame_num as u32,
            sort_k: 0, sort_j: 0, sort_lo: 0, sort_n: 0,
        };
        ctx.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        Ok(())
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        target: &'a wgpu::TextureView,
        depth: &'a wgpu::TextureView,
        resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let target_view = resources.pre_aa.get().unwrap_or(target);
        let color_attachments: &'a [Option<wgpu::RenderPassColorAttachment<'a>>] = Box::leak(Box::new([
            Some(wgpu::RenderPassColorAttachment {
                view: target_view, resolve_target: None, depth_slice: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            }),
        ]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("Corona Render"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None, occlusion_query_set: None, multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.emitter_count == 0 { return Ok(()); }

        // ── Bind group rebuild when buffer pointers change ────────────────────

        let part_ptr   = &self.particle_buf as *const _ as usize;
        let camera_ptr = ctx.scene.camera   as *const _ as usize;
        let key = (part_ptr, camera_ptr);

        if self.bg_key != Some(key) {
            self.bg = Some(Self::build_bg(
                ctx.device, &self.bgl,
                &self.uniform_buf, &self.particle_buf, &self.emitter_buf,
                &self.compact_buf, &self.emitter_alive_buf, &self.draw_args_staging,
                ctx.scene.camera,
                &self.prefix_buf, &self.block_sums_buf, &self.sort_key_buf,
                &self.particle_view, &self.particle_sampler,
            ));
            self.bg_key = Some(key);
        }

        let bg = self.bg.as_ref().unwrap();
        let wg = self.max_particles.div_ceil(WG);
        let ec = self.emitter_count;

        // ── Pass 1: Simulate ─────────────────────────────────────────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Simulate"), timestamp_writes: None,
            });
            p.set_pipeline(&self.simulate_pipeline);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // ── Pass 2: Emit ─────────────────────────────────────────────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Emit"), timestamp_writes: None,
            });
            p.set_pipeline(&self.emit_pipeline);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // ── Pass 3: Scan local (prefix scan + sort-key sentinel reset) ────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona ScanLocal"), timestamp_writes: None,
            });
            p.set_pipeline(&self.scan_local_pipeline);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // ── Pass 4: Scan blocks (cumulative per-emitter offsets) ──────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona ScanBlocks"), timestamp_writes: None,
            });
            p.set_pipeline(&self.scan_blocks_pipeline);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // ── Pass 5: Scatter (compact_buf + sort_key_buf) ─────────────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona Scatter"), timestamp_writes: None,
            });
            p.set_pipeline(&self.scatter_pipeline);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // ── Pass 6: Build draw args ───────────────────────────────────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Corona BuildMulti"), timestamp_writes: None,
            });
            p.set_pipeline(&self.build_multi_pipeline);
            p.set_bind_group(0, bg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // Copy STORAGE staging → INDIRECT buffer (the STORAGE+INDIRECT conflict fix).
        let args_size = ec as u64
            * std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        unsafe { &mut *ctx.encoder_ptr }.copy_buffer_to_buffer(
            &self.draw_args_staging, 0, &self.draw_args_buf, 0, args_size,
        );

        // ── Passes 7+: Bitonic sort per emitter (opt-in) ─────────────────────
        // Each bitonic stage requires a separate compute pass. For emitters with
        // 262K particles this is ~66 dispatches each. Leave depth_sort_enabled=false
        // for additive effects where draw order doesn't matter.

        if !self.depth_sort_enabled || self.sort_steps.is_empty() {
            // Skip sort — proceed directly to render.
        } else {

        let step_size = std::mem::size_of::<SortStep>() as u64;

        for (step_idx, step) in self.sort_steps.iter().enumerate() {
            // Copy {k, j, lo, n} from sort_steps_buf into the sort_* fields of
            // uniform_buf (offset 16 = after the first 4 u32 base fields).
            unsafe { &mut *ctx.encoder_ptr }.copy_buffer_to_buffer(
                &self.sort_steps_buf,
                step_idx as u64 * step_size,
                &self.uniform_buf,
                16,  // byte offset of sort_k in CoronaUniforms
                step_size,
            );

            let particle_count = step.n;
            let blocks = particle_count.div_ceil(WG);

            if step.j == 0 {
                // cs_sort_local: initial block sort (k=2..256 in shared memory).
                let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Corona SortLocal"), timestamp_writes: None,
                });
                p.set_pipeline(&self.sort_local_pipeline);
                p.set_bind_group(0, bg, &[]);
                p.dispatch_workgroups(blocks, 1, 1);
            } else if step.j == u32::MAX {
                // cs_sort_local_merge: tail steps (j=128..1) for a global k-stage.
                let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Corona SortLocalMerge"), timestamp_writes: None,
                });
                p.set_pipeline(&self.sort_local_merge_pipeline);
                p.set_bind_group(0, bg, &[]);
                p.dispatch_workgroups(blocks, 1, 1);
            } else {
                // cs_sort_global: one compare-swap step for j >= 256.
                let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Corona SortGlobal"), timestamp_writes: None,
                });
                p.set_pipeline(&self.sort_global_pipeline);
                p.set_bind_group(0, bg, &[]);
                p.dispatch_workgroups(blocks, 1, 1);
            }
        }

        } // end depth_sort_enabled else branch

        // ── Render pass ──────────────────────────────────────────────────────

        let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        rp.set_pipeline(&self.render_pipeline);
        rp.set_bind_group(0, bg, &[]);

        // One draw_indirect per emitter — each draws only its alive, sorted particles.
        let stride = std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        for i in 0..ec {
            rp.draw_indirect(&self.draw_args_buf, i as u64 * stride);
        }

        Ok(())
    }
}
