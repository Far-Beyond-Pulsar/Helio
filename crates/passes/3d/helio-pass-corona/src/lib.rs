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
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use pulsar_scenedb::gpu::BufferKey;

pub mod components;
pub use components::CoronaEmitterComponent;

// ── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_MAX_PARTICLES: u32 = libhelio::CORONA_MAX_PARTICLES;
// Redesigned (was 64 slots + CPU-side per-frame compaction over live
// `particle_count`/`emit_rate` values read from a renderer-owned Vec):
// every emitter now gets a fixed, non-overlapping particle range chosen by
// its row index — `SLOT_SIZE` each, `MAX_EMITTERS` slots, statically sized so
// `MAX_EMITTERS * SLOT_SIZE == CORONA_MAX_PARTICLES`. This is what makes the
// pass able to read `CoronaEmitterComponent`'s SceneDB buffer directly, with
// zero CPU touch per frame (see `prepare()`): `particle_offset`/
// `particle_count` never need recomputing from the current live emitter set,
// so there is nothing left for the CPU to read back. The trade-off is a
// lower emitter ceiling (4 instead of 64) unless `CORONA_MAX_PARTICLES` is
// raised to match a higher `MAX_EMITTERS` for a future demo that needs more
// simultaneous emitters.
const MAX_EMITTERS: u32 = DEFAULT_MAX_PARTICLES / libhelio::CORONA_MAX_PARTICLES_PER_EMITTER;
const SLOT_SIZE: u32 = libhelio::CORONA_MAX_PARTICLES_PER_EMITTER;
// corona.wgsl hardcodes this as a `const` (WGSL can't `include!` a Rust
// constant) -- this assertion fails the build loudly if the two ever drift,
// instead of silently mis-sizing every emitter's particle range.
const _: () = assert!(SLOT_SIZE == 262144);
const WG: u32 = 256;
const ATLAS_SIZE: u32 = 128; // 128×128 atlas, 4×4 cells of 32×32 each
const ATLAS_CELLS: u32 = 4; // cells per row/column
const _CELL_SIZE: u32 = ATLAS_SIZE / ATLAS_CELLS; // 32

// ── CPU structs ──────────────────────────────────────────────────────────────

/// Matches GpuCoronaUniforms in corona.wgsl (8 × u32/f32 = 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CoronaUniforms {
    delta_time: f32,
    total_particles: u32,
    emitter_count: u32,
    frame_count: u32,
    // Written per sort-dispatch via copy_buffer_to_buffer from sort_steps_buf.
    sort_k: u32,
    sort_j: u32,
    sort_lo: u32,
    sort_n: u32,
}

/// One entry in sort_steps_buf (16 bytes). Pre-built at prepare() time.
/// Copied into uniforms[16..32] before each sort dispatch.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortStep {
    k: u32,
    j: u32,
    lo: u32,
    n: u32,
}

// ── Pass ─────────────────────────────────────────────────────────────────────

pub struct CoronaPass {
    // ── Pipelines ────────────────────────────────────────────────────────────
    simulate_pipeline: wgpu::ComputePipeline,
    emit_pipeline: wgpu::ComputePipeline,
    scan_local_pipeline: wgpu::ComputePipeline,
    scan_blocks_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    build_multi_pipeline: wgpu::ComputePipeline,
    sort_local_pipeline: wgpu::ComputePipeline,
    sort_global_pipeline: wgpu::ComputePipeline,
    sort_local_merge_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,

    // Compute writes particle storage, while the vertex stage may only read it
    // on WebGPU. Keep compatible layouts rather than exposing writable storage
    // to the vertex stage.
    compute_bgl: wgpu::BindGroupLayout,
    render_bgl: wgpu::BindGroupLayout,

    // ── GPU buffers ──────────────────────────────────────────────────────────
    uniform_buf: wgpu::Buffer, // CoronaUniforms (32 bytes, UNIFORM | COPY_DST)
    particle_buf: wgpu::Buffer,
    /// Fallback buffer, bound only until `"corona_emitters"` exists (before
    /// any `CoronaEmitterComponent` has ever been inserted) — never written
    /// to otherwise; SceneDB's own buffer is bound directly once it exists
    /// (see `execute()`), read_write, so the compute shader's own per-frame
    /// `spawn_cursor` advance persists in place across frames with no CPU
    /// involvement at all.
    emitter_buf: wgpu::Buffer,
    /// Per-emitter-slot spawn cursor — purely transient, pass-owned GPU
    /// state, deliberately separate from the SceneDB-authored
    /// `CoronaEmitterComponent` row. See its creation site in `new()` for why.
    spawn_cursor_buf: wgpu::Buffer,
    compact_buf: wgpu::Buffer,
    emitter_alive_buf: wgpu::Buffer, // non-atomic u32 per emitter
    draw_args_staging: wgpu::Buffer, // STORAGE | COPY_SRC
    draw_args_buf: wgpu::Buffer,     // INDIRECT | COPY_DST
    prefix_buf: wgpu::Buffer,        // u32 per particle slot
    block_sums_buf: wgpu::Buffer,    // u32 per 256-block
    sort_key_buf: wgpu::Buffer,      // f32 per particle slot
    // Pre-built sort steps; 16 bytes per step, STORAGE | COPY_SRC.
    // Entries are copied into uniform_buf[16..32] before each sort dispatch.
    sort_steps_buf: wgpu::Buffer,

    // ── Particle texture (4×4 atlas, 128×128) ────────────────────────────────
    _particle_tex: wgpu::Texture,
    particle_view: wgpu::TextureView,
    particle_sampler: wgpu::Sampler,

    // ── Bind groups (rebuilt when camera or particle buffer pointer changes) ─
    compute_bg: Option<wgpu::BindGroup>,
    render_bg: Option<wgpu::BindGroup>,
    bg_key: Option<(usize, usize)>, // (particle_buf ptr, camera_buf ptr)

    // ── State ────────────────────────────────────────────────────────────────
    max_particles: u32,
    emitter_count: u32,
    max_sort_steps: u32, // capacity of sort_steps_buf in step count

    // Pre-computed once, at construction, for the fixed `MAX_EMITTERS` ×
    // `SLOT_SIZE` slot layout — never recomputed per frame (see `MAX_EMITTERS`'s
    // doc for why the layout no longer depends on which emitters are live).
    sort_steps: Vec<SortStep>,
    /// Enable per-emitter back-to-front depth sort before rendering.
    /// Costs ~50–200 compute dispatches per frame depending on emitter sizes.
    /// Leave false for additive effects (fire, sparks) — order-independent.
    /// Set true only for alpha-blended volumetric effects (smoke, clouds).
    pub depth_sort_enabled: bool,
}

impl CoronaPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera_buf: &wgpu::Buffer,
        surface_format: wgpu::TextureFormat,
    ) -> Self {
        let source = include_str!("../shaders/corona.wgsl");
        let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Corona Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Corona Render Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/corona_render.wgsl").into()),
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
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let emitter_size = std::mem::size_of::<libhelio::GpuCoronaEmitter>() as u64;
        let emitter_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Emitters"),
            size: MAX_EMITTERS as u64 * emitter_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Per-emitter-slot spawn cursor: purely transient, pass-owned GPU
        // state (`cs_emit` reads and advances it in place every frame), kept
        // OUT of `CoronaEmitterComponent` deliberately -- that struct is
        // SceneDB-authored (the frontend rewrites its `transform`/color/etc.
        // fields via `World::get_mut` whenever an emitter moves), and a
        // `#[gpu(layout = packed)]` write re-uploads the WHOLE row. If
        // spawn_cursor lived in that same row, every such authored update
        // would stomp the GPU's own advanced cursor back to a stale
        // CPU-shadowed value, restarting the emission ring each time.
        // Zero-initialized once; nothing but `cs_emit` ever touches it again.
        let spawn_cursor_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Spawn Cursors"),
            size: MAX_EMITTERS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &spawn_cursor_buf,
            0,
            bytemuck::cast_slice(&vec![0u32; MAX_EMITTERS as usize]),
        );

        let compact_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Compact"),
            size: DEFAULT_MAX_PARTICLES as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let emitter_alive_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Emitter Alive"),
            size: MAX_EMITTERS as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let draw_args_size =
            MAX_EMITTERS as u64 * std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        let draw_args_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona DrawArgs Staging"),
            size: draw_args_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let draw_args_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona DrawArgs"),
            size: draw_args_size,
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let prefix_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona Prefix"),
            size: DEFAULT_MAX_PARTICLES as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let max_blocks = DEFAULT_MAX_PARTICLES.div_ceil(WG);
        let block_sums_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona BlockSums"),
            size: max_blocks as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let sort_key_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona SortKeys"),
            size: DEFAULT_MAX_PARTICLES as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Initial sort_steps_buf capacity: enough for 4 emitters of max size.
        // Grows on first prepare() call with real emitter data.
        let initial_sort_cap = 256u32;
        let sort_steps_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Corona SortSteps"),
            size: initial_sort_cap as u64 * std::mem::size_of::<SortStep>() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── 4×4 sprite atlas ─────────────────────────────────────────────────

        let tex_data = Self::make_atlas(ATLAS_SIZE, ATLAS_CELLS);
        let particle_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Corona Atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
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
                bytes_per_row: Some(ATLAS_SIZE * 4),
                rows_per_image: Some(ATLAS_SIZE),
            },
            wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
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

        let compute_bgl = Self::create_bgl(device, false);
        let render_bgl = Self::create_bgl(device, true);

        let compute_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Corona Compute PL"),
            bind_group_layouts: &[Some(&compute_bgl)],
            immediate_size: 0,
        });
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Corona Render PL"),
            bind_group_layouts: &[Some(&render_bgl)],
            immediate_size: 0,
        });

        // ── Compute pipelines ────────────────────────────────────────────────

        let mk_compute = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("Corona {entry}")),
                layout: Some(&compute_pl),
                module: &compute_shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let simulate_pipeline = mk_compute("cs_simulate");
        let emit_pipeline = mk_compute("cs_emit");
        let scan_local_pipeline = mk_compute("cs_scan_local");
        let scan_blocks_pipeline = mk_compute("cs_scan_blocks");
        let scatter_pipeline = mk_compute("cs_scatter");
        let build_multi_pipeline = mk_compute("cs_build_multi");
        let sort_local_pipeline = mk_compute("cs_sort_local");
        let sort_global_pipeline = mk_compute("cs_sort_global");
        let sort_local_merge_pipeline = mk_compute("cs_sort_local_merge");

        // ── Render pipeline ──────────────────────────────────────────────────

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Corona Render"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
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

        // ── Initial bind group ───────────────────────────────────────────────

        let camera_ptr = camera_buf as *const _ as usize;
        let part_ptr = &particle_buf as *const _ as usize;

        let compute_bg = Some(Self::build_bg(
            device,
            &compute_bgl,
            &uniform_buf,
            &particle_buf,
            &emitter_buf,
            &compact_buf,
            &emitter_alive_buf,
            &draw_args_staging,
            camera_buf,
            &prefix_buf,
            &block_sums_buf,
            &sort_key_buf,
            &particle_view,
            &particle_sampler,
            &spawn_cursor_buf,
        ));
        let render_bg = Some(Self::build_bg(
            device,
            &render_bgl,
            &uniform_buf,
            &particle_buf,
            &emitter_buf,
            &compact_buf,
            &emitter_alive_buf,
            &draw_args_staging,
            camera_buf,
            &prefix_buf,
            &block_sums_buf,
            &sort_key_buf,
            &particle_view,
            &particle_sampler,
            &spawn_cursor_buf,
        ));

        // Fixed slot layout, computed once — never revisited per frame (see
        // `MAX_EMITTERS`'s doc). Every slot gets the same treatment
        // regardless of whether an emitter currently occupies it.
        let mut sort_steps = Vec::new();
        for slot in 0..MAX_EMITTERS {
            Self::push_sort_steps(slot * SLOT_SIZE, SLOT_SIZE, &mut sort_steps);
        }
        let mut sort_steps_buf = sort_steps_buf;
        let mut max_sort_steps = initial_sort_cap;
        Self::upload_sort_steps(device, queue, &sort_steps, &mut sort_steps_buf, &mut max_sort_steps);

        Self {
            simulate_pipeline,
            emit_pipeline,
            scan_local_pipeline,
            scan_blocks_pipeline,
            scatter_pipeline,
            build_multi_pipeline,
            sort_local_pipeline,
            sort_global_pipeline,
            sort_local_merge_pipeline,
            render_pipeline,
            compute_bgl,
            render_bgl,
            uniform_buf,
            particle_buf,
            emitter_buf,
            spawn_cursor_buf,
            compact_buf,
            emitter_alive_buf,
            draw_args_staging,
            draw_args_buf,
            prefix_buf,
            block_sums_buf,
            sort_key_buf,
            sort_steps_buf,
            _particle_tex: particle_tex,
            particle_view,
            particle_sampler,
            compute_bg,
            render_bg,
            bg_key: Some((part_ptr, camera_ptr)),
            max_particles: DEFAULT_MAX_PARTICLES,
            emitter_count: 0,
            max_sort_steps,
            sort_steps: Vec::new(),
            depth_sort_enabled: false,
        }
    }

    // ── BGL helpers ──────────────────────────────────────────────────────────

    fn uniform_entry(binding: u32, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn storage_entry(
        binding: u32,
        vis: wgpu::ShaderStages,
        ro: bool,
    ) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }

    fn create_bgl(device: &wgpu::Device, for_render: bool) -> wgpu::BindGroupLayout {
        use wgpu::ShaderStages as SS;

        let storage_visibility = if for_render { SS::VERTEX } else { SS::COMPUTE };
        let uniform_visibility = if for_render {
            SS::VERTEX | SS::FRAGMENT
        } else {
            SS::COMPUTE
        };
        let label = if for_render {
            "Corona Render BGL"
        } else {
            "Corona Compute BGL"
        };

        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &[
                Self::uniform_entry(0, uniform_visibility),
                Self::storage_entry(1, storage_visibility, for_render),
                Self::storage_entry(2, storage_visibility, for_render),
                Self::storage_entry(3, storage_visibility, for_render),
                Self::storage_entry(4, SS::COMPUTE, false),
                Self::storage_entry(5, SS::COMPUTE, false),
                Self::storage_entry(6, uniform_visibility, true),
                Self::storage_entry(7, SS::COMPUTE, false),
                Self::storage_entry(8, SS::COMPUTE, false),
                Self::storage_entry(9, SS::COMPUTE, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: SS::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: SS::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // Per-emitter-slot spawn cursor: pass-owned, purely transient
                // GPU state (see `spawn_cursor_buf`'s doc), never mirrored by
                // SceneDB and never read by the render bind group -- present
                // here too only so `build_bg` has one uniform entry list for
                // both layouts.
                Self::storage_entry(12, SS::COMPUTE, false),
            ],
        })
    }

    // ── Bind group builder ────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn build_bg(
        device: &wgpu::Device,
        bgl: &wgpu::BindGroupLayout,
        uniforms: &wgpu::Buffer,
        particles: &wgpu::Buffer,
        emitters: &wgpu::Buffer,
        compact: &wgpu::Buffer,
        alive: &wgpu::Buffer,
        draw_staging: &wgpu::Buffer,
        camera: &wgpu::Buffer,
        prefix: &wgpu::Buffer,
        block_sums: &wgpu::Buffer,
        sort_keys: &wgpu::Buffer,
        tex_view: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        spawn_cursors: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Corona BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: emitters.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: compact.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: alive.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: draw_staging.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: camera.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: prefix.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: block_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: sort_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::TextureView(tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: spawn_cursors.as_entire_binding(),
                },
            ],
        })
    }

    // ── Sort step pre-computation ─────────────────────────────────────────────

    /// Compute all bitonic sort dispatches for one emitter (particle_offset=lo, particle_count=n).
    /// Appends to `out`. Steps are: initial local sort, then for each k-stage:
    ///   global steps (j ≥ 256), then one local-merge step (j = 128..1).
    fn push_sort_steps(lo: u32, n: u32, out: &mut Vec<SortStep>) {
        if n == 0 {
            return;
        }

        // Initial block sort (k = 2..256, all in shared memory).
        // j == 0 signals cs_sort_local (not cs_sort_global).
        out.push(SortStep {
            k: 256,
            j: 0,
            lo,
            n,
        });

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
            out.push(SortStep {
                k,
                j: u32::MAX,
                lo,
                n,
            });
            k <<= 1;
        }
    }

    /// Rebuild sort_steps from the current emitter configuration.
    /// Grow `sort_steps_buf` if needed and upload `sort_steps`. The steps
    /// themselves are computed once, at construction, for the fixed slot
    /// layout (see `MAX_EMITTERS`'s doc) — this only ever runs again if a
    /// future caller changes `MAX_EMITTERS`/`SLOT_SIZE` at runtime, which
    /// nothing in this pass currently does.
    fn upload_sort_steps(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sort_steps: &[SortStep],
        sort_steps_buf: &mut wgpu::Buffer,
        max_sort_steps: &mut u32,
    ) {
        let needed = sort_steps.len() as u32;
        if needed > *max_sort_steps {
            *max_sort_steps = needed.next_power_of_two().max(256);
            *sort_steps_buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Corona SortSteps"),
                size: *max_sort_steps as u64 * std::mem::size_of::<SortStep>() as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
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
        let cell = atlas_size / cells; // 32
        let mut data = vec![0u8; (atlas_size * atlas_size * 4) as usize];

        for sprite in 0..16u32 {
            let col = sprite % cells;
            let row = sprite / cells;
            let ox = col * cell;
            let oy = row * cell;

            for py in 0..cell {
                for px in 0..cell {
                    let half = cell as f32 * 0.5;
                    let dx = (px as f32 + 0.5 - half) / half;
                    let dy = (py as f32 + 0.5 - half) / half;
                    let r = (dx * dx + dy * dy).sqrt();
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
                            let ring = 1.0
                                - ((r - (inner + outer) * 0.5).abs() / ((outer - inner) * 0.5))
                                    .clamp(0.0, 1.0);
                            ring * ring
                        }
                        2 => {
                            // Stars: 4-8 points
                            let points = 4.0 + col as f32 * 1.5;
                            let angle = dy.atan2(dx);
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
                    data[base] = 255;
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
    fn name(&self) -> &'static str {
        "Corona"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            "pre_aa",
            "full_res_depth",
            "corona_emitters",
            "depth",
            "main_scene",
        ]
    }

    fn writes(&self) -> &'static [&'static str] {
        // Draws (LoadOp::Load) directly onto pre_aa — see render_pass_descriptor()
        // below. Declaring this lets the render graph see the real dependency
        // for subpass fusion.
        &["pre_aa"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
        builder.read("full_res_depth");
        builder.read("corona_emitters");
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Resolve `CoronaEmitterComponent`'s `"corona_emitters"` buffer by
        // key, generically, every frame — no renderer method, no stored
        // SceneDB handle, no CPU compaction. `particle_offset`/
        // `particle_count`/`spawn_cursor` are all either fixed-by-slot
        // (authored once, see `MAX_EMITTERS`'s doc) or advanced in place by
        // this pass's own compute shaders directly on the SceneDB buffer, so
        // there is nothing left for the CPU to read back or recompute here.
        self.emitter_count = if ctx.scene_buffers.contains(BufferKey::of("corona_emitters")) {
            MAX_EMITTERS
        } else {
            0
        };
        self.max_particles = DEFAULT_MAX_PARTICLES;

        let uniforms = CoronaUniforms {
            delta_time: ctx.delta_time,
            total_particles: self.max_particles,
            emitter_count: self.emitter_count,
            frame_count: ctx.frame_num as u32,
            sort_k: 0,
            sort_j: 0,
            sort_lo: 0,
            sort_n: 0,
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
            label: Some("Corona Render"),
            color_attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if self.emitter_count == 0 {
            return Ok(());
        }

        // ── Bind group rebuild when buffer pointers change ────────────────────

        let part_ptr = &self.particle_buf as *const _ as usize;
        let camera_ptr = ctx.scene.camera as *const _ as usize;
        let emitter_buf = ctx
            .scene_buffers
            .get(BufferKey::of("corona_emitters"))
            .map(|handle| &handle.buffer)
            .unwrap_or(&self.emitter_buf);
        let emitter_ptr = emitter_buf as *const _ as usize;
        let key = (part_ptr ^ emitter_ptr, camera_ptr);

        if self.bg_key != Some(key) {
            self.compute_bg = Some(Self::build_bg(
                ctx.device,
                &self.compute_bgl,
                &self.uniform_buf,
                &self.particle_buf,
                emitter_buf,
                &self.compact_buf,
                &self.emitter_alive_buf,
                &self.draw_args_staging,
                ctx.scene.camera,
                &self.prefix_buf,
                &self.block_sums_buf,
                &self.sort_key_buf,
                &self.particle_view,
                &self.particle_sampler,
                &self.spawn_cursor_buf,
            ));
            self.render_bg = Some(Self::build_bg(
                ctx.device,
                &self.render_bgl,
                &self.uniform_buf,
                &self.particle_buf,
                emitter_buf,
                &self.compact_buf,
                &self.emitter_alive_buf,
                &self.draw_args_staging,
                ctx.scene.camera,
                &self.prefix_buf,
                &self.block_sums_buf,
                &self.sort_key_buf,
                &self.particle_view,
                &self.particle_sampler,
                &self.spawn_cursor_buf,
            ));
            self.bg_key = Some(key);
        }

        let compute_bg = self.compute_bg.as_ref().unwrap();
        let render_bg = self.render_bg.as_ref().unwrap();
        let wg = self.max_particles.div_ceil(WG);
        let ec = self.emitter_count;

        // ── Pass 1: Simulate ─────────────────────────────────────────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Corona Simulate"),
                    timestamp_writes: None,
                },
            );
            p.set_pipeline(&self.simulate_pipeline);
            p.set_bind_group(0, compute_bg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // ── Pass 2: Emit ─────────────────────────────────────────────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Corona Emit"),
                    timestamp_writes: None,
                },
            );
            p.set_pipeline(&self.emit_pipeline);
            p.set_bind_group(0, compute_bg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // ── Pass 3: Scan local (prefix scan + sort-key sentinel reset) ────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Corona ScanLocal"),
                    timestamp_writes: None,
                },
            );
            p.set_pipeline(&self.scan_local_pipeline);
            p.set_bind_group(0, compute_bg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // ── Pass 4: Scan blocks (cumulative per-emitter offsets) ──────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Corona ScanBlocks"),
                    timestamp_writes: None,
                },
            );
            p.set_pipeline(&self.scan_blocks_pipeline);
            p.set_bind_group(0, compute_bg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // ── Pass 5: Scatter (compact_buf + sort_key_buf) ─────────────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Corona Scatter"),
                    timestamp_writes: None,
                },
            );
            p.set_pipeline(&self.scatter_pipeline);
            p.set_bind_group(0, compute_bg, &[]);
            p.dispatch_workgroups(wg, 1, 1);
        }

        // ── Pass 6: Build draw args ───────────────────────────────────────────
        {
            let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                &wgpu::ComputePassDescriptor {
                    label: Some("Corona BuildMulti"),
                    timestamp_writes: None,
                },
            );
            p.set_pipeline(&self.build_multi_pipeline);
            p.set_bind_group(0, compute_bg, &[]);
            p.dispatch_workgroups(ec, 1, 1);
        }

        // Copy STORAGE staging → INDIRECT buffer (the STORAGE+INDIRECT conflict fix).
        let args_size = ec as u64 * std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        unsafe { &mut *ctx.compute_encoder_ptr }.copy_buffer_to_buffer(
            &self.draw_args_staging,
            0,
            &self.draw_args_buf,
            0,
            args_size,
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
                unsafe { &mut *ctx.compute_encoder_ptr }.copy_buffer_to_buffer(
                    &self.sort_steps_buf,
                    step_idx as u64 * step_size,
                    &self.uniform_buf,
                    16, // byte offset of sort_k in CoronaUniforms
                    step_size,
                );

                let particle_count = step.n;
                let blocks = particle_count.div_ceil(WG);

                if step.j == 0 {
                    // cs_sort_local: initial block sort (k=2..256 in shared memory).
                    let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                        &wgpu::ComputePassDescriptor {
                            label: Some("Corona SortLocal"),
                            timestamp_writes: None,
                        },
                    );
                    p.set_pipeline(&self.sort_local_pipeline);
                    p.set_bind_group(0, compute_bg, &[]);
                    p.dispatch_workgroups(blocks, 1, 1);
                } else if step.j == u32::MAX {
                    // cs_sort_local_merge: tail steps (j=128..1) for a global k-stage.
                    let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                        &wgpu::ComputePassDescriptor {
                            label: Some("Corona SortLocalMerge"),
                            timestamp_writes: None,
                        },
                    );
                    p.set_pipeline(&self.sort_local_merge_pipeline);
                    p.set_bind_group(0, compute_bg, &[]);
                    p.dispatch_workgroups(blocks, 1, 1);
                } else {
                    // cs_sort_global: one compare-swap step for j >= 256.
                    let mut p = unsafe { &mut *ctx.compute_encoder_ptr }.begin_compute_pass(
                        &wgpu::ComputePassDescriptor {
                            label: Some("Corona SortGlobal"),
                            timestamp_writes: None,
                        },
                    );
                    p.set_pipeline(&self.sort_global_pipeline);
                    p.set_bind_group(0, compute_bg, &[]);
                    p.dispatch_workgroups(blocks, 1, 1);
                }
            }
        } // end depth_sort_enabled else branch

        // ── Render pass ──────────────────────────────────────────────────────

        let rp = unsafe { &mut *ctx.active_render_pass_ptr().unwrap() };
        rp.set_pipeline(&self.render_pipeline);
        rp.set_bind_group(0, render_bg, &[]);

        // One draw_indirect per emitter — each draws only its alive, sorted particles.
        let stride = std::mem::size_of::<libhelio::GpuCoronaDrawIndirect>() as u64;
        for i in 0..ec {
            rp.draw_indirect(&self.draw_args_buf, i as u64 * stride);
        }

        Ok(())
    }
}
