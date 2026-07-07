//! GPU-native post-processing pipeline for Helio.
//!
//! Sub-stages (execution order in `execute()`):
//!   0. `cs_volume_blend`       — blend active post-process volumes → output (compute)
//!   1. `cs_exposure`           — luminance histogram → avg luminance (compute)
//!   2. `cs_bloom_down_extract` — extract brights from HDR → bloom mip 0 (compute)
//!   3. `cs_bloom_down`         — 2x downsample mip chain, 4 passes (compute)
//!   4. `fs_uber`               — tonemap, color grade, vignette, CA, grain (render)
//!
//! Bind groups:
//!   Main BGLs (group 0): uniforms, samplers, hdr/depth, bloom, noise, custom, volumes, blend output
//!   Bloom BGL (group 1): per-dispatch bloom src (sampled) + dst (storage write)
//!   Blend BGL (group 0, separate layout): postprocess, camera, pp_volumes, blend_output
//!
//! See also `postprocess.wgsl` for shader-level injection points:
//!   INJECTION_POINT_0 (pre-blend), INJECTION_POINT_1 (post-tonemap),
//!   INJECTION_POINT_2 (post-grain), INJECTION_POINT_3 (final)

use bytemuck;
use helio_core::graph::ResourceBuilder;
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const BASE_SHADER_SRC: &str = include_str!("../shaders/postprocess.wgsl");

const BLOOM_MIPS: u32 = 5;
const WG_BLOOM: u32 = 8;
const WG_EXPOSURE_X: u32 = 16;
const WG_EXPOSURE_Y: u32 = 16;
#[allow(dead_code)]
const MAX_PP_VOLUMES: u32 = 256;

/// Position in the uber-shader effect chain where a user effect is injected.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UserEffectPosition {
    PreBlend = 0,    // before exposure / bloom / color grade
    PostTonemap = 1, // after tonemap, before vignette / CA / grain
    PostGrain = 2,   // after grain, before DoF / motion blur
    Final = 3,       // after all built-in effects
}

/// A single user-defined effect entry in the shader injection chain.
#[derive(Clone, Debug)]
pub struct UserEffectEntry {
    /// Which injection point to splice into.
    pub position: UserEffectPosition,
    /// Complete WGSL function body (e.g. `return color * 0.5;`).
    /// The function receives `(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32>`.
    pub body: String,
}

pub struct PostProcessPass {
    avg_luminance_buf: wgpu::Buffer,

    exposure_pipeline: wgpu::ComputePipeline,
    bloom_extract_pipeline: wgpu::ComputePipeline,
    bloom_down_pipeline: wgpu::ComputePipeline,
    uber_pipeline: wgpu::RenderPipeline,
    volume_blend_pipeline: wgpu::ComputePipeline,

    // Separate BGLs for compute vs render
    compute_main_bgl: wgpu::BindGroupLayout,
    render_main_bgl: wgpu::BindGroupLayout,
    bloom_compute_bgl: wgpu::BindGroupLayout,
    blend_bgl: wgpu::BindGroupLayout,

    compute_main_bg: Option<wgpu::BindGroup>,
    render_main_bg: Option<wgpu::BindGroup>,
    main_bg_key: Option<(usize, usize, usize, usize)>,

    // Bloom BGs
    bloom_extract_bg: Option<(usize, wgpu::BindGroup)>,
    bloom_down_bgs: Vec<wgpu::BindGroup>,

    // Volume blend BG
    blend_bg: Option<wgpu::BindGroup>,
    blend_bg_key: Option<(usize, usize)>,

    bloom_textures: Vec<wgpu::Texture>,
    bloom_sampled_views: Vec<wgpu::TextureView>,
    bloom_storage_views: Vec<wgpu::TextureView>,

    linear_sampler: wgpu::Sampler,
    point_sampler: wgpu::Sampler,

    width: u32,
    height: u32,
    format: wgpu::TextureFormat,

    first_frame: bool,

    // ── GPU volume blending ────────────────────────────────────────────────
    blend_output_buf: wgpu::Buffer,

    // ── Bloom gating ───────────────────────────────────────────────────────
    bloom_active: bool,

    // ── Custom effect infrastructure ───────────────────────────────────────
    noise_texture: wgpu::Texture,
    noise_view: wgpu::TextureView,
    noise_sampler: wgpu::Sampler,
    custom_params_buf: wgpu::Buffer,
    custom_params: Vec<[f32; 4]>,

    uber_pl: wgpu::PipelineLayout,

    // Current user shader snippet (the one baked into uber_pipeline).
    user_shader_snippet: Option<String>,
    // Pending snippet queued by set_user_shader — applied in prepare().
    pending_shader_snippet: Option<String>,
    // Multi-effect chain entries.
    user_effect_entries: Vec<UserEffectEntry>,
    // Cached built shader source to avoid rebuilding identical configs.
    cached_shader_source: Option<String>,
}

impl PostProcessPass {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self::new_with_user_effects(device, queue, width, height, format, None)
    }

    pub fn new_with_user_effects(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        user_effects_fn: Option<&str>,
    ) -> Self {
        let initial_entries = user_effects_fn.map(|body| {
            vec![UserEffectEntry {
                // Legacy API: inject at FINAL position to match old pass-through
                // behavior where user_effects was the only thing running.
                position: UserEffectPosition::Final,
                body: body.to_string(),
            }]
        }).unwrap_or_default();

        let initial_src = Self::build_shader_source(&initial_entries);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(initial_src.clone().into()),
        });

        let avg_luminance_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Avg Luminance"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Linear Sampler"),
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let point_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Point Sampler"),
            min_filter: wgpu::FilterMode::Nearest,
            mag_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let (bloom_textures, bloom_sampled_views, bloom_storage_views) =
            Self::create_bloom_mips(device, width, height);

        // ── Blend output buffer (storage, written by cs_volume_blend) ───────
        let blend_output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Blend Output"),
            size: std::mem::size_of::<libhelio::GpuPostProcessUniforms>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // ── Shared BGL entry helpers ────────────────────────────────────────

        let uniform_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let sampled_tex_entry =
            |binding: u32, vis: wgpu::ShaderStages, depth: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Texture {
                    sample_type: if depth {
                        wgpu::TextureSampleType::Depth
                    } else {
                        wgpu::TextureSampleType::Float { filterable: true }
                    },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            };
        let sampler_entry =
            |binding: u32, vis: wgpu::ShaderStages, filtering: bool| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Sampler(if filtering {
                    wgpu::SamplerBindingType::Filtering
                } else {
                    wgpu::SamplerBindingType::NonFiltering
                }),
                count: None,
            };
        let storage_buf_entry =
            |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: vis,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            };
        let storage_ro_entry = |binding: u32, vis: wgpu::ShaderStages| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let cv = wgpu::ShaderStages::COMPUTE;
        let fv = wgpu::ShaderStages::FRAGMENT;
        let cfv = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT;

        // ── compute_main_bgl: b0-b16 (uses storage for b15-b16) ─────────────
        let compute_main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Compute Main BGL"),
            entries: &[
                uniform_entry(0, cfv),
                uniform_entry(1, cfv),
                sampled_tex_entry(2, cfv, false),
                sampled_tex_entry(3, fv, true),
                sampler_entry(4, cfv, true),
                sampler_entry(5, fv, false),
                storage_buf_entry(11, cfv),
                sampled_tex_entry(12, cfv, false),
                sampler_entry(13, cfv, false),
                storage_ro_entry(14, cfv),
                storage_ro_entry(15, cfv),
                storage_buf_entry(16, cfv),
            ],
        });

        // ── render_main_bgl: b0-b14 (bloom sampled at b6-b10) ──────────────
        let render_main_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Render Main BGL"),
            entries: &[
                uniform_entry(0, fv),
                uniform_entry(1, fv),
                sampled_tex_entry(2, fv, false),
                sampled_tex_entry(3, fv, true),
                sampler_entry(4, fv, true),
                sampler_entry(5, fv, false),
                sampled_tex_entry(6, fv, false),
                sampled_tex_entry(7, fv, false),
                sampled_tex_entry(8, fv, false),
                sampled_tex_entry(9, fv, false),
                sampled_tex_entry(10, fv, false),
                storage_buf_entry(11, fv),
                sampled_tex_entry(12, fv, false),
                sampler_entry(13, fv, false),
                storage_ro_entry(14, fv),
            ],
        });

        // ── bloom_compute_bgl: per-dispatch src + dst ──────────────────────
        let bloom_compute_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Bloom Compute BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: cv,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: cv,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // ── blend_bgl: postprocess (b0), camera (b1), pp_volumes (b15), blend_output (b16) ──
        let blend_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PostProcess Volume Blend BGL"),
            entries: &[
                uniform_entry(0, cv),
                uniform_entry(1, cv),
                storage_ro_entry(15, cv),
                storage_buf_entry(16, cv),
            ],
        });

        // Precompute bloom_down BGs
        let bloom_down_bgs = Self::make_bloom_down_bgs(
            device,
            &bloom_compute_bgl,
            &bloom_sampled_views,
            &bloom_storage_views,
        );

        // ── Pipeline layouts ──────────────────────────────────────────────────
        let exposure_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Exposure PL"),
            bind_group_layouts: &[Some(&compute_main_bgl)],
            immediate_size: 0,
        });
        let bloom_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Bloom PL"),
            bind_group_layouts: &[Some(&compute_main_bgl), Some(&bloom_compute_bgl)],
            immediate_size: 0,
        });
        let blend_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Volume Blend PL"),
            bind_group_layouts: &[Some(&blend_bgl)],
            immediate_size: 0,
        });
        let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PostProcess Render PL"),
            bind_group_layouts: &[Some(&render_main_bgl)],
            immediate_size: 0,
        });

        let mk_compute = |label: &str, entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let exposure_pipeline = mk_compute("PostProcess Exposure", "cs_exposure", &exposure_pl);
        let volume_blend_pipeline = mk_compute("PostProcess Volume Blend", "cs_volume_blend", &blend_pl);
        let bloom_extract_pipeline = mk_compute("PostProcess Bloom Extract", "cs_bloom_down_extract", &bloom_pl);
        let bloom_down_pipeline = mk_compute("PostProcess Bloom Down", "cs_bloom_down", &bloom_pl);

        let uber_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PostProcess Uber Pipeline"),
            layout: Some(&render_pl),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_uber"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
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

        // ── Noise texture ──────────────────────────────────────────────────
        let noise_size = 64u32;
        let noise_data: Vec<u8> = {
            let mut rng: u64 = 987654321;
            let mut data = Vec::with_capacity((noise_size * noise_size) as usize);
            for _ in 0..noise_size * noise_size {
                rng = rng
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                data.push((rng >> 40) as u8);
            }
            data
        };
        let noise_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PostProcess Noise"),
            size: wgpu::Extent3d { width: noise_size, height: noise_size, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &noise_texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &noise_data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(noise_size), rows_per_image: Some(noise_size) },
            wgpu::Extent3d { width: noise_size, height: noise_size, depth_or_array_layers: 1 },
        );
        let noise_view = noise_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let noise_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PostProcess Noise Sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });
        let custom_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PostProcess Custom Params"),
            size: 64 * 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let stored_snippet = user_effects_fn.map(|s| s.to_string());

        Self {
            avg_luminance_buf,
            exposure_pipeline,
            bloom_extract_pipeline,
            bloom_down_pipeline,
            uber_pipeline,
            volume_blend_pipeline,
            compute_main_bgl,
            render_main_bgl,
            bloom_compute_bgl,
            blend_bgl,
            compute_main_bg: None,
            render_main_bg: None,
            main_bg_key: None,
            bloom_extract_bg: None,
            bloom_down_bgs,
            blend_bg: None,
            blend_bg_key: None,
            bloom_textures,
            bloom_sampled_views,
            bloom_storage_views,
            linear_sampler,
            point_sampler,
            width,
            height,
            format,
            first_frame: true,
            blend_output_buf,
            bloom_active: true,
            noise_texture,
            noise_view,
            noise_sampler,
            custom_params_buf,
            custom_params: Vec::new(),
            user_shader_snippet: stored_snippet,
            pending_shader_snippet: None,
            user_effect_entries: initial_entries,
            cached_shader_source: Some(initial_src),
            uber_pl: render_pl,
        }
    }

    // ── Shader source builder ────────────────────────────────────────────────

    /// Build the complete WGSL source by splicing user effect entries into the
    /// base shader at `//%P0` through `//%P3` markers.
    ///
    /// Each entry is either:
    /// - A complete `fn user_effects(...)` definition (old API via `new_with_user_effects`)
    ///   → placed verbatim at module scope; a call is emitted at the marker.
    /// - A bare expression body (new API via `add_user_effect`)
    ///   → wrapped in a generated `fn` and placed at module scope; a call emitted at the marker.
    fn build_shader_source(entries: &[UserEffectEntry]) -> String {
        let base = BASE_SHADER_SRC;
        let mut result = base.to_string();

        // Collect module-scope definitions and per-position calls.
        let mut defs = String::new();
        let mut calls_by_pos: [Vec<String>; 4] = [vec![], vec![], vec![], vec![]];
        let mut entry_index = 0usize;

        for e in entries {
            let pos = e.position as usize;
            if pos >= 4 { continue; }

            let trimmed = e.body.trim();

            if trimmed.starts_with("fn ") {
                // Old API: one or more complete function definitions.
                // The last `fn` is usually the main `user_effects(...)` entry point.
                // Extract its name for the call site.
                let fn_name = "user_effects";
                defs.push_str(&format!("{}\n", trimmed));
                calls_by_pos[pos].push(format!("    color = {}(color, uv, dims);\n", fn_name));
            } else {
                // New API: bare expression body — wrap in a generated function.
                let fn_name = format!("userfx_{}", entry_index);
                entry_index += 1;
                defs.push_str(&format!(
                    "fn {}(color: vec3<f32>, uv: vec2<f32>, dims: vec2<f32>) -> vec3<f32> {{ return {}; }}\n",
                    fn_name, trimmed
                ));
                calls_by_pos[pos].push(format!("    color = {}(color, uv, dims);\n", fn_name));
            }
        }

        // Replace markers with calls, then append definitions at module scope.
        for (pos, calls) in calls_by_pos.iter().enumerate() {
            let marker = format!("//%P{}", pos);
            if calls.is_empty() {
                result = result.replace(&marker, &format!("//%P{} (empty)", pos));
            } else {
                let splice: String = calls.iter().flat_map(|c| c.chars()).collect();
                result = result.replace(&marker, &splice);
            }
        }

        if !defs.is_empty() {
            result.push_str("\n// ── Injected user effects ──\n");
            result.push_str(&defs);
        }

        result
    }

    fn rebuild_uber_from_entries(&mut self, device: &wgpu::Device) {
        let source = Self::build_shader_source(&self.user_effect_entries);
        if self.cached_shader_source.as_deref() == Some(&source) {
            return; // identical — skip rebuild
        }
        let shader_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PostProcess Shader"),
            source: wgpu::ShaderSource::Wgsl(source.clone().into()),
        });
        self.uber_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("PostProcess Uber"),
            layout: Some(&self.uber_pl),
            vertex: wgpu::VertexState {
                module: &shader_mod,
                entry_point: Some("vs_fullscreen"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_mod,
                entry_point: Some("fs_uber"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
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
        self.cached_shader_source = Some(source);
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Gate bloom compute dispatches on/off.
    /// Call each frame from the renderer when the blended bloom settings are known.
    pub fn set_bloom_active(&mut self, active: bool) {
        self.bloom_active = active;
    }

    /// Queue a new user shader snippet to be applied at the start of the next frame.
    /// The pipeline rebuild happens in `prepare()`, not on the calling thread.
    /// Pass `None` to restore the default no-op.
    pub fn set_user_shader(&mut self, wgsl: Option<&str>) {
        self.pending_shader_snippet = wgsl.map(|s| s.to_string());
    }

    /// Add a user effect entry at the given position in the effect chain.
    /// Effects are applied in the order they are added.
    /// Call `commit_user_effects()` to rebuild the uber-pipeline.
    pub fn add_user_effect(&mut self, position: UserEffectPosition, body: &str) {
        self.user_effect_entries.push(UserEffectEntry {
            position,
            body: body.to_string(),
        });
    }

    /// Remove all user effect entries and rebuild the pipeline.
    pub fn clear_user_effects(&mut self, device: &wgpu::Device) {
        self.user_effect_entries.clear();
        self.rebuild_uber_from_entries(device);
    }

    /// Rebuild the uber-pipeline with the current set of user effect entries.
    /// Called automatically if `set_user_shader()` is used (legacy path).
    pub fn commit_user_effects(&mut self, device: &wgpu::Device) {
        self.rebuild_uber_from_entries(device);
    }

    /// Upload custom float4 parameters that the shader reads from `pp_custom`.
    pub fn set_custom_params(&mut self, params: &[[f32; 4]]) {
        self.custom_params.clear();
        self.custom_params.extend_from_slice(params);
    }

    // ── Internal helpers ────────────────────────────────────────────────────

    fn create_bloom_mips(
        device: &wgpu::Device,
        width: u32,
        height: u32,
    ) -> (Vec<wgpu::Texture>, Vec<wgpu::TextureView>, Vec<wgpu::TextureView>) {
        let mut textures = Vec::with_capacity(BLOOM_MIPS as usize);
        let mut sampled_views = Vec::with_capacity(BLOOM_MIPS as usize);
        let mut storage_views = Vec::with_capacity(BLOOM_MIPS as usize);

        for i in 0..BLOOM_MIPS {
            let mw = (width >> (i + 1)).max(1);
            let mh = (height >> (i + 1)).max(1);
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom Mip {}", i)),
                size: wgpu::Extent3d { width: mw, height: mh, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            sampled_views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Bloom Mip {} Sampled", i)),
                ..Default::default()
            }));
            storage_views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                label: Some(&format!("Bloom Mip {} Storage", i)),
                ..Default::default()
            }));
            textures.push(tex);
        }
        (textures, sampled_views, storage_views)
    }

    fn make_bloom_down_bgs(
        device: &wgpu::Device,
        bloom_compute_bgl: &wgpu::BindGroupLayout,
        bloom_sampled_views: &[wgpu::TextureView],
        bloom_storage_views: &[wgpu::TextureView],
    ) -> Vec<wgpu::BindGroup> {
        (1..BLOOM_MIPS as usize)
            .map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("PostProcess Bloom Down BG mip{}", i)),
                    layout: bloom_compute_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&bloom_sampled_views[i - 1]) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&bloom_storage_views[i]) },
                    ],
                })
            })
            .collect()
    }

    fn rebuild_bind_groups(
        &mut self,
        device: &wgpu::Device,
        postprocess_buf: &wgpu::Buffer,
        pre_aa_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        camera_buf: &wgpu::Buffer,
        pp_volumes_buf: &wgpu::Buffer,
    ) {
        self.compute_main_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Compute Main BG"),
            layout: &self.compute_main_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: postprocess_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(pre_aa_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.point_sampler) },
                wgpu::BindGroupEntry { binding: 11, resource: self.avg_luminance_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.noise_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(&self.noise_sampler) },
                wgpu::BindGroupEntry { binding: 14, resource: self.custom_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: pp_volumes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: self.blend_output_buf.as_entire_binding() },
            ],
        }));

        self.render_main_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Render Main BG"),
            layout: &self.render_main_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: postprocess_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(pre_aa_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.linear_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(&self.point_sampler) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[0]) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[1]) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[2]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[3]) },
                wgpu::BindGroupEntry { binding: 10, resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[4]) },
                wgpu::BindGroupEntry { binding: 11, resource: self.avg_luminance_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: wgpu::BindingResource::TextureView(&self.noise_view) },
                wgpu::BindGroupEntry { binding: 13, resource: wgpu::BindingResource::Sampler(&self.noise_sampler) },
                wgpu::BindGroupEntry { binding: 14, resource: self.custom_params_buf.as_entire_binding() },
            ],
        }));
    }

    fn rebuild_blend_bg(
        &mut self,
        device: &wgpu::Device,
        postprocess_buf: &wgpu::Buffer,
        camera_buf: &wgpu::Buffer,
        pp_volumes_buf: &wgpu::Buffer,
    ) {
        self.blend_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PostProcess Volume Blend BG"),
            layout: &self.blend_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: postprocess_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: camera_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 15, resource: pp_volumes_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 16, resource: self.blend_output_buf.as_entire_binding() },
            ],
        }));
    }

    fn mip_dims(&self, mip: u32) -> (u32, u32) {
        ((self.width >> (mip + 1)).max(1), (self.height >> (mip + 1)).max(1))
    }
}

impl RenderPass for PostProcessPass {
    fn name(&self) -> &'static str {
        "PostProcess"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["pre_aa"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        builder.read("pre_aa");
    }

    fn on_resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        let (textures, sampled_views, storage_views) = Self::create_bloom_mips(device, width, height);
        self.bloom_textures = textures;
        self.bloom_sampled_views = sampled_views;
        self.bloom_storage_views = storage_views;
        self.bloom_down_bgs = Self::make_bloom_down_bgs(
            device, &self.bloom_compute_bgl, &self.bloom_sampled_views, &self.bloom_storage_views,
        );
        self.compute_main_bg = None;
        self.render_main_bg = None;
        self.main_bg_key = None;
        self.bloom_extract_bg = None;
        self.blend_bg = None;
        self.blend_bg_key = None;
        self.first_frame = true;
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if self.first_frame {
            self.first_frame = false;
            let initial: f32 = 0.18;
            ctx.queue.write_buffer(&self.avg_luminance_buf, 0, bytemuck::bytes_of(&initial));
        }

        // Deferred shader rebuild: if a snippet was queued, apply it now.
        if self.pending_shader_snippet.is_some() {
            let pending = self.pending_shader_snippet.take().unwrap();
            let changed = self.user_shader_snippet.as_deref() != Some(pending.as_str());
            if changed {
                // Convert legacy single-snippet API into a single Final entry
                // (matching old pass-through behavior).
                self.user_effect_entries.retain(|e| e.position != UserEffectPosition::Final);
                self.user_effect_entries.push(UserEffectEntry {
                    position: UserEffectPosition::Final,
                    body: pending.clone(),
                });
                self.rebuild_uber_from_entries(ctx.device);
                self.user_shader_snippet = Some(pending);
            }
        }

        // Upload custom params
        if !self.custom_params.is_empty() {
            ctx.queue.write_buffer(
                &self.custom_params_buf,
                0,
                bytemuck::cast_slice(&self.custom_params),
            );
        }

        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let pre_aa_view = match ctx.resources.pre_aa.get() {
            Some(v) => v,
            None => return Ok(()),
        };
        let postprocess_buf = match ctx.resources.postprocess_uniforms.get() {
            Some(v) => v,
            None => return Ok(()),
        };

        // Volume buffer — may be empty, that's fine.
        let pp_volumes_buf = ctx.resources.pp_volumes.get();
        let pp_volume_count = ctx.resources.pp_volume_count;

        let camera_buf = ctx.scene.camera;

        let bg_key = (
            pre_aa_view as *const _ as usize,
            ctx.depth as *const _ as usize,
            camera_buf as *const _ as usize,
            postprocess_buf as *const _ as usize,
        );
        if self.main_bg_key != Some(bg_key) {
            let vol_buf = pp_volumes_buf.unwrap_or(postprocess_buf); // fallback to any valid buf
            self.rebuild_bind_groups(ctx.device, postprocess_buf, pre_aa_view, ctx.depth, camera_buf, vol_buf);
            self.main_bg_key = Some(bg_key);
        }

        let vol_key = (
            camera_buf as *const _ as usize,
            pp_volumes_buf.map(|b| b as *const _ as usize).unwrap_or(0),
        );
        if self.blend_bg_key != Some(vol_key) {
            if let Some(vol_buf) = pp_volumes_buf {
                self.rebuild_blend_bg(ctx.device, postprocess_buf, camera_buf, vol_buf);
            }
            self.blend_bg_key = Some(vol_key);
        }

        // Bloom extract BG
        let hdr_ptr = pre_aa_view as *const _ as usize;
        if self.bloom_extract_bg.as_ref().map(|(k, _)| *k) != Some(hdr_ptr) {
            let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PostProcess Bloom Extract BG"),
                layout: &self.bloom_compute_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&self.bloom_sampled_views[1]) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.bloom_storage_views[0]) },
                ],
            });
            self.bloom_extract_bg = Some((hdr_ptr, bg));
        }

        let compute_bg = self.compute_main_bg.as_ref().unwrap();
        let render_bg = self.render_main_bg.as_ref().unwrap();
        let extract_bg = &self.bloom_extract_bg.as_ref().unwrap().1;
        let ce = ctx.compute_encoder_ptr;

        // 0. GPU volume blending (if volumes are present)
        if pp_volume_count > 0 {
            if let Some(ref blend_bg) = self.blend_bg {
                let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PostProcess Volume Blend"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.volume_blend_pipeline);
                cpass.set_bind_group(0, blend_bg, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Copy blended result from storage buffer to uniform buffer
            let postprocess_size = std::mem::size_of::<libhelio::GpuPostProcessUniforms>() as u64;
            unsafe { &mut *ce }.copy_buffer_to_buffer(
                &self.blend_output_buf, 0,
                postprocess_buf, 0,
                postprocess_size,
            );
        }

        // 1. Auto-exposure histogram
        {
            let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PostProcess Exposure"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.exposure_pipeline);
            cpass.set_bind_group(0, compute_bg, &[]);
            let gx = (self.width / (4 * WG_EXPOSURE_X)).max(1);
            let gy = (self.height / (4 * WG_EXPOSURE_Y)).max(1);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // 2. Bloom (only when active)
        if self.bloom_active {
            // 2a. Bloom extract: HDR → mip 0
            {
                let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PostProcess Bloom Extract"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_extract_pipeline);
                cpass.set_bind_group(0, compute_bg, &[]);
                cpass.set_bind_group(1, extract_bg, &[]);
                let (mw, mh) = self.mip_dims(0);
                cpass.dispatch_workgroups(
                    (mw + WG_BLOOM - 1) / WG_BLOOM,
                    (mh + WG_BLOOM - 1) / WG_BLOOM,
                    1,
                );
            }

            // 2b. Bloom downsample
            for i in 0..(BLOOM_MIPS as usize - 1) {
                let (mw, mh) = self.mip_dims(i as u32 + 1);
                let mut cpass = unsafe { &mut *ce }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("PostProcess Bloom Down mip{}", i + 1)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.bloom_down_pipeline);
                cpass.set_bind_group(0, compute_bg, &[]);
                cpass.set_bind_group(1, &self.bloom_down_bgs[i], &[]);
                cpass.dispatch_workgroups(
                    (mw + WG_BLOOM - 1) / WG_BLOOM,
                    (mh + WG_BLOOM - 1) / WG_BLOOM,
                    1,
                );
            }
        }

        // 3. Uber render pass
        {
            let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("PostProcess Uber"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: ctx.target,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.uber_pipeline);
            pass.set_bind_group(0, render_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        Ok(())
    }
}
