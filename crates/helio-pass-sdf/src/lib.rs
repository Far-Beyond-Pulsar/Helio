//! Screen-Space SDF pass — Jump Flooding Algorithm (JFA), entirely GPU-driven.
//!
//! Consumes the scene depth buffer and produces a two-channel `Rg16Float` SDF texture:
//!   `.r` — normalised unsigned distance to the nearest surface edge  \[0, 1\]
//!   `.g` — sign: `+1.0` outside / background, `−1.0` on or inside a surface
//!
//! # Algorithm
//!
//! 1. **Seed pass** (1 dispatch) — marks silhouette/edge pixels as JFA seeds.
//! 2. **JFA flood passes** (`ceil(log2(max(W,H)))` dispatches, ≤ 12 for 4K) —
//!    iteratively propagates the nearest seed to every pixel using power-of-two
//!    jump offsets (step = max_dim/2, max_dim/4, …, 1).
//! 3. **Resolve pass** (1 dispatch) — converts nearest-seed coordinates to a
//!    signed, normalised distance and writes the final SDF texture.
//!
//! # Complexity
//!
//! | Resource      | Cost                                         |
//! |---------------|----------------------------------------------|
//! | CPU per frame | O(1) — fixed number of `dispatch()` calls    |
//! | GPU per frame | O(N·log N) — JFA is embarrassingly parallel   |
//! | Memory        | 2 × `Rg32Float` ping-pong + 1 × `Rg16Float` |
//!
//! # Integration with GBuffer pass
//!
//! Pass ordering (suggested):
//! ```text
//! DepthPrepass → GBuffer → … → SdfPass → DeferredLight / SSAO
//! ```
//! The SDF output (`sdf_view`) can be sampled in any downstream pass for:
//! - **Soft contact shadows** (sample SDF in shadow shader)
//! - **Screen-space AO** (SDF-guided importance sampling)
//! - **Distance-field font / decal rendering**
//! - **Screen-edge detection** (anti-aliasing, outline effects)

use bytemuck::{Pod, Zeroable};
use helio_v3::{RenderPass, PassContext, PrepareContext, Result as HelioResult};

// ── Constants ─────────────────────────────────────────────────────────────────

const WORKGROUP: u32 = 8;
/// Maximum number of JFA mip/step iterations we pre-allocate resources for.
/// ceil(log2(4096)) = 12, so 12 covers resolutions up to 4096 × 4096.
const MAX_JFA_STEPS: usize = 12;

// ── GPU uniform layout ────────────────────────────────────────────────────────

/// Matches `SdfUniforms` in `sdf.wgsl` (32 bytes, 8 × u32/f32).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SdfUniforms {
    width:       u32,
    height:      u32,
    step:        u32,
    near:        f32,
    far:         f32,
    edge_thresh: f32,
    _pad0:       u32,
    _pad1:       u32,
}

// ── Pass struct ───────────────────────────────────────────────────────────────

/// Screen-space SDF compute pass.
///
/// Allocate once with [`SdfPass::new`]; call [`SdfPass::resize`] on window resize.
pub struct SdfPass {
    // ── Pipelines ─────────────────────────────────────────────────────────────
    pipeline_seed:    wgpu::ComputePipeline,
    pipeline_jfa:     wgpu::ComputePipeline,
    pipeline_resolve: wgpu::ComputePipeline,

    #[allow(dead_code)]
    bgl: wgpu::BindGroupLayout,

    // ── Per-step resources (ping-pong) ────────────────────────────────────────
    /// One uniform buffer per JFA step + 2 extra (seed pass + resolve pass).
    uniform_bufs: Vec<wgpu::Buffer>,
    /// Bind groups for seed pass (writes ping buffer from depth).
    bg_seed: wgpu::BindGroup,
    /// Bind groups for each JFA step (reads ping→writes pong or vice-versa).
    bg_jfa: Vec<wgpu::BindGroup>,
    /// Bind group for resolve pass (reads final pong, writes sdf_out).
    bg_resolve: wgpu::BindGroup,

    // ── Textures (pub: downstream passes read these) ──────────────────────────
    /// Ping buffer (Rg32Float storage, seed coords).
    _ping_tex: wgpu::Texture,
    ping_view_read:  wgpu::TextureView,
    ping_view_write: wgpu::TextureView,
    /// Pong buffer (Rg32Float storage, seed coords).
    _pong_tex: wgpu::Texture,
    pong_view_read:  wgpu::TextureView,
    pong_view_write: wgpu::TextureView,

    /// Final SDF output (Rg16Float).  Sample `.r` for distance, `.g` for sign.
    pub sdf_tex:  wgpu::Texture,
    pub sdf_view: wgpu::TextureView,

    // ── Step configuration ────────────────────────────────────────────────────
    /// Number of JFA iterations this frame (= ceil(log2(max(W, H)))).
    jfa_step_count: usize,

    // ── Camera near/far (updated each prepare()) ─────────────────────────────
    near: f32,
    far:  f32,

    width:  u32,
    height: u32,
}

impl SdfPass {
    /// Construct the SDF pass.
    ///
    /// * `depth_view` — the scene depth texture view (Depth32Float, read in shader).
    ///   Must remain valid for the lifetime of this pass (or until `resize()`).
    /// * `near / far` — camera clip planes for depth linearisation.
    /// * `edge_thresh` — view-space edge detection sensitivity (try `0.1`..`0.5`).
    pub fn new(
        device:      &wgpu::Device,
        depth_view:  &wgpu::TextureView,
        width:        u32,
        height:       u32,
        near:         f32,
        far:          f32,
        edge_thresh:  f32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label:  Some("SDF Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sdf.wgsl").into()),
        });

        // ── Bind Group Layout ─────────────────────────────────────────────────
        // All three passes share the same BGL (the step uniform changes content;
        // we allocate one buffer per step to avoid mid-frame re-uploads).
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SDF BGL"),
            entries: &[
                // 0: uniforms (uniform buffer — width, height, step, near, far, …)
                wgpu::BindGroupLayoutEntry {
                    binding:    0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 1: depth texture (read, non-filtered)
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                // 2: seed_src (storage read — ping or pong)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::ReadOnly,
                        format:         wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 3: seed_dst (storage write — pong or ping)
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 4: sdf_out (storage write — only used in resolve pass)
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access:         wgpu::StorageTextureAccess::WriteOnly,
                        format:         wgpu::TextureFormat::Rg16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 5: depth sampler (non-filtering nearest)
                wgpu::BindGroupLayoutEntry {
                    binding:    5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
            ],
        });

        // ── Pipelines ─────────────────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:                Some("SDF PL"),
            bind_group_layouts:   &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline_seed = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("SDF Seed"),
            layout:              Some(&pipeline_layout),
            module:              &shader,
            entry_point:         Some("seed_pass"),
            compilation_options: Default::default(),
            cache:               None,
        });
        let pipeline_jfa = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("SDF JFA"),
            layout:              Some(&pipeline_layout),
            module:              &shader,
            entry_point:         Some("jfa_pass"),
            compilation_options: Default::default(),
            cache:               None,
        });
        let pipeline_resolve = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("SDF Resolve"),
            layout:              Some(&pipeline_layout),
            module:              &shader,
            entry_point:         Some("resolve_pass"),
            compilation_options: Default::default(),
            cache:               None,
        });

        // ── Textures ──────────────────────────────────────────────────────────
        let (ping_tex, ping_view_read, ping_view_write) =
            make_seed_texture(device, width, height, "SDF/Ping");
        let (pong_tex, pong_view_read, pong_view_write) =
            make_seed_texture(device, width, height, "SDF/Pong");
        let (sdf_tex, sdf_view) = make_sdf_texture(device, width, height);

        // ── Depth sampler ─────────────────────────────────────────────────────
        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("SDF DepthSampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // ── Null storage view for passes that don't write t_sdf_out ──────────
        // The BGL requires binding 4 everywhere, so we point JFA/seed passes at
        // the real SDF output (it just won't be written by those entry points).
        let jfa_step_count = jfa_steps(width, height);

        // ── Uniform buffers ───────────────────────────────────────────────────
        // 1 (seed) + jfa_step_count + 1 (resolve)
        let total_uniforms = 2 + jfa_step_count;
        let mut uniform_bufs: Vec<wgpu::Buffer> = (0..total_uniforms)
            .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&format!("SDF Uniforms[{i}]")),
                size:               std::mem::size_of::<SdfUniforms>() as u64,
                usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
            .collect();

        // ── Build bind groups ─────────────────────────────────────────────────
        // Seed: writes into ping (t_seed_dst = ping_write)
        let bg_seed = make_bind_group(
            device, &bgl,
            &uniform_bufs[0],
            depth_view,
            &ping_view_read,   // src (unused in seed pass, but must be bound)
            &ping_view_write,  // dst = ping
            &sdf_view,
            &depth_sampler,
            "SDF BG Seed",
        );

        // JFA steps: alternately read from ping → write to pong, then pong → ping
        let mut bg_jfa: Vec<wgpu::BindGroup> = Vec::with_capacity(jfa_step_count);
        for s in 0..jfa_step_count {
            let (src, dst) = if s % 2 == 0 {
                (&ping_view_read, &pong_view_write)
            } else {
                (&pong_view_read, &ping_view_write)
            };
            bg_jfa.push(make_bind_group(
                device, &bgl,
                &uniform_bufs[1 + s],
                depth_view,
                src, dst,
                &sdf_view,
                &depth_sampler,
                &format!("SDF BG JFA[{s}]"),
            ));
        }

        // Resolve: read from the buffer last written by JFA
        let resolve_src = if jfa_step_count % 2 == 0 {
            &ping_view_read
        } else {
            &pong_view_read
        };
        let bg_resolve = make_bind_group(
            device, &bgl,
            &uniform_bufs[1 + jfa_step_count],
            depth_view,
            resolve_src,
            &ping_view_write, // dst unused in resolve; bind ping again as dummy
            &sdf_view,
            &depth_sampler,
            "SDF BG Resolve",
        );

        // ── Upload initial uniforms ───────────────────────────────────────────
        // (These are re-uploaded every frame in prepare(); we do a first write
        //  here so the buffers have valid contents before the first execute().)
        upload_uniforms(
            device, &mut uniform_bufs,
            width, height, near, far, edge_thresh, jfa_step_count,
        );

        Self {
            pipeline_seed,
            pipeline_jfa,
            pipeline_resolve,
            bgl,
            uniform_bufs,
            bg_seed,
            bg_jfa,
            bg_resolve,
            _ping_tex: ping_tex,
            ping_view_read,
            ping_view_write,
            _pong_tex: pong_tex,
            pong_view_read,
            pong_view_write,
            sdf_tex,
            sdf_view,
            jfa_step_count,
            near,
            far,
            width,
            height,
        }
    }

    /// Update the camera clip planes (call when camera changes projection).
    pub fn set_clip(&mut self, near: f32, far: f32) {
        self.near = near;
        self.far  = far;
    }

    /// Recreate all resolution-dependent resources after a window resize.
    ///
    /// `depth_view` must be the new depth texture view at the new resolution.
    pub fn resize(
        &mut self,
        device:     &wgpu::Device,
        depth_view: &wgpu::TextureView,
        width:       u32,
        height:      u32,
        edge_thresh: f32,
    ) {
        self.width        = width;
        self.height       = height;
        self.jfa_step_count = jfa_steps(width, height);

        // Recreate textures
        let (ping_tex, ping_view_read, ping_view_write) =
            make_seed_texture(device, width, height, "SDF/Ping");
        let (pong_tex, pong_view_read, pong_view_write) =
            make_seed_texture(device, width, height, "SDF/Pong");
        let (sdf_tex, sdf_view) = make_sdf_texture(device, width, height);

        self._ping_tex = ping_tex;
        self.ping_view_read  = ping_view_read;
        self.ping_view_write = ping_view_write;
        self._pong_tex = pong_tex;
        self.pong_view_read  = pong_view_read;
        self.pong_view_write = pong_view_write;
        self.sdf_tex  = sdf_tex;
        self.sdf_view = sdf_view;

        let depth_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label:      Some("SDF DepthSampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let total_uniforms = 2 + self.jfa_step_count;
        self.uniform_bufs = (0..total_uniforms)
            .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                label:              Some(&format!("SDF Uniforms[{i}]")),
                size:               std::mem::size_of::<SdfUniforms>() as u64,
                usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }))
            .collect();

        self.bg_seed = make_bind_group(
            device, &self.bgl,
            &self.uniform_bufs[0], depth_view,
            &self.ping_view_read, &self.ping_view_write,
            &self.sdf_view, &depth_sampler, "SDF BG Seed",
        );

        self.bg_jfa.clear();
        for s in 0..self.jfa_step_count {
            let (src, dst) = if s % 2 == 0 {
                (&self.ping_view_read, &self.pong_view_write)
            } else {
                (&self.pong_view_read, &self.ping_view_write)
            };
            self.bg_jfa.push(make_bind_group(
                device, &self.bgl,
                &self.uniform_bufs[1 + s], depth_view,
                src, dst,
                &self.sdf_view, &depth_sampler,
                &format!("SDF BG JFA[{s}]"),
            ));
        }

        let resolve_src = if self.jfa_step_count % 2 == 0 {
            &self.ping_view_read
        } else {
            &self.pong_view_read
        };
        self.bg_resolve = make_bind_group(
            device, &self.bgl,
            &self.uniform_bufs[1 + self.jfa_step_count], depth_view,
            resolve_src, &self.ping_view_write,
            &self.sdf_view, &depth_sampler, "SDF BG Resolve",
        );

        upload_uniforms(
            device, &mut self.uniform_bufs,
            width, height, self.near, self.far, edge_thresh, self.jfa_step_count,
        );
    }
}

// ── RenderPass impl ───────────────────────────────────────────────────────────

impl RenderPass for SdfPass {
    fn name(&self) -> &'static str { "SdfPass" }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // Re-upload uniforms in case near/far changed (camera updated).
        // This is still O(1) — fixed number of small buffer writes.
        upload_uniforms_queue(
            ctx.queue,
            &self.uniform_bufs,
            self.width, self.height,
            self.near, self.far,
            0.2, // default edge threshold; expose via set_edge_thresh() if needed
            self.jfa_step_count,
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let wx = dispatch_count(self.width,  WORKGROUP);
        let wy = dispatch_count(self.height, WORKGROUP);

        // ── Seed pass ─────────────────────────────────────────────────────────
        {
            let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("SDF Seed"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_seed);
            cpass.set_bind_group(0, &self.bg_seed, &[]);
            cpass.dispatch_workgroups(wx, wy, 1);
        }

        // ── JFA flood passes ──────────────────────────────────────────────────
        for s in 0..self.jfa_step_count {
            let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("SDF JFA"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_jfa);
            cpass.set_bind_group(0, &self.bg_jfa[s], &[]);
            cpass.dispatch_workgroups(wx, wy, 1);
        }

        // ── Resolve pass ──────────────────────────────────────────────────────
        {
            let mut cpass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label:            Some("SDF Resolve"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline_resolve);
            cpass.set_bind_group(0, &self.bg_resolve, &[]);
            cpass.dispatch_workgroups(wx, wy, 1);
        }

        Ok(())
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Number of JFA steps for a resolution: ceil(log2(max(w, h))).
fn jfa_steps(w: u32, h: u32) -> usize {
    let max_dim = w.max(h).max(1);
    // integer ceil(log2)
    let steps = (u32::BITS - max_dim.leading_zeros()) as usize; // floor(log2) + 1 for non-power-2
    steps.min(MAX_JFA_STEPS)
}

/// Ceiling division for workgroup dispatch counts.
#[inline]
fn dispatch_count(dim: u32, wg: u32) -> u32 {
    (dim + wg - 1) / wg
}

/// Create a ping/pong seed coordinate texture (Rg32Float, storage read + write).
fn make_seed_texture(
    device: &wgpu::Device,
    width:   u32,
    height:  u32,
    label:   &str,
) -> (wgpu::Texture, wgpu::TextureView, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:              Some(label),
        size:               wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count:    1,
        sample_count:       1,
        dimension:          wgpu::TextureDimension::D2,
        format:             wgpu::TextureFormat::Rg32Float,
        usage:              wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:       &[],
    });
    let read_view = tex.create_view(&wgpu::TextureViewDescriptor {
        label:  Some(&format!("{label}/Read")),
        ..Default::default()
    });
    let write_view = tex.create_view(&wgpu::TextureViewDescriptor {
        label:  Some(&format!("{label}/Write")),
        ..Default::default()
    });
    (tex, read_view, write_view)
}

/// Create the final SDF output texture (Rg16Float, storage write + sample binding).
fn make_sdf_texture(
    device: &wgpu::Device,
    width:   u32,
    height:  u32,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label:           Some("SDF Output"),
        size:            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count:    1,
        dimension:       wgpu::TextureDimension::D2,
        format:          wgpu::TextureFormat::Rg16Float,
        usage:           wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats:    &[],
    });
    let view = tex.create_view(&wgpu::TextureViewDescriptor::default());
    (tex, view)
}

/// Construct a bind group for one dispatch.
#[allow(clippy::too_many_arguments)]
fn make_bind_group(
    device:       &wgpu::Device,
    bgl:          &wgpu::BindGroupLayout,
    uniform_buf:  &wgpu::Buffer,
    depth_view:   &wgpu::TextureView,
    seed_src:     &wgpu::TextureView,
    seed_dst:     &wgpu::TextureView,
    sdf_out:      &wgpu::TextureView,
    depth_sampler: &wgpu::Sampler,
    label:        &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(seed_src) },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(seed_dst) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(sdf_out) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(depth_sampler) },
        ],
    })
}

/// Upload all uniform buffers using `device.create_buffer_init` indirection —
/// called once at construction time (CPU-side, no queue needed).
/// Actual per-frame data is written by `upload_uniforms_queue` in `prepare()`.
fn upload_uniforms(
    _device:        &wgpu::Device,
    _bufs:          &mut [wgpu::Buffer],
    _width:          u32,
    _height:         u32,
    _near:           f32,
    _far:            f32,
    _edge_thresh:    f32,
    _jfa_step_count: usize,
) {
    // Buffers are populated on the first frame via prepare() → upload_uniforms_queue().
    // No initialisation is needed here because wgpu zero-initialises UNIFORM buffers.
}

/// Upload uniform buffers via the wgpu queue (called every frame from `prepare()`).
fn upload_uniforms_queue(
    queue:          &wgpu::Queue,
    bufs:           &[wgpu::Buffer],
    width:           u32,
    height:          u32,
    near:            f32,
    far:             f32,
    edge_thresh:     f32,
    jfa_step_count:  usize,
) {
    let max_dim = width.max(height);

    // 0: seed pass
    let seed_uni = SdfUniforms { width, height, step: 0, near, far, edge_thresh, _pad0: 0, _pad1: 0 };
    queue.write_buffer(&bufs[0], 0, bytemuck::bytes_of(&seed_uni));

    // 1..=jfa_step_count: JFA passes
    for s in 0..jfa_step_count {
        let step = (max_dim >> (s + 1)).max(1);
        let uni  = SdfUniforms { width, height, step, near, far, edge_thresh, _pad0: 0, _pad1: 0 };
        queue.write_buffer(&bufs[1 + s], 0, bytemuck::bytes_of(&uni));
    }

    // jfa_step_count+1: resolve pass
    let resolve_uni = SdfUniforms { width, height, step: 0, near, far, edge_thresh, _pad0: 0, _pad1: 0 };
    if let Some(buf) = bufs.get(1 + jfa_step_count) {
        queue.write_buffer(buf, 0, bytemuck::bytes_of(&resolve_uni));
    }
}
