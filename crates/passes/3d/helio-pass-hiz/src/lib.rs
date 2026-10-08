//! Hi-Z (Hierarchical Z) pyramid builder.
//!
//! Two-phase build each frame — fully GPU-driven, O(1) CPU:
//!
//!  Phase 1 — Depth copy  (texture -> buffer -> texture)
//!    Copies the exact previous-frame `Depth32Float` render-target bytes into
//!    mip-0 of the R32Float HiZ textures. The intermediate buffer avoids depth
//!    operations in translated shaders. Downlevel GL adapters that cannot copy
//!    depth textures receive a conservative far-depth pyramid instead, keeping
//!    rendering correct while disabling dynamic occlusion and SSR.
//!
//!  Phase 2 — Mip chain  (hiz_build.wgsl, ~log2(max_dim) dispatches)
//!    Downsamples using MAX-reduction so each texel stores the farthest depth
//!    in its 2x2 footprint — "conservative Hi-Z".
//!
//! The finished pyramid is consumed NEXT FRAME by OcclusionCullPass (temporal
//! approach: frame N-1 depth tests visibility of frame N geometry).
//!
//! The HiZ texture is owned by the render graph and declared via `declare_resources`.
//! The pass recreates mip views and bind groups lazily during `execute()` from the
//! graph-owned texture accessed via `ctx.resource_pool`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::graph::{ResourceBuilder, ResourceSize};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};
use helio_core::ResourceRegistry;

/// Marker opting a shader into [`HIZ`]. Must appear in the source.
pub const HIZ_MARKER: &str = "//!use helio_hiz";

/// Hi-Z screen-space ray marching, shared by SSR and water — as a
/// `helio-core` shader snippet (see `helio_core::shader::ShaderSnippet`).
///
/// Separate from `helio_core::shader::PRELUDE` because it is only wanted by
/// the passes that march the pyramid, and prepending it everywhere would
/// push every other shader's diagnostics further out of alignment for
/// nothing.
pub const HIZ: &str = include_str!("../shaders/hiz_trace.wgsl");

/// [`helio_core::shader::ShaderSnippet`] for [`HIZ`]. Passed explicitly to
/// `helio_core::shader::resolve_with`/`module_with` by any shader opting in
/// via [`HIZ_MARKER`] — `helio-core` itself never names this snippet.
pub const HIZ_SNIPPET: helio_core::shader::ShaderSnippet =
    helio_core::wgsl_snippet!(HIZ_MARKER, "../shaders/hiz_trace.wgsl");
const WORKGROUP_SIZE: u32 = 8;
const MAX_MIP_LEVELS: u32 = 12;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HiZUniforms {
    src_size: [u32; 2],
    dst_size: [u32; 2],
}

pub struct HiZBuildPass {
    // Mip-chain downsampling pipeline
    mip_pipeline: wgpu::ComputePipeline,
    mip_bgl: wgpu::BindGroupLayout,
    mip_bind_groups: Vec<wgpu::BindGroup>,
    mip_uniforms: Vec<wgpu::Buffer>,
    mip_dispatch_groups: Vec<(u32, u32)>,

    // Exact GPU copy staging (Depth32Float -> bytes -> R32Float mip-0).
    depth_copy_buffer: wgpu::Buffer,
    depth_copy_bytes_per_row: u32,
    depth_copy_supported: bool,

    // Safe fallback for downlevel adapters without depth texture/buffer copies.
    // Both pyramids are filled with far depth, conservatively disabling dynamic
    // occlusion and SSR instead of rejecting the entire renderer.
    fallback_pipeline: wgpu::ComputePipeline,
    fallback_bgl: wgpu::BindGroupLayout,
    fallback_bind_group: Option<wgpu::BindGroup>,

    // ── Min pyramid ("hiz_min") ─────────────────────────────────────────────
    // Same policy and same dimensions as the max chain above, opposite reduction.
    // Shares this pass so pyramid sizing/mip-count/resize logic lives in exactly
    // one place; a consumer that needs min-depth reads the resource rather than
    // growing its own builder. Mip 0 is identical to the max chain's (a plain
    // depth copy — min and max of one texel are the same); only levels 1+
    // diverge.
    min_mip_pipeline: wgpu::ComputePipeline,
    min_mip_bind_groups: Vec<wgpu::BindGroup>,
    min_mip_views: Vec<wgpu::TextureView>,

    // HiZ sampler (always owned by this pass)
    pub hiz_sampler: Arc<wgpu::Sampler>,
    // Per-mip views created from graph-owned texture; rebuilt on resize
    mip_views: Vec<wgpu::TextureView>,
    width: u32,
    height: u32,

    /// Which depth the max pyramid holds and which depth the next copy reads.
    max_source: PyramidSource,
}

/// What a frame's depth was drawn from: everything the CPU knows that decides
/// which geometry reached it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct DepthInputs {
    camera_generation: u64,
    /// `SceneBufferProjection::content_signature`.
    scene_signature: u64,
    /// `helio_core::resource_keys::depth_draw_signature`, 0 when no pass
    /// publishes one.
    draw_signature: u64,
}

/// Reuse bookkeeping for the max pyramid.
///
/// The pyramid is built from the depth texture as it stands when HiZBuild
/// runs, which is ahead of this frame's geometry: it holds the PREVIOUS
/// frame's depth. So a pyramid is tagged with the inputs of the frame that
/// drew that depth, not the frame that copied it, and is reused only while the
/// current frame's inputs equal that tag -- i.e. only while the depth it came
/// from is exactly what this frame would draw again.
///
/// Tagging with the copying frame's inputs instead (the old behaviour) froze
/// any pyramid built just after a change: with the camera at rest, the
/// pyramid built on frame 0 from a depth buffer nothing had drawn into yet
/// was reused forever and occlusion culling hid every object.
#[derive(Clone, Copy, Debug, Default)]
struct PyramidSource {
    /// Inputs of the depth the pyramid was built from; `None` when unknown
    /// (never built, or built from a depth no tracked frame drew).
    built_from: Option<DepthInputs>,
    /// Inputs of the frame that last ran: what the depth texture holds now.
    depth_holds: Option<DepthInputs>,
}

impl PyramidSource {
    /// Starts a frame drawn from `current`. Returns whether the pyramid must
    /// be rebuilt from the depth texture this frame.
    fn begin_frame(&mut self, current: DepthInputs) -> bool {
        let rebuild = self.built_from != Some(current);
        if rebuild {
            self.built_from = self.depth_holds;
        }
        self.depth_holds = Some(current);
        rebuild
    }

    /// The depth texture was reallocated: neither it nor the pyramid holds
    /// anything a tracked frame drew.
    fn invalidate(&mut self) {
        *self = Self::default();
    }
}

impl HiZBuildPass {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, width: u32, height: u32) -> Self {
        let hiz_sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("HiZ Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }));

        // Phase 2: mip-chain downsampling pipeline
        let mip_shader = helio_core::shader::module(device, "HiZ Build Shader", helio_core::include_wgsl!("../shaders/hiz_build.wgsl"));

        let mip_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HiZ Mip BGL"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        let mip_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HiZ Mip PL"),
            bind_group_layouts: &[Some(&mip_bgl)],
            immediate_size: 0,
        });

        let mip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HiZ Mip Pipeline"),
            layout: Some(&mip_pl),
            module: &mip_shader,
            entry_point: Some("main_max"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Same module and layout as the max chain — only the reduction differs.
        let min_mip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HiZ Min Mip Pipeline"),
            layout: Some(&mip_pl),
            module: &mip_shader,
            entry_point: Some("main_min"),
            compilation_options: Default::default(),
            cache: None,
        });

        let (depth_copy_buffer, depth_copy_bytes_per_row) =
            create_depth_copy_buffer(device, width, height);
        let depth_copy_supported = depth_texture_buffer_copies_supported(device);
        if !depth_copy_supported {
            log::warn!(
                "HiZ depth copies are unavailable on this downlevel adapter; dynamic occlusion \
                 culling and SSR use a conservative far-depth fallback"
            );
        }
        let fallback_shader = helio_core::shader::module(device, "HiZ Far Depth Fallback Shader", helio_core::include_wgsl!("../shaders/hiz_far_depth_fallback.wgsl"));
        let fallback_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HiZ Far Depth Fallback BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::R32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });
        let fallback_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HiZ Far Depth Fallback PL"),
            bind_group_layouts: &[Some(&fallback_bgl)],
            immediate_size: 0,
        });
        let fallback_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HiZ Far Depth Fallback Pipeline"),
            layout: Some(&fallback_layout),
            module: &fallback_shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Mip uniforms and bind groups are built lazily from the graph-owned texture.
        // Mip uniforms and dispatch groups are built once here (width/height-dependent).
        // Both chains are the same size, so both share these.
        let (mip_uniforms, mip_dispatch_groups) =
            build_mip_uniforms(device, queue, width, height, "HiZ Mip Uniform");

        // mip_views and mip_bind_groups need the wgpu::Texture handle which is
        // available via ctx.resource_pool in execute().
        Self {
            mip_pipeline,
            mip_bgl,
            mip_bind_groups: Vec::new(),
            mip_uniforms,
            mip_dispatch_groups,
            depth_copy_buffer,
            depth_copy_bytes_per_row,
            depth_copy_supported,
            fallback_pipeline,
            fallback_bgl,
            fallback_bind_group: None,
            min_mip_pipeline,
            min_mip_bind_groups: Vec::new(),
            min_mip_views: Vec::new(),
            hiz_sampler,
            mip_views: Vec::new(),
            width,
            height,
            max_source: PyramidSource::default(),
        }
    }

    /// Builds the min-depth pyramid consumed by SsrPass after mip 0 is seeded.
    ///
    /// Assumes `min_mip_views` / `min_mip_bind_groups` are already populated.
    fn build_min_pyramid(&mut self, ctx: &mut PassContext) {
        let encoder = unsafe { &mut *ctx.encoder_ptr };

        // Levels 1+ via MIN-reduction.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ Min MipChain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.min_mip_pipeline);
            for (bg, &(wg_x, wg_y)) in self
                .min_mip_bind_groups
                .iter()
                .zip(self.mip_dispatch_groups.iter())
            {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
    }
}

fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}

#[cfg(all(
    not(target_arch = "wasm32"),
    any(target_os = "windows", target_os = "linux", target_os = "android")
))]
fn depth_texture_buffer_copies_supported(device: &wgpu::Device) -> bool {
    // Native Vulkan and DX12 implementations expose the WebGPU depth-copy
    // contract. WGPU's GL compatibility backend may omit it, and Device does
    // not retain the Adapter properties needed to query the downlevel flag
    // directly. Backend identity is therefore the exact capability boundary
    // available to an externally supplied Device on GLES-capable targets.
    unsafe { device.as_hal::<wgpu::hal::api::Gles>() }.is_none()
}

#[cfg(all(
    not(target_arch = "wasm32"),
    not(any(target_os = "windows", target_os = "linux", target_os = "android"))
))]
fn depth_texture_buffer_copies_supported(_device: &wgpu::Device) -> bool {
    // WGPU does not compile its GLES HAL on these native targets. Their native
    // backends implement depth texture/buffer copies, so the exact copy path is
    // available without a backend identity probe.
    true
}

#[cfg(target_arch = "wasm32")]
fn depth_texture_buffer_copies_supported(_device: &wgpu::Device) -> bool {
    true
}

fn create_depth_copy_buffer(device: &wgpu::Device, width: u32, height: u32) -> (wgpu::Buffer, u32) {
    let unpadded_bytes_per_row = width
        .max(1)
        .saturating_mul(std::mem::size_of::<f32>() as u32);
    let bytes_per_row = unpadded_bytes_per_row
        .div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
        .saturating_mul(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let size = u64::from(bytes_per_row).saturating_mul(u64::from(height.max(1)));
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HiZ Depth Copy Buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    (buffer, bytes_per_row)
}

/// Per-level src/dst sizes and dispatch extents for a pyramid of `width`x`height`.
fn build_mip_uniforms(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    label: &str,
) -> (Vec<wgpu::Buffer>, Vec<(u32, u32)>) {
    let mip_count = mip_levels(width, height).min(MAX_MIP_LEVELS);
    let levels = mip_count.saturating_sub(1);
    let mut uniforms = Vec::with_capacity(levels as usize);
    let mut dispatch_groups = Vec::with_capacity(levels as usize);

    for mip in 0..levels {
        let src_w = (width >> mip).max(1);
        let src_h = (height >> mip).max(1);
        let dst_w = (width >> (mip + 1)).max(1);
        let dst_h = (height >> (mip + 1)).max(1);
        let u = HiZUniforms {
            src_size: [src_w, src_h],
            dst_size: [dst_w, dst_h],
        };
        // `create_buffer_init` uses mappedAtCreation on WebGPU. Dawn can
        // reject that synchronous mapping under browser resource pressure,
        // so initialize through the queue instead.
        let ub = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<HiZUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&ub, 0, bytemuck::bytes_of(&u));
        uniforms.push(ub);
        dispatch_groups.push((
            dst_w.div_ceil(WORKGROUP_SIZE),
            dst_h.div_ceil(WORKGROUP_SIZE),
        ));
    }

    (uniforms, dispatch_groups)
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &'static str {
        "HiZBuild"
    }

    fn reads(&self) -> &'static [&'static str] {
        // The signature orders this pass after every depth-draw producer, so
        // `execute` sees this frame's value.
        &["depth", helio_core::resource_keys::DEPTH_DRAW_SIGNATURE]
    }

    fn writes(&self) -> &'static [&'static str] {
        &[
            "hiz",
            "hiz_min",
            "hiz_sampler",
        ]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // Both are R32Float, which the resource pool automatically gives a full
        // mip chain (see graph::resource_lifetime).
        builder.write_color_raw(
            "hiz",
            wgpu::TextureFormat::R32Float,
            ResourceSize::MatchSurface,
        );
        builder
            .with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST);
        builder.write_color_raw(
            "hiz_min",
            wgpu::TextureFormat::R32Float,
            ResourceSize::MatchSurface,
        );
        builder
            .with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST);
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // Graph textures are re-allocated by the pool. Clear lazy views/bind groups
        // so they are rebuilt from the new graph-owned texture in execute().
        self.mip_views.clear();
        self.mip_bind_groups.clear();
        self.min_mip_views.clear();
        self.min_mip_bind_groups.clear();
        self.fallback_bind_group = None;
        self.max_source.invalidate();
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if ctx.resize && (self.width != ctx.width || self.height != ctx.height) {
            self.width = ctx.width.max(1);
            self.height = ctx.height.max(1);
            (self.depth_copy_buffer, self.depth_copy_bytes_per_row) =
                create_depth_copy_buffer(ctx.device, self.width, self.height);
            (self.mip_uniforms, self.mip_dispatch_groups) = build_mip_uniforms(
                ctx.device,
                ctx.queue,
                self.width,
                self.height,
                "HiZ Mip Uniform",
            );
            self.max_source.invalidate();
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── Lazy init: build mip views and bind groups from graph-owned texture ──
        if self.mip_views.is_empty() {
            let hiz_texture = ctx
                .resource_pool
                .get_texture("hiz")
                .expect("HiZ texture 'hiz' must be declared as a graph resource");
            let mip_count = mip_levels(self.width, self.height).min(MAX_MIP_LEVELS);

            // Create per-mip single-level views
            let mut mip_views = Vec::with_capacity(mip_count as usize);
            for mip in 0..mip_count {
                mip_views.push(hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("HiZ Mip View"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                }));
            }
            self.mip_views = mip_views;

            // Build mip bind groups from the existing uniforms + new views
            let mut mip_bind_groups = Vec::with_capacity((mip_count.saturating_sub(1)) as usize);
            for mip in 0..(mip_count.saturating_sub(1)) {
                let bg = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HiZ Mip BG"),
                    layout: &self.mip_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.mip_uniforms[mip as usize].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.mip_views[mip as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &self.mip_views[(mip + 1) as usize],
                            ),
                        },
                    ],
                });
                mip_bind_groups.push(bg);
            }
            self.mip_bind_groups = mip_bind_groups;
        }

        // ── Lazy init: min pyramid views + bind groups ───────────────────────
        if self.min_mip_views.is_empty() {
            let tex = ctx
                .resource_pool
                .get_texture("hiz_min")
                .expect("HiZ texture 'hiz_min' must be declared as a graph resource");
            let mip_count = mip_levels(self.width, self.height).min(MAX_MIP_LEVELS);

            let mut views = Vec::with_capacity(mip_count as usize);
            for mip in 0..mip_count {
                views.push(tex.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("HiZ Min Mip View"),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                }));
            }
            self.min_mip_views = views;

            let mut bgs = Vec::with_capacity((mip_count.saturating_sub(1)) as usize);
            for mip in 0..(mip_count.saturating_sub(1)) {
                bgs.push(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HiZ Min Mip BG"),
                    layout: &self.mip_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            // Same dimensions as the max chain, so the same
                            // per-level src/dst sizes apply.
                            binding: 0,
                            resource: self.mip_uniforms[mip as usize].as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(
                                &self.min_mip_views[mip as usize],
                            ),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(
                                &self.min_mip_views[(mip + 1) as usize],
                            ),
                        },
                    ],
                }));
            }
            self.min_mip_bind_groups = bgs;
        }

        let hiz_texture = ctx
            .resource_pool
            .get_texture("hiz")
            .expect("HiZ texture 'hiz' must be declared as a graph resource");
        let hiz_min_texture = ctx
            .resource_pool
            .get_texture("hiz_min")
            .expect("HiZ texture 'hiz_min' must be declared as a graph resource");
        let copy_extent = wgpu::Extent3d {
            width: self.width.max(1),
            height: self.height.max(1),
            depth_or_array_layers: 1,
        };
        let copy_layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(self.depth_copy_bytes_per_row),
            rows_per_image: Some(self.height.max(1)),
        };
        let encoder = unsafe { &mut *ctx.encoder_ptr };

        // The min pyramid is optional: only SSR and WaterSim's reflections
        // read it, and they declare that before the frame executes. The depth
        // copy feeds both pyramids, so it is needed only when one of them is
        // rebuilt this frame.
        let min_wanted = helio_core::is_demanded(ctx.registry, "hiz_min");
        let max_rebuild = self.max_source.begin_frame(DepthInputs {
            camera_generation: ctx.camera_generation,
            scene_signature: ctx.scene_buffers.content_signature(),
            draw_signature: ctx
                .registry
                .get(helio_core::resource_keys::depth_draw_signature())
                .unwrap_or(0),
        });

        if self.depth_copy_supported && (min_wanted || max_rebuild) {
            let depth_texture = ctx
                .registry
                .get::<&wgpu::Texture>(helio_core::ResourceKey::new("depth_texture"))
                .expect("Renderer must publish the active depth texture for HiZ");
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: depth_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::DepthOnly,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.depth_copy_buffer,
                    layout: copy_layout,
                },
                copy_extent,
            );
            if min_wanted {
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &self.depth_copy_buffer,
                        layout: copy_layout,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: hiz_min_texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    copy_extent,
                );
            }
        } else if !self.depth_copy_supported {
            if self.fallback_bind_group.is_none() {
                self.fallback_bind_group =
                    Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("HiZ Far Depth Fallback BG"),
                        layout: &self.fallback_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::TextureView(&self.mip_views[0]),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: wgpu::BindingResource::TextureView(
                                    &self.min_mip_views[0],
                                ),
                            },
                        ],
                    }));
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ Far Depth Fallback"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.fallback_pipeline);
            pass.set_bind_group(0, self.fallback_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(
                self.width.div_ceil(WORKGROUP_SIZE),
                self.height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        // ── Min pyramid: rebuilt every frame ─────────────────────────────────
        // Deliberately outside the max pyramid's reuse early-out below. That optimization
        // is sound for the max chain because its consumer (occlusion culling) is
        // temporal by design and tolerates a frame-stale pyramid. SSR is not: it
        // reflects the *current* frame, and a static camera does not imply static
        // depth — anything that moves while the camera holds still would otherwise
        // reflect a frozen pyramid.
        if min_wanted {
            self.build_min_pyramid(ctx);
        }

        // ── HiZ reuse: skip the rebuild while the pyramid already holds the
        // depth this frame would draw (see `PyramidSource`) ──────────────────
        if !max_rebuild {
            return Ok(());
        }

        if self.depth_copy_supported {
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.depth_copy_buffer,
                    layout: copy_layout,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: hiz_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                copy_extent,
            );
        }

        // Phase 2: build the remaining mip levels via MAX-reduction
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ MipChain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mip_pipeline);
            for (bg, &(wg_x, wg_y)) in self
                .mip_bind_groups
                .iter()
                .zip(self.mip_dispatch_groups.iter())
            {
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
        Ok(())
    }

    fn publish<'a>(&self, frame: &mut ResourceRegistry<'a>) {
        // SAFETY: every borrow below is extended out of `self`'s owned `Arc`
        // fields, which are pass-lifetime (owned by the RenderGraph across
        // many frames), never frame-scoped -- `'a` is always shorter than
        // these fields' real lifetime. `publish(&self, ..)` (not `&'a self`)
        // cannot express that relationship, so the borrow checker sees an
        // unrelated, shorter lifetime here instead; matches
        // `ResourceRegistry::write_texture_binding`'s own transmute for the
        // same reason.
        //
        // The graph routes "hiz" texture view via pre_pass_actions before execute().
        // We only need to publish the sampler (not owned by the graph).
        let hiz_sampler: &'a wgpu::Sampler = unsafe { std::mem::transmute(&*self.hiz_sampler) };
        frame.write(helio_core::ResourceKey::new("hiz_sampler"), hiz_sampler, "HiZBuild");
    }
}

#[cfg(test)]
mod tests {
    use super::{DepthInputs, PyramidSource};

    fn inputs(camera_generation: u64, scene_signature: u64, draw_signature: u64) -> DepthInputs {
        DepthInputs { camera_generation, scene_signature, draw_signature }
    }

    /// Runs `frames` frames at `current`, returning which of them rebuilt.
    fn run(source: &mut PyramidSource, current: DepthInputs, frames: usize) -> Vec<bool> {
        (0..frames).map(|_| source.begin_frame(current)).collect()
    }

    #[test]
    fn static_view_rebuilds_until_the_pyramid_holds_its_own_depth() {
        let mut source = PyramidSource::default();
        // Frame 0 copies a depth nothing drew; frame 1 copies frame 0's.
        assert_eq!(run(&mut source, inputs(1, 7, 3), 4), [true, true, false, false]);
    }

    /// Objects present before the first frame, camera never moves: the draw
    /// counts reach the CPU frames after the upload, so the first frames draw
    /// nothing. The pyramid built from those empty depths must not outlive
    /// the frame whose draws finally reach depth.
    #[test]
    fn late_draws_under_a_static_camera_rebuild_the_pyramid() {
        let mut source = PyramidSource::default();
        let nothing_drawn = inputs(1, 7, 0);
        let drawn = inputs(1, 7, 42);
        assert_eq!(run(&mut source, nothing_drawn, 5), [true, true, false, false, false]);
        // First frame drawing the scene still copies the previous, empty
        // depth; the next copies the drawn one; then it is reused.
        assert_eq!(run(&mut source, drawn, 5), [true, true, false, false, false]);
    }

    /// A change is seen by the frame that copies the depth drawn BEFORE it,
    /// so that rebuild alone is stale: the pyramid must rebuild once more
    /// from the first depth drawn after the change.
    #[test]
    fn a_change_rebuilds_from_the_first_depth_drawn_after_it() {
        for changed in [inputs(2, 7, 42), inputs(1, 8, 42), inputs(1, 7, 43)] {
            let mut source = PyramidSource::default();
            run(&mut source, inputs(1, 7, 42), 4);
            assert_eq!(run(&mut source, changed, 4), [true, true, false, false], "{changed:?}");
        }
    }

    #[test]
    fn a_moving_camera_rebuilds_every_frame() {
        let mut source = PyramidSource::default();
        for camera in 0..6 {
            assert!(source.begin_frame(inputs(camera, 7, 42)));
        }
    }

    #[test]
    fn invalidation_forgets_the_depth_texture() {
        let mut source = PyramidSource::default();
        run(&mut source, inputs(1, 7, 42), 4);
        source.invalidate();
        assert_eq!(run(&mut source, inputs(1, 7, 42), 3), [true, true, false]);
    }
}
