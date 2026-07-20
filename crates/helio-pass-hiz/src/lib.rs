//! Hi-Z (Hierarchical Z) pyramid builder.
//!
//! Two-phase build each frame — fully GPU-driven, O(1) CPU:
//!
//!  Phase 1 — Depth copy  (hiz_depth_copy.wgsl, one dispatch)
//!    Reads the `Depth32Float` render-attachment texture written by DepthPrepassPass
//!    and writes each depth value into mip-0 of the R32Float HiZ texture.
//!    This is necessary because Depth32Float cannot be bound as a storage texture.
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
use helio_core::{FrameResources, PassContext, PrepareContext, RenderPass, Result as HelioResult};
const WORKGROUP_SIZE: u32 = 8;
const MAX_MIP_LEVELS: u32 = 12;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HiZUniforms {
    src_size: [u32; 2],
    dst_size: [u32; 2],
}

/// Metadata for loaded static HiZ data.
#[derive(Clone, Debug)]
pub struct StaticHizMetadata {
    pub grid_resolution: [u32; 3],
    pub world_bounds_min: [f32; 3],
    pub world_bounds_max: [f32; 3],
    pub mip_count: u32,
}

pub struct HiZBuildPass {
    // Mip-chain downsampling pipeline
    mip_pipeline: wgpu::ComputePipeline,
    mip_bgl: wgpu::BindGroupLayout,
    mip_bind_groups: Vec<wgpu::BindGroup>,
    mip_uniforms: Vec<wgpu::Buffer>,
    mip_dispatch_groups: Vec<(u32, u32)>,

    // Depth-copy pipeline (Depth32Float -> R32Float mip-0)
    copy_pipeline: wgpu::ComputePipeline,
    copy_bgl: wgpu::BindGroupLayout,
    copy_bind_group: Option<wgpu::BindGroup>,
    copy_bind_group_key: Option<usize>,

    // ── Min pyramid ("hiz_min") ─────────────────────────────────────────────
    // Same policy and same dimensions as the max chain above, opposite reduction.
    // Shares this pass so pyramid sizing/mip-count/resize logic lives in exactly
    // one place; a consumer that needs min-depth reads the resource rather than
    // growing its own builder. Mip 0 is identical to the max chain's (a plain
    // depth copy — min and max of one texel are the same), so it reuses
    // `copy_pipeline`; only levels 1+ diverge.
    min_mip_pipeline: wgpu::ComputePipeline,
    min_mip_bind_groups: Vec<wgpu::BindGroup>,
    min_mip_views: Vec<wgpu::TextureView>,
    min_copy_bind_group: Option<wgpu::BindGroup>,
    min_copy_bind_group_key: Option<usize>,

    // HiZ sampler (always owned by this pass)
    pub hiz_sampler: Arc<wgpu::Sampler>,
    // Per-mip views created from graph-owned texture; rebuilt on resize
    mip_views: Vec<wgpu::TextureView>,
    width: u32,
    height: u32,

    // Camera tracking for HiZ reuse optimization (skip rebuild if camera static)
    prev_camera_generation: u64,
    /// Whether this is the first frame (forces a full rebuild regardless of generation).
    first_frame: bool,

    // Static HiZ: Pre-baked voxel-based occlusion for static geometry
    /// Pre-baked 3D voxel occlusion grid (camera-independent).
    /// Contains 6 layers (±X, ±Y, ±Z directions) with hierarchical mips.
    pub static_hiz_texture: Option<Arc<wgpu::Texture>>,
    pub static_hiz_view: Option<Arc<wgpu::TextureView>>,
    pub static_hiz_sampler: Option<Arc<wgpu::Sampler>>,
    static_hiz_metadata: Option<StaticHizMetadata>,
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
        let mip_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HiZ Build Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hiz_build.wgsl").into()),
        });

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

        // Phase 1: depth-copy pipeline
        let copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HiZ Depth Copy Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/hiz_depth_copy.wgsl").into(),
            ),
        });

        let copy_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HiZ Copy BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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

        let copy_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("HiZ Copy PL"),
            bind_group_layouts: &[Some(&copy_bgl)],
            immediate_size: 0,
        });

        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HiZ Copy Pipeline"),
            layout: Some(&copy_pl),
            module: &copy_shader,
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
            copy_pipeline,
            copy_bgl,
            copy_bind_group: None,
            copy_bind_group_key: None,
            min_mip_pipeline,
            min_mip_bind_groups: Vec::new(),
            min_mip_views: Vec::new(),
            min_copy_bind_group: None,
            min_copy_bind_group_key: None,
            hiz_sampler,
            mip_views: Vec::new(),
            width,
            height,
            prev_camera_generation: 0,
            first_frame: true,
            static_hiz_texture: None,
            static_hiz_view: None,
            static_hiz_sampler: None,
            static_hiz_metadata: None,
        }
    }

    /// Returns metadata about the loaded static HiZ voxel grid, or `None` if no
    /// static HiZ data has been loaded.
    pub fn static_hiz_metadata(&self) -> Option<&StaticHizMetadata> {
        self.static_hiz_metadata.as_ref()
    }

    /// Seeds and builds the min-depth pyramid consumed by SsrPass.
    ///
    /// Assumes `min_mip_views` / `min_mip_bind_groups` are already populated.
    fn build_min_pyramid(&mut self, ctx: &mut PassContext) {
        let depth_key = ctx.depth as *const _ as usize;
        if self.min_copy_bind_group_key != Some(depth_key) {
            self.min_copy_bind_group =
                Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("HiZ Min Copy BG"),
                    layout: &self.copy_bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(ctx.depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.min_mip_views[0]),
                        },
                    ],
                }));
            self.min_copy_bind_group_key = Some(depth_key);
        }

        let encoder = unsafe { &mut *ctx.encoder_ptr };

        // Seed mip 0. Same shader as the max chain: at 1:1 there is nothing to
        // reduce, so the two pyramids only diverge from level 1 on.
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ Min Seed"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, self.min_copy_bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(
                self.width.div_ceil(WORKGROUP_SIZE),
                self.height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }

        // Remaining levels via MIN-reduction.
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

    /// Load pre-baked static HiZ data from Nebula baker.
    ///
    /// This creates a 3D texture containing omnidirectional voxel occlusion data
    /// for static geometry. The texture has 6 layers (±X, ±Y, ±Z) with hierarchical
    /// mips, allowing camera-independent occlusion queries at runtime.
    ///
    /// # Arguments
    /// * `device` - GPU device for texture creation
    /// * `queue` - Command queue for data upload
    /// * `grid_resolution` - Voxel grid dimensions [width, height, depth]
    /// * `world_bounds_min` - AABB min corner in world space
    /// * `world_bounds_max` - AABB max corner in world space
    /// * `mip_count` - Number of mip levels in the hierarchy
    /// * `texels` - Raw R32F texel data (all mips, all layers)
    pub fn load_static_hiz(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        grid_resolution: [u32; 3],
        world_bounds_min: [f32; 3],
        world_bounds_max: [f32; 3],
        mip_count: u32,
        texels: &[u8],
    ) {
        // Create 3D texture (6 layers packed into Z dimension)
        let depth_with_layers = grid_resolution[2] * 6;

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Static HiZ Voxel Grid"),
            size: wgpu::Extent3d {
                width: grid_resolution[0],
                height: grid_resolution[1],
                depth_or_array_layers: depth_with_layers,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload texel data for each mip level
        let mut offset = 0;
        for mip in 0..mip_count {
            let mip_w = (grid_resolution[0] >> mip).max(1);
            let mip_h = (grid_resolution[1] >> mip).max(1);
            let mip_d = (grid_resolution[2] >> mip).max(1);
            let mip_depth_with_layers = mip_d * 6;

            let bytes_per_texel = 4; // R32Float
            let row_bytes = mip_w * bytes_per_texel;
            let layer_bytes = row_bytes * mip_h;
            let mip_bytes = (layer_bytes * mip_depth_with_layers) as usize;

            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: mip,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &texels[offset..offset + mip_bytes],
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row_bytes),
                    rows_per_image: Some(mip_h),
                },
                wgpu::Extent3d {
                    width: mip_w,
                    height: mip_h,
                    depth_or_array_layers: mip_depth_with_layers,
                },
            );

            offset += mip_bytes;
        }

        let view = Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("Static HiZ View"),
            format: Some(wgpu::TextureFormat::R32Float),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        }));

        let sampler = Arc::new(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Static HiZ Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        }));

        self.static_hiz_texture = Some(Arc::new(texture));
        self.static_hiz_view = Some(view);
        self.static_hiz_sampler = Some(sampler);
        self.static_hiz_metadata = Some(StaticHizMetadata {
            grid_resolution,
            world_bounds_min,
            world_bounds_max,
            mip_count,
        });

        log::info!(
            "Loaded static HiZ: {}x{}x{} voxels, {} mips, bounds [{:?} to {:?}]",
            grid_resolution[0],
            grid_resolution[1],
            grid_resolution[2],
            mip_count,
            world_bounds_min,
            world_bounds_max
        );
    }
}

fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
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
        dispatch_groups.push((dst_w.div_ceil(WORKGROUP_SIZE), dst_h.div_ceil(WORKGROUP_SIZE)));
    }

    (uniforms, dispatch_groups)
}

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &'static str {
        "HiZBuild"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["depth"]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["hiz", "hiz_min", "hiz_sampler", "static_hiz", "static_hiz_sampler"]
    }

    fn declare_resources(&self, builder: &mut ResourceBuilder) {
        // Both are R32Float, which the resource pool automatically gives a full
        // mip chain (see graph::resource_lifetime).
        builder.write_color_raw("hiz", wgpu::TextureFormat::R32Float, ResourceSize::MatchSurface);
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
        builder.write_color_raw("hiz_min", wgpu::TextureFormat::R32Float, ResourceSize::MatchSurface);
        builder.with_extra_usage(wgpu::TextureUsages::STORAGE_BINDING);
    }

    fn on_resize(&mut self, _device: &wgpu::Device, _width: u32, _height: u32) {
        // Graph textures are re-allocated by the pool. Clear lazy views/bind groups
        // so they are rebuilt from the new graph-owned texture in execute().
        self.mip_views.clear();
        self.mip_bind_groups.clear();
        self.copy_bind_group = None;
        self.copy_bind_group_key = None;
        self.min_mip_views.clear();
        self.min_mip_bind_groups.clear();
        self.min_copy_bind_group = None;
        self.min_copy_bind_group_key = None;
        self.first_frame = true;
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, _ctx: &PrepareContext) -> HelioResult<()> {
        // Static HiZ mip uniforms are initialized once in `new()` and do not
        // need to be re-uploaded every frame unless the pass is recreated.
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── Lazy init: build mip views and bind groups from graph-owned texture ──
        if self.mip_views.is_empty() {
            let hiz_texture = ctx.resource_pool.get_texture("hiz")
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
            let tex = ctx.resource_pool.get_texture("hiz_min")
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

        // ── Min pyramid: rebuilt every frame ─────────────────────────────────
        // Deliberately outside the camera-static early-out below. That optimization
        // is sound for the max chain because its consumer (occlusion culling) is
        // temporal by design and tolerates a frame-stale pyramid. SSR is not: it
        // reflects the *current* frame, and a static camera does not imply static
        // depth — anything that moves while the camera holds still would otherwise
        // reflect a frozen pyramid.
        self.build_min_pyramid(ctx);

        // ── HiZ Reuse optimization: skip rebuild if camera static ─────────────
        let camera_gen = ctx.scene.camera_generation;
        let resolution_changed = false;

        if !self.first_frame && camera_gen == self.prev_camera_generation && self.copy_bind_group.is_some() && !resolution_changed {
            return Ok(());
        }

        self.first_frame = false;
        self.prev_camera_generation = camera_gen;

        // Rebuild depth-copy bind group if the depth texture view pointer changed
        let depth_key = ctx.depth as *const _ as usize;
        if self.copy_bind_group_key != Some(depth_key) {
            self.copy_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HiZ Copy BG"),
                layout: &self.copy_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(ctx.depth),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&self.mip_views[0]),
                    },
                ],
            }));
            self.copy_bind_group_key = Some(depth_key);
        }

        // Phase 1: copy depth -> HiZ mip-0
        {
            let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ DepthCopy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, self.copy_bind_group.as_ref().unwrap(), &[]);
            let wg_x = self.width.div_ceil(WORKGROUP_SIZE);
            let wg_y = self.height.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Phase 2: build the remaining mip levels via MAX-reduction
        {
            let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
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

    fn publish<'a>(&'a self, frame: &mut FrameResources<'a>) {
        // The graph routes "hiz" texture view via pre_pass_actions before execute().
        // We only need to publish the sampler (not owned by the graph).
        frame.hiz_sampler.write(&*self.hiz_sampler, "HiZBuild");

        // Expose static HiZ if loaded
        if let Some(ref view) = self.static_hiz_view {
            frame.static_hiz.write(&**view, "HiZBuild");
        }
        if let Some(ref sampler) = self.static_hiz_sampler {
            frame.static_hiz_sampler.write(&**sampler, "HiZBuild");
        }
    }
}
