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
//! `hiz_view` and `hiz_sampler` are Arc-wrapped so OcclusionCullPass can hold
//! its own reference to the persistent texture without lifetime issues.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_v3::{FrameResources, PassContext, PrepareContext, RenderPass, Result as HelioResult};

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
struct StaticHizMetadata {
    grid_resolution: [u32; 3],
    world_bounds_min: [f32; 3],
    world_bounds_max: [f32; 3],
    mip_count: u32,
}

pub struct HiZBuildPass {
    // Mip-chain downsampling pipeline
    mip_pipeline: wgpu::ComputePipeline,
    #[allow(dead_code)]
    mip_bgl: wgpu::BindGroupLayout,
    mip_bind_groups: Vec<wgpu::BindGroup>,
    mip_uniforms: Vec<wgpu::Buffer>,

    // Depth-copy pipeline (Depth32Float -> R32Float mip-0)
    copy_pipeline: wgpu::ComputePipeline,
    copy_bgl: wgpu::BindGroupLayout,
    copy_bind_group: Option<wgpu::BindGroup>,
    copy_bind_group_key: Option<usize>,

    // Shared HiZ texture resources (Arc so OcclusionCullPass can co-own)
    pub hiz_view: Arc<wgpu::TextureView>,
    pub hiz_sampler: Arc<wgpu::Sampler>,
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
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let mip_count = mip_levels(width, height).min(MAX_MIP_LEVELS);

        // HiZ texture
        let hiz_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("HiZ Texture"),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        let hiz_view = Arc::new(hiz_texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("HiZ Full View"),
            ..Default::default()
        }));

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

        // Per-mip single-level views
        let mut mip_views = Vec::with_capacity(mip_count as usize);
        for mip in 0..mip_count {
            mip_views.push(hiz_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("HiZ Mip View"),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            }));
        }

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
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Build per-mip bind groups for mip[i]->mip[i+1] downsampling
        let mut mip_bind_groups = Vec::with_capacity((mip_count - 1) as usize);
        let mut mip_uniforms = Vec::with_capacity((mip_count - 1) as usize);
        for mip in 0..(mip_count - 1) {
            let ub = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("HiZ Mip Uniform"),
                size: std::mem::size_of::<HiZUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HiZ Mip BG"),
                layout: &mip_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&mip_views[mip as usize]),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(
                            &mip_views[(mip + 1) as usize],
                        ),
                    },
                ],
            });
            mip_uniforms.push(ub);
            mip_bind_groups.push(bg);
        }

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

        Self {
            mip_pipeline,
            mip_bgl,
            mip_bind_groups,
            mip_uniforms,
            copy_pipeline,
            copy_bgl,
            copy_bind_group: None,
            copy_bind_group_key: None,
            hiz_view,
            hiz_sampler,
            mip_views,
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

impl RenderPass for HiZBuildPass {
    fn name(&self) -> &'static str {
        "HiZBuild"
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let w = self.width;
        let h = self.height;
        for (mip, ub) in self.mip_uniforms.iter().enumerate() {
            let mip = mip as u32;
            let src_w = (w >> mip).max(1);
            let src_h = (h >> mip).max(1);
            let dst_w = (w >> (mip + 1)).max(1);
            let dst_h = (h >> (mip + 1)).max(1);
            ctx.write_buffer(
                ub,
                0,
                bytemuck::bytes_of(&HiZUniforms {
                    src_size: [src_w, src_h],
                    dst_size: [dst_w, dst_h],
                }),
            );
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // ── HiZ Reuse optimization: skip rebuild if camera static ─────────────
        // Use camera_generation counter to detect actual camera data changes.
        let camera_gen = ctx.scene.camera_generation;

        // Check if resolution changed (window resize invalidates HiZ pyramid)
        let resolution_changed = ctx.width != self.width || ctx.height != self.height;

        // Skip HiZ rebuild if camera hasn't changed and resolution is the same
        if !self.first_frame && camera_gen == self.prev_camera_generation && self.copy_bind_group.is_some() && !resolution_changed {
            // Camera static and resolution unchanged - reuse existing HiZ pyramid from previous frame
            return Ok(());
        }

        // Camera moved or resolution changed - update generation and rebuild pyramid
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
        // A separate compute pass provides an implicit GPU barrier so Phase 2
        // sees the freshly written mip-0 data.
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ DepthCopy"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, self.copy_bind_group.as_ref().unwrap(), &[]);
            let wg_x = self.width.div_ceil(WORKGROUP_SIZE);
            let wg_y = self.height.div_ceil(WORKGROUP_SIZE);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        } // pass drops here -> implicit GPU barrier

        // Phase 2: build the remaining mip levels via MAX-reduction
        // O(log resolution) dispatches, fixed at ~10 for <= 4K resolution.
        {
            let mut pass = ctx.encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HiZ MipChain"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.mip_pipeline);
            for (mip, bg) in self.mip_bind_groups.iter().enumerate() {
                let mip = mip as u32;
                let dst_w = (ctx.width >> (mip + 1)).max(1);
                let dst_h = (ctx.height >> (mip + 1)).max(1);
                let wg_x = dst_w.div_ceil(WORKGROUP_SIZE);
                let wg_y = dst_h.div_ceil(WORKGROUP_SIZE);
                pass.set_bind_group(0, bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
        }
        Ok(())
    }

    fn publish<'a>(&'a self, frame: &mut FrameResources<'a>) {
        // Expose the current frame's HiZ for any downstream pass that needs it.
        // OcclusionCullPass uses its own Arc ref so it always reads the
        // previous frame's data (temporal), not this freshly built pyramid.
        frame.hiz = Some(&*self.hiz_view);
        frame.hiz_sampler = Some(&*self.hiz_sampler);

        // Expose static HiZ if loaded
        if let Some(ref view) = self.static_hiz_view {
            frame.static_hiz = Some(&**view);
        }
        if let Some(ref sampler) = self.static_hiz_sampler {
            frame.static_hiz_sampler = Some(&**sampler);
        }
    }
}
