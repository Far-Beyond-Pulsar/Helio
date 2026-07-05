//! Hi-Z occlusion-culling pass.
//!
//! Runs AFTER IndirectDispatchPass (frustum cull) each frame, using the PREVIOUS
//! frame's Hi-Z pyramid (temporal approach).  For each DRAW CALL the shader:
//!  1. Tests the representative instance's bounding sphere against the Hi-Z buffer
//!  2. Writes `indirect[slot * 5 + 1]` = 0 (occluded) or leaves the frustum-cull value
//!
//! Frame 0 is skipped since no Hi-Z pyramid exists yet.
//! Bind-group is rebuilt lazily when buffer pointers change (e.g. scene grows).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_v3::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

const WORKGROUP_SIZE: u32 = 64;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullParams {
    screen_width:         u32,
    screen_height:        u32,
    draw_count:           u32,
    hiz_mip_count:        u32,
    static_hiz_available: u32,
    grid_resolution_x:    u32,
    grid_resolution_y:    u32,
    grid_resolution_z:    u32,
    world_bounds_min_x:   f32,
    world_bounds_min_y:   f32,
    world_bounds_min_z:   f32,
    world_bounds_max_x:   f32,
    world_bounds_max_y:   f32,
    world_bounds_max_z:   f32,
}

pub struct OcclusionCullPass {
    pipeline:        wgpu::ComputePipeline,
    bgl:             wgpu::BindGroupLayout,
    cull_params_buf: wgpu::Buffer,
    hiz_sampler:     Arc<wgpu::Sampler>,
    cull_stats_buf:  wgpu::Buffer,

    /// Placeholder 3D texture used when no static HiZ is loaded.
    placeholder_static_hiz_view:   wgpu::TextureView,
    placeholder_static_hiz_sampler: wgpu::Sampler,

    /// Metadata for the static HiZ voxel grid (set from HiZBuildPass).
    static_hiz_bounds_min:    [f32; 3],
    static_hiz_bounds_max:    [f32; 3],
    static_hiz_grid_resolution: [u32; 3],

    /// Cached bind group, invalidated when buffer pointers change.
    bind_group:     Option<wgpu::BindGroup>,
    /// (camera, instances, draw_calls, indirect, hiz_view, static_hiz_view, static_hiz_sampler)
    bind_group_key: Option<(usize, usize, usize, usize, usize, usize, usize, usize)>,
    screen_width:   u32,
    screen_height:  u32,
}

impl OcclusionCullPass {
    /// Create the occlusion-cull pass.
    ///
    /// The HiZ texture view is read from `ctx.resources.hiz` each frame (routed
    /// by the graph). `hiz_sampler` is owned by HiZBuildPass and shared via Arc.
    pub fn new(
        device: &wgpu::Device,
        hiz_sampler: Arc<wgpu::Sampler>,
        screen_width: u32,
        screen_height: u32,
        cull_stats_buf: wgpu::Buffer,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("OcclusionCull Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/occlusion_cull.wgsl").into(),
            ),
        });

        let cull_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("OcclusionCull CullParams"),
            size:               std::mem::size_of::<CullParams>() as u64,
            usage:              wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Placeholder 3D texture for static HiZ when none is loaded.
        let placeholder_static_hiz = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("OcclusionCull Placeholder Static HiZ"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let placeholder_static_hiz_view = placeholder_static_hiz.create_view(&wgpu::TextureViewDescriptor {
            label: Some("OcclusionCull Placeholder Static HiZ View"),
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });
        let placeholder_static_hiz_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("OcclusionCull Placeholder Static HiZ Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Bind group layout must match occlusion_cull.wgsl binding declarations.
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label:   Some("OcclusionCull BGL"),
            entries: &[
                // 0: Camera uniform
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
                // 1: CullParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding:    1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 2: GpuInstanceData[] (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding:    2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 3: GpuDrawCall[] (read-only) — for mapping draw index → first instance
                wgpu::BindGroupLayoutEntry {
                    binding:    3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 4: Hi-Z texture
                wgpu::BindGroupLayoutEntry {
                    binding:    4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled:   false,
                    },
                    count: None,
                },
                // 5: Hi-Z sampler (non-filtering, nearest)
                wgpu::BindGroupLayoutEntry {
                    binding:    5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // 6: indirect draw buffer (read + write, u32 raw view)
                wgpu::BindGroupLayoutEntry {
                    binding:    6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
                // 7: Static HiZ 3D voxel texture (pre-baked PVS, R32Float, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding:    7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type:    wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled:   false,
                    },
                    count: None,
                },
                // 8: Static HiZ sampler (nearest, non-filtering — R32Float is non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding:    8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // 9: Culling stats (read_write, atomic counters)
                wgpu::BindGroupLayoutEntry {
                    binding:    9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty:                 wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size:   None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label:              Some("OcclusionCull PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size:     0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label:               Some("OcclusionCull Pipeline"),
            layout:              Some(&pl),
            module:              &shader,
            entry_point:         Some("main"),
            compilation_options: Default::default(),
            cache:               None,
        });

        Self {
            pipeline,
            bgl,
            cull_params_buf,
            hiz_sampler,
            cull_stats_buf,
            placeholder_static_hiz_view,
            placeholder_static_hiz_sampler,
            static_hiz_bounds_min: [0.0; 3],
            static_hiz_bounds_max: [0.0; 3],
            static_hiz_grid_resolution: [0; 3],
            bind_group:     None,
            bind_group_key: None,
            screen_width,
            screen_height,
        }
    }

    /// Update internal-resolution dimensions used by cull uniforms.
    pub fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    /// Set the static HiZ voxel grid metadata (called when pre-baked data is loaded).
    pub fn set_static_hiz_metadata(
        &mut self,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        resolution: [u32; 3],
    ) {
        self.static_hiz_bounds_min = bounds_min;
        self.static_hiz_bounds_max = bounds_max;
        self.static_hiz_grid_resolution = resolution;
    }
}

impl RenderPass for OcclusionCullPass {
    fn name(&self) -> &'static str {
        "OcclusionCull"
    }

    fn reads(&self) -> &'static [&'static str] {
        &["hiz", "static_hiz", "static_hiz_sampler"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        let static_hiz_available = ctx.frame_resources.static_hiz.is_some();
        let p = CullParams {
            screen_width:         self.screen_width,
            screen_height:        self.screen_height,
            draw_count:           ctx.scene.draw_calls.len() as u32,
            hiz_mip_count:        mip_levels(self.screen_width, self.screen_height),
            static_hiz_available: if static_hiz_available { 1 } else { 0 },
            grid_resolution_x:    self.static_hiz_grid_resolution[0],
            grid_resolution_y:    self.static_hiz_grid_resolution[1],
            grid_resolution_z:    self.static_hiz_grid_resolution[2],
            world_bounds_min_x:   self.static_hiz_bounds_min[0],
            world_bounds_min_y:   self.static_hiz_bounds_min[1],
            world_bounds_min_z:   self.static_hiz_bounds_min[2],
            world_bounds_max_x:   self.static_hiz_bounds_max[0],
            world_bounds_max_y:   self.static_hiz_bounds_max[1],
            world_bounds_max_z:   self.static_hiz_bounds_max[2],
        };
        ctx.write_buffer(&self.cull_params_buf, 0, bytemuck::bytes_of(&p));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        // Temporal Hi-Z: frame 0 has no valid pyramid yet — skip culling.
        if ctx.frame_num == 0 {
            return Ok(());
        }

        let draw_count = ctx.scene.draw_count;
        if draw_count == 0 {
            return Ok(());
        }

        // Lazy bind-group rebuild: rebuild whenever any buffer pointer or the
        // HiZ texture view changes (e.g. scene grows, graph reallocates on resize).
        let hiz_view = ctx.resources.hiz.as_ref()
            .expect("OcclusionCull: 'hiz' view not routed by graph — is HiZBuildPass declared?");

        // Resolve static HiZ resources (use placeholder when no pre-baked data is loaded).
        let static_hiz_view = ctx.resources.static_hiz.get()
            .unwrap_or(&self.placeholder_static_hiz_view);
        let static_hiz_sampler = ctx.resources.static_hiz_sampler.get()
            .unwrap_or(&self.placeholder_static_hiz_sampler);

        let key = (
            ctx.scene.camera       as *const _ as usize,
            ctx.scene.instances     as *const _ as usize,
            ctx.scene.draw_calls    as *const _ as usize,
            ctx.scene.indirect      as *const _ as usize,
            hiz_view               as *const _ as usize,
            static_hiz_view        as *const _ as usize,
            static_hiz_sampler     as *const _ as usize,
            &self.cull_stats_buf   as *const _ as usize,
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label:  Some("OcclusionCull BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding:  0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  1,
                        resource: self.cull_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  2,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  3,
                        resource: ctx.scene.draw_calls.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  4,
                        resource: wgpu::BindingResource::TextureView(hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding:  5,
                        resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding:  6,
                        resource: ctx.scene.indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding:  7,
                        resource: wgpu::BindingResource::TextureView(static_hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding:  8,
                        resource: wgpu::BindingResource::Sampler(static_hiz_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding:  9,
                        resource: self.cull_stats_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let wg = draw_count.div_ceil(WORKGROUP_SIZE);
        let mut pass = unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label:            Some("OcclusionCull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(wg, 1, 1);
        Ok(())
    }
}

fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}
