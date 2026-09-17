//! Hi-Z occlusion-culling pass.
//!
//! Runs AFTER IndirectDispatchPass (frustum cull) each frame, using the PREVIOUS
//! frame's Hi-Z pyramid (temporal approach). One workgroup per draw-call group
//! cooperatively Hi-Z-tests each instance that already survived frustum culling
//! (read from `compacted_indices`) and compacts real survivors into
//! `compacted_indices_2`, writing the final per-group visible count into
//! `indirect[slot * 5 + 1]`. Downstream draws must read `compacted_indices_2`.
//!
//! The first frame that actually has live instances has no Hi-Z pyramid yet,
//! so instead of testing anything it copies `compacted_indices` straight
//! through to `compacted_indices_2` unchanged (see `hiz_warmed_up`'s doc for
//! why this is gated on "first frame with real instances", not `frame_num ==
//! 0` -- the two are not the same frame). Bind-group is rebuilt lazily when
//! buffer pointers change (e.g. scene grows).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

pub use helio_pass_gbuffer::CulledBatchFrameData;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullParams {
    screen_width: u32,
    screen_height: u32,
    draw_count: u32,
    hiz_mip_count: u32,
    static_hiz_available: u32,
    grid_resolution_x: u32,
    grid_resolution_y: u32,
    grid_resolution_z: u32,
    world_bounds_min_x: f32,
    world_bounds_min_y: f32,
    world_bounds_min_z: f32,
    world_bounds_max_x: f32,
    world_bounds_max_y: f32,
    world_bounds_max_z: f32,
}

/// Below this many instances, `compacted_indices_2_buf` still allocates at
/// this floor -- matches `ObjectBatchPass`'s own `MIN_SCRATCH_CAPACITY` idiom.
const MIN_CAPACITY: u32 = 256;

pub struct OcclusionCullPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    cull_params_buf: wgpu::Buffer,
    hiz_sampler: Arc<wgpu::Sampler>,
    cull_stats_buf: wgpu::Buffer,
    /// This pass's own output -- no longer a central `GpuScene` field (see
    /// `CulledBatchFrameData`'s doc, now in this crate): final (frustum +
    /// occlusion) surviving instance slots, one `u32` per live instance
    /// (worst case).
    compacted_indices_2_buf: wgpu::Buffer,
    instance_capacity: u32,

    /// Placeholder 3D texture used when no static HiZ is loaded.
    placeholder_static_hiz_view: wgpu::TextureView,
    placeholder_static_hiz_sampler: wgpu::Sampler,

    /// Metadata for the static HiZ voxel grid (set from HiZBuildPass).
    static_hiz_bounds_min: [f32; 3],
    static_hiz_bounds_max: [f32; 3],
    static_hiz_grid_resolution: [u32; 3],

    /// Cached bind group, invalidated when buffer pointers change.
    bind_group: Option<wgpu::BindGroup>,
    /// True once this pass has actually run its real Hi-Z test against a
    /// depth buffer built from a frame that drew real geometry. `frame_num
    /// == 0` is NOT an equivalent condition: `batch.draw_count` comes from
    /// `ObjectBatchPass`'s own async GPU->CPU readback of its compute
    /// results, which lags a frame behind the GPU work that produced it --
    /// on frame 0 it reads 0 regardless of how many objects were actually
    /// spawned, so `execute()`'s `draw_count == 0` early-out fires before
    /// the frame-0 bypass below ever runs. The bypass then never executes,
    /// `compacted_indices_2_buf` stays zeroed, and the very first real
    /// dispatch (frame 1) Hi-Z-tests against a pyramid built from frame 0's
    /// EMPTY depth buffer (nothing was drawn, so nothing was written to
    /// it) -- in reversed-Z that background clears to 0.0/far, so every
    /// real object's near-depth reads as "closer than the empty
    /// background" and gets marked occluded. Occlusion culling mutates
    /// `indirect_dispatch.indirect` in place, so that zeroes every
    /// instance count; the next frame's depth buffer is then ALSO empty
    /// (nothing drew), and the cycle never recovers -- a permanent
    /// deadlock, not a one-frame glitch. Gating the bypass on "have I ever
    /// dispatched a real test" instead of "is this frame_num 0" fixes it:
    /// the bypass now runs on whichever frame is actually first to see
    /// `draw_count > 0`, guaranteeing real geometry lands in depth before
    /// Hi-Z testing ever reads from it.
    hiz_warmed_up: bool,
    /// (camera, instances, draw_calls, indirect, hiz_view, static_hiz_view,
    /// static_hiz_sampler, cull_stats_buf, compacted_indices, compacted_indices_2,
    /// coordinate_spaces)
    bind_group_key: Option<(
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
        usize,
    )>,
    screen_width: u32,
    screen_height: u32,
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
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/occlusion_cull.wgsl").into()),
        });

        let cull_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OcclusionCull CullParams"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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
        let placeholder_static_hiz_view =
            placeholder_static_hiz.create_view(&wgpu::TextureViewDescriptor {
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
            label: Some("OcclusionCull BGL"),
            entries: &[
                // 0: Camera storage
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 1: CullParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 2: GpuInstanceData[] (read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: GpuDrawCall[] (read-only) — for mapping draw index → first instance
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: Hi-Z texture
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 5: Hi-Z sampler (non-filtering, nearest)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // 6: indirect draw buffer (read + write, u32 raw view)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: Static HiZ 3D voxel texture (pre-baked PVS, R32Float, non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                // 8: Static HiZ sampler (nearest, non-filtering — R32Float is non-filterable)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // 9: Culling stats (read_write, atomic counters)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 10: compacted_indices (read-only) — frustum-stage survivors
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 11: compacted_indices_2 (read_write) — final surviving set
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 12: coordinate_spaces (read-only) — see occlusion_cull.wgsl
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("OcclusionCull PL"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("OcclusionCull Pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let compacted_indices_2_buf = create_compacted_indices_2_buf(device, MIN_CAPACITY);

        Self {
            pipeline,
            bgl,
            cull_params_buf,
            hiz_sampler,
            cull_stats_buf,
            compacted_indices_2_buf,
            instance_capacity: MIN_CAPACITY,
            placeholder_static_hiz_view,
            placeholder_static_hiz_sampler,
            static_hiz_bounds_min: [0.0; 3],
            static_hiz_bounds_max: [0.0; 3],
            static_hiz_grid_resolution: [0; 3],
            bind_group: None,
            hiz_warmed_up: false,
            bind_group_key: None,
            screen_width,
            screen_height,
        }
    }

    /// Grows `compacted_indices_2_buf` to at least `instance_count` rows
    /// (next-power-of-two, floor `MIN_CAPACITY`). Returns `true` if it
    /// reallocated (the caller must then rebuild the bind group).
    fn ensure_capacity(&mut self, device: &wgpu::Device, instance_count: u32) -> bool {
        if instance_count <= self.instance_capacity {
            return false;
        }
        self.instance_capacity = instance_count.next_power_of_two().max(MIN_CAPACITY);
        self.compacted_indices_2_buf =
            create_compacted_indices_2_buf(device, self.instance_capacity);
        true
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

fn create_compacted_indices_2_buf(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("OcclusionCull CompactedIndices2"),
        size: (capacity as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

impl RenderPass for OcclusionCullPass {
    fn name(&self) -> &'static str {
        "OcclusionCull"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            "hiz",
            "static_hiz",
            "static_hiz_sampler",
            "object_batch",
            "indirect_dispatch",
        ]
    }

    fn writes(&self) -> &'static [&'static str] {
        &["culled_batch"]
    }

    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        builder.read("object_batch");
        builder.read("indirect_dispatch");
        builder.write_buffer("culled_batch");
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        // `indirect_dispatch.indirect` is mutated IN PLACE by this pass
        // (its `instance_count` field, refined from frustum-only down to
        // frustum+occlusion survivors) -- there is no separate owned
        // `indirect` buffer here, so `culled_batch` simply republishes the
        // same buffer reference `indirect_dispatch` already holds.
        let Some(indirect_dispatch) = frame.read::<helio_pass_indirect_dispatch::IndirectDispatchFrameData<'a>>(helio_core::ResourceKey::new("indirect_dispatch"), "OcclusionCull") else {
            return;
        };
        let compacted_indices: &'a wgpu::Buffer = unsafe { std::mem::transmute(&self.compacted_indices_2_buf) };
        frame.write(helio_core::ResourceKey::new("culled_batch"), 
            crate::CulledBatchFrameData {
                indirect: indirect_dispatch.indirect,
                compacted_indices,
            },
            "OcclusionCull",
        );
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a helio_core::ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        // `set_screen_size` was previously dead code -- nothing called it, so
        // `screen_width`/`screen_height` stayed frozen at this pass's
        // construction-time resolution forever, while the "hiz" texture it
        // samples (graph-pooled) DID get correctly reallocated on resize by
        // `HiZBuildPass::on_resize`/its own `ctx.resize` handling in
        // `prepare`. The resulting mismatch corrupts `pick_mip`'s mip-level
        // math and `screen_radius_px`'s UV footprint sizing against the
        // pyramid's real dimensions -- mirrors `HiZBuildPass::prepare`'s own
        // `ctx.resize` sync so both passes agree on the current resolution.
        if ctx.resize {
            self.set_screen_size(ctx.width.max(1), ctx.height.max(1));
        }

        let batch = ctx.pass_resources.get::<helio_pass_gbuffer::ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new("object_batch"));
        let draw_count = batch.map(|b| b.draw_count).unwrap_or(0);
        self.ensure_capacity(ctx.device, batch.map(|b| b.instance_count).unwrap_or(0));

        // Plain (non-panicking) lookup: "static_hiz" is legitimately optional
        // (only present once real baked data is loaded via `load_static_hiz`)
        // -- `read_texture_view` falls through to a debug-only panic on a
        // missing key, which fires before this `.is_some()` ever sees it.
        let static_hiz_available = ctx.pass_resources.get(helio_core::ResourceKey::new("static_hiz"))
            .or_else(|| ctx.pass_resources.texture_binding("static_hiz"))
            .is_some();
        let p = CullParams {
            screen_width: self.screen_width,
            screen_height: self.screen_height,
            draw_count,
            hiz_mip_count: mip_levels(self.screen_width, self.screen_height),
            static_hiz_available: if static_hiz_available { 1 } else { 0 },
            grid_resolution_x: self.static_hiz_grid_resolution[0],
            grid_resolution_y: self.static_hiz_grid_resolution[1],
            grid_resolution_z: self.static_hiz_grid_resolution[2],
            world_bounds_min_x: self.static_hiz_bounds_min[0],
            world_bounds_min_y: self.static_hiz_bounds_min[1],
            world_bounds_min_z: self.static_hiz_bounds_min[2],
            world_bounds_max_x: self.static_hiz_bounds_max[0],
            world_bounds_max_y: self.static_hiz_bounds_max[1],
            world_bounds_max_z: self.static_hiz_bounds_max[2],
        };
        ctx.write_buffer(&self.cull_params_buf, 0, bytemuck::bytes_of(&p));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let Some(batch) = ctx.resources.get::<helio_pass_gbuffer::ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new("object_batch")) else {
            return Ok(());
        };
        let Some(indirect_dispatch) = ctx.resources.get::<helio_pass_indirect_dispatch::IndirectDispatchFrameData<'_>>(helio_core::ResourceKey::new("indirect_dispatch")) else {
            return Ok(());
        };
        let Some(coord_data) = ctx.resources.get::<helio_pass_gbuffer::CoordinateSpacesFrameData<'_>>(helio_core::ResourceKey::new("coordinate_spaces")) else {
            return Ok(());
        };
        let draw_count = batch.draw_count;
        if draw_count == 0 {
            return Ok(());
        }

        // Temporal Hi-Z: the first frame with real instances has no valid
        // pyramid yet (see `hiz_warmed_up`'s doc for why this is NOT the
        // same as `frame_num == 0`) — skip real occlusion testing, but
        // downstream draws always read `compacted_indices_2`, so pass the
        // frustum-culled list through unchanged instead of leaving it
        // stale/uninitialized.
        if !self.hiz_warmed_up {
            let instance_count = batch.instance_count as u64;
            if instance_count > 0 {
                unsafe { &mut *ctx.encoder_ptr }.copy_buffer_to_buffer(
                    indirect_dispatch.compacted_indices,
                    0,
                    &self.compacted_indices_2_buf,
                    0,
                    instance_count * 4,
                );
            }
            // Only declare Hi-Z warmed up once real instances actually got
            // copied through this frame -- that's what guarantees GBuffer
            // has real geometry to write into depth this frame, which is
            // the one thing frame N+1's Hi-Z pyramid actually needs to be
            // valid. If `instance_count` was 0 here (draw_count > 0 but no
            // live instances yet -- shouldn't normally happen, but this
            // must not gamble on it), stay un-warmed and retry the bypass
            // next frame instead of moving on to a real test with nothing
            // real backing it either.
            if batch.instance_count > 0 {
                self.hiz_warmed_up = true;
            }
            return Ok(());
        }

        // Lazy bind-group rebuild: rebuild whenever any buffer pointer or the
        // HiZ texture view changes (e.g. scene grows, graph reallocates on resize).
        let hiz_view =
            ctx.resources.read_texture_view(helio_core::ResourceKey::new("hiz"), "OcclusionCull").expect(
                "OcclusionCull: 'hiz' view not routed by graph — is HiZBuildPass declared?",
            );

        // Resolve static HiZ resources (use placeholder when no pre-baked data is
        // loaded). Plain (non-panicking) lookups, same reasoning as `prepare`'s
        // `static_hiz_available` above -- `read_texture_view`/`read_sampler` fall
        // through to a debug-only panic on a missing key, which would fire before
        // `unwrap_or` ever sees it, even though this resource is legitimately optional.
        let static_hiz_view = ctx.resources.get(helio_core::ResourceKey::new("static_hiz"))
            .or_else(|| ctx.resources.texture_binding("static_hiz"))
            .unwrap_or(&self.placeholder_static_hiz_view);
        let static_hiz_sampler = ctx.resources.get(helio_core::ResourceKey::new("static_hiz_sampler"))
            .unwrap_or(&self.placeholder_static_hiz_sampler);

        let key = (
            ctx.camera as *const _ as usize,
            batch.instances as *const _ as usize,
            batch.draw_calls as *const _ as usize,
            indirect_dispatch.indirect as *const _ as usize,
            hiz_view as *const _ as usize,
            static_hiz_view as *const _ as usize,
            static_hiz_sampler as *const _ as usize,
            &self.cull_stats_buf as *const _ as usize,
            indirect_dispatch.compacted_indices as *const _ as usize,
            &self.compacted_indices_2_buf as *const _ as usize,
            coord_data.coordinate_spaces as *const _ as usize,
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("OcclusionCull BG"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.cull_params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: batch.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: batch.draw_calls.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::TextureView(hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Sampler(&self.hiz_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: indirect_dispatch.indirect.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: wgpu::BindingResource::TextureView(static_hiz_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: wgpu::BindingResource::Sampler(static_hiz_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: self.cull_stats_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: indirect_dispatch.compacted_indices.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 11,
                        resource: self.compacted_indices_2_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 12,
                        resource: coord_data.coordinate_spaces.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        // One workgroup per draw-call group — its 64 lanes cooperatively
        // Hi-Z-test and compact that group's frustum survivors.
        let mut pass =
            unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("OcclusionCull"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(draw_count, 1, 1);
        Ok(())
    }
}

fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}
