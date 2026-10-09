//! Hi-Z occlusion-culling and visible-range compaction pass.
//!
//! Runs AFTER IndirectDispatchPass (frustum cull) each frame, using the PREVIOUS
//! frame's Hi-Z pyramid (temporal approach). One workgroup per draw-call group
//! cooperatively Hi-Z-tests each instance that already survived frustum culling
//! (read from `compacted_indices`) and compacts real survivors into
//! `compacted_indices_2`, updating the source indirect instance counts. A
//! second GPU dispatch packs non-empty indirect draw records inside each
//! material range and writes survivor counts for indirect-count rendering.
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
    /// Baked PVS grid (`occlusion_cull.wgsl` documents the layout); 0 = none.
    pvs_available: u32,
    pvs_grid: [u32; 3],
    pvs_min: [f32; 3],
    pvs_cell_size: f32,
    /// u32 words per source cell: the baker's u64 words × 2.
    pvs_words_per_cell: u32,
    _pad: [u32; 3],
}

/// The baked PVS grid shape last uploaded, and the bitfield it came from.
#[derive(Clone, Copy, PartialEq)]
struct PvsGrid {
    grid: [u32; 3],
    min: [f32; 3],
    cell_size: f32,
    words_per_cell: u32,
    /// Identity of the baked bitfield (address, length): re-upload on change.
    source: (usize, usize),
}

/// Below this many instances, `compacted_indices_2_buf` still allocates at
/// this floor -- matches `ObjectBatchPass`'s own `MIN_SCRATCH_CAPACITY` idiom.
const MIN_CAPACITY: u32 = 256;

pub struct OcclusionCullPass {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    range_compact_pipeline: wgpu::ComputePipeline,
    range_compact_bgl: wgpu::BindGroupLayout,
    range_compact_params: wgpu::Buffer,
    range_compact_bind_group: Option<wgpu::BindGroup>,
    range_compact_key: Option<[wgpu::Buffer; 7]>,
    cull_params_buf: wgpu::Buffer,
    hiz_sampler: Arc<wgpu::Sampler>,
    cull_stats_buf: wgpu::Buffer,
    /// This pass's own output -- no longer a central `GpuScene` field (see
    /// `CulledBatchFrameData`'s doc, now in this crate): final (frustum +
    /// occlusion) surviving instance slots, one `u32` per live instance
    /// (worst case).
    compacted_indices_2_buf: wgpu::Buffer,
    /// Packed indirect records consumed by render passes after occlusion.
    compacted_indirect_buf: wgpu::Buffer,
    instance_capacity: u32,

    /// The baked PVS bitfield on the GPU (Helio#256), uploaded once per bake
    /// from the `baked_pvs` frame input. Four zero bytes when none is baked
    /// (`pvs_available = 0` then keeps the shader from reading it).
    pvs_buf: wgpu::Buffer,
    pvs_grid: Option<PvsGrid>,

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
    /// ([camera, instances, draw_calls, indirect, pvs_buf, cull_stats_buf,
    /// compacted_indices, compacted_indices_2, coordinate_spaces], hiz_view)
    bind_group_key: Option<([wgpu::Buffer; 9], wgpu::TextureView)>,
    screen_width: u32,
    screen_height: u32,
}

impl OcclusionCullPass {
    /// Create the occlusion-cull pass.
    ///
    /// The HiZ texture view is read from `ctx.registry.hiz` each frame (routed
    /// by the graph). `hiz_sampler` is owned by HiZBuildPass and shared via Arc.
    pub fn new(
        device: &wgpu::Device,
        hiz_sampler: Arc<wgpu::Sampler>,
        screen_width: u32,
        screen_height: u32,
        cull_stats_buf: wgpu::Buffer,
    ) -> Self {
        let shader = helio_core::shader::module(device, "OcclusionCull Shader", helio_core::include_wgsl!("../shaders/occlusion_cull.wgsl"));

        let cull_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OcclusionCull CullParams"),
            size: std::mem::size_of::<CullParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let pvs_buf = create_pvs_buf(device, 4);

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
                // 7: Baked PVS bitfield (read-only storage)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
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

        let range_compact_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("OcclusionCull RangeCompaction BGL"),
            entries: &[
                storage_layout_entry(0, true),
                storage_layout_entry(1, false),
                storage_layout_entry(2, true),
                storage_layout_entry(3, true),
                storage_layout_entry(4, true),
                storage_layout_entry(5, true),
                storage_layout_entry(6, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let compact_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("OcclusionCull RangeCompaction Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/compact_ranges.wgsl").into(),
            ),
        });
        let compact_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("OcclusionCull RangeCompaction PL"),
            bind_group_layouts: &[Some(&range_compact_bgl)],
            immediate_size: 0,
        });
        let range_compact_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("OcclusionCull RangeCompaction Pipeline"),
            layout: Some(&compact_layout),
            module: &compact_shader,
            entry_point: Some("compact_ranges"),
            compilation_options: Default::default(),
            cache: None,
        });
        let range_compact_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OcclusionCull RangeCompaction Params"),
            size: 768,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let compacted_indices_2_buf = create_compacted_indices_2_buf(device, MIN_CAPACITY);
        let compacted_indirect_buf = create_compacted_indirect_buf(device, MIN_CAPACITY);

        Self {
            pipeline,
            bgl,
            range_compact_pipeline,
            range_compact_bgl,
            range_compact_params,
            range_compact_bind_group: None,
            range_compact_key: None,
            cull_params_buf,
            hiz_sampler,
            cull_stats_buf,
            compacted_indices_2_buf,
            compacted_indirect_buf,
            instance_capacity: MIN_CAPACITY,
            pvs_buf,
            pvs_grid: None,
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
        self.compacted_indirect_buf = create_compacted_indirect_buf(device, self.instance_capacity);
        true
    }

    fn record_range_compaction(
        &mut self,
        ctx: &mut PassContext,
        batch: &helio_pass_gbuffer::ObjectBatchFrameData<'_>,
        source_indirect: &wgpu::Buffer,
        draw_count: u32,
    ) {
        let mut cmds = ctx.graphics_cmds();
        let bytes = (draw_count as u64 * 20).max(4);
        cmds.copy_buffer_to_buffer(source_indirect, 0, &self.compacted_indirect_buf, 0, bytes);

        // Legacy/test frames can have no GPU range slots; preserve the
        // copied list unchanged in that case.
        let slots = batch.range_slot_capacity;
        if slots == 0 {
            return;
        }
        let key = [
            source_indirect.clone(),
            batch.range_counts_gpu.clone(),
            batch.opaque_ranges_gpu.clone(),
            batch.transparent_ranges_gpu.clone(),
            batch.forward_ranges_gpu.clone(),
            batch.draw_counts_gpu.clone(),
            self.compacted_indirect_buf.clone(),
        ];
        if self.range_compact_key.as_ref() != Some(&key) {
            self.range_compact_bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("OcclusionCull RangeCompaction BG"),
                layout: &self.range_compact_bgl,
                entries: &[
                    buffer_entry(0, source_indirect),
                    buffer_entry(1, &self.compacted_indirect_buf),
                    buffer_entry(2, batch.range_counts_gpu),
                    buffer_entry(3, batch.opaque_ranges_gpu),
                    buffer_entry(4, batch.transparent_ranges_gpu),
                    buffer_entry(5, batch.forward_ranges_gpu),
                    buffer_entry(6, batch.draw_counts_gpu),
                    uniform_range_entry(7, &self.range_compact_params, 8),
                ],
            }));
            self.range_compact_key = Some(key);
        }
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("OcclusionCull RangeCompaction"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.range_compact_pipeline);
        for bucket in 0..3u32 {
            pass.set_bind_group(
                0,
                self.range_compact_bind_group.as_ref().unwrap(),
                &[bucket * 256],
            );
            pass.dispatch_workgroups_indirect(batch.range_counts_gpu, 16 + bucket as u64 * 12);
        }
    }

    /// Update internal-resolution dimensions used by cull uniforms.
    pub fn set_screen_size(&mut self, width: u32, height: u32) {
        self.screen_width = width;
        self.screen_height = height;
    }

    /// Test-only observability for `set_screen_size`'s effect -- lets an
    /// integration test assert this pass's cull-uniform resolution actually
    /// tracks the graph's real resolution across a resize, instead of only
    /// being inferable indirectly from occlusion-test outcomes.
    pub fn screen_size(&self) -> (u32, u32) {
        (self.screen_width, self.screen_height)
    }

    /// Whether a baked PVS is currently uploaded and used by the cull.
    pub fn pvs_active(&self) -> bool {
        self.pvs_grid.is_some()
    }

    /// Follow the `baked_pvs` frame input: upload its bitfield when a bake
    /// arrives or changes, drop it when it goes away.
    fn sync_pvs(&mut self, ctx: &PrepareContext) {
        let baked = ctx
            .registry
            .get::<helio_bake_types::BakedPvsRef<'_>>(helio_core::ResourceKey::new("baked_pvs"));
        let Some(pvs) = baked.filter(|pvs| {
            pvs.cell_size > 0.0 && pvs.words_per_cell > 0 && !pvs.bits.is_empty()
        }) else {
            self.pvs_grid = None;
            return;
        };
        let grid = PvsGrid {
            grid: pvs.grid_dims,
            min: pvs.world_min,
            cell_size: pvs.cell_size,
            words_per_cell: pvs.words_per_cell * 2,
            source: (pvs.bits.as_ptr() as usize, pvs.bits.len()),
        };
        if self.pvs_grid == Some(grid) {
            return;
        }
        let bytes: &[u8] = bytemuck::cast_slice(pvs.bits);
        if self.pvs_buf.size() < bytes.len() as u64 {
            self.pvs_buf = create_pvs_buf(ctx.device, bytes.len() as u64);
        }
        ctx.queue.write_buffer(&self.pvs_buf, 0, bytes);
        self.pvs_grid = Some(grid);
    }
}

fn create_pvs_buf(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("OcclusionCull Baked PVS"),
        size: size.max(4).next_multiple_of(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_compacted_indices_2_buf(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("OcclusionCull CompactedIndices2"),
        size: (capacity as u64 * 4).max(4),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn create_compacted_indirect_buf(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("OcclusionCull CompactedIndirect"),
        size: (capacity as u64 * 20).max(20),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::INDIRECT
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn storage_layout_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn uniform_range_entry(binding: u32, buffer: &wgpu::Buffer, size: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: std::num::NonZeroU64::new(size),
        }),
    }
}

impl RenderPass for OcclusionCullPass {
    fn name(&self) -> &'static str {
        "OcclusionCull"
    }

    fn reads(&self) -> &'static [&'static str] {
        &[
            "hiz",
            "object_batch",
            "indirect_dispatch",
        ]
    }

    fn declare_resources(&self, builder: &mut helio_core::graph::ResourceBuilder) {
        builder.read("object_batch");
        builder.read("indirect_dispatch");
        builder.write_buffer("culled_batch");
    }

    fn publish<'a>(&self, frame: &mut helio_core::ResourceRegistry<'a>) {
        // The source indirect records are refined by Hi-Z, then copied and
        // compacted into this pass's output buffer for downstream geometry.
        //
        // Plain (non-panicking) lookup: this is legitimately optional (a
        // graph that omits `IndirectDispatchPass`, e.g. a focused test
        // graph, has nothing to republish yet) -- the `else { return; }`
        // below already handles absence gracefully, but `frame.read()`
        // falls through to a debug-only panic on a missing key before ever
        // returning `None`, defeating that.
        let Some(_indirect_dispatch) = frame.get::<helio_pass_indirect_dispatch::IndirectDispatchFrameData<'a>>(helio_core::ResourceKey::new("indirect_dispatch")) else {
            return;
        };
        let compacted_indices: &'a wgpu::Buffer = unsafe { std::mem::transmute(&self.compacted_indices_2_buf) };
        let indirect: &'a wgpu::Buffer = unsafe { std::mem::transmute(&self.compacted_indirect_buf) };
        frame.write(helio_core::ResourceKey::new("culled_batch"), 
            crate::CulledBatchFrameData {
                indirect,
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

        let batch = ctx.registry.get::<helio_pass_gbuffer::ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new("object_batch"));
        let draw_count = batch.map(|b| b.draw_count).unwrap_or(0);
        let range_slots = batch.map(|b| b.range_slot_capacity).unwrap_or(0);
        for bucket in 0..3u32 {
            let params = [range_slots, bucket];
            ctx.queue.write_buffer(
                &self.range_compact_params,
                bucket as u64 * 256,
                bytemuck::cast_slice(&params),
            );
        }
        self.ensure_capacity(ctx.device, batch.map(|b| b.instance_count).unwrap_or(0));

        // `baked_pvs` is optional: published by helio-bake's BakeInjectPass
        // only after a bake that included a PVS (`BakeConfig::with_pvs`).
        self.sync_pvs(ctx);
        let pvs = self.pvs_grid;
        let p = CullParams {
            screen_width: self.screen_width,
            screen_height: self.screen_height,
            draw_count,
            hiz_mip_count: mip_levels(self.screen_width, self.screen_height),
            pvs_available: pvs.is_some() as u32,
            pvs_grid: pvs.map_or([0; 3], |p| p.grid),
            pvs_min: pvs.map_or([0.0; 3], |p| p.min),
            pvs_cell_size: pvs.map_or(1.0, |p| p.cell_size),
            pvs_words_per_cell: pvs.map_or(0, |p| p.words_per_cell),
            _pad: [0; 3],
        };
        ctx.write_buffer(&self.cull_params_buf, 0, bytemuck::bytes_of(&p));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let Some(batch) = ctx.registry.get::<helio_pass_gbuffer::ObjectBatchFrameData<'_>>(helio_core::ResourceKey::new("object_batch")) else {
            return Ok(());
        };
        let Some(indirect_dispatch) = ctx.registry.get::<helio_pass_indirect_dispatch::IndirectDispatchFrameData<'_>>(helio_core::ResourceKey::new("indirect_dispatch")) else {
            return Ok(());
        };
        let Some(coord_data) = ctx.registry.get::<helio_pass_gbuffer::CoordinateSpacesFrameData<'_>>(helio_core::resource_keys::coordinate_spaces()) else {
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
                ctx.graphics_cmds().copy_buffer_to_buffer(
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
            self.record_range_compaction(ctx, &batch, indirect_dispatch.indirect, draw_count);
            return Ok(());
        }

        // Lazy bind-group rebuild: rebuild whenever any buffer or the
        // HiZ texture view changes (e.g. scene grows, graph reallocates on resize).
        let hiz_view =
            ctx.registry.read_texture_view(helio_core::ResourceKey::new("hiz"), "OcclusionCull").expect(
                "OcclusionCull: 'hiz' view not routed by graph — is HiZBuildPass declared?",
            );

        let key = (
            [
                ctx.camera.clone(),
                batch.instances.clone(),
                batch.draw_calls.clone(),
                indirect_dispatch.indirect.clone(),
                self.pvs_buf.clone(),
                self.cull_stats_buf.clone(),
                indirect_dispatch.compacted_indices.clone(),
                self.compacted_indices_2_buf.clone(),
                coord_data.coordinate_spaces.clone(),
            ],
            hiz_view.clone(),
        );
        if self.bind_group_key.as_ref() != Some(&key) {
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
                        resource: self.pvs_buf.as_entire_binding(),
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
        {
        let mut cmds = ctx.graphics_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("OcclusionCull"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(draw_count, 1, 1);
        }
        self.record_range_compaction(ctx, &batch, indirect_dispatch.indirect, draw_count);
        Ok(())
    }
}

fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}
