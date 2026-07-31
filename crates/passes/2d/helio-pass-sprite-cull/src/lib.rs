//! GPU-side culling + depth sort for a growable, handle-addressed instance
//! pool — built for `helio-pass-sprite-batch`'s `SpriteBatchPass`, but
//! deliberately has no Cargo dependency on that crate: it only needs raw
//! `Arc<wgpu::Buffer>` handles (instance data + alive flags) and two counts,
//! matching how `helio-pass-shadow-cull` stays decoupled from
//! `helio-pass-shadow` — the two are wired together by whoever builds the
//! render graph (`SpriteCullPass::new` takes the buffers, then
//! `SpriteBatchPass::use_gpu_culling` takes this pass's outputs), not by one
//! crate importing the other's types.
//!
//! Runs as its own [`RenderPass`], added to the graph *before* the pass that
//! renders the culled/sorted result — this is what lets that pass's
//! `execute()` never know the visible instance count on the CPU at all.
//! Every frame, entirely on the GPU:
//!
//! 1. `cs_cull` (`shaders/sprite_cull.wgsl`) — one thread per pool slot.
//!    Alive + in-view slots are atomically compacted into `indices_a`/`keys_a`,
//!    and the surviving count is atomically accumulated directly into
//!    [`SpriteCullPass::indirect_buf`]'s `instance_count` field.
//! 2. Four digit passes of a GPU LSD radix sort (`shaders/sprite_sort.wgsl`)
//!    over the compacted list — see that shader's module doc comment for the
//!    three-kernel (histogram / scan / scatter) design.
//!
//! The CPU touches none of this per frame beyond four tiny fixed-size
//! dispatches and one 4-byte buffer-to-buffer copy (propagating the cull
//! pass's GPU-computed visible count into the sort kernels' bounds check) —
//! there is no per-instance CPU work in `prepare()`/`execute()` regardless
//! of pool size.
//!
//! The instance byte layout this reads (`position`/`size`/`depth` at
//! specific offsets in `shaders/sprite_cull.wgsl`) is a *protocol*, not a
//! Cargo-level type dependency — it must stay byte-identical to
//! `helio_pass_sprite_batch::SpriteInstance`'s `#[repr(C)]` layout, the same
//! way `helio-pass-shadow-cull`'s shader hardcodes `helio-core`'s
//! `GpuInstance` layout without depending on that crate for it.
//!
//! # Fixed capacity
//!
//! This pass is wired to specific `instances`/`slot_alive` buffers *once* at
//! construction and does not handle them being reallocated afterward — size
//! the pool (e.g. `SpriteBatchPass::reserve`) *before* wiring GPU culling in.

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result};
use std::sync::Arc;

const WG_SIZE: u32 = 256;
/// `DrawIndexedIndirectArgs` is 5 × u32 = 20 bytes: index_count,
/// instance_count, first_index, base_vertex, first_instance.
const INDIRECT_ARGS_SIZE: u64 = 20;
const INDIRECT_INSTANCE_COUNT_OFFSET: u64 = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniforms {
    view_min: [f32; 2],
    view_max: [f32; 2],
    slot_count: u32,
    max_visible: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SortUniforms {
    shift: u32,
    num_blocks: u32,
    _pad0: u32,
    _pad1: u32,
}

/// GPU-side compute pass: culls a growable sprite pool against a view rect
/// and depth-sorts the survivors, entirely on the GPU. See the module doc
/// comment for the full design.
pub struct SpriteCullPass {
    slot_capacity: u32,
    max_visible: u32,
    num_blocks: u32,

    cull_pipeline: wgpu::ComputePipeline,
    cull_uniform_buf: wgpu::Buffer,
    cull_bind_group: wgpu::BindGroup,
    view_min: [f32; 2],
    view_max: [f32; 2],
    view_dirty: bool,

    hist_pipeline: wgpu::ComputePipeline,
    scan_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,

    /// Holds the GPU-copied visible count each frame — see `execute()`.
    count_buf: wgpu::Buffer,
    /// One bind group per digit pass — shift is baked in at construction, and
    /// every buffer reference is fixed for the pass's lifetime, so nothing
    /// here needs rebuilding per frame.
    hist_bind_groups: [wgpu::BindGroup; 4],
    scatter_bind_groups: [wgpu::BindGroup; 4],
    scan_bind_group: wgpu::BindGroup,

    /// Final draw-order buffer after 4 (even) digit passes — always lands
    /// back in the "a" ping-pong buffer; see `radix_sort_indices`'s CPU-side
    /// sibling in `helio-pass-sprite-batch` for the same parity argument.
    pub draw_order_buf: Arc<wgpu::Buffer>,
    /// `DrawIndexedIndirectArgs`, `instance_count` GPU-written by `cs_cull`.
    /// Bind this to `SpriteBatchPass::use_gpu_culling`.
    pub indirect_buf: Arc<wgpu::Buffer>,
}

impl SpriteCullPass {
    /// `instances_buf`/`alive_buf` must be
    /// [`crate::SpriteBatchPass::instances_buffer`]/
    /// [`crate::SpriteBatchPass::alive_buffer`] on the pass this culls for,
    /// sized for at least `slot_capacity` slots. `max_visible` bounds how
    /// many sprites can be visible at once — size it for the worst case your
    /// scene can put on screen simultaneously, not the total pool size.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        instances_buf: Arc<wgpu::Buffer>,
        alive_buf: Arc<wgpu::Buffer>,
        slot_capacity: u32,
        max_visible: u32,
    ) -> Self {
        let num_blocks = max_visible.div_ceil(WG_SIZE).max(1);

        // ── Cull ────────────────────────────────────────────────────────────
        let cull_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sprite Cull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sprite_cull.wgsl").into()),
        });
        let cull_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Cull BGL"),
            entries: &[
                uniform_entry(0),
                storage_entry(1, true),
                storage_entry(2, true),
                storage_entry(3, false),
                storage_entry(4, false),
                storage_entry(5, false),
            ],
        });
        let cull_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Cull PL"),
            bind_group_layouts: &[Some(&cull_bgl)],
            immediate_size: 0,
        });
        let cull_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Cull Pipeline"),
            layout: Some(&cull_pl),
            module: &cull_shader,
            entry_point: Some("cs_cull"),
            compilation_options: Default::default(),
            cache: None,
        });
        let cull_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Cull Uniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let keys_a = create_u32_buffer(device, "Sprite Sort Keys A", max_visible);
        let keys_b = create_u32_buffer(device, "Sprite Sort Keys B", max_visible);
        let indices_a = Arc::new(create_u32_buffer(device, "Sprite Sort Indices A", max_visible));
        let indices_b = create_u32_buffer(device, "Sprite Sort Indices B", max_visible);

        let indirect_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Indirect Draw Args"),
            size: INDIRECT_ARGS_SIZE,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        // Static fields, written once: index_count=6 (one quad), instance_count=0
        // (reset every frame before culling), first_index=0, base_vertex=0,
        // first_instance=0.
        queue.write_buffer(&indirect_buf, 0, bytemuck::cast_slice(&[6u32, 0, 0, 0, 0]));

        let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sprite Cull BG"),
            layout: &cull_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: cull_uniform_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: instances_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: alive_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: indices_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: keys_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: indirect_buf.as_entire_binding() },
            ],
        });

        // ── Sort ────────────────────────────────────────────────────────────
        let sort_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Sprite Sort Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/sprite_sort.wgsl").into()),
        });

        let hist_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Sort Histogram BGL"),
            entries: &[uniform_entry(0), uniform_entry(1), storage_entry(2, true), storage_entry(3, false)],
        });
        let scan_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Sort Scan BGL"),
            entries: &[uniform_entry(0), storage_entry(1, false)],
        });
        let scatter_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Sprite Sort Scatter BGL"),
            entries: &[
                uniform_entry(0),
                uniform_entry(1),
                storage_entry(2, true),
                storage_entry(3, true),
                storage_entry(4, false),
                storage_entry(5, false),
                storage_entry(6, true),
            ],
        });

        let hist_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Sort Histogram PL"),
            bind_group_layouts: &[Some(&hist_bgl)],
            immediate_size: 0,
        });
        let scan_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Sort Scan PL"),
            bind_group_layouts: &[Some(&scan_bgl)],
            immediate_size: 0,
        });
        let scatter_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Sprite Sort Scatter PL"),
            bind_group_layouts: &[Some(&scatter_bgl)],
            immediate_size: 0,
        });

        let hist_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Sort Histogram Pipeline"),
            layout: Some(&hist_pl),
            module: &sort_shader,
            entry_point: Some("cs_histogram"),
            compilation_options: Default::default(),
            cache: None,
        });
        let scan_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Sort Scan Pipeline"),
            layout: Some(&scan_pl),
            module: &sort_shader,
            entry_point: Some("cs_scan"),
            compilation_options: Default::default(),
            cache: None,
        });
        let scatter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Sprite Sort Scatter Pipeline"),
            layout: Some(&scatter_pl),
            module: &sort_shader,
            entry_point: Some("cs_scatter"),
            compilation_options: Default::default(),
            cache: None,
        });

        let block_hist_buf = create_u32_buffer(device, "Sprite Sort Block Histogram", num_blocks * 256);

        // Digit-pass shift/num_blocks are compile-time-fixed given
        // `max_visible`, so their uniform buffers are written once here and
        // never touched again. `count` (the GPU-computed visible count) is
        // dynamic — see `count_buf` below, refreshed once per frame in
        // `execute()`, shared by all four passes.
        let pass_uniforms: [wgpu::Buffer; 4] = std::array::from_fn(|i| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Sprite Sort Pass Uniforms"),
                size: std::mem::size_of::<SortUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let u = SortUniforms { shift: (i as u32) * 8, num_blocks, _pad0: 0, _pad1: 0 };
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(&u));
            buf
        });

        let count_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Sort Visible Count"),
            size: 16, // CountUniform{count:u32} is 4 bytes in WGSL; pad the allocation for backend safety margin.
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Ping-pong role per pass, mirroring the CPU radix sort's parity:
        // pass 0 reads A writes B, pass 1 reads B writes A, pass 2 reads A
        // writes B, pass 3 reads B writes A — so after 4 (even) passes the
        // sorted result is back in A.
        let hist_bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            let src_keys = if i % 2 == 0 { &keys_a } else { &keys_b };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sprite Sort Histogram BG"),
                layout: &hist_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pass_uniforms[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: count_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: src_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: block_hist_buf.as_entire_binding() },
                ],
            })
        });
        let scan_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sprite Sort Scan BG"),
            layout: &scan_bgl,
            // Shift is unused by `cs_scan`; any pass_uniforms entry works —
            // `num_blocks` (the only field it reads) is identical across all four.
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pass_uniforms[0].as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: block_hist_buf.as_entire_binding() },
            ],
        });
        let scatter_bind_groups: [wgpu::BindGroup; 4] = std::array::from_fn(|i| {
            let (src_keys, src_indices, dst_keys, dst_indices) = if i % 2 == 0 {
                (&keys_a, &*indices_a, &keys_b, &indices_b)
            } else {
                (&keys_b, &indices_b, &keys_a, &*indices_a)
            };
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Sprite Sort Scatter BG"),
                layout: &scatter_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: pass_uniforms[i].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: count_buf.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: src_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: src_indices.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: dst_keys.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: dst_indices.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 6, resource: block_hist_buf.as_entire_binding() },
                ],
            })
        });

        Self {
            slot_capacity,
            max_visible,
            num_blocks,
            cull_pipeline,
            cull_uniform_buf,
            cull_bind_group,
            view_min: [0.0, 0.0],
            view_max: [0.0, 0.0],
            view_dirty: true,
            hist_pipeline,
            scan_pipeline,
            scatter_pipeline,
            count_buf,
            hist_bind_groups,
            scatter_bind_groups,
            scan_bind_group,
            draw_order_buf: indices_a,
            indirect_buf,
        }
    }

    /// Sets the world-space view rect sprites are culled against — must
    /// match the paired `SpriteBatchPass::set_camera`'s effective view, or
    /// sprites will be culled against the wrong bounds. Unlike the batch
    /// pass, this rect isn't derived from the render target size
    /// automatically (this pass has no render target to read a size from) —
    /// callers using `half_extent: None` on the batch pass must resize this
    /// to match themselves.
    pub fn set_view_rect(&mut self, center: [f32; 2], half_extent: [f32; 2]) {
        self.view_min = [center[0] - half_extent[0], center[1] - half_extent[1]];
        self.view_max = [center[0] + half_extent[0], center[1] + half_extent[1]];
        self.view_dirty = true;
    }
}

fn create_u32_buffer(device: &wgpu::Device, label: &str, count: u32) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (count.max(1) as u64) * 4,
        // COPY_SRC costs nothing at runtime and is what lets
        // `draw_order_buf` (== `indices_a`) be read back at all, including
        // by `tests/gpu_sort_validation.rs`.
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

impl RenderPass for SpriteCullPass {
    fn name(&self) -> &'static str {
        "SpriteCull"
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a libhelio::FrameResources<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None // compute-only pass
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> Result<()> {
        if self.view_dirty {
            self.view_dirty = false;
            let u = CullUniforms {
                view_min: self.view_min,
                view_max: self.view_max,
                slot_count: self.slot_capacity,
                max_visible: self.max_visible,
                _pad0: 0,
                _pad1: 0,
            };
            ctx.write_buffer(&self.cull_uniform_buf, 0, bytemuck::bytes_of(&u));
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> Result<()> {
        self.record(unsafe { &mut *ctx.encoder_ptr });
        Ok(())
    }
}

impl SpriteCullPass {
    /// Records the cull + 4-pass sort dispatch sequence. Shared by the
    /// `RenderPass::execute()` trait impl and
    /// [`SpriteCullPass::run_once_for_testing`] — the graph-integrated path
    /// and the standalone test path must record identically, or a test pass
    /// proves nothing about the real one.
    fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        // Reset the atomic visible-count / instance_count field to zero
        // before culling. The other four `DrawIndexedIndirectArgs` fields
        // are static and were written once at construction.
        encoder.clear_buffer(&self.indirect_buf, INDIRECT_INSTANCE_COUNT_OFFSET, Some(4));

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpriteCull"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.cull_pipeline);
            pass.set_bind_group(0, &self.cull_bind_group, &[]);
            pass.dispatch_workgroups(self.slot_capacity.div_ceil(WG_SIZE), 1, 1);
        }

        // Propagate the GPU-computed visible count into the sort kernels'
        // bounds check. This is the only thing that varies per digit pass'
        // *count* — dispatch size is always the fixed worst-case
        // `num_blocks` (out-of-range threads no-op via `gid.x < count`), so
        // no indirect dispatch is needed, only this one 4-byte copy.
        encoder.copy_buffer_to_buffer(&self.indirect_buf, INDIRECT_INSTANCE_COUNT_OFFSET, &self.count_buf, 0, 4);

        for i in 0..4 {
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpriteSort Histogram"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.hist_pipeline);
                pass.set_bind_group(0, &self.hist_bind_groups[i], &[]);
                pass.dispatch_workgroups(self.num_blocks, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpriteSort Scan"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scan_pipeline);
                pass.set_bind_group(0, &self.scan_bind_group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpriteSort Scatter"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.scatter_pipeline);
                pass.set_bind_group(0, &self.scatter_bind_groups[i], &[]);
                pass.dispatch_workgroups(self.num_blocks, 1, 1);
            }
        }
    }

    /// Runs cull + sort once, outside a `RenderGraph`, blocking until the
    /// GPU is done. Not for the hot path — this exists for integration tests
    /// that need to read back [`SpriteCullPass::draw_order_buf`]/
    /// [`SpriteCullPass::indirect_buf`] afterward without standing up a full
    /// graph + `PassContext`.
    pub fn run_once_for_testing(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.view_dirty {
            self.view_dirty = false;
            let u = CullUniforms {
                view_min: self.view_min,
                view_max: self.view_max,
                slot_count: self.slot_capacity,
                max_visible: self.max_visible,
                _pad0: 0,
                _pad1: 0,
            };
            queue.write_buffer(&self.cull_uniform_buf, 0, bytemuck::bytes_of(&u));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SpriteCullPass::run_once_for_testing"),
        });
        self.record(&mut encoder);
        queue.submit([encoder.finish()]);
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
    }
}
