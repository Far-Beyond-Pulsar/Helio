//! Per-portal-*chain* GPU frustum culling.
//!
//! For each active portal chain (a sequence of up to `libhelio::MAX_CHAIN_DEPTH`
//! portals — see `libhelio::GpuPortalChain`'s docs for why chains, not single
//! portals, are what makes portals reflect each other automatically), tests
//! every draw-call group's instances — mapped through that chain's *composed*
//! transform — against the main camera frustum, and compacts survivors into
//! two shared output buffers (`portal_indirect_buf` / `portal_compacted_indices_buf`,
//! plus a parallel `portal_compacted_chains_buf` recording which chain each
//! survivor was selected under). `helio-pass-portal-instances` reads all
//! three to draw the duplicated, chain-clipped content — one indirect draw
//! call per draw group, same shape as the ordinary non-portal G-buffer pass,
//! not one per chain.
//!
//! # Two passes: `select` then `finalize`
//!
//! Reserved capacity is sized **per draw group, not per chain** — chains can
//! number in the hundreds (`portal_count^depth`) while draw groups stay in
//! the dozens to low hundreds for realistic scenes, and only a handful of
//! chains ever have any given group's content in frustum simultaneously,
//! so reserving per-chain capacity would multiply the wrong axis. Because
//! several chains' `select` workgroups append into the *same* draw group's
//! region concurrently, the final survivor count isn't known until they've
//! all finished — `finalize` is a tiny second dispatch (one thread per draw
//! group) that turns the settled count into a `DrawIndexedIndirect`.
//!
//! # Buffers produced
//!
//! | Buffer                          | Format                                                      |
//! |----------------------------------|--------------------------------------------------------------|
//! | `portal_indirect_buf`            | `PORTAL_DRAW_CAPACITY × 20` bytes                             |
//! | `portal_compacted_indices_buf`   | `PORTAL_DRAW_CAPACITY × PORTAL_GROUP_CHAIN_CAPACITY × 4` bytes |
//! | `portal_compacted_chains_buf`    | same shape as `portal_compacted_indices_buf`                  |
//!
//! All fixed-size, allocated once — not resized to track scene growth —
//! mirroring `helio-pass-shadow-cull`'s own atlas buffers. A scene exceeding
//! a cap silently drops the excess (see `portal_cull.wgsl`'s bounds checks)
//! rather than corrupting adjacent memory.
//!
//! # Integration
//!
//! ```ignore
//! let cull_pass = PortalCullPass::new(device);
//! let indirect_buf = Arc::clone(&cull_pass.portal_indirect_buf);
//! let compacted_buf = Arc::clone(&cull_pass.portal_compacted_indices_buf);
//! let compacted_chains_buf = Arc::clone(&cull_pass.portal_compacted_chains_buf);
//! graph.add_pass(Box::new(cull_pass));
//!
//! graph.add_pass(Box::new(
//!     PortalInstancePass::new(device, indirect_buf, compacted_buf, compacted_chains_buf),
//! ));
//! ```

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Fixed cap on draw-call groups considered. Realistic scenes have dozens to
/// low hundreds of distinct mesh+material combinations; this is generous
/// headroom, not a tight budget.
pub const PORTAL_DRAW_CAPACITY: u32 = 512;

/// How many (instance, chain) survivor slots are reserved **per draw
/// group** — not per chain, see the module doc for why that's the right
/// axis. `PORTAL_DRAW_CAPACITY × PORTAL_GROUP_CHAIN_CAPACITY` is the actual
/// total allocation, so this is deliberately modest.
pub const PORTAL_GROUP_CHAIN_CAPACITY: u32 = 1024;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniforms {
    frustum_planes: [[f32; 4]; 6],
    draw_count: u32,
    chain_count: u32,
    group_capacity: u32,
    _pad: u32,
}

pub struct PortalCullPass {
    select_pipeline: wgpu::ComputePipeline,
    finalize_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,

    /// One `DrawIndexedIndirect` per draw group (not per chain).
    pub portal_indirect_buf: Arc<wgpu::Buffer>,

    /// Shared compacted original-instance-slot buffer. Draw group `g`'s
    /// region is `[g * PORTAL_GROUP_CHAIN_CAPACITY, (g+1) * PORTAL_GROUP_CHAIN_CAPACITY)`.
    pub portal_compacted_indices_buf: Arc<wgpu::Buffer>,

    /// Parallel to `portal_compacted_indices_buf` — which chain each
    /// compacted entry was selected under.
    pub portal_compacted_chains_buf: Arc<wgpu::Buffer>,

    /// Per-draw-group selected-instance totals (across all chains
    /// combined), zeroed every frame by the CPU. Doubles as the atomic
    /// counter `select` claims write slots from and the diagnostic readback
    /// confirming the cull selects content.
    portal_stats_buf: wgpu::Buffer,
    portal_stats_staging: wgpu::Buffer,
    copy_pending: bool,

    bind_group: Option<wgpu::BindGroup>,
    /// (camera, instances, draw_calls, coordinate_spaces, portal_views, portal_chains)
    bind_group_key: Option<(usize, usize, usize, usize, usize, usize)>,

    draw_count: u32,
    chain_count: u32,
}

impl PortalCullPass {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PortalCull Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/portal_cull.wgsl").into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull Uniforms"),
            size: std::mem::size_of::<CullUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let portal_indirect_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/PortalIndirect"),
            size: (PORTAL_DRAW_CAPACITY as u64) * 20,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let compacted_len = (PORTAL_DRAW_CAPACITY as u64) * (PORTAL_GROUP_CHAIN_CAPACITY as u64);
        let portal_compacted_indices_buf =
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PortalCull/CompactedIndices"),
                size: compacted_len * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }));
        let portal_compacted_chains_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/CompactedChains"),
            size: compacted_len * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let portal_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/Stats"),
            size: (PORTAL_DRAW_CAPACITY as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let portal_stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/StatsStaging"),
            size: (PORTAL_DRAW_CAPACITY as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PortalCull BGL"),
            entries: &[
                storage_entry(0, true), // camera
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
                storage_entry(2, true),   // instances
                storage_entry(3, true),   // draw_calls
                storage_entry(4, true),   // coordinate_spaces
                storage_entry(5, true),   // portal_views
                storage_entry(6, false),  // portal_indirect
                storage_entry(7, false),  // portal_compacted_indices
                storage_entry(8, false),  // portal_stats
                storage_entry(9, true),   // portal_chains
                storage_entry(10, false), // portal_compacted_chains
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PortalCull PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let select_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PortalCull Select Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("select"),
            compilation_options: Default::default(),
            cache: None,
        });
        let finalize_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PortalCull Finalize Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("finalize"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            select_pipeline,
            finalize_pipeline,
            bind_group_layout,
            uniform_buf,
            portal_indirect_buf,
            portal_compacted_indices_buf,
            portal_compacted_chains_buf,
            portal_stats_buf,
            portal_stats_staging,
            copy_pending: false,
            bind_group: None,
            bind_group_key: None,
            draw_count: 0,
            chain_count: 0,
        }
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

impl RenderPass for PortalCullPass {
    fn name(&self) -> &'static str {
        "PortalCull"
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
        self.draw_count = ctx.scene.draw_calls.len() as u32;
        self.chain_count = ctx.scene.portal_chains.len() as u32;
        let planes = extract_frustum_planes(ctx.scene.camera.data().view_proj);

        let uniforms = CullUniforms {
            frustum_planes: planes,
            draw_count: self.draw_count,
            chain_count: self.chain_count,
            group_capacity: PORTAL_GROUP_CHAIN_CAPACITY,
            _pad: 0,
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        // Zero the per-group stats/counters before this frame's dispatch —
        // `select` uses these as its live atomic claim counters, and the
        // readback below needs exactly this frame's totals.
        let zeros = vec![0u32; PORTAL_DRAW_CAPACITY as usize];
        ctx.queue
            .write_buffer(&self.portal_stats_buf, 0, bytemuck::cast_slice(&zeros));
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if ctx.frame_num < 3 || ctx.frame_num % 120 == 0 {
            log::info!(
                "[PortalCull] frame={} draw_count={} chain_count={}",
                ctx.frame_num,
                self.draw_count,
                self.chain_count,
            );
        }
        if self.draw_count == 0 || self.chain_count == 0 {
            return Ok(());
        }

        let key = (
            ctx.scene.camera as *const wgpu::Buffer as usize,
            ctx.scene.instances as *const wgpu::Buffer as usize,
            ctx.scene.draw_calls as *const wgpu::Buffer as usize,
            ctx.scene.coordinate_spaces as *const wgpu::Buffer as usize,
            ctx.scene.portal_views as *const wgpu::Buffer as usize,
            ctx.scene.portal_chains as *const wgpu::Buffer as usize,
        );
        if self.bind_group_key != Some(key) {
            self.bind_group = Some(ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("PortalCull BG"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ctx.scene.camera.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: self.uniform_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: ctx.scene.instances.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: ctx.scene.draw_calls.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: ctx.scene.coordinate_spaces.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: ctx.scene.portal_views.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 6,
                        resource: self.portal_indirect_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 7,
                        resource: self.portal_compacted_indices_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 8,
                        resource: self.portal_stats_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 9,
                        resource: ctx.scene.portal_chains.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 10,
                        resource: self.portal_compacted_chains_buf.as_entire_binding(),
                    },
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let draw_workgroups = self.draw_count.min(PORTAL_DRAW_CAPACITY);
        let chain_workgroups = self.chain_count;

        {
            let mut pass =
                unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PortalCull Select"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.select_pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(draw_workgroups, chain_workgroups, 1);
        }
        {
            let mut pass =
                unsafe { &mut *ctx.encoder_ptr }.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("PortalCull Finalize"),
                    timestamp_writes: None,
                });
            pass.set_pipeline(&self.finalize_pipeline);
            pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(draw_workgroups.div_ceil(64), 1, 1);
        }

        // ── Diagnostic readback: every 60 frames, copy the per-group
        // selected-instance totals off the GPU and log them on the next
        // frame (after this encoder has been submitted). Tells us whether
        // the cull is actually selecting anything, without touching the
        // frame's real data flow.
        if self.copy_pending {
            self.copy_pending = false;
            let completion = std::sync::Arc::new(std::sync::Mutex::new(None));
            let callback_completion = std::sync::Arc::clone(&completion);
            self.portal_stats_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    *callback_completion.lock().unwrap() = Some(result);
                });
            ctx.device.poll(wgpu::PollType::wait_indefinitely());
            let result = completion.lock().unwrap().take();
            match result {
                Some(Ok(())) => {
                    let data = self
                        .portal_stats_staging
                        .slice(..)
                        .get_mapped_range()
                        .expect("portal stats mapped range");
                    let mut counts = Vec::with_capacity(
                        (self.draw_count as usize).min(PORTAL_DRAW_CAPACITY as usize),
                    );
                    for chunk in data.chunks_exact(4).take(self.draw_count as usize) {
                        counts.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    log::info!("[PortalCull] per-group selected-instance counts (across all chains): {counts:?}");
                    drop(data);
                    self.portal_stats_staging.unmap();
                }
                _ => log::warn!("[PortalCull] stats readback map failed"),
            }
        }
        if ctx.frame_num % 60 == 0 {
            let size = (PORTAL_DRAW_CAPACITY as u64) * 4;
            unsafe { &mut *ctx.encoder_ptr }.copy_buffer_to_buffer(
                &self.portal_stats_buf,
                0,
                &self.portal_stats_staging,
                0,
                size,
            );
            self.copy_pending = true;
        }
        Ok(())
    }
}

/// Extract 6 frustum planes from a view-projection matrix (Gribb/Hartmann method).
/// Identical to `helio-pass-indirect-dispatch`'s own copy — see there for the
/// full derivation notes; duplicated per this codebase's established
/// per-pass-duplication convention for small shared helpers.
fn extract_frustum_planes(vp: [f32; 16]) -> [[f32; 4]; 6] {
    let row = |r: usize| -> [f32; 4] { [vp[r], vp[4 + r], vp[8 + r], vp[12 + r]] };
    let r0 = row(0);
    let r1 = row(1);
    let r2 = row(2);
    let r3 = row(3);
    let add = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    };
    let sub = |a: [f32; 4], b: [f32; 4]| -> [f32; 4] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    };
    let normalize = |p: [f32; 4]| -> [f32; 4] {
        let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if len > 1e-10 {
            [p[0] / len, p[1] / len, p[2] / len, p[3] / len]
        } else {
            p
        }
    };
    [
        normalize(add(r3, r0)),
        normalize(sub(r3, r0)),
        normalize(add(r3, r1)),
        normalize(sub(r3, r1)),
        normalize(r2),
        normalize(sub(r3, r2)),
    ]
}
