//! Per-portal GPU frustum culling.
//!
//! For each active portal, tests every draw-call group's instances — mapped
//! through that portal's `pair_map_inverse` coordinate space — against the
//! main camera frustum, and compacts survivors into a portal-private slice of
//! two shared output buffers (`portal_indirect_buf` / `portal_compacted_indices_buf`).
//! `helio-pass-portal-instances` reads these to draw the duplicated,
//! portal-clipped content.
//!
//! # Buffers produced
//!
//! | Buffer                          | Format                                                    |
//! |----------------------------------|-----------------------------------------------------------|
//! | `portal_indirect_buf`            | `MAX_PORTAL_SLOTS × PORTAL_DRAW_CAPACITY × 20` bytes       |
//! | `portal_compacted_indices_buf`   | `MAX_PORTAL_SLOTS × PORTAL_INSTANCE_CAPACITY × 4` bytes    |
//!
//! Both are fixed-size, allocated once — not resized to track scene growth —
//! mirroring `helio-pass-shadow-cull`'s own atlas buffers. A scene exceeding
//! either cap silently drops the excess (see `portal_cull.wgsl`'s
//! `arrayLength()` bounds checks) rather than corrupting adjacent memory;
//! both caps are generous enough that this is not expected in normal use.
//!
//! # Integration
//!
//! ```ignore
//! let cull_pass = PortalCullPass::new(device);
//! let indirect_buf = Arc::clone(&cull_pass.portal_indirect_buf);
//! let compacted_buf = Arc::clone(&cull_pass.portal_compacted_indices_buf);
//! graph.add_pass(Box::new(cull_pass));
//!
//! graph.add_pass(Box::new(PortalInstancePass::new(device, indirect_buf, compacted_buf)));
//! ```

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use helio_core::{PassContext, PrepareContext, RenderPass, Result as HelioResult};

/// Coordinate-space slots (shared with sublevels) — see `libhelio::MAX_COORDINATE_SPACES`.
/// Slot 0 (world space) never has a portal, so one slice is always unused;
/// not worth special-casing away.
pub const MAX_PORTAL_SLOTS: u32 = libhelio::MAX_COORDINATE_SPACES;

/// Fixed cap on draw-call groups considered per portal. See module docs.
pub const PORTAL_DRAW_CAPACITY: u32 = 4096;

/// Fixed cap on instance slots considered per portal. See module docs.
pub const PORTAL_INSTANCE_CAPACITY: u32 = 65536;

/// Diagnostic probe sizes (see the readback in `execute`): how many leading
/// indirect commands and compacted slots per portal to copy back and log.
const PROBE_DRAWS: u64 = 8;
const PROBE_INDICES: u64 = 32;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CullUniforms {
    frustum_planes: [[f32; 4]; 6],
    draw_count: u32,
    portal_count: u32,
    draw_capacity: u32,
    instance_capacity: u32,
}

pub struct PortalCullPass {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buf: wgpu::Buffer,

    /// Per-portal compacted indirect draw commands. Slice `p` (draws for
    /// portal coordinate-space slot `p`) starts at byte offset
    /// `p * PORTAL_DRAW_CAPACITY * 20`.
    pub portal_indirect_buf: Arc<wgpu::Buffer>,

    /// Per-portal compacted original instance slots. Slice `p` starts at
    /// element offset `p * PORTAL_INSTANCE_CAPACITY`.
    pub portal_compacted_indices_buf: Arc<wgpu::Buffer>,

    /// Per-portal selected-instance totals, zeroed every frame by the CPU.
    /// Diagnostic readback (see `execute`) — confirms the cull selects content.
    portal_stats_buf: wgpu::Buffer,
    portal_stats_staging: wgpu::Buffer,
    /// Diagnostic: first few indirect commands + compacted slots per portal.
    portal_probe_buf: wgpu::Buffer,
    portal_probe_staging: wgpu::Buffer,
    copy_pending: bool,
    probe_pending: bool,

    bind_group: Option<wgpu::BindGroup>,
    /// (camera, instances, draw_calls, coordinate_spaces, portal_views)
    bind_group_key: Option<(usize, usize, usize, usize, usize)>,

    draw_count: u32,
    portal_count: u32,
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
            size: (MAX_PORTAL_SLOTS as u64) * (PORTAL_DRAW_CAPACITY as u64) * 20,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let portal_compacted_indices_buf = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/CompactedIndices"),
            size: (MAX_PORTAL_SLOTS as u64) * (PORTAL_INSTANCE_CAPACITY as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));
        let portal_stats_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/Stats"),
            size: (MAX_PORTAL_SLOTS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let portal_stats_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/StatsStaging"),
            size: (MAX_PORTAL_SLOTS as u64) * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        // Diagnostic probe: per portal, PROBE_DRAWS indirect commands then
        // PROBE_INDICES compacted slots, laid out as
        // `portal * (PROBE_DRAWS*20 + PROBE_INDICES*4)`.
        let portal_probe_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/Probe"),
            size: (MAX_PORTAL_SLOTS as u64) * (PROBE_DRAWS * 20 + PROBE_INDICES * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let portal_probe_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PortalCull/ProbeStaging"),
            size: (MAX_PORTAL_SLOTS as u64) * (PROBE_DRAWS * 20 + PROBE_INDICES * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PortalCull BGL"),
            entries: &[
                storage_entry(0, true),  // camera
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
                storage_entry(2, true),  // instances
                storage_entry(3, true),  // draw_calls
                storage_entry(4, true),  // coordinate_spaces
                storage_entry(5, true),  // portal_views
                storage_entry(6, false), // portal_indirect
                storage_entry(7, false), // portal_compacted_indices
                storage_entry(8, false), // portal_stats
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PortalCull PL"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PortalCull Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buf,
            portal_indirect_buf,
            portal_compacted_indices_buf,
            portal_stats_buf,
            portal_stats_staging,
            portal_probe_buf,
            portal_probe_staging,
            copy_pending: false,
            probe_pending: false,
            bind_group: None,
            bind_group_key: None,
            draw_count: 0,
            portal_count: 0,
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
        self.portal_count = ctx.scene.portal_views.len() as u32;
        let planes = extract_frustum_planes(ctx.scene.camera.data().view_proj);

        let uniforms = CullUniforms {
            frustum_planes: planes,
            draw_count: self.draw_count,
            portal_count: self.portal_count,
            draw_capacity: PORTAL_DRAW_CAPACITY,
            instance_capacity: PORTAL_INSTANCE_CAPACITY,
        };
        ctx.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        // Zero the per-portal stats before this frame's dispatch so the
        // readback reflects exactly this frame's selections.
        let zeros = [0u32; MAX_PORTAL_SLOTS as usize];
        ctx.queue.write_buffer(
            &self.portal_stats_buf,
            0,
            bytemuck::cast_slice(&zeros),
        );
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        if ctx.frame_num < 3 || ctx.frame_num % 120 == 0 {
            log::info!(
                "[PortalCull] frame={} draw_count={} portal_count={}",
                ctx.frame_num, self.draw_count, self.portal_count,
            );
        }
        if self.draw_count == 0 || self.portal_count == 0 {
            return Ok(());
        }

        let key = (
            ctx.scene.camera as *const wgpu::Buffer as usize,
            ctx.scene.instances as *const wgpu::Buffer as usize,
            ctx.scene.draw_calls as *const wgpu::Buffer as usize,
            ctx.scene.coordinate_spaces as *const wgpu::Buffer as usize,
            ctx.scene.portal_views as *const wgpu::Buffer as usize,
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
                ],
            }));
            self.bind_group_key = Some(key);
        }

        let draw_workgroups = self.draw_count.min(PORTAL_DRAW_CAPACITY);
        let portal_workgroups = self.portal_count.min(MAX_PORTAL_SLOTS);

        let mut pass = unsafe { &mut *ctx.encoder_ptr }
            .begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PortalCull"),
                timestamp_writes: None,
            });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
        pass.dispatch_workgroups(draw_workgroups, portal_workgroups, 1);
        drop(pass);

        // ── Diagnostic readback: every 60 frames, copy the per-portal
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
                    let mut counts = Vec::with_capacity(MAX_PORTAL_SLOTS as usize);
                    for chunk in data.chunks_exact(4) {
                        counts.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                    }
                    log::info!("[PortalCull] per-portal selected-instance counts: {counts:?}");
                    drop(data);
                    self.portal_stats_staging.unmap();
                }
                _ => log::warn!("[PortalCull] stats readback map failed"),
            }
            self.probe_pending = true;
        }
        if self.probe_pending {
            self.probe_pending = false;
            let completion = std::sync::Arc::new(std::sync::Mutex::new(None));
            let callback_completion = std::sync::Arc::clone(&completion);
            self.portal_probe_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |result| {
                    *callback_completion.lock().unwrap() = Some(result);
                });
            ctx.device.poll(wgpu::PollType::wait_indefinitely());
            let result = completion.lock().unwrap().take();
            match result {
                Some(Ok(())) => {
                    let data = self
                        .portal_probe_staging
                        .slice(..)
                        .get_mapped_range()
                        .expect("portal probe mapped range");
                    let draws = PROBE_DRAWS as usize;
                    let indices = PROBE_INDICES as usize;
                    let portal_stride = draws * 20 + indices * 4;
                    let portals = (self.portal_count as usize).min(MAX_PORTAL_SLOTS as usize);
                    for p in 0..portals {
                        let base = p * portal_stride;
                        let mut cmd = Vec::new();
                        for d in 0..draws {
                            let o = base + d * 20;
                            let read = |i: usize| u32::from_le_bytes([
                                data[o + i * 4], data[o + i * 4 + 1], data[o + i * 4 + 2], data[o + i * 4 + 3],
                            ]);
                            cmd.push((read(0), read(1), read(2), read(3), read(4)));
                        }
                        log::info!(
                            "[PortalCull] portal[{p}] indirect (index_count, instance_count, first_index, base_vertex, first_instance): {cmd:?}"
                        );
                        let mut slots = Vec::new();
                        for i in 0..indices {
                            let o = base + draws * 20 + i * 4;
                            slots.push(u32::from_le_bytes([
                                data[o], data[o + 1], data[o + 2], data[o + 3],
                            ]));
                        }
                        log::info!("[PortalCull] portal[{p}] compacted slots (first {indices}): {slots:?}");
                    }
                    drop(data);
                    self.portal_probe_staging.unmap();
                }
                _ => log::warn!("[PortalCull] probe readback map failed"),
            }
        }
        if ctx.frame_num % 60 == 0 {
            let size = (MAX_PORTAL_SLOTS as u64) * 4;
            unsafe { &mut *ctx.encoder_ptr }
                .copy_buffer_to_buffer(&self.portal_stats_buf, 0, &self.portal_stats_staging, 0, size);
            self.copy_pending = true;
        }
        if ctx.frame_num % 60 == 0 {
            let probes = (self.portal_count as u64).min(MAX_PORTAL_SLOTS as u64);
            let byte_len = probes * (PROBE_DRAWS * 20 + PROBE_INDICES * 4);
            let mut encoder = unsafe { &mut *ctx.encoder_ptr };
            for p in 0..probes {
                // Indirect commands: portal p's slice, first PROBE_DRAWS entries.
                let src = p * (PORTAL_DRAW_CAPACITY as u64) * 20;
                let dst = p * (PROBE_DRAWS * 20 + PROBE_INDICES * 4);
                encoder.copy_buffer_to_buffer(
                    &self.portal_indirect_buf,
                    src,
                    &self.portal_probe_buf,
                    dst,
                    PROBE_DRAWS * 20,
                );
                // Compacted slots: portal p's slice, first PROBE_INDICES slots.
                let src = p * (PORTAL_INSTANCE_CAPACITY as u64) * 4;
                let dst = p * (PROBE_DRAWS * 20 + PROBE_INDICES * 4) + PROBE_DRAWS * 20;
                encoder.copy_buffer_to_buffer(
                    &self.portal_compacted_indices_buf,
                    src,
                    &self.portal_probe_buf,
                    dst,
                    PROBE_INDICES * 4,
                );
            }
            encoder.copy_buffer_to_buffer(&self.portal_probe_buf, 0, &self.portal_probe_staging, 0, byte_len);
            self.probe_pending = true;
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
