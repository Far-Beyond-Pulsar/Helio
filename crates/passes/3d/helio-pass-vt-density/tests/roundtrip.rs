//! T3 — FEEDBACK ROUNDTRIP (Helio#238 issue test 3).
//!
//! The full frame path is: five raster fragment stages write packed
//! `(slot << 8) | (mip+1)` words into the quarter-res density target beside
//! their samples (`vt_feedback_write`), then `VtDensityCompactPass` reduces
//! the finished target to one per-slot maximum. This test exercises the REAL
//! kernels end-to-end on a live device — `vt_feedback.wgsl`'s `cs_clear` and
//! `cs_compact`, byte-for-byte the source the passes ship — feeding them
//! known cell writes through a stand-in writer kernel (the writers themselves
//! are five copies of one two-line `textureStore(vt_density, cell,
//! vec4<u32>(vt_pack(...)))`, whose pack math is pinned bit-for-bit against
//! `libhelio::pack_feedback` by helio-core's module tests).
//!
//! Asserted here, exactly:
//! - a cleared target compacts to ALL-ZERO output (sentinel survives the whole
//!   pipeline; blank cells never fabricate slot-0/mip-0 demand);
//! - known writes land as EXACT per-slot maxima regardless of which cell held
//!   them, including multi-slot mixing and the base-mip special case
//!   `pack(0, 0) == 1`;
//! - max semantics: several cells of one slot reduce to the LARGEST wanted
//!   mip (the +1 bias keeps pack order == mip order);
//! - the out-of-range-slot guard (`slot >= 256`) drops the store AND does not
//!   count it in the touched-store counter;
//! - the touched-store counter equals exactly the number of in-range stores.
//!
//! Headless machines skip gracefully (no adapter ⇒ no assertion).

use std::sync::Arc;

use helio_pass_vt_density::{FEEDBACK_BUFFER_WORDS, FEEDBACK_SLOT_COUNT};
use wgpu::util::DeviceExt as _;

/// One stand-in fragment feedback store: quarter-res cell + packed word.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CellWrite {
    cell: [u32; 2],
    word: u32,
    _pad: u32,
}

fn device_or_skip(label: &str) -> Option<(Arc<wgpu::Device>, Arc<wgpu::Queue>)> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    })) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("skipping: no GPU adapter (headless)");
            return None;
        }
    };
    match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some(label),
        ..Default::default()
    })) {
        Ok((d, q)) => Some((Arc::new(d), Arc::new(q))),
        Err(_) => {
            eprintln!("skipping: no device");
            None
        }
    }
}

/// `libhelio::pack_feedback` — duplicated inline because this crate cannot see
/// libhelio's copy without dragging the whole renderer facade into a pass
/// crate; helio-core's `vt_declares_the_shared_contract_surface` +
/// libhelio's own unit tests pin the two together bit-for-bit.
fn pack(slot: u32, wanted_mip: u32) -> u32 {
    (slot << 8) | (wanted_mip.min(254) + 1)
}

struct Harness {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    /// Quarter-res target dimensions for this scenario.
    qw: u32,
    qh: u32,
    density: wgpu::Texture,
    clear_pipeline: wgpu::ComputePipeline,
    compact_pipeline: wgpu::ComputePipeline,
    write_pipeline: wgpu::ComputePipeline,
    bgl_clear: wgpu::BindGroupLayout,
    bgl_compact: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    bgl_write: wgpu::BindGroupLayout,
}

const WORKGROUP: u32 = 8;

impl Harness {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>, qw: u32, qh: u32) -> Self {
        let module = helio_core::shader::module(
            &device,
            "vt_feedback_test",
            include_str!("../shaders/vt_feedback.wgsl"),
        );
        // Per-entry-point layouts mirror the pass's `clear_group0_bgl` /
        // `compact_group0_bgl`: bind groups must fill their layout fully, so
        // clear (uniform @0 + write view @1) and compact (uniform @2 + read
        // view @3) each get exactly their own two entries.
        let bgl_clear = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("T3 BGL Clear"),
            entries: &[
                uniform(0),
                storage_texture(1, wgpu::StorageTextureAccess::WriteOnly),
            ],
        });
        let bgl_compact = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("T3 BGL Compact"),
            entries: &[
                uniform(2),
                storage_texture(3, wgpu::StorageTextureAccess::ReadOnly),
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("T3 BGL1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let mk_pipe = |label: &str, entry: &'static str| {
            let groups: [Option<&wgpu::BindGroupLayout>; 2] = [Some(&bgl_compact), Some(&bgl1)];
            let single: [Option<&wgpu::BindGroupLayout>; 1] = [Some(&bgl_clear)];
            let layouts: &[Option<&wgpu::BindGroupLayout>] =
                if entry == "cs_compact" { &groups } else { &single };
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                immediate_size: 0,
                bind_group_layouts: layouts,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let clear_pipeline = mk_pipe("T3 Clear", "cs_clear");
        let compact_pipeline = mk_pipe("T3 Compact", "cs_compact");

        // Stand-in writer: one invocation per CellWrite triple.
        let writer_src = r#"
struct CellWrite { cell: vec2<u32>, word: u32, _pad: u32 };
@group(0) @binding(0) var dst: texture_storage_2d<r32uint, write>;
@group(0) @binding(1) var<storage, read> writes: array<CellWrite>;
@compute @workgroup_size(1, 1, 1)
fn cs_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= arrayLength(&writes) { return; }
    let w = writes[gid.x];
    textureStore(dst, w.cell, vec4<u32>(w.word, 0u, 0u, 0u));
}
"#;
        let write_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("T3 Writer"),
            source: wgpu::ShaderSource::Wgsl(writer_src.into()),
        });
        let bgl_write = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("T3 BGL Write"),
            entries: &[
                storage_texture(0, wgpu::StorageTextureAccess::WriteOnly),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
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
        let write_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("T3 Write PL"),
            immediate_size: 0,
            bind_group_layouts: &[Some(&bgl_write)],
        });
        let write_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("T3 Write"),
                layout: Some(&write_pl),
                module: &write_module,
                entry_point: Some("cs_write"),
                compilation_options: Default::default(),
                cache: None,
            });

        let density = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("T3 Density"),
            size: wgpu::Extent3d { width: qw, height: qh, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });

        Self {
            device,
            queue,
            qw,
            qh,
            density,
            clear_pipeline,
            compact_pipeline,
            write_pipeline,
            bgl_clear,
            bgl_compact,
            bgl1,
            bgl_write,
        }
    }

    fn dims_buf(&self) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T3 Dims"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Runs ONE full frame: clear → stand-in writes → compact → mapped readback.
    fn run_frame(&self, writes: &[CellWrite]) -> Vec<u32> {
        let dims = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("T3 Dims Init"),
            contents: bytemuck::bytes_of(&[self.qw, self.qh, 0u32, 0u32]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T3 Out"),
            size: FEEDBACK_BUFFER_WORDS * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero the output like the pass's prepare() does every frame.
        self.queue.write_buffer(&out_buf, 0, &vec![0u8; (FEEDBACK_BUFFER_WORDS * 4) as usize]);

        let write_view = self
            .density
            .create_view(&wgpu::TextureViewDescriptor {
                format: Some(wgpu::TextureFormat::R32Uint),
                dimension: Some(wgpu::TextureViewDimension::D2),
                usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
                ..Default::default()
            });
        let read_view = self.density.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R32Uint),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            ..Default::default()
        });

        let bg_clear = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("T3 BG Clear"),
            layout: &self.bgl_clear,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dims.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&write_view),
                },
            ],
        });
        let bg_compact0 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("T3 BG Compact0"),
            layout: &self.bgl_compact,
            entries: &[
                wgpu::BindGroupEntry { binding: 2, resource: dims.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&read_view),
                },
            ],
        });
        let bg_compact1 = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("T3 BG Compact1"),
            layout: &self.bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            }],
        });

        // Stand-in writer bindings exist only when there IS something to write:
        // a zero-sized storage buffer is a bind-group validation error.
        let writes_buf = (!writes.is_empty()).then(|| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("T3 Writes"),
                contents: bytemuck::cast_slice(writes),
                usage: wgpu::BufferUsages::STORAGE,
            })
        });
        let bg_write = writes_buf.as_ref().map(|buf| {
            self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("T3 BG Write"),
                layout: &self.bgl_write,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&write_view),
                    },
                    wgpu::BindGroupEntry { binding: 1, resource: buf.as_entire_binding() },
                ],
            })
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("T3 Frame") });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("T3 Frame Pass"),
                timestamp_writes: None,
            });
            let (gx, gy) = (
                self.qw.div_ceil(WORKGROUP).max(1),
                self.qh.div_ceil(WORKGROUP).max(1),
            );
            cp.set_pipeline(&self.clear_pipeline);
            cp.set_bind_group(0, &bg_clear, &[]);
            cp.dispatch_workgroups(gx, gy, 1);

            if let (Some(bg_write), false) = (&bg_write, writes.is_empty()) {
                cp.set_pipeline(&self.write_pipeline);
                cp.set_bind_group(0, bg_write, &[]);
                cp.dispatch_workgroups(writes.len() as u32, 1, 1);
            }

            cp.set_pipeline(&self.compact_pipeline);
            cp.set_bind_group(0, &bg_compact0, &[]);
            cp.set_bind_group(1, &bg_compact1, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit([encoder.finish()]);

        // Readback.
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("T3 Staging"),
            size: FEEDBACK_BUFFER_WORDS * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("T3 Copy") });
        enc.copy_buffer_to_buffer(
            &out_buf,
            0,
            &staging,
            0,
            FEEDBACK_BUFFER_WORDS * 4,
        );
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let map = {
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).expect("map callback channel");
            });
            self.device.poll(wgpu::PollType::wait_indefinitely());
            rx.recv().expect("map result").expect("staging maps");
            let mapped = slice.get_mapped_range().expect("mapped range");
            mapped.to_vec()
        };
        staging.unmap();
        bytemuck::cast_slice::<u8, u32>(&map).to_vec()
    }
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

fn storage_texture(binding: u32, access: wgpu::StorageTextureAccess) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access,
            format: wgpu::TextureFormat::R32Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

fn slot_word(words: &[u32], slot: usize) -> u32 {
    words[slot]
}

#[test]
fn cleared_target_compacts_to_all_zero_sentinel() {
    let Some((device, queue)) = device_or_skip("T3 cleared roundtrip") else { return };
    let h = Harness::new(device, queue, 64, 48);
    let words = h.run_frame(&[]);
    assert!(
        words[..FEEDBACK_SLOT_COUNT as usize].iter().all(|&w| w == 0),
        "a cleared frame must compact to untouched sentinels"
    );
    assert_eq!(words[FEEDBACK_SLOT_COUNT as usize], 0, "zero touched stores");
}

#[test]
fn known_writes_roundtrip_to_exact_per_slot_maxima() {
    let Some((device, queue)) = device_or_skip("T3 exact maxima") else { return };
    let h = Harness::new(device, queue, 64, 48);

    // Mixed slots, scattered cells, including the base-mip special case
    // pack(0,0)=1 and a multi-cell single-slot max.
    let writes = vec![
        CellWrite { cell: [0, 0], word: pack(3, 2), _pad: 0 },
        CellWrite { cell: [7, 3], word: pack(3, 5), _pad: 0 }, // same slot, finer
        CellWrite { cell: [2, 2], word: pack(0, 0), _pad: 0 }, // == 1: sentinel-adjacent
        CellWrite { cell: [60, 40], word: pack(255, 11), _pad: 0 },
        CellWrite { cell: [13, 9], word: pack(3, 4), _pad: 0 }, // coarser than mip 5
    ];
    let words = h.run_frame(&writes);

    assert_eq!(words[FEEDBACK_SLOT_COUNT as usize], 5, "every in-range store counted");
    assert_eq!(words[3], pack(3, 5), "slot 3 reduces to its FINEST demanded mip");
    assert_eq!(words[0], pack(0, 0), "base-mip demand packs to exactly 1");
    assert_eq!(words[255], pack(255, 11), "the last bindless slot round-trips");
    for (slot, &word) in words.iter().enumerate().take(FEEDBACK_SLOT_COUNT as usize) {
        if ![0usize, 3usize, 255].contains(&slot) {
            assert_eq!(word, 0, "untouched slot {slot} must stay sentinel");
        }
    }
}

#[test]
fn out_of_range_slots_are_dropped_and_not_counted() {
    let Some((device, queue)) = device_or_skip("T3 range guard") else { return };
    let h = Harness::new(device, queue, 32, 32);
    let writes = vec![
        CellWrite { cell: [1, 1], word: pack(300, 2), _pad: 0 }, // slot 300: dropped
        CellWrite { cell: [2, 2], word: pack(4, 3), _pad: 0 },   // in range
    ];
    let words = h.run_frame(&writes);
    assert_eq!(words[FEEDBACK_SLOT_COUNT as usize], 1, "only the in-range store counts");
    assert_eq!(words[4], pack(4, 3));
    assert!(
        words[..FEEDBACK_SLOT_COUNT as usize].iter().enumerate().all(|(s, &w)| s == 4 || w == 0),
        "dropped stores must not fabricate any slot"
    );
    let _ = slot_word(&words, 4); // keep helper referenced for readability parity
}

#[test]
fn reruns_after_clear_are_independent_frames() {
    let Some((device, queue)) = device_or_skip("T3 frame independence") else { return };
    let h = Harness::new(device, queue, 32, 32);
    let loud = vec![CellWrite { cell: [5, 5], word: pack(9, 7), _pad: 0 }];
    let words = h.run_frame(&loud);
    assert_eq!(words[9], pack(9, 7));
    // Next frame starts from a fresh clear: previous demand must not leak.
    let words = h.run_frame(&[]);
    assert_eq!(words[9], 0, "frame N+1 must not inherit frame N's demand");
    assert_eq!(words[FEEDBACK_SLOT_COUNT as usize], 0);
}
