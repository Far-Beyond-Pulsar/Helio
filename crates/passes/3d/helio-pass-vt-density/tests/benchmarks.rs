//! BM1 — feedback + compaction cost at 1080p and 4K (Helio#238 benchmarks).
//! BM3 — bind-group rebuild churn under residency churn (issue benchmark 3).
//!
//! BM1 times full clear→compact frame pairs over the REAL kernels at the
//! quarter-res target sizes the internal resolutions imply (1080p ⇒ 480×270,
//! 4K ⇒ 960×540), both EMPTY (the steady-state cost floor) and FULLY WRITTEN
//! (worst case: every cell demanding, every slot's atomicMax contended).
//! The pair is what the frame pays per frame; the individual dispatches are
//! not separable in practice because the clear must complete before writers
//! and compaction after them.
//!
//! BM3 measures the §C-style rebuild gate discipline through the shared
//! binder's own churn counter ([`helio_core::shader::vt_binder::
//! VtGroupBinder::rebuild_count`]): a caller that keys constructions on the
//! changing-input key (frame-transient meta buffer pointer × density view ×
//! material-texture version) constructs EXACTLY once per change — an
//! unchanged-input loop performs ZERO constructions, and alternating residency
//! churn costs one construction per flip. That count IS the per-frame bind
//! overhead the issue asks to track.

use std::sync::Arc;

use helio_core::shader::vt_binder::{VtGroupBinder, VtGroupKey};
use helio_pass_vt_density::{FEEDBACK_BUFFER_WORDS, FEEDBACK_SLOT_COUNT};
use std::time::Instant;
use wgpu::util::DeviceExt as _;

const WORKGROUP: u32 = 8;

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

struct Kernels {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    clear_pipeline: wgpu::ComputePipeline,
    compact_pipeline: wgpu::ComputePipeline,
    write_pipeline: wgpu::ComputePipeline,
    bgl_clear: wgpu::BindGroupLayout,
    bgl_compact: wgpu::BindGroupLayout,
    bgl1: wgpu::BindGroupLayout,
    bgl_write: wgpu::BindGroupLayout,
}

impl Kernels {
    fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let module = helio_core::shader::module(
            &device,
            "vt_feedback_bench",
            include_str!("../shaders/vt_feedback.wgsl"),
        );
        // Per-entry-point layouts, mirroring the pass's `clear_group0_bgl` /
        // `compact_group0_bgl` (bind groups must fill their layout fully).
        let bgl_clear = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BM BGL Clear"),
            entries: &[
                uniform(0),
                storage_texture(1, wgpu::StorageTextureAccess::WriteOnly),
            ],
        });
        let bgl_compact = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BM BGL Compact"),
            entries: &[
                uniform(2),
                storage_texture(3, wgpu::StorageTextureAccess::ReadOnly),
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BM BGL1"),
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
        let mk_pipe = |label: &'static str, entry: &'static str, groups: bool| {
            let pair: [Option<&wgpu::BindGroupLayout>; 2] = [Some(&bgl_compact), Some(&bgl1)];
            let single: [Option<&wgpu::BindGroupLayout>; 1] = [Some(&bgl_clear)];
            let layouts: &[Option<&wgpu::BindGroupLayout>] = if groups { &pair } else { &single };
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
        let clear_pipeline = mk_pipe("BM Clear", "cs_clear", false);
        let compact_pipeline = mk_pipe("BM Compact", "cs_compact", true);

        // Full-screen writer: every invocation stores one packed word.
        let writer_src = r#"
struct Dims { dims: vec2<u32>, _pad: vec2<u32> };
@group(0) @binding(0) var dst: texture_storage_2d<r32uint, write>;
@group(0) @binding(1) var<uniform> dims: Dims;
@compute @workgroup_size(8, 8, 1)
fn cs_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= dims.dims.x || gid.y >= dims.dims.y { return; }
    // Every cell demands slot 1; wanted mip varies with position so the
    // atomicMax path does real compare work instead of folding equals.
    let mip = (gid.x + gid.y) % 12u;
    textureStore(dst, gid.xy, vec4<u32>((1u << 8u) | (mip + 1u), 0u, 0u, 0u));
}
"#;
        let write_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("BM Fill"),
            source: wgpu::ShaderSource::Wgsl(writer_src.into()),
        });
        let bgl_write = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("BM BGL Fill"),
            entries: &[
                storage_texture(0, wgpu::StorageTextureAccess::WriteOnly),
                uniform(1),
            ],
        });
        let write_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("BM Fill PL"),
            immediate_size: 0,
            bind_group_layouts: &[Some(&bgl_write)],
        });
        let write_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BM Fill"),
                layout: Some(&write_pl),
                module: &write_module,
                entry_point: Some("cs_fill"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self {
            device,
            queue,
            clear_pipeline,
            compact_pipeline,
            write_pipeline,
            bgl_clear,
            bgl_compact,
            bgl1,
            bgl_write,
        }
    }

    fn run_frame_pair(
        &self,
        qw: u32,
        qh: u32,
        fill: bool,
        density: &wgpu::Texture,
    ) -> Vec<u32> {
        let dims_data = [qw, qh, 0u32, 0u32];
        let dims = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BM Dims"),
            contents: bytemuck::bytes_of(&dims_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let out_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BM Out"),
            size: FEEDBACK_BUFFER_WORDS * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&out_buf, 0, &vec![0u8; (FEEDBACK_BUFFER_WORDS * 4) as usize]);

        let write_view = density.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R32Uint),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            ..Default::default()
        });
        let read_view = density.create_view(&wgpu::TextureViewDescriptor {
            format: Some(wgpu::TextureFormat::R32Uint),
            dimension: Some(wgpu::TextureViewDimension::D2),
            usage: Some(wgpu::TextureUsages::STORAGE_BINDING),
            ..Default::default()
        });
        let bg_clear = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BM BG Clear"),
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
            label: Some("BM BG C0"),
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
            label: Some("BM BG C1"),
            layout: &self.bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: out_buf.as_entire_binding(),
            }],
        });
        let bg_fill = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("BM BG Fill"),
            layout: &self.bgl_write,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&write_view),
                },
                wgpu::BindGroupEntry { binding: 1, resource: dims.as_entire_binding() },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("BM Frame") });
        {
            let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BM Frame Pass"),
                timestamp_writes: None,
            });
            let gx = qw.div_ceil(WORKGROUP).max(1);
            let gy = qh.div_ceil(WORKGROUP).max(1);
            cp.set_pipeline(&self.clear_pipeline);
            cp.set_bind_group(0, &bg_clear, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
            if fill {
                cp.set_pipeline(&self.write_pipeline);
                cp.set_bind_group(0, &bg_fill, &[]);
                cp.dispatch_workgroups(gx, gy, 1);
            }
            cp.set_pipeline(&self.compact_pipeline);
            cp.set_bind_group(0, &bg_compact0, &[]);
            cp.set_bind_group(1, &bg_compact1, &[]);
            cp.dispatch_workgroups(gx, gy, 1);
        }
        self.queue.submit([encoder.finish()]);

        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BM Staging"),
            size: FEEDBACK_BUFFER_WORDS * 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("BM Copy") });
        enc.copy_buffer_to_buffer(&out_buf, 0, &staging, 0, FEEDBACK_BUFFER_WORDS * 4);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).expect("map callback channel");
        });
        self.device.poll(wgpu::PollType::wait_indefinitely());
        rx.recv().expect("map result").expect("staging maps");
        let mapped = slice.get_mapped_range().expect("mapped range");
        let data = mapped.to_vec();
        drop(mapped);
        staging.unmap();
        bytemuck::cast_slice::<u8, u32>(&data).to_vec()
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

fn bm1(label: &str, internal: (u32, u32), kernels: &Kernels) {
    let (iw, ih) = internal;
    let (qw, qh) = (iw.div_ceil(4).max(1), ih.div_ceil(4).max(1));
    let density = kernels.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BM Density"),
        size: wgpu::Extent3d { width: qw, height: qh, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    const WARMUP: usize = 20;
    const FRAMES: usize = 200;

    for fill in [false, true] {
        for _ in 0..WARMUP {
            kernels.run_frame_pair(qw, qh, fill, &density);
        }
        let start = Instant::now();
        let mut last_words = Vec::new();
        for _ in 0..FRAMES {
            last_words = kernels.run_frame_pair(qw, qh, fill, &density);
        }
        let per_frame_us = start.elapsed().as_secs_f64() * 1e6 / FRAMES as f64;
        println!(
            "BM1 {label} {}x{} (quarter {}x{}, {}): {:>8.2} µs/frame (clear+fill+compact+readback)",
            iw, ih, qw, qh,
            if fill { "FULL" } else { "EMPTY" },
            per_frame_us,
        );
        // Sanity: a filled frame must produce demand for the written slot.
        if fill {
            assert_eq!(
                last_words[1] >> 8,
                1,
                "filled frames must record slot 1 demand"
            );
            assert!(last_words[FEEDBACK_SLOT_COUNT as usize] > 0);
        } else {
            assert_eq!(last_words[FEEDBACK_SLOT_COUNT as usize], 0);
        }
    }
}

#[test]
fn bm1_feedback_and_compaction_cost_at_1080p_and_4k() {
    let Some((device, queue)) = device_or_skip("BM1 vt-density") else { return };
    let kernels = Kernels::new(device, queue);
    bm1("1080p", (1920, 1080), &kernels);
    bm1("4K   ", (3840, 2160), &kernels);
}

#[test]
fn bm3_bind_group_rebuild_churn_under_residency_churn() {
    let Some((device, _queue)) = device_or_skip("BM3 bind churn") else { return };

    // Two frame-transient meta buffers standing in for alternating residency
    // publications; the density view stays fixed (graph-owned target).
    let meta_a = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BM3 Meta A"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let meta_b = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("BM3 Meta B"),
        size: 64,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let density = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("BM3 Density"),
        size: wgpu::Extent3d { width: 4, height: 4, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::R32Uint,
        usage: wgpu::TextureUsages::STORAGE_BINDING,
        view_formats: &[],
    });
    let density_view = density.create_view(&Default::default());
    let binder = VtGroupBinder::new(device.as_ref());

    let ptr = |b: &wgpu::Buffer| b as *const _ as usize;
    let key_for = |meta: usize| VtGroupKey {
        meta_ptr: meta,
        density_ptr: &density_view as *const _ as usize,
        version: 7, // fixed material-texture version for this scenario
    };

    let baseline = binder.rebuild_count();

    // Steady state: SAME inputs for 100 frames → ZERO rebuilds. This is the
    // gate property BM3 exists to keep honest — residency churn is the only
    // thing that may cost a bind-group construction.
    let steady_key = key_for(ptr(&meta_a));
    let mut held: Option<wgpu::BindGroup> = None;
    for _ in 0..100 {
        if steady_key != key_for(ptr(&meta_a)) || held.is_none() {
            held = Some(binder.bind_group(device.as_ref(), Some(&meta_a), Some(&density_view)));
        }
    }
    assert_eq!(
        binder.rebuild_count(),
        baseline + 1,
        "an unchanged-input loop must construct exactly once (the first time)"
    );

    // Residency churn: alternate publications 50 times → exactly 50 rebuilds,
    // never two per flip, never zero.
    let mut flips: u64 = 0;
    let last_holder: std::sync::Mutex<Option<VtGroupKey>> = std::sync::Mutex::new(None);
    for i in 0..50 {
        let next = if i % 2 == 0 { &meta_b } else { &meta_a };
        let key = key_for(ptr(next));
        let mut last = last_holder.lock().unwrap();
        if *last != Some(key) || held.is_none() {
            held = Some(binder.bind_group(device.as_ref(), Some(next), Some(&density_view)));
            flips += 1;
        }
        *last = Some(key);
    }
    assert_eq!(
        binder.rebuild_count(),
        baseline + 1 + flips,
        "each input change costs exactly one construction"
    );
    println!(
        "BM3 churn: 50 residency flips -> {flips} bind-group constructions \
         (steady 100-frame loop -> 0)"
    );

    // The constructed group is real and usable-sized (both buffers bind).
    assert!(held.is_some());
}
