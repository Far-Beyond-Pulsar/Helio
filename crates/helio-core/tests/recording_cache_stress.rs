//! Stress coverage for the recording cache (Helio#311).
//!
//! A graph of synthetic passes, each with a known recording behaviour, runs
//! for many frames. For each pass the test checks two things:
//!
//! * **hits**: the cache's hit/miss counters match what the design predicts
//!   for that behaviour (and the full table is printed, for eyeballing);
//! * **correctness**: replayed recordings really execute. Every pass's GPU
//!   work increments or copies counters, which are read back at the end and
//!   compared with the values a frame-by-frame encoder would produce. The
//!   same graph is also run with the cache switched off, and must agree.
//!
//! Both are run with worker recording on and off.
//!
//! `recording_cache_cpu_cost` (ignored by default; run with
//! `cargo test -p helio-core --test recording_cache_stress -- --ignored --nocapture`)
//! times frames of a heavy graph with and without the cache.
//!
//! Skips when no adapter exists, and when the backend cannot reuse command
//! buffers (anything but Vulkan and D3D12), printing why.

use helio_core::{
    PassContext, PrepareContext, RenderFrameStorage, RenderGraph, RenderPass, ResourceRegistry,
    Result as HelioResult,
};
use std::sync::Arc;
mod support;

const FRAMES: u64 = 120;

const COUNTER_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> counter: atomic<u32>;
@compute @workgroup_size(1)
fn main() { atomicAdd(&counter, 1u); }
"#;

const TRIANGLE_SHADER: &str = r#"
@vertex
fn vs(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(i) - 1);
    let y = f32(i32(i & 1u) * 2 - 1);
    return vec4<f32>(x, y, 0.0, 1.0);
}
@fragment
fn fs() -> @location(0) vec4<f32> { return vec4<f32>(1.0, 0.5, 0.25, 1.0); }
"#;

/// What a pass records each frame; each maps to a predicted cache behaviour.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Behaviour {
    /// Same dispatch, same bind group every frame. Predicted: 1 miss.
    Static,
    /// Alternates two bind groups. Predicted: 2 misses (two variants).
    PingPong,
    /// A copy on every 4th frame only, like a readback state machine.
    /// Predicted: 2 misses.
    EveryFourth,
    /// A new bind group every frame. Predicted: every frame misses.
    Rebuild,
    /// Dispatch size cycles through 7 values, more than the 4 variants a unit
    /// keeps. Predicted: every frame misses (LRU thrash).
    Cycle7,
    /// Uploads the frame index in `prepare` and copies it in `execute`.
    /// Commands are identical every frame. Predicted: 1 miss.
    UploadCopy,
}

struct Shared {
    device: Arc<wgpu::Device>,
    counter_pipeline: wgpu::ComputePipeline,
    counter_layout: wgpu::BindGroupLayout,
}

impl Shared {
    fn new(device: &Arc<wgpu::Device>) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stress counter"),
            source: wgpu::ShaderSource::Wgsl(COUNTER_SHADER.into()),
        });
        let counter_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("stress counter"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let counter_layout = counter_pipeline.get_bind_group_layout(0);
        Self {
            device: Arc::clone(device),
            counter_pipeline,
            counter_layout,
        }
    }

    fn counter(&self, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn bind(&self, buffer: &wgpu::Buffer) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("stress counter"),
            layout: &self.counter_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        })
    }
}

struct StressPass {
    name: &'static str,
    writes: &'static [&'static str],
    behaviour: Behaviour,
    shared: Arc<Shared>,
    /// Counters (or copy sources/destinations), per behaviour.
    buffers: Vec<wgpu::Buffer>,
    bind_groups: Vec<wgpu::BindGroup>,
    /// Extra dispatches per frame, to make the graph heavier.
    repeat: u32,
}

impl StressPass {
    fn new(
        name: &'static str,
        writes: &'static [&'static str],
        behaviour: Behaviour,
        shared: &Arc<Shared>,
        repeat: u32,
    ) -> Self {
        let buffers: Vec<wgpu::Buffer> = match behaviour {
            Behaviour::PingPong | Behaviour::EveryFourth => {
                vec![shared.counter(name), shared.counter(name)]
            }
            Behaviour::UploadCopy => vec![
                shared.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("stress upload"),
                    size: 4,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                shared.counter(name),
            ],
            _ => vec![shared.counter(name)],
        };
        // Only counter buffers are storage buffers; the upload pass only copies.
        let bind_groups = if behaviour == Behaviour::UploadCopy {
            Vec::new()
        } else {
            buffers.iter().map(|b| shared.bind(b)).collect()
        };
        Self {
            name,
            writes,
            behaviour,
            shared: Arc::clone(shared),
            buffers,
            bind_groups,
            repeat,
        }
    }

    fn dispatch(&self, ctx: &PassContext, bind_group: &wgpu::BindGroup, workgroups: u32) {
        let mut cmds = ctx.compute_cmds();
        let mut pass = cmds.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.name),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.shared.counter_pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        for _ in 0..self.repeat.max(1) {
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    /// The value its first counter must hold after `frames` frames.
    fn expected(&self, frames: u64) -> Vec<u32> {
        let repeat = u64::from(self.repeat.max(1));
        match self.behaviour {
            Behaviour::Static | Behaviour::Rebuild => vec![(frames * repeat) as u32],
            Behaviour::PingPong => vec![
                (frames.div_ceil(2) * repeat) as u32,
                ((frames / 2) * repeat) as u32,
            ],
            Behaviour::Cycle7 => {
                vec![((0..frames).map(|f| 1 + f % 7).sum::<u64>() * repeat) as u32]
            }
            // The copy source is never written by this pass; it only proves
            // the replayed copy runs without error.
            Behaviour::EveryFourth => vec![0, 0],
            Behaviour::UploadCopy => vec![(frames - 1) as u32, (frames - 1) as u32],
        }
    }
}

impl RenderPass for StressPass {
    fn name(&self) -> &'static str {
        self.name
    }

    fn writes(&self) -> &'static [&'static str] {
        self.writes
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn prepare(&mut self, ctx: &PrepareContext) -> HelioResult<()> {
        if self.behaviour == Behaviour::UploadCopy {
            ctx.queue
                .write_buffer(&self.buffers[0], 0, &(ctx.frame_num as u32).to_le_bytes());
        }
        Ok(())
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let frame = ctx.frame_num;
        match self.behaviour {
            Behaviour::Static => self.dispatch(ctx, &self.bind_groups[0], 1),
            Behaviour::PingPong => {
                self.dispatch(ctx, &self.bind_groups[(frame % 2) as usize], 1)
            }
            Behaviour::Rebuild => {
                let fresh = self.shared.bind(&self.buffers[0]);
                self.dispatch(ctx, &fresh, 1);
            }
            Behaviour::Cycle7 => self.dispatch(ctx, &self.bind_groups[0], 1 + (frame % 7) as u32),
            Behaviour::EveryFourth => {
                if frame % 4 == 0 {
                    ctx.compute_cmds()
                        .copy_buffer_to_buffer(&self.buffers[0], 0, &self.buffers[1], 0, 4);
                }
            }
            Behaviour::UploadCopy => {
                ctx.compute_cmds()
                    .copy_buffer_to_buffer(&self.buffers[0], 0, &self.buffers[1], 0, 4);
            }
        }
        Ok(())
    }
}

/// Draws into the render pass the graph opens for it, every frame the same.
/// Predicted: 1 miss.
struct StressDraw {
    color: wgpu::TextureView,
    pipeline: wgpu::RenderPipeline,
}

impl StressDraw {
    fn new(device: &wgpu::Device) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stress draw target"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("stress triangle"),
            source: wgpu::ShaderSource::Wgsl(TRIANGLE_SHADER.into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("stress triangle"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::TextureFormat::Rgba8Unorm.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        Self {
            color: texture.create_view(&Default::default()),
            pipeline,
        }
    }
}

impl RenderPass for StressDraw {
    fn name(&self) -> &'static str {
        "StressDraw"
    }

    fn writes(&self) -> &'static [&'static str] {
        &["stress_draw"]
    }

    fn render_pass_descriptor<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a ResourceRegistry<'a>,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        None
    }

    fn render_pass_descriptor_with_storage<'a>(
        &'a self,
        _target: &'a wgpu::TextureView,
        _depth: &'a wgpu::TextureView,
        _resources: &'a ResourceRegistry<'a>,
        storage: &'a mut RenderFrameStorage,
    ) -> Option<wgpu::RenderPassDescriptor<'a>> {
        let color_attachments = storage.retain_boxed_slice(Box::new([Some(
            wgpu::RenderPassColorAttachment {
                view: &self.color,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            },
        )]));
        Some(wgpu::RenderPassDescriptor {
            label: Some("StressDraw"),
            color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    fn execute(&mut self, ctx: &mut PassContext) -> HelioResult<()> {
        let mut pass = ctx
            .render_cmds()
            .expect("StressDraw requires the graph render pass");
        pass.set_pipeline(&self.pipeline);
        pass.draw(0..3, 0..1);
        Ok(())
    }
}

struct Gpu {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    backend: wgpu::Backend,
}

async fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let mut adapter = None;
    for force_fallback_adapter in [false, true] {
        if let Ok(found) = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter,
                apply_limit_buckets: false,
            })
            .await
        {
            adapter = Some(found);
            break;
        }
    }
    let adapter = adapter?;
    let backend = adapter.get_info().backend;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("recording cache stress"),
            required_features: wgpu::Features::empty(),
            required_limits: adapter.limits(),
            ..Default::default()
        })
        .await
        .ok()?;
    Some(Gpu {
        device: Arc::new(device),
        queue: Arc::new(queue),
        backend,
    })
}

fn frame_view(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("stress frame"),
            size: wgpu::Extent3d {
                width: 16,
                height: 16,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

fn read_u32(gpu: &Gpu, src: &wgpu::Buffer) -> u32 {
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("stress readback"),
        size: 4,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu.device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(src, 0, &readback, 0, 4);
    gpu.queue.submit([encoder.finish()]);
    readback.slice(..).map_async(wgpu::MapMode::Read, |result| result.unwrap());
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");
    let mapped = readback.slice(..).get_mapped_range().expect("mapped");
    let value = u32::from_le_bytes(mapped[..4].try_into().unwrap());
    drop(mapped);
    readback.unmap();
    value
}

const BEHAVIOURS: [(&str, &[&str], Behaviour); 6] = [
    ("Static", &["stress_static"], Behaviour::Static),
    ("PingPong", &["stress_ping_pong"], Behaviour::PingPong),
    ("EveryFourth", &["stress_every_fourth"], Behaviour::EveryFourth),
    ("Rebuild", &["stress_rebuild"], Behaviour::Rebuild),
    ("Cycle7", &["stress_cycle7"], Behaviour::Cycle7),
    ("UploadCopy", &["stress_upload_copy"], Behaviour::UploadCopy),
];

struct Run {
    stats: helio_core::RecordingCacheStats,
    counters: Vec<(&'static str, Vec<u32>, Vec<u32>)>,
}

/// Runs the stress graph for `frames` frames; returns the cache stats and,
/// per pass, (name, read-back counters, expected counters).
fn run(gpu: &Gpu, cache: bool, parallel: bool, frames: u64) -> Run {
    let shared = Arc::new(Shared::new(&gpu.device));
    let mut graph = RenderGraph::new(&gpu.device, &gpu.queue);
    graph.set_recording_cache(cache);
    graph.set_parallel_recording(parallel);
    // The graph owns the passes; keep handles to their buffers (cheap clones)
    // to read back after the run.
    let mut probes = Vec::new();
    for (name, writes, behaviour) in BEHAVIOURS {
        let pass = StressPass::new(name, writes, behaviour, &shared, 1);
        probes.push((name, pass.buffers.clone(), pass.expected(frames)));
        graph.add_pass(Box::new(pass));
    }
    graph.add_pass(Box::new(StressDraw::new(&gpu.device)));
    graph.lock(16, 16);

    let mut scene = support::SceneInputAdapter::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue));
    let target = frame_view(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
    let depth = frame_view(&gpu.device, wgpu::TextureFormat::Depth32Float);
    for frame in 0..frames {
        scene.frame_count = frame;
        graph
            .execute(&scene, &target, &depth)
            .expect("stress frame must execute");
    }
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("poll");

    let counters = probes
        .into_iter()
        .map(|(name, buffers, expected)| {
            let values = buffers
                .iter()
                .take(expected.len())
                .map(|b| read_u32(gpu, b))
                .collect();
            (name, values, expected)
        })
        .collect();
    Run {
        stats: graph.recording_cache_stats(),
        counters,
    }
}

fn print_stats(label: &str, stats: &helio_core::RecordingCacheStats) {
    eprintln!(
        "\n== {label}: cache {} {}",
        if stats.active { "active" } else { "inactive" },
        stats.inactive_reason.unwrap_or("")
    );
    eprintln!("{:<14} {:>6} {:>7} {:>9}  last miss", "pass", "hits", "misses", "variants");
    for unit in &stats.units {
        eprintln!(
            "{:<14} {:>6} {:>7} {:>9}  {}{}",
            unit.passes.join("+"),
            unit.hits,
            unit.misses,
            unit.cached_variants,
            unit.uncacheable
                .map(|r| format!("[uncacheable: {r}] "))
                .unwrap_or_default(),
            unit.last_miss.as_deref().unwrap_or("-"),
        );
    }
    let hits: u64 = stats.units.iter().map(|u| u.hits).sum();
    let total: u64 = stats.units.iter().map(|u| u.hits + u.misses).sum();
    if total > 0 {
        eprintln!(
            "overall hit rate: {:.1}% ({hits}/{total} unit-frames)",
            100.0 * hits as f64 / total as f64
        );
    }
}

fn unit<'a>(stats: &'a helio_core::RecordingCacheStats, name: &str) -> &'a helio_core::UnitCacheStats {
    stats
        .units
        .iter()
        .find(|u| u.passes == [name])
        .unwrap_or_else(|| panic!("no cache stats for {name}"))
}

#[test]
fn recording_cache_hits_match_each_pass_behaviour_and_replays_execute() {
    pollster::block_on(async {
        let Some(gpu) = gpu().await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: recording cache stress");
            return;
        };

        // Reference: the cache off encodes every frame.
        let reference = run(&gpu, false, false, FRAMES);
        print_stats("cache off (reference)", &reference.stats);
        assert!(!reference.stats.active);
        for (name, values, expected) in &reference.counters {
            assert_eq!(values, expected, "{name}: reference run produced wrong counters");
        }

        for parallel in [false, true] {
            let label = if parallel { "cache on, worker recording" } else { "cache on, serial" };
            let result = run(&gpu, true, parallel, FRAMES);
            print_stats(label, &result.stats);
            if !result.stats.active {
                assert!(
                    !matches!(gpu.backend, wgpu::Backend::Vulkan | wgpu::Backend::Dx12),
                    "the cache must be active on {:?}: {:?}",
                    gpu.backend,
                    result.stats.inactive_reason
                );
                eprintln!("recording cache inactive on {:?}; skipping hit checks", gpu.backend);
                return;
            }

            // Replays must do exactly what encoding every frame does.
            for ((name, values, expected), (_, reference_values, _)) in
                result.counters.iter().zip(&reference.counters)
            {
                assert_eq!(values, expected, "{label}: {name} counters wrong after replays");
                assert_eq!(values, reference_values, "{label}: {name} differs from the reference");
            }

            let frames = FRAMES;
            let check = |name: &str, misses: u64| {
                let u = unit(&result.stats, name);
                assert_eq!(u.uncacheable, None, "{label}: {name} must be cacheable");
                assert_eq!(u.hits + u.misses, frames, "{label}: {name} counts every frame");
                assert_eq!(u.misses, misses, "{label}: {name} misses ({:?})", u.last_miss);
            };
            check("Static", 1);
            check("PingPong", 2);
            check("EveryFourth", 2);
            check("UploadCopy", 1);
            check("StressDraw", 1);
            check("Rebuild", frames);
            check("Cycle7", frames);
        }
    });
}

/// Times frames of a heavy graph (many passes, many dispatches each) with and
/// without the cache. Prints the numbers; asserts only that the cache does not
/// make frames slower.
#[test]
#[ignore = "timing benchmark; run explicitly with --ignored --nocapture"]
fn recording_cache_cpu_cost() {
    pollster::block_on(async {
        let Some(gpu) = gpu().await else {
            eprintln!("GPU_VALIDATION_SKIPPED_NO_ADAPTER: recording cache cpu cost");
            return;
        };
        const PASSES: usize = 64;
        const DISPATCHES: u32 = 64;
        const WARMUP: u64 = 10;
        const MEASURED: u64 = 200;
        static NAMES: std::sync::OnceLock<Vec<&'static str>> = std::sync::OnceLock::new();
        let names = NAMES.get_or_init(|| {
            (0..PASSES)
                .map(|i| &*Box::leak(format!("Heavy{i}").into_boxed_str()))
                .collect()
        });
        static WRITES: std::sync::OnceLock<Vec<&'static [&'static str]>> = std::sync::OnceLock::new();
        let writes = WRITES.get_or_init(|| {
            (0..PASSES)
                .map(|i| {
                    let name: &'static str = Box::leak(format!("heavy_{i}").into_boxed_str());
                    &*Box::leak(vec![name].into_boxed_slice())
                })
                .collect()
        });

        let mut results = Vec::new();
        for (cache, parallel) in [(false, false), (true, false), (false, true), (true, true)] {
            let shared = Arc::new(Shared::new(&gpu.device));
            let mut graph = RenderGraph::new(&gpu.device, &gpu.queue);
            graph.set_recording_cache(cache);
            graph.set_parallel_recording(parallel);
            for i in 0..PASSES {
                graph.add_pass(Box::new(StressPass::new(
                    names[i],
                    writes[i],
                    Behaviour::Static,
                    &shared,
                    DISPATCHES,
                )));
            }
            graph.lock(16, 16);
            let mut scene =
                support::SceneInputAdapter::new(Arc::clone(&gpu.device), Arc::clone(&gpu.queue));
            let target = frame_view(&gpu.device, wgpu::TextureFormat::Rgba8Unorm);
            let depth = frame_view(&gpu.device, wgpu::TextureFormat::Depth32Float);
            let mut measured = std::time::Duration::ZERO;
            for frame in 0..WARMUP + MEASURED {
                scene.frame_count = frame;
                let start = std::time::Instant::now();
                graph.execute(&scene, &target, &depth).expect("heavy frame");
                if frame >= WARMUP {
                    measured += start.elapsed();
                }
                // Keep the GPU from falling arbitrarily far behind.
                if frame % 16 == 0 {
                    let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
                }
            }
            let _ = gpu.device.poll(wgpu::PollType::wait_indefinitely());
            let per_frame = measured / MEASURED as u32;
            let stats = graph.recording_cache_stats();
            let hits: u64 = stats.units.iter().map(|u| u.hits).sum();
            let total: u64 = stats.units.iter().map(|u| u.hits + u.misses).sum();
            eprintln!(
                "cache {:<3} workers {:<3}: {:>8.3} ms/frame CPU ({} passes x {} dispatches), hits {hits}/{total}",
                if cache { "on" } else { "off" },
                if parallel { "on" } else { "off" },
                per_frame.as_secs_f64() * 1e3,
                PASSES,
                DISPATCHES,
            );
            results.push(((cache, parallel), per_frame, stats.active));
        }
        for parallel in [false, true] {
            let off = results.iter().find(|r| r.0 == (false, parallel)).unwrap();
            let on = results.iter().find(|r| r.0 == (true, parallel)).unwrap();
            if on.2 {
                eprintln!(
                    "workers {}: cache speedup {:.2}x",
                    if parallel { "on" } else { "off" },
                    off.1.as_secs_f64() / on.1.as_secs_f64()
                );
                assert!(
                    on.1 <= off.1.mul_f64(1.10),
                    "the cache made frames slower: {:?} vs {:?}",
                    on.1,
                    off.1
                );
            }
        }
    });
}
