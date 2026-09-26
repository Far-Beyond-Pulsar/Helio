//! Asynchronous GPU samples are attributed to the frame that produced them,
//! never to the camera/stage that happens to receive the delayed readback.
use helio::Renderer;
use helio_pass_tiny_voxel::engine::LazyEngineVoxelPass;
use std::{collections::BTreeMap, fs::File, io::Write, path::Path};

pub(super) struct VisibilityAudit {
    buffer: wgpu::Buffer,
    row: u32,
    width: u32,
}
impl VisibilityAudit {
    pub(super) fn encode(
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
    ) -> Self {
        assert_eq!(texture.format(), wgpu::TextureFormat::Rgba16Float);
        let row = (texture.width() * 8).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel sunlight audit"),
            size: u64::from(row) * u64::from(texture.height()),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(texture.height()),
                },
            },
            texture.size(),
        );
        Self {
            buffer,
            row,
            width: texture.width(),
        }
    }

    pub(super) fn save(self, device: &wgpu::Device, path: &Path) {
        let (tx, rx) = std::sync::mpsc::channel();
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = self.buffer.slice(..).get_mapped_range().unwrap();
        let mut counts = [0usize; 4];
        for row in data.chunks_exact(self.row as usize) {
            for pixel in row[..self.width as usize * 8].chunks_exact(8) {
                // Shader outputs exactly 1 (visible), 0 (occluded), or -1
                // (exhausted). Compare binary16 values without rounding.
                let kind = match u16::from_le_bytes([pixel[0], pixel[1]]) {
                    0x3c00 => 0,
                    0x0000 => 1,
                    0xbc00 => 2,
                    _ => 3,
                };
                counts[kind] += 1;
            }
        }
        std::fs::write(
            path.with_extension("sun.csv"),
            format!(
                "visible,occluded,exhausted,invalid\n{},{},{},{}\n",
                counts[0], counts[1], counts[2], counts[3],
            ),
        )
        .unwrap();
        assert_eq!(
            counts[2] + counts[3],
            0,
            "{}: sunlight traversal failed: {counts:?}",
            path.display()
        );
    }
}

pub(super) fn save_trace_work(
    device: &wgpu::Device,
    buffer: &wgpu::Buffer,
    hits: &[u8],
    path: &Path,
    name: &str,
) {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = buffer.slice(..).get_mapped_range().unwrap();
    let mut csv = File::create(path.with_extension(format!("{name}.csv"))).unwrap();
    writeln!(csv, "result,metric,n,mean,p50,p95,p99,max").unwrap();
    for result in ["all", "empty", "exact_hit", "far_hit"] {
        let mut values = [Vec::new(), Vec::new(), Vec::new()];
        for (pixel, (record, hit)) in data.chunks_exact(64).zip(hits.chunks_exact(32)).enumerate() {
            // Instrumentation must preserve the hit result and ray parameter.
            assert_eq!(
                &record[..32],
                hit,
                "{} {name}: changed primary ray {pixel}",
                path.display()
            );
            let work = &record[32..];
            let status = u32::from_le_bytes(hit[12..16].try_into().unwrap());
            let kind = if status & 3 == 0 {
                "empty"
            } else if (status >> 2) & 31 == 0 {
                "exact_hit"
            } else {
                "far_hit"
            };
            if result != "all" && result != kind {
                continue;
            }
            for a in 0..3 {
                values[a].push(u32::from_le_bytes(
                    work[a * 4..a * 4 + 4].try_into().unwrap(),
                ));
            }
        }
        for (metric, mut v) in ["leaf_visits", "exact_steps", "far_steps"]
            .into_iter()
            .zip(values)
        {
            if v.is_empty() {
                continue;
            }
            v.sort_unstable();
            let n = v.len();
            let mean = v.iter().map(|&x| u64::from(x)).sum::<u64>() as f64 / n as f64;
            writeln!(
                csv,
                "{result},{metric},{n},{mean:.3},{},{},{},{}",
                v[(n - 1) / 2],
                v[(n - 1) * 95 / 100],
                v[(n - 1) * 99 / 100],
                v[n - 1]
            )
            .unwrap();
        }
    }
}

pub(super) struct FlightProfiler {
    labels: BTreeMap<u64, (usize, String)>,
    graph_frames: BTreeMap<u64, u64>,
    graph_epoch: u64,
    last_graph_cpu: Option<u64>,
    gpu: File,
    memory: File,
    last_graph: Option<u64>,
    last_voxel: Option<u64>,
    last_memory: Option<(u64, u64)>,
}

impl FlightProfiler {
    pub(super) fn new(output: &Path) -> Self {
        let mut gpu = File::create(output.join("gpu.csv")).unwrap();
        writeln!(gpu, "domain,epoch,engine_frame,flight_frame,stage,pass,gpu_ms,lag_frames,readback_drops,query_overflows").unwrap();
        let mut memory = File::create(output.join("memory.csv")).unwrap();
        writeln!(memory, "flight_frame,stage,buffers_bytes,textures_bytes,material_capacity_bytes,primary_hits_bytes").unwrap();
        Self {
            labels: BTreeMap::new(),
            graph_frames: BTreeMap::new(),
            graph_epoch: 0,
            last_graph_cpu: None,
            gpu,
            memory,
            last_graph: None,
            last_voxel: None,
            last_memory: None,
        }
    }

    pub(super) fn record(&mut self, renderer: &Renderer, frame: usize, stage: &str) {
        let graph = renderer.timing_snapshot();
        let current = graph.cpu_frame_index;
        // The harness renders once per flight frame. PassContext uses that
        // persistent renderer clock; the graph profiler uses a separate clock
        // that restarts when resize constructs a new graph.
        self.labels.insert(frame as u64, (frame, stage.into()));
        if self
            .last_graph_cpu
            .is_some_and(|previous| current <= previous)
        {
            self.graph_epoch += 1;
            self.graph_frames.clear();
            self.last_graph = None;
        }
        self.last_graph_cpu = Some(current);
        self.graph_frames.insert(current, frame as u64);
        if let Some(source) = graph.gpu_frame_index {
            if self.last_graph != Some(source) {
                self.last_graph = Some(source);
                // This value uses the graph's enclosing timestamp scope. Do
                // not substitute a sum of potentially overlapping child passes.
                if let Some(ms) = renderer.gpu_frame_ms() {
                    self.row(
                        "graph",
                        self.graph_epoch,
                        source,
                        current,
                        self.graph_frames[&source],
                        "whole_graph",
                        f64::from(ms),
                        graph.readback_drops,
                        graph.query_overflows,
                    );
                }
                for pass in &graph.passes {
                    if let Some(ms) = pass.gpu_ms {
                        self.row(
                            "graph",
                            self.graph_epoch,
                            source,
                            current,
                            self.graph_frames[&source],
                            pass.name,
                            f64::from(ms),
                            graph.readback_drops,
                            graph.query_overflows,
                        );
                    }
                }
            }
        }
        let voxel = renderer.find_pass::<LazyEngineVoxelPass>().unwrap();
        let profiler = voxel.stage_profiler().expect("terrain profiling enabled");
        assert!(
            profiler.supported(),
            "GPU timestamp queries required for a profiling run"
        );
        if let Some(source) = profiler.last_completed_frame() {
            if self.last_voxel != Some(source) {
                self.last_voxel = Some(source);
                for pass in profiler.get_last_timings() {
                    self.row(
                        "voxel",
                        0,
                        source,
                        frame as u64,
                        source,
                        pass.name,
                        pass.duration_ns as f64 / 1_000_000.0,
                        profiler.dropped_readbacks(),
                        profiler.query_overflows(),
                    );
                }
            }
        }
        let memory = voxel.memory_stats().unwrap();
        let allocation = (memory.buffers_bytes, memory.textures_bytes);
        if self.last_memory != Some(allocation) {
            writeln!(
                self.memory,
                "{frame},{stage},{},{},{},{}",
                memory.buffers_bytes,
                memory.textures_bytes,
                memory.material_capacity_bytes,
                memory.primary_hits_bytes
            )
            .unwrap();
            self.last_memory = Some(allocation);
        }
    }

    fn row(
        &mut self,
        domain: &str,
        epoch: u64,
        source: u64,
        current: u64,
        flight_source: u64,
        name: &str,
        ms: f64,
        drops: u64,
        overflows: u64,
    ) {
        let (frame, stage) = self
            .labels
            .get(&flight_source)
            .expect("GPU result has a recorded source frame");
        assert!(ms.is_finite() && ms >= 0.0);
        let name = name.replace('"', "\"\"");
        writeln!(
            self.gpu,
            "{domain},{epoch},{source},{frame},{stage},\"{name}\",{ms:.6},{},{drops},{overflows}",
            current.saturating_sub(source)
        )
        .unwrap();
    }
}
