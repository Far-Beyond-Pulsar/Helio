/// GPU profiler — zero overhead unless `debug_printout` is enabled in RendererConfig.
///
/// When disabled:
/// - No timestamp query allocations.
/// - No readback buffers.
/// - No stall, no async polling.
///
/// When enabled:
/// - One `wgpu::QuerySet` per frame (timestamp queries).
/// - Async readback via `QUERY_RESOLVE` buffer.
/// - Results available on next frame (latency = 1).

use std::collections::HashMap;

pub struct GpuProfiler {
    enabled:    bool,
    query_set:  Option<wgpu::QuerySet>,
    resolve_buf: Option<wgpu::Buffer>,
    readback_buf: Option<wgpu::Buffer>,
    slots:      Vec<String>,
    /// t[i*2] = begin timestamp, t[i*2+1] = end timestamp (nanoseconds).
    timestamps: Vec<u64>,
    max_scopes: u32,
    period_ns:  f32, // queue.get_timestamp_period()
}

#[derive(Clone, Debug, Default)]
pub struct PassTiming {
    pub label:   String,
    pub gpu_ms:  f32,
}

pub struct ProfilerScope {
    pub index: u32,
    pub enabled: bool,
}

impl GpuProfiler {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool, max_scopes: u32) -> Self {
        if !enabled {
            return Self {
                enabled: false,
                query_set: None,
                resolve_buf: None,
                readback_buf: None,
                slots: Vec::new(),
                timestamps: Vec::new(),
                max_scopes,
                period_ns: 1.0,
            };
        }

        let n = max_scopes * 2;  // begin + end per scope
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("profiler_queries"),
            ty:    wgpu::QueryType::Timestamp,
            count: n,
        });

        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("profiler_resolve"),
            size:               n as u64 * 8,
            usage:              wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label:              Some("profiler_readback"),
            size:               n as u64 * 8,
            usage:              wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let period_ns = queue.get_timestamp_period();

        Self {
            enabled: true,
            query_set: Some(query_set),
            resolve_buf: Some(resolve_buf),
            readback_buf: Some(readback_buf),
            slots:      Vec::new(),
            timestamps: vec![0u64; n as usize],
            max_scopes,
            period_ns,
        }
    }

    /// Register a scope label, returning its slot index. Call once at init.
    pub fn register_scope(&mut self, label: &str) -> u32 {
        assert!(
            self.slots.len() < self.max_scopes as usize,
            "Exceeded max profiler scopes ({})", self.max_scopes
        );
        let idx = self.slots.len() as u32;
        self.slots.push(label.to_owned());
        idx
    }

    /// Write begin timestamp for a scope (if profiler enabled).
    pub fn begin_scope(&self, encoder: &mut wgpu::CommandEncoder, idx: u32) {
        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, idx * 2);
        }
    }

    /// Write end timestamp for a scope (if profiler enabled).
    pub fn end_scope(&self, encoder: &mut wgpu::CommandEncoder, idx: u32) {
        if let Some(qs) = &self.query_set {
            encoder.write_timestamp(qs, idx * 2 + 1);
        }
    }

    /// Resolve timestamp queries at end of frame. Must be called before encoder.finish().
    pub fn resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        if let (Some(qs), Some(rb)) = (&self.query_set, &self.resolve_buf) {
            let count = self.slots.len() as u32 * 2;
            if count > 0 {
                encoder.resolve_query_set(qs, 0..count, rb, 0);
                if let Some(readback) = &self.readback_buf {
                    encoder.copy_buffer_to_buffer(rb, 0, readback, 0, count as u64 * 8);
                }
            }
        }
    }

    /// Non-blocking readback from the PREVIOUS frame's results. Returns empty map if disabled.
    pub fn collect_previous_frame(&mut self) -> HashMap<String, PassTiming> {
        if !self.enabled { return HashMap::new(); }
        let readback = match &self.readback_buf { Some(b) => b, None => return HashMap::new() };

        let slice = readback.slice(..);
        // Poll without blocking — if data isn't ready, skip this frame.
        // This imposes zero stall.
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        let mapped_ok = matches!(rx.try_recv(), Ok(Ok(())));
        if mapped_ok {
            let view = slice.get_mapped_range();
            let raw: &[u64] = bytemuck::cast_slice(&view);
            for (i, t) in raw.iter().enumerate() {
                if i < self.timestamps.len() { self.timestamps[i] = *t; }
            }
            drop(view);
            readback.unmap();
        }

        let mut out = HashMap::new();
        for (i, label) in self.slots.iter().enumerate() {
            let t0 = self.timestamps[i*2];
            let t1 = self.timestamps[i*2+1];
            if t0 == 0 && t1 == 0 { continue; }
            let gpu_ms = (t1.saturating_sub(t0)) as f32 * self.period_ns / 1_000_000.0;
            out.insert(label.clone(), PassTiming { label: label.clone(), gpu_ms });
        }
        out
    }
}

/// CPU-side timing for frame statistics (monotonic clock, nanoseconds).
#[derive(Clone, Debug, Default)]
pub struct CpuFrameStats {
    pub prep_ms:      f32,   // everything before graph.execute()
    pub graph_ms:     f32,   // graph.execute() through encoder.finish()
    pub submit_ms:    f32,   // queue.submit()
    pub present_ms:   f32,   // surface.present()
}
