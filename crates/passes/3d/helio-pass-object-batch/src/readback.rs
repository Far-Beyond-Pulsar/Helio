//! Small, bounded, async CPU readback of the GPU-computed range tables and
//! draw/shadow counts.
//!
//! `helio-pass-gbuffer` (and `helio-pass-shadow`/`helio-pass-transparent`/
//! `helio-pass-forward-lit`) select a PSO per `(material_class, graph_hash)`
//! range and issue one `multi_draw_indexed_indirect(indirect, start * 20,
//! count)` call per range -- `start`/`count` must be plain `u32`s the CPU
//! holds before recording that draw call, not something read from a GPU
//! buffer at submit time. Every OTHER byte this pipeline produces
//! (`instances`/`aabbs`/`draw_calls`/the shadow-partitioned indirect lists)
//! is read by later passes entirely on the GPU with zero CPU involvement --
//! only this one small, already-bounded table (`#ranges <= #groups <=
//! #objects`, and a real scene's distinct material combinations stay far
//! below even that) ever needs to reach the CPU, and it does so the same
//! way any GPU-driven renderer reads back a bounded indirect-count/range
//! table: an async, non-blocking `map_async`, read whenever its callback
//! has actually fired (never stalling the frame to wait for it), trailing
//! whatever the GPU actually produced by however many frames the GPU
//! happens to be behind -- typically one.
//!
//! This is emphatically NOT a readback of per-instance data: `instances`/
//! `aabbs` (the only per-object-sized buffers) never touch the CPU.
//!
//! # Ordering (why this is correct, not racy)
//!
//! [`RangeReadback::poll_and_kick_off`] is called from `ObjectBatchPass::
//! prepare()`, which the render graph runs BEFORE this same pass's
//! `execute()` each frame (the same ordering every other pass's `prepare()`-
//! side uniform upload already relies on). So at the moment this runs for
//! frame N, `scratch`'s buffers still hold whatever frame (N-1)'s
//! `execute()` last wrote -- copying them now is copying valid, complete
//! data, and it is submitted (hence GPU-ordered) strictly before frame N's
//! own `execute()` dispatch overwrites them with fresh values.

use std::sync::{Arc, Mutex};

use super::{RangeTuple, ScratchBuffers};

type MapResult = Result<(), wgpu::BufferAsyncError>;
/// `map_async`'s callback can fire from a driver/polling thread, so its
/// slot needs `Send + Sync` -- an `mpsc::Receiver` isn't `Sync` (and
/// `ObjectBatchPass` must be, to be stored as a graph pass), so each
/// pending map is a shared slot the callback fills and `try_recv`-style
/// polling below drains with a non-blocking `lock()` + `take()`.
type MapSlot = Arc<Mutex<Option<MapResult>>>;

fn new_map_slot() -> MapSlot {
    Arc::new(Mutex::new(None))
}

fn try_recv(slot: &MapSlot) -> Option<MapResult> {
    slot.lock().expect("map slot mutex poisoned").take()
}

/// One staging "generation": a copy of last-known counts + range tables,
/// with up to four independent `map_async` calls in flight (or already
/// resolved) against it.
struct Generation {
    counts_staging: wgpu::Buffer,
    opaque_staging: wgpu::Buffer,
    transparent_staging: wgpu::Buffer,
    forward_staging: wgpu::Buffer,
    pending: Option<PendingMaps>,
}

struct PendingMaps {
    counts_slot: MapSlot,
    opaque_slot: MapSlot,
    transparent_slot: MapSlot,
    forward_slot: MapSlot,
    counts_ready: bool,
    opaque_ready: bool,
    transparent_ready: bool,
    forward_ready: bool,
}

impl PendingMaps {
    fn all_ready(&self) -> bool {
        self.counts_ready && self.opaque_ready && self.transparent_ready && self.forward_ready
    }
}

impl Generation {
    fn new(device: &wgpu::Device, capacity: u32) -> Self {
        let range_bytes = (capacity.max(1) as u64) * super::RANGE_BYTES;
        let make = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(4),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        Self {
            // [group_count, bucket_opaque, bucket_transparent, bucket_forward, shadow_static, shadow_movable, instance_count] -- 7 u32s.
            counts_staging: make("ObjBatch Counts Staging", 7 * 4),
            opaque_staging: make("ObjBatch Opaque Staging", range_bytes),
            transparent_staging: make("ObjBatch Transparent Staging", range_bytes),
            forward_staging: make("ObjBatch Forward Staging", range_bytes),
            pending: None,
        }
    }

    fn range_capacity_bytes(&self) -> u64 {
        self.opaque_staging.size()
    }
}

const GENERATIONS: usize = 2;

pub struct RangeReadback {
    gens: Vec<Generation>,
    next: usize,
    capacity: u32,

    opaque: Vec<RangeTuple>,
    transparent: Vec<RangeTuple>,
    forward: Vec<RangeTuple>,
    draw_count: u32,
    shadow_static: u32,
    shadow_movable: u32,
    instance_count: u32,
    /// Bumped whenever a freshly-harvested generation's `shadow_static`
    /// count differs from the previous one -- `helio-pass-shadow`'s static-
    /// atlas cache invalidation signal, matching what `Scene::
    /// rebuild_shadow_partition_buffers`'s `static_objects_generation`
    /// used to track (increment on Static/Stationary object add/remove).
    /// An add+remove that nets to the same count in one frame is missed
    /// (a stale static atlas for one frame, self-correcting the moment the
    /// count next changes) -- the same category of approximation `Scene`'s
    /// own dirty-flag tracking made no stronger guarantee about either
    /// (nothing here ever tracked object *identity*, only set membership).
    shadow_static_generation: u64,
}

impl RangeReadback {
    pub fn new() -> Self {
        Self {
            gens: Vec::new(),
            next: 0,
            capacity: 0,
            opaque: Vec::new(),
            transparent: Vec::new(),
            forward: Vec::new(),
            draw_count: 0,
            shadow_static: 0,
            shadow_movable: 0,
            instance_count: 0,
            shadow_static_generation: 0,
        }
    }

    pub fn opaque(&self) -> &[RangeTuple] {
        &self.opaque
    }
    pub fn transparent(&self) -> &[RangeTuple] {
        &self.transparent
    }
    pub fn forward(&self) -> &[RangeTuple] {
        &self.forward
    }
    pub fn counts(&self) -> (u32, u32, u32) {
        (self.draw_count, self.shadow_static, self.shadow_movable)
    }
    /// Live instance count -- same value as `instances_buffer()`'s valid
    /// prefix length.
    pub fn instance_count(&self) -> u32 {
        self.instance_count
    }
    /// See [`Self::shadow_static_generation`]'s field doc.
    pub fn shadow_static_generation(&self) -> u64 {
        self.shadow_static_generation
    }

    /// See this module's doc for the full ordering rationale. Never blocks.
    pub fn poll_and_kick_off(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &ScratchBuffers,
    ) {
        device.poll(wgpu::PollType::Poll).ok();

        let capacity = scratch_capacity_hint(scratch);
        if self.gens.len() < GENERATIONS || self.capacity != capacity {
            // (Re)build generations sized to the current scratch capacity.
            // Any maps in flight against the old (about to be dropped)
            // buffers simply never get read -- harmless: the public arrays
            // just keep whatever they last held until a new generation's
            // map resolves.
            self.gens = (0..GENERATIONS)
                .map(|_| Generation::new(device, capacity))
                .collect();
            self.capacity = capacity;
            self.next = 0;
        }

        // 1. Advance every generation's pending maps; harvest the first one
        // that's fully resolved this call.
        for gen in self.gens.iter_mut() {
            let Some(pending) = gen.pending.as_mut() else {
                continue;
            };
            if !pending.counts_ready {
                if let Some(r) = try_recv(&pending.counts_slot) {
                    pending.counts_ready = r.is_ok();
                    if r.is_err() {
                        gen.pending = None;
                        continue;
                    }
                }
            }
            let Some(pending) = gen.pending.as_mut() else {
                continue;
            };
            if !pending.opaque_ready {
                if let Some(r) = try_recv(&pending.opaque_slot) {
                    pending.opaque_ready = r.is_ok();
                }
            }
            if !pending.transparent_ready {
                if let Some(r) = try_recv(&pending.transparent_slot) {
                    pending.transparent_ready = r.is_ok();
                }
            }
            if !pending.forward_ready {
                if let Some(r) = try_recv(&pending.forward_slot) {
                    pending.forward_ready = r.is_ok();
                }
            }
            if pending.all_ready() {
                let prev_shadow_static = self.shadow_static;
                read_generation_into(
                    gen,
                    &mut self.opaque,
                    &mut self.transparent,
                    &mut self.forward,
                    &mut self.draw_count,
                    &mut self.shadow_static,
                    &mut self.shadow_movable,
                    &mut self.instance_count,
                );
                if self.shadow_static != prev_shadow_static {
                    self.shadow_static_generation = self.shadow_static_generation.wrapping_add(1);
                }
                gen.counts_staging.unmap();
                gen.opaque_staging.unmap();
                gen.transparent_staging.unmap();
                gen.forward_staging.unmap();
                gen.pending = None;
            }
        }

        // 2. Queue a fresh copy+map into the next round-robin slot, unless
        // it's still waiting on its own previous map (the GPU/CPU would
        // have to be more than `GENERATIONS` frames behind -- this frame
        // just skips kicking off a new copy rather than stomping on a
        // buffer still being read).
        let slot = self.next;
        self.next = (self.next + 1) % self.gens.len().max(1);
        if self.gens[slot].pending.is_some() {
            return;
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("ObjBatch Readback Copy"),
        });
        {
            let gen = &self.gens[slot];
            encoder.copy_buffer_to_buffer(&scratch.group_count, 0, &gen.counts_staging, 0, 4);
            encoder.copy_buffer_to_buffer(
                &scratch.range_bucket_counts,
                0,
                &gen.counts_staging,
                4,
                12,
            );
            encoder.copy_buffer_to_buffer(&scratch.shadow_counts, 0, &gen.counts_staging, 16, 8);
            // `frame_uniform`'s first u32 is `count` -- the gather's live
            // instance count (see `FrameUniformGpu`'s doc in `src/lib.rs`).
            encoder.copy_buffer_to_buffer(&scratch.frame_uniform, 0, &gen.counts_staging, 24, 4);
            let range_bytes = gen.range_capacity_bytes();
            encoder.copy_buffer_to_buffer(
                &scratch.opaque_ranges,
                0,
                &gen.opaque_staging,
                0,
                range_bytes.min(scratch.opaque_ranges.size()),
            );
            encoder.copy_buffer_to_buffer(
                &scratch.transparent_ranges,
                0,
                &gen.transparent_staging,
                0,
                range_bytes.min(scratch.transparent_ranges.size()),
            );
            encoder.copy_buffer_to_buffer(
                &scratch.forward_ranges,
                0,
                &gen.forward_staging,
                0,
                range_bytes.min(scratch.forward_ranges.size()),
            );
        }
        queue.submit([encoder.finish()]);

        let gen = &mut self.gens[slot];
        let counts_slot = new_map_slot();
        {
            let slot = Arc::clone(&counts_slot);
            gen.counts_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    *slot.lock().expect("map slot mutex poisoned") = Some(r);
                });
        }
        let opaque_slot = new_map_slot();
        {
            let slot = Arc::clone(&opaque_slot);
            gen.opaque_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    *slot.lock().expect("map slot mutex poisoned") = Some(r);
                });
        }
        let transparent_slot = new_map_slot();
        {
            let slot = Arc::clone(&transparent_slot);
            gen.transparent_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    *slot.lock().expect("map slot mutex poisoned") = Some(r);
                });
        }
        let forward_slot = new_map_slot();
        {
            let slot = Arc::clone(&forward_slot);
            gen.forward_staging
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |r| {
                    *slot.lock().expect("map slot mutex poisoned") = Some(r);
                });
        }

        gen.pending = Some(PendingMaps {
            counts_slot,
            opaque_slot,
            transparent_slot,
            forward_slot,
            counts_ready: false,
            opaque_ready: false,
            transparent_ready: false,
            forward_ready: false,
        });
    }
}

/// `ScratchBuffers` doesn't publish its own capacity (it's derived from
/// buffer sizes, which are only meaningful together) -- recover it the same
/// way `object_batch.wgsl`'s own capacity math does, from a fixed-4-bytes-
/// per-row buffer's size.
fn scratch_capacity_hint(scratch: &ScratchBuffers) -> u32 {
    (scratch.keys_a.size() / 4) as u32
}

fn read_generation_into(
    gen: &Generation,
    opaque: &mut Vec<RangeTuple>,
    transparent: &mut Vec<RangeTuple>,
    forward: &mut Vec<RangeTuple>,
    draw_count: &mut u32,
    shadow_static: &mut u32,
    shadow_movable: &mut u32,
    instance_count: &mut u32,
) {
    let counts_view = gen
        .counts_staging
        .slice(..)
        .get_mapped_range()
        .expect("counts staging buffer was mapped (map_async already resolved Ok)");
    let counts: &[u32] = bytemuck::cast_slice(&counts_view);
    let group_count = counts[0];
    let n_opaque = counts[1];
    let n_transparent = counts[2];
    let n_forward = counts[3];
    let n_shadow_static = counts[4];
    let n_shadow_movable = counts[5];
    let live_instance_count = counts[6];
    drop(counts_view);

    *draw_count = group_count;
    *shadow_static = n_shadow_static;
    *shadow_movable = n_shadow_movable;
    *instance_count = live_instance_count;

    *opaque = read_ranges(&gen.opaque_staging, n_opaque);
    *transparent = read_ranges(&gen.transparent_staging, n_transparent);
    *forward = read_ranges(&gen.forward_staging, n_forward);
}

fn read_ranges(staging: &wgpu::Buffer, count: u32) -> Vec<RangeTuple> {
    let view = staging
        .slice(..)
        .get_mapped_range()
        .expect("range staging buffer was mapped (map_async already resolved Ok)");
    let words: &[u32] = bytemuck::cast_slice(&view);
    // GpuRangeOut: {material_class, graph_hash_lo, graph_hash_hi, start, count} -- 5 u32s.
    let n = (count as usize).min(words.len() / 5);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let base = i * 5;
        let class = words[base];
        let hash_lo = words[base + 1] as u64;
        let hash_hi = words[base + 2] as u64;
        let start = words[base + 3];
        let cnt = words[base + 4];
        out.push((class, (hash_hi << 32) | hash_lo, start, cnt));
    }
    drop(view);
    out
}
