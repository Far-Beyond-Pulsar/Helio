//! Named frame statistics: counts any pass adds to, totalled per frame and
//! read by name from outside the renderer.
//!
//! A stat is a name. The first contribution to a name creates it; every pass
//! that contributes to the same name adds into one total, so the mesh,
//! foliage and virtual-geometry passes all add to [`VERTICES`] and a reader
//! gets the frame's vertices without knowing who drew them. A plugin exposes
//! its own counts the same way, under a name of its own (`meshes.draws`), or
//! pools them with others' under a shared one.
//!
//! Two ways in, neither of which costs the frame anything measurable:
//!
//! - **GPU.** Counts a GPU-driven pass only knows on the GPU (indirect draws
//!   its culling wrote, clusters it rasterized) go in with one
//!   `atomicAdd(&stats[slot], n)` from the shader that already writes them.
//!   [`FrameStats::slot`] gives the index (at build time); the counters are one
//!   storage buffer, [`FrameStats::buffer`], a `array<atomic<u32>>`.
//! - **CPU.** [`FrameStats::add`] for counts known on the CPU. Helio's command
//!   wrappers (`RenderCmds`) count every draw they record into
//!   [`DRAW_CALLS`], [`VERTICES`] and [`INSTANCES`] this way, adding once per
//!   pass rather than once per draw. Indirect draws are not counted there:
//!   their counts are on the GPU, so their pass adds them on the GPU.
//!
//! The frame's GPU counters are copied out at the start of the next frame, in
//! the same submission that clears them ([`FrameStats::cycle`]), into a ring
//! of staging buffers mapped without waiting. [`FrameStats::latest`] is the
//! newest frame whose copy has landed: a frame or two behind, never a stall.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Draw commands recorded on the CPU (`draw`, `draw_indexed`, and each draw of
/// an indirect or multi-draw call), plus draws a GPU-driven pass issues on the
/// GPU (a culled-in indirect draw).
pub const DRAW_CALLS: &str = "draw_calls";
/// Vertices submitted: a direct draw's vertex (or index) count times its
/// instances, and what GPU-driven passes add for the draws they issue.
pub const VERTICES: &str = "vertices";
/// Instances drawn, the same way.
pub const INSTANCES: &str = "instances";

/// Stats one renderer holds at most. A GPU counter is 4 bytes, so the buffer is
/// 4 KiB; past this, contributions go to [`OVERFLOW`].
pub const CAPACITY: u32 = 1024;
/// Where contributions to stats past [`CAPACITY`] land.
pub const OVERFLOW: &str = "stats.overflow";

/// Staging buffers the counters are copied into: one mapping while the next is
/// copied, a third for a slow map.
const RING: usize = 3;

/// A stat's index: its GPU counter (`stats[index]`) and its CPU total.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StatSlot(u32);

impl StatSlot {
    /// The GPU counter's index in [`FrameStats::buffer`].
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// One frame's totals, by name.
#[derive(Clone, Debug, Default)]
pub struct StatsSnapshot {
    /// The frame they were counted in.
    pub frame: u64,
    values: Vec<(Arc<str>, u64)>,
}

impl StatsSnapshot {
    /// The total of `name` (`None`: nothing has contributed to it).
    pub fn get(&self, name: &str) -> Option<u64> {
        self.values
            .iter()
            .find(|(n, _)| &**n == name)
            .map(|(_, v)| *v)
    }

    /// Every stat and its total, in the order they were created.
    pub fn iter(&self) -> impl Iterator<Item = (&str, u64)> {
        self.values.iter().map(|(n, v)| (&**n, *v))
    }
}

struct Names {
    list: Vec<Arc<str>>,
    index: HashMap<Arc<str>, u32>,
}

enum StagingState {
    Free,
    /// Copied in this frame's submission, not yet asked to map.
    Copied,
    /// Asked to map; the callback sets the flag.
    Mapping(Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>),
}

struct Staging {
    buffer: wgpu::Buffer,
    state: StagingState,
    /// The frame copied into it, its CPU totals and how many stats existed.
    frame: u64,
    cpu: Vec<u64>,
    count: u32,
}

struct Frame {
    /// The frame being counted now.
    number: u64,
    /// Whether its GPU counters were cleared (so they hold one frame).
    cleared: bool,
}

struct Inner {
    names: Mutex<Names>,
    cpu: Box<[AtomicU64]>,
    gpu: wgpu::Buffer,
    ring: Mutex<Vec<Staging>>,
    frame: Mutex<Frame>,
    latest: Mutex<Arc<StatsSnapshot>>,
}

/// A renderer's named frame statistics (see the module docs). Clones share
/// them.
#[derive(Clone)]
pub struct FrameStats {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for FrameStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameStats")
            .field("stats", &self.inner.names.lock().map_or(0, |n| n.list.len()))
            .finish()
    }
}

impl FrameStats {
    pub fn new(device: &wgpu::Device) -> Self {
        let bytes = u64::from(CAPACITY) * 4;
        let gpu = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Helio Frame Stats"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ring = (0..RING)
            .map(|_| Staging {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Helio Frame Stats Readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: StagingState::Free,
                frame: 0,
                cpu: Vec::new(),
                count: 0,
            })
            .collect();
        let stats = Self {
            inner: Arc::new(Inner {
                names: Mutex::new(Names {
                    list: Vec::new(),
                    index: HashMap::new(),
                }),
                cpu: (0..CAPACITY).map(|_| AtomicU64::new(0)).collect(),
                gpu,
                ring: Mutex::new(ring),
                frame: Mutex::new(Frame {
                    number: 0,
                    cleared: false,
                }),
                latest: Mutex::new(Arc::new(StatsSnapshot::default())),
            }),
        };
        // The built-in stats exist from the start, so a reader sees them at
        // zero rather than missing; the overflow takes the last counter.
        for name in [DRAW_CALLS, VERTICES, INSTANCES] {
            stats.slot(name);
        }
        stats
    }

    fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
        m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// The slot of stat `name`, created if no one has contributed to it yet.
    /// Take it once (at build time) and keep it: contributing through a slot
    /// takes no lock. Past [`CAPACITY`] stats, the [`OVERFLOW`] slot.
    pub fn slot(&self, name: &str) -> StatSlot {
        let mut names = Self::lock(&self.inner.names);
        if let Some(&index) = names.index.get(name) {
            return StatSlot(index);
        }
        // The last counter is the overflow's.
        let full = names.list.len() as u32 >= CAPACITY - 1;
        let (name, index) = if full {
            if let Some(&index) = names.index.get(OVERFLOW) {
                return StatSlot(index);
            }
            log::warn!("frame stats: more than {CAPACITY} stats; `{name}` and later ones count as `{OVERFLOW}`");
            (OVERFLOW, CAPACITY - 1)
        } else {
            (name, names.list.len() as u32)
        };
        let name: Arc<str> = Arc::from(name);
        names.list.push(name.clone());
        names.index.insert(name, index);
        StatSlot(index)
    }

    /// A shader's stats as pipeline-override constants: each `(override,
    /// stat)` pair becomes `(override, slot)`, creating stats no one has
    /// contributed to yet. The shader declares `override <name>: u32;` and
    /// adds with `atomicAdd(&stats[<name>], n)` into [`Self::buffer`]: its
    /// slots are compiled in, nothing is bound or read per frame. Pass the
    /// result as the pipeline's `compilation_options.constants`.
    pub fn slot_constants<'n>(&self, pairs: &[(&'n str, &str)]) -> Vec<(&'n str, f64)> {
        pairs
            .iter()
            .map(|&(name, stat)| (name, f64::from(self.slot(stat).index())))
            .collect()
    }

    /// The GPU counters: `array<atomic<u32>, CAPACITY>`, a pass's shader adding
    /// to `stats[slot.index()]`. Bind it read-write.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.inner.gpu
    }

    /// Add `n` to `slot`'s total for this frame (CPU-side; any thread).
    pub fn add(&self, slot: StatSlot, n: u64) {
        if n != 0 {
            self.inner.cpu[slot.0 as usize].fetch_add(n, Ordering::Relaxed);
        }
    }

    /// Add `n` to stat `name` (created if new). Looks the name up: prefer a
    /// kept [`StatSlot`] and [`Self::add`] for anything done every frame.
    pub fn add_named(&self, name: &str, n: u64) {
        let slot = self.slot(name);
        self.add(slot, n);
    }

    /// The newest frame whose totals have come back.
    pub fn latest(&self) -> Arc<StatsSnapshot> {
        Self::lock(&self.inner.latest).clone()
    }

    /// The renderer's per-frame step, at the start of a frame, recorded into
    /// the frame's first submission: the last frame's GPU counters copied out
    /// (with its CPU totals) when a staging buffer is free, then cleared for
    /// this frame. Copies whose maps landed become [`Self::latest`]. A frame
    /// with no free staging buffer is not reported (the reader sees the one
    /// before); counting never waits.
    pub fn cycle(&self, encoder: &mut wgpu::CommandEncoder) {
        self.collect();
        let count = Self::lock(&self.inner.names).list.len() as u32;
        let bytes = u64::from(count.max(1)) * 4;
        let mut frame = Self::lock(&self.inner.frame);
        // The CPU totals are the last frame's either way.
        let cpu: Vec<u64> = self.inner.cpu[..count as usize]
            .iter()
            .map(|c| c.swap(0, Ordering::Relaxed))
            .collect();
        if frame.cleared {
            let mut ring = Self::lock(&self.inner.ring);
            if let Some(staging) = ring
                .iter_mut()
                .find(|s| matches!(s.state, StagingState::Free))
            {
                encoder.copy_buffer_to_buffer(&self.inner.gpu, 0, &staging.buffer, 0, bytes);
                staging.state = StagingState::Copied;
                staging.frame = frame.number;
                staging.cpu = cpu;
                staging.count = count;
            }
        }
        encoder.clear_buffer(&self.inner.gpu, 0, Some(u64::from(CAPACITY) * 4));
        frame.cleared = true;
        frame.number += 1;
    }

    /// After the submission holding [`Self::cycle`]'s copy: ask for the copied
    /// staging buffers to map. The map completes on a later submit or poll.
    pub fn submitted(&self) {
        let mut ring = Self::lock(&self.inner.ring);
        for staging in ring.iter_mut() {
            if matches!(staging.state, StagingState::Copied) {
                let done = Arc::new(Mutex::new(None));
                let callback = Arc::clone(&done);
                let bytes = u64::from(staging.count.max(1)) * 4;
                staging
                    .buffer
                    .slice(..bytes)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        *Self::lock(&callback) = Some(result);
                    });
                staging.state = StagingState::Mapping(done);
            }
        }
    }

    /// Take every staging buffer whose map landed; the newest becomes
    /// [`Self::latest`].
    fn collect(&self) {
        let names = Self::lock(&self.inner.names).list.clone();
        let mut ring = Self::lock(&self.inner.ring);
        let mut newest: Option<StatsSnapshot> = None;
        for staging in ring.iter_mut() {
            let StagingState::Mapping(done) = &staging.state else {
                continue;
            };
            let Some(result) = Self::lock(done).take() else {
                continue;
            };
            let count = staging.count as usize;
            let bytes = (count.max(1) * 4) as u64;
            if result.is_ok() {
                let gpu: Vec<u32> = match staging.buffer.slice(..bytes).get_mapped_range() {
                    Ok(view) => bytemuck::cast_slice::<u8, u32>(&view)[..count].to_vec(),
                    Err(_) => Vec::new(),
                };
                if gpu.len() == count
                    && newest.as_ref().is_none_or(|n| n.frame < staging.frame)
                {
                    newest = Some(StatsSnapshot {
                        frame: staging.frame,
                        values: names[..count]
                            .iter()
                            .zip(gpu.iter().zip(&staging.cpu))
                            .map(|(name, (&g, &c))| (name.clone(), u64::from(g) + c))
                            .collect(),
                    });
                }
            }
            staging.buffer.unmap();
            staging.state = StagingState::Free;
        }
        if let Some(snapshot) = newest {
            let mut latest = Self::lock(&self.inner.latest);
            if latest.frame <= snapshot.frame {
                *latest = Arc::new(snapshot);
            }
        }
    }
}

/// Draws, vertices and instances one recorder counted, added to the frame's
/// stats once, when it is done.
#[derive(Default)]
pub(crate) struct DrawTally {
    pub(crate) draws: u64,
    pub(crate) vertices: u64,
    pub(crate) instances: u64,
}

impl DrawTally {
    pub(crate) fn direct(&mut self, vertices: u32, instances: u32) {
        self.draws += 1;
        self.vertices += u64::from(vertices) * u64::from(instances);
        self.instances += u64::from(instances);
    }

    pub(crate) fn flush(&mut self, stats: &FrameStats) {
        let taken = std::mem::take(self);
        if taken.draws == 0 {
            return;
        }
        // Built in, so these slots are the first three.
        stats.add(StatSlot(0), taken.draws);
        stats.add(StatSlot(1), taken.vertices);
        stats.add(StatSlot(2), taken.instances);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        pollster::block_on(async {
            let instance = wgpu::Instance::new(crate::wgpu_instance_descriptor());
            let adapter = instance.request_adapter(&Default::default()).await.ok()?;
            adapter.request_device(&Default::default()).await.ok()
        })
    }

    #[test]
    fn built_in_stats_hold_the_first_slots() {
        let Some((device, _)) = device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let stats = FrameStats::new(&device);
        assert_eq!(stats.slot(DRAW_CALLS), StatSlot(0));
        assert_eq!(stats.slot(VERTICES), StatSlot(1));
        assert_eq!(stats.slot(INSTANCES), StatSlot(2));
    }

    #[test]
    fn a_name_is_created_by_its_first_contribution_and_shared_after() {
        let Some((device, _)) = device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let stats = FrameStats::new(&device);
        let a = stats.slot("foliage.blades");
        let b = stats.slot("foliage.blades");
        assert_eq!(a, b, "the same name is the same stat");
        assert_ne!(a, stats.slot("meshes.draws"));
    }

    /// GPU and CPU contributions to one name are totalled for the frame they
    /// were made in, and the next frame starts from zero.
    #[test]
    fn gpu_and_cpu_contributions_total_per_frame() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let stats = FrameStats::new(&device);
        let custom = stats.slot("test.custom");
        let wait = |stats: &FrameStats| {
            stats.submitted();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };

        let frame = |stats: &FrameStats| {
            let mut encoder = device.create_command_encoder(&Default::default());
            stats.cycle(&mut encoder);
            queue.submit([encoder.finish()]);
            wait(stats);
        };

        // Frame 1 starts; its GPU work adds 5 to `vertices` and 7 to the
        // custom stat (written ahead of the next submission, as a pass's
        // atomics land before the next frame's copy), its CPU 3 to `vertices`.
        frame(&stats);
        queue.write_buffer(stats.buffer(), u64::from(StatSlot(1).index()) * 4, &5u32.to_le_bytes());
        queue.write_buffer(stats.buffer(), u64::from(custom.index()) * 4, &7u32.to_le_bytes());
        stats.add_named(VERTICES, 3);
        // Frame 2's start copies frame 1 out; frame 3's start collects it.
        frame(&stats);
        frame(&stats);
        let latest = stats.latest();
        assert_eq!(latest.frame, 1);
        assert_eq!(latest.get(VERTICES), Some(8), "GPU 5 + CPU 3");
        assert_eq!(latest.get("test.custom"), Some(7));
        assert_eq!(latest.get(DRAW_CALLS), Some(0), "built in, nothing drawn");
        assert_eq!(latest.get("never.contributed"), None);

        // Frame 2 counted nothing: cleared, not carried over.
        frame(&stats);
        let latest = stats.latest();
        assert_eq!(latest.frame, 2);
        assert_eq!(latest.get(VERTICES), Some(0));
        assert_eq!(latest.get("test.custom"), Some(0));
    }
}
