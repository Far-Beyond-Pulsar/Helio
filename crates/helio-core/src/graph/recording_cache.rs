//! The recording cache: record a frame once, resubmit it (Helio#311).
//!
//! With the cache on, every unit (a pass, or a fused chain) records into
//! [`crate::cmd_ir`] streams instead of wgpu encoders. The streams are
//! compared with the recordings cached for that unit:
//!
//! * **hit**: the unit's cached `wgpu::ReusableCommandBuffer`s are submitted
//!   again. Nothing is encoded or validated by wgpu; wgpu only records the
//!   per-submission barriers and lifetime tracking.
//! * **miss**: the streams are encoded once into reusable command buffers and
//!   cached as a new variant (at most [`MAX_VARIANTS`] per unit, least
//!   recently used evicted, unused ones dropped after [`EVICT_AFTER_FRAMES`]).
//!
//! Correctness does not depend on passes: a pass that records different
//! commands (a CPU branch, a rebuilt bind group, a new dispatch size) simply
//! misses. Ping-pong passes alternate between two cached variants. Making
//! passes record the same commands every frame only raises the hit rate; see
//! [`RecordingCacheStats`] for which units miss and why.
//!
//! Units whose command buffers cannot be reused (acceleration structures,
//! surface textures, or a backend without support: anything but Vulkan and
//! D3D12) are recorded straight into wgpu, as without the cache.

use crate::cmd_ir::{self, Cmd, Stream};

/// Cached recordings kept per unit.
pub(crate) const MAX_VARIANTS: usize = 4;

/// A cached recording unused for this many frames is dropped, releasing the
/// resources it holds (old render targets after a resize, replaced buffers).
pub(crate) const EVICT_AFTER_FRAMES: u64 = 16;

/// Environment variable that switches the recording cache off (`0`, `off` or
/// `false`); it is on by default.
pub(crate) const RECORDING_CACHE_ENV: &str = "HELIO_RECORDING_CACHE";

/// Environment variable that prints, every 300 frames, which units missed and
/// why (`1`, `on` or `true`).
pub(crate) const RECORDING_CACHE_LOG_ENV: &str = "HELIO_RECORDING_CACHE_LOG";

pub(crate) fn log_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var(RECORDING_CACHE_LOG_ENV).is_ok_and(|value| {
            matches!(value.to_ascii_lowercase().as_str(), "1" | "on" | "true")
        })
    })
}

/// One cached recording of a unit.
pub(crate) struct CachedRecording {
    pub(crate) compute_cmds: Stream,
    pub(crate) graphics_cmds: Stream,
    /// `None` for an empty stream.
    pub(crate) compute: Option<wgpu::ReusableCommandBuffer>,
    pub(crate) graphics: Option<wgpu::ReusableCommandBuffer>,
    pub(crate) last_used: u64,
}

/// GPU timing for a cached unit. Recordings write timestamps at fixed
/// indices into this query set and resolve them into `resolve`, so the timing
/// commands are identical every frame; the graph's profiler copies the
/// results into its own readback each frame.
pub(crate) struct UnitTimer {
    pub(crate) query_set: wgpu::QuerySet,
    pub(crate) resolve: wgpu::Buffer,
    pub(crate) query_count: u32,
    /// `(label, begin, end)` query indices.
    pub(crate) spans: Vec<(&'static str, u32, u32)>,
}

impl UnitTimer {
    /// Queries `4k`/`4k+1` time pass `k`'s compute stream and `4k+2`/`4k+3`
    /// its graphics stream. A fused chain is one hardware render pass, timed
    /// once on the graphics stream at `2`/`3` under `chain_label`.
    pub(crate) fn new(device: &wgpu::Device, passes: &[&'static str], chain_label: Option<&'static str>) -> Self {
        let query_count = 4 * passes.len() as u32;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("Helio Cached Unit Timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: query_count,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Helio Cached Unit Timestamp Resolve"),
            size: u64::from(query_count) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mut spans = Vec::with_capacity(2 * passes.len());
        for (k, name) in passes.iter().enumerate() {
            let q = 4 * k as u32;
            spans.push((*name, q, q + 1));
            if chain_label.is_none() {
                spans.push((*name, q + 2, q + 3));
            }
        }
        if let Some(label) = chain_label {
            spans.push((label, 2, 3));
        }
        Self {
            query_set,
            resolve,
            query_count,
            spans,
        }
    }
}

/// Everything the cache keeps for one unit, indexed by the unit's first pass.
#[derive(Default)]
pub(crate) struct UnitCache {
    pub(crate) variants: Vec<CachedRecording>,
    /// Streams being recorded this frame; kept to reuse their allocations.
    pub(crate) scratch_compute: Stream,
    pub(crate) scratch_graphics: Stream,
    pub(crate) timer: Option<UnitTimer>,
    /// Set once this unit's command buffers turned out not to be reusable; it
    /// is then recorded straight into wgpu until the graph is rebuilt.
    pub(crate) uncacheable: Option<&'static str>,
    pub(crate) hits: u64,
    pub(crate) misses: u64,
    pub(crate) last_miss: Option<String>,
}

impl UnitCache {
    pub(crate) fn find(&self, compute: &[Cmd], graphics: &[Cmd]) -> Option<usize> {
        self.variants
            .iter()
            .position(|v| v.compute_cmds == compute && v.graphics_cmds == graphics)
    }

    /// Why `compute`/`graphics` missed: how they differ from the most
    /// recently used variant.
    pub(crate) fn describe_miss(&self, compute: &[Cmd], graphics: &[Cmd]) -> String {
        let Some(latest) = self.variants.iter().max_by_key(|v| v.last_used) else {
            return "first recording".to_string();
        };
        if latest.compute_cmds != compute {
            format!(
                "compute stream: {}",
                cmd_ir::first_difference(&latest.compute_cmds, compute)
            )
        } else {
            format!(
                "graphics stream: {}",
                cmd_ir::first_difference(&latest.graphics_cmds, graphics)
            )
        }
    }

    /// Adds a variant, evicting the least recently used one when full, and
    /// returns its index.
    pub(crate) fn insert(&mut self, recording: CachedRecording) -> usize {
        if self.variants.len() >= MAX_VARIANTS {
            let oldest = self
                .variants
                .iter()
                .enumerate()
                .min_by_key(|(_, v)| v.last_used)
                .map(|(i, _)| i)
                .expect("a full cache has variants");
            self.variants.swap_remove(oldest);
        }
        self.variants.push(recording);
        self.variants.len() - 1
    }

    /// Drops variants unused for [`EVICT_AFTER_FRAMES`]. Dropping a reusable
    /// command buffer is safe while a submission of it is in flight.
    pub(crate) fn evict_stale(&mut self, frame: u64) {
        self.variants
            .retain(|v| frame.saturating_sub(v.last_used) < EVICT_AFTER_FRAMES);
    }
}

/// Recording-cache diagnostics for the last frames, from
/// [`crate::graph::RenderGraph::recording_cache_stats`].
#[derive(Clone, Debug, Default)]
pub struct RecordingCacheStats {
    /// Whether the cache was used for the last frame.
    pub active: bool,
    /// Why it was not, when it was not.
    pub inactive_reason: Option<&'static str>,
    pub units: Vec<UnitCacheStats>,
}

#[derive(Clone, Debug)]
pub struct UnitCacheStats {
    /// The unit's passes: one, or a fused chain.
    pub passes: Vec<&'static str>,
    pub hits: u64,
    pub misses: u64,
    pub cached_variants: usize,
    /// Set when the unit is recorded straight into wgpu instead.
    pub uncacheable: Option<&'static str>,
    /// How the last miss differed from the previous recording.
    pub last_miss: Option<String>,
}

/// The result of encoding one recorded stream.
pub(crate) enum Encoded {
    Reusable(wgpu::ReusableCommandBuffer),
    /// Could not be made reusable; submit it once.
    Once(wgpu::CommandBuffer, &'static str),
}

/// Encodes a recorded stream into a reusable command buffer, or a normal one
/// when that is not possible. `None` for an empty stream.
pub(crate) fn encode_stream(device: &wgpu::Device, cmds: &[Cmd], label: &'static str) -> Option<Encoded> {
    if cmds.is_empty() {
        return None;
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(label),
    });
    let reusable = encoder.mark_reusable();
    cmd_ir::encode(cmds, &mut encoder);
    let command_buffer = encoder.finish();
    if !reusable {
        return Some(Encoded::Once(
            command_buffer,
            "the backend cannot resubmit command buffers",
        ));
    }
    Some(match command_buffer.into_reusable() {
        Ok(reusable) => Encoded::Reusable(reusable),
        Err(command_buffer) => Encoded::Once(
            command_buffer,
            "its command buffers use acceleration structures, surface textures or deferred actions",
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::{UnitCache, MAX_VARIANTS};

    #[test]
    fn an_empty_cache_finds_nothing_and_explains_the_first_miss() {
        let cache = UnitCache::default();
        assert_eq!(cache.find(&[], &[]), None);
        assert_eq!(cache.describe_miss(&[], &[]), "first recording");
        assert!(MAX_VARIANTS >= 2, "ping-pong passes need two variants");
    }
}
