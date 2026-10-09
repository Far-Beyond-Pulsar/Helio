//! The recording cache: record a frame once, resubmit it (Helio#330, part 5).
//!
//! With the cache on, a cacheable unit records into [`crate::cmd_ir`] streams
//! instead of wgpu encoders, and the streams are compared with the recordings
//! cached for that unit:
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
//! misses. Ping-pong passes alternate between cached variants. Making passes
//! record the same commands every frame only raises the hit rate; see
//! [`RecordingCacheStats`] for which units miss and why.
//!
//! A unit is one pass outside a fused chain, without a render bundle, whose
//! [`RenderPass::supports_recording_cache`](crate::RenderPass::supports_recording_cache)
//! is true. Units whose command buffers cannot be reused (acceleration
//! structures, surface textures, or a backend without support: anything but
//! Vulkan and D3D12) go back to being recorded straight into wgpu. The cache
//! is off while the finish breakdown is on. GPU timing is recorded into the
//! streams like any other command: the profiler's query indices restart every
//! frame, so a frame of the same shape writes the same timestamps.
//!
//! Adapted from Tristan Poland's `codex/persistent-command-buffers` branch.

use crate::cmd_ir::{self, Cmd, Stream};

/// Cached recordings kept per unit.
pub(crate) const MAX_VARIANTS: usize = 4;

/// A cached recording unused for this many frames is dropped, releasing the
/// resources it holds (old render targets after a resize, replaced buffers).
pub(crate) const EVICT_AFTER_FRAMES: u64 = 16;

/// Why a backend cannot use the cache at all.
pub(crate) const BACKEND_UNSUPPORTED: &str = "the backend cannot resubmit command buffers";

/// Environment variable that switches the recording cache on (`1`, `on` or
/// `true`); it is off by default while it is being brought up.
pub(crate) const RECORDING_CACHE_ENV: &str = "HELIO_RECORDING_CACHE";

pub(crate) fn enabled_by_env() -> bool {
    std::env::var(RECORDING_CACHE_ENV)
        .is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "on" | "true"))
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

/// Everything the cache keeps for one unit, indexed by the unit's pass.
#[derive(Default)]
pub(crate) struct UnitCache {
    pub(crate) variants: Vec<CachedRecording>,
    /// Streams being recorded this frame; kept to reuse their allocations.
    pub(crate) scratch_compute: Stream,
    pub(crate) scratch_graphics: Stream,
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

/// Recording-cache diagnostics, from
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
    pub pass: &'static str,
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
pub(crate) fn encode_stream(
    device: &wgpu::Device,
    cmds: &[Cmd],
    label: &'static str,
) -> Option<Encoded> {
    if cmds.is_empty() {
        return None;
    }
    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    let reusable = encoder.mark_reusable();
    cmd_ir::encode(cmds, &mut encoder);
    let command_buffer = encoder.finish();
    if !reusable {
        return Some(Encoded::Once(command_buffer, BACKEND_UNSUPPORTED));
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
