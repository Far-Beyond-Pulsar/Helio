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
//! A unit is a pass outside a fused chain (with or without a prebuilt render
//! bundle) or a whole fused chain, recorded as one render pass, when
//! [`RenderPass::supports_recording_cache`](crate::RenderPass::supports_recording_cache)
//! is true for each of its passes. It is recorded straight into wgpu instead
//! when its command buffers cannot be reused (acceleration structures, surface
//! textures), while it backs off after [`MISS_STREAK_LIMIT`] misses in a row,
//! and always on a backend without support (anything but Vulkan and D3D12).
//! The cache is off while the finish breakdown is on. GPU timing is recorded
//! into the streams like any other command: the profiler's query indices
//! restart every frame, so a frame of the same shape writes the same
//! timestamps.
//!
//! Adapted from Tristan Poland's `codex/persistent-command-buffers` branch.

use crate::cmd_ir::{self, Cmd, Stream};

/// Cached recordings kept per unit.
pub(crate) const MAX_VARIANTS: usize = 4;

/// A cached recording unused for this many frames is dropped, releasing the
/// resources it holds (old render targets after a resize, replaced buffers).
pub(crate) const EVICT_AFTER_FRAMES: u64 = 16;

/// A unit that misses this many frames in a row is recorded straight into
/// wgpu for [`BACKOFF_FRAMES`] frames before the cache tries it again: a miss
/// costs more than recording directly, since the stream is compared, then
/// replayed into an encoder that is finished on the render thread.
pub(crate) const MISS_STREAK_LIMIT: u32 = 8;

/// How long a unit that kept missing is recorded directly.
pub(crate) const BACKOFF_FRAMES: u64 = 64;

/// Why a unit is being recorded directly while it backs off.
pub(crate) const BACKING_OFF: &str = "it missed 8 frames in a row; retried later";

/// Why a backend cannot use the cache at all.
pub(crate) const BACKEND_UNSUPPORTED: &str = "the backend cannot resubmit command buffers";

/// Environment variable that switches the recording cache off (`0`, `off`
/// or `false`); it is on by default.
pub(crate) const RECORDING_CACHE_ENV: &str = "HELIO_RECORDING_CACHE";

pub(crate) fn enabled_by_env() -> bool {
    !std::env::var(RECORDING_CACHE_ENV)
        .is_ok_and(|value| matches!(value.to_ascii_lowercase().as_str(), "0" | "off" | "false"))
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
    /// Frames recorded directly while backing off.
    pub(crate) bypassed: u64,
    pub(crate) last_miss: Option<String>,
    /// The unit's pass name, or its fused chain's label.
    pub(crate) label: &'static str,
    /// Consecutive misses (see [`MISS_STREAK_LIMIT`]).
    miss_streak: u32,
    /// Recorded directly until this frame.
    backoff_until: u64,
}

/// What a unit submits this frame for one of its streams.
pub(crate) enum UnitBuffer {
    /// The cached recording with this variant index.
    Cached(usize),
    Once(wgpu::CommandBuffer),
    /// Encoded reusable this frame, but not cached.
    Uncached(wgpu::ReusableCommandBuffer),
}

/// A unit's command buffers for this frame; `None` for an empty stream.
pub(crate) struct Resolved {
    pub(crate) compute: Option<UnitBuffer>,
    pub(crate) graphics: Option<UnitBuffer>,
    /// Encoding showed the backend cannot resubmit command buffers at all.
    pub(crate) backend_unsupported: bool,
}

impl UnitCache {
    /// Resolves this frame's recording: resubmits the matching cached
    /// variant, or encodes the streams and caches them as a new one.
    pub(crate) fn resolve(
        &mut self,
        device: &wgpu::Device,
        compute: Stream,
        graphics: Stream,
        frame: u64,
    ) -> Resolved {
        if let Some(variant) = self.find(&compute, &graphics) {
            self.hits += 1;
            self.miss_streak = 0;
            let recording = &mut self.variants[variant];
            recording.last_used = frame;
            let has = (recording.compute.is_some(), recording.graphics.is_some());
            self.recycle(compute, graphics);
            return Resolved {
                compute: has.0.then_some(UnitBuffer::Cached(variant)),
                graphics: has.1.then_some(UnitBuffer::Cached(variant)),
                backend_unsupported: false,
            };
        }
        self.misses += 1;
        self.last_miss = Some(self.describe_miss(&compute, &graphics));
        self.miss_streak += 1;
        if self.miss_streak >= MISS_STREAK_LIMIT {
            self.miss_streak = 0;
            self.backoff_until = frame + 1 + BACKOFF_FRAMES;
            // Nothing suggests these recordings come back.
            self.variants.clear();
        }
        let encoded_compute = encode_stream(device, &compute, "Helio Cached Compute Unit");
        let encoded_graphics = encode_stream(device, &graphics, "Helio Cached Graphics Unit");
        let once = [&encoded_compute, &encoded_graphics]
            .into_iter()
            .find_map(|encoded| match encoded {
                Some(Encoded::Once(_, reason)) => Some(*reason),
                _ => None,
            });
        if let Some(reason) = once {
            // Submit this frame's encoding once; the graph records the unit
            // straight into wgpu from now on.
            let backend_unsupported = reason == BACKEND_UNSUPPORTED;
            if !backend_unsupported {
                self.uncacheable = Some(reason);
            }
            self.recycle(compute, graphics);
            let once = |encoded: Option<Encoded>| {
                encoded.map(|encoded| match encoded {
                    Encoded::Once(buffer, _) => UnitBuffer::Once(buffer),
                    Encoded::Reusable(buffer) => UnitBuffer::Uncached(buffer),
                })
            };
            return Resolved {
                compute: once(encoded_compute),
                graphics: once(encoded_graphics),
                backend_unsupported,
            };
        }
        let reusable = |encoded: Option<Encoded>| match encoded {
            Some(Encoded::Reusable(buffer)) => Some(buffer),
            _ => None,
        };
        let compute_buffer = reusable(encoded_compute);
        let graphics_buffer = reusable(encoded_graphics);
        let has = (compute_buffer.is_some(), graphics_buffer.is_some());
        let variant = self.insert(CachedRecording {
            compute_cmds: compute,
            graphics_cmds: graphics,
            compute: compute_buffer,
            graphics: graphics_buffer,
            last_used: frame,
        });
        Resolved {
            compute: has.0.then_some(UnitBuffer::Cached(variant)),
            graphics: has.1.then_some(UnitBuffer::Cached(variant)),
            backend_unsupported: false,
        }
    }

    /// Whether the unit is recorded directly this frame because it kept
    /// missing; counts the frame if so.
    pub(crate) fn backing_off(&mut self, frame: u64) -> bool {
        let backing_off = frame < self.backoff_until;
        if backing_off {
            self.bypassed += 1;
        }
        backing_off
    }

    /// Takes the scratch streams for recording this frame.
    pub(crate) fn take_scratch(&mut self) -> (Stream, Stream) {
        let mut compute = std::mem::take(&mut self.scratch_compute);
        let mut graphics = std::mem::take(&mut self.scratch_graphics);
        compute.clear();
        graphics.clear();
        (compute, graphics)
    }

    /// Keeps the streams' allocations for the next frame.
    fn recycle(&mut self, mut compute: Stream, mut graphics: Stream) {
        compute.clear();
        graphics.clear();
        self.scratch_compute = compute;
        self.scratch_graphics = graphics;
    }

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
    /// Passes recorded straight into wgpu last frame, and why.
    pub direct: Vec<(&'static str, &'static str)>,
}

#[derive(Clone, Debug)]
pub struct UnitCacheStats {
    /// The pass, or a fused chain's passes joined with `+`.
    pub pass: &'static str,
    pub hits: u64,
    pub misses: u64,
    /// Frames it was recorded directly because it missed
    /// `MISS_STREAK_LIMIT` (8) frames in a row.
    pub bypassed: u64,
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
