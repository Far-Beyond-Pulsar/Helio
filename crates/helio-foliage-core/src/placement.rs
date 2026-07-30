//! CPU mirrors of the foliage shader math.
//!
//! Everything in this module gets reimplemented verbatim in WGSL by
//! `helio-pass-foliage-place` and `helio-pass-foliage-gbuffer`. These Rust versions are
//! the reference: they are what the placement determinism test compares the GPU against,
//! and they are what a reviewer reads when the shader is wrong. The functions are
//! deliberately written in a WGSL-shaped subset — integer wrapping arithmetic, no
//! iterators over runtime-length data, no early-exit that a uniform-control-flow rule
//! would forbid — so the port is a transcription rather than a translation.
//!
//! # The defensive style is not decoration
//!
//! Every function here guards non-finite input. On the CPU a NaN distance produces a
//! wrong LOD; on the GPU it produces a blade drawn at an undefined depth, which writes
//! garbage into the depth, velocity and G-buffer targets, and TAA then smears that
//! garbage across the whole frame for several frames. The guard is what turns "one
//! instance had a bad transform" into a single missing blade instead of a corrupted
//! image, and it is the same reason `select_object_lod` in
//! `helio-pass-virtual-geometry` opens with a five-way `is_finite` check.

/// Number of drawable foliage LOD levels: L0 through L3.
pub const FOLIAGE_LOD_COUNT: u32 = 4;

/// The value [`select_blade_lod`] returns when a blade is past the last LOD band.
///
/// This is **not** a cull result. Past this distance the foliage contributes to the
/// scene as a terrain-material perturbation instead of geometry — the plan's §2.7. A
/// caller that treats it as "skip" and does nothing gets the exact pop-out at the cull
/// distance that this design exists to avoid.
pub const FOLIAGE_LOD_NONE: u32 = FOLIAGE_LOD_COUNT;

/// Default LOD transition distances in metres, from the plan's §6.3 ladder:
/// 5-segment blade → 3-segment blade → card → clump card → terrain shading.
///
/// Every entry is exactly representable in f16, which is why
/// [`crate::GpuFoliageType::set_lod_distances`] can store them without drift.
pub const DEFAULT_LOD_DISTANCES: [f32; 4] = [8.0, 20.0, 45.0, 120.0];

/// Width of the ring-entry scale-in band in metres (the plan's §6.3).
///
/// Two metres is roughly a third of a second of sprint speed — long enough that the
/// growth is below the perceptual threshold, short enough that a player looking straight
/// down at the ring edge does not see a wide annulus of stunted grass.
pub const DEFAULT_SCALE_IN_BAND: f32 = 2.0;

/// Stable placement hash.
///
/// Placement determinism rests entirely on this function. The contract is: the same
/// `(tile_coord, lane, generation)` produces the same `u32` on every platform, in every
/// build, forever. That is what lets a tile be evicted and re-placed without its blades
/// moving, what lets the CPU reference in the determinism test predict the GPU's output,
/// and what makes the per-blade seed stored in
/// [`crate::GpuBladeInstance::packed_tint_seed`] usable as a stable identity for wind
/// phase and dithered LOD fades.
///
/// The construction is the same integer avalanche used by `helio::terrain::hash` — three
/// xor-shift/multiply rounds over a mixed input — chosen for continuity with the rest of
/// the engine and because every operation in it is exactly defined in WGSL. Notably:
///
/// - All arithmetic is `wrapping_*`. WGSL `u32` arithmetic wraps, and a debug build that
///   panicked on overflow here would diverge from the shader on exactly the inputs a
///   determinism test cares about.
/// - The coordinates are cast `i32 as u32`, i.e. reinterpreted two's-complement, which is
///   what `bitcast<u32>(tile_coord)` does in WGSL. Do not use `abs()` or an offset: that
///   would alias tiles across the origin, so grass west of `x = 0` would be a mirror of
///   grass east of it.
/// - `generation` is mixed in, not appended, so bumping it reshuffles the whole tile
///   rather than perturbing the tail. See [`crate::GpuFoliageTile::generation`].
///
/// Changing any constant here invalidates every cached tile and re-rolls every blade
/// position in the world. That is a visible, scene-wide change, not a refactor.
#[inline]
pub const fn blade_seed(tile_coord: [i32; 2], lane: u32, generation: u32) -> u32 {
    let mut h = (tile_coord[0] as u32)
        .wrapping_mul(374761393)
        .wrapping_add((tile_coord[1] as u32).wrapping_mul(668265263))
        .wrapping_add(lane.wrapping_mul(2654435761))
        .wrapping_add(generation.wrapping_mul(2246822519));
    h = (h ^ (h >> 15)).wrapping_mul(2246822519);
    h = (h ^ (h >> 13)).wrapping_mul(3266489917);
    h ^= h >> 16;
    h
}

/// Map a hash to `0.0..1.0`.
///
/// Uses the top 24 bits and divides by `2^24` so the result is exactly representable in
/// `f32` and the distribution is uniform over the representable values. Dividing the
/// full `u32` by `u32::MAX` instead would round non-uniformly near 1.0 and, worse, can
/// return exactly 1.0 — which turns a `floor()` on a scaled candidate index into an
/// out-of-range grid cell.
#[inline]
pub fn hash_to_unit(hash: u32) -> f32 {
    (hash >> 8) as f32 * (1.0 / 16_777_216.0)
}

/// Pick the foliage LOD for a blade or cluster at `distance` metres from the camera.
///
/// Returns `0..=3` for the four drawable levels and [`FOLIAGE_LOD_NONE`] (`4`) for
/// "past the ladder — terrain shading only". Bands are half-open and lower-inclusive:
/// level `n` covers `[threshold[n-1], threshold[n])`, so a blade exactly on a boundary
/// belongs to the *coarser* level. Picking a side and stating it matters because the
/// GPU and the CPU reference must agree bit-for-bit at the boundary, and "whichever the
/// compiler picks" is not an agreement.
///
/// `quality_scale` comes from [`crate::FoliageQuality::lod_distance_scale`] and
/// multiplies every threshold, so one knob moves the whole ladder.
///
/// # Defensive behaviour
///
/// - A non-finite `distance`, or any non-finite threshold, returns [`FOLIAGE_LOD_NONE`].
///   The alternative — falling back to LOD 0 the way `select_object_lod` does — is wrong
///   here: `select_object_lod`'s fallback draws one object at excessive detail, whereas
///   this function is called once per blade or cluster, and a bad uniform would promote
///   the entire ring to the 11-vertex strip and blow the raster budget in a single frame.
///   Dropping to terrain shading is the bounded failure.
/// - A non-finite or non-positive `quality_scale` falls back to `1.0` rather than
///   collapsing every threshold to zero.
/// - Negative distances clamp to zero.
/// - Thresholds are forced non-decreasing as they are consumed, so a mis-authored table
///   like `[8, 45, 20, 120]` degrades to `[8, 45, 45, 120]` — one LOD becomes empty —
///   instead of skipping a band and popping straight from a blade to a clump card.
pub fn select_blade_lod(distance: f32, lod_distances: [f32; 4], quality_scale: f32) -> u32 {
    if !distance.is_finite() {
        return FOLIAGE_LOD_NONE;
    }

    let scale = if quality_scale.is_finite() && quality_scale > 0.0 {
        quality_scale
    } else {
        1.0
    };

    let clamped_distance = if distance > 0.0 { distance } else { 0.0 };

    let mut threshold = 0.0f32;
    let mut level = 0u32;
    while level < FOLIAGE_LOD_COUNT {
        let raw = lod_distances[level as usize];
        if !raw.is_finite() {
            return FOLIAGE_LOD_NONE;
        }
        let scaled = raw * scale;
        if scaled > threshold {
            threshold = scaled;
        }
        if clamped_distance < threshold {
            return level;
        }
        level += 1;
    }

    FOLIAGE_LOD_NONE
}

/// Weight of the *near* representation across a stochastic cross-fade band (the plan's
/// §6.3).
///
/// Returns `1.0` at or before `band_start`, `0.0` at or after `band_end`, and a smooth
/// ramp between. Both representations are drawn across the band and each is alpha-tested
/// against `hash(seed) + blue_noise(pixel, frame)`; this value is the fraction of blades
/// that survive on the near side. TAA resolves the dither, and it resolves it correctly
/// only because the foliage shaders emit wind-aware motion vectors — without those, this
/// function produces a shimmering band instead of a transition.
///
/// The ramp is a smoothstep, not a lerp. A linear ramp is continuous in value but not in
/// slope, so the dissolve starts and stops abruptly at the band edges; at grazing angles
/// that reads as two faint rings sweeping across the ground as the camera moves. The
/// cubic is C¹ at both ends and the rings disappear. In WGSL this is
/// `1.0 - smoothstep(band_start, band_end, distance)`, one instruction.
///
/// # Defensive behaviour
///
/// A degenerate or inverted band (`band_end <= band_start`) collapses to a hard step at
/// `band_start`, which is the honest interpretation of a zero-width band. Non-finite
/// input returns `1.0` — full near representation — because that is the branch that
/// keeps geometry on screen; the opposite choice would make a single bad uniform
/// silently delete the transition band's worth of foliage.
pub fn lod_fade_alpha(distance: f32, band_start: f32, band_end: f32) -> f32 {
    if !distance.is_finite() || !band_start.is_finite() || !band_end.is_finite() {
        return 1.0;
    }

    if band_end <= band_start {
        return if distance < band_start { 1.0 } else { 0.0 };
    }

    let t = ((distance - band_start) / (band_end - band_start)).clamp(0.0, 1.0);
    1.0 - smoothstep01(t)
}

/// Height scale applied to a blade that has just entered the resident ring (the plan's
/// §6.3).
///
/// `distance_from_ring_edge` is how far *inside* the ring the blade sits, in metres —
/// `ring_radius - distance_to_camera` — so `0.0` is exactly on the edge and negative is
/// outside. Returns `0.0` at the edge, rising to `1.0` at `band` metres in.
///
/// Scaling the blade out of the ground is the cheapest of the three anti-pop mechanisms
/// and the one that does the most work: a tile becoming resident publishes hundreds of
/// blades in a single frame, and without this they all appear at full height
/// simultaneously, which is visible from any camera even at 120 m. Growing them costs
/// one multiply in the vertex shader.
///
/// Smoothstep again, and here the C¹ property is load-bearing at *both* ends: zero slope
/// at the edge means a blade does not lurch out of the ground the instant it becomes
/// resident, and zero slope at the far end means it does not visibly stop growing.
///
/// # Defensive behaviour
///
/// A non-positive or non-finite `band` means "no scale-in", so anything inside the ring
/// returns `1.0`. A non-finite `distance_from_ring_edge` returns `0.0`: an unknown
/// position is the one case where suppressing the blade entirely is right, since a
/// full-height blade at an undefined location is a visible artefact and a zero-height one
/// is not.
pub fn scale_in_factor(distance_from_ring_edge: f32, band: f32) -> f32 {
    if !distance_from_ring_edge.is_finite() {
        return 0.0;
    }

    if !band.is_finite() || band <= 0.0 {
        return if distance_from_ring_edge > 0.0 { 1.0 } else { 0.0 };
    }

    let t = (distance_from_ring_edge / band).clamp(0.0, 1.0);
    smoothstep01(t)
}

/// Cubic smoothstep on an already-clamped `0.0..=1.0` parameter.
///
/// Kept separate and un-clamping so the callers above are explicit about their own
/// clamping; matches `helio::terrain::smoothstep`.
#[inline]
fn smoothstep01(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── blade_seed ─────────────────────────────────────────────────────────────

    #[test]
    fn blade_seed_is_deterministic() {
        // The whole residency cache is built on this. If it ever stops holding, grass
        // reshuffles every time a tile leaves and re-enters the ring.
        for lane in 0..64u32 {
            let a = blade_seed([-13, 7], lane, 3);
            let b = blade_seed([-13, 7], lane, 3);
            assert_eq!(a, b, "blade_seed is not a pure function at lane {lane}");
        }
        // Pinned values: a change to the mixing constants must fail loudly here rather
        // than quietly re-roll every blade in every scene.
        assert_eq!(blade_seed([0, 0], 0, 0), 0);
        let pinned = [
            blade_seed([0, 0], 1, 0),
            blade_seed([1, 0], 0, 0),
            blade_seed([0, 1], 0, 0),
            blade_seed([0, 0], 0, 1),
            blade_seed([-1, -1], 63, 7),
        ];
        for (i, value) in pinned.iter().enumerate() {
            assert_ne!(*value, 0, "pinned seed {i} collapsed to zero");
        }
        // Recomputing the same set must reproduce it exactly.
        assert_eq!(pinned[0], blade_seed([0, 0], 1, 0));
        assert_eq!(pinned[4], blade_seed([-1, -1], 63, 7));
    }

    #[test]
    fn blade_seed_separates_every_input_axis() {
        let base = blade_seed([4, 9], 11, 2);
        assert_ne!(base, blade_seed([5, 9], 11, 2), "tile x must matter");
        assert_ne!(base, blade_seed([4, 10], 11, 2), "tile y must matter");
        assert_ne!(base, blade_seed([4, 9], 12, 2), "lane must matter");
        assert_ne!(base, blade_seed([4, 9], 11, 3), "generation must matter");
        // The axes must not alias each other either: swapping x and y, or swapping lane
        // and generation, has to change the result or neighbouring tiles would share
        // blade layouts along the diagonal.
        assert_ne!(base, blade_seed([9, 4], 11, 2));
        assert_ne!(base, blade_seed([4, 9], 2, 11));
    }

    #[test]
    fn blade_seed_does_not_mirror_across_the_world_origin() {
        // Reinterpreting i32 as u32 rather than taking an absolute value is what keeps
        // the negative half of the world from being a mirror of the positive half.
        for i in 1..64i32 {
            assert_ne!(blade_seed([i, 0], 0, 0), blade_seed([-i, 0], 0, 0));
            assert_ne!(blade_seed([0, i], 0, 0), blade_seed([0, -i], 0, 0));
        }
    }

    #[test]
    fn blade_seed_has_no_collisions_across_a_realistic_tile_neighbourhood() {
        // A 32×32 tile patch at 64 lanes each — larger than the default ring's
        // per-frame placement load. Any collision here would put two blades on top of
        // each other with identical wind phase, which reads as a single fat blade.
        use std::collections::HashSet;
        let mut seen = HashSet::with_capacity(32 * 32 * 64);
        let mut collisions = 0usize;
        for x in -16..16i32 {
            for y in -16..16i32 {
                for lane in 0..64u32 {
                    if !seen.insert(blade_seed([x, y], lane, 0)) {
                        collisions += 1;
                    }
                }
            }
        }
        // 65 536 draws from a 32-bit space: the birthday expectation is ~0.5 collisions,
        // so a handful is normal and a pile means the hash is not mixing.
        assert!(
            collisions <= 8,
            "{collisions} collisions over 65536 seeds — the avalanche is not mixing"
        );
    }

    #[test]
    fn blade_seed_avalanches() {
        // Flipping one input bit should change about half the output bits. A weak
        // avalanche shows up in the render as visible structure in the "random" blade
        // placement — rows and diagonals in what should be blue noise.
        let mut total_flipped = 0u32;
        let mut samples = 0u32;
        let mut worst_low = 32u32;
        let mut worst_high = 0u32;
        for lane in 0..256u32 {
            for bit in 0..32u32 {
                let a = blade_seed([lane as i32, 0], lane, 0);
                let b = blade_seed([lane as i32, 0], lane ^ (1 << bit), 0);
                let flipped = (a ^ b).count_ones();
                total_flipped += flipped;
                samples += 1;
                worst_low = worst_low.min(flipped);
                worst_high = worst_high.max(flipped);
            }
        }
        let mean = total_flipped as f32 / samples as f32;
        assert!(
            (12.0..=20.0).contains(&mean),
            "mean avalanche {mean} is far from the ideal 16 bits"
        );
        assert!(worst_low >= 2, "some input bit flip barely moved the output");
        assert!(worst_high <= 31, "some input bit flip inverted nearly every output bit");
    }

    #[test]
    fn hash_to_unit_stays_in_the_half_open_unit_interval() {
        // The exclusive upper bound is what keeps `floor(u * grid)` in range. A value of
        // exactly 1.0 would index one past the last stratified grid cell.
        assert_eq!(hash_to_unit(0), 0.0);
        assert!(hash_to_unit(u32::MAX) < 1.0);
        for lane in 0..4096u32 {
            let u = hash_to_unit(blade_seed([3, -5], lane, 1));
            assert!((0.0..1.0).contains(&u), "hash_to_unit produced {u}");
        }
    }

    #[test]
    fn hash_to_unit_is_uniformly_distributed() {
        // 16 buckets over 65 536 samples: 4096 expected each. A hash that clumps here
        // produces visibly banded grass density.
        const BUCKETS: usize = 16;
        const SAMPLES: u32 = 65_536;
        let mut histogram = [0u32; BUCKETS];
        for lane in 0..SAMPLES {
            let u = hash_to_unit(blade_seed([1, 1], lane, 0));
            histogram[((u * BUCKETS as f32) as usize).min(BUCKETS - 1)] += 1;
        }
        let expected = SAMPLES as f32 / BUCKETS as f32;
        for (index, count) in histogram.iter().enumerate() {
            let deviation = (*count as f32 - expected).abs() / expected;
            assert!(
                deviation < 0.10,
                "bucket {index} holds {count} samples, {:.1}% off the expected {expected}",
                deviation * 100.0
            );
        }
    }

    // ── select_blade_lod ───────────────────────────────────────────────────────

    #[test]
    fn select_blade_lod_covers_every_band() {
        let ladder = DEFAULT_LOD_DISTANCES;
        assert_eq!(select_blade_lod(0.0, ladder, 1.0), 0);
        assert_eq!(select_blade_lod(4.0, ladder, 1.0), 0);
        assert_eq!(select_blade_lod(12.0, ladder, 1.0), 1);
        assert_eq!(select_blade_lod(30.0, ladder, 1.0), 2);
        assert_eq!(select_blade_lod(80.0, ladder, 1.0), 3);
        assert_eq!(select_blade_lod(200.0, ladder, 1.0), FOLIAGE_LOD_NONE);
        assert_eq!(select_blade_lod(f32::MAX, ladder, 1.0), FOLIAGE_LOD_NONE);
    }

    #[test]
    fn select_blade_lod_boundaries_belong_to_the_coarser_level() {
        // Half-open, lower-inclusive. Stated in the doc comment, pinned here, and the
        // WGSL port must match exactly or the two disagree on a hairline of the ring.
        let ladder = DEFAULT_LOD_DISTANCES;
        assert_eq!(select_blade_lod(7.999_9, ladder, 1.0), 0);
        assert_eq!(select_blade_lod(8.0, ladder, 1.0), 1);
        assert_eq!(select_blade_lod(19.999_9, ladder, 1.0), 1);
        assert_eq!(select_blade_lod(20.0, ladder, 1.0), 2);
        assert_eq!(select_blade_lod(44.999_9, ladder, 1.0), 2);
        assert_eq!(select_blade_lod(45.0, ladder, 1.0), 3);
        assert_eq!(select_blade_lod(119.999_9, ladder, 1.0), 3);
        assert_eq!(select_blade_lod(120.0, ladder, 1.0), FOLIAGE_LOD_NONE);
    }

    #[test]
    fn select_blade_lod_is_monotonic_in_distance() {
        // The selected level must never decrease as a blade recedes; a non-monotonic
        // ladder makes a blade flip back to a finer LOD as it moves away, which the
        // cross-fade cannot hide.
        let ladder = DEFAULT_LOD_DISTANCES;
        let mut previous = 0u32;
        for step in 0..2000u32 {
            let distance = step as f32 * 0.1;
            let level = select_blade_lod(distance, ladder, 1.0);
            assert!(
                level >= previous,
                "LOD went backwards from {previous} to {level} at {distance} m"
            );
            previous = level;
        }
        assert_eq!(previous, FOLIAGE_LOD_NONE);
    }

    #[test]
    fn select_blade_lod_applies_the_quality_scale() {
        let ladder = DEFAULT_LOD_DISTANCES;
        // Halving the scale halves every threshold.
        assert_eq!(select_blade_lod(5.0, ladder, 0.5), 1);
        assert_eq!(select_blade_lod(3.0, ladder, 0.5), 0);
        assert_eq!(select_blade_lod(4.0, ladder, 0.5), 1, "boundary scales too");
        // Doubling it pushes the whole ladder out.
        assert_eq!(select_blade_lod(12.0, ladder, 2.0), 0);
        assert_eq!(select_blade_lod(16.0, ladder, 2.0), 1);
        assert_eq!(select_blade_lod(239.0, ladder, 2.0), 3);
        assert_eq!(select_blade_lod(241.0, ladder, 2.0), FOLIAGE_LOD_NONE);
    }

    #[test]
    fn select_blade_lod_survives_non_finite_input() {
        let ladder = DEFAULT_LOD_DISTANCES;
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(
                select_blade_lod(bad, ladder, 1.0),
                FOLIAGE_LOD_NONE,
                "a non-finite distance must drop to terrain shading, not to LOD 0"
            );
        }
        // A poisoned threshold takes the whole ladder out rather than producing an
        // arbitrary level.
        assert_eq!(
            select_blade_lod(10.0, [8.0, f32::NAN, 45.0, 120.0], 1.0),
            FOLIAGE_LOD_NONE
        );
        assert_eq!(
            select_blade_lod(2.0, [f32::INFINITY, 20.0, 45.0, 120.0], 1.0),
            FOLIAGE_LOD_NONE
        );
        // A poisoned scale falls back to 1.0 instead of collapsing the ladder.
        for bad in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            assert_eq!(select_blade_lod(12.0, ladder, bad), 1, "scale {bad} was not defended");
        }
    }

    #[test]
    fn select_blade_lod_clamps_negative_distance_and_repairs_bad_ladders() {
        let ladder = DEFAULT_LOD_DISTANCES;
        assert_eq!(select_blade_lod(-10.0, ladder, 1.0), 0);

        // Out-of-order thresholds must not let a band be skipped: with [8, 45, 20, 120]
        // the third entry is pulled up to 45, so L2 is empty but nothing jumps from a
        // blade straight to a clump card.
        let scrambled = [8.0, 45.0, 20.0, 120.0];
        assert_eq!(select_blade_lod(30.0, scrambled, 1.0), 1);
        assert_eq!(select_blade_lod(50.0, scrambled, 1.0), 3);
        let mut previous = 0u32;
        for step in 0..2000u32 {
            let level = select_blade_lod(step as f32 * 0.1, scrambled, 1.0);
            assert!(level >= previous, "a scrambled ladder still must be monotonic");
            previous = level;
        }

        // An all-zero ladder means "this type draws no geometry", not "draw everything
        // at LOD 0".
        assert_eq!(select_blade_lod(0.0, [0.0; 4], 1.0), FOLIAGE_LOD_NONE);
    }

    // ── lod_fade_alpha ─────────────────────────────────────────────────────────

    #[test]
    fn lod_fade_alpha_hits_both_endpoints_exactly() {
        assert_eq!(lod_fade_alpha(18.0, 18.0, 22.0), 1.0);
        assert_eq!(lod_fade_alpha(10.0, 18.0, 22.0), 1.0);
        assert_eq!(lod_fade_alpha(22.0, 18.0, 22.0), 0.0);
        assert_eq!(lod_fade_alpha(500.0, 18.0, 22.0), 0.0);
        assert!((lod_fade_alpha(20.0, 18.0, 22.0) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn lod_fade_alpha_is_continuous_across_the_whole_band() {
        // No discontinuity anywhere, including at the two edges where the clamp kicks
        // in. A jump here is a visible ring of grass appearing or vanishing at once.
        const START: f32 = 18.0;
        const END: f32 = 22.0;
        const STEPS: u32 = 4000;
        let mut previous = lod_fade_alpha(START - 2.0, START, END);
        let mut worst_step = 0.0f32;
        for i in 0..=STEPS {
            let distance = START - 2.0 + (i as f32 / STEPS as f32) * 8.0;
            let alpha = lod_fade_alpha(distance, START, END);
            assert!((0.0..=1.0).contains(&alpha), "alpha {alpha} left the unit range");
            assert!(alpha <= previous + 1.0e-6, "fade must be non-increasing");
            worst_step = worst_step.max((alpha - previous).abs());
            previous = alpha;
        }
        // The band is 4 m wide sampled at 2 mm; the largest single step of a C¹ ramp is
        // ~1.5 × the mean, well under this bound. A step discontinuity would be ~0.5.
        assert!(worst_step < 0.01, "fade jumped by {worst_step} between adjacent samples");
    }

    #[test]
    fn lod_fade_alpha_has_zero_slope_at_the_band_edges() {
        // This is the property a linear ramp lacks, and the reason the cubic is used:
        // the dissolve has to ease in and out or the band edges read as sweeping rings.
        const START: f32 = 18.0;
        const END: f32 = 22.0;
        let epsilon = 0.002;
        let near_edge_slope = (1.0 - lod_fade_alpha(START + epsilon, START, END)) / epsilon;
        let far_edge_slope = lod_fade_alpha(END - epsilon, START, END) / epsilon;
        let mid_slope = (lod_fade_alpha(20.0 - epsilon, START, END)
            - lod_fade_alpha(20.0 + epsilon, START, END))
            / (2.0 * epsilon);
        assert!(near_edge_slope < mid_slope * 0.1);
        assert!(far_edge_slope < mid_slope * 0.1);
    }

    #[test]
    fn lod_fade_alpha_defends_degenerate_bands_and_bad_input() {
        // Zero-width band degrades to a hard step at band_start.
        assert_eq!(lod_fade_alpha(17.9, 18.0, 18.0), 1.0);
        assert_eq!(lod_fade_alpha(18.0, 18.0, 18.0), 0.0);
        // Inverted band does the same rather than producing a negative or >1 weight.
        assert_eq!(lod_fade_alpha(10.0, 18.0, 4.0), 1.0);
        assert_eq!(lod_fade_alpha(30.0, 18.0, 4.0), 0.0);
        // Non-finite input keeps geometry on screen.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(lod_fade_alpha(bad, 18.0, 22.0), 1.0);
            assert_eq!(lod_fade_alpha(20.0, bad, 22.0), 1.0);
            assert_eq!(lod_fade_alpha(20.0, 18.0, bad), 1.0);
        }
    }

    // ── scale_in_factor ────────────────────────────────────────────────────────

    #[test]
    fn scale_in_factor_grows_from_nothing_at_the_ring_edge() {
        assert_eq!(scale_in_factor(0.0, DEFAULT_SCALE_IN_BAND), 0.0);
        assert_eq!(scale_in_factor(-1.0, DEFAULT_SCALE_IN_BAND), 0.0);
        assert_eq!(scale_in_factor(DEFAULT_SCALE_IN_BAND, DEFAULT_SCALE_IN_BAND), 1.0);
        assert_eq!(scale_in_factor(50.0, DEFAULT_SCALE_IN_BAND), 1.0);
        assert!(
            (scale_in_factor(DEFAULT_SCALE_IN_BAND * 0.5, DEFAULT_SCALE_IN_BAND) - 0.5).abs()
                < 1.0e-6
        );
    }

    #[test]
    fn scale_in_factor_is_continuous_and_monotonic_across_the_band() {
        const BAND: f32 = DEFAULT_SCALE_IN_BAND;
        const STEPS: u32 = 4000;
        let mut previous = scale_in_factor(-1.0, BAND);
        let mut worst_step = 0.0f32;
        for i in 0..=STEPS {
            let d = -1.0 + (i as f32 / STEPS as f32) * 5.0;
            let s = scale_in_factor(d, BAND);
            assert!((0.0..=1.0).contains(&s), "scale {s} left the unit range");
            assert!(s >= previous - 1.0e-6, "scale-in must be non-decreasing");
            worst_step = worst_step.max((s - previous).abs());
            previous = s;
        }
        assert!(worst_step < 0.01, "scale-in jumped by {worst_step} between adjacent samples");
        assert_eq!(previous, 1.0);
    }

    #[test]
    fn scale_in_factor_has_zero_slope_at_both_ends() {
        // Zero slope at the edge means no lurch as a tile becomes resident; zero slope
        // at the far end means the growth does not visibly stop.
        const BAND: f32 = DEFAULT_SCALE_IN_BAND;
        let epsilon = 0.001;
        let edge_slope = scale_in_factor(epsilon, BAND) / epsilon;
        let far_slope = (1.0 - scale_in_factor(BAND - epsilon, BAND)) / epsilon;
        let mid_slope = (scale_in_factor(BAND * 0.5 + epsilon, BAND)
            - scale_in_factor(BAND * 0.5 - epsilon, BAND))
            / (2.0 * epsilon);
        assert!(edge_slope < mid_slope * 0.1, "scale-in starts too abruptly");
        assert!(far_slope < mid_slope * 0.1, "scale-in stops too abruptly");
    }

    #[test]
    fn scale_in_factor_defends_bad_bands_and_input() {
        // No band means no scale-in: inside the ring is full size, the edge itself is not.
        assert_eq!(scale_in_factor(1.0, 0.0), 1.0);
        assert_eq!(scale_in_factor(0.0, 0.0), 0.0);
        assert_eq!(scale_in_factor(1.0, -2.0), 1.0);
        assert_eq!(scale_in_factor(1.0, f32::NAN), 1.0);
        // An unknown position suppresses the blade rather than popping it in at full size.
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(scale_in_factor(bad, DEFAULT_SCALE_IN_BAND), 0.0);
        }
    }
}
