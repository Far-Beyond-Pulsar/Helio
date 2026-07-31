//! Foliage quality presets.
//!
//! Mirrors the shape of `helio_pass_virtual_geometry::LodQuality`: a four-variant enum
//! whose variants are the only platform difference in the whole foliage stack. Nothing
//! in the plan's platform matrix is `#[cfg]`-ed out — every pass compiles and runs
//! everywhere, and a mobile tier is a `FoliageQuality::Low` rather than a second code
//! path. That is a deliberate constraint: the moment a quality tier becomes a `cfg`, the
//! low tier stops being tested by anyone's desktop build.

/// Quality preset scaling the foliage ring, density, LOD ladder and cull granularity.
///
/// The four knobs are not independent in practice — they trade against each other for a
/// roughly fixed frame cost — but they are separate methods so a caller with an unusual
/// budget (a wide vista at low density, say) can mix them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FoliageQuality {
    /// Mobile and integrated tier. Caps the ring at 48 m and the blade arena at 4 MiB,
    /// per the plan's platform matrix — storage-buffer limits on mobile are the binding
    /// constraint, not fill rate.
    Low,
    /// The reference tier the 3.0 ms perf gate is measured against.
    #[default]
    Medium,
    High,
    /// Everything widened. Ultra's ring reaches the 256 m the terrain capture is sized
    /// for, so it is the tier where the capture resolution, not the ring, becomes the
    /// limit.
    Ultra,
}

impl FoliageQuality {
    /// Radius of the resident tile ring in metres.
    ///
    /// This is the outer edge of *placement*, not of drawing — geometry stops at the
    /// last LOD distance and the terrain material takes over beyond it. Setting a ring
    /// radius shorter than the type's final LOD distance is the one combination that
    /// pops: blades would be culled by residency before the cross-fade to terrain
    /// shading finished. [`FoliageQuality::clamps_lod_ladder`] is the check for that.
    pub fn ring_radius(self) -> f32 {
        match self {
            FoliageQuality::Low => 48.0,
            FoliageQuality::Medium => 128.0,
            FoliageQuality::High => 176.0,
            FoliageQuality::Ultra => 256.0,
        }
    }

    /// Multiplier applied to every foliage type's authored `density`.
    ///
    /// Density scales the blade count quadratically with the ring radius, so this and
    /// [`FoliageQuality::ring_radius`] together are what actually decide arena
    /// occupancy. Low is a 0.35× density over a 48 m ring — about 5% of Medium's blade
    /// count, which is what makes the 4 MiB arena in
    /// [`FoliageQuality::blade_arena_bytes`] sufficient rather than merely a cap.
    pub fn density_multiplier(self) -> f32 {
        match self {
            FoliageQuality::Low => 0.35,
            FoliageQuality::Medium => 1.0,
            FoliageQuality::High => 1.35,
            FoliageQuality::Ultra => 2.0,
        }
    }

    /// Multiplier applied to every LOD transition distance, i.e. the `quality_scale`
    /// argument to [`crate::select_blade_lod`].
    ///
    /// Pushing the ladder out moves blades into finer LODs at a given distance, so this
    /// buys silhouette quality with raster cost.
    ///
    /// Every value here is constrained from above by
    /// [`FoliageQuality::ring_radius`]: the scaled ladder must finish *inside* the ring,
    /// or residency evicts blades that are still mid-cross-fade. Low is the binding
    /// case — its 48 m ring is fixed by the mobile storage limit in the plan's §13, so
    /// its scale cannot exceed 48/120 = 0.4 against the default ladder, and 0.35 leaves
    /// a tile of margin. That coupling is not obvious from either number alone, which is
    /// why [`FoliageQuality::clamps_lod_ladder`] exists and is asserted over every
    /// preset in this module's tests.
    pub fn lod_distance_scale(self) -> f32 {
        match self {
            FoliageQuality::Low => 0.35,
            FoliageQuality::Medium => 1.0,
            FoliageQuality::High => 1.3,
            FoliageQuality::Ultra => 1.75,
        }
    }

    /// Blades per cull cluster — the granularity of the second cull dispatch and of the
    /// L3 clump card.
    ///
    /// Always a power of two, and never finer than 16: at 1 M blades a 4×4 cluster is
    /// already 62 500 lanes, and halving the cluster doubles the cull dispatch to buy
    /// culling precision that the Hi-Z test cannot actually resolve at that footprint.
    /// The plan's §12 names raising this to 64 (8×8) as the first lever if the raster
    /// budget slips, which is exactly what Low does.
    pub fn cluster_granularity(self) -> u32 {
        match self {
            FoliageQuality::Low => 64,
            FoliageQuality::Medium => 16,
            FoliageQuality::High => 16,
            FoliageQuality::Ultra => 16,
        }
    }

    /// Hard ceiling on the blade arena in bytes.
    ///
    /// This is a budget, not a hint. When placement exhausts it the overflow counter
    /// increments and the excess blades are dropped — the same contract as
    /// `DEFAULT_MAX_PUBLISHED_MESHLETS`, and the same failure mode: silently truncated
    /// geometry that looks like a culling bug because the *same* blades lose the race
    /// every frame in a static scene. Anything reading this should also be surfacing the
    /// overflow counter.
    /// # Why this is the density knob, and the trap in raising quality instead
    ///
    /// The arena is divided into one equal fixed slab per resident tile, so what actually
    /// sets blades-per-square-metre is
    /// `blade_capacity / ring_capacity / FOLIAGE_TILE_SIZE_METERS²`. Raising the *quality
    /// preset* to get denser grass does not work: a higher preset grows the ring radius
    /// too, the tile count grows as its square, and the slab per tile barely moves. Ultra
    /// has four times Medium's arena and almost exactly the same per-tile density. Range
    /// and density are separate axes, and this is the density one.
    ///
    /// Medium's 64 MiB gives 4 M blades over a 128 m ring (1024 tiles of 8 m), i.e. 4096
    /// blades per 64 m² tile — 64 per m². Authored densities above that are clamped
    /// uniformly (`FoliagePlacePass::density_scale`), which thins the whole field rather
    /// than leaving bald patches, and logs once.
    ///
    /// The structural inefficiency here is real and unfixed: a tile at the ring's edge
    /// gets exactly the same slab as one under the camera, even though its blades only
    /// ever draw as clump cards at 1/16 the instance count. Making slabs vary with
    /// distance would buy a lot of density back, but it would make `blade_offset` depend
    /// on a tile's position rather than its slot, which is precisely what the fixed-slab
    /// layout exists to prevent.
    pub fn blade_arena_bytes(self) -> u64 {
        match self {
            FoliageQuality::Low => 8 * 1024 * 1024,
            FoliageQuality::Medium => 64 * 1024 * 1024,
            FoliageQuality::High => 128 * 1024 * 1024,
            FoliageQuality::Ultra => 256 * 1024 * 1024,
        }
    }

    /// Blade capacity implied by [`FoliageQuality::blade_arena_bytes`].
    pub fn blade_capacity(self) -> u64 {
        self.blade_arena_bytes() / std::mem::size_of::<crate::GpuBladeInstance>() as u64
    }

    /// Whether this preset's ring is too small for a LOD ladder ending at
    /// `final_lod_distance` metres once [`FoliageQuality::lod_distance_scale`] is applied.
    ///
    /// A `true` result means blades will be evicted by residency before they finish
    /// cross-fading into terrain shading, which is a visible ring of popping foliage.
    /// The fix is to shorten the type's ladder, not to widen the ring — the ring is
    /// sized against the memory budget.
    pub fn clamps_lod_ladder(self, final_lod_distance: f32) -> bool {
        final_lod_distance.is_finite()
            && final_lod_distance * self.lod_distance_scale() > self.ring_radius()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DEFAULT_LOD_DISTANCES;

    const ALL: [FoliageQuality; 4] = [
        FoliageQuality::Low,
        FoliageQuality::Medium,
        FoliageQuality::High,
        FoliageQuality::Ultra,
    ];

    #[test]
    fn medium_is_the_default_reference_tier() {
        assert_eq!(FoliageQuality::default(), FoliageQuality::Medium);
        assert_eq!(FoliageQuality::Medium.density_multiplier(), 1.0);
        assert_eq!(
            FoliageQuality::Medium.lod_distance_scale(),
            1.0,
            "the perf gate is measured with an unscaled ladder"
        );
    }

    #[test]
    fn presets_are_monotonic() {
        for pair in ALL.windows(2) {
            let (lower, higher) = (pair[0], pair[1]);
            assert!(
                higher.ring_radius() > lower.ring_radius(),
                "{higher:?} must reach further than {lower:?}"
            );
            assert!(higher.density_multiplier() > lower.density_multiplier());
            assert!(higher.lod_distance_scale() > lower.lod_distance_scale());
            assert!(higher.cluster_granularity() <= lower.cluster_granularity());
            assert!(higher.blade_arena_bytes() > lower.blade_arena_bytes());
        }
    }

    #[test]
    fn cluster_granularity_is_a_power_of_two_of_at_least_sixteen() {
        for quality in ALL {
            let granularity = quality.cluster_granularity();
            assert!(granularity.is_power_of_two(), "{quality:?} granularity is not a power of two");
            assert!(granularity >= 16, "{quality:?} granularity is finer than a 4x4 cluster");
        }
    }

    #[test]
    fn low_honours_the_platform_matrix_caps() {
        // Both numbers are stated in the plan's §13 as the mobile storage-limit answer.
        assert_eq!(FoliageQuality::Low.blade_arena_bytes(), 4 * 1024 * 1024);
        assert_eq!(FoliageQuality::Low.ring_radius(), 48.0);
        assert_eq!(FoliageQuality::Medium.blade_arena_bytes(), 24 * 1024 * 1024);
    }

    #[test]
    fn blade_capacity_follows_the_sixteen_byte_record() {
        assert_eq!(FoliageQuality::Medium.blade_capacity(), 24 * 1024 * 1024 / 16);
        assert_eq!(FoliageQuality::Medium.blade_capacity(), 1_572_864);
        assert!(
            FoliageQuality::Medium.blade_capacity() >= 1_000_000,
            "the reference tier must hold the 1 M blades the perf gate targets"
        );
    }

    #[test]
    fn every_preset_reaches_past_its_scaled_lod_ladder() {
        // The one combination that pops: residency evicting a blade before it has faded
        // into terrain shading. Every shipped preset must clear its own default ladder.
        let final_lod = DEFAULT_LOD_DISTANCES[3];
        for quality in ALL {
            assert!(
                !quality.clamps_lod_ladder(final_lod),
                "{quality:?} evicts blades at {} m before the {} m ladder ends",
                quality.ring_radius(),
                final_lod * quality.lod_distance_scale()
            );
        }
        // And the check itself must actually fire on a ladder that is too long.
        assert!(FoliageQuality::Low.clamps_lod_ladder(400.0));
        assert!(!FoliageQuality::Low.clamps_lod_ladder(f32::NAN));
    }

    #[test]
    fn low_uses_the_eight_by_eight_cluster_fallback() {
        // §12's named lever for recovering raster budget.
        assert_eq!(FoliageQuality::Low.cluster_granularity(), 64);
        assert_eq!(FoliageQuality::Medium.cluster_granularity(), 16);
    }
}
