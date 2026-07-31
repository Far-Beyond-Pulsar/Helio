//! The LOD cross-fade weights, and their agreement with `helio-foliage-core`.
//!
//! `FoliagePlacePass` reimplements `select_blade_lod` to decide which of the four
//! `visible_blades` regions a blade lands in; this pass reimplements the threshold
//! ladder to decide how much of that blade survives the dither. The two must agree at
//! every boundary, because inside a cross-fade band a single plant is drawn twice — once
//! as a blade and once as a card — and a disagreement about where the band is shears one
//! representation against the other, which is precisely the artefact the fade exists to
//! hide.

use helio_foliage_core::{select_blade_lod, DEFAULT_LOD_DISTANCES, FOLIAGE_LOD_NONE};
use helio_pass_foliage_gbuffer::{cross_fade_alpha, lod_threshold};

const LADDER: [f32; 4] = DEFAULT_LOD_DISTANCES; // [8, 20, 45, 120]
const BAND: f32 = 4.0;

#[test]
fn thresholds_are_the_boundaries_select_blade_lod_uses() {
    // The upper bound of LOD n is the distance at which `select_blade_lod` starts
    // returning n+1. Half-open and lower-inclusive on both sides, or the two disagree on
    // a hairline of the ring.
    for level in 0..4u32 {
        let threshold = lod_threshold(LADDER, level, 1.0);
        assert_eq!(threshold, LADDER[level as usize]);
        let just_inside = select_blade_lod(threshold - 0.001, LADDER, 1.0);
        let at_boundary = select_blade_lod(threshold, LADDER, 1.0);
        assert_eq!(just_inside, level);
        assert_eq!(at_boundary, (level + 1).min(FOLIAGE_LOD_NONE));
    }
}

#[test]
fn the_quality_scale_moves_the_whole_ladder() {
    for scale in [0.35f32, 1.0, 1.3, 1.75] {
        for level in 0..4u32 {
            assert!(
                (lod_threshold(LADDER, level, scale) - LADDER[level as usize] * scale).abs()
                    < 1.0e-4
            );
        }
    }
    // A poisoned scale falls back to 1.0 rather than collapsing every threshold to zero,
    // matching `select_blade_lod`'s defence.
    for bad in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
        assert_eq!(lod_threshold(LADDER, 1, bad), LADDER[1]);
    }
}

#[test]
fn a_scrambled_ladder_is_repaired_rather_than_skipping_a_band() {
    // `[8, 45, 20, 120]` must degrade to an empty L2, not pop straight from a blade to a
    // clump card. Same repair `select_blade_lod` applies.
    let scrambled = [8.0, 45.0, 20.0, 120.0];
    let thresholds: Vec<f32> = (0..4).map(|l| lod_threshold(scrambled, l, 1.0)).collect();
    assert_eq!(thresholds, vec![8.0, 45.0, 45.0, 120.0]);
    assert!(thresholds.windows(2).all(|w| w[1] >= w[0]));
}

#[test]
fn adjacent_lod_weights_sum_to_one_across_every_band() {
    // The property the stochastic dither rests on. If the near and far weights did not
    // sum to one, the band would render at the wrong *density* — a visible ring of
    // thinned or doubled grass sweeping across the ground as the camera moves, which is
    // worse than the pop it replaced.
    for level in 0..3u32 {
        let boundary = LADDER[level as usize];
        for step in 0..=64 {
            let distance = boundary - BAND + (step as f32 / 64.0) * BAND;
            let near = cross_fade_alpha(level, distance, LADDER, 1.0, BAND);
            let far = cross_fade_alpha(level + 1, distance, LADDER, 1.0, BAND);
            assert!(
                (near + far - 1.0).abs() < 1.0e-5,
                "LOD {level}/{} at {distance} m sum to {} instead of 1.0",
                level + 1,
                near + far
            );
        }
    }
}

#[test]
fn a_blade_well_inside_its_band_is_fully_present() {
    // Only the band edges dither. Everywhere else the alpha test is a no-op, which is
    // what keeps the cost of the fade proportional to the band width rather than to the
    // ring.
    assert_eq!(cross_fade_alpha(0, 0.0, LADDER, 1.0, BAND), 1.0);
    assert_eq!(cross_fade_alpha(0, 3.0, LADDER, 1.0, BAND), 1.0);
    assert_eq!(cross_fade_alpha(1, 12.0, LADDER, 1.0, BAND), 1.0);
    assert_eq!(cross_fade_alpha(2, 30.0, LADDER, 1.0, BAND), 1.0);
    assert_eq!(cross_fade_alpha(3, 80.0, LADDER, 1.0, BAND), 1.0);
}

#[test]
fn the_last_lod_fades_out_rather_than_being_culled() {
    // Past the final threshold the terrain material takes over (the plan's §2.7). L3
    // must therefore reach zero *smoothly* at 120 m — a hard cut there is exactly the
    // pop-out at the cull distance this whole design exists to avoid.
    assert!(cross_fade_alpha(3, 117.0, LADDER, 1.0, BAND) > 0.0);
    assert!(cross_fade_alpha(3, 117.0, LADDER, 1.0, BAND) < 1.0);
    assert_eq!(cross_fade_alpha(3, 120.0, LADDER, 1.0, BAND), 0.0);
    assert_eq!(cross_fade_alpha(3, 500.0, LADDER, 1.0, BAND), 0.0);
}

#[test]
fn every_weight_stays_in_the_unit_interval_and_is_continuous() {
    // A weight outside [0, 1] would make the alpha test either always or never pass, and
    // a discontinuity is a ring of grass appearing or vanishing all at once.
    for level in 0..4u32 {
        let mut previous = cross_fade_alpha(level, 0.0, LADDER, 1.0, BAND);
        for step in 0..=4000u32 {
            let distance = step as f32 * 0.05;
            let alpha = cross_fade_alpha(level, distance, LADDER, 1.0, BAND);
            assert!(
                (0.0..=1.0).contains(&alpha),
                "LOD {level} weight {alpha} left the unit range at {distance} m"
            );
            assert!(
                (alpha - previous).abs() < 0.02,
                "LOD {level} weight jumped by {} at {distance} m",
                (alpha - previous).abs()
            );
            previous = alpha;
        }
    }
}

#[test]
fn a_zero_width_band_degrades_to_a_hard_switch() {
    // Not a supported authoring mode, but the honest interpretation of one: no fade,
    // just the pop. It must not produce a weight outside the unit range or NaN, which
    // would take the alpha test with it.
    for level in 0..4u32 {
        for distance in [0.0f32, 7.999, 8.0, 19.9, 44.9, 119.9, 200.0] {
            let alpha = cross_fade_alpha(level, distance, LADDER, 1.0, 0.0);
            assert!((0.0..=1.0).contains(&alpha));
        }
    }
}
