// Tests for helio-pass-virtual-geometry: LodQuality public API.
// Imports the actual crate type — no GPU device required.

use helio_pass_virtual_geometry::LodQuality;

// ── Variant existence tests ───────────────────────────────────────────────────

#[test]
fn lod_quality_low_exists() {
    let q = LodQuality::Low;
    let _ = q;
}

#[test]
fn lod_quality_medium_exists() {
    let q = LodQuality::Medium;
    let _ = q;
}

#[test]
fn lod_quality_high_exists() {
    let q = LodQuality::High;
    let _ = q;
}

#[test]
fn lod_quality_ultra_exists() {
    let q = LodQuality::Ultra;
    let _ = q;
}

// ── Default trait ─────────────────────────────────────────────────────────────

#[test]
fn lod_quality_default_is_medium() {
    let q: LodQuality = Default::default();
    assert_eq!(q, LodQuality::Medium);
}

// ── Thresholds return 7 elements ─────────────────────────────────────────────

#[test]
fn thresholds_return_7_elements() {
    for q in [LodQuality::Low, LodQuality::Medium, LodQuality::High, LodQuality::Ultra] {
        let t = q.thresholds();
        assert_eq!(t.len(), 7, "{:?}", q);
    }
}

// ── Invariant: thresholds are strictly decreasing within each level ──────────

#[test]
fn thresholds_strictly_decreasing() {
    for q in [LodQuality::Low, LodQuality::Medium, LodQuality::High, LodQuality::Ultra] {
        let t = q.thresholds();
        for i in 0..6 {
            assert!(t[i] > t[i + 1], "{:?}: t[{i}]={} not > t[{}]={}", q, t[i], i + 1, t[i + 1]);
        }
    }
}

// ── Invariant: higher quality = lower thresholds (more permissive) ───────────
// This is the FIX for the inverted threshold bug.

#[test]
fn higher_quality_has_lower_s0() {
    let low = LodQuality::Low.thresholds();
    let med = LodQuality::Medium.thresholds();
    let high = LodQuality::High.thresholds();
    let ultra = LodQuality::Ultra.thresholds();
    assert!(low[0] > med[0], "Low s0 should be > Medium s0");
    assert!(med[0] > high[0], "Medium s0 should be > High s0");
    assert!(high[0] > ultra[0], "High s0 should be > Ultra s0");
}

#[test]
fn higher_quality_has_lower_s6() {
    let low = LodQuality::Low.thresholds();
    let med = LodQuality::Medium.thresholds();
    let high = LodQuality::High.thresholds();
    let ultra = LodQuality::Ultra.thresholds();
    assert!(low[6] > med[6]);
    assert!(med[6] > high[6]);
    assert!(high[6] > ultra[6]);
}

// ── All thresholds positive and below 1.0 ────────────────────────────────────

#[test]
fn all_thresholds_positive() {
    for q in [LodQuality::Low, LodQuality::Medium, LodQuality::High, LodQuality::Ultra] {
        let t = q.thresholds();
        for (i, &v) in t.iter().enumerate() {
            assert!(v > 0.0, "{:?} t[{i}]={v} not positive", q);
        }
    }
}

#[test]
fn all_thresholds_below_1() {
    for q in [LodQuality::Low, LodQuality::Medium, LodQuality::High, LodQuality::Ultra] {
        let t = q.thresholds();
        for (i, &v) in t.iter().enumerate() {
            assert!(v < 1.0, "{:?} t[{i}]={v} >= 1.0", q);
        }
    }
}

// ── Trait: Copy / Clone ───────────────────────────────────────────────────────

#[test]
fn lod_quality_is_copy() {
    let a = LodQuality::High;
    let b = a;
    let _ = a;
    assert_eq!(b, LodQuality::High);
}

#[test]
fn lod_quality_is_clone() {
    let a = LodQuality::Ultra;
    let b = a.clone();
    assert_eq!(b, LodQuality::Ultra);
}

// ── Trait: Debug ──────────────────────────────────────────────────────────────

#[test]
fn lod_quality_debug_contains_variant_name() {
    for (q, name) in [
        (LodQuality::Low, "Low"),
        (LodQuality::Medium, "Medium"),
        (LodQuality::High, "High"),
        (LodQuality::Ultra, "Ultra"),
    ] {
        let s = format!("{:?}", q);
        assert!(s.contains(name), "debug output: {s}");
    }
}

// ── Trait: PartialEq ─────────────────────────────────────────────────────────

#[test]
fn lod_quality_eq_same_variant() {
    assert_eq!(LodQuality::Medium, LodQuality::Medium);
}

#[test]
fn lod_quality_ne_different_variants() {
    assert_ne!(LodQuality::Low, LodQuality::High);
}

// ── screen_radius formula tests ───────────────────────────────────────────────

fn screen_radius(obj_radius: f32, fov_rad: f32, dist: f32) -> f32 {
    let cot_half_fov = 1.0 / (fov_rad / 2.0).tan();
    obj_radius * cot_half_fov / dist
}

#[test]
fn screen_radius_decreases_with_distance() {
    let fov = std::f32::consts::FRAC_PI_2;
    let r1 = screen_radius(1.0, fov, 10.0);
    let r2 = screen_radius(1.0, fov, 100.0);
    assert!(r1 > r2);
}

#[test]
fn screen_radius_scales_linearly_with_object_size() {
    let fov = std::f32::consts::FRAC_PI_2;
    let dist = 50.0f32;
    let r1 = screen_radius(1.0, fov, dist);
    let r2 = screen_radius(2.0, fov, dist);
    assert!((r2 - 2.0 * r1).abs() < 1e-5);
}

// ── LOD selection correctness ────────────────────────────────────────────────
// Verify that the inverted threshold fix works: Ultra should give LOD0
// at SMALLER screen coverage than Low.

#[test]
fn ultra_gives_full_detail_at_small_screen_coverage() {
    let ultra = LodQuality::Ultra.thresholds();
    let sr = 0.01; // 1% screen coverage
    // Ultra s0 = 0.008, so 1% > 0.8% → LOD0 on Ultra
    assert!(sr >= ultra[0], "sr={sr} ultra_s0={}", ultra[0]);
}

#[test]
fn low_quality_uses_coarse_lod_at_small_screen_coverage() {
    let low = LodQuality::Low.thresholds();
    let sr = 0.01; // 1% screen coverage
    // Low s0 = 0.18, so 1% < 18% → NOT LOD0 on Low
    assert!(sr < low[0], "sr={sr} low_s0={}", low[0]);
}
