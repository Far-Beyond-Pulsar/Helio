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

// ── Thresholds: exact values ──────────────────────────────────────────────────

#[test]
fn low_thresholds_lod_s0_is_0_02() {
    let (s0, _) = LodQuality::Low.thresholds();
    assert!((s0 - 0.02f32).abs() < 1e-6f32, "s0 = {s0}");
}

#[test]
fn low_thresholds_lod_s1_is_0_004() {
    let (_, s1) = LodQuality::Low.thresholds();
    assert!((s1 - 0.004f32).abs() < 1e-6f32, "s1 = {s1}");
}

#[test]
fn medium_thresholds_lod_s0_is_0_05() {
    let (s0, _) = LodQuality::Medium.thresholds();
    assert!((s0 - 0.05f32).abs() < 1e-6f32, "s0 = {s0}");
}

#[test]
fn medium_thresholds_lod_s1_is_0_010() {
    let (_, s1) = LodQuality::Medium.thresholds();
    assert!((s1 - 0.010f32).abs() < 1e-6f32, "s1 = {s1}");
}

#[test]
fn high_thresholds_lod_s0_is_0_10() {
    let (s0, _) = LodQuality::High.thresholds();
    assert!((s0 - 0.10f32).abs() < 1e-6f32, "s0 = {s0}");
}

#[test]
fn high_thresholds_lod_s1_is_0_020() {
    let (_, s1) = LodQuality::High.thresholds();
    assert!((s1 - 0.020f32).abs() < 1e-6f32, "s1 = {s1}");
}

#[test]
fn ultra_thresholds_lod_s0_is_0_18() {
    let (s0, _) = LodQuality::Ultra.thresholds();
    assert!((s0 - 0.18f32).abs() < 1e-6f32, "s0 = {s0}");
}

#[test]
fn ultra_thresholds_lod_s1_is_0_040() {
    let (_, s1) = LodQuality::Ultra.thresholds();
    assert!((s1 - 0.040f32).abs() < 1e-6f32, "s1 = {s1}");
}

// ── Invariant: lod_s0 > lod_s1 for all quality levels ────────────────────────

#[test]
fn low_s0_greater_than_s1() {
    let (s0, s1) = LodQuality::Low.thresholds();
    assert!(s0 > s1, "s0={s0} s1={s1}");
}

#[test]
fn medium_s0_greater_than_s1() {
    let (s0, s1) = LodQuality::Medium.thresholds();
    assert!(s0 > s1, "s0={s0} s1={s1}");
}

#[test]
fn high_s0_greater_than_s1() {
    let (s0, s1) = LodQuality::High.thresholds();
    assert!(s0 > s1, "s0={s0} s1={s1}");
}

#[test]
fn ultra_s0_greater_than_s1() {
    let (s0, s1) = LodQuality::Ultra.thresholds();
    assert!(s0 > s1, "s0={s0} s1={s1}");
}

// ── Monotonic quality ordering ────────────────────────────────────────────────

#[test]
fn low_s0_less_than_medium_s0() {
    let (low_s0, _) = LodQuality::Low.thresholds();
    let (med_s0, _) = LodQuality::Medium.thresholds();
    assert!(low_s0 < med_s0);
}

#[test]
fn medium_s0_less_than_high_s0() {
    let (med_s0, _) = LodQuality::Medium.thresholds();
    let (high_s0, _) = LodQuality::High.thresholds();
    assert!(med_s0 < high_s0);
}

#[test]
fn high_s0_less_than_ultra_s0() {
    let (high_s0, _) = LodQuality::High.thresholds();
    let (ultra_s0, _) = LodQuality::Ultra.thresholds();
    assert!(high_s0 < ultra_s0);
}

#[test]
fn low_s1_less_than_medium_s1() {
    let (_, low_s1) = LodQuality::Low.thresholds();
    let (_, med_s1) = LodQuality::Medium.thresholds();
    assert!(low_s1 < med_s1);
}

#[test]
fn medium_s1_less_than_high_s1() {
    let (_, med_s1) = LodQuality::Medium.thresholds();
    let (_, high_s1) = LodQuality::High.thresholds();
    assert!(med_s1 < high_s1);
}

#[test]
fn high_s1_less_than_ultra_s1() {
    let (_, high_s1) = LodQuality::High.thresholds();
    let (_, ultra_s1) = LodQuality::Ultra.thresholds();
    assert!(high_s1 < ultra_s1);
}

// ── Trait: Copy / Clone ───────────────────────────────────────────────────────

#[test]
fn lod_quality_is_copy() {
    let a = LodQuality::High;
    let b = a; // copy
    let _ = a; // still usable
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
fn lod_quality_debug_low() {
    let s = format!("{:?}", LodQuality::Low);
    assert!(s.contains("Low"), "debug output: {s}");
}

#[test]
fn lod_quality_debug_medium() {
    let s = format!("{:?}", LodQuality::Medium);
    assert!(s.contains("Medium"), "debug output: {s}");
}

#[test]
fn lod_quality_debug_high() {
    let s = format!("{:?}", LodQuality::High);
    assert!(s.contains("High"), "debug output: {s}");
}

#[test]
fn lod_quality_debug_ultra() {
    let s = format!("{:?}", LodQuality::Ultra);
    assert!(s.contains("Ultra"), "debug output: {s}");
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

/// screen_radius = (obj_radius * cot(fov/2)) / dist
fn screen_radius(obj_radius: f32, fov_rad: f32, dist: f32) -> f32 {
    let cot_half_fov = 1.0 / (fov_rad / 2.0).tan();
    obj_radius * cot_half_fov / dist
}

#[test]
fn screen_radius_decreases_with_distance() {
    let fov = std::f32::consts::FRAC_PI_2; // 90°
    let r1 = screen_radius(1.0, fov, 10.0);
    let r2 = screen_radius(1.0, fov, 100.0);
    assert!(r1 > r2, "r1={r1} r2={r2}");
}

#[test]
fn screen_radius_scales_linearly_with_object_size() {
    let fov = std::f32::consts::FRAC_PI_2;
    let dist = 50.0f32;
    let r1 = screen_radius(1.0, fov, dist);
    let r2 = screen_radius(2.0, fov, dist);
    assert!((r2 - 2.0 * r1).abs() < 1e-5f32);
}

#[test]
fn lod_quality_medium_active_at_5_percent_screen_coverage() {
    let (s0, _) = LodQuality::Medium.thresholds();
    let sr = 0.06f32; // 6% screen coverage
    // LOD 0 is active when screen_radius >= s0
    assert!(sr >= s0, "sr={sr} s0={s0}");
}

#[test]
fn lod_quality_medium_switches_lod1_at_5_percent() {
    let (s0, _) = LodQuality::Medium.thresholds();
    let sr = 0.04f32; // 4% — below s0
    assert!(sr < s0, "should be below threshold");
}
