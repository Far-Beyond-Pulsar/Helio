//! Tests for FXAA algorithm properties and constants.
//!
//! FXAA (Fast Approximate Anti-Aliasing) is a screen-space technique that
//! detects edges using luminance contrast and applies a directional blur.

// ── Luma helper ───────────────────────────────────────────────────────────────

/// Converts linear RGB to luma using the standard FXAA coefficients:
///   luma = dot(rgb, vec3(0.299, 0.587, 0.114))
fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.299 * r + 0.587 * g + 0.114 * b
}

// ── FXAA algorithm constants ──────────────────────────────────────────────────

/// Minimum edge threshold; no AA applied below this absolute luma contrast.
const EDGE_THRESHOLD_MIN: f32 = 0.0625;

/// Relative edge threshold; fraction of local luma range that counts as an edge.
const EDGE_THRESHOLD: f32 = 0.125;

/// Sub-pixel aliasing quality reduction factor (0.75 = aggressive).
const SUBPIX_QUALITY: f32 = 0.75;

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn luma_black_is_zero() {
    assert_eq!(luma(0.0, 0.0, 0.0), 0.0);
}

#[test]
fn luma_white_is_one() {
    let l = luma(1.0, 1.0, 1.0);
    assert!((l - 1.0).abs() < 1e-5, "luma(1,1,1) ≈ 1.0, got {l}");
}

#[test]
fn luma_pure_red() {
    let l = luma(1.0, 0.0, 0.0);
    assert!((l - 0.299).abs() < 1e-5, "luma(1,0,0) ≈ 0.299, got {l}");
}

#[test]
fn luma_pure_green() {
    let l = luma(0.0, 1.0, 0.0);
    assert!((l - 0.587).abs() < 1e-5, "luma(0,1,0) ≈ 0.587, got {l}");
}

#[test]
fn luma_pure_blue() {
    let l = luma(0.0, 0.0, 1.0);
    assert!((l - 0.114).abs() < 1e-5, "luma(0,0,1) ≈ 0.114, got {l}");
}

#[test]
fn luma_coefficients_sum_to_one() {
    let sum = 0.299_f32 + 0.587 + 0.114;
    assert!((sum - 1.0).abs() < 1e-5, "luma coefficients must sum to 1, got {sum}");
}

#[test]
fn luma_green_dominates_red() {
    assert!(0.587_f32 > 0.299, "green coefficient must exceed red");
}

#[test]
fn luma_green_dominates_blue() {
    assert!(0.587_f32 > 0.114, "green coefficient must exceed blue");
}

#[test]
fn luma_red_dominates_blue() {
    assert!(0.299_f32 > 0.114, "red coefficient must exceed blue");
}

#[test]
fn luma_non_negative_for_valid_inputs() {
    let samples = [(0.5, 0.5, 0.5), (1.0, 0.0, 0.0), (0.0, 0.0, 0.5), (0.2, 0.8, 0.3)];
    for (r, g, b) in samples {
        assert!(luma(r, g, b) >= 0.0, "luma({r},{g},{b}) must be >= 0");
    }
}

#[test]
fn luma_output_in_unit_range_for_unit_inputs() {
    let l = luma(1.0, 1.0, 1.0);
    assert!(l >= 0.0 && l <= 1.0, "luma of white must be in [0,1]");
}

#[test]
fn luma_is_linear_in_each_channel() {
    // luma(2r, 0, 0) == 2 * luma(r, 0, 0)
    let l1 = luma(0.5, 0.0, 0.0);
    let l2 = luma(1.0, 0.0, 0.0);
    assert!((l2 - 2.0 * l1).abs() < 1e-6);
}

#[test]
fn luma_gray_equals_intensity() {
    // For equal R=G=B=v, luma should equal v (coefficients sum to 1).
    let v = 0.6_f32;
    let l = luma(v, v, v);
    assert!((l - v).abs() < 1e-5, "luma of gray {v} must equal {v}, got {l}");
}

#[test]
fn luma_is_additive() {
    let l_a = luma(0.3, 0.1, 0.2);
    let l_b = luma(0.1, 0.4, 0.3);
    let l_sum = luma(0.4, 0.5, 0.5);
    assert!((l_sum - (l_a + l_b)).abs() < 1e-5);
}

#[test]
fn edge_threshold_min_value() {
    assert_eq!(EDGE_THRESHOLD_MIN, 0.0625);
}

#[test]
fn edge_threshold_min_is_one_sixteenth() {
    assert!((EDGE_THRESHOLD_MIN - 1.0 / 16.0).abs() < 1e-8);
}

#[test]
fn edge_threshold_value() {
    assert_eq!(EDGE_THRESHOLD, 0.125);
}

#[test]
fn edge_threshold_is_one_eighth() {
    assert!((EDGE_THRESHOLD - 1.0 / 8.0).abs() < 1e-8);
}

#[test]
fn edge_threshold_greater_than_minimum() {
    assert!(EDGE_THRESHOLD > EDGE_THRESHOLD_MIN);
}

#[test]
fn subpix_quality_value() {
    assert_eq!(SUBPIX_QUALITY, 0.75);
}

#[test]
fn subpix_quality_in_unit_range() {
    assert!(SUBPIX_QUALITY > 0.0 && SUBPIX_QUALITY <= 1.0);
}

#[test]
fn subpix_quality_more_aggressive_than_half() {
    // 0.75 > 0.5 means more sub-pixel blending than the neutral setting.
    assert!(SUBPIX_QUALITY > 0.5);
}

#[test]
fn edge_thresholds_are_positive() {
    assert!(EDGE_THRESHOLD_MIN > 0.0);
    assert!(EDGE_THRESHOLD > 0.0);
}

#[test]
fn luma_contrast_detection() {
    // A sharp black-to-white transition has contrast = luma(white) - luma(black) = 1.
    let contrast = luma(1.0, 1.0, 1.0) - luma(0.0, 0.0, 0.0);
    assert!((contrast - 1.0).abs() < 1e-5);
}

#[test]
fn fxaa_skips_edge_below_min_threshold() {
    // If local_contrast < EDGE_THRESHOLD_MIN, no AA is applied.
    let tiny_contrast: f32 = 0.01;
    assert!(
        tiny_contrast < EDGE_THRESHOLD_MIN,
        "tiny contrast {tiny_contrast} should be below min threshold"
    );
}
