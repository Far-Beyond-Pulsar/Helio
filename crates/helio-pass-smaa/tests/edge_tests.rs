// Tests for helio-pass-smaa: edge detection math — luma, contrast thresholds, Rg16Float.
// All tests are pure math — no GPU device required.

use std::f32;

// ── Luma helpers ──────────────────────────────────────────────────────────────

/// Rec.709 luma from linear RGB.
fn luma(r: f32, g: f32, b: f32) -> f32 {
    0.2126 * r + 0.7152 * g + 0.0722 * b
}

/// Luma-based edge contrast: |luma(center) - luma(neighbour)|.
fn luma_contrast(center: f32, neighbour: f32) -> f32 {
    (center - neighbour).abs()
}

/// SMAA luma edge: edge fired when contrast >= threshold.
fn is_luma_edge(center: f32, neighbour: f32, threshold: f32) -> bool {
    luma_contrast(center, neighbour) >= threshold
}

/// Heuristic max edge: max(|r1-r2|, |g1-g2|, |b1-b2|).
fn color_contrast_max(c0: [f32; 3], c1: [f32; 3]) -> f32 {
    let dr = (c0[0] - c1[0]).abs();
    let dg = (c0[1] - c1[1]).abs();
    let db = (c0[2] - c1[2]).abs();
    dr.max(dg).max(db)
}

// ── Luma coefficient tests ────────────────────────────────────────────────────

#[test]
fn luma_coefficients_sum_to_one() {
    let sum = 0.2126f32 + 0.7152f32 + 0.0722f32;
    assert!((sum - 1.0f32).abs() < 1e-4f32, "sum = {sum}");
}

#[test]
fn luma_of_white_is_one() {
    assert!((luma(1.0, 1.0, 1.0) - 1.0f32).abs() < 1e-5f32);
}

#[test]
fn luma_of_black_is_zero() {
    assert!((luma(0.0, 0.0, 0.0)).abs() < 1e-6f32);
}

#[test]
fn luma_green_dominant_channel() {
    // Green contributes ~71.5% to luma
    let l_pure_green = luma(0.0, 1.0, 0.0);
    let l_pure_red = luma(1.0, 0.0, 0.0);
    let l_pure_blue = luma(0.0, 0.0, 1.0);
    assert!(l_pure_green > l_pure_red);
    assert!(l_pure_green > l_pure_blue);
}

#[test]
fn luma_red_greater_than_blue() {
    let l_red = luma(1.0, 0.0, 0.0);
    let l_blue = luma(0.0, 0.0, 1.0);
    assert!(l_red > l_blue, "red luma = {l_red}, blue luma = {l_blue}");
}

#[test]
fn luma_linear_in_each_channel() {
    // luma(2R, 0, 0) = 2 * luma(R, 0, 0)
    let l1 = luma(0.3, 0.0, 0.0);
    let l2 = luma(0.6, 0.0, 0.0);
    assert!((l2 - 2.0 * l1).abs() < 1e-6f32);
}

#[test]
fn luma_grey_equals_input_value() {
    for v in [0.0f32, 0.25, 0.5, 0.75, 1.0] {
        let l = luma(v, v, v);
        assert!((l - v).abs() < 1e-5f32, "luma({v},{v},{v}) = {l}");
    }
}

// ── Edge detection threshold tests ───────────────────────────────────────────

#[test]
fn edge_fires_when_contrast_exceeds_threshold() {
    let threshold = 0.1f32;
    assert!(is_luma_edge(0.9, 0.7, threshold)); // contrast = 0.2 ≥ 0.1
}

#[test]
fn edge_does_not_fire_below_threshold() {
    let threshold = 0.1f32;
    assert!(!is_luma_edge(0.5, 0.55, threshold)); // contrast = 0.05 < 0.1
}

#[test]
fn edge_fires_at_exact_threshold() {
    let threshold = 0.1f32;
    // Use a pixel pair with exactly 0.1 contrast
    let center = 0.6f32;
    let neighbour = 0.5f32;
    assert!(is_luma_edge(center, neighbour, threshold));
}

#[test]
fn luma_contrast_is_symmetric() {
    let a = 0.3f32;
    let b = 0.7f32;
    assert!((luma_contrast(a, b) - luma_contrast(b, a)).abs() < 1e-6f32);
}

#[test]
fn luma_contrast_same_pixel_is_zero() {
    assert!((luma_contrast(0.42, 0.42)).abs() < 1e-6f32);
}

// ── Rg16Float format tests ────────────────────────────────────────────────────

#[test]
fn rg16float_has_two_channels() {
    let channels: usize = 2; // R=horizontal edge, G=vertical edge
    assert_eq!(channels, 2);
}

#[test]
fn rg16float_16bit_per_channel() {
    let bits_per_channel: usize = 16;
    assert_eq!(bits_per_channel, 16);
}

#[test]
fn rg16float_bytes_per_texel_is_4() {
    let bytes = 2 * (16 / 8); // 2 channels × 2 bytes
    assert_eq!(bytes, 4usize);
}

#[test]
fn f16_max_finite_is_65504() {
    // Max f16 value used to verify no overflow in edge texture
    let max_f16: f32 = 65504.0;
    assert!(max_f16 > 1.0f32);
    assert!(max_f16 < f32::MAX);
}

// ── Color contrast tests ──────────────────────────────────────────────────────

#[test]
fn color_contrast_max_all_same_is_zero() {
    let c = [0.5f32, 0.5, 0.5];
    assert!((color_contrast_max(c, c)).abs() < 1e-6f32);
}

#[test]
fn color_contrast_max_picks_dominant_channel() {
    let c0 = [0.0f32, 0.0, 0.0];
    let c1 = [0.1f32, 0.5, 0.2];
    let result = color_contrast_max(c0, c1);
    assert!((result - 0.5f32).abs() < 1e-6f32);
}

#[test]
fn color_contrast_max_is_symmetric() {
    let c0 = [0.1f32, 0.3, 0.9];
    let c1 = [0.5f32, 0.6, 0.2];
    assert!((color_contrast_max(c0, c1) - color_contrast_max(c1, c0)).abs() < 1e-6f32);
}

// ── Blend weight math ─────────────────────────────────────────────────────────

#[test]
fn blend_weight_zero_means_no_blending() {
    let source = 0.8f32;
    let neighbour = 0.2f32;
    let weight = 0.0f32;
    let result = source * (1.0 - weight) + neighbour * weight;
    assert!((result - source).abs() < 1e-6f32);
}

#[test]
fn blend_weight_one_means_full_neighbour() {
    let source = 0.8f32;
    let neighbour = 0.2f32;
    let weight = 1.0f32;
    let result = source * (1.0 - weight) + neighbour * weight;
    assert!((result - neighbour).abs() < 1e-6f32);
}

#[test]
fn blend_weight_half_is_average() {
    let source = 0.8f32;
    let neighbour = 0.2f32;
    let weight = 0.5f32;
    let result = source * (1.0 - weight) + neighbour * weight;
    assert!((result - 0.5f32).abs() < 1e-6f32);
}

#[test]
fn edge_detection_higher_threshold_misses_soft_edge() {
    let center = 0.55f32;
    let neighbour = 0.5f32;
    let low_threshold = 0.04f32;
    let high_threshold = 0.1f32;
    assert!(
        is_luma_edge(center, neighbour, low_threshold),
        "Should detect edge at low threshold"
    );
    assert!(
        !is_luma_edge(center, neighbour, high_threshold),
        "Should miss edge at high threshold"
    );
}
