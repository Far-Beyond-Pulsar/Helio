//! Extensive tests for the mip level calculation formula used by HiZBuildPass.
//!
//! The private `mip_levels(w, h)` function is:
//!   `(u32::BITS - max_dim.leading_zeros()).max(1)`
//! where `max_dim = w.max(h)`.
//!
//! This is equivalent to `floor(log2(max_dim)) + 1`, clamped to 1.

/// Local copy of the private `mip_levels` function for white-box testing.
fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}

// ── Power-of-two sizes ────────────────────────────────────────────────────────

#[test]
fn mip_1x1() {
    assert_eq!(mip_levels(1, 1), 1);
}

#[test]
fn mip_2x2() {
    assert_eq!(mip_levels(2, 2), 2);
}

#[test]
fn mip_4x4() {
    assert_eq!(mip_levels(4, 4), 3);
}

#[test]
fn mip_8x8() {
    assert_eq!(mip_levels(8, 8), 4);
}

#[test]
fn mip_16x16() {
    assert_eq!(mip_levels(16, 16), 5);
}

#[test]
fn mip_32x32() {
    assert_eq!(mip_levels(32, 32), 6);
}

#[test]
fn mip_64x64() {
    assert_eq!(mip_levels(64, 64), 7);
}

#[test]
fn mip_128x128() {
    assert_eq!(mip_levels(128, 128), 8);
}

#[test]
fn mip_256x256() {
    assert_eq!(mip_levels(256, 256), 9);
}

#[test]
fn mip_512x512() {
    assert_eq!(mip_levels(512, 512), 10);
}

#[test]
fn mip_1024x1024() {
    assert_eq!(mip_levels(1024, 1024), 11);
}

#[test]
fn mip_2048x2048() {
    assert_eq!(mip_levels(2048, 2048), 12);
}

#[test]
fn mip_4096x4096() {
    // 4096 = 2^12 → 13 mip levels (exceeds MAX_MIP_LEVELS=12, so the pass clamps).
    assert_eq!(mip_levels(4096, 4096), 13);
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn mip_0x0_returns_one() {
    // max(0,0) = 0 → leading_zeros(0) = 32 → 32-32 = 0 → max(0,1) = 1
    assert_eq!(mip_levels(0, 0), 1);
}

#[test]
fn mip_result_always_at_least_one() {
    for d in [0_u32, 1, 3, 7, 15, 100, 999, 4095] {
        assert!(mip_levels(d, d) >= 1, "mip_levels({d},{d}) must be ≥ 1");
    }
}

// ── Non-power-of-two ─────────────────────────────────────────────────────────

#[test]
fn mip_1920x1080() {
    // max = 1920 (11-bit value) → 11 levels
    assert_eq!(mip_levels(1920, 1080), 11);
}

#[test]
fn mip_1280x720() {
    // max = 1280 (11-bit value: 10100000000) → 11 levels
    assert_eq!(mip_levels(1280, 720), 11);
}

#[test]
fn mip_800x600() {
    // max = 800 (10-bit value) → 10 levels
    assert_eq!(mip_levels(800, 600), 10);
}

// ── Non-square ───────────────────────────────────────────────────────────────

#[test]
fn mip_1024x512_equals_1024x1024() {
    // The larger dimension (1024) drives the mip count in both cases.
    assert_eq!(mip_levels(1024, 512), mip_levels(1024, 1024));
}

#[test]
fn mip_512x1024_equals_1024x1024() {
    assert_eq!(mip_levels(512, 1024), mip_levels(1024, 1024));
}

#[test]
fn mip_is_symmetric() {
    assert_eq!(mip_levels(1920, 1080), mip_levels(1080, 1920));
    assert_eq!(mip_levels(300, 400), mip_levels(400, 300));
    assert_eq!(mip_levels(1, 2048), mip_levels(2048, 1));
}

// ── Mono­tonicity ─────────────────────────────────────────────────────────────

#[test]
fn mip_monotone_with_increasing_size() {
    assert!(mip_levels(2048, 2048) >= mip_levels(1024, 1024));
    assert!(mip_levels(1024, 1024) >= mip_levels(512, 512));
    assert!(mip_levels(512, 512) >= mip_levels(256, 256));
}

// ── Algebraic property: for N = 2^k, result = k + 1 ─────────────────────────

#[test]
fn mip_power_of_two_formula() {
    for k in 0_u32..13 {
        let n = 1_u32 << k;
        let expected = k + 1;
        assert_eq!(
            mip_levels(n, n),
            expected,
            "mip_levels({n},{n}) should be {expected}"
        );
    }
}

// ── Regression: max-dimension selection ──────────────────────────────────────

#[test]
fn mip_uses_max_dimension_not_min() {
    // A 1x1024 texture should produce 11 levels (driven by 1024), not 1.
    assert_eq!(mip_levels(1, 1024), 11);
}

#[test]
fn mip_1920x1080_driven_by_width() {
    // 1920 > 1080, so the result equals mip_levels(1920, 1920).
    assert_eq!(mip_levels(1920, 1080), mip_levels(1920, 1920));
}

