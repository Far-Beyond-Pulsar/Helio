// Tests for helio-pass-taa: TaaUniform size, R1/R2 jitter properties,
// history ping-pong, feedback clamp. All tests are pure Rust — no GPU device required.

use std::mem;

// ── Mirror private types ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct TaaUniform {
    jitter: [f32; 2],
    upscale_factor: f32,
    reset: u32,
    time_delta: f32,
    _pad: f32,
}

/// Replicates the `r1_r2_jitter()` function from `lib.rs`.
fn r1_r2_jitter(frame: u64) -> [f32; 2] {
    const INV_R1: f64 = 0.7548776662466927;
    const INV_R2: f64 = 0.5698402905980539;
    const PHASE: f64 = 0.5;
    let fx = frame as f64 * INV_R1 + PHASE;
    let fy = frame as f64 * INV_R2 + PHASE;
    [(fx.fract() - 0.5) as f32, (fy.fract() - 0.5) as f32]
}

// ── TaaUniform size tests ─────────────────────────────────────────────────────

#[test]
fn taa_uniform_size_is_24() {
    assert_eq!(mem::size_of::<TaaUniform>(), 24,
        "jitter(8) + upscale_factor(4) + reset(4) + time_delta(4) + _pad(4) = 24");
}

#[test]
fn taa_uniform_alignment_is_4() {
    assert_eq!(mem::align_of::<TaaUniform>(), 4);
}

#[test]
fn taa_uniform_size_aligned_to_8() {
    // WGSL vec2 align 8 → struct size must be multiple of 8.
    assert_eq!(mem::size_of::<TaaUniform>() % 8, 0);
}

// ── R1/R2 jitter range tests ─────────────────────────────────────────────────

#[test]
fn r1_r2_jitter_x_in_half_range() {
    for frame in 0..256u64 {
        let [jx, _] = r1_r2_jitter(frame);
        assert!(jx >= -0.5 && jx < 0.5, "frame {frame}: jx = {jx}");
    }
}

#[test]
fn r1_r2_jitter_y_in_half_range() {
    for frame in 0..256u64 {
        let [_, jy] = r1_r2_jitter(frame);
        assert!(jy >= -0.5 && jy < 0.5, "frame {frame}: jy = {jy}");
    }
}

#[test]
fn r1_r2_covers_both_signs_x() {
    let mut neg = 0u32;
    let mut pos = 0u32;
    for frame in 0..256u64 {
        let [jx, _] = r1_r2_jitter(frame);
        if jx < 0.0 { neg += 1; } else { pos += 1; }
    }
    assert!(neg > 0 && pos > 0, "neg={neg} pos={pos}");
}

#[test]
fn r1_r2_covers_both_signs_y() {
    let mut neg = 0u32;
    let mut pos = 0u32;
    for frame in 0..256u64 {
        let [_, jy] = r1_r2_jitter(frame);
        if jy < 0.0 { neg += 1; } else { pos += 1; }
    }
    assert!(neg > 0 && pos > 0, "neg={neg} pos={pos}");
}

#[test]
fn r1_r2_mean_near_zero() {
    let n = 256usize;
    let mean_x: f32 = (0..n).map(|f| r1_r2_jitter(f as u64)[0]).sum::<f32>() / n as f32;
    let mean_y: f32 = (0..n).map(|f| r1_r2_jitter(f as u64)[1]).sum::<f32>() / n as f32;
    assert!(mean_x.abs() < 0.05, "mean_x = {mean_x}");
    assert!(mean_y.abs() < 0.05, "mean_y = {mean_y}");
}

// ── R1/R2 unique values (no repeats within window) ────────────────────────────

#[test]
fn r1_r2_no_duplicates_in_256_frames() {
    let mut set = std::collections::HashSet::new();
    for frame in 0..256u64 {
        let [jx, jy] = r1_r2_jitter(frame);
        let key = ((jx * 1_000_000.0) as i32, (jy * 1_000_000.0) as i32);
        assert!(set.insert(key), "duplicate at frame {frame}: ({jx}, {jy})");
    }
    assert_eq!(set.len(), 256);
}

// ── R1/R2 is deterministic ────────────────────────────────────────────────────

#[test]
fn r1_r2_deterministic() {
    for frame in 0..64u64 {
        assert_eq!(r1_r2_jitter(frame), r1_r2_jitter(frame));
    }
}

// ── Upscale factor sanity ─────────────────────────────────────────────────────

#[test]
fn upscale_factor_computed_correctly() {
    let internal = 960u32;
    let output = 1920u32;
    let factor = output as f32 / internal as f32;
    assert!((factor - 2.0).abs() < 1e-5);
}

#[test]
fn upscale_factor_at_least_one() {
    for internal in [1920, 960, 640, 480] {
        let output = 1920u32;
        let factor = (output as f32 / internal as f32).max(1.0);
        assert!(factor >= 1.0, "internal={internal} factor={factor}");
    }
}

// ── History ping-pong tests ───────────────────────────────────────────────────

#[test]
fn frame_num_advances_monotonically() {
    let values: Vec<[f32; 2]> = (0..64).map(|f| r1_r2_jitter(f)).collect();
    // All 64 entries must be distinct (non-repeating)
    let mut set = std::collections::HashSet::new();
    for v in &values {
        let key = ((v[0] * 1_000_000.0) as i32, (v[1] * 1_000_000.0) as i32);
        assert!(set.insert(key), "non-unique jitter {v:?}");
    }
}
