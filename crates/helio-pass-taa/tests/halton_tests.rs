// Tests for helio-pass-taa: R1/R2 low-discrepancy jitter sequence.
//
// The R1/R2 sequence is a 2D generalisation of the golden ratio based on the
// plastic ratio (ρ₁ ≈ 1.324717957, ρ₂ = ρ₁² ≈ 1.754877666).
//
// Properties tested:
//   - All values in [-0.5, 0.5]
//   - Mean ≈ 0
//   - No repeats within a large window
//   - Deterministic
//
// All tests are pure math — no GPU device required.

/// Replicates the `r1_r2_jitter()` function from `lib.rs`.
fn r1_r2_jitter(frame: u64) -> [f32; 2] {
    const INV_R1: f64 = 0.7548776662466927;
    const INV_R2: f64 = 0.5698402905980539;
    const PHASE: f64 = 0.5;
    let fx = frame as f64 * INV_R1 + PHASE;
    let fy = frame as f64 * INV_R2 + PHASE;
    [(fx.fract() - 0.5) as f32, (fy.fract() - 0.5) as f32]
}

// ── Range tests ───────────────────────────────────────────────────────────────

#[test]
fn all_x_between_minus_half_and_half() {
    for frame in 0..256u64 {
        let [jx, _] = r1_r2_jitter(frame);
        assert!(
            jx >= -0.5 && jx < 0.5,
            "frame {frame}: jx = {jx} not in [-0.5, 0.5)"
        );
    }
}

#[test]
fn all_y_between_minus_half_and_half() {
    for frame in 0..256u64 {
        let [_, jy] = r1_r2_jitter(frame);
        assert!(
            jy >= -0.5 && jy < 0.5,
            "frame {frame}: jy = {jy} not in [-0.5, 0.5)"
        );
    }
}

// ── Uniqueness tests ──────────────────────────────────────────────────────────

#[test]
fn first_256_values_all_unique() {
    let mut set = std::collections::HashSet::new();
    for frame in 0..256u64 {
        let [jx, jy] = r1_r2_jitter(frame);
        let key = ((jx * 1_000_000.0) as i32, (jy * 1_000_000.0) as i32);
        assert!(set.insert(key), "duplicate at frame {frame}: ({jx}, {jy})");
    }
}

// ── Mean tests ────────────────────────────────────────────────────────────────

#[test]
fn x_mean_close_to_0() {
    let mean = (0..256).map(|f| r1_r2_jitter(f)[0]).sum::<f32>() / 256.0;
    assert!(mean.abs() < 0.05, "mean_x = {mean}");
}

#[test]
fn y_mean_close_to_0() {
    let mean = (0..256).map(|f| r1_r2_jitter(f)[1]).sum::<f32>() / 256.0;
    assert!(mean.abs() < 0.05, "mean_y = {mean}");
}

// ── Sign coverage tests ───────────────────────────────────────────────────────

#[test]
fn x_covers_both_halves() {
    let below = (0..256).filter(|f| r1_r2_jitter(*f)[0] < 0.0).count();
    let above = (0..256).filter(|f| r1_r2_jitter(*f)[0] >= 0.0).count();
    assert!(below > 0 && above > 0, "below={below} above={above}");
}

#[test]
fn y_covers_both_halves() {
    let below = (0..256).filter(|f| r1_r2_jitter(*f)[1] < 0.0).count();
    let above = (0..256).filter(|f| r1_r2_jitter(*f)[1] >= 0.0).count();
    assert!(below > 0 && above > 0, "below={below} above={above}");
}

// ── Deterministic test ────────────────────────────────────────────────────────

#[test]
fn deterministic_sequence() {
    let first: Vec<[f32; 2]> = (0..64).map(r1_r2_jitter).collect();
    let second: Vec<[f32; 2]> = (0..64).map(r1_r2_jitter).collect();
    assert_eq!(first, second);
}

// ── Variance (low-discrepancy) test ───────────────────────────────────────────

#[test]
fn sequence_variance_x_is_reasonable() {
    let samples: Vec<f32> = (0..256).map(|f| r1_r2_jitter(f)[0]).collect();
    let mean = samples.iter().sum::<f32>() / 256.0;
    let var = samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / 256.0;
    // Uniform distribution on [-0.5, 0.5] has variance 1/12 ≈ 0.0833
    // Low-discrepancy should be close to this.
    assert!((var - 1.0 / 12.0).abs() < 0.02, "var_x = {var}");
}

#[test]
fn sequence_variance_y_is_reasonable() {
    let samples: Vec<f32> = (0..256).map(|f| r1_r2_jitter(f)[1]).collect();
    let mean = samples.iter().sum::<f32>() / 256.0;
    let var = samples.iter().map(|y| (y - mean) * (y - mean)).sum::<f32>() / 256.0;
    assert!((var - 1.0 / 12.0).abs() < 0.02, "var_y = {var}");
}

// ── R1/R2 sequence doesn't repeat at N=16 (unlike Halton) ────────────────────

#[test]
fn no_repeat_at_modulo_16() {
    let base: Vec<[f32; 2]> = (0..16).map(r1_r2_jitter).collect();
    let next: Vec<[f32; 2]> = (16..32).map(r1_r2_jitter).collect();
    assert_ne!(base, next, "R1/R2 repeats at 16 — this would be wrong");
}
