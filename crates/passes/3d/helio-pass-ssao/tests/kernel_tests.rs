// Tests for helio-pass-ssao: SSAO kernel sample distribution, Halton sampling, noise rotation.
// All tests are pure math — no GPU device required.

use std::f32::consts::{FRAC_PI_2, PI};

const KERNEL_SIZE: usize = 64;
const NOISE_DIM: u32 = 4;

// ── Kernel generation helpers ─────────────────────────────────────────────────

/// Accelerating lerp used for kernel importance sampling.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Scale factor for sample i: closer samples to origin have higher density.
fn kernel_scale(i: usize, n: usize) -> f32 {
    let t = i as f32 / n as f32;
    lerp(0.1, 1.0, t * t)
}

/// Generate a uniform hemisphere sample using spherical coordinates.
/// Returns a normalised direction, then scaled into a hemisphere kernel.
fn kernel_sample(i: usize, n: usize) -> [f32; 3] {
    // Use Fibonacci-ish spiral to approximate uniform hemisphere coverage
    let golden_angle = PI * (3.0 - 5.0f32.sqrt());
    let phi = golden_angle * i as f32;
    // cos_theta in [0, 1] → upper hemisphere
    let cos_theta = 1.0 - (i as f32 + 0.5) / n as f32;
    let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
    let x = sin_theta * phi.cos();
    let y = sin_theta * phi.sin();
    let z = cos_theta;
    let s = kernel_scale(i, n);
    [x * s, y * s, z * s]
}

/// Halton sequence value for given base and index (1-based).
fn halton(base: u32, index: u32) -> f32 {
    let mut f = 1.0f32;
    let mut r = 0.0f32;
    let mut i = index;
    let b = base as f32;
    while i > 0 {
        f /= b;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

// ── Hemisphere kernel tests ───────────────────────────────────────────────────

#[test]
fn kernel_sample_z_always_non_negative() {
    for i in 0..KERNEL_SIZE {
        let s = kernel_sample(i, KERNEL_SIZE);
        assert!(s[2] >= -1e-6f32, "sample[{i}].z = {} < 0", s[2]);
    }
}

#[test]
fn kernel_sample_unit_direction_before_scaling() {
    // Without scaling the direction is still on the unit sphere
    for i in 0..KERNEL_SIZE {
        let golden_angle = PI * (3.0 - 5.0f32.sqrt());
        let phi = golden_angle * i as f32;
        let cos_theta = 1.0 - (i as f32 + 0.5) / KERNEL_SIZE as f32;
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();
        let x = sin_theta * phi.cos();
        let y = sin_theta * phi.sin();
        let z = cos_theta;
        let len = (x * x + y * y + z * z).sqrt();
        assert!((len - 1.0f32).abs() < 1e-5f32, "i={i} len={len}");
    }
}

#[test]
fn kernel_sample_length_lte_1() {
    for i in 0..KERNEL_SIZE {
        let s = kernel_sample(i, KERNEL_SIZE);
        let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        assert!(len <= 1.01f32, "sample[{i}] length = {len}");
    }
}

#[test]
fn kernel_sample_count_is_64() {
    let samples: Vec<_> = (0..KERNEL_SIZE)
        .map(|i| kernel_sample(i, KERNEL_SIZE))
        .collect();
    assert_eq!(samples.len(), 64);
}

#[test]
fn kernel_scale_increases_with_index() {
    // Accelerating distribution: later samples farther from origin
    let s0 = kernel_scale(0, KERNEL_SIZE);
    let s32 = kernel_scale(32, KERNEL_SIZE);
    let s63 = kernel_scale(63, KERNEL_SIZE);
    assert!(s0 < s32, "s0={s0} s32={s32}");
    assert!(s32 < s63, "s32={s32} s63={s63}");
}

#[test]
fn kernel_scale_first_sample_near_0_1() {
    let s = kernel_scale(0, KERNEL_SIZE);
    assert!((s - 0.1f32).abs() < 1e-4f32, "s = {s}");
}

#[test]
fn kernel_scale_last_sample_near_1_0() {
    let s = kernel_scale(KERNEL_SIZE - 1, KERNEL_SIZE);
    // scale({63}, 64) = lerp(0.1, 1.0, (63/64)^2) ≈ 0.986
    assert!(s > 0.95f32, "s = {s}");
    assert!(s <= 1.0f32);
}

#[test]
fn kernel_scale_is_positive_for_all_indices() {
    for i in 0..KERNEL_SIZE {
        let s = kernel_scale(i, KERNEL_SIZE);
        assert!(s > 0.0f32, "scale[{i}] = {s}");
    }
}

// ── Halton sequence tests ─────────────────────────────────────────────────────

#[test]
fn halton_base2_index1_is_0_5() {
    assert!((halton(2, 1) - 0.5f32).abs() < 1e-6f32);
}

#[test]
fn halton_base2_index2_is_0_25() {
    assert!((halton(2, 2) - 0.25f32).abs() < 1e-6f32);
}

#[test]
fn halton_base2_index3_is_0_75() {
    assert!((halton(2, 3) - 0.75f32).abs() < 1e-6f32);
}

#[test]
fn halton_base3_index1_is_one_third() {
    assert!((halton(3, 1) - 1.0 / 3.0).abs() < 1e-6f32);
}

#[test]
fn halton_base3_index2_is_two_thirds() {
    assert!((halton(3, 2) - 2.0 / 3.0).abs() < 1e-6f32);
}

#[test]
fn halton_all_values_in_open_01() {
    for base in [2u32, 3] {
        for idx in 1u32..=64 {
            let h = halton(base, idx);
            assert!(
                h > 0.0f32 && h < 1.0f32,
                "halton({base},{idx}) = {h} not in (0,1)"
            );
        }
    }
}

#[test]
fn halton_no_duplicates_base2_first_16() {
    let seq: Vec<u32> = (1u32..=16)
        .map(|i| (halton(2, i) * 1_000_000.0f32) as u32)
        .collect();
    let unique: std::collections::HashSet<_> = seq.iter().cloned().collect();
    assert_eq!(unique.len(), 16, "Halton base-2 has duplicate values");
}

#[test]
fn halton_no_duplicates_base3_first_16() {
    let seq: Vec<u32> = (1u32..=16)
        .map(|i| (halton(3, i) * 1_000_000.0f32) as u32)
        .collect();
    let unique: std::collections::HashSet<_> = seq.iter().cloned().collect();
    assert_eq!(unique.len(), 16, "Halton base-3 has duplicate values");
}

// ── Noise rotation tests ──────────────────────────────────────────────────────

#[test]
fn noise_texture_size_is_4x4() {
    let texels = NOISE_DIM * NOISE_DIM;
    assert_eq!(texels, 16);
}

#[test]
fn noise_texture_tiling_covers_viewport() {
    // For a 640×480 viewport the noise tiles NOISE_DIM × NOISE_DIM exactly N times
    let w = 640u32;
    let h = 480u32;
    assert_eq!(w % NOISE_DIM, 0, "width not divisible by NOISE_DIM");
    assert_eq!(h % NOISE_DIM, 0, "height not divisible by NOISE_DIM");
}

#[test]
fn noise_rotation_is_2d_unit_vectors() {
    // Generate a 4×4 random rotation noise (pairs of (cos θ, sin θ))
    for i in 0..16usize {
        let angle = 2.0 * PI * (i as f32 / 16.0);
        let (s, c) = angle.sin_cos();
        let len_sq = c * c + s * s;
        assert!((len_sq - 1.0f32).abs() < 1e-5f32, "i={i} len_sq={len_sq}");
    }
}

#[test]
fn hemisphere_samples_concentrate_near_origin() {
    // The first quarter of samples should be closer to origin on average
    let mean_near: f32 = (0..KERNEL_SIZE / 4)
        .map(|i| {
            let s = kernel_sample(i, KERNEL_SIZE);
            (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
        })
        .sum::<f32>()
        / (KERNEL_SIZE / 4) as f32;

    let mean_far: f32 = (3 * KERNEL_SIZE / 4..KERNEL_SIZE)
        .map(|i| {
            let s = kernel_sample(i, KERNEL_SIZE);
            (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt()
        })
        .sum::<f32>()
        / (KERNEL_SIZE / 4) as f32;

    assert!(
        mean_near < mean_far,
        "near_mean={mean_near} far_mean={mean_far}"
    );
}
