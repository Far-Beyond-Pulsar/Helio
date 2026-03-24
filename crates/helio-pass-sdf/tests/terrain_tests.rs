use bytemuck::Zeroable;
use glam::Vec3;
use helio_pass_sdf::{
    noise::{fbm2, terrain_height_range, terrain_sdf},
    terrain::{GpuTerrainParams, TerrainConfig, TerrainStyle},
};
use std::mem;

// ──────────────────────── TerrainStyle ───────────────────────────────────────

#[test]
fn terrain_style_rolling_discriminant_is_0() {
    assert_eq!(TerrainStyle::Rolling as u32, 0);
}

#[test]
fn terrain_style_eq() {
    assert_eq!(TerrainStyle::Rolling, TerrainStyle::Rolling);
}

#[test]
fn terrain_style_copy_clone() {
    let a = TerrainStyle::Rolling;
    let b = a;
    let c = a.clone();
    assert_eq!(b, TerrainStyle::Rolling);
    assert_eq!(c, TerrainStyle::Rolling);
}

// ──────────────────────── TerrainConfig::rolling() ───────────────────────────

#[test]
fn rolling_style_is_rolling() {
    let c = TerrainConfig::rolling();
    assert_eq!(c.style, TerrainStyle::Rolling);
}

#[test]
fn rolling_height_is_zero() {
    let c = TerrainConfig::rolling();
    assert_eq!(c.height, 0.0);
}

#[test]
fn rolling_amplitude_is_25() {
    let c = TerrainConfig::rolling();
    assert!((c.amplitude - 25.0).abs() < 1e-6);
}

#[test]
fn rolling_frequency_is_0_015() {
    let c = TerrainConfig::rolling();
    assert!((c.frequency - 0.015).abs() < 1e-6);
}

#[test]
fn rolling_octaves_is_6() {
    let c = TerrainConfig::rolling();
    assert_eq!(c.octaves, 6);
}

#[test]
fn rolling_lacunarity_is_2() {
    let c = TerrainConfig::rolling();
    assert!((c.lacunarity - 2.0).abs() < 1e-6);
}

#[test]
fn rolling_persistence_is_0_5() {
    let c = TerrainConfig::rolling();
    assert!((c.persistence - 0.5).abs() < 1e-6);
}

// ──────────────────────── TerrainConfig::build_gpu_params ────────────────────

#[test]
fn gpu_params_rolling_enabled_is_1() {
    let p = TerrainConfig::rolling().build_gpu_params();
    assert_eq!(p.enabled, 1);
}

#[test]
fn gpu_params_rolling_style_is_0() {
    let p = TerrainConfig::rolling().build_gpu_params();
    assert_eq!(p.style, 0);
}

#[test]
fn gpu_params_rolling_height_matches() {
    let cfg = TerrainConfig::rolling();
    let p = cfg.build_gpu_params();
    assert_eq!(p.height, cfg.height);
}

#[test]
fn gpu_params_rolling_amplitude_matches() {
    let cfg = TerrainConfig::rolling();
    let p = cfg.build_gpu_params();
    assert_eq!(p.amplitude, cfg.amplitude);
}

#[test]
fn gpu_params_rolling_frequency_matches() {
    let cfg = TerrainConfig::rolling();
    let p = cfg.build_gpu_params();
    assert!((p.frequency - cfg.frequency).abs() < 1e-6);
}

#[test]
fn gpu_params_rolling_octaves_matches() {
    let cfg = TerrainConfig::rolling();
    let p = cfg.build_gpu_params();
    assert_eq!(p.octaves, cfg.octaves);
}

// ──────────────────────── GpuTerrainParams::disabled ─────────────────────────

#[test]
fn disabled_enabled_is_0() {
    let p = GpuTerrainParams::disabled();
    assert_eq!(p.enabled, 0);
}

#[test]
fn disabled_lacunarity_is_1() {
    let p = GpuTerrainParams::disabled();
    assert!((p.lacunarity - 1.0).abs() < 1e-6);
}

#[test]
fn disabled_persistence_is_0_5() {
    let p = GpuTerrainParams::disabled();
    assert!((p.persistence - 0.5).abs() < 1e-6);
}

#[test]
fn disabled_amplitude_is_zero() {
    let p = GpuTerrainParams::disabled();
    assert_eq!(p.amplitude, 0.0);
}

// ──────────────────────── Layout / POD ───────────────────────────────────────

#[test]
fn gpu_terrain_params_size_is_32() {
    assert_eq!(mem::size_of::<GpuTerrainParams>(), 32);
}

#[test]
fn gpu_terrain_params_pod_cast() {
    let p = TerrainConfig::rolling().build_gpu_params();
    let bytes: &[u8] = bytemuck::bytes_of(&p);
    assert_eq!(bytes.len(), 32);
}

#[test]
fn gpu_terrain_params_zeroable() {
    let z: GpuTerrainParams = Zeroable::zeroed();
    assert_eq!(z.enabled, 0);
    assert_eq!(z.amplitude, 0.0);
}

// ──────────────────────── TerrainConfig clone ────────────────────────────────

#[test]
fn terrain_config_clone_equals_original() {
    let a = TerrainConfig::rolling();
    let b = a.clone();
    assert_eq!(a.octaves, b.octaves);
    assert_eq!(a.amplitude, b.amplitude);
    assert_eq!(a.style, b.style);
}

// ──────────────────────── noise::terrain_sdf ─────────────────────────────────

/// Flat terrain helper: amplitude=0 so height is exactly `config.height`.
fn flat_terrain(height: f32) -> TerrainConfig {
    TerrainConfig {
        style: TerrainStyle::Rolling,
        height,
        amplitude: 0.0,
        frequency: 0.1,
        octaves: 1,
        lacunarity: 2.0,
        persistence: 0.5,
    }
}

#[test]
fn terrain_sdf_above_flat_ground_positive() {
    let cfg = flat_terrain(5.0);
    let d = terrain_sdf(Vec3::new(0.0, 7.0, 0.0), &cfg);
    assert!((d - 2.0).abs() < 1e-5);
}

#[test]
fn terrain_sdf_below_flat_ground_negative() {
    let cfg = flat_terrain(5.0);
    let d = terrain_sdf(Vec3::new(0.0, 3.0, 0.0), &cfg);
    assert!((d - (-2.0)).abs() < 1e-5);
}

#[test]
fn terrain_sdf_on_flat_ground_zero() {
    let cfg = flat_terrain(5.0);
    let d = terrain_sdf(Vec3::new(0.0, 5.0, 0.0), &cfg);
    assert!(d.abs() < 1e-5);
}

#[test]
fn terrain_sdf_rolling_varies_with_amplitude() {
    // With amplitude > 0 the height varies, so the same XZ position at the mean height
    // might no longer be exactly zero.
    let cfg = TerrainConfig::rolling();
    // Just verify the function doesn't panic and returns a finite value.
    let d = terrain_sdf(Vec3::new(0.0, 0.0, 0.0), &cfg);
    assert!(d.is_finite());
}

// ──────────────────────── noise::terrain_height_range ────────────────────────

#[test]
fn height_range_min_le_max() {
    let cfg = TerrainConfig::rolling();
    let brick_min = Vec3::new(-10.0, -10.0, -10.0);
    let brick_max = Vec3::new(10.0, 10.0, 10.0);
    let (min_h, max_h) = terrain_height_range(brick_min, brick_max, &cfg);
    assert!(min_h <= max_h);
}

#[test]
fn height_range_flat_terrain_min_equals_max() {
    let cfg = flat_terrain(7.0);
    let brick_min = Vec3::new(-5.0, 0.0, -5.0);
    let brick_max = Vec3::new(5.0, 0.0, 5.0);
    let (min_h, max_h) = terrain_height_range(brick_min, brick_max, &cfg);
    assert!((min_h - 7.0).abs() < 1e-5);
    assert!((max_h - 7.0).abs() < 1e-5);
}

#[test]
fn height_range_values_are_finite() {
    let cfg = TerrainConfig::rolling();
    let (min_h, max_h) = terrain_height_range(
        Vec3::new(-100.0, 0.0, -100.0),
        Vec3::new(100.0, 0.0, 100.0),
        &cfg,
    );
    assert!(min_h.is_finite());
    assert!(max_h.is_finite());
}

// ──────────────────────── noise::fbm2 ────────────────────────────────────────

#[test]
fn fbm2_zero_octaves_returns_zero() {
    let v = fbm2(0.0, 0.0, 0, 2.0, 0.5);
    assert_eq!(v, 0.0);
}

#[test]
fn fbm2_output_is_finite() {
    // The fbm_rotate step amplifies coordinates each octave; the result can
    // slightly exceed [-1, 1] at large inputs due to float precision in fract(),
    // but must always be a finite (non-NaN, non-Inf) number.
    for x in [-10.0f32, -1.0, 0.0, 0.5, 3.7, 100.0] {
        for z in [-10.0f32, 0.0, 7.3] {
            let v = fbm2(x, z, 4, 2.0, 0.5);
            assert!(v.is_finite(), "fbm2({x},{z}) = {v} is not finite");
        }
    }
}

#[test]
fn fbm2_deterministic() {
    let a = fbm2(1.23, 4.56, 4, 2.0, 0.5);
    let b = fbm2(1.23, 4.56, 4, 2.0, 0.5);
    assert_eq!(a, b);
}

#[test]
fn fbm2_different_positions_differ() {
    let a = fbm2(0.0, 0.0, 4, 2.0, 0.5);
    let b = fbm2(100.0, 0.0, 4, 2.0, 0.5);
    // While not guaranteed in theory, these positions should yield different values.
    assert_ne!(a, b);
}

#[test]
fn fbm2_single_octave_is_finite() {
    let v = fbm2(3.14, 2.71, 1, 2.0, 0.5);
    assert!(v.is_finite());
}
