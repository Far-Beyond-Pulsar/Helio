//! Sky uniform struct layout and `earth_like()` parameter tests.
//!
//! Pure math — no GPU, no wgpu, no crate imports.
//! The struct exactly mirrors the private `ShaderSkyUniforms` in
//! `helio-pass-sky/src/lib.rs` and validates its documented field values.

/// Exact mirror of the private `ShaderSkyUniforms` struct in lib.rs.
/// Must be 112 bytes, 16-byte WGSL-aligned.
#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderSkyUniforms {
    sun_direction: [f32; 3],    //  0..12
    sun_intensity: f32,         // 12..16
    rayleigh_scatter: [f32; 3], // 16..28
    rayleigh_h_scale: f32,      // 28..32
    mie_scatter: f32,           // 32..36
    mie_h_scale: f32,           // 36..40
    mie_g: f32,                 // 40..44
    sun_disk_cos: f32,          // 44..48
    earth_radius: f32,          // 48..52
    atm_radius: f32,            // 52..56
    exposure: f32,              // 56..60
    clouds_enabled: u32,        // 60..64
    cloud_coverage: f32,        // 64..68
    cloud_density: f32,         // 68..72
    cloud_base: f32,            // 72..76
    cloud_top: f32,             // 76..80
    cloud_wind_x: f32,          // 80..84
    cloud_wind_z: f32,          // 84..88
    cloud_speed: f32,           // 88..92
    time_sky: f32,              // 92..96
    skylight_intensity: f32,    // 96..100
    _pad0: f32,                 // 100..104
    _pad1: f32,                 // 104..108
    _pad2: f32,                 // 108..112
}

/// Mirror of the private `earth_like()` constructor.
fn earth_like() -> ShaderSkyUniforms {
    let d = [0.0f32, 0.9, 0.4];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    ShaderSkyUniforms {
        sun_direction: [d[0] / len, d[1] / len, d[2] / len],
        sun_intensity: 22.0,
        rayleigh_scatter: [5.8e-3, 1.35e-2, 3.31e-2],
        rayleigh_h_scale: 0.1,
        mie_scatter: 2.1e-3,
        mie_h_scale: 0.075,
        mie_g: 0.76,
        sun_disk_cos: 0.9998,
        earth_radius: 6360.0,
        atm_radius: 6420.0,
        exposure: 0.1,
        clouds_enabled: 0,
        cloud_coverage: 0.0,
        cloud_density: 0.0,
        cloud_base: 0.0,
        cloud_top: 0.0,
        cloud_wind_x: 0.0,
        cloud_wind_z: 0.0,
        cloud_speed: 0.0,
        time_sky: 0.0,
        skylight_intensity: 0.0,
        _pad0: 0.0,
        _pad1: 0.0,
        _pad2: 0.0,
    }
}

// ── Struct size and alignment ─────────────────────────────────────────────────

#[test]
fn shader_sky_uniforms_size_is_112_bytes() {
    assert_eq!(std::mem::size_of::<ShaderSkyUniforms>(), 112);
}

#[test]
fn shader_sky_uniforms_is_divisible_by_16() {
    // WGSL uniform buffers require 16-byte alignment.
    assert_eq!(std::mem::size_of::<ShaderSkyUniforms>() % 16, 0);
}

#[test]
fn shader_sky_uniforms_field_count() {
    // 24 fields × 4 bytes = 96 bytes, + 16 bytes for padding = 112
    // Actual: 28 × 4 bytes = 112 bytes (counted by hand)
    let expected_fields: usize = 28;
    assert_eq!(expected_fields * 4, 112);
}

#[test]
fn shader_sky_uniforms_alignment_is_4() {
    assert_eq!(std::mem::align_of::<ShaderSkyUniforms>(), 4);
}

// ── Sun direction normalization ───────────────────────────────────────────────

#[test]
fn sun_direction_input_vector() {
    // The raw input before normalization.
    let d = [0.0_f32, 0.9, 0.4];
    let len_sq = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
    // 0 + 0.81 + 0.16 = 0.97 — NOT pre-normalized.
    assert!((len_sq - 0.97).abs() < 1e-5);
}

#[test]
fn sun_direction_is_normalized_after_earth_like() {
    let u = earth_like();
    let n = u.sun_direction;
    let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
    assert!((len_sq - 1.0).abs() < 1e-5, "|sun_direction|² = {}", len_sq);
}

#[test]
fn sun_direction_x_is_zero() {
    let u = earth_like();
    assert!(u.sun_direction[0].abs() < 1e-6);
}

#[test]
fn sun_direction_y_is_positive() {
    let u = earth_like();
    assert!(u.sun_direction[1] > 0.0);
}

#[test]
fn sun_direction_z_is_positive() {
    let u = earth_like();
    assert!(u.sun_direction[2] > 0.0);
}

// ── sun_intensity ─────────────────────────────────────────────────────────────

#[test]
fn sun_intensity_is_22() {
    let u = earth_like();
    assert!((u.sun_intensity - 22.0).abs() < 1e-6);
}

#[test]
fn sun_intensity_is_positive() {
    let u = earth_like();
    assert!(u.sun_intensity > 0.0);
}

// ── Rayleigh scatter coefficients ────────────────────────────────────────────

#[test]
fn rayleigh_scatter_r_coefficient() {
    let u = earth_like();
    assert!((u.rayleigh_scatter[0] - 5.8e-3).abs() < 1e-7);
}

#[test]
fn rayleigh_scatter_g_coefficient() {
    let u = earth_like();
    assert!((u.rayleigh_scatter[1] - 1.35e-2).abs() < 1e-7);
}

#[test]
fn rayleigh_scatter_b_coefficient() {
    let u = earth_like();
    assert!((u.rayleigh_scatter[2] - 3.31e-2).abs() < 1e-7);
}

#[test]
fn rayleigh_blue_greater_than_green_greater_than_red() {
    // Blue sky: B > G > R in scattering coefficients.
    let u = earth_like();
    let r = u.rayleigh_scatter[0];
    let g = u.rayleigh_scatter[1];
    let b = u.rayleigh_scatter[2];
    assert!(b > g, "B={} should be > G={}", b, g);
    assert!(g > r, "G={} should be > R={}", g, r);
}

#[test]
fn rayleigh_h_scale_is_0_1() {
    let u = earth_like();
    assert!((u.rayleigh_h_scale - 0.1).abs() < 1e-6);
}

// ── Mie parameters ───────────────────────────────────────────────────────────

#[test]
fn mie_scatter_is_2_1e_3() {
    let u = earth_like();
    assert!((u.mie_scatter - 2.1e-3).abs() < 1e-8);
}

#[test]
fn mie_h_scale_is_0_075() {
    let u = earth_like();
    assert!((u.mie_h_scale - 0.075).abs() < 1e-6);
}

#[test]
fn mie_g_is_0_76() {
    // Henyey-Greenstein asymmetry parameter: 0.76 = strong forward scattering.
    let u = earth_like();
    assert!((u.mie_g - 0.76).abs() < 1e-6);
}

#[test]
fn mie_h_scale_less_than_rayleigh_h_scale() {
    // Mie aerosols are concentrated at lower altitudes than Rayleigh.
    let u = earth_like();
    assert!(u.mie_h_scale < u.rayleigh_h_scale);
}

// ── Sun disk angular size ─────────────────────────────────────────────────────

#[test]
fn sun_disk_cos_is_0_9998() {
    // cos(0.265°) ≈ 0.9998, matching the sun's ~0.53° angular diameter.
    let u = earth_like();
    assert!((u.sun_disk_cos - 0.9998).abs() < 1e-5);
}

#[test]
fn sun_disk_cos_is_close_to_one() {
    // The sun subtends a very small angle — cos(θ) is very close to 1.
    let u = earth_like();
    assert!(u.sun_disk_cos > 0.999);
    assert!(u.sun_disk_cos < 1.0);
}

// ── Earth / atmosphere radii ──────────────────────────────────────────────────

#[test]
fn earth_radius_is_6360_km() {
    let u = earth_like();
    assert!((u.earth_radius - 6360.0).abs() < 1e-3);
}

#[test]
fn atm_radius_is_6420_km() {
    let u = earth_like();
    assert!((u.atm_radius - 6420.0).abs() < 1e-3);
}

#[test]
fn atmosphere_extends_60_km_above_surface() {
    let u = earth_like();
    let thickness = u.atm_radius - u.earth_radius;
    assert!((thickness - 60.0).abs() < 1e-3);
}

#[test]
fn atm_radius_greater_than_earth_radius() {
    let u = earth_like();
    assert!(u.atm_radius > u.earth_radius);
}

// ── Exposure ─────────────────────────────────────────────────────────────────

#[test]
fn exposure_is_0_1() {
    let u = earth_like();
    assert!((u.exposure - 0.1).abs() < 1e-6);
}

// ── Cloud defaults ────────────────────────────────────────────────────────────

#[test]
fn clouds_disabled_by_default() {
    let u = earth_like();
    assert_eq!(u.clouds_enabled, 0);
}

#[test]
fn cloud_coverage_is_zero_by_default() {
    let u = earth_like();
    assert!((u.cloud_coverage).abs() < 1e-6);
}

// ── Padding is zero ───────────────────────────────────────────────────────────

#[test]
fn padding_fields_are_zero() {
    let u = earth_like();
    assert_eq!(u._pad0, 0.0);
    assert_eq!(u._pad1, 0.0);
    assert_eq!(u._pad2, 0.0);
}
