// Tests for helio-pass-sky-lut: LUT dimensions, uniform layout, atmospheric constants.
// All tests are pure Rust / math — no GPU device is required.

use std::mem;

// ── Mirror private structs ────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct ShaderSkyUniforms {
    sun_direction: [f32; 3],
    sun_intensity: f32,
    rayleigh_scatter: [f32; 3],
    rayleigh_h_scale: f32,
    mie_scatter: f32,
    mie_h_scale: f32,
    mie_g: f32,
    sun_disk_cos: f32,
    earth_radius: f32,
    atm_radius: f32,
    exposure: f32,
    clouds_enabled: u32,
    cloud_coverage: f32,
    cloud_density: f32,
    cloud_base: f32,
    cloud_top: f32,
    cloud_wind_x: f32,
    cloud_wind_z: f32,
    cloud_speed: f32,
    time_sky: f32,
    skylight_intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const LUT_WIDTH: u32 = 192;
const LUT_HEIGHT: u32 = 108;

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

// ── LUT dimension tests ───────────────────────────────────────────────────────

#[test]
fn lut_width_is_192() {
    assert_eq!(LUT_WIDTH, 192);
}

#[test]
fn lut_height_is_108() {
    assert_eq!(LUT_HEIGHT, 108);
}

#[test]
fn lut_pixel_count() {
    assert_eq!(LUT_WIDTH * LUT_HEIGHT, 20736);
}

#[test]
fn lut_aspect_ratio_approx_16_over_9() {
    let ratio = LUT_WIDTH as f32 / LUT_HEIGHT as f32;
    assert!(
        (ratio - 16.0f32 / 9.0f32).abs() < 1e-4f32,
        "ratio = {ratio}"
    );
}

#[test]
fn lut_width_divisible_by_4() {
    assert_eq!(LUT_WIDTH % 4, 0);
}

#[test]
fn lut_height_divisible_by_4() {
    assert_eq!(LUT_HEIGHT % 4, 0);
}

#[test]
fn lut_width_greater_than_height() {
    assert!(LUT_WIDTH > LUT_HEIGHT);
}

// ── Uniform layout tests ──────────────────────────────────────────────────────

#[test]
fn shader_sky_uniforms_size_is_112() {
    assert_eq!(mem::size_of::<ShaderSkyUniforms>(), 112);
}

#[test]
fn shader_sky_uniforms_size_divisible_by_16() {
    assert_eq!(
        mem::size_of::<ShaderSkyUniforms>() % 16,
        0,
        "Uniforms must be 16-byte aligned for WGSL"
    );
}

#[test]
fn shader_sky_uniforms_field_count_times_4_eq_size() {
    // 28 fields × 4 bytes = 112
    let expected = 28usize * 4;
    assert_eq!(mem::size_of::<ShaderSkyUniforms>(), expected);
}

// ── earth_like() factory tests ────────────────────────────────────────────────

#[test]
fn earth_like_sun_direction_is_unit_vector() {
    let u = earth_like();
    let len_sq = u.sun_direction[0] * u.sun_direction[0]
        + u.sun_direction[1] * u.sun_direction[1]
        + u.sun_direction[2] * u.sun_direction[2];
    assert!((len_sq - 1.0f32).abs() < 1e-5f32, "len_sq = {len_sq}");
}

#[test]
fn earth_like_sun_direction_y_positive() {
    let u = earth_like();
    assert!(u.sun_direction[1] > 0.0f32);
}

#[test]
fn earth_like_sun_intensity_is_22() {
    let u = earth_like();
    assert!((u.sun_intensity - 22.0f32).abs() < 1e-6f32);
}

#[test]
fn earth_like_earth_radius_6360() {
    let u = earth_like();
    assert!((u.earth_radius - 6360.0f32).abs() < 1e-3f32);
}

#[test]
fn earth_like_atm_radius_6420() {
    let u = earth_like();
    assert!((u.atm_radius - 6420.0f32).abs() < 1e-3f32);
}

#[test]
fn earth_like_atm_radius_greater_than_earth_radius() {
    let u = earth_like();
    assert!(u.atm_radius > u.earth_radius);
}

#[test]
fn earth_like_atm_thickness_is_60() {
    let u = earth_like();
    assert!((u.atm_radius - u.earth_radius - 60.0f32).abs() < 1e-3f32);
}

#[test]
fn earth_like_rayleigh_blue_greater_than_green_greater_than_red() {
    let u = earth_like();
    // Rayleigh scatter inversely proportional to λ^4 → blue > green > red
    assert!(u.rayleigh_scatter[2] > u.rayleigh_scatter[1]);
    assert!(u.rayleigh_scatter[1] > u.rayleigh_scatter[0]);
}

#[test]
fn earth_like_rayleigh_red_is_5_8e_3() {
    let u = earth_like();
    assert!((u.rayleigh_scatter[0] - 5.8e-3f32).abs() < 1e-6f32);
}

#[test]
fn earth_like_mie_g_in_range() {
    let u = earth_like();
    assert!(u.mie_g > 0.0f32 && u.mie_g < 1.0f32, "mie_g = {}", u.mie_g);
}

#[test]
fn earth_like_mie_g_forward_scattering() {
    let u = earth_like();
    // Positive g means forward-scattering Mie phase
    assert!(u.mie_g > 0.5f32);
}

#[test]
fn earth_like_exposure_positive() {
    let u = earth_like();
    assert!(u.exposure > 0.0f32);
}

#[test]
fn earth_like_clouds_disabled_by_default() {
    let u = earth_like();
    assert_eq!(u.clouds_enabled, 0u32);
}

#[test]
fn earth_like_pads_are_zero() {
    let u = earth_like();
    assert_eq!(u._pad0, 0.0f32);
    assert_eq!(u._pad1, 0.0f32);
    assert_eq!(u._pad2, 0.0f32);
}

#[test]
fn earth_like_sun_disk_cos_near_one() {
    let u = earth_like();
    assert!(u.sun_disk_cos > 0.999f32 && u.sun_disk_cos <= 1.0f32);
}

#[test]
fn earth_like_cloud_params_zero() {
    let u = earth_like();
    assert_eq!(u.cloud_coverage, 0.0f32);
    assert_eq!(u.cloud_density, 0.0f32);
    assert_eq!(u.cloud_base, 0.0f32);
    assert_eq!(u.cloud_top, 0.0f32);
}

#[test]
fn earth_like_mie_h_scale_smaller_than_rayleigh_h_scale() {
    let u = earth_like();
    // Mie scattering concentrates in lower troposphere → smaller scale height
    assert!(u.mie_h_scale < u.rayleigh_h_scale);
}

#[test]
fn earth_like_mie_scatter_smaller_than_rayleigh_red() {
    let u = earth_like();
    assert!(u.mie_scatter < u.rayleigh_scatter[0]);
}

#[test]
fn earth_like_rayleigh_blue_red_ratio_exceeds_five() {
    let u = earth_like();
    let ratio = u.rayleigh_scatter[2] / u.rayleigh_scatter[0];
    assert!(ratio > 5.0f32, "ratio = {ratio}");
}
