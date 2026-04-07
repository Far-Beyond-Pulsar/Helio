// Tests for helio-pass-sky-lut: panoramic UV ↔ azimuth/elevation math.
// All tests are pure math — no GPU device required.

use std::f32::consts::{FRAC_PI_2, PI};

const LUT_WIDTH: u32 = 192;
const LUT_HEIGHT: u32 = 108;

// ── Panoramic mapping helpers (mirrors the LUT shader math) ──────────────────

/// Map a UV in [0,1]² to (azimuth, elevation) in radians.
/// azimuth ∈ [0, 2π), elevation ∈ [-π/2, π/2].
fn uv_to_azimuth_elevation(u: f32, v: f32) -> (f32, f32) {
    let azimuth = u * 2.0 * PI;
    let elevation = (v - 0.5) * PI;
    (azimuth, elevation)
}

/// Inverse: (azimuth, elevation) → UV
fn azimuth_elevation_to_uv(azimuth: f32, elevation: f32) -> (f32, f32) {
    let u = azimuth / (2.0 * PI);
    let v = elevation / PI + 0.5;
    (u, v)
}

/// UV to unit direction vector (right-hand, Y-up).
fn uv_to_direction(u: f32, v: f32) -> [f32; 3] {
    let (az, el) = uv_to_azimuth_elevation(u, v);
    let cos_el = el.cos();
    [cos_el * az.cos(), el.sin(), cos_el * az.sin()]
}

/// Pixel centre UV for (col, row) in a LUT_WIDTH × LUT_HEIGHT texture.
fn pixel_uv(col: u32, row: u32) -> (f32, f32) {
    let u = (col as f32 + 0.5) / LUT_WIDTH as f32;
    let v = (row as f32 + 0.5) / LUT_HEIGHT as f32;
    (u, v)
}

// ── UV range tests ────────────────────────────────────────────────────────────

#[test]
fn uv_corners_stay_in_range() {
    for &u in &[0.0f32, 0.5, 1.0] {
        for &v in &[0.0f32, 0.5, 1.0] {
            let (az, el) = uv_to_azimuth_elevation(u, v);
            assert!(az >= 0.0 && az <= 2.0 * PI + 1e-5, "az = {az}");
            assert!(
                el >= -FRAC_PI_2 - 1e-5 && el <= FRAC_PI_2 + 1e-5,
                "el = {el}"
            );
        }
    }
}

#[test]
fn uv_0_0_gives_azimuth_0_elevation_neg_90() {
    let (az, el) = uv_to_azimuth_elevation(0.0, 0.0);
    assert!((az - 0.0f32).abs() < 1e-6);
    assert!((el - (-FRAC_PI_2)).abs() < 1e-6, "el = {el}");
}

#[test]
fn uv_0_1_gives_azimuth_0_elevation_pos_90() {
    let (az, el) = uv_to_azimuth_elevation(0.0, 1.0);
    assert!((az - 0.0f32).abs() < 1e-6);
    assert!((el - FRAC_PI_2).abs() < 1e-6, "el = {el}");
}

#[test]
fn uv_midpoint_gives_horizon() {
    let (_az, el) = uv_to_azimuth_elevation(0.5, 0.5);
    assert!(el.abs() < 1e-6, "elevation at v=0.5 should be 0, got {el}");
}

#[test]
fn uv_1_0_gives_azimuth_2pi() {
    let (az, _el) = uv_to_azimuth_elevation(1.0, 0.0);
    assert!((az - 2.0 * PI).abs() < 1e-5, "az = {az}");
}

// ── Round-trip tests ──────────────────────────────────────────────────────────

#[test]
fn round_trip_uv_to_az_el_back_to_uv() {
    let cases = [(0.1, 0.2), (0.5, 0.5), (0.8, 0.9), (0.0, 0.5), (1.0, 0.5)];
    for (u_in, v_in) in cases {
        let (az, el) = uv_to_azimuth_elevation(u_in, v_in);
        let (u_out, v_out) = azimuth_elevation_to_uv(az, el);
        assert!((u_out - u_in).abs() < 1e-5, "u mismatch: {u_in} → {u_out}");
        assert!((v_out - v_in).abs() < 1e-5, "v mismatch: {v_in} → {v_out}");
    }
}

// ── Direction vector tests ────────────────────────────────────────────────────

#[test]
fn direction_is_unit_length_at_various_uvs() {
    let uvs = [
        (0.0, 0.5),
        (0.25, 0.5),
        (0.5, 0.5),
        (0.75, 0.5),
        (0.5, 0.0),
        (0.5, 1.0),
    ];
    for (u, v) in uvs {
        let d = uv_to_direction(u, v);
        let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        assert!((len - 1.0f32).abs() < 1e-5, "len = {len} at ({u},{v})");
    }
}

#[test]
fn zenith_direction_points_up() {
    // v=1 means elevation = +π/2, so direction = (0, 1, 0)
    let d = uv_to_direction(0.0, 1.0);
    assert!(d[1] > 0.999f32, "y = {}", d[1]);
}

#[test]
fn nadir_direction_points_down() {
    // v=0 means elevation = -π/2, so direction = (0, -1, 0)
    let d = uv_to_direction(0.0, 0.0);
    assert!(d[1] < -0.999f32, "y = {}", d[1]);
}

#[test]
fn horizon_direction_has_zero_y() {
    let d = uv_to_direction(0.0, 0.5);
    assert!(d[1].abs() < 1e-5, "y = {}", d[1]);
}

// ── Pixel UV tests ────────────────────────────────────────────────────────────

#[test]
fn first_pixel_uv_near_bottom_left() {
    let (u, v) = pixel_uv(0, 0);
    let expected_u = 0.5 / LUT_WIDTH as f32;
    let expected_v = 0.5 / LUT_HEIGHT as f32;
    assert!((u - expected_u).abs() < 1e-6);
    assert!((v - expected_v).abs() < 1e-6);
}

#[test]
fn last_pixel_uv_near_top_right() {
    let (u, v) = pixel_uv(LUT_WIDTH - 1, LUT_HEIGHT - 1);
    let expected_u = (LUT_WIDTH as f32 - 0.5) / LUT_WIDTH as f32;
    let expected_v = (LUT_HEIGHT as f32 - 0.5) / LUT_HEIGHT as f32;
    assert!((u - expected_u).abs() < 1e-6);
    assert!((v - expected_v).abs() < 1e-6);
}

#[test]
fn pixel_uvs_all_in_0_1_range() {
    for col in [0u32, LUT_WIDTH / 2, LUT_WIDTH - 1] {
        for row in [0u32, LUT_HEIGHT / 2, LUT_HEIGHT - 1] {
            let (u, v) = pixel_uv(col, row);
            assert!(u > 0.0 && u < 1.0, "u={u} for col={col}");
            assert!(v > 0.0 && v < 1.0, "v={v} for row={row}");
        }
    }
}

#[test]
fn middle_pixel_uv_near_0_5() {
    let mid_col = LUT_WIDTH / 2;
    let mid_row = LUT_HEIGHT / 2;
    let (u, v) = pixel_uv(mid_col, mid_row);
    // Should be close to 0.5 for even-sized textures
    assert!((u - 0.5f32).abs() < 1.0 / LUT_WIDTH as f32 + 1e-3);
    assert!((v - 0.5f32).abs() < 1.0 / LUT_HEIGHT as f32 + 1e-3);
}

// ── Azimuth step tests ────────────────────────────────────────────────────────

#[test]
fn azimuth_step_per_pixel() {
    let step = 2.0 * PI / LUT_WIDTH as f32;
    // Each column spans exactly step radians in azimuth
    assert!((step - 2.0 * PI / 192.0f32).abs() < 1e-6, "step = {step}");
}

#[test]
fn elevation_step_per_pixel() {
    let step = PI / LUT_HEIGHT as f32;
    assert!((step - PI / 108.0f32).abs() < 1e-6, "step = {step}");
}

#[test]
fn total_azimuth_coverage_is_2pi() {
    let total: f32 = (2.0 * PI / LUT_WIDTH as f32) * LUT_WIDTH as f32;
    assert!((total - 2.0 * PI).abs() < 1e-4, "total = {total}");
}

#[test]
fn total_elevation_coverage_is_pi() {
    let total: f32 = (PI / LUT_HEIGHT as f32) * LUT_HEIGHT as f32;
    assert!((total - PI).abs() < 1e-4, "total = {total}");
}

// ── Solid angle / area tests ──────────────────────────────────────────────────

#[test]
fn uv_v_half_maps_to_horizon() {
    for col in 0..LUT_WIDTH {
        let (_, v) = pixel_uv(col, LUT_HEIGHT / 2);
        let (_az, el) = uv_to_azimuth_elevation(0.0, v);
        // Should be near the horizon (small elevation)
        assert!(el.abs() < 0.05f32, "el={el} for row={}", LUT_HEIGHT / 2);
    }
}

#[test]
fn sun_direction_normalization_check() {
    let d = [0.0f32, 0.9, 0.4];
    let len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let normalized = [d[0] / len, d[1] / len, d[2] / len];
    let check_len = (normalized[0] * normalized[0]
        + normalized[1] * normalized[1]
        + normalized[2] * normalized[2])
        .sqrt();
    assert!((check_len - 1.0f32).abs() < 1e-6);
}

#[test]
fn opposite_azimuths_produce_mirrored_x_z() {
    let d1 = uv_to_direction(0.0, 0.5); // azimuth = 0
    let d2 = uv_to_direction(0.5, 0.5); // azimuth = π
                                        // x components should have opposite signs
    assert!(d1[0] > 0.0, "d1.x = {}", d1[0]);
    assert!(d2[0] < 0.0, "d2.x = {}", d2[0]);
}
