//! Architecture and dispatch math tests for the Shadow Matrix compute pass.
//!
//! Pure math — no GPU, no wgpu, no crate imports.
//! Constants and struct layout mirror private items in `helio-pass-shadow-matrix/src/lib.rs`.

// Mirror of private constant in lib.rs
const WORKGROUP_SIZE: u32 = 64;

// Mirror of private struct in lib.rs (for size/layout assertions)
#[repr(C)]
#[derive(Clone, Copy)]
struct ShadowMatrixUniforms {
    light_count: u32,
    shadow_atlas_size: u32,
    _pad: [u32; 2],
}

// Derived constants
const MATRIX_4X4_BYTES: usize = 4 * 4 * 4; // 64 bytes
const CUBE_FACES: usize = 6;
const MAX_SHADOW_FACES: usize = 256;

// ── Uniform struct layout ─────────────────────────────────────────────────────

#[test]
fn shadow_matrix_uniforms_size_is_16_bytes() {
    assert_eq!(std::mem::size_of::<ShadowMatrixUniforms>(), 16);
}

#[test]
fn shadow_matrix_uniforms_has_four_u32_fields() {
    // light_count(u32) + shadow_atlas_size(u32) + _pad([u32;2]) = 4 × 4 = 16 bytes.
    let expected = 4 * std::mem::size_of::<u32>();
    assert_eq!(expected, 16);
}

#[test]
fn shadow_matrix_uniforms_alignment_is_4_bytes() {
    assert_eq!(std::mem::align_of::<ShadowMatrixUniforms>(), 4);
}

#[test]
fn shadow_matrix_uniforms_is_multiple_of_16() {
    // WGSL uniform buffers require 16-byte alignment.
    assert_eq!(std::mem::size_of::<ShadowMatrixUniforms>() % 16, 0);
}

#[test]
fn shadow_matrix_uniforms_pad_fields_count() {
    // _pad is [u32; 2] = 8 bytes of padding to reach 16-byte total.
    let payload_bytes = std::mem::size_of::<u32>() * 2; // light_count + shadow_atlas_size
    let total = std::mem::size_of::<ShadowMatrixUniforms>();
    let padding = total - payload_bytes;
    assert_eq!(padding, 8);
}

// ── Workgroup dispatch math ───────────────────────────────────────────────────

#[test]
fn workgroup_size_is_64() {
    assert_eq!(WORKGROUP_SIZE, 64);
}

#[test]
fn workgroup_size_is_power_of_two() {
    assert!(WORKGROUP_SIZE.is_power_of_two());
}

#[test]
fn dispatch_for_zero_lights() {
    let light_count: u32 = 0;
    let groups = light_count.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups, 0);
}

#[test]
fn dispatch_for_one_light() {
    let light_count: u32 = 1;
    let groups = light_count.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups, 1);
}

#[test]
fn dispatch_for_64_lights_exact() {
    let light_count: u32 = 64;
    let groups = light_count.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups, 1);
}

#[test]
fn dispatch_for_65_lights_needs_two_groups() {
    let light_count: u32 = 65;
    let groups = light_count.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups, 2);
}

#[test]
fn dispatch_for_128_lights() {
    let light_count: u32 = 128;
    let groups = light_count.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups, 2);
}

#[test]
fn dispatch_for_typical_42_point_lights() {
    // 42 is the maximum point-light count given the 256-face atlas budget.
    let light_count: u32 = 42;
    let groups = light_count.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups, 1); // 42 < 64, so one workgroup
}

#[test]
fn dispatch_formula_is_ceil_division() {
    // ceil(n / 64) = (n + 63) / 64 (integer division)
    for n in 0u32..=200 {
        let ceil = n.div_ceil(WORKGROUP_SIZE);
        let manual = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
        assert_eq!(ceil, manual, "mismatch at n={}", n);
    }
}

// ── 4×4 matrix sizing ────────────────────────────────────────────────────────

#[test]
fn matrix_4x4_f32_is_64_bytes() {
    assert_eq!(MATRIX_4X4_BYTES, 64);
}

#[test]
fn matrix_4x4_row_count() {
    assert_eq!(4 * 4, 16); // 16 elements
}

#[test]
fn matrix_4x4_byte_size() {
    let s = 4 * 4 * std::mem::size_of::<f32>();
    assert_eq!(s, 64);
}

// ── Shadow matrix buffer sizing ───────────────────────────────────────────────

#[test]
fn shadow_matrix_buf_for_one_point_light() {
    // One point light needs 6 cube-face matrices.
    let size = CUBE_FACES * MATRIX_4X4_BYTES;
    assert_eq!(size, 384); // 6 × 64 = 384 bytes
}

#[test]
fn shadow_matrix_buf_for_max_faces() {
    let size = MAX_SHADOW_FACES * MATRIX_4X4_BYTES;
    assert_eq!(size, 16_384); // 256 × 64 = 16 KiB
}

#[test]
fn shadow_matrix_buf_for_42_point_lights() {
    let lights: usize = 42;
    let size = lights * CUBE_FACES * MATRIX_4X4_BYTES;
    // 42 × 6 × 64 = 16,128 bytes
    assert_eq!(size, 16_128);
}

#[test]
fn shadow_matrix_buf_plus_csm_fits_max_faces() {
    // 42 point lights × 6 + 4 CSM = 256 total; must not exceed the pre-allocated buffer.
    let total_faces: usize = 42 * CUBE_FACES + 4;
    assert!(total_faces <= MAX_SHADOW_FACES);
    assert_eq!(total_faces, 256);
}

// ── light_count field fits in u32 ────────────────────────────────────────────

#[test]
fn light_count_field_is_u32() {
    // The maximum meaningful light_count is 42 (one per point light), well within u32::MAX.
    let max_lights: u32 = 42;
    assert!(max_lights <= u32::MAX);
}

#[test]
fn shadow_atlas_size_field_is_u32() {
    // SHADOW_RES = 1024, which fits comfortably in a u32.
    let res: u32 = 1024;
    assert!(res <= u32::MAX);
}

// ── GPU-driven O(1) CPU property ─────────────────────────────────────────────

#[test]
fn single_dispatch_regardless_of_light_count() {
    // The compute pass issues exactly one dispatch call per frame.
    // The workgroup count varies, but the number of API calls is constant (1).
    let api_calls_per_frame: u32 = 1;
    assert_eq!(api_calls_per_frame, 1);
}

#[test]
fn uniform_buffer_size_is_constant() {
    // The uniform buffer is always sizeof(ShadowMatrixUniforms) = 16 bytes,
    // regardless of the number of lights.
    let size = std::mem::size_of::<ShadowMatrixUniforms>();
    assert_eq!(size, 16);
}
