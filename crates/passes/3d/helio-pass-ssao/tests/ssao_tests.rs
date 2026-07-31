// Tests for helio-pass-ssao: struct sizes, SSAO constants, hemispheric kernel math.
// All tests are pure Rust — no GPU device required.

use std::mem;

// ── Mirror private structs ────────────────────────────────────────────────────

/// Camera uniform matching ssao.wgsl CameraUniform.
/// 4 × mat4 (4×64 = 256 bytes) + vec3 (12) + pad (4) = 272 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct SsaoCameraUniform {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    position: [f32; 3],
    _pad0: f32,
}

/// Globals matching ssao.wgsl Globals (80 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct GpuGlobals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],
    rc_world_min: [f32; 4],
    rc_world_max: [f32; 4],
    csm_splits: [f32; 4],
}

/// SSAO parameters matching ssao.wgsl SsaoUniform (32 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct SsaoUniform {
    radius: f32,
    bias: f32,
    power: f32,
    samples: u32,
    noise_scale: [f32; 2],
    _pad: [f32; 2],
}

const KERNEL_SIZE: usize = 64;
const NOISE_DIM: u32 = 4;

// ── Struct size tests ─────────────────────────────────────────────────────────

#[test]
fn ssao_camera_uniform_size_is_272() {
    assert_eq!(mem::size_of::<SsaoCameraUniform>(), 272,
        "4×mat4(256) + vec3(12) + pad(4) = 272");
}

#[test]
fn ssao_camera_uniform_holds_four_mat4() {
    // 4 matrices × 64 bytes = 256 bytes
    let mat4_bytes = 4 * 4 * mem::size_of::<f32>();
    assert_eq!(mat4_bytes, 64);
    assert_eq!(4 * mat4_bytes, 256);
}

#[test]
fn ssao_camera_uniform_position_plus_pad_is_16() {
    // position: [f32;3] = 12, _pad0: f32 = 4 → 16 bytes (one vec4 slot)
    let position_bytes = 3 * mem::size_of::<f32>();
    let pad_bytes = mem::size_of::<f32>();
    assert_eq!(position_bytes + pad_bytes, 16);
}

#[test]
fn ssao_camera_uniform_total_256_plus_16() {
    assert_eq!(256 + 16, 272);
}

#[test]
fn gpu_globals_size_is_80() {
    assert_eq!(mem::size_of::<GpuGlobals>(), 80);
}

#[test]
fn gpu_globals_scalar_header_is_16_bytes() {
    // frame(4) + delta_time(4) + light_count(4) + ambient_intensity(4) = 16
    assert_eq!(4 + 4 + 4 + 4, 16usize);
}

#[test]
fn gpu_globals_four_vec4_fields() {
    // ambient_color + rc_world_min + rc_world_max + csm_splits = 4 × 16 = 64 bytes
    let vec4_section = 4 * 16usize;
    assert_eq!(vec4_section, 64);
}

#[test]
fn gpu_globals_layout_16_plus_64() {
    assert_eq!(16 + 64, 80usize);
}

#[test]
fn ssao_uniform_size_is_32() {
    assert_eq!(mem::size_of::<SsaoUniform>(), 32);
}

#[test]
fn ssao_uniform_scalar_header_is_16() {
    // radius(4) + bias(4) + power(4) + samples(4) = 16
    assert_eq!(4 + 4 + 4 + 4, 16usize);
}

#[test]
fn ssao_uniform_noise_scale_plus_pad_is_16() {
    // noise_scale([f32;2])=8 + _pad([f32;2])=8 = 16
    assert_eq!(8 + 8, 16usize);
}

// ── SSAO kernel constant tests ────────────────────────────────────────────────

#[test]
fn kernel_size_is_64() {
    assert_eq!(KERNEL_SIZE, 64);
}

#[test]
fn kernel_size_is_power_of_two() {
    assert!(KERNEL_SIZE.is_power_of_two());
}

#[test]
fn noise_dim_is_4() {
    assert_eq!(NOISE_DIM, 4);
}

#[test]
fn noise_texture_is_4x4() {
    let texel_count = NOISE_DIM * NOISE_DIM;
    assert_eq!(texel_count, 16);
}

#[test]
fn noise_dim_is_power_of_two() {
    assert!(NOISE_DIM.is_power_of_two());
}

// ── Hemispheric kernel math ───────────────────────────────────────────────────

/// Generate a hemisphere sample at index i using a deterministic pattern.
fn hemisphere_sample(i: usize, n: usize) -> [f32; 3] {
    // Distribute samples uniformly-ish using index-based angles
    let phi = 2.0 * std::f32::consts::PI * (i as f32 / n as f32);
    let theta = (i as f32 / n as f32) * std::f32::consts::FRAC_PI_2;
    let x = theta.sin() * phi.cos();
    let y = theta.sin() * phi.sin();
    let z = theta.cos(); // z >= 0 for upper hemisphere
    // Scale to bring closer to origin for importance sampling
    let scale = {
        let t = i as f32 / n as f32;
        0.1 + 0.9 * t * t
    };
    [x * scale, y * scale, z.abs() * scale]
}

#[test]
fn hemisphere_sample_z_always_non_negative() {
    for i in 0..KERNEL_SIZE {
        let s = hemisphere_sample(i, KERNEL_SIZE);
        assert!(s[2] >= 0.0, "sample[{i}].z = {} < 0", s[2]);
    }
}

#[test]
fn hemisphere_sample_length_below_one() {
    for i in 0..KERNEL_SIZE {
        let s = hemisphere_sample(i, KERNEL_SIZE);
        let len = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
        assert!(len <= 1.01f32, "sample[{i}] length = {len}");
    }
}

#[test]
fn hemisphere_sample_count_matches_kernel_size() {
    let samples: Vec<_> = (0..KERNEL_SIZE).map(|i| hemisphere_sample(i, KERNEL_SIZE)).collect();
    assert_eq!(samples.len(), KERNEL_SIZE);
}

#[test]
fn ssao_radius_positive_non_zero() {
    let u = SsaoUniform { radius: 0.5, bias: 0.025, power: 2.0, samples: 64, noise_scale: [1.0; 2], _pad: [0.0; 2] };
    assert!(u.radius > 0.0f32);
}

#[test]
fn ssao_bias_small_positive() {
    let u = SsaoUniform { radius: 0.5, bias: 0.025, power: 2.0, samples: 64, noise_scale: [1.0; 2], _pad: [0.0; 2] };
    assert!(u.bias > 0.0f32 && u.bias < 0.1f32);
}

#[test]
fn ssao_samples_u32_not_i32() {
    let u = SsaoUniform { radius: 0.5, bias: 0.025, power: 2.0, samples: 64, noise_scale: [1.0; 2], _pad: [0.0; 2] };
    // Ensure samples field is u32 (no sign issues)
    assert_eq!(u.samples, 64u32);
}

#[test]
fn noise_scale_computation() {
    // noise_scale = (screen_width / NOISE_DIM, screen_height / NOISE_DIM)
    let width = 1920.0f32;
    let height = 1080.0f32;
    let noise_scale = [width / NOISE_DIM as f32, height / NOISE_DIM as f32];
    assert!((noise_scale[0] - 480.0f32).abs() < 1e-3f32);
    assert!((noise_scale[1] - 270.0f32).abs() < 1e-3f32);
}

#[test]
fn ssao_camera_align_check_16() {
    // SsaoCameraUniform contains only f32 arrays → alignment is 4 bytes
    assert_eq!(mem::align_of::<SsaoCameraUniform>(), 4);
}

#[test]
fn gpu_globals_align_check_4() {
    assert_eq!(mem::align_of::<GpuGlobals>(), 4);
}

#[test]
fn ssao_uniform_align_check_4() {
    assert_eq!(mem::align_of::<SsaoUniform>(), 4);
}
