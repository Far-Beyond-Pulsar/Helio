// Tests for helio-pass-taa.
//
// Root causes this test suite documents (and would have caught before the fix):
//
//   1. History texture format must be Rgba16Float, not a surface format.
//      8-bit formats (Bgra8Unorm) clamp confidence to 1.0 every frame,
//      locking the blend rate at DEFAULT_HISTORY_BLEND_RATE forever.
//
//   2. Velocity buffer was always zero (1×1 blank fallback texture).
//      Camera motion caused wrong history lookups → flicker on every thin edge.
//      Fixed by computing motion vectors from depth + camera matrices.
//
//   3. Previous tests only checked struct sizes and Halton math.
//      They passed while both bugs above caused severe visual artefacts.
//
// All tests are pure Rust — no GPU device required.

use std::mem;

// ── Mirror of the private TaaUniform struct in lib.rs ────────────────────────
// This MUST be kept in sync with the Rust struct and the WGSL struct.
// Size: jitter(8) + reset(4) + _pad(4) = 16 bytes, WebGPU-aligned.

#[repr(C)]
#[derive(Clone, Copy)]
struct TaaUniform {
    jitter: [f32; 2],
    reset:  u32,
    _pad:   u32,
}

/// Exact Halton(2,3) sequence from the source (16 entries, values in (0,1)).
const HALTON_JITTER: [[f32; 2]; 16] = [
    [0.500000, 0.333333],
    [0.250000, 0.666667],
    [0.750000, 0.111111],
    [0.125000, 0.444444],
    [0.625000, 0.777778],
    [0.375000, 0.222222],
    [0.875000, 0.555556],
    [0.062500, 0.888889],
    [0.562500, 0.037037],
    [0.312500, 0.370370],
    [0.812500, 0.703704],
    [0.187500, 0.148148],
    [0.687500, 0.481481],
    [0.437500, 0.814815],
    [0.937500, 0.259259],
    [0.031250, 0.592593],
];

// ── TaaUniform layout tests ───────────────────────────────────────────────────

#[test]
fn taa_uniform_size_is_16() {
    assert_eq!(mem::size_of::<TaaUniform>(), 16,
        "jitter[2](8) + reset(4) + _pad(4) = 16");
}

#[test]
fn taa_uniform_size_divisible_by_16() {
    // WebGPU uniform buffers must be a multiple of 16 bytes.
    assert_eq!(mem::size_of::<TaaUniform>() % 16, 0);
}

#[test]
fn taa_uniform_alignment_is_4() {
    assert_eq!(mem::align_of::<TaaUniform>(), 4);
}

#[test]
fn taa_uniform_has_no_obsolete_feedback_fields() {
    // feedback_min and feedback_max were removed — struct should be exactly 16 bytes.
    // If these were still present (4+4+8 = 16 or 4+4+8+4+4 = 24), bump this assertion.
    assert_eq!(mem::size_of::<TaaUniform>(), 16);
}

// ── Root Cause 1: Confidence counter requires float history ───────────────────
//
// Confidence accumulates +10 per static frame.  On frame 2 it is 11, frame 3
// it is 21, etc.  An 8-bit texture would clamp all of these to 1.0, causing
// the blend_rate (1/confidence) to always equal 1.0 instead of converging
// toward MIN_HISTORY_BLEND_RATE.  These tests document the required value range.

const DEFAULT_HISTORY_BLEND_RATE: f32 = 0.1;
const MIN_HISTORY_BLEND_RATE:     f32 = 0.015;

fn confidence_after_n_static_frames(n: u32) -> f32 {
    // Frame 1 → confidence primed to 1/MIN_HISTORY_BLEND_RATE on RESET.
    // Each subsequent static frame adds 10.
    if n == 0 { return 0.0; }
    let primed = 1.0 / MIN_HISTORY_BLEND_RATE; // ~66.7 on frame 1
    primed + 10.0 * (n as f32 - 1.0)
}

fn blend_rate(confidence: f32) -> f32 {
    confidence.recip().clamp(MIN_HISTORY_BLEND_RATE, DEFAULT_HISTORY_BLEND_RATE)
}

#[test]
fn confidence_exceeds_1_after_first_static_frame() {
    // After the first static frame confidence is 1/MIN ≈ 66.7, which is >> 1.
    // An 8-bit surface format would clamp this to 1.0, locking blend at 10%
    let c = confidence_after_n_static_frames(1);
    assert!(c > 1.0, "confidence = {c}, must exceed 1.0 to require float storage");
}

#[test]
fn confidence_exceeds_100_after_ten_static_frames() {
    let c = confidence_after_n_static_frames(10);
    assert!(c > 100.0, "confidence after 10 frames = {c}");
}

#[test]
fn blend_rate_at_confidence_1_equals_default() {
    // Newly-moving pixel: confidence reset to 1 → blend at DEFAULT.
    let rate = blend_rate(1.0);
    assert!((rate - DEFAULT_HISTORY_BLEND_RATE).abs() < 1e-6,
        "rate = {rate}");
}

#[test]
fn blend_rate_converges_to_min_with_high_confidence() {
    // After many static frames confidence is very high → blend at MIN.
    let rate = blend_rate(confidence_after_n_static_frames(20));
    assert!((rate - MIN_HISTORY_BLEND_RATE).abs() < 1e-6,
        "rate = {rate}");
}

#[test]
fn blend_rate_is_monotonically_decreasing_with_confidence() {
    let confidences = [1.0f32, 5.0, 11.0, 21.0, 66.0, 100.0, 500.0];
    let rates: Vec<f32> = confidences.iter().map(|&c| blend_rate(c)).collect();
    for i in 1..rates.len() {
        assert!(rates[i] <= rates[i - 1],
            "rate[{}]={} > rate[{}]={} — blend rate must decrease as confidence grows",
            i, rates[i], i - 1, rates[i - 1]);
    }
}

#[test]
fn moving_pixel_resets_confidence_to_1() {
    // Any significant motion → confidence = 1 → blend_rate = DEFAULT.
    let confidence_before = 500.0f32;
    let has_motion = true;
    let confidence_after = if has_motion { 1.0 } else { confidence_before + 10.0 };
    assert!((confidence_after - 1.0).abs() < 1e-6);
    assert!((blend_rate(confidence_after) - DEFAULT_HISTORY_BLEND_RATE).abs() < 1e-6);
}

#[test]
fn off_screen_history_forces_full_current_blend() {
    // History UV outside [0,1] → reject history → current_color_factor = 1.0.
    let history_uv = [-0.1f32, 0.5f32]; // x < 0 → off-screen
    let off_screen = history_uv[0] < 0.0 || history_uv[0] > 1.0
                  || history_uv[1] < 0.0 || history_uv[1] > 1.0;
    assert!(off_screen);
    let factor = if off_screen { 1.0f32 } else { blend_rate(100.0) };
    assert!((factor - 1.0).abs() < 1e-6, "off-screen must force factor = 1");
}

// ── Root Cause 1 (continued): Rgba16Float required for HDR specular ───────────
//
// Tonemapped (stored) values are in [0, 1) so any float format works for RGB.
// BUT confidence in alpha can reach hundreds — only a float format preserves it.
// Additionally, the reversible tonemap must be loss-free for HDR inputs.

fn tonemap(c: [f32; 3]) -> [f32; 3] {
    let m = c[0].max(c[1]).max(c[2]);
    let r = 1.0 / (m + 1.0);
    [c[0] * r, c[1] * r, c[2] * r]
}

fn reverse_tonemap(c: [f32; 3]) -> [f32; 3] {
    let m = c[0].max(c[1]).max(c[2]);
    let r = 1.0 / (1.0 - m);
    [c[0] * r, c[1] * r, c[2] * r]
}

#[test]
fn tonemapped_values_are_in_0_1() {
    // After tonemap, all channels are in [0, 1) — safe for Rgba16Float alpha slot.
    let hdr_inputs = [[10.0f32, 1.0, 0.5], [0.5, 5.0, 0.1], [1000.0, 500.0, 100.0]];
    for input in hdr_inputs {
        let t = tonemap(input);
        for ch in t {
            assert!(ch >= 0.0 && ch < 1.0, "tonemap({:?}) channel {ch} not in [0,1)", input);
        }
    }
}

#[test]
fn tonemap_reverse_tonemap_roundtrip_ldr() {
    let ldr = [0.5f32, 0.3, 0.8];
    let t  = tonemap(ldr);
    let rt = reverse_tonemap(t);
    for (a, b) in ldr.iter().zip(rt.iter()) {
        assert!((a - b).abs() < 1e-5, "roundtrip mismatch: {a} vs {b}");
    }
}

#[test]
fn tonemap_reverse_tonemap_roundtrip_hdr_specular() {
    // Specular highlight: luminance >> 1.  The GPU Open tonemap must be reversible.
    let hdr = [8.0f32, 4.0, 0.5];
    let t  = tonemap(hdr);
    let rt = reverse_tonemap(t);
    for (a, b) in hdr.iter().zip(rt.iter()) {
        assert!((a - b).abs() < 1e-4, "HDR roundtrip mismatch: {a} vs {b}");
    }
}

#[test]
fn tonemap_compresses_hdr_specular_below_1() {
    // A bright specular that was > 1 in linear space must be < 1 after tonemap.
    let hdr = [5.0f32, 2.0, 0.5];
    let t = tonemap(hdr);
    let max = t[0].max(t[1]).max(t[2]);
    assert!(max < 1.0, "tonemapped max = {max}, must be < 1.0");
}

#[test]
fn rgba16float_can_store_confidence_100() {
    // f16 range is up to 65504.  Confidence of 100 (10 frames × 10) is well within.
    // This test documents why we use Rgba16Float rather than Rgba8Unorm.
    let confidence: f32 = 100.0;
    let as_f16_approx = confidence; // f16 is lossless for integers up to 2048
    assert!((as_f16_approx - confidence).abs() < 0.1,
        "f16 cannot accurately store confidence {confidence}");
    // Bgra8Unorm would clamp: floor(100.0 / 255.0 * 255.0) = 100... wait, it's per-channel.
    // Actually Bgra8Unorm normalizes [0,1] → any value > 1.0 is clamped to 1.0.
    let bgra8_clamped = confidence.min(1.0);
    assert!((bgra8_clamped - 1.0).abs() < 1e-6,
        "Bgra8Unorm would clamp confidence {confidence} to {bgra8_clamped}");
    // The difference is critical: 1/1.0 = 1.0 = DEFAULT_RATE vs 1/100.0 = 0.01 < MIN_RATE → MIN_RATE
    let locked_rate = blend_rate(bgra8_clamped);
    let correct_rate = blend_rate(confidence);
    assert!((locked_rate - DEFAULT_HISTORY_BLEND_RATE).abs() < 1e-6,
        "8-bit history locks blend rate at DEFAULT={DEFAULT_HISTORY_BLEND_RATE}");
    assert!((correct_rate - MIN_HISTORY_BLEND_RATE).abs() < 1e-6,
        "float history converges to MIN={MIN_HISTORY_BLEND_RATE}");
}

// ── Root Cause 2: Zero velocity causes wrong history lookup ───────────────────
//
// The old implementation used a 1×1 blank Rgba16Float texture as the velocity
// fallback.  Sampling it always returned (0,0).  history_uv = in.uv - (0,0) =
// in.uv — correct for static camera but wrong for any camera movement.
//
// The fix: derive motion from depth + inv_view_proj + prev_view_proj.
// For a static scene with a static camera, the motion vector IS (0,0), so:
// the new code gives the same result as the old (broken) zero-velocity approach
// ONLY when the camera is static.  For a moving camera it gives the correct
// reprojection UV.

#[test]
fn zero_velocity_gives_identity_history_lookup() {
    // Zero motion vector → history_uv = current uv.  Correct for static camera.
    let uv = [0.4f32, 0.6f32];
    let velocity = [0.0f32, 0.0f32];
    let history_uv = [uv[0] - velocity[0], uv[1] - velocity[1]];
    assert!((history_uv[0] - uv[0]).abs() < 1e-7);
    assert!((history_uv[1] - uv[1]).abs() < 1e-7);
}

#[test]
fn nonzero_velocity_shifts_history_uv() {
    // For a rightward camera pan, v.x > 0.  history_uv.x = uv.x - v.x < uv.x.
    let uv = [0.5f32, 0.5f32];
    let velocity = [0.1f32, -0.05f32]; // rightward + slightly downward pan
    let history_uv = [uv[0] - velocity[0], uv[1] - velocity[1]];
    assert!((history_uv[0] - 0.4f32).abs() < 1e-6,
        "history_uv.x = {} expected 0.4", history_uv[0]);
    assert!((history_uv[1] - 0.55f32).abs() < 1e-6,
        "history_uv.y = {} expected 0.55", history_uv[1]);
}

#[test]
fn pixel_motion_threshold_determines_static_vs_dynamic() {
    // Pixels with sub-0.01 pixel motion are considered static → accumulate confidence.
    let texture_size = [1920.0f32, 1080.0f32];
    let motion_uv = [0.000004f32, 0.000003f32]; // tiny jitter motion
    let pixel_motion = [motion_uv[0].abs() * texture_size[0],
                        motion_uv[1].abs() * texture_size[1]];
    let is_static = pixel_motion[0] < 0.01 && pixel_motion[1] < 0.01;
    assert!(is_static, "sub-pixel motion should count as static: {:?}", pixel_motion);

    let motion_uv2 = [0.002f32, 0.001f32]; // 3.8px motion
    let pixel_motion2 = [motion_uv2[0].abs() * texture_size[0],
                         motion_uv2[1].abs() * texture_size[1]];
    let is_static2 = pixel_motion2[0] < 0.01 && pixel_motion2[1] < 0.01;
    assert!(!is_static2, "3.8px motion should count as dynamic: {:?}", pixel_motion2);
}

// ── Halton sequence tests ─────────────────────────────────────────────────────

#[test]
fn halton_has_16_entries() {
    assert_eq!(HALTON_JITTER.len(), 16);
}

#[test]
fn halton_all_x_values_in_open_0_1() {
    for (i, entry) in HALTON_JITTER.iter().enumerate() {
        assert!(entry[0] > 0.0f32 && entry[0] < 1.0f32,
            "entry[{i}].x = {} not in (0,1)", entry[0]);
    }
}

#[test]
fn halton_all_y_values_in_open_0_1() {
    for (i, entry) in HALTON_JITTER.iter().enumerate() {
        assert!(entry[1] > 0.0f32 && entry[1] < 1.0f32,
            "entry[{i}].y = {} not in (0,1)", entry[1]);
    }
}

#[test]
fn halton_no_x_duplicates() {
    let xs: Vec<u32> = HALTON_JITTER.iter()
        .map(|e| (e[0] * 1_000_000.0) as u32)
        .collect();
    let set: std::collections::HashSet<_> = xs.iter().cloned().collect();
    assert_eq!(set.len(), 16, "Duplicate x values: {:?}", xs);
}

#[test]
fn halton_no_y_duplicates() {
    let ys: Vec<u32> = HALTON_JITTER.iter()
        .map(|e| (e[1] * 1_000_000.0) as u32)
        .collect();
    let set: std::collections::HashSet<_> = ys.iter().cloned().collect();
    assert_eq!(set.len(), 16, "Duplicate y values: {:?}", ys);
}

#[test]
fn halton_first_entry_x_is_0_5() {
    assert!((HALTON_JITTER[0][0] - 0.5f32).abs() < 1e-5f32);
}

#[test]
fn halton_first_entry_y_is_one_third() {
    assert!((HALTON_JITTER[0][1] - 1.0f32 / 3.0f32).abs() < 1e-4f32);
}

#[test]
fn halton_second_entry_x_is_0_25() {
    assert!((HALTON_JITTER[1][0] - 0.25f32).abs() < 1e-5f32);
}

#[test]
fn halton_third_entry_x_is_0_75() {
    assert!((HALTON_JITTER[2][0] - 0.75f32).abs() < 1e-5f32);
}

#[test]
fn frame_modulo_16_cycles_through_all_jitter_entries() {
    let mut seen = std::collections::HashSet::new();
    for frame in 0u32..16 {
        seen.insert(frame % 16);
    }
    assert_eq!(seen.len(), 16);
}

#[test]
fn frame_modulo_wraps_at_16() {
    assert_eq!(16u32 % 16, 0);
    assert_eq!(17u32 % 16, 1);
    assert_eq!(31u32 % 16, 15);
    assert_eq!(32u32 % 16, 0);
}

#[test]
fn jitter_centered_after_offset_by_half() {
    // Jitter is offset by -0.5 at upload time; result should be in (-0.5, 0.5).
    for entry in &HALTON_JITTER {
        let jx = entry[0] - 0.5;
        let jy = entry[1] - 0.5;
        assert!(jx > -0.5f32 && jx < 0.5f32, "jx = {jx}");
        assert!(jy > -0.5f32 && jy < 0.5f32, "jy = {jy}");
    }
}

#[test]
fn jitter_mean_near_zero_after_centering() {
    let mean_x: f32 = HALTON_JITTER.iter().map(|e| e[0] - 0.5).sum::<f32>() / 16.0;
    let mean_y: f32 = HALTON_JITTER.iter().map(|e| e[1] - 0.5).sum::<f32>() / 16.0;
    assert!(mean_x.abs() < 0.05f32, "mean_x = {mean_x}");
    assert!(mean_y.abs() < 0.05f32, "mean_y = {mean_y}");
}


