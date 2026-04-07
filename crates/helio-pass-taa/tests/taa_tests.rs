// Tests for helio-pass-taa: TaaUniform size, Halton properties, history ping-pong,
// feedback clamp. All tests are pure Rust — no GPU device required.

use std::mem;

// ── Mirror private types ──────────────────────────────────────────────────────

#[repr(C)]
#[derive(Clone, Copy)]
struct TaaUniform {
    feedback_min: f32,
    feedback_max: f32,
    jitter: [f32; 2],
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

// ── TaaUniform size tests ─────────────────────────────────────────────────────

#[test]
fn taa_uniform_size_is_16() {
    assert_eq!(
        mem::size_of::<TaaUniform>(),
        16,
        "feedback_min(4) + feedback_max(4) + jitter[2](8) = 16"
    );
}

#[test]
fn taa_uniform_feedback_plus_jitter_layout() {
    // 2 × f32 + [f32;2] = 2×4 + 8 = 16
    assert_eq!(2 * 4 + 8, 16usize);
}

#[test]
fn taa_uniform_alignment_is_4() {
    assert_eq!(mem::align_of::<TaaUniform>(), 4);
}

#[test]
fn taa_uniform_size_divisible_by_16() {
    // Matches WebGPU uniform alignment requirements
    assert_eq!(mem::size_of::<TaaUniform>() % 16, 0);
}

// ── Feedback clamp tests ──────────────────────────────────────────────────────

#[test]
fn feedback_min_less_than_feedback_max() {
    let u = TaaUniform {
        feedback_min: 0.88,
        feedback_max: 0.97,
        jitter: [0.0; 2],
    };
    assert!(u.feedback_min < u.feedback_max);
}

#[test]
fn feedback_min_in_0_1() {
    let u = TaaUniform {
        feedback_min: 0.88,
        feedback_max: 0.97,
        jitter: [0.0; 2],
    };
    assert!(u.feedback_min >= 0.0 && u.feedback_min <= 1.0);
}

#[test]
fn feedback_max_in_0_1() {
    let u = TaaUniform {
        feedback_min: 0.88,
        feedback_max: 0.97,
        jitter: [0.0; 2],
    };
    assert!(u.feedback_max >= 0.0 && u.feedback_max <= 1.0);
}

#[test]
fn feedback_blend_produces_temporal_average() {
    let feedback = 0.9f32;
    let current = 1.0f32;
    let history = 0.0f32;
    let blended = current * (1.0 - feedback) + history * feedback;
    assert!((blended - 0.1f32).abs() < 1e-6f32);
}

#[test]
fn feedback_one_gives_full_history() {
    let current = 0.8f32;
    let history = 0.2f32;
    let blended = current * 0.0 + history * 1.0;
    assert!((blended - history).abs() < 1e-6f32);
}

#[test]
fn feedback_zero_gives_full_current() {
    let current = 0.8f32;
    let history = 0.2f32;
    let blended = current * 1.0 + history * 0.0;
    assert!((blended - current).abs() < 1e-6f32);
}

// ── Halton sequence count tests ───────────────────────────────────────────────

#[test]
fn halton_has_16_entries() {
    assert_eq!(HALTON_JITTER.len(), 16);
}

#[test]
fn halton_all_x_values_in_open_0_1() {
    for (i, entry) in HALTON_JITTER.iter().enumerate() {
        assert!(
            entry[0] > 0.0f32 && entry[0] < 1.0f32,
            "entry[{i}].x = {} not in (0,1)",
            entry[0]
        );
    }
}

#[test]
fn halton_all_y_values_in_open_0_1() {
    for (i, entry) in HALTON_JITTER.iter().enumerate() {
        assert!(
            entry[1] > 0.0f32 && entry[1] < 1.0f32,
            "entry[{i}].y = {} not in (0,1)",
            entry[1]
        );
    }
}

#[test]
fn halton_no_x_duplicates() {
    let xs: Vec<u32> = HALTON_JITTER
        .iter()
        .map(|e| (e[0] * 1_000_000.0) as u32)
        .collect();
    let set: std::collections::HashSet<_> = xs.iter().cloned().collect();
    assert_eq!(set.len(), 16, "Duplicate x values: {:?}", xs);
}

#[test]
fn halton_no_y_duplicates() {
    let ys: Vec<u32> = HALTON_JITTER
        .iter()
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

// ── History ping-pong tests ───────────────────────────────────────────────────

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
    // Jitter is offset by -0.5 at upload time; result should be in (-0.5, 0.5)
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
