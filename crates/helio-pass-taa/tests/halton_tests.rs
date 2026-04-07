// Tests for helio-pass-taa: Halton sequence — base-2 (x), base-3 (y),
// all values in (0,1), no repeats, low-discrepancy properties.
// All tests are pure math — no GPU device required.

/// Exact Halton(2,3) sequence copied from the source.
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

/// Compute Halton(base, index) for 1-based index.
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

// ── Base-2 (x channel) correctness ───────────────────────────────────────────

#[test]
fn halton_x_matches_base2_entry_0() {
    let expected = halton(2, 1);
    assert!(
        (HALTON_JITTER[0][0] - expected).abs() < 1e-5f32,
        "got {}, expected {expected}",
        HALTON_JITTER[0][0]
    );
}

#[test]
fn halton_x_matches_base2_entry_1() {
    let expected = halton(2, 2);
    assert!((HALTON_JITTER[1][0] - expected).abs() < 1e-5f32);
}

#[test]
fn halton_x_matches_base2_entry_2() {
    let expected = halton(2, 3);
    assert!((HALTON_JITTER[2][0] - expected).abs() < 1e-5f32);
}

#[test]
fn halton_x_matches_base2_entry_3() {
    let expected = halton(2, 4);
    assert!((HALTON_JITTER[3][0] - expected).abs() < 1e-5f32);
}

#[test]
fn halton_x_matches_base2_all_16() {
    for i in 0..16usize {
        let expected = halton(2, i as u32 + 1);
        assert!(
            (HALTON_JITTER[i][0] - expected).abs() < 1e-5f32,
            "entry[{i}].x = {}, expected {expected}",
            HALTON_JITTER[i][0]
        );
    }
}

#[test]
fn halton_base2_values_are_binary_fractions() {
    // All base-2 Halton values are exact binary fractions with denominators that are
    // powers of 2 ≤ 32 for the first 16 entries.
    for (i, entry) in HALTON_JITTER.iter().enumerate() {
        let x = entry[0];
        // Multiply by 32 (= 2^5) and check it's near an integer
        let scaled = x * 32.0f32;
        let rounded = scaled.round();
        assert!(
            (scaled - rounded).abs() < 1e-3f32,
            "entry[{i}].x = {x} not a clean binary fraction"
        );
    }
}

// ── Base-3 (y channel) correctness ───────────────────────────────────────────

#[test]
fn halton_y_matches_base3_entry_0() {
    let expected = halton(3, 1);
    assert!(
        (HALTON_JITTER[0][1] - expected).abs() < 1e-4f32,
        "got {}, expected {expected}",
        HALTON_JITTER[0][1]
    );
}

#[test]
fn halton_y_matches_base3_entry_1() {
    let expected = halton(3, 2);
    assert!((HALTON_JITTER[1][1] - expected).abs() < 1e-4f32);
}

#[test]
fn halton_y_matches_base3_entry_2() {
    let expected = halton(3, 3);
    assert!((HALTON_JITTER[2][1] - expected).abs() < 1e-4f32);
}

#[test]
fn halton_y_matches_base3_all_16() {
    for i in 0..16usize {
        let expected = halton(3, i as u32 + 1);
        assert!(
            (HALTON_JITTER[i][1] - expected).abs() < 1e-4f32,
            "entry[{i}].y = {}, expected {expected}",
            HALTON_JITTER[i][1]
        );
    }
}

// ── Range tests ───────────────────────────────────────────────────────────────

#[test]
fn all_x_strictly_between_0_and_1() {
    for (i, e) in HALTON_JITTER.iter().enumerate() {
        assert!(e[0] > 0.0f32, "entry[{i}].x = {} not > 0", e[0]);
        assert!(e[0] < 1.0f32, "entry[{i}].x = {} not < 1", e[0]);
    }
}

#[test]
fn all_y_strictly_between_0_and_1() {
    for (i, e) in HALTON_JITTER.iter().enumerate() {
        assert!(e[1] > 0.0f32, "entry[{i}].y = {} not > 0", e[1]);
        assert!(e[1] < 1.0f32, "entry[{i}].y = {} not < 1", e[1]);
    }
}

// ── Uniqueness tests ──────────────────────────────────────────────────────────

#[test]
fn x_values_all_unique() {
    let xs: Vec<u32> = HALTON_JITTER
        .iter()
        .map(|e| (e[0] * 1_000_000.0f32) as u32)
        .collect();
    let uniq: std::collections::HashSet<_> = xs.iter().copied().collect();
    assert_eq!(uniq.len(), 16, "Duplicate x values in Halton sequence");
}

#[test]
fn y_values_all_unique() {
    let ys: Vec<u32> = HALTON_JITTER
        .iter()
        .map(|e| (e[1] * 1_000_000.0f32) as u32)
        .collect();
    let uniq: std::collections::HashSet<_> = ys.iter().copied().collect();
    assert_eq!(uniq.len(), 16, "Duplicate y values in Halton sequence");
}

// ── Low-discrepancy properties ────────────────────────────────────────────────

#[test]
fn x_mean_close_to_0_5() {
    let mean = HALTON_JITTER.iter().map(|e| e[0]).sum::<f32>() / 16.0;
    assert!((mean - 0.5f32).abs() < 0.05f32, "mean_x = {mean}");
}

#[test]
fn y_mean_close_to_0_5() {
    let mean = HALTON_JITTER.iter().map(|e| e[1]).sum::<f32>() / 16.0;
    assert!((mean - 0.5f32).abs() < 0.05f32, "mean_y = {mean}");
}

#[test]
fn x_covers_both_halves_of_0_1() {
    let below_half = HALTON_JITTER.iter().filter(|e| e[0] < 0.5).count();
    let above_half = HALTON_JITTER.iter().filter(|e| e[0] >= 0.5).count();
    assert_eq!(below_half, above_half, "x not evenly split at 0.5");
}

#[test]
fn y_covers_multiple_thirds() {
    let t0 = HALTON_JITTER.iter().filter(|e| e[1] < 1.0 / 3.0).count();
    let t1 = HALTON_JITTER
        .iter()
        .filter(|e| e[1] >= 1.0 / 3.0 && e[1] < 2.0 / 3.0)
        .count();
    let t2 = HALTON_JITTER.iter().filter(|e| e[1] >= 2.0 / 3.0).count();
    // All three thirds should be represented
    assert!(t0 > 0 && t1 > 0 && t2 > 0, "t0={t0} t1={t1} t2={t2}");
}

#[test]
fn sequence_variance_x_is_reasonable() {
    let mean = HALTON_JITTER.iter().map(|e| e[0]).sum::<f32>() / 16.0;
    let var = HALTON_JITTER
        .iter()
        .map(|e| (e[0] - mean) * (e[0] - mean))
        .sum::<f32>()
        / 16.0;
    // Low-discrepancy; variance should be roughly 1/12 (uniform distribution variance)
    let expected_var = 1.0f32 / 12.0f32;
    assert!((var - expected_var).abs() < 0.05f32, "var_x = {var}");
}

#[test]
fn halton_base2_produces_van_der_corput_sequence() {
    // Van der Corput: each new sample fills the largest gap
    let computed: Vec<f32> = (1u32..=8).map(|i| halton(2, i)).collect();
    // Expected: 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625
    let expected = [0.5f32, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625];
    for (i, (&c, &e)) in computed.iter().zip(expected.iter()).enumerate() {
        assert!(
            (c - e).abs() < 1e-5f32,
            "index {i}: computed {c}, expected {e}"
        );
    }
}

#[test]
fn jitter_after_centering_covers_both_signs() {
    let negatives_x = HALTON_JITTER.iter().filter(|e| e[0] - 0.5 < 0.0).count();
    let positives_x = HALTON_JITTER.iter().filter(|e| e[0] - 0.5 > 0.0).count();
    assert!(negatives_x > 0 && positives_x > 0);
}
