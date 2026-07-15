use nebula_visibility::{PvsConfig, PvsOutput, CHUNK_TAG};
use nebula_core::traits::BakeOutput;

// ── Chunk tag ─────────────────────────────────────────────────────────────────

#[test]
fn pvs_chunk_tag_is_pvss() {
    assert_eq!(CHUNK_TAG.to_bytes(), *b"PVSS");
}

// ── PvsConfig defaults ────────────────────────────────────────────────────────

#[test]
fn pvs_config_default_cell_size_is_3() {
    assert!((PvsConfig::default().cell_size - 3.0).abs() < f32::EPSILON);
}

#[test]
fn pvs_config_default_ray_budget_is_256() {
    assert_eq!(PvsConfig::default().ray_budget, 256);
}

#[test]
fn pvs_config_default_conservative_is_true() {
    assert!(PvsConfig::default().conservative);
}

#[test]
fn pvs_config_default_visibility_threshold_is_1() {
    assert_eq!(PvsConfig::default().visibility_threshold, 1);
}

#[test]
fn pvs_config_default_max_ray_distance_is_500() {
    assert!((PvsConfig::default().max_ray_distance - 500.0).abs() < f32::EPSILON);
}

// ── Fast preset ───────────────────────────────────────────────────────────────

#[test]
fn pvs_config_fast_cell_size_is_8() {
    assert!((PvsConfig::fast().cell_size - 8.0).abs() < f32::EPSILON);
}

#[test]
fn pvs_config_fast_ray_budget_is_32() {
    assert_eq!(PvsConfig::fast().ray_budget, 32);
}

#[test]
fn pvs_config_fast_conservative_is_false() {
    assert!(!PvsConfig::fast().conservative);
}

// ── Ultra preset ──────────────────────────────────────────────────────────────

#[test]
fn pvs_config_ultra_cell_size_is_1_5() {
    assert!((PvsConfig::ultra().cell_size - 1.5).abs() < f32::EPSILON);
}

#[test]
fn pvs_config_ultra_ray_budget_is_2048() {
    assert_eq!(PvsConfig::ultra().ray_budget, 2048);
}

#[test]
fn pvs_config_ultra_conservative_is_true() {
    assert!(PvsConfig::ultra().conservative);
}

// ── Preset ordering ───────────────────────────────────────────────────────────

#[test]
fn pvs_config_cell_size_ordering_fast_coarser() {
    assert!(PvsConfig::fast().cell_size > PvsConfig::default().cell_size);
    assert!(PvsConfig::default().cell_size > PvsConfig::ultra().cell_size);
}

#[test]
fn pvs_config_ray_budget_ordering() {
    assert!(PvsConfig::fast().ray_budget < PvsConfig::default().ray_budget);
    assert!(PvsConfig::default().ray_budget < PvsConfig::ultra().ray_budget);
}

// ── BakeOutput trait ──────────────────────────────────────────────────────────

#[test]
fn pvs_output_kind_name_is_pvs() {
    assert_eq!(PvsOutput::kind_name(), "pvs");
}

// ── Helpers for constructing minimal PvsOutput ────────────────────────────────

/// Build a tiny 4-cell (2×1×2) PVS grid for testing.
/// Layout: 4 cells total, words_per_cell = 1.
fn make_4cell_pvs() -> PvsOutput {
    let cell_count = 4u32;
    let words_per_cell = 1u32; // ceil(4 / 64) = 1
    // Each source cell sees all cells (all bits set).
    let bits = vec![0xFFFF_FFFF_FFFF_FFFFu64; (cell_count * words_per_cell) as usize];
    PvsOutput {
        world_min: [0.0, 0.0, 0.0],
        world_max: [6.0, 3.0, 6.0],
        grid_dims: [2, 1, 2],
        cell_size: 3.0,
        cell_count,
        words_per_cell,
        bits,
        config_json: "{}".to_string(),
    }
}

// ── is_visible ────────────────────────────────────────────────────────────────

#[test]
fn pvs_is_visible_all_cells_see_all_cells() {
    let pvs = make_4cell_pvs();
    for from in 0..4 {
        for to in 0..4 {
            assert!(pvs.is_visible(from, to), "cell {from} should see cell {to}");
        }
    }
}

#[test]
fn pvs_is_visible_blind_pvs_returns_false() {
    let pvs = PvsOutput {
        world_min: [0.0; 3], world_max: [6.0, 3.0, 6.0],
        grid_dims: [2, 1, 2], cell_size: 3.0, cell_count: 4,
        words_per_cell: 1, bits: vec![0u64; 4],
        config_json: "{}".to_string(),
    };
    for from in 0..4 {
        for to in 0..4 {
            assert!(!pvs.is_visible(from, to), "blind PVS should not see anything");
        }
    }
}

#[test]
fn pvs_is_visible_out_of_range_returns_false() {
    let pvs = make_4cell_pvs();
    // Requesting cell index 9999 should not panic and should return false.
    assert!(!pvs.is_visible(0, 9999));
}

#[test]
fn pvs_is_visible_selective_bits() {
    // 2-cell grid: cell 0 can see cell 1, cell 1 cannot see cell 0.
    let pvs = PvsOutput {
        world_min: [0.0; 3], world_max: [6.0, 3.0, 3.0],
        grid_dims: [2, 1, 1], cell_size: 3.0, cell_count: 2,
        words_per_cell: 1,
        // bits for cell 0: bit 1 set (can see cell 1), bit 0 clear
        // bits for cell 1: all clear (sees nothing)
        bits: vec![0b10, 0b00],
        config_json: "{}".to_string(),
    };
    assert!(!pvs.is_visible(0, 0)); // cell 0 does not see itself
    assert!(pvs.is_visible(0, 1));  // cell 0 does see cell 1
    assert!(!pvs.is_visible(1, 0)); // cell 1 does not see cell 0
    assert!(!pvs.is_visible(1, 1)); // cell 1 does not see itself
}

// ── cell_at ───────────────────────────────────────────────────────────────────

#[test]
fn pvs_cell_at_origin_is_cell_0() {
    let pvs = make_4cell_pvs();
    // grid is 2×1×2, cell_size 3.0, min at origin.
    // Point (0.5, 0.5, 0.5) → dx=0, dy=0, dz=0 → flat index 0.
    let idx = pvs.cell_at([0.5, 0.5, 0.5]);
    assert_eq!(idx, Some(0));
}

#[test]
fn pvs_cell_at_second_x_cell() {
    let pvs = make_4cell_pvs();
    // Point (3.5, 0.5, 0.5) → dx=1, dy=0, dz=0 → flat index = dz * gy * gx + dy * gx + dx = 0 + 0 + 1 = 1
    let idx = pvs.cell_at([3.5, 0.5, 0.5]);
    assert_eq!(idx, Some(1));
}

#[test]
fn pvs_cell_at_second_z_row() {
    let pvs = make_4cell_pvs();
    // Point (0.5, 0.5, 3.5) → dx=0, dy=0, dz=1 → flat index = 1 * 1 * 2 + 0 + 0 = 2
    let idx = pvs.cell_at([0.5, 0.5, 3.5]);
    assert_eq!(idx, Some(2));
}

#[test]
fn pvs_cell_at_outside_grid_returns_none() {
    let pvs = make_4cell_pvs();
    assert_eq!(pvs.cell_at([-1.0, 0.0, 0.0]), None);
    assert_eq!(pvs.cell_at([100.0, 0.0, 0.0]), None);
    assert_eq!(pvs.cell_at([0.0, -1.0, 0.0]), None);
    assert_eq!(pvs.cell_at([0.0, 0.0, 100.0]), None);
}

// ── Serialize / deserialize ───────────────────────────────────────────────────

#[test]
fn pvs_output_serialize_deserialize_roundtrip() {
    let pvs = make_4cell_pvs();
    let bytes = pvs.serialize_to_bytes().expect("serialize");
    let back = PvsOutput::deserialize_from_bytes(&bytes).expect("deserialize");
    assert_eq!(back.cell_count, 4);
    assert_eq!(back.words_per_cell, 1);
    assert_eq!(back.bits, pvs.bits);
    assert_eq!(back.grid_dims, [2, 1, 2]);
    assert_eq!(back.cell_size, 3.0);
}

#[test]
fn pvs_output_serialize_produces_non_empty_bytes() {
    let pvs = make_4cell_pvs();
    let bytes = pvs.serialize_to_bytes().expect("serialize");
    assert!(!bytes.is_empty());
}

#[test]
fn pvs_output_deserialize_corrupt_returns_error() {
    let result = PvsOutput::deserialize_from_bytes(b"\xFF\x00\x00");
    assert!(result.is_err());
}

// ── Serde (JSON) round-trip ───────────────────────────────────────────────────

#[test]
fn pvs_config_json_roundtrip() {
    let cfg = PvsConfig::ultra();
    let json = serde_json::to_string(&cfg).expect("serialize");
    let back: PvsConfig = serde_json::from_str(&json).expect("deserialize");
    assert!((back.cell_size - cfg.cell_size).abs() < f32::EPSILON);
    assert_eq!(back.ray_budget, cfg.ray_budget);
    assert_eq!(back.conservative, cfg.conservative);
}

// ── Baker (GPU-gated) ─────────────────────────────────────────────────────────

#[test]
#[ignore = "requires GPU adapter"]
fn pvs_baker_produces_output_for_empty_scene() {
    use nebula_core::{context::BakeContext, progress::NullReporter, scene::SceneGeometry};
    use nebula_visibility::PvsBaker;
    use nebula_core::traits::BakePass;

    pollster::block_on(async {
        let ctx = BakeContext::new().await.expect("BakeContext");
        let scene = SceneGeometry::default();
        let cfg = PvsConfig::fast();
        let out = PvsBaker.execute(&scene, &cfg, &ctx, &NullReporter).await
            .expect("bake");
        assert_eq!(out.cell_count, out.grid_dims[0] * out.grid_dims[1] * out.grid_dims[2]);
    });
}
