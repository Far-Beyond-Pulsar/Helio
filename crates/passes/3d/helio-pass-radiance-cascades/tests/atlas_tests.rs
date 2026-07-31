// Probe atlas addressing tests for helio-pass-radiance-cascades.
// All helpers defined locally — no crate imports required.

const PROBE_DIM: u32 = 8;
const DIR_DIM: u32 = 4;
const ATLAS_W: u32 = 32; // PROBE_DIM * DIR_DIM
const ATLAS_H: u32 = 256; // PROBE_DIM * PROBE_DIM * DIR_DIM

/// X coordinate in the probe atlas.
/// px ∈ [0, PROBE_DIM), dx ∈ [0, DIR_DIM)  →  result ∈ [0, ATLAS_W)
fn probe_atlas_x(px: u32, dx: u32) -> u32 {
    px * DIR_DIM + dx
}

/// Y coordinate in the probe atlas.
/// py, pz ∈ [0, PROBE_DIM), dy ∈ [0, DIR_DIM)  →  result ∈ [0, ATLAS_H)
fn probe_atlas_y(py: u32, pz: u32, dy: u32) -> u32 {
    (py * PROBE_DIM + pz) * DIR_DIM + dy
}

// ── probe_atlas_x spot checks ─────────────────────────────────────────────────

#[test]
fn atlas_x_origin() {
    assert_eq!(probe_atlas_x(0, 0), 0);
}

#[test]
fn atlas_x_maximum() {
    // px=7, dx=3  →  7*4+3 = 31
    assert_eq!(probe_atlas_x(7, 3), 31);
}

#[test]
fn atlas_x_stride_by_px() {
    // Each probe index advances x by DIR_DIM=4
    assert_eq!(probe_atlas_x(1, 0), 4);
    assert_eq!(probe_atlas_x(2, 0), 8);
    assert_eq!(probe_atlas_x(3, 0), 12);
}

#[test]
fn atlas_x_stride_by_dx() {
    // Each direction index advances x by 1
    assert_eq!(probe_atlas_x(0, 1), 1);
    assert_eq!(probe_atlas_x(0, 2), 2);
    assert_eq!(probe_atlas_x(0, 3), 3);
}

#[test]
fn atlas_x_spot_check_px3_dx2() {
    // px=3, dx=2  →  3*4+2 = 14
    assert_eq!(probe_atlas_x(3, 2), 14);
}

#[test]
fn atlas_x_spot_check_px5_dx1() {
    // 5*4+1 = 21
    assert_eq!(probe_atlas_x(5, 1), 21);
}

#[test]
fn atlas_x_spot_check_px7_dx0() {
    // 7*4+0 = 28
    assert_eq!(probe_atlas_x(7, 0), 28);
}

// ── probe_atlas_y spot checks ─────────────────────────────────────────────────

#[test]
fn atlas_y_origin() {
    assert_eq!(probe_atlas_y(0, 0, 0), 0);
}

#[test]
fn atlas_y_maximum() {
    // py=7, pz=7, dy=3  →  (7*8+7)*4+3 = 63*4+3 = 255
    assert_eq!(probe_atlas_y(7, 7, 3), 255);
}

#[test]
fn atlas_y_spot_check_py2_pz5_dy1() {
    // (2*8+5)*4+1 = 21*4+1 = 85
    assert_eq!(probe_atlas_y(2, 5, 1), 85);
}

#[test]
fn atlas_y_spot_check_py0_pz1_dy0() {
    // (0*8+1)*4+0 = 4
    assert_eq!(probe_atlas_y(0, 1, 0), 4);
}

#[test]
fn atlas_y_stride_by_pz() {
    // Each pz increment advances y by DIR_DIM=4
    assert_eq!(probe_atlas_y(0, 0, 0), 0);
    assert_eq!(probe_atlas_y(0, 1, 0), 4);
    assert_eq!(probe_atlas_y(0, 2, 0), 8);
}

#[test]
fn atlas_y_stride_by_py() {
    // Each py increment advances y by PROBE_DIM*DIR_DIM = 8*4 = 32
    assert_eq!(probe_atlas_y(0, 0, 0), 0);
    assert_eq!(probe_atlas_y(1, 0, 0), 32);
    assert_eq!(probe_atlas_y(2, 0, 0), 64);
    assert_eq!(probe_atlas_y(7, 0, 0), 224);
}

#[test]
fn atlas_y_stride_by_dy() {
    // Each dy increment advances y by 1
    assert_eq!(probe_atlas_y(0, 0, 1), 1);
    assert_eq!(probe_atlas_y(0, 0, 2), 2);
    assert_eq!(probe_atlas_y(0, 0, 3), 3);
}

// ── Bounds checks: all outputs within [0, ATLAS_W) and [0, ATLAS_H) ──────────

#[test]
fn all_x_coordinates_within_atlas_width() {
    for px in 0..PROBE_DIM {
        for dx in 0..DIR_DIM {
            let x = probe_atlas_x(px, dx);
            assert!(
                x < ATLAS_W,
                "probe_atlas_x({},{}) = {} out of bounds (ATLAS_W={})",
                px, dx, x, ATLAS_W
            );
        }
    }
}

#[test]
fn all_y_coordinates_within_atlas_height() {
    for py in 0..PROBE_DIM {
        for pz in 0..PROBE_DIM {
            for dy in 0..DIR_DIM {
                let y = probe_atlas_y(py, pz, dy);
                assert!(
                    y < ATLAS_H,
                    "probe_atlas_y({},{},{}) = {} out of bounds (ATLAS_H={})",
                    py, pz, dy, y, ATLAS_H
                );
            }
        }
    }
}

// ── First probe (0,0,0): all 16 direction pairs ───────────────────────────────

#[test]
fn first_probe_all_directions_x_range() {
    // px=0: x = dx ∈ {0,1,2,3}
    for dx in 0..DIR_DIM {
        assert_eq!(probe_atlas_x(0, dx), dx);
    }
}

#[test]
fn first_probe_all_directions_y_range() {
    // py=0, pz=0: y = dy ∈ {0,1,2,3}
    for dy in 0..DIR_DIM {
        assert_eq!(probe_atlas_y(0, 0, dy), dy);
    }
}

// ── Last probe (7,7,7): all 16 direction pairs ────────────────────────────────

#[test]
fn last_probe_last_direction_is_final_texel() {
    // px=7, dx=3  →  x=31 = ATLAS_W-1
    // py=7, pz=7, dy=3  →  y=255 = ATLAS_H-1
    assert_eq!(probe_atlas_x(7, 3), ATLAS_W - 1);
    assert_eq!(probe_atlas_y(7, 7, 3), ATLAS_H - 1);
}

#[test]
fn last_probe_first_direction() {
    // px=7, dx=0  →  x=28
    // py=7, pz=7, dy=0  →  y=252
    assert_eq!(probe_atlas_x(7, 0), 28);
    assert_eq!(probe_atlas_y(7, 7, 0), 252);
}

// ── Monotonicity ──────────────────────────────────────────────────────────────

#[test]
fn atlas_x_increases_with_px() {
    for px in 0..PROBE_DIM - 1 {
        assert!(probe_atlas_x(px, 0) < probe_atlas_x(px + 1, 0));
    }
}

#[test]
fn atlas_x_increases_with_dx() {
    for dx in 0..DIR_DIM - 1 {
        assert!(probe_atlas_x(0, dx) < probe_atlas_x(0, dx + 1));
    }
}

#[test]
fn atlas_y_increases_with_py() {
    for py in 0..PROBE_DIM - 1 {
        assert!(probe_atlas_y(py, 0, 0) < probe_atlas_y(py + 1, 0, 0));
    }
}

#[test]
fn atlas_y_increases_with_pz() {
    for pz in 0..PROBE_DIM - 1 {
        assert!(probe_atlas_y(0, pz, 0) < probe_atlas_y(0, pz + 1, 0));
    }
}

#[test]
fn atlas_y_increases_with_dy() {
    for dy in 0..DIR_DIM - 1 {
        assert!(probe_atlas_y(0, 0, dy) < probe_atlas_y(0, 0, dy + 1));
    }
}

// ── Uniqueness: no two distinct probe+direction tuples collide ────────────────

#[test]
fn distinct_px_gives_distinct_x() {
    // px=1 and px=2 with same dx → different x
    assert_ne!(probe_atlas_x(1, 0), probe_atlas_x(2, 0));
    assert_ne!(probe_atlas_x(3, 2), probe_atlas_x(4, 2));
}

#[test]
fn distinct_dx_gives_distinct_x() {
    assert_ne!(probe_atlas_x(0, 0), probe_atlas_x(0, 1));
    assert_ne!(probe_atlas_x(5, 1), probe_atlas_x(5, 3));
}

#[test]
fn distinct_py_gives_distinct_y() {
    assert_ne!(probe_atlas_y(0, 0, 0), probe_atlas_y(1, 0, 0));
    assert_ne!(probe_atlas_y(3, 2, 1), probe_atlas_y(4, 2, 1));
}

#[test]
fn distinct_pz_gives_distinct_y() {
    assert_ne!(probe_atlas_y(0, 0, 0), probe_atlas_y(0, 1, 0));
    assert_ne!(probe_atlas_y(2, 3, 2), probe_atlas_y(2, 4, 2));
}

#[test]
fn all_texel_coordinates_are_unique() {
    // Exhaustively verify that every (px,py,pz,dx,dy) maps to a unique texel.
    let mut seen = std::collections::HashSet::new();
    for px in 0..PROBE_DIM {
        for py in 0..PROBE_DIM {
            for pz in 0..PROBE_DIM {
                for dx in 0..DIR_DIM {
                    for dy in 0..DIR_DIM {
                        let x = probe_atlas_x(px, dx);
                        let y = probe_atlas_y(py, pz, dy);
                        assert!(
                            seen.insert((x, y)),
                            "Collision at ({},{}) for probe ({},{},{}) dir ({},{})",
                            x, y, px, py, pz, dx, dy
                        );
                    }
                }
            }
        }
    }
    // Every texel in the atlas is covered exactly once.
    assert_eq!(seen.len() as u32, ATLAS_W * ATLAS_H);
}
