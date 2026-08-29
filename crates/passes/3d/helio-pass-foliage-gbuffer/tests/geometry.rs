//! CPU mirror of the blade geometry, pinned.
//!
//! `foliage_gbuffer.wgsl` derives every vertex from `@builtin(vertex_index)`, and there
//! is no buffer anywhere to inspect if it gets it wrong — a folded strip or an
//! inside-out blade is a screenful of flickering polygons with no intermediate state.
//! These asserts are the only place the geometry can be looked at directly, so they pin
//! the actual numbers rather than properties of them.
//!
//! When the WGSL changes, these change with it. That is the point: the diff makes the
//! silhouette change visible in review instead of only on a GPU.

use helio_pass_foliage_gbuffer::*;

/// Positions are exact rational values at these inputs, but they go through `f32`
/// division, so compare with a tolerance rather than `==`.
fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1.0e-6
}

fn assert_position(lod: u32, index: u32, height: f32, width: f32, expected: [f32; 3]) {
    let got = blade_local_position(lod, index, height, width);
    assert!(
        close(got[0], expected[0]) && close(got[1], expected[1]) && close(got[2], expected[2]),
        "LOD {lod} vertex {index}: got {got:?}, expected {expected:?}"
    );
}

// ── Row / side derivation ─────────────────────────────────────────────────────

#[test]
fn vertices_walk_up_the_blade_two_at_a_time_and_end_in_a_collapsed_tip() {
    // LOD 0: 5 segments, 11 vertices. Rows 0,0,1,1,2,2,3,3,4,4 then the tip.
    let expected_rows = [0u32, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5];
    let expected_sides = [
        -1.0f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 0.0,
    ];
    for index in 0..LOD_VERTEX_COUNTS[0] {
        let v = blade_vertex(0, index);
        assert_eq!(v.row, expected_rows[index as usize], "row at index {index}");
        assert_eq!(
            v.side, expected_sides[index as usize],
            "side at index {index}"
        );
        assert_eq!(v.is_tip, index == 10, "tip flag at index {index}");
    }

    // LOD 1: 3 segments, 7 vertices.
    for index in 0..LOD_VERTEX_COUNTS[1] {
        let v = blade_vertex(1, index);
        assert_eq!(v.is_tip, index == 6);
        assert_eq!(v.row, if index == 6 { 3 } else { index >> 1 });
    }
}

#[test]
fn cards_are_plain_quads_with_no_tip_and_no_taper() {
    for lod in [2u32, 3] {
        for index in 0..LOD_VERTEX_COUNTS[lod as usize] {
            let v = blade_vertex(lod, index);
            assert!(!v.is_tip, "LOD {lod} must have no collapsed tip vertex");
            // Constant width top to bottom: a card is a rectangle, and tapering it would
            // put a different silhouette on screen from the blade it replaces at exactly
            // the distance where both are drawn.
            assert_eq!(v.width_frac, 1.0);
            assert_eq!(v.row, index >> 1);
            assert_eq!(v.side, if index & 1 == 1 { 1.0 } else { -1.0 });
        }
    }
}

#[test]
fn the_root_is_exactly_zero_and_the_tip_is_exactly_one() {
    // Not "approximately". Every wind band scales with `height_frac`, so a root that is
    // 1e-7 rather than 0 lets the blade's base leave the terrain, which reveals the
    // ground plane under it. And a tip that is not exactly 1 makes the LOD's maximum
    // displacement depend on the segment count, so a blade visibly changes its wind
    // amplitude at the L0→L1 boundary.
    for lod in 0..4u32 {
        assert_eq!(blade_vertex(lod, 0).height_frac, 0.0);
        assert_eq!(blade_vertex(lod, 1).height_frac, 0.0);
        let last = LOD_VERTEX_COUNTS[lod as usize] - 1;
        assert_eq!(
            blade_vertex(lod, last).height_frac,
            1.0,
            "LOD {lod} top row is not at height_frac 1.0"
        );
    }
}

// ── Pinned positions ──────────────────────────────────────────────────────────

#[test]
fn lod0_blade_positions_are_pinned() {
    // A 1 m blade 0.2 m wide. Round numbers so the expected values are readable:
    // half-width 0.1, rows every 0.2, curl 0.35 · t².
    const H: f32 = 1.0;
    const W: f32 = 0.2;

    assert_position(0, 0, H, W, [-0.1, 0.0, 0.0]);
    assert_position(0, 1, H, W, [0.1, 0.0, 0.0]);
    // Row 1: t = 0.2, taper 1 - 0.04 = 0.96, curl 0.35 · 0.04 = 0.014.
    assert_position(0, 2, H, W, [-0.096, 0.2, 0.014]);
    assert_position(0, 3, H, W, [0.096, 0.2, 0.014]);
    // Row 2: t = 0.4, taper 0.84, curl 0.35 · 0.16 = 0.056.
    assert_position(0, 4, H, W, [-0.084, 0.4, 0.056]);
    // Row 4: t = 0.8, taper 0.36, curl 0.35 · 0.64 = 0.224.
    assert_position(0, 8, H, W, [-0.036, 0.8, 0.224]);
    assert_position(0, 9, H, W, [0.036, 0.8, 0.224]);
    // Tip: collapsed to the centreline, at full height, fully curled.
    assert_position(0, 10, H, W, [0.0, 1.0, 0.35]);
}

#[test]
fn lod1_blade_positions_are_pinned() {
    // Same blade, three segments. The tip must land in the *same place* as LOD 0's —
    // that is the silhouette continuity the L0→L1 cross-fade depends on.
    const H: f32 = 1.0;
    const W: f32 = 0.2;

    assert_position(1, 0, H, W, [-0.1, 0.0, 0.0]);
    assert_position(1, 1, H, W, [0.1, 0.0, 0.0]);
    let third = 1.0f32 / 3.0;
    let taper = 1.0 - third * third;
    assert_position(1, 2, H, W, [-0.1 * taper, third, 0.35 * third * third]);
    assert_position(1, 6, H, W, [0.0, 1.0, 0.35]);

    let lod0_tip = blade_local_position(0, LOD_VERTEX_COUNTS[0] - 1, H, W);
    let lod1_tip = blade_local_position(1, LOD_VERTEX_COUNTS[1] - 1, H, W);
    assert_eq!(lod0_tip, lod1_tip, "the two blade LODs must share a tip");
}

#[test]
fn card_positions_are_pinned_and_flat() {
    // L2: a 1 m × 0.2 m rectangle, no curl.
    assert_position(2, 0, 1.0, 0.2, [-0.1, 0.0, 0.0]);
    assert_position(2, 1, 1.0, 0.2, [0.1, 0.0, 0.0]);
    assert_position(2, 2, 1.0, 0.2, [-0.1, 1.0, 0.0]);
    assert_position(2, 3, 1.0, 0.2, [0.1, 1.0, 0.0]);

    // L3 is the same geometry; the clump's extra size comes from the LOD scales the
    // caller applies before it gets here, not from the vertex derivation.
    let clump_h = 1.0 * LOD_HEIGHT_SCALE[3];
    let clump_w = 0.2 * LOD_WIDTH_SCALE[3];
    assert_position(3, 0, clump_h, clump_w, [-0.5 * clump_w, 0.0, 0.0]);
    assert_position(3, 3, clump_h, clump_w, [0.5 * clump_w, clump_h, 0.0]);
    assert!(
        LOD_WIDTH_SCALE[3] > LOD_WIDTH_SCALE[2],
        "a clump card must be wider than one blade"
    );

    // Every card vertex is planar in the blade's local Z.
    for lod in [2u32, 3] {
        for index in 0..LOD_VERTEX_COUNTS[lod as usize] {
            assert_eq!(blade_local_position(lod, index, 1.0, 0.2)[2], 0.0);
        }
    }
}

// ── Winding ───────────────────────────────────────────────────────────────────

/// Signed area of a triangle in the blade's local XY plane. Sign is the winding.
fn signed_area(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> f32 {
    let ab = [b[0] - a[0], b[1] - a[1]];
    let ac = [c[0] - a[0], c[1] - a[1]];
    0.5 * (ab[0] * ac[1] - ab[1] * ac[0])
}

#[test]
fn every_strip_triangle_at_every_lod_shares_a_winding() {
    // `cull_mode: None` means a flipped triangle is not culled, so this cannot show up
    // as a hole. It shows up as a lighting seam instead: the fragment shader flips the
    // normal by `@builtin(front_facing)`, so one inconsistently wound triangle in the
    // middle of a blade lights as if it faced the other way. That is much harder to spot
    // and much harder to attribute than a missing triangle.
    for lod in 0..4u32 {
        let count = LOD_VERTEX_COUNTS[lod as usize];
        let positions: Vec<[f32; 3]> = (0..count)
            .map(|index| blade_local_position(lod, index, 1.0, 0.2))
            .collect();

        let mut sign = 0.0f32;
        for triangle in 0..count - 2 {
            let [i0, i1, i2] = strip_triangle(triangle);
            let area = signed_area(
                positions[i0 as usize],
                positions[i1 as usize],
                positions[i2 as usize],
            );
            assert!(
                area.abs() > 1.0e-9,
                "LOD {lod} triangle {triangle} is degenerate — a strip with a zero-area \
                 triangle in it has a fold"
            );
            if sign == 0.0 {
                sign = area.signum();
            }
            assert_eq!(
                area.signum(),
                sign,
                "LOD {lod} triangle {triangle} winds the other way ({area})"
            );
        }
        // Pinned rather than merely consistent: the value is what the front face *is*,
        // and flipping it would invert every blade's lighting at once.
        assert_eq!(
            sign, 1.0,
            "LOD {lod} strip is not counter-clockwise in local XY"
        );
    }
}

#[test]
fn strip_triangle_flips_odd_triangles_the_way_webgpu_does() {
    assert_eq!(strip_triangle(0), [0, 1, 2]);
    assert_eq!(strip_triangle(1), [2, 1, 3]);
    assert_eq!(strip_triangle(2), [2, 3, 4]);
    assert_eq!(strip_triangle(3), [4, 3, 5]);
}

// ── Normals ───────────────────────────────────────────────────────────────────

#[test]
fn normals_are_unit_length_everywhere_including_the_collapsed_tip() {
    // The analytic normal exists precisely because the differenced one would divide by
    // the zero-width edge at the tip.
    for lod in 0..4u32 {
        for index in 0..LOD_VERTEX_COUNTS[lod as usize] {
            let n = blade_local_normal(lod, index, 1.0);
            let length = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!(
                (length - 1.0).abs() < 1.0e-5,
                "LOD {lod} vertex {index} normal is not unit length ({length})"
            );
        }
    }
}

#[test]
fn a_card_normal_faces_straight_out_of_its_plane() {
    // No curl means no tilt: a card's normal is +Z in blade-local space at every vertex,
    // which is what lets L2/L3 inherit the blade's yaw and keep the silhouette direction
    // across the boundary.
    for lod in [2u32, 3] {
        for index in 0..LOD_VERTEX_COUNTS[lod as usize] {
            let n = blade_local_normal(lod, index, 1.0);
            assert_eq!(n, [0.0, 0.0, 1.0]);
        }
    }
}

#[test]
fn a_blade_normal_tilts_forward_as_the_blade_curls() {
    // The normal must rotate with the curl, or a strongly curled blade lights as if it
    // were flat and the curl is invisible under any directional light.
    let root = blade_local_normal(0, 0, 1.0);
    let tip = blade_local_normal(0, 10, 1.0);
    assert_eq!(
        root,
        [0.0, 0.0, 1.0],
        "the root is vertical, so its normal is unturned"
    );
    assert!(tip[1] < 0.0, "the tip's normal must tip over with the curl");
    assert!(
        tip[2] > 0.0,
        "and must not flip past horizontal at the default curl"
    );
}

// ── Defensive behaviour ───────────────────────────────────────────────────────

#[test]
fn an_out_of_range_lod_or_vertex_index_degenerates_rather_than_folding() {
    // Neither can happen in a correct draw. Both come from GPU-side state, though, and
    // the WGSL cannot panic — so both sides clamp, and a producer that writes the wrong
    // vertex count draws a sliver instead of a blade turned inside out.
    let clamped = blade_vertex(99, 0);
    assert_eq!(clamped, blade_vertex(3, 0));

    let past_the_end = blade_vertex(0, 64);
    assert_eq!(past_the_end.row, LOD_SEGMENTS[0]);
    assert_eq!(past_the_end.height_frac, 1.0);
    assert!(!past_the_end.is_tip);
}
