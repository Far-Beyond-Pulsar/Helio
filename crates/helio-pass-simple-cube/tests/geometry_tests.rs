//! Cube geometry tests using local mirrors of `cube_vertices()` and `cube_indices()`.
//!
//! Pure math — no GPU, no wgpu, no crate imports.
//! The vertex struct, helper function, and geometry exactly mirror the private
//! implementation in `helio-pass-simple-cube/src/lib.rs`.

/// Exact mirror of the private `CubeVertex` struct in lib.rs.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

fn v(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> CubeVertex {
    CubeVertex {
        position,
        normal,
        color,
    }
}

/// Exact mirror of the private `cube_vertices()` function in lib.rs.
fn cube_vertices() -> [CubeVertex; 24] {
    let r = [1.0_f32, 0.25, 0.25]; // +X  red
    let c = [0.25_f32, 1.0, 1.0]; // -X  cyan
    let g = [0.25_f32, 1.0, 0.25]; // +Y  green
    let m = [1.0_f32, 0.25, 1.0]; // -Y  magenta
    let b = [0.3_f32, 0.5, 1.0]; // +Z  blue
    let y = [1.0_f32, 1.0, 0.25]; // -Z  yellow

    [
        // +X face (verts 0–3)
        v([0.5, -0.5, -0.5], [1., 0., 0.], r),
        v([0.5, 0.5, -0.5], [1., 0., 0.], r),
        v([0.5, 0.5, 0.5], [1., 0., 0.], r),
        v([0.5, -0.5, 0.5], [1., 0., 0.], r),
        // -X face (verts 4–7)
        v([-0.5, -0.5, 0.5], [-1., 0., 0.], c),
        v([-0.5, 0.5, 0.5], [-1., 0., 0.], c),
        v([-0.5, 0.5, -0.5], [-1., 0., 0.], c),
        v([-0.5, -0.5, -0.5], [-1., 0., 0.], c),
        // +Y face (verts 8–11)
        v([-0.5, 0.5, -0.5], [0., 1., 0.], g),
        v([-0.5, 0.5, 0.5], [0., 1., 0.], g),
        v([0.5, 0.5, 0.5], [0., 1., 0.], g),
        v([0.5, 0.5, -0.5], [0., 1., 0.], g),
        // -Y face (verts 12–15)
        v([-0.5, -0.5, 0.5], [0., -1., 0.], m),
        v([-0.5, -0.5, -0.5], [0., -1., 0.], m),
        v([0.5, -0.5, -0.5], [0., -1., 0.], m),
        v([0.5, -0.5, 0.5], [0., -1., 0.], m),
        // +Z face (verts 16–19)
        v([-0.5, -0.5, 0.5], [0., 0., 1.], b),
        v([0.5, -0.5, 0.5], [0., 0., 1.], b),
        v([0.5, 0.5, 0.5], [0., 0., 1.], b),
        v([-0.5, 0.5, 0.5], [0., 0., 1.], b),
        // -Z face (verts 20–23)
        v([0.5, -0.5, -0.5], [0., 0., -1.], y),
        v([-0.5, -0.5, -0.5], [0., 0., -1.], y),
        v([-0.5, 0.5, -0.5], [0., 0., -1.], y),
        v([0.5, 0.5, -0.5], [0., 0., -1.], y),
    ]
}

/// Exact mirror of the private `cube_indices()` function in lib.rs.
fn cube_indices() -> [u16; 36] {
    let mut idx = [0u16; 36];
    for face in 0..6u16 {
        let b = face * 4;
        let o = (face * 6) as usize;
        idx[o] = b;
        idx[o + 1] = b + 1;
        idx[o + 2] = b + 2;
        idx[o + 3] = b;
        idx[o + 4] = b + 2;
        idx[o + 5] = b + 3;
    }
    idx
}

// ── Face normal directions ────────────────────────────────────────────────────

#[test]
fn pos_x_face_normal_is_1_0_0() {
    let verts = cube_vertices();
    for i in 0..4 {
        assert_eq!(verts[i].normal, [1.0, 0.0, 0.0]);
    }
}

#[test]
fn neg_x_face_normal_is_neg1_0_0() {
    let verts = cube_vertices();
    for i in 4..8 {
        assert_eq!(verts[i].normal, [-1.0, 0.0, 0.0]);
    }
}

#[test]
fn pos_y_face_normal_is_0_1_0() {
    let verts = cube_vertices();
    for i in 8..12 {
        assert_eq!(verts[i].normal, [0.0, 1.0, 0.0]);
    }
}

#[test]
fn neg_y_face_normal_is_0_neg1_0() {
    let verts = cube_vertices();
    for i in 12..16 {
        assert_eq!(verts[i].normal, [0.0, -1.0, 0.0]);
    }
}

#[test]
fn pos_z_face_normal_is_0_0_1() {
    let verts = cube_vertices();
    for i in 16..20 {
        assert_eq!(verts[i].normal, [0.0, 0.0, 1.0]);
    }
}

#[test]
fn neg_z_face_normal_is_0_0_neg1() {
    let verts = cube_vertices();
    for i in 20..24 {
        assert_eq!(verts[i].normal, [0.0, 0.0, -1.0]);
    }
}

#[test]
fn all_normals_are_unit_length() {
    let verts = cube_vertices();
    for (i, vert) in verts.iter().enumerate() {
        let n = vert.normal;
        let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        assert!((len_sq - 1.0).abs() < 1e-6, "vertex {} |n|²={}", i, len_sq);
    }
}

// ── Face vertex positions ─────────────────────────────────────────────────────

#[test]
fn pos_x_face_all_x_coords_are_half() {
    let verts = cube_vertices();
    for i in 0..4 {
        assert!((verts[i].position[0] - 0.5).abs() < 1e-6, "vertex {}", i);
    }
}

#[test]
fn neg_x_face_all_x_coords_are_neg_half() {
    let verts = cube_vertices();
    for i in 4..8 {
        assert!((verts[i].position[0] - (-0.5)).abs() < 1e-6, "vertex {}", i);
    }
}

#[test]
fn pos_y_face_all_y_coords_are_half() {
    let verts = cube_vertices();
    for i in 8..12 {
        assert!((verts[i].position[1] - 0.5).abs() < 1e-6, "vertex {}", i);
    }
}

#[test]
fn neg_y_face_all_y_coords_are_neg_half() {
    let verts = cube_vertices();
    for i in 12..16 {
        assert!((verts[i].position[1] - (-0.5)).abs() < 1e-6, "vertex {}", i);
    }
}

#[test]
fn pos_z_face_all_z_coords_are_half() {
    let verts = cube_vertices();
    for i in 16..20 {
        assert!((verts[i].position[2] - 0.5).abs() < 1e-6, "vertex {}", i);
    }
}

#[test]
fn neg_z_face_all_z_coords_are_neg_half() {
    let verts = cube_vertices();
    for i in 20..24 {
        assert!((verts[i].position[2] - (-0.5)).abs() < 1e-6, "vertex {}", i);
    }
}

// ── Face colors ───────────────────────────────────────────────────────────────

#[test]
fn pos_x_face_color() {
    let verts = cube_vertices();
    for i in 0..4 {
        assert_eq!(verts[i].color, [1.0, 0.25, 0.25], "vertex {}", i);
    }
}

#[test]
fn neg_x_face_color() {
    let verts = cube_vertices();
    for i in 4..8 {
        assert_eq!(verts[i].color, [0.25, 1.0, 1.0], "vertex {}", i);
    }
}

#[test]
fn pos_y_face_color() {
    let verts = cube_vertices();
    for i in 8..12 {
        assert_eq!(verts[i].color, [0.25, 1.0, 0.25], "vertex {}", i);
    }
}

#[test]
fn neg_y_face_color() {
    let verts = cube_vertices();
    for i in 12..16 {
        assert_eq!(verts[i].color, [1.0, 0.25, 1.0], "vertex {}", i);
    }
}

#[test]
fn pos_z_face_color() {
    let verts = cube_vertices();
    for i in 16..20 {
        assert_eq!(verts[i].color, [0.3, 0.5, 1.0], "vertex {}", i);
    }
}

#[test]
fn neg_z_face_color() {
    let verts = cube_vertices();
    for i in 20..24 {
        assert_eq!(verts[i].color, [1.0, 1.0, 0.25], "vertex {}", i);
    }
}

// ── Index buffer structure ────────────────────────────────────────────────────

#[test]
fn index_buffer_has_36_entries() {
    let idx = cube_indices();
    assert_eq!(idx.len(), 36);
}

#[test]
fn face_0_indices_are_0_1_2_and_0_2_3() {
    let idx = cube_indices();
    assert_eq!(idx[0..6], [0, 1, 2, 0, 2, 3]);
}

#[test]
fn face_1_indices_are_4_5_6_and_4_6_7() {
    let idx = cube_indices();
    assert_eq!(idx[6..12], [4, 5, 6, 4, 6, 7]);
}

#[test]
fn face_5_indices_are_20_21_22_and_20_22_23() {
    let idx = cube_indices();
    assert_eq!(idx[30..36], [20, 21, 22, 20, 22, 23]);
}

#[test]
fn all_indices_are_below_24() {
    let idx = cube_indices();
    for (i, &index) in idx.iter().enumerate() {
        assert!(
            (index as usize) < 24,
            "index {} at position {} is out of range",
            index,
            i
        );
    }
}

#[test]
fn each_face_quad_split_shares_diagonal() {
    // Each quad is split as: (b, b+1, b+2) and (b, b+2, b+3).
    // The diagonal is from b to b+2.
    let idx = cube_indices();
    for face in 0..6usize {
        let o = face * 6;
        let b = (face * 4) as u16;
        // First tri: b, b+1, b+2
        assert_eq!(idx[o], b, "face {} tri0[0]", face);
        assert_eq!(idx[o + 1], b + 1, "face {} tri0[1]", face);
        assert_eq!(idx[o + 2], b + 2, "face {} tri0[2]", face);
        // Second tri: b, b+2, b+3 (shares diagonal b → b+2)
        assert_eq!(idx[o + 3], b, "face {} tri1[0]", face);
        assert_eq!(idx[o + 4], b + 2, "face {} tri1[1]", face);
        assert_eq!(idx[o + 5], b + 3, "face {} tri1[2]", face);
    }
}

#[test]
fn no_degenerate_triangles() {
    // A triangle is degenerate if any two of its three indices are equal.
    let idx = cube_indices();
    for tri in 0..12usize {
        let a = idx[tri * 3];
        let b = idx[tri * 3 + 1];
        let c = idx[tri * 3 + 2];
        assert_ne!(a, b, "tri {} degenerate: a==b", tri);
        assert_ne!(b, c, "tri {} degenerate: b==c", tri);
        assert_ne!(a, c, "tri {} degenerate: a==c", tri);
    }
}

// ── All positions within [-0.5, 0.5]³ ────────────────────────────────────────

#[test]
fn all_positions_within_unit_half_extents() {
    let verts = cube_vertices();
    for (i, vert) in verts.iter().enumerate() {
        for (j, &coord) in vert.position.iter().enumerate() {
            assert!(
                coord >= -0.5 && coord <= 0.5,
                "vertex {} coord[{}] = {} outside [-0.5, 0.5]",
                i,
                j,
                coord
            );
        }
    }
}

// ── Uniqueness of normals per-face ────────────────────────────────────────────

#[test]
fn six_faces_have_six_distinct_normals() {
    let verts = cube_vertices();
    // Collect the normal from vert 0 of each face (each 4-vert block).
    let face_normals: Vec<[u32; 3]> = (0..6)
        .map(|f| {
            let n = verts[f * 4].normal;
            // Map to integers to avoid fp equality issues
            [n[0].to_bits(), n[1].to_bits(), n[2].to_bits()]
        })
        .collect();
    for i in 0..6 {
        for j in (i + 1)..6 {
            assert_ne!(
                face_normals[i], face_normals[j],
                "faces {} and {} share normal",
                i, j
            );
        }
    }
}
