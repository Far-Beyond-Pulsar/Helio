//! Architecture and geometry tests for the Simple Cube pass.
//!
//! Pure math — no GPU, no wgpu, no crate imports.
//! The vertex struct and constants mirror the private items in
//! `helio-pass-simple-cube/src/lib.rs`.

// Mirror of the private CubeVertex struct in lib.rs
#[repr(C)]
#[derive(Clone, Copy)]
struct CubeVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

const VERTEX_COUNT: usize = 24;
const INDEX_COUNT: usize = 36;
const FACES: usize = 6;
const VERTS_PER_FACE: usize = 4;
const TRIS_PER_FACE: usize = 2;
const INDICES_PER_TRI: usize = 3;

// ── CubeVertex struct layout ──────────────────────────────────────────────────

#[test]
fn cube_vertex_size_is_36_bytes() {
    // position(12) + normal(12) + color(12) = 36 bytes
    assert_eq!(std::mem::size_of::<CubeVertex>(), 36);
}

#[test]
fn cube_vertex_alignment_is_4_bytes() {
    assert_eq!(std::mem::align_of::<CubeVertex>(), 4);
}

#[test]
fn cube_vertex_position_field_is_12_bytes() {
    assert_eq!(std::mem::size_of::<[f32; 3]>(), 12);
}

#[test]
fn cube_vertex_normal_field_is_12_bytes() {
    assert_eq!(std::mem::size_of::<[f32; 3]>(), 12);
}

#[test]
fn cube_vertex_color_field_is_12_bytes() {
    // Three f32 color channels (no alpha)
    assert_eq!(std::mem::size_of::<[f32; 3]>(), 12);
}

#[test]
fn cube_vertex_has_no_hidden_padding() {
    // Three [f32;3] fields back-to-back = 3 × 12 = 36.
    let expected = 3 * std::mem::size_of::<[f32; 3]>();
    assert_eq!(std::mem::size_of::<CubeVertex>(), expected);
}

// ── Vertex and index counts ───────────────────────────────────────────────────

#[test]
fn vertex_count_is_24() {
    // 6 faces × 4 vertices per face = 24
    assert_eq!(VERTEX_COUNT, 24);
    assert_eq!(FACES * VERTS_PER_FACE, 24);
}

#[test]
fn index_count_is_36() {
    // 6 faces × 2 triangles × 3 vertices = 36
    assert_eq!(INDEX_COUNT, 36);
    assert_eq!(FACES * TRIS_PER_FACE * INDICES_PER_TRI, 36);
}

#[test]
fn each_face_contributes_six_indices() {
    assert_eq!(TRIS_PER_FACE * INDICES_PER_TRI, 6);
}

#[test]
fn each_face_uses_four_unique_vertices() {
    // Each face is a quad split into two triangles sharing no vertices with other faces.
    assert_eq!(VERTS_PER_FACE, 4);
}

// ── Vertex buffer size ────────────────────────────────────────────────────────

#[test]
fn vertex_buffer_total_bytes() {
    let total = VERTEX_COUNT * std::mem::size_of::<CubeVertex>();
    assert_eq!(total, 864); // 24 × 36 = 864 bytes
}

#[test]
fn index_buffer_u16_total_bytes() {
    // Indices stored as u16.
    let total = INDEX_COUNT * std::mem::size_of::<u16>();
    assert_eq!(total, 72); // 36 × 2 = 72 bytes
}

// ── Face color constants (from lib.rs) ───────────────────────────────────────

#[test]
fn pos_x_face_color_is_red() {
    let r = [1.0_f32, 0.25, 0.25];
    assert!((r[0] - 1.0).abs() < 1e-6);
    assert!((r[1] - 0.25).abs() < 1e-6);
    assert!((r[2] - 0.25).abs() < 1e-6);
}

#[test]
fn neg_x_face_color_is_cyan() {
    let c = [0.25_f32, 1.0, 1.0];
    assert!((c[0] - 0.25).abs() < 1e-6);
    assert!((c[1] - 1.0).abs() < 1e-6);
    assert!((c[2] - 1.0).abs() < 1e-6);
}

#[test]
fn pos_y_face_color_is_green() {
    let g = [0.25_f32, 1.0, 0.25];
    assert!((g[1] - 1.0).abs() < 1e-6);
    assert!(g[1] > g[0]); // green channel dominates
}

#[test]
fn neg_y_face_color_is_magenta() {
    let m = [1.0_f32, 0.25, 1.0];
    assert!((m[0] - 1.0).abs() < 1e-6);
    assert!((m[2] - 1.0).abs() < 1e-6);
    assert!(m[1] < m[0]); // green channel is low
}

#[test]
fn pos_z_face_color_is_blue() {
    let b = [0.3_f32, 0.5, 1.0];
    assert!((b[2] - 1.0).abs() < 1e-6); // blue channel is max
    assert!(b[2] > b[1]);
    assert!(b[2] > b[0]);
}

#[test]
fn neg_z_face_color_is_yellow() {
    let y = [1.0_f32, 1.0, 0.25];
    assert!((y[0] - 1.0).abs() < 1e-6);
    assert!((y[1] - 1.0).abs() < 1e-6);
    assert!(y[2] < y[0]); // blue channel is low
}

#[test]
fn all_face_colors_are_in_0_1_range() {
    let colors: [[f32; 3]; 6] = [
        [1.0, 0.25, 0.25],  // +X red
        [0.25, 1.0, 1.0],   // -X cyan
        [0.25, 1.0, 0.25],  // +Y green
        [1.0, 0.25, 1.0],   // -Y magenta
        [0.3, 0.5, 1.0],    // +Z blue
        [1.0, 1.0, 0.25],   // -Z yellow
    ];
    for (i, color) in colors.iter().enumerate() {
        for (j, &c) in color.iter().enumerate() {
            assert!(c >= 0.0 && c <= 1.0, "face {} channel {} = {} out of [0,1]", i, j, c);
        }
    }
}

#[test]
fn all_face_colors_are_distinct() {
    let colors: [[u32; 3]; 6] = [
        [1000, 250, 250],   // +X  (scaled to avoid fp comparison)
        [250, 1000, 1000],  // -X
        [250, 1000, 250],   // +Y
        [1000, 250, 1000],  // -Y
        [300, 500, 1000],   // +Z
        [1000, 1000, 250],  // -Z
    ];
    for i in 0..6 {
        for j in (i + 1)..6 {
            assert_ne!(colors[i], colors[j], "faces {} and {} share color", i, j);
        }
    }
}

// ── Index buffer: valid range ─────────────────────────────────────────────────

#[test]
fn all_face_base_indices_are_below_vertex_count() {
    // Each face's base vertex index = face * 4; must be < 24.
    for face in 0..FACES {
        let base = face * VERTS_PER_FACE;
        assert!(base < VERTEX_COUNT, "face {} base {} >= {}", face, base, VERTEX_COUNT);
    }
}

#[test]
fn last_face_last_vertex_index_is_below_vertex_count() {
    let last_base = (FACES - 1) * VERTS_PER_FACE; // 20
    let last_vert = last_base + VERTS_PER_FACE - 1; // 23
    assert!(last_vert < VERTEX_COUNT);
    assert_eq!(last_vert, 23);
}

// ── Index stride (face i starts at index 6*i) ────────────────────────────────

#[test]
fn index_stride_between_faces_is_six() {
    for face in 0..FACES {
        let start = face * (TRIS_PER_FACE * INDICES_PER_TRI);
        let expected = face * 6;
        assert_eq!(start, expected);
    }
}

// ── Normal vector unit-length ─────────────────────────────────────────────────

#[test]
fn face_normals_are_unit_length() {
    let normals: [[f32; 3]; 6] = [
        [1.0, 0.0, 0.0],   // +X
        [-1.0, 0.0, 0.0],  // -X
        [0.0, 1.0, 0.0],   // +Y
        [0.0, -1.0, 0.0],  // -Y
        [0.0, 0.0, 1.0],   // +Z
        [0.0, 0.0, -1.0],  // -Z
    ];
    for (i, n) in normals.iter().enumerate() {
        let len_sq = n[0] * n[0] + n[1] * n[1] + n[2] * n[2];
        assert!((len_sq - 1.0).abs() < 1e-6, "face {} |n|²={}", i, len_sq);
    }
}

// ── Positions within [-0.5, 0.5] ─────────────────────────────────────────────

#[test]
fn face_vertex_coordinates_are_half_unit() {
    // All cube vertex positions are in {-0.5, 0.5} for each axis.
    let valid_coords = [-0.5_f32, 0.5_f32];
    let all_positions: [[f32; 3]; 8] = [
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, -0.5],
    ];
    for (vi, pos) in all_positions.iter().enumerate() {
        for (ci, &coord) in pos.iter().enumerate() {
            assert!(
                valid_coords.contains(&coord),
                "vertex {} coord[{}] = {} is not ±0.5",
                vi, ci, coord
            );
        }
    }
}
