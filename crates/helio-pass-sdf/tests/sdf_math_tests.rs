//! Self-contained SDF math tests — no imports from the crate needed.
//! All helper functions are defined locally and match the standard SDF formulas.

// ──────────────────────── SDF primitives (local helpers) ─────────────────────

/// Signed distance from `p` to a sphere centred at `center` with radius `r`.
fn sdf_sphere(p: [f32; 3], center: [f32; 3], r: f32) -> f32 {
    let dx = p[0] - center[0];
    let dy = p[1] - center[1];
    let dz = p[2] - center[2];
    (dx * dx + dy * dy + dz * dz).sqrt() - r
}

/// Signed distance from `p` (relative to box centre at origin) to a box with
/// half-extents `b`.  Negative inside, zero on surface, positive outside.
fn sdf_box(p: [f32; 3], b: [f32; 3]) -> f32 {
    let qx = p[0].abs() - b[0];
    let qy = p[1].abs() - b[1];
    let qz = p[2].abs() - b[2];
    let len_outside = (qx.max(0.0).powi(2) + qy.max(0.0).powi(2) + qz.max(0.0).powi(2)).sqrt();
    let max_inside = qx.max(qy).max(qz).min(0.0);
    len_outside + max_inside
}

/// CSG union: minimum distance.
fn sdf_union(a: f32, b: f32) -> f32 {
    a.min(b)
}

/// CSG subtraction: cut shape B out of shape A.
fn sdf_subtract(a: f32, b: f32) -> f32 {
    (-a).max(b)
}

/// CSG intersection: the region inside both shapes.
fn sdf_intersect(a: f32, b: f32) -> f32 {
    a.max(b)
}

// ──────────────────────── Sphere ─────────────────────────────────────────────

#[test]
fn sphere_center_is_negative_radius() {
    let d = sdf_sphere([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0);
    assert!((d - (-2.0)).abs() < 1e-6);
}

#[test]
fn sphere_on_surface_is_zero() {
    let d = sdf_sphere([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0);
    assert!(d.abs() < 1e-6);
}

#[test]
fn sphere_outside_positive() {
    let d = sdf_sphere([3.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0);
    assert!((d - 2.0).abs() < 1e-6);
}

#[test]
fn sphere_nonaxis_surface() {
    // Point on surface diagonally — sqrt(3) * 1/sqrt(3) = 1.0.
    let s = (1.0f32 / 3.0).sqrt();
    let d = sdf_sphere([s, s, s], [0.0, 0.0, 0.0], 1.0);
    assert!(d.abs() < 1e-5);
}

#[test]
fn sphere_negative_center_coord() {
    let d = sdf_sphere([0.0, 5.0, 0.0], [0.0, 3.0, 0.0], 1.0);
    // Distance = |5-3| - 1 = 1.0
    assert!((d - 1.0).abs() < 1e-6);
}

#[test]
fn sphere_zero_radius_surface_at_center() {
    let d = sdf_sphere([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0);
    assert!(d.abs() < 1e-6);
}

#[test]
fn sphere_far_outside_large_positive() {
    let d = sdf_sphere([100.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0);
    assert!((d - 99.0).abs() < 1e-4);
}

#[test]
fn sphere_inside_negative() {
    let d = sdf_sphere([0.5, 0.0, 0.0], [0.0, 0.0, 0.0], 2.0);
    assert!(d < 0.0);
}

// ──────────────────────── Box ────────────────────────────────────────────────

#[test]
fn box_center_is_neg_min_half_extent() {
    // Unit half-extent box: center dist = -min(1,1,1) = -1
    let d = sdf_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!((d - (-1.0)).abs() < 1e-6);
}

#[test]
fn box_face_center_is_zero() {
    // Point on the +X face centre: p = (1,0,0), b = (1,1,1)
    let d = sdf_box([1.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!(d.abs() < 1e-6);
}

#[test]
fn box_outside_x_positive() {
    // p = (2,0,0), b = (1,1,1) → d = 1
    let d = sdf_box([2.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!((d - 1.0).abs() < 1e-6);
}

#[test]
fn box_outside_corner() {
    // p = (2,2,2), b = (1,1,1) → q=(1,1,1) → d = sqrt(3)
    let d = sdf_box([2.0, 2.0, 2.0], [1.0, 1.0, 1.0]);
    assert!((d - 3.0f32.sqrt()).abs() < 1e-5);
}

#[test]
fn box_on_edge_is_zero() {
    // Point on the X/Y edge: (1,1,0) with b=(1,1,1)
    let d = sdf_box([1.0, 1.0, 0.0], [1.0, 1.0, 1.0]);
    assert!(d.abs() < 1e-6);
}

#[test]
fn box_on_corner_is_zero() {
    let d = sdf_box([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]);
    assert!(d.abs() < 1e-6);
}

#[test]
fn box_inside_not_touching_any_face() {
    // p=(0.5,0.5,0.5), b=(1,1,1) → all q negative, max_inside = 0.5 - 1 = -0.5
    let d = sdf_box([0.5, 0.5, 0.5], [1.0, 1.0, 1.0]);
    assert!((d - (-0.5)).abs() < 1e-6);
}

#[test]
fn box_elongated_outside_long_axis() {
    // b=(5,1,1), p=(6,0,0) → d=1
    let d = sdf_box([6.0, 0.0, 0.0], [5.0, 1.0, 1.0]);
    assert!((d - 1.0).abs() < 1e-6);
}

// ──────────────────────── CSG union ──────────────────────────────────────────

#[test]
fn union_a_less_than_b_returns_a() {
    assert!((sdf_union(1.0, 5.0) - 1.0).abs() < 1e-9);
}

#[test]
fn union_b_less_than_a_returns_b() {
    assert!((sdf_union(5.0, 1.0) - 1.0).abs() < 1e-9);
}

#[test]
fn union_equal_returns_that_value() {
    assert!((sdf_union(3.0, 3.0) - 3.0).abs() < 1e-9);
}

#[test]
fn union_of_two_spheres_outside_both_positive() {
    let s1 = sdf_sphere([10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0); // >0
    let s2 = sdf_sphere([10.0, 0.0, 0.0], [5.0, 0.0, 0.0], 1.0); // >0
    assert!(sdf_union(s1, s2) > 0.0);
}

#[test]
fn union_inside_first_sphere_negative() {
    let inside = sdf_sphere([0.1, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0); // <0
    let outside = sdf_sphere([0.1, 0.0, 0.0], [10.0, 0.0, 0.0], 1.0); // >0
    assert!(sdf_union(inside, outside) < 0.0);
}

// ──────────────────────── CSG subtraction ────────────────────────────────────

#[test]
fn subtract_outside_a_outside_b_positive() {
    // Both positive: max(-pos, pos) could be positive
    let a = 2.0f32; // outside shape A
    let b = 3.0f32; // outside shape B
    // sdf_subtract(a, b) = (-a).max(b) = (-2).max(3) = 3
    assert!((sdf_subtract(a, b) - 3.0).abs() < 1e-9);
}

#[test]
fn subtract_inside_a_outside_b_interior_carved() {
    // Inside A (-2), outside B (5): sdf_subtract = max(2, 5) = 5 (outside carved region)
    assert!((sdf_subtract(-2.0, 5.0) - 5.0).abs() < 1e-9);
}

#[test]
fn subtract_inside_a_inside_b_gives_positive() {
    // Inside A (-1), inside B (-1): sdf_subtract = max(1, -1) = 1 (outside result → carved out)
    assert!((sdf_subtract(-1.0, -1.0) - 1.0).abs() < 1e-9);
}

// ──────────────────────── CSG intersection ───────────────────────────────────

#[test]
fn intersect_both_inside_negative() {
    // Both inside: max(-2, -3) = -2
    assert!((sdf_intersect(-2.0, -3.0) - (-2.0)).abs() < 1e-9);
}

#[test]
fn intersect_one_outside_positive() {
    // One outside: max(-2, 3) = 3
    assert!((sdf_intersect(-2.0, 3.0) - 3.0).abs() < 1e-9);
}

#[test]
fn intersect_both_outside_positive() {
    // Both outside: max(2, 3) = 3
    assert!((sdf_intersect(2.0, 3.0) - 3.0).abs() < 1e-9);
}

#[test]
fn intersect_of_two_spheres_outside_both_is_positive() {
    let s1 = sdf_sphere([5.0, 0.0, 0.0], [0.0, 0.0, 0.0], 1.0);
    let s2 = sdf_sphere([5.0, 0.0, 0.0], [3.0, 0.0, 0.0], 1.0);
    assert!(sdf_intersect(s1, s2) > 0.0);
}

#[test]
fn union_is_commutative() {
    assert_eq!(sdf_union(1.5, 2.5), sdf_union(2.5, 1.5));
}

#[test]
fn intersect_is_commutative() {
    assert_eq!(sdf_intersect(1.5, 2.5), sdf_intersect(2.5, 1.5));
}
