use glam::Vec3;
use helio_pass_sdf::edit_bvh::Aabb;

fn aabb(min: [f32; 3], max: [f32; 3]) -> Aabb {
    Aabb::new(Vec3::from(min), Vec3::from(max))
}

// ──────────────────────── Construction ───────────────────────────────────────

#[test]
fn new_stores_min() {
    let a = aabb([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
    assert_eq!(a.min, Vec3::new(-1.0, -2.0, -3.0));
}

#[test]
fn new_stores_max() {
    let a = aabb([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
    assert_eq!(a.max, Vec3::new(1.0, 2.0, 3.0));
}

#[test]
fn from_point_radius_min_correct() {
    let a = Aabb::from_point_radius(Vec3::new(0.0, 0.0, 0.0), 3.0);
    assert_eq!(a.min, Vec3::splat(-3.0));
}

#[test]
fn from_point_radius_max_correct() {
    let a = Aabb::from_point_radius(Vec3::new(0.0, 0.0, 0.0), 3.0);
    assert_eq!(a.max, Vec3::splat(3.0));
}

#[test]
fn from_point_radius_nonzero_center() {
    let c = Vec3::new(1.0, 2.0, 3.0);
    let a = Aabb::from_point_radius(c, 1.0);
    assert_eq!(a.min, Vec3::new(0.0, 1.0, 2.0));
    assert_eq!(a.max, Vec3::new(2.0, 3.0, 4.0));
}

#[test]
fn from_point_radius_zero_radius_is_point() {
    let a = Aabb::from_point_radius(Vec3::new(5.0, 5.0, 5.0), 0.0);
    assert_eq!(a.min, a.max);
}

// ──────────────────────── overlaps ───────────────────────────────────────────

#[test]
fn overlaps_same_box_true() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!(a.overlaps(&a));
}

#[test]
fn overlaps_touching_face_inclusive() {
    // Boxes share the face x=1.
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = aabb([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    assert!(a.overlaps(&b));
    assert!(b.overlaps(&a));
}

#[test]
fn overlaps_separated_x_false() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = aabb([2.0, 0.0, 0.0], [3.0, 1.0, 1.0]);
    assert!(!a.overlaps(&b));
}

#[test]
fn overlaps_separated_y_false() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = aabb([0.0, 2.0, 0.0], [1.0, 3.0, 1.0]);
    assert!(!a.overlaps(&b));
}

#[test]
fn overlaps_separated_z_false() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = aabb([0.0, 0.0, 2.0], [1.0, 1.0, 3.0]);
    assert!(!a.overlaps(&b));
}

#[test]
fn overlaps_partial_overlap_true() {
    let a = aabb([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let b = aabb([1.0, 1.0, 1.0], [3.0, 3.0, 3.0]);
    assert!(a.overlaps(&b));
}

#[test]
fn overlaps_fully_contained_true() {
    let outer = aabb([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]);
    let inner = aabb([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
    assert!(outer.overlaps(&inner));
    assert!(inner.overlaps(&outer));
}

#[test]
fn overlaps_just_missed_x() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = aabb([1.001, 0.0, 0.0], [2.0, 1.0, 1.0]);
    assert!(!a.overlaps(&b));
}

// ──────────────────────── union ──────────────────────────────────────────────

#[test]
fn union_contains_both_mins() {
    let a = aabb([-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = aabb([0.0, -1.0, 0.0], [2.0, 2.0, 1.0]);
    let u = a.union(b);
    assert_eq!(u.min, Vec3::new(-1.0, -1.0, 0.0));
}

#[test]
fn union_contains_both_maxs() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = aabb([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
    let u = a.union(b);
    assert_eq!(u.max, Vec3::new(2.0, 3.0, 4.0));
}

#[test]
fn union_is_commutative() {
    let a = aabb([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
    let b = aabb([0.0, 0.0, 0.0], [4.0, 4.0, 4.0]);
    let ab = a.union(b);
    let ba = b.union(a);
    assert_eq!(ab.min, ba.min);
    assert_eq!(ab.max, ba.max);
}

#[test]
fn union_associative() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = aabb([1.0, 1.0, 1.0], [2.0, 2.0, 2.0]);
    let c = aabb([-1.0, -1.0, -1.0], [0.5, 0.5, 0.5]);
    let left = a.union(b).union(c);
    let right = a.union(b.union(c));
    assert_eq!(left.min, right.min);
    assert_eq!(left.max, right.max);
}

#[test]
fn union_idempotent() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let u = a.union(a);
    assert_eq!(u.min, a.min);
    assert_eq!(u.max, a.max);
}

// ──────────────────────── half_area ──────────────────────────────────────────

#[test]
fn half_area_unit_cube() {
    // Unit cube: d=(1,1,1) → half_area = 1*1 + 1*1 + 1*1 = 3
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!((a.half_area() - 3.0).abs() < 1e-6);
}

#[test]
fn half_area_rectangular_box() {
    // d=(2,3,4) → half_area = 2*3 + 3*4 + 4*2 = 6 + 12 + 8 = 26
    let a = aabb([0.0, 0.0, 0.0], [2.0, 3.0, 4.0]);
    assert!((a.half_area() - 26.0).abs() < 1e-6);
}

#[test]
fn half_area_zero_volume_line() {
    // d=(0,0,5) → half_area = 0
    let a = aabb([0.0, 0.0, 0.0], [0.0, 0.0, 5.0]);
    assert!((a.half_area() - 0.0).abs() < 1e-6);
}

#[test]
fn half_area_flat_plane() {
    // d=(2,0,4) → half_area = 2*0 + 0*4 + 4*2 = 8
    let a = aabb([0.0, 0.0, 0.0], [2.0, 0.0, 4.0]);
    assert!((a.half_area() - 8.0).abs() < 1e-6);
}

// ──────────────────────── expand ─────────────────────────────────────────────

#[test]
fn expand_by_zero_unchanged() {
    let a = aabb([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
    let e = a.expand(0.0);
    assert_eq!(e.min, a.min);
    assert_eq!(e.max, a.max);
}

#[test]
fn expand_by_one_shrinks_min_grows_max() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let e = a.expand(0.5);
    assert_eq!(e.min, Vec3::splat(-0.5));
    assert_eq!(e.max, Vec3::splat(1.5));
}

#[test]
fn expand_is_symmetric() {
    let a = aabb([0.0, 0.0, 0.0], [2.0, 4.0, 6.0]);
    let e = a.expand(1.0);
    assert_eq!(e.min, Vec3::new(-1.0, -1.0, -1.0));
    assert_eq!(e.max, Vec3::new(3.0, 5.0, 7.0));
}

#[test]
fn aabb_copy_clone() {
    let a = aabb([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let b = a;   // Copy
    let c = a.clone();
    assert_eq!(b.min, a.min);
    assert_eq!(c.max, a.max);
}
