use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use helio_pass_sdf::{
    edit_list::{BooleanOp, GpuSdfEdit, SdfEdit, SdfEditList},
    primitives::{SdfShapeParams, SdfShapeType},
};
use std::mem;

// ──────────────────────── SdfShapeType discriminants ────────────────────────

#[test]
fn shape_type_sphere_discriminant_is_0() {
    assert_eq!(SdfShapeType::Sphere as u32, 0);
}

#[test]
fn shape_type_cube_discriminant_is_1() {
    assert_eq!(SdfShapeType::Cube as u32, 1);
}

#[test]
fn shape_type_capsule_discriminant_is_2() {
    assert_eq!(SdfShapeType::Capsule as u32, 2);
}

#[test]
fn shape_type_torus_discriminant_is_3() {
    assert_eq!(SdfShapeType::Torus as u32, 3);
}

#[test]
fn shape_type_cylinder_discriminant_is_4() {
    assert_eq!(SdfShapeType::Cylinder as u32, 4);
}

#[test]
fn shape_type_eq() {
    assert_eq!(SdfShapeType::Sphere, SdfShapeType::Sphere);
    assert_ne!(SdfShapeType::Sphere, SdfShapeType::Cube);
}

#[test]
fn shape_type_copy_clone() {
    let a = SdfShapeType::Torus;
    let b = a; // Copy
    let c = a.clone(); // Clone
    assert_eq!(a, b);
    assert_eq!(a, c);
}

#[test]
fn shape_type_debug_nonempty() {
    let s = format!("{:?}", SdfShapeType::Cylinder);
    assert!(!s.is_empty());
}

// ──────────────────────── SdfShapeParams factory methods ─────────────────────

#[test]
fn sphere_params_stores_radius() {
    let p = SdfShapeParams::sphere(5.0);
    assert_eq!(p.param0, 5.0);
}

#[test]
fn sphere_params_unused_fields_are_zero() {
    let p = SdfShapeParams::sphere(3.0);
    assert_eq!(p.param1, 0.0);
    assert_eq!(p.param2, 0.0);
    assert_eq!(p.param3, 0.0);
}

#[test]
fn sphere_zero_radius() {
    let p = SdfShapeParams::sphere(0.0);
    assert_eq!(p.param0, 0.0);
}

#[test]
fn sphere_large_radius() {
    let p = SdfShapeParams::sphere(1_000_000.0);
    assert_eq!(p.param0, 1_000_000.0);
}

#[test]
fn sphere_negative_radius_stored() {
    // No clamping is enforced; the struct is a data carrier.
    let p = SdfShapeParams::sphere(-1.0);
    assert_eq!(p.param0, -1.0);
}

#[test]
fn cube_params_stores_half_extents() {
    let p = SdfShapeParams::cube(1.0, 2.0, 3.0);
    assert_eq!(p.param0, 1.0);
    assert_eq!(p.param1, 2.0);
    assert_eq!(p.param2, 3.0);
}

#[test]
fn cube_params_w_is_zero() {
    let p = SdfShapeParams::cube(1.0, 2.0, 3.0);
    assert_eq!(p.param3, 0.0);
}

#[test]
fn cube_params_uniform() {
    let p = SdfShapeParams::cube(4.0, 4.0, 4.0);
    assert_eq!(p.param0, p.param1);
    assert_eq!(p.param1, p.param2);
}

#[test]
fn capsule_params_stores_radius_and_half_height() {
    let p = SdfShapeParams::capsule(0.5, 2.0);
    assert_eq!(p.param0, 0.5);
    assert_eq!(p.param1, 2.0);
}

#[test]
fn capsule_params_unused_are_zero() {
    let p = SdfShapeParams::capsule(1.0, 1.0);
    assert_eq!(p.param2, 0.0);
    assert_eq!(p.param3, 0.0);
}

#[test]
fn torus_params_stores_major_and_minor_radii() {
    let p = SdfShapeParams::torus(3.0, 1.0);
    assert_eq!(p.param0, 3.0);
    assert_eq!(p.param1, 1.0);
}

#[test]
fn torus_params_unused_are_zero() {
    let p = SdfShapeParams::torus(3.0, 1.0);
    assert_eq!(p.param2, 0.0);
    assert_eq!(p.param3, 0.0);
}

#[test]
fn cylinder_params_stores_radius_and_half_height() {
    let p = SdfShapeParams::cylinder(1.5, 3.0);
    assert_eq!(p.param0, 1.5);
    assert_eq!(p.param1, 3.0);
}

#[test]
fn cylinder_params_unused_are_zero() {
    let p = SdfShapeParams::cylinder(1.5, 3.0);
    assert_eq!(p.param2, 0.0);
    assert_eq!(p.param3, 0.0);
}

// ──────────────────────── Layout / POD ───────────────────────────────────────

#[test]
fn shape_params_size_is_16() {
    assert_eq!(mem::size_of::<SdfShapeParams>(), 16);
}

#[test]
fn shape_params_align_is_4() {
    assert_eq!(mem::align_of::<SdfShapeParams>(), 4);
}

#[test]
fn shape_params_is_pod_cast() {
    let p = SdfShapeParams::sphere(7.0);
    let bytes: &[u8] = bytemuck::bytes_of(&p);
    assert_eq!(bytes.len(), 16);
}

#[test]
fn shape_params_zeroable_all_zero() {
    let z: SdfShapeParams = Zeroable::zeroed();
    assert_eq!(z.param0, 0.0);
    assert_eq!(z.param1, 0.0);
    assert_eq!(z.param2, 0.0);
    assert_eq!(z.param3, 0.0);
}

#[test]
fn shape_params_cast_roundtrip() {
    let p = SdfShapeParams::cube(1.0, 2.0, 3.0);
    let bytes: &[u8] = bytemuck::bytes_of(&p);
    let back: &SdfShapeParams = bytemuck::from_bytes(bytes);
    assert_eq!(back.param0, 1.0);
    assert_eq!(back.param1, 2.0);
    assert_eq!(back.param2, 3.0);
}

#[test]
fn shape_params_copy_clone() {
    let a = SdfShapeParams::cylinder(2.0, 5.0);
    let b = a;
    let c = a.clone();
    assert_eq!(b.param0, 2.0);
    assert_eq!(c.param1, 5.0);
}

// ──────────────────────── BooleanOp ──────────────────────────────────────────

#[test]
fn boolean_op_union_discriminant_is_0() {
    assert_eq!(BooleanOp::Union as u32, 0);
}

#[test]
fn boolean_op_subtraction_discriminant_is_1() {
    assert_eq!(BooleanOp::Subtraction as u32, 1);
}

#[test]
fn boolean_op_intersection_discriminant_is_2() {
    assert_eq!(BooleanOp::Intersection as u32, 2);
}

// ──────────────────────── SdfEditList ────────────────────────────────────────

fn make_sphere_edit() -> SdfEdit {
    SdfEdit {
        shape: SdfShapeType::Sphere,
        op: BooleanOp::Union,
        transform: Mat4::IDENTITY,
        params: SdfShapeParams::sphere(1.0),
        blend_radius: 0.0,
    }
}

#[test]
fn edit_list_new_is_empty() {
    let list = SdfEditList::new();
    assert!(list.is_empty());
    assert_eq!(list.len(), 0);
}

#[test]
fn edit_list_new_starts_dirty() {
    let list = SdfEditList::new();
    assert!(list.is_dirty());
}

#[test]
fn edit_list_add_returns_correct_index() {
    let mut list = SdfEditList::new();
    let i0 = list.add(make_sphere_edit());
    let i1 = list.add(make_sphere_edit());
    assert_eq!(i0, 0);
    assert_eq!(i1, 1);
}

#[test]
fn edit_list_len_tracks_add_remove() {
    let mut list = SdfEditList::new();
    list.add(make_sphere_edit());
    list.add(make_sphere_edit());
    assert_eq!(list.len(), 2);
    list.remove(0);
    assert_eq!(list.len(), 1);
}

#[test]
fn edit_list_generation_increments_on_mutate() {
    let mut list = SdfEditList::new();
    let g0 = list.generation();
    list.add(make_sphere_edit());
    assert!(list.generation() > g0);
    let g1 = list.generation();
    list.remove(0);
    assert!(list.generation() > g1);
}

#[test]
fn edit_list_flush_clears_dirty_flag() {
    let mut list = SdfEditList::new();
    list.add(make_sphere_edit());
    assert!(list.is_dirty());
    let _gpu = list.flush_gpu_data();
    assert!(!list.is_dirty());
}

#[test]
fn edit_list_flush_returns_correct_count() {
    let mut list = SdfEditList::new();
    list.add(make_sphere_edit());
    list.add(make_sphere_edit());
    let gpu = list.flush_gpu_data();
    assert_eq!(gpu.len(), 2);
}

#[test]
fn gpu_sdf_edit_size_is_96() {
    assert_eq!(mem::size_of::<GpuSdfEdit>(), 96);
}

#[test]
fn gpu_sdf_edit_to_gpu_shape_type() {
    let edit = SdfEdit {
        shape: SdfShapeType::Torus,
        op: BooleanOp::Subtraction,
        transform: Mat4::IDENTITY,
        params: SdfShapeParams::torus(3.0, 0.5),
        blend_radius: 0.1,
    };
    let gpu = edit.to_gpu();
    assert_eq!(gpu.shape_type, SdfShapeType::Torus as u32);
    assert_eq!(gpu.boolean_op, BooleanOp::Subtraction as u32);
    assert_eq!(gpu.params.param0, 3.0);
    assert_eq!(gpu.params.param1, 0.5);
}

#[test]
fn edit_list_clear_empties() {
    let mut list = SdfEditList::new();
    list.add(make_sphere_edit());
    list.add(make_sphere_edit());
    list.clear();
    assert!(list.is_empty());
}

#[test]
fn edit_list_set_updates_entry() {
    let mut list = SdfEditList::new();
    list.add(make_sphere_edit());
    let updated = SdfEdit {
        shape: SdfShapeType::Cube,
        op: BooleanOp::Intersection,
        transform: Mat4::IDENTITY,
        params: SdfShapeParams::cube(2.0, 2.0, 2.0),
        blend_radius: 0.0,
    };
    list.set(0, updated);
    assert_eq!(list.edits()[0].shape, SdfShapeType::Cube);
}
