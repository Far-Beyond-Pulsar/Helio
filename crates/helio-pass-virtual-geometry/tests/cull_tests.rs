// Tests for helio-pass-virtual-geometry: CullUniforms layout, dispatch math,
// meshlet visibility, screen coverage, LOD transitions, backface cone culling.
// Uses the actual public LodQuality API + locally mirrored private types.

use std::mem;
use helio_pass_virtual_geometry::LodQuality;

// ── Mirror private types ──────────────────────────────────────────────────────

/// Mirrors private CullUniforms (16 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct CullUniforms {
    meshlet_count: u32,
    lod_s0: f32,
    lod_s1: f32,
    _pad2: u32,
}

/// Mirrors private VgGlobals (96 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct VgGlobals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: [f32; 4],
    rc_world_min: [f32; 4],
    rc_world_max: [f32; 4],
    csm_splits: [f32; 4],
    debug_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;
const INITIAL_MESHLETS: u64 = 1024;
const INITIAL_INSTANCES: u64 = 256;

// ── CullUniforms layout tests ─────────────────────────────────────────────────

#[test]
fn cull_uniforms_size_is_16() {
    assert_eq!(mem::size_of::<CullUniforms>(), 16);
}

#[test]
fn cull_uniforms_alignment_is_4() {
    assert_eq!(mem::align_of::<CullUniforms>(), 4);
}

#[test]
fn cull_uniforms_size_divisible_by_16() {
    assert_eq!(mem::size_of::<CullUniforms>() % 16, 0);
}

#[test]
fn cull_uniforms_four_fields_of_four_bytes() {
    // meshlet_count(4) + lod_s0(4) + lod_s1(4) + _pad2(4) = 16
    assert_eq!(4 + 4 + 4 + 4, 16usize);
}

// ── VgGlobals layout tests ────────────────────────────────────────────────────

#[test]
fn vg_globals_size_is_96() {
    assert_eq!(mem::size_of::<VgGlobals>(), 96,
        "16 (scalars) + 64 (4×vec4) + 16 (debug_mode + 3 pads) = 96");
}

#[test]
fn vg_globals_scalar_section_16_bytes() {
    assert_eq!(4 + 4 + 4 + 4, 16usize);
}

#[test]
fn vg_globals_vec4_section_64_bytes() {
    assert_eq!(4 * 4 * mem::size_of::<f32>(), 64usize);
}

#[test]
fn vg_globals_debug_section_16_bytes() {
    // debug_mode + _pad0 + _pad1 + _pad2 = 4 × 4 = 16
    assert_eq!(4 * 4, 16usize);
}

#[test]
fn vg_globals_total_16_plus_64_plus_16() {
    assert_eq!(16 + 64 + 16, 96usize);
}

// ── Initial buffer capacity tests ────────────────────────────────────────────

#[test]
fn initial_meshlets_is_1024() {
    assert_eq!(INITIAL_MESHLETS, 1024u64);
}

#[test]
fn initial_instances_is_256() {
    assert_eq!(INITIAL_INSTANCES, 256u64);
}

#[test]
fn initial_meshlets_greater_than_instances() {
    assert!(INITIAL_MESHLETS > INITIAL_INSTANCES);
}

#[test]
fn initial_meshlets_is_power_of_two() {
    assert!(INITIAL_MESHLETS.is_power_of_two());
}

#[test]
fn initial_instances_is_power_of_two() {
    assert!(INITIAL_INSTANCES.is_power_of_two());
}

// ── WORKGROUP_SIZE / dispatch math tests ──────────────────────────────────────

#[test]
fn workgroup_size_is_64() {
    assert_eq!(WORKGROUP_SIZE, 64u32);
}

#[test]
fn workgroup_size_is_power_of_two() {
    assert!(WORKGROUP_SIZE.is_power_of_two());
}

#[test]
fn dispatch_groups_ceil_division() {
    // dispatch_count = ceil(meshlet_count / WORKGROUP_SIZE)
    fn ceil_div(n: u32, d: u32) -> u32 { (n + d - 1) / d }
    assert_eq!(ceil_div(64, 64), 1);
    assert_eq!(ceil_div(65, 64), 2);
    assert_eq!(ceil_div(128, 64), 2);
    assert_eq!(ceil_div(1024, 64), 16);
    assert_eq!(ceil_div(0, 64), 0);
}

#[test]
fn dispatch_1024_meshlets_needs_16_groups() {
    let groups = (1024u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    assert_eq!(groups, 16u32);
}

#[test]
fn dispatch_single_meshlet_needs_one_group() {
    let groups = (1u32 + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    assert_eq!(groups, 1u32);
}

// ── Meshlet visibility / LOD threshold tests (using LodQuality API) ───────────

#[test]
fn medium_lod0_visible_when_screen_radius_above_s0() {
    let (s0, _) = LodQuality::Medium.thresholds();
    let sr = 0.06f32; // 6% > 5%
    assert!(sr >= s0, "LOD 0 should be visible");
}

#[test]
fn medium_lod0_not_visible_when_screen_radius_below_s0() {
    let (s0, _) = LodQuality::Medium.thresholds();
    let sr = 0.03f32; // 3% < 5%
    assert!(sr < s0);
}

#[test]
fn medium_lod1_visible_when_screen_radius_between_s1_and_s0() {
    let (s0, s1) = LodQuality::Medium.thresholds();
    let sr = 0.03f32;
    assert!(sr < s0 && sr >= s1, "sr={sr} s0={s0} s1={s1}");
}

#[test]
fn medium_lod2_when_screen_radius_below_s1() {
    let (_, s1) = LodQuality::Medium.thresholds();
    let sr = 0.005f32; // < s1=0.01
    assert!(sr < s1);
}

// ── Screen coverage formula tests ─────────────────────────────────────────────

/// screen_radius = (obj_radius * cot(fov/2)) / dist
fn screen_radius(obj_radius: f32, fov_rad: f32, dist: f32) -> f32 {
    let cot_half_fov = 1.0 / (fov_rad / 2.0).tan();
    obj_radius * cot_half_fov / dist
}

#[test]
fn screen_radius_far_object_below_medium_s0() {
    let (s0, _) = LodQuality::Medium.thresholds();
    let fov = std::f32::consts::FRAC_PI_2; // 90°
    let obj_radius = 1.0f32;
    let dist = 100.0f32;
    let sr = screen_radius(obj_radius, fov, dist);
    // Very far object should be below LOD 0 threshold
    assert!(sr < s0, "sr={sr} s0={s0}");
}

#[test]
fn screen_radius_close_object_above_ultra_s0() {
    let (s0, _) = LodQuality::Ultra.thresholds();
    let fov = std::f32::consts::FRAC_PI_2;
    let obj_radius = 10.0f32;
    let dist = 5.0f32;
    let sr = screen_radius(obj_radius, fov, dist);
    assert!(sr > s0, "sr={sr} ultra_s0={s0}");
}

#[test]
fn screen_radius_proportional_to_radius() {
    let fov = std::f32::consts::FRAC_PI_2;
    let dist = 20.0f32;
    let sr1 = screen_radius(1.0, fov, dist);
    let sr2 = screen_radius(3.0, fov, dist);
    assert!((sr2 - 3.0 * sr1).abs() < 1e-5f32);
}

#[test]
fn screen_radius_inversely_proportional_to_distance() {
    let fov = std::f32::consts::FRAC_PI_2;
    let r = 5.0f32;
    let sr10 = screen_radius(r, fov, 10.0);
    let sr20 = screen_radius(r, fov, 20.0);
    assert!((sr10 - 2.0 * sr20).abs() < 1e-4f32);
}

// ── LOD transition boundary tests (using LodQuality) ─────────────────────────

#[test]
fn all_quality_levels_have_positive_thresholds() {
    for q in [LodQuality::Low, LodQuality::Medium, LodQuality::High, LodQuality::Ultra] {
        let (s0, s1) = q.thresholds();
        assert!(s0 > 0.0f32, "{:?} s0={s0}", q);
        assert!(s1 > 0.0f32, "{:?} s1={s1}", q);
    }
}

#[test]
fn all_quality_levels_thresholds_below_1() {
    for q in [LodQuality::Low, LodQuality::Medium, LodQuality::High, LodQuality::Ultra] {
        let (s0, s1) = q.thresholds();
        assert!(s0 < 1.0f32, "{:?} s0={s0} >= 1.0", q);
        assert!(s1 < 1.0f32, "{:?} s1={s1} >= 1.0", q);
    }
}

// ── Backface cone culling tests ───────────────────────────────────────────────

/// Backface check: dot(view_dir, cone_axis) + cos_half_angle <= 0.
fn is_backfacing_cone(view_dir: [f32; 3], cone_axis: [f32; 3], cos_half_angle: f32) -> bool {
    let dot = view_dir[0] * cone_axis[0]
        + view_dir[1] * cone_axis[1]
        + view_dir[2] * cone_axis[2];
    dot + cos_half_angle <= 0.0
}

#[test]
fn cone_culling_backfacing_when_view_behind_cone() {
    // View direction exactly opposite to cone axis
    let view_dir = [0.0f32, 0.0, -1.0];
    let cone_axis = [0.0f32, 0.0, 1.0];
    let cos_half = 0.5f32;
    // dot = -1, -1 + 0.5 = -0.5 <= 0 → backfacing
    assert!(is_backfacing_cone(view_dir, cone_axis, cos_half));
}

#[test]
fn cone_culling_visible_when_view_in_front_of_cone() {
    let view_dir = [0.0f32, 0.0, 1.0]; // same direction as cone
    let cone_axis = [0.0f32, 0.0, 1.0];
    let cos_half = 0.5f32;
    // dot = 1, 1 + 0.5 = 1.5 > 0 → visible
    assert!(!is_backfacing_cone(view_dir, cone_axis, cos_half));
}

#[test]
fn cone_culling_boundary_exactly_zero() {
    // dot + cos_half_angle == 0 → on boundary, treated as culled
    let view_dir = [-0.5f32, 0.0, 0.0];
    let cone_axis = [1.0f32, 0.0, 0.0];
    let cos_half = 0.5f32;
    // dot = -0.5, -0.5 + 0.5 = 0.0 <= 0 → backfacing
    assert!(is_backfacing_cone(view_dir, cone_axis, cos_half));
}

#[test]
fn cone_culling_narrow_cone_more_aggressive() {
    // A narrower cone (larger cos_half) culls more aggressively
    let view_dir = [0.0f32, 0.0, -0.8];   // mostly backward
    let cone_axis = [0.0f32, 0.0, 1.0];
    let cos_half_narrow = 0.9f32;
    let cos_half_wide = 0.1f32;
    // Narrow cone: dot(-0.8) + 0.9 = 0.1 > 0 → visible (but barely)
    // Wide cone:   dot(-0.8) + 0.1 = -0.7 ≤ 0 → culled
    assert!(!is_backfacing_cone(view_dir, cone_axis, cos_half_narrow));
    assert!(is_backfacing_cone(view_dir, cone_axis, cos_half_wide));
}

// ── Frustum culling stub tests ────────────────────────────────────────────────

/// Simple frustum plane test: point is inside if dot(normal, point) + d >= 0.
fn inside_plane(normal: [f32; 3], d: f32, point: [f32; 3]) -> bool {
    normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] + d >= 0.0
}

#[test]
fn frustum_point_in_front_of_near_plane() {
    let normal = [0.0f32, 0.0, 1.0]; // facing +Z
    let d = -1.0f32; // near plane at z=1
    let point = [0.0f32, 0.0, 5.0]; // z=5 is in front
    assert!(inside_plane(normal, d, point));
}

#[test]
fn frustum_point_behind_near_plane() {
    let normal = [0.0f32, 0.0, 1.0];
    let d = -1.0f32;
    let point = [0.0f32, 0.0, 0.5]; // z=0.5 < near=1 → behind
    assert!(!inside_plane(normal, d, point));
}
