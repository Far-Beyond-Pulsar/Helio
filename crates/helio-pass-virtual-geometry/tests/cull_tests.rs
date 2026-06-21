// Tests for helio-pass-virtual-geometry: CullUniforms layout, dispatch math,
// meshlet visibility, screen coverage, LOD transitions, backface cone culling.
// Uses the actual public LodQuality API + locally mirrored private types.

use std::mem;
use helio_pass_virtual_geometry::LodQuality;

// ── Mirror private types ──────────────────────────────────────────────────────

/// Mirrors private CullUniforms (48 bytes: 16 header + 32 thresholds).
#[repr(C)]
#[derive(Clone, Copy)]
struct CullUniforms {
    meshlet_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    lod_thresholds: [f32; 7],
    _pad3: f32,
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
fn cull_uniforms_size_is_48() {
    assert_eq!(mem::size_of::<CullUniforms>(), 48);
}

#[test]
fn cull_uniforms_alignment_is_4() {
    assert_eq!(mem::align_of::<CullUniforms>(), 4);
}

#[test]
fn cull_uniforms_size_divisible_by_16() {
    assert_eq!(mem::size_of::<CullUniforms>() % 16, 0);
}

// ── VgGlobals layout tests ────────────────────────────────────────────────────

#[test]
fn vg_globals_size_is_96() {
    assert_eq!(mem::size_of::<VgGlobals>(), 96);
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

// ── WORKGROUP_SIZE / dispatch math tests ──────────────────────────────────────

#[test]
fn workgroup_size_is_64() {
    assert_eq!(WORKGROUP_SIZE, 64u32);
}

#[test]
fn dispatch_groups_ceil_division() {
    fn ceil_div(n: u32, d: u32) -> u32 { (n + d - 1) / d }
    assert_eq!(ceil_div(64, 64), 1);
    assert_eq!(ceil_div(65, 64), 2);
    assert_eq!(ceil_div(128, 64), 2);
    assert_eq!(ceil_div(1024, 64), 16);
    assert_eq!(ceil_div(0, 64), 0);
}

// ── Meshlet visibility / LOD threshold tests ────────────────────────────────

#[test]
fn medium_lod0_visible_when_screen_radius_above_s0() {
    let t = LodQuality::Medium.thresholds();
    let sr = 0.06f32;
    assert!(sr >= t[0], "LOD 0 should be visible");
}

#[test]
fn medium_lod0_not_visible_when_screen_radius_below_s0() {
    let t = LodQuality::Medium.thresholds();
    let sr = 0.03f32;
    assert!(sr < t[0]);
}

#[test]
fn medium_lod1_visible_when_screen_radius_between_s1_and_s0() {
    let t = LodQuality::Medium.thresholds();
    let sr = 0.04f32;
    assert!(sr < t[0] && sr >= t[1], "sr={sr} s0={} s1={}", t[0], t[1]);
}

// ── Screen coverage formula tests ─────────────────────────────────────────────

fn screen_radius(obj_radius: f32, fov_rad: f32, dist: f32) -> f32 {
    let cot_half_fov = 1.0 / (fov_rad / 2.0).tan();
    obj_radius * cot_half_fov / dist
}

#[test]
fn screen_radius_far_object_below_medium_s0() {
    let t = LodQuality::Medium.thresholds();
    let fov = std::f32::consts::FRAC_PI_2;
    let sr = screen_radius(1.0, fov, 100.0);
    assert!(sr < t[0], "sr={sr} s0={}", t[0]);
}

#[test]
fn screen_radius_close_object_above_ultra_s0() {
    let t = LodQuality::Ultra.thresholds();
    let fov = std::f32::consts::FRAC_PI_2;
    let sr = screen_radius(10.0, fov, 5.0);
    assert!(sr > t[0], "sr={sr} ultra_s0={}", t[0]);
}

#[test]
fn all_quality_levels_have_positive_thresholds() {
    for q in [LodQuality::Low, LodQuality::Medium, LodQuality::High, LodQuality::Ultra] {
        let t = q.thresholds();
        for &v in &t {
            assert!(v > 0.0, "{:?} threshold {v}", q);
        }
    }
}

// ── Backface cone culling tests ───────────────────────────────────────────────

fn is_backfacing_cone(view_dir: [f32; 3], cone_axis: [f32; 3], cos_half_angle: f32) -> bool {
    let dot = view_dir[0] * cone_axis[0]
        + view_dir[1] * cone_axis[1]
        + view_dir[2] * cone_axis[2];
    dot + cos_half_angle <= 0.0
}

#[test]
fn cone_culling_backfacing_when_view_behind_cone() {
    assert!(is_backfacing_cone([0.0, 0.0, -1.0], [0.0, 0.0, 1.0], 0.5));
}

#[test]
fn cone_culling_visible_when_view_in_front_of_cone() {
    assert!(!is_backfacing_cone([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 0.5));
}

// ── Frustum culling stub tests ────────────────────────────────────────────────

fn inside_plane(normal: [f32; 3], d: f32, point: [f32; 3]) -> bool {
    normal[0] * point[0] + normal[1] * point[1] + normal[2] * point[2] + d >= 0.0
}

#[test]
fn frustum_point_in_front_of_near_plane() {
    assert!(inside_plane([0.0, 0.0, 1.0], -1.0, [0.0, 0.0, 5.0]));
}

#[test]
fn frustum_point_behind_near_plane() {
    assert!(!inside_plane([0.0, 0.0, 1.0], -1.0, [0.0, 0.0, 0.5]));
}
