// Tests for helio-pass-virtual-geometry: CullUniforms layout, dispatch math,
// meshlet visibility, screen coverage, LOD transitions, backface cone culling.
// Uses the actual public LodQuality API + locally mirrored private types.

use helio_pass_virtual_geometry::LodQuality;
use std::mem;

// ── Mirror private types ──────────────────────────────────────────────────────

/// Mirrors private CullUniforms (48 bytes).
#[repr(C)]
#[derive(Clone, Copy)]
struct CullUniforms {
    object_count: u32,
    screen_width: u32,
    screen_height: u32,
    hiz_mip_count: u32,
    draw_capacity: u32,
    lod_error_threshold_px: f32,
    object_dispatch_width: u32,
    work_item_count: u32,
    work_dispatch_width: u32,
    _pad0: u32,
    _pad1: u32,
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
const INITIAL_OBJECTS: u64 = 256;
const INITIAL_INSTANCES: u64 = 256;

#[test]
fn cull_shader_parses_and_validates() {
    let source = include_str!("../shaders/vg_cull.wgsl");
    let module = naga::front::wgsl::parse_str(source).expect("VG cull shader must parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("VG cull shader must validate");
}

#[test]
fn gbuffer_shader_parses_and_validates() {
    let source = include_str!("../shaders/vg_gbuffer.wgsl");
    let module = naga::front::wgsl::parse_str(source).expect("VG draw shader must parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("VG draw shader must validate");
}

fn validate_webgpu_shader(label: &str, source: &str) {
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|error| panic!("{label} must parse as WGSL: {error}"));
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .unwrap_or_else(|error| panic!("{label} must validate: {error}"));
}

#[test]
fn browser_shader_variants_parse_and_validate() {
    const MAX_TEXTURES: usize = 16;

    let fix_material_shader = |source: &str| {
        let source = source
            .replace(
                "binding_array<texture_2d<f32>, 256>",
                &format!("binding_array<texture_2d<f32>, {MAX_TEXTURES}>"),
            )
            .replace(
                "binding_array<sampler, 256>",
                &format!("binding_array<sampler, {MAX_TEXTURES}>"),
            );
        libhelio::shader::apply_webgpu_material_bindings(&source, MAX_TEXTURES)
    };

    let gbuffer = fix_material_shader(include_str!(
        "../../helio-pass-gbuffer/shaders/gbuffer.wgsl"
    ));
    assert!(!gbuffer.contains("binding_array"));
    assert!(gbuffer.contains("@binding(33) var scene_sampler_15"));
    validate_webgpu_shader("WebGPU GBuffer shader", &gbuffer);

    let vg_gbuffer = fix_material_shader(include_str!("../shaders/vg_gbuffer.wgsl"));
    assert!(!vg_gbuffer.contains("binding_array"));
    validate_webgpu_shader("WebGPU virtual-geometry GBuffer shader", &vg_gbuffer);

    validate_webgpu_shader(
        "Corona compute shader",
        include_str!("../../helio-pass-corona/shaders/corona.wgsl"),
    );
    let corona_render = include_str!("../../helio-pass-corona/shaders/corona_render.wgsl");
    assert!(!corona_render.contains("read_write"));
    validate_webgpu_shader("Corona render shader", corona_render);

    validate_webgpu_shader(
        "FXAA shader",
        include_str!("../../helio-pass-fxaa/shaders/fxaa.wgsl"),
    );
}

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
fn initial_objects_is_256() {
    assert_eq!(INITIAL_OBJECTS, 256u64);
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
fn dispatch_uses_parallel_object_lanes_and_meshlet_spans() {
    fn grid(workgroup_count: u32, limit: u32) -> (u32, u32) {
        if workgroup_count == 0 {
            return (0, 0);
        }
        let width = workgroup_count.min(limit);
        (width, workgroup_count.div_ceil(width))
    }

    assert_eq!(grid(1_u32.div_ceil(WORKGROUP_SIZE), 65_535), (1, 1));
    assert_eq!(grid(65_536_u32.div_ceil(WORKGROUP_SIZE), 65_535), (1024, 1));
    assert_eq!(grid(65_536, 65_535), (65_535, 2));
    assert_eq!(grid(0, 65_535), (0, 0));
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
    for q in [
        LodQuality::Low,
        LodQuality::Medium,
        LodQuality::High,
        LodQuality::Ultra,
    ] {
        let t = q.thresholds();
        for &v in &t {
            assert!(v > 0.0, "{:?} threshold {v}", q);
        }
    }
}

// ── Backface cone culling tests ───────────────────────────────────────────────

fn meshopt_perspective_cone_reject(
    camera: [f32; 3],
    cone_apex: [f32; 3],
    cone_axis: [f32; 3],
    cone_cutoff: f32,
) -> bool {
    let to_apex = [
        cone_apex[0] - camera[0],
        cone_apex[1] - camera[1],
        cone_apex[2] - camera[2],
    ];
    let length =
        (to_apex[0] * to_apex[0] + to_apex[1] * to_apex[1] + to_apex[2] * to_apex[2]).sqrt();
    if length <= 1.0e-6 {
        return false;
    }
    let dot = (to_apex[0] * cone_axis[0] + to_apex[1] * cone_axis[1] + to_apex[2] * cone_axis[2])
        / length;
    dot >= cone_cutoff
}

#[test]
fn cone_culling_backfacing_when_view_behind_cone() {
    assert!(meshopt_perspective_cone_reject(
        [0.0, 0.0, -10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        0.5,
    ));
}

#[test]
fn cone_culling_visible_when_view_in_front_of_cone() {
    assert!(!meshopt_perspective_cone_reject(
        [0.0, 0.0, 10.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        0.5,
    ));
}

#[test]
fn cone_culling_is_disabled_at_the_apex_singularity() {
    assert!(!meshopt_perspective_cone_reject(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
        [0.0, 0.0, 1.0],
        0.5,
    ));
}

#[test]
fn conservative_hiz_uses_the_farthest_corner() {
    let samples = [0.2_f32, 0.3, 0.9, 0.4];
    let farthest = samples.into_iter().fold(0.0_f32, f32::max);
    assert_eq!(farthest, 0.9);
    assert!(!(0.8 > farthest), "a visible corner must prevent occlusion");
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
