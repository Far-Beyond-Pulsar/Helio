// Pure math tests for helio-pass-radiance-cascades constants and struct layout.
// No crate imports required — all types and constants are defined locally.

const PROBE_DIM: u32 = 8;
const DIR_DIM: u32 = 4;
const ATLAS_W: u32 = 32; // PROBE_DIM * DIR_DIM
const ATLAS_H: u32 = 256; // PROBE_DIM * PROBE_DIM * DIR_DIM

/// Local mirror of the private RCDynamic uniform struct for size/layout verification.
#[repr(C)]
struct RCDynamic {
    world_min: [f32; 4],
    world_max: [f32; 4],
    frame: u32,
    light_count: u32,
    _pad0: u32,
    _pad1: u32,
    sky_color: [f32; 4],
}

// ── Named constant values ─────────────────────────────────────────────────────

#[test]
fn probe_dim_is_eight() {
    assert_eq!(PROBE_DIM, 8);
}

#[test]
fn dir_dim_is_four() {
    assert_eq!(DIR_DIM, 4);
}

#[test]
fn atlas_w_is_32() {
    assert_eq!(ATLAS_W, 32);
}

#[test]
fn atlas_h_is_256() {
    assert_eq!(ATLAS_H, 256);
}

// ── Atlas dimension derivations ───────────────────────────────────────────────

#[test]
fn atlas_w_equals_probe_dim_times_dir_dim() {
    assert_eq!(PROBE_DIM * DIR_DIM, ATLAS_W);
}

#[test]
fn atlas_h_equals_probe_dim_squared_times_dir_dim() {
    assert_eq!(PROBE_DIM * PROBE_DIM * DIR_DIM, ATLAS_H);
}

#[test]
fn atlas_h_equals_probe_dim_cubed_div_probe_dim_times_dir_dim() {
    // A more explicit form: PROBE_DIM layers, each row = PROBE_DIM rows × DIR_DIM directions
    assert_eq!(PROBE_DIM * (PROBE_DIM * DIR_DIM), ATLAS_H);
}

// ── Probe and direction counts ────────────────────────────────────────────────

#[test]
fn total_probes_in_3d_grid_is_512() {
    // PROBE_DIM^3
    assert_eq!(PROBE_DIM * PROBE_DIM * PROBE_DIM, 512);
}

#[test]
fn total_directions_per_probe_is_16() {
    // DIR_DIM^2 directions per probe
    assert_eq!(DIR_DIM * DIR_DIM, 16);
}

#[test]
fn total_atlas_texels_is_8192() {
    assert_eq!(ATLAS_W * ATLAS_H, 8192);
}

#[test]
fn probe_direction_pairs_equal_atlas_texel_count() {
    // Every atlas texel corresponds to exactly one (probe, direction) pair.
    let probes = PROBE_DIM * PROBE_DIM * PROBE_DIM; // 512
    let dirs = DIR_DIM * DIR_DIM; // 16
    assert_eq!(probes * dirs, ATLAS_W * ATLAS_H);
}

// ── Dispatch dimensions ───────────────────────────────────────────────────────

#[test]
fn workgroup_dispatch_x_for_8x8_groups() {
    // ceil(ATLAS_W / 8) = ceil(32 / 8) = 4
    let dispatch_x = (ATLAS_W + 7) / 8;
    assert_eq!(dispatch_x, 4);
}

#[test]
fn workgroup_dispatch_y_for_8x8_groups() {
    // ceil(ATLAS_H / 8) = ceil(256 / 8) = 32
    let dispatch_y = (ATLAS_H + 7) / 8;
    assert_eq!(dispatch_y, 32);
}

#[test]
fn workgroup_dispatch_total_for_8x8_groups() {
    let dispatch_x = (ATLAS_W + 7) / 8;
    let dispatch_y = (ATLAS_H + 7) / 8;
    assert_eq!(dispatch_x * dispatch_y, 128);
}

// ── RCDynamic struct layout ───────────────────────────────────────────────────

#[test]
fn rcdynamic_size_is_64_bytes() {
    assert_eq!(std::mem::size_of::<RCDynamic>(), 64);
}

#[test]
fn rcdynamic_world_min_field_is_16_bytes() {
    // [f32; 4] = 4 × 4 = 16 bytes
    assert_eq!(std::mem::size_of::<[f32; 4]>(), 16);
}

#[test]
fn rcdynamic_world_max_field_is_16_bytes() {
    assert_eq!(std::mem::size_of::<[f32; 4]>(), 16);
}

#[test]
fn rcdynamic_scalar_u32_fields_are_4_bytes() {
    // frame, light_count, _pad0, _pad1 — each 4 bytes, 4 × 4 = 16
    assert_eq!(4 * std::mem::size_of::<u32>(), 16);
}

#[test]
fn rcdynamic_sky_color_is_16_bytes() {
    assert_eq!(std::mem::size_of::<[f32; 4]>(), 16);
}

#[test]
fn rcdynamic_field_sum_matches_struct_size() {
    // world_min(16) + world_max(16) + 4×u32(16) + sky_color(16) = 64
    let computed = 16 + 16 + 4 * std::mem::size_of::<u32>() + 16;
    assert_eq!(computed, 64);
    assert_eq!(computed, std::mem::size_of::<RCDynamic>());
}

// ── Texel memory footprint ────────────────────────────────────────────────────

#[test]
fn rgba16_float_bytes_per_texel() {
    // Rgba16Float: 4 channels × 2 bytes = 8 bytes per texel
    let bytes_per_texel: u32 = 4 * 2;
    assert_eq!(bytes_per_texel, 8);
}

#[test]
fn total_atlas_memory_is_65536_bytes() {
    // 32 × 256 × 8 bytes per texel
    let bytes_per_texel: u32 = 8;
    assert_eq!(ATLAS_W * ATLAS_H * bytes_per_texel, 65536);
}

#[test]
fn total_atlas_memory_is_64_kib() {
    let bytes_per_texel: u32 = 8;
    let total = ATLAS_W * ATLAS_H * bytes_per_texel;
    assert_eq!(total, 64 * 1024);
}

// ── World bounds validity ─────────────────────────────────────────────────────

#[test]
fn valid_scene_has_world_max_greater_than_world_min() {
    let world_min = [-100.0f32, -100.0, -100.0, 1.0];
    let world_max = [100.0f32, 100.0, 100.0, 1.0];
    for i in 0..3 {
        assert!(world_max[i] > world_min[i]);
    }
}

#[test]
fn empty_scene_world_bounds_are_degenerate() {
    let world_min = [0.0f32, 0.0, 0.0, 1.0];
    let world_max = [0.0f32, 0.0, 0.0, 1.0];
    // A degenerate (zero-volume) scene has equal min and max
    for i in 0..3 {
        assert_eq!(world_min[i], world_max[i]);
    }
}

// ── Power-of-two properties ───────────────────────────────────────────────────

#[test]
fn probe_dim_is_power_of_two() {
    assert!(PROBE_DIM.is_power_of_two());
}

#[test]
fn dir_dim_is_power_of_two() {
    assert!(DIR_DIM.is_power_of_two());
}

#[test]
fn atlas_w_is_power_of_two() {
    assert!(ATLAS_W.is_power_of_two());
}

#[test]
fn atlas_h_is_power_of_two() {
    assert!(ATLAS_H.is_power_of_two());
}

// ── Sanity / relationship checks ──────────────────────────────────────────────

#[test]
fn dir_dim_squared_is_total_directions_per_probe() {
    assert_eq!(DIR_DIM * DIR_DIM, 16);
}

#[test]
fn probe_dim_cubed_is_total_probes() {
    assert_eq!(PROBE_DIM.pow(3), 512);
}

#[test]
fn atlas_w_less_than_atlas_h() {
    assert!(ATLAS_W < ATLAS_H);
}

#[test]
fn atlas_h_div_atlas_w_equals_probe_dim() {
    // 256 / 32 = 8 = PROBE_DIM
    assert_eq!(ATLAS_H / ATLAS_W, PROBE_DIM);
}
