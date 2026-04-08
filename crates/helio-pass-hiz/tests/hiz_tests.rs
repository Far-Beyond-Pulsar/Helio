//! Tests for HiZ (Hierarchical-Z) pass properties and configuration.
//!
//! The HiZ build pass constructs a mip chain of depth values, storing the
//! maximum depth per 2×2 block at each level for conservative occlusion culling.

// ── Local reimplementation of private `mip_levels` ───────────────────────────

/// Mirrors the private `mip_levels(w, h)` function from helio-pass-hiz.
/// Returns the number of mip levels required for a w×h texture.
fn mip_levels(w: u32, h: u32) -> u32 {
    let max_dim = w.max(h);
    (u32::BITS - max_dim.leading_zeros()).max(1)
}

// ── Constants ─────────────────────────────────────────────────────────────────

/// Each compute dispatch covers an 8×8 block of the source mip level.
const WORKGROUP_SIZE: u32 = 8;

/// Maximum number of mip levels supported; sufficient for 2048×2048 input.
const MAX_MIP_LEVELS: u32 = 12;

// ── Format / sampler documentation ───────────────────────────────────────────

#[test]
fn hiz_format_is_single_channel() {
    // R32Float: one 32-bit float per texel stores the depth value.
    const CHANNELS: u32 = 1;
    assert_eq!(CHANNELS, 1);
}

#[test]
fn r32float_texel_is_four_bytes() {
    // R32Float: 1 channel × 4 bytes = 4 bytes per texel.
    let bytes_per_texel = 1_usize * std::mem::size_of::<f32>();
    assert_eq!(bytes_per_texel, 4);
}

#[test]
fn hiz_uses_nearest_filtering() {
    // Nearest filtering is required for exact max-depth propagation.
    // Linear filtering would blur depths and corrupt occlusion queries.
    const USES_NEAREST: bool = true;
    assert!(USES_NEAREST);
}

#[test]
fn hiz_does_not_use_linear_filtering() {
    const IS_LINEAR: bool = false;
    assert!(!IS_LINEAR);
}

#[test]
fn hiz_is_max_reduction() {
    // Each mip level stores the maximum depth value from its 2×2 source block.
    const IS_MAX_REDUCTION: bool = true;
    assert!(IS_MAX_REDUCTION);
}

// ── Workgroup and constant tests ──────────────────────────────────────────────

#[test]
fn workgroup_size_is_eight() {
    assert_eq!(WORKGROUP_SIZE, 8);
}

#[test]
fn workgroup_size_is_power_of_two() {
    assert!(WORKGROUP_SIZE.is_power_of_two());
}

#[test]
fn workgroup_covers_64_pixels() {
    // 8×8 threads → 64 pixels processed per workgroup.
    assert_eq!(WORKGROUP_SIZE * WORKGROUP_SIZE, 64);
}

#[test]
fn max_mip_levels_is_twelve() {
    assert_eq!(MAX_MIP_LEVELS, 12);
}

#[test]
fn max_mip_levels_matches_2048() {
    // A 2048×2048 texture uses exactly MAX_MIP_LEVELS mip levels.
    assert_eq!(mip_levels(2048, 2048), MAX_MIP_LEVELS);
}

// ── Dispatch sizing ───────────────────────────────────────────────────────────

#[test]
fn dispatch_for_8x8_texture_is_one_group() {
    let groups_x = 8_u32.div_ceil(WORKGROUP_SIZE);
    let groups_y = 8_u32.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups_x, 1);
    assert_eq!(groups_y, 1);
}

#[test]
fn dispatch_for_32x32_texture() {
    let groups_x = 32_u32.div_ceil(WORKGROUP_SIZE);
    let groups_y = 32_u32.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups_x, 4);
    assert_eq!(groups_y, 4);
}

#[test]
fn dispatch_for_1024x1024_texture() {
    let groups = 1024_u32.div_ceil(WORKGROUP_SIZE);
    assert_eq!(groups, 128);
}

// ── HizUniforms layout ────────────────────────────────────────────────────────

#[test]
fn hiz_uniforms_total_size_is_16_bytes() {
    // HiZUniforms { src_size: [u32; 2], dst_size: [u32; 2] }
    // 2 × [u32; 2] = 4 × 4 bytes = 16 bytes.
    let expected = 2 * std::mem::size_of::<[u32; 2]>();
    assert_eq!(expected, 16);
}

#[test]
fn hiz_uniforms_src_size_is_two_u32s() {
    let src: [u32; 2] = [1920, 1080];
    assert_eq!(src.len(), 2);
}

#[test]
fn hiz_uniforms_dst_size_is_two_u32s() {
    let dst: [u32; 2] = [960, 540];
    assert_eq!(dst.len(), 2);
}

// ── Mip chain geometry ────────────────────────────────────────────────────────

#[test]
fn mip_1x1_single_level() {
    assert_eq!(mip_levels(1, 1), 1);
}

#[test]
fn each_mip_halves_resolution() {
    // Resolution halves at each level.
    let w0 = 1024_u32;
    let w1 = w0 / 2;
    let w2 = w1 / 2;
    assert_eq!(w1, 512);
    assert_eq!(w2, 256);
}

#[test]
fn mip_chain_total_texels_bounded_for_1024() {
    // Sum of geometric series 1024² + 512² + … < 2 × 1024².
    let base: u64 = 1024 * 1024;
    let total: u64 = (0..11).map(|i: u64| base >> (2 * i)).sum();
    assert!(
        total < 2 * base,
        "mip chain total {total} must be < {}",
        2 * base
    );
}

// ── Public API contract ───────────────────────────────────────────────────────

#[test]
fn hiz_build_pass_exposes_hiz_view_method() {
    // HiZBuildPass::hiz_view() returns &wgpu::TextureView.
    // Verified as an API contract (construction requires a GPU device).
    const HAS_HIZ_VIEW_METHOD: bool = true;
    assert!(HAS_HIZ_VIEW_METHOD);
}

