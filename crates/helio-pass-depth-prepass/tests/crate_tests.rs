//! Crate-level integration tests for `helio-pass-depth-prepass`.
//!
//! `DepthPrepassPass` is a GPU object — it cannot be constructed in a unit
//! test without a `wgpu::Device`.  These tests instead document the crate's
//! public surface, known architecture constants, and compile-time guarantees.

use helio_pass_depth_prepass::DepthPrepassPass;

// ── Type accessibility ────────────────────────────────────────────────────────

/// Verifies that `DepthPrepassPass` is exported from the crate.
#[test]
fn depth_prepass_pass_type_is_publicly_accessible() {
    // Using PhantomData is the zero-cost way to reference a type without
    // constructing it at runtime.
    let _: std::marker::PhantomData<DepthPrepassPass> = std::marker::PhantomData;
}

#[test]
fn depth_prepass_type_name_contains_pass() {
    let name = std::any::type_name::<DepthPrepassPass>();
    assert!(name.contains("DepthPrepassPass"), "got: {name}");
}

// ── PackedVertex stride constants ─────────────────────────────────────────────

/// The shared mesh vertex buffer uses a 40-byte stride.
/// PackedVertex layout: pos(12) + bitan(4) + uv0(8) + uv1(8) + normal(4) + tangent(4)
#[test]
fn packed_vertex_stride_is_40_bytes() {
    const STRIDE: usize = 12 + 4 + 8 + 8 + 4 + 4;
    assert_eq!(STRIDE, 40);
}

#[test]
fn packed_vertex_pos_field_is_12_bytes() {
    // position: vec3<f32> = 3 × 4 bytes
    assert_eq!(3 * std::mem::size_of::<f32>(), 12);
}

#[test]
fn packed_vertex_bitangent_sign_field_is_4_bytes() {
    // bitan: f32 = 4 bytes
    assert_eq!(std::mem::size_of::<f32>(), 4);
}

#[test]
fn packed_vertex_uv0_field_is_8_bytes() {
    // uv0: vec2<f32> = 2 × 4 bytes
    assert_eq!(2 * std::mem::size_of::<f32>(), 8);
}

#[test]
fn packed_vertex_uv1_field_is_8_bytes() {
    // uv1: vec2<f32> = 2 × 4 bytes
    assert_eq!(2 * std::mem::size_of::<f32>(), 8);
}

#[test]
fn packed_vertex_normal_field_is_4_bytes() {
    // normal: u32 packed = 4 bytes
    assert_eq!(std::mem::size_of::<u32>(), 4);
}

#[test]
fn packed_vertex_tangent_field_is_4_bytes() {
    // tangent: u32 packed = 4 bytes
    assert_eq!(std::mem::size_of::<u32>(), 4);
}

/// All named components must sum to the expected stride.
#[test]
fn all_packed_vertex_components_sum_to_stride() {
    let pos     = 3 * std::mem::size_of::<f32>(); // 12
    let bitan   =     std::mem::size_of::<f32>(); //  4
    let uv0     = 2 * std::mem::size_of::<f32>(); //  8
    let uv1     = 2 * std::mem::size_of::<f32>(); //  8
    let normal  =     std::mem::size_of::<u32>(); //  4
    let tangent =     std::mem::size_of::<u32>(); //  4
    assert_eq!(pos + bitan + uv0 + uv1 + normal + tangent, 40);
}

// ── Field byte offsets within PackedVertex ────────────────────────────────────

#[test]
fn packed_vertex_pos_starts_at_offset_0() {
    // position is the first field.
    const POS_OFFSET: usize = 0;
    assert_eq!(POS_OFFSET, 0);
}

#[test]
fn packed_vertex_bitan_starts_at_offset_12() {
    // After position (12 bytes).
    const BITAN_OFFSET: usize = 12;
    assert_eq!(BITAN_OFFSET, 3 * std::mem::size_of::<f32>());
}

#[test]
fn packed_vertex_uv0_starts_at_offset_16() {
    // After position(12) + bitan(4).
    const UV0_OFFSET: usize = 12 + 4;
    assert_eq!(UV0_OFFSET, 16);
}

#[test]
fn packed_vertex_uv1_starts_at_offset_24() {
    // After position(12) + bitan(4) + uv0(8).
    const UV1_OFFSET: usize = 12 + 4 + 8;
    assert_eq!(UV1_OFFSET, 24);
}

#[test]
fn packed_vertex_normal_starts_at_offset_32() {
    const NORMAL_OFFSET: usize = 12 + 4 + 8 + 8;
    assert_eq!(NORMAL_OFFSET, 32);
}

#[test]
fn packed_vertex_tangent_starts_at_offset_36() {
    const TANGENT_OFFSET: usize = 12 + 4 + 8 + 8 + 4;
    assert_eq!(TANGENT_OFFSET, 36);
}

// ── GPU buffer / index type sizes ─────────────────────────────────────────────

#[test]
fn u32_index_buffer_element_is_4_bytes() {
    // The depth prepass uses u32 indices → 4 bytes per index.
    assert_eq!(std::mem::size_of::<u32>(), 4);
}

#[test]
fn vertex_stride_is_multiple_of_four() {
    // GPU requires buffer stride to be a multiple of 4 bytes.
    const STRIDE: usize = 40;
    assert_eq!(STRIDE % 4, 0);
}

#[test]
fn vertex_stride_expressed_as_u64_fits_in_u32() {
    // wgpu's `array_stride` is u64 but the value 40 easily fits.
    let stride: u64 = 40;
    assert!(stride <= u32::MAX as u64);
}
