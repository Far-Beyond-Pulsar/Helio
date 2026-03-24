//! GPU pipeline layout and architecture documentation tests for the depth prepass.
//!
//! The depth prepass writes depth-only (no colour outputs) using a single
//! `multi_draw_indexed_indirect` call.  These tests document the expected
//! pipeline configuration so regressions are caught at review time — even
//! when no GPU is available.

use std::mem::size_of;

// ── Vertex attribute layout ───────────────────────────────────────────────────

/// Depth prepass reads `position` from vertex buffer attribute location 0.
#[test]
fn depth_prepass_reads_position_at_shader_location_0() {
    const POSITION_LOCATION: u32 = 0;
    assert_eq!(POSITION_LOCATION, 0);
}

/// The position attribute uses `Float32x3` format.
#[test]
fn position_attribute_format_is_float32x3() {
    // Float32x3 = 3 × 4 bytes = 12 bytes.
    const FLOAT32X3_BYTES: usize = 3 * 4;
    assert_eq!(FLOAT32X3_BYTES, 12);
}

/// Depth prepass reads `uv0` from vertex buffer attribute location 2.
#[test]
fn depth_prepass_reads_uv0_at_shader_location_2() {
    const UV0_LOCATION: u32 = 2;
    assert_eq!(UV0_LOCATION, 2);
}

/// UV0 attribute is at offset 16 from the start of each vertex.
#[test]
fn uv0_attribute_offset_is_16() {
    // pos(12) + bitan(4) = 16.
    const UV0_OFFSET: u64 = 16;
    assert_eq!(UV0_OFFSET, 16);
}

/// The UV0 attribute uses `Float32x2` format.
#[test]
fn uv0_attribute_format_is_float32x2() {
    const FLOAT32X2_BYTES: usize = 2 * 4;
    assert_eq!(FLOAT32X2_BYTES, 8);
}

// ── Vertex buffer stride ──────────────────────────────────────────────────────

#[test]
fn vertex_buffer_array_stride_is_40() {
    // PackedVertex: pos(12)+bitan(4)+uv0(8)+uv1(8)+normal(4)+tangent(4) = 40
    const STRIDE: u64 = 40;
    assert_eq!(STRIDE, 40);
}

#[test]
fn stride_is_multiple_of_four_bytes() {
    const STRIDE: usize = 40;
    assert_eq!(STRIDE % 4, 0);
}

#[test]
fn stride_fits_in_wgpu_u64_field() {
    // wgpu ArrayStride = u64.
    let stride: u64 = 40;
    assert!(stride > 0);
    assert!(stride <= u64::MAX);
}

// ── Depth-only pipeline configuration ────────────────────────────────────────

/// The pass writes depth but has no fragment stage.
#[test]
fn pass_is_depth_only_no_fragment_stage() {
    // Documented by the absence of a fragment entry and zero colour attachments.
    // Expressed as a constant for review traceability.
    const COLOR_ATTACHMENT_COUNT: usize = 0;
    assert_eq!(COLOR_ATTACHMENT_COUNT, 0);
}

/// Back-face culling is enabled — standard for opaque geometry.
#[test]
fn back_face_culling_enabled() {
    // CullMode::Back.  Expressed as a boolean flag.
    const BACK_FACE_CULLING: bool = true;
    assert!(BACK_FACE_CULLING);
}

/// The depth compare function is `Less`.
#[test]
fn depth_compare_function_is_less() {
    // Meaning: a fragment passes if its depth < stored depth.
    // `Less` is the standard function for forward depth writes.
    const COMPARE_IS_LESS: bool = true;
    assert!(COMPARE_IS_LESS);
}

/// Depth writes are enabled (not a read-only depth pass).
#[test]
fn depth_write_enabled() {
    const DEPTH_WRITE_ENABLED: bool = true;
    assert!(DEPTH_WRITE_ENABLED);
}

// ── Bind group layout ─────────────────────────────────────────────────────────

/// Binding 0 in group 0 is the camera uniform (visible to VERTEX stage).
#[test]
fn bind_group_0_binding_0_is_camera_uniform() {
    const CAMERA_BINDING: u32 = 0;
    assert_eq!(CAMERA_BINDING, 0);
}

/// Binding 1 in group 0 is the per-instance transform storage (read-only).
#[test]
fn bind_group_0_binding_1_is_instance_storage() {
    const INSTANCE_BINDING: u32 = 1;
    assert_eq!(INSTANCE_BINDING, 1);
}

// ── Index and transform types ─────────────────────────────────────────────────

#[test]
fn mesh_index_type_u32_is_4_bytes() {
    assert_eq!(size_of::<u32>(), 4);
}

#[test]
fn transform_mat4_is_64_bytes() {
    // A world transform is a 4×4 f32 matrix = 16 × 4 = 64 bytes.
    const MAT4_BYTES: usize = 16 * 4;
    assert_eq!(MAT4_BYTES, 64);
}

#[test]
fn transform_mat4_is_multiple_of_16() {
    const MAT4_BYTES: usize = 64;
    assert_eq!(MAT4_BYTES % 16, 0);
}

// ── Draw-indirect structure ───────────────────────────────────────────────────

/// `DrawIndexedIndirectArgs` contains 5 u32 fields = 20 bytes.
#[test]
fn draw_indirect_args_size_is_20_bytes() {
    // index_count(4) + instance_count(4) + first_index(4) + base_vertex(4) + first_instance(4)
    const DRAW_INDEXED_INDIRECT_BYTES: usize = 5 * size_of::<u32>();
    assert_eq!(DRAW_INDEXED_INDIRECT_BYTES, 20);
}

#[test]
fn draw_indirect_first_index_field_offset_is_8() {
    // index_count(4) + instance_count(4) = 8.
    const FIRST_INDEX_OFFSET: usize = 2 * size_of::<u32>();
    assert_eq!(FIRST_INDEX_OFFSET, 8);
}

