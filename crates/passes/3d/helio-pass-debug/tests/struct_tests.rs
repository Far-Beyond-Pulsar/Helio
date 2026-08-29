//! Structural layout tests for `DebugVertex`.
//!
//! `DebugVertex` is `#[repr(C)]` and must match the vertex input layout
//! declared in `debug_draw.wgsl`:
//!   location(0) position: vec3<f32>   (offset  0, 12 bytes)
//!   [implicit pad: 4 bytes at offset 12]
//!   location(1) color:    vec4<f32>   (offset 16, 16 bytes)
//!
//! Total stride: 32 bytes.

use bytemuck::{bytes_of, Zeroable};
use helio_pass_debug::DebugVertex;
use std::mem::{align_of, offset_of, size_of};

// ── Size ─────────────────────────────────────────────────────────────────────

#[test]
fn size_of_debug_vertex_is_32() {
    // [f32;3] + f32 + [f32;4] = 4 + 4 + 4 + 4 + 4*4 = 32 bytes.
    assert_eq!(size_of::<DebugVertex>(), 32);
}

#[test]
fn size_is_eight_f32_values() {
    assert_eq!(size_of::<DebugVertex>(), 8 * size_of::<f32>());
}

#[test]
fn size_matches_wgsl_stride() {
    // The pipeline's array_stride must equal this size.
    assert_eq!(size_of::<DebugVertex>(), 32);
}

#[test]
fn size_is_multiple_of_four() {
    assert_eq!(size_of::<DebugVertex>() % 4, 0);
}

// ── Alignment ────────────────────────────────────────────────────────────────

#[test]
fn align_of_debug_vertex_is_four() {
    // Largest scalar type is f32 (4 bytes) → alignment is 4.
    assert_eq!(align_of::<DebugVertex>(), 4);
}

#[test]
fn size_divisible_by_alignment() {
    assert_eq!(size_of::<DebugVertex>() % align_of::<DebugVertex>(), 0);
}

// ── Field offsets (repr(C)) ───────────────────────────────────────────────────

#[test]
fn position_is_at_offset_0() {
    assert_eq!(offset_of!(DebugVertex, position), 0);
}

#[test]
fn pad_is_at_offset_12() {
    // `position` is [f32; 3] = 12 bytes, so `_pad` follows immediately.
    assert_eq!(offset_of!(DebugVertex, _pad), 12);
}

#[test]
fn color_is_at_offset_16() {
    // After position (12) and _pad (4) = 16 bytes.
    assert_eq!(offset_of!(DebugVertex, color), 16);
}

#[test]
fn pad_offset_equals_three_f32_bytes() {
    assert_eq!(offset_of!(DebugVertex, _pad), 3 * size_of::<f32>());
}

#[test]
fn color_offset_equals_four_f32_bytes() {
    assert_eq!(offset_of!(DebugVertex, color), 4 * size_of::<f32>());
}

#[test]
fn color_plus_its_size_equals_total_struct_size() {
    assert_eq!(
        offset_of!(DebugVertex, color) + 4 * size_of::<f32>(),
        size_of::<DebugVertex>()
    );
}

// ── Construction ─────────────────────────────────────────────────────────────

#[test]
fn can_construct_debug_vertex() {
    let v = DebugVertex {
        position: [1.0, 2.0, 3.0],
        _pad: 0.0,
        color: [1.0, 0.0, 0.0, 1.0],
    };
    assert_eq!(v.position[0], 1.0);
    assert_eq!(v.position[1], 2.0);
    assert_eq!(v.position[2], 3.0);
}

#[test]
fn color_rgba_accessible_via_index() {
    let v = DebugVertex {
        position: [0.0; 3],
        _pad: 0.0,
        color: [1.0, 0.5, 0.25, 0.75],
    };
    assert_eq!(v.color[0], 1.0); // R
    assert_eq!(v.color[1], 0.5); // G
    assert_eq!(v.color[2], 0.25); // B
    assert_eq!(v.color[3], 0.75); // A
}

#[test]
fn pad_field_is_accessible_and_mutable() {
    let mut v = DebugVertex {
        position: [0.0; 3],
        _pad: 0.0,
        color: [0.0; 4],
    };
    v._pad = 1.0;
    assert_eq!(v._pad, 1.0);
}

// ── Mutation ─────────────────────────────────────────────────────────────────

#[test]
fn can_mutate_position_components() {
    let mut v = DebugVertex {
        position: [0.0; 3],
        _pad: 0.0,
        color: [1.0; 4],
    };
    v.position[0] = 5.0;
    v.position[1] = -3.0;
    v.position[2] = 100.0;
    assert_eq!(v.position, [5.0, -3.0, 100.0]);
}

#[test]
fn can_mutate_color_alpha() {
    let mut v = DebugVertex {
        position: [0.0; 3],
        _pad: 0.0,
        color: [1.0, 1.0, 1.0, 1.0],
    };
    v.color[3] = 0.0;
    assert_eq!(v.color[3], 0.0);
}

// ── Copy / Clone ─────────────────────────────────────────────────────────────

#[test]
fn debug_vertex_is_copy() {
    let a = DebugVertex {
        position: [1.0, 2.0, 3.0],
        _pad: 0.0,
        color: [0.5; 4],
    };
    let b = a; // copy
    assert_eq!(b.position, a.position);
    assert_eq!(b.color, a.color);
}

#[test]
fn debug_vertex_is_clone() {
    let a = DebugVertex {
        position: [1.0, 2.0, 3.0],
        _pad: 0.0,
        color: [0.5; 4],
    };
    let b = a.clone();
    assert_eq!(b.position, a.position);
}

// ── bytemuck: zeroed / Pod ─────────────────────────────────────────────────

#[test]
fn zeroed_vertex_has_zero_position() {
    let v: DebugVertex = Zeroable::zeroed();
    assert_eq!(v.position, [0.0f32; 3]);
}

#[test]
fn zeroed_vertex_has_zero_color() {
    let v: DebugVertex = Zeroable::zeroed();
    assert_eq!(v.color, [0.0f32; 4]);
}

#[test]
fn zeroed_vertex_has_zero_pad() {
    let v: DebugVertex = Zeroable::zeroed();
    assert_eq!(v._pad, 0.0f32);
}

#[test]
fn zeroed_vertex_bytes_are_all_zero() {
    let v: DebugVertex = Zeroable::zeroed();
    assert!(bytes_of(&v).iter().all(|&b| b == 0));
}

#[test]
fn bytes_of_debug_vertex_is_32_bytes() {
    let v: DebugVertex = Zeroable::zeroed();
    assert_eq!(bytes_of(&v).len(), 32);
}
