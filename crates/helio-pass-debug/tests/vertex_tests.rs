//! Vertex geometry and encoding tests for `DebugVertex`.
//!
//! These tests verify that `DebugVertex` can correctly represent debug draw
//! geometry (line segments, points) and that its byte encoding matches the
//! wgsl shader's vertex buffer layout with stride 32 bytes.

use bytemuck::{bytes_of, cast_slice, Zeroable};
use helio_pass_debug::DebugVertex;
use std::mem::size_of;

// ── Canonical vertex constructions ────────────────────────────────────────────

#[test]
fn origin_vertex_has_zero_position() {
    let v = DebugVertex {
        position: [0.0, 0.0, 0.0],
        _pad: 0.0,
        color: [1.0, 1.0, 1.0, 1.0],
    };
    assert_eq!(v.position, [0.0f32; 3]);
}

#[test]
fn unit_x_axis_vertex() {
    let v = DebugVertex {
        position: [1.0, 0.0, 0.0],
        _pad: 0.0,
        color: [1.0, 0.0, 0.0, 1.0], // red axis
    };
    assert_eq!(v.position[0], 1.0);
    assert_eq!(v.position[1], 0.0);
    assert_eq!(v.position[2], 0.0);
}

#[test]
fn unit_y_axis_vertex() {
    let v = DebugVertex {
        position: [0.0, 1.0, 0.0],
        _pad: 0.0,
        color: [0.0, 1.0, 0.0, 1.0], // green axis
    };
    assert_eq!(v.position[1], 1.0);
    assert_eq!(v.color[1], 1.0);
}

#[test]
fn unit_z_axis_vertex() {
    let v = DebugVertex {
        position: [0.0, 0.0, 1.0],
        _pad: 0.0,
        color: [0.0, 0.0, 1.0, 1.0], // blue axis
    };
    assert_eq!(v.position[2], 1.0);
    assert_eq!(v.color[2], 1.0);
}

// ── Color keywords as vertex configurations ───────────────────────────────────

#[test]
fn opaque_red_vertex() {
    let v = DebugVertex {
        position: [0.0; 3],
        _pad: 0.0,
        color: [1.0, 0.0, 0.0, 1.0],
    };
    assert_eq!(v.color, [1.0, 0.0, 0.0, 1.0]);
}

#[test]
fn opaque_white_vertex() {
    let v = DebugVertex {
        position: [0.0; 3],
        _pad: 0.0,
        color: [1.0, 1.0, 1.0, 1.0],
    };
    assert!(v.color.iter().all(|&c| c == 1.0));
}

#[test]
fn fully_transparent_vertex_has_zero_alpha() {
    let v = DebugVertex {
        position: [0.0; 3],
        _pad: 0.0,
        color: [1.0, 1.0, 1.0, 0.0],
    };
    assert_eq!(v.color[3], 0.0);
}

#[test]
fn opaque_alpha_is_one() {
    let v = DebugVertex {
        position: [1.0, 2.0, 3.0],
        _pad: 0.0,
        color: [0.5, 0.3, 0.8, 1.0],
    };
    assert_eq!(v.color[3], 1.0);
}

// ── Vertex buffer stride ──────────────────────────────────────────────────────

#[test]
fn vertex_stride_matches_size_of_debug_vertex() {
    // The wgsl pipeline must use array_stride == size_of::<DebugVertex>().
    assert_eq!(size_of::<DebugVertex>(), 32);
}

#[test]
fn three_vertices_occupy_96_bytes() {
    let verts = [
        DebugVertex {
            position: [0.0; 3],
            _pad: 0.0,
            color: [1.0; 4],
        },
        DebugVertex {
            position: [1.0; 3],
            _pad: 0.0,
            color: [0.0, 0.0, 0.0, 1.0],
        },
        DebugVertex {
            position: [2.0; 3],
            _pad: 0.0,
            color: [0.5; 4],
        },
    ];
    let bytes_total: &[u8] = cast_slice(&verts);
    assert_eq!(bytes_total.len(), 3 * 32);
    assert_eq!(bytes_total.len(), 96);
}

#[test]
fn vertex_array_is_tightly_packed() {
    assert_eq!(size_of::<[DebugVertex; 5]>(), 5 * size_of::<DebugVertex>());
}

// ── Position value range ──────────────────────────────────────────────────────

#[test]
fn position_can_hold_negative_values() {
    let v = DebugVertex {
        position: [-100.0, -200.0, -300.0],
        _pad: 0.0,
        color: [1.0; 4],
    };
    assert_eq!(v.position[0], -100.0);
    assert_eq!(v.position[1], -200.0);
    assert_eq!(v.position[2], -300.0);
}

#[test]
fn position_can_hold_large_world_space_values() {
    let v = DebugVertex {
        position: [1e6, 2e6, 3e6],
        _pad: 0.0,
        color: [1.0; 4],
    };
    assert_eq!(v.position[0], 1e6f32);
}

#[test]
fn fractional_position_preserved() {
    let v = DebugVertex {
        position: [0.1, 0.2, 0.3],
        _pad: 0.0,
        color: [1.0; 4],
    };
    // f32 precision — exact bit match.
    assert_eq!(v.position, [0.1f32, 0.2f32, 0.3f32]);
}

// ── Padding field behaviour ───────────────────────────────────────────────────

#[test]
fn two_vertices_differing_only_in_pad_have_different_bytes() {
    let a = DebugVertex {
        position: [1.0, 2.0, 3.0],
        _pad: 0.0,
        color: [1.0; 4],
    };
    let b = DebugVertex {
        position: [1.0, 2.0, 3.0],
        _pad: 99.0, // different pad
        color: [1.0; 4],
    };
    // The bytes differ because _pad is part of the struct.
    assert_ne!(bytes_of(&a), bytes_of(&b));
}

#[test]
fn zeroed_vertex_pad_is_zero() {
    let v: DebugVertex = Zeroable::zeroed();
    assert_eq!(v._pad, 0.0f32);
}

// ── Byte encoding round-trips ─────────────────────────────────────────────────

#[test]
fn bytes_round_trip_preserves_position() {
    let v = DebugVertex {
        position: [3.14, 2.71, 1.41],
        _pad: 0.0,
        color: [0.0; 4],
    };
    let bytes = bytes_of(&v);
    let r: &DebugVertex = bytemuck::from_bytes(bytes);
    assert_eq!(r.position, v.position);
}

#[test]
fn bytes_round_trip_preserves_color() {
    let v = DebugVertex {
        position: [0.0; 3],
        _pad: 0.0,
        color: [0.9, 0.1, 0.5, 0.75],
    };
    let bytes = bytes_of(&v);
    let r: &DebugVertex = bytemuck::from_bytes(bytes);
    assert_eq!(r.color, v.color);
}

#[test]
fn cast_slice_for_empty_vertex_slice() {
    let verts: &[DebugVertex] = &[];
    let bytes: &[u8] = cast_slice(verts);
    assert_eq!(bytes.len(), 0);
}

#[test]
fn position_and_color_bytes_do_not_overlap() {
    let v = DebugVertex {
        position: [1.0, 0.0, 0.0],
        _pad: 0.0,
        color: [0.0, 0.0, 0.0, 0.0],
    };
    let bytes = bytes_of(&v);
    // position occupies bytes [0,12); color occupies bytes [16, 32).
    // Bytes 12–16 are the pad.
    let pos_bytes: &[u8] = &bytes[0..12];
    let col_bytes: &[u8] = &bytes[16..32];
    // position encodes 1.0 in first 4 bytes; color starts with 0.0.
    let pos_x = f32::from_le_bytes(pos_bytes[0..4].try_into().unwrap());
    let col_r = f32::from_le_bytes(col_bytes[0..4].try_into().unwrap());
    assert_eq!(pos_x, 1.0);
    assert_eq!(col_r, 0.0);
}

#[test]
fn vertex_as_f32_slice_via_cast_has_eight_elements() {
    let v = DebugVertex {
        position: [1.0, 2.0, 3.0],
        _pad: 0.0,
        color: [1.0, 0.5, 0.0, 1.0],
    };
    let bytes = bytes_of(&v);
    // 32 bytes / 4 bytes per f32 = 8 f32 values.
    let floats: &[f32] = bytemuck::cast_slice(bytes);
    assert_eq!(floats.len(), 8);
}

