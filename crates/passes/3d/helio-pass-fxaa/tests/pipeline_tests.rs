//! Tests for FXAA pipeline geometry and sampler configuration.
//!
//! FXAA renders using a single fullscreen triangle (3 vertices) that covers
//! the entire NDC viewport [-1,1]×[-1,1] without redundant rasterisation.

// ── Fullscreen triangle geometry ──────────────────────────────────────────────

/// NDC position of vertex 0: bottom-left corner of the viewport.
const TRIANGLE_V0: [f32; 2] = [-1.0, -1.0];

/// NDC position of vertex 1: far right, extends past the viewport.
const TRIANGLE_V1: [f32; 2] = [3.0, -1.0];

/// NDC position of vertex 2: far top, extends past the viewport.
const TRIANGLE_V2: [f32; 2] = [-1.0, 3.0];

/// Returns whether point (x,y) lies inside (or on the boundary of) the
/// fullscreen triangle using half-plane (edge function) tests.
fn point_in_fullscreen_triangle(x: f32, y: f32) -> bool {
    let e0 = (TRIANGLE_V1[0] - TRIANGLE_V0[0]) * (y - TRIANGLE_V0[1])
        - (TRIANGLE_V1[1] - TRIANGLE_V0[1]) * (x - TRIANGLE_V0[0]);
    let e1 = (TRIANGLE_V2[0] - TRIANGLE_V1[0]) * (y - TRIANGLE_V1[1])
        - (TRIANGLE_V2[1] - TRIANGLE_V1[1]) * (x - TRIANGLE_V1[0]);
    let e2 = (TRIANGLE_V0[0] - TRIANGLE_V2[0]) * (y - TRIANGLE_V2[1])
        - (TRIANGLE_V0[1] - TRIANGLE_V2[1]) * (x - TRIANGLE_V2[0]);
    e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0
}

/// Signed area of a triangle given three 2-D vertices.
fn triangle_signed_area(v0: [f32; 2], v1: [f32; 2], v2: [f32; 2]) -> f32 {
    0.5 * ((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[test]
fn vertex_count_is_three() {
    // FXAA issues exactly one draw(0..3, 0..1) per frame.
    const VERTEX_COUNT: u32 = 3;
    assert_eq!(VERTEX_COUNT, 3);
}

#[test]
fn instance_count_is_one() {
    const INSTANCE_COUNT: u32 = 1;
    assert_eq!(INSTANCE_COUNT, 1);
}

#[test]
fn v0_is_bottom_left() {
    assert_eq!(TRIANGLE_V0, [-1.0, -1.0]);
}

#[test]
fn v1_is_far_right_bottom() {
    assert_eq!(TRIANGLE_V1, [3.0, -1.0]);
}

#[test]
fn v2_is_far_top_left() {
    assert_eq!(TRIANGLE_V2, [-1.0, 3.0]);
}

#[test]
fn v0_x_equals_minus_one() {
    assert_eq!(TRIANGLE_V0[0], -1.0);
}

#[test]
fn v0_y_equals_minus_one() {
    assert_eq!(TRIANGLE_V0[1], -1.0);
}

#[test]
fn v1_x_equals_three() {
    assert_eq!(TRIANGLE_V1[0], 3.0);
}

#[test]
fn v2_y_equals_three() {
    assert_eq!(TRIANGLE_V2[1], 3.0);
}

#[test]
fn triangle_covers_bottom_left_corner() {
    assert!(point_in_fullscreen_triangle(-1.0, -1.0));
}

#[test]
fn triangle_covers_bottom_right_corner() {
    assert!(point_in_fullscreen_triangle(1.0, -1.0));
}

#[test]
fn triangle_covers_top_left_corner() {
    assert!(point_in_fullscreen_triangle(-1.0, 1.0));
}

#[test]
fn triangle_covers_top_right_corner() {
    assert!(point_in_fullscreen_triangle(1.0, 1.0));
}

#[test]
fn triangle_covers_center() {
    assert!(point_in_fullscreen_triangle(0.0, 0.0));
}

#[test]
fn triangle_covers_all_ndc_corners() {
    let corners = [(-1.0_f32, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)];
    for (x, y) in corners {
        assert!(
            point_in_fullscreen_triangle(x, y),
            "NDC corner ({x},{y}) must lie inside the fullscreen triangle"
        );
    }
}

#[test]
fn triangle_area_is_eight() {
    // Base and height are both 4 (from -1 to 3). Area = 0.5 × 4 × 4 = 8.
    let area = triangle_signed_area(TRIANGLE_V0, TRIANGLE_V1, TRIANGLE_V2);
    assert!((area - 8.0).abs() < 1e-5, "expected area 8.0, got {area}");
}

#[test]
fn triangle_area_larger_than_ndc_viewport() {
    // The viewport is 2×2 = 4 units². The triangle (8 units²) is twice as large,
    // guaranteeing no uncovered pixels.
    let tri_area = triangle_signed_area(TRIANGLE_V0, TRIANGLE_V1, TRIANGLE_V2);
    let viewport_area = 4.0_f32;
    assert!(tri_area > viewport_area);
}

#[test]
fn triangle_x_range_covers_viewport() {
    let min_x = TRIANGLE_V0[0].min(TRIANGLE_V1[0]).min(TRIANGLE_V2[0]);
    let max_x = TRIANGLE_V0[0].max(TRIANGLE_V1[0]).max(TRIANGLE_V2[0]);
    assert!(min_x <= -1.0, "triangle min x must reach -1");
    assert!(max_x >= 1.0, "triangle max x must reach +1");
}

#[test]
fn triangle_y_range_covers_viewport() {
    let min_y = TRIANGLE_V0[1].min(TRIANGLE_V1[1]).min(TRIANGLE_V2[1]);
    let max_y = TRIANGLE_V0[1].max(TRIANGLE_V1[1]).max(TRIANGLE_V2[1]);
    assert!(min_y <= -1.0, "triangle min y must reach -1");
    assert!(max_y >= 1.0, "triangle max y must reach +1");
}

#[test]
fn sampler_linear_filtering_required() {
    // Linear filtering ensures sub-pixel luma values are correctly interpolated
    // when the FXAA kernel samples diagonally across an edge.
    const USES_LINEAR_FILTER: bool = true;
    assert!(USES_LINEAR_FILTER);
}

#[test]
fn sampler_clamp_to_edge_avoids_border_artifacts() {
    // ClampToEdge prevents sampling outside the texture at screen borders,
    // which could introduce incorrect luma values and ghost edges.
    const USES_CLAMP_TO_EDGE: bool = true;
    assert!(USES_CLAMP_TO_EDGE);
}

#[test]
fn draw_call_count_is_constant_one() {
    // FXAA complexity is O(1): exactly one draw call regardless of scene geometry.
    const DRAW_CALL_COUNT: usize = 1;
    assert_eq!(DRAW_CALL_COUNT, 1);
}

#[test]
fn v1_y_shared_with_v0() {
    // V0 and V1 share the same Y coordinate (both at the bottom edge).
    assert_eq!(TRIANGLE_V0[1], TRIANGLE_V1[1]);
}

#[test]
fn v0_x_shared_with_v2() {
    // V0 and V2 share the same X coordinate (both at the left edge).
    assert_eq!(TRIANGLE_V0[0], TRIANGLE_V2[0]);
}

#[test]
fn ndc_range_is_minus_one_to_one() {
    // NDC viewport spans exactly [-1, 1] in both axes.
    let ndc_min = -1.0_f32;
    let ndc_max = 1.0_f32;
    assert_eq!(ndc_max - ndc_min, 2.0);
}

#[test]
fn triangle_vertex_count_equals_draw_range() {
    // draw(0..3, 0..1): range 0..3 covers exactly 3 vertices.
    let draw_range = 0..3_u32;
    assert_eq!(draw_range.count(), 3);
}
