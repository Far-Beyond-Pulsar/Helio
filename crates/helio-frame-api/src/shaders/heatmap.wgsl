// The frame's heatmap colours (`helio_frame_api::diagnostics::HEATMAP_SNIPPET`):
// the ramp every diagnostic view draws with, so a kind's cost view and the
// overlay's own views read the same.

/// Blue, cyan, green, yellow, red for `value` in 0..1.
fn heatmap_color(value: f32) -> vec3<f32> {
    let t = clamp(value, 0.0, 1.0);

    if t < 0.25 {
        return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0);
    } else if t < 0.5 {
        return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
    } else if t < 0.75 {
        return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
    } else {
        return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
    }
}

const HEATMAP_SCALE_BAR_HEIGHT: f32 = 40.0;

/// The legend strip along the bottom of the display: the ramp left to right
/// with tick marks. `uv` is the display position in 0..1; `.w` is 0 above
/// the strip (the view draws its own colour there) and 1 inside it.
fn heatmap_scale_bar(uv: vec2<f32>, display_size: vec2<f32>) -> vec4<f32> {
    let y = clamp(uv.y * display_size.y, 0.0, display_size.y - 1.0);
    let start_y = display_size.y - HEATMAP_SCALE_BAR_HEIGHT;
    if y < start_y {
        return vec4<f32>(0.0);
    }
    let local_y = y - start_y;
    let is_top_border = local_y < 2.0;
    let is_bottom_border = local_y >= HEATMAP_SCALE_BAR_HEIGHT - 2.0;

    // Ticks at 0%, 25%, 50%, 75%, 100%.
    let tick_width = 2.0 / display_size.x;
    let is_tick = (abs(uv.x - 0.0) < tick_width) ||
                  (abs(uv.x - 0.25) < tick_width) ||
                  (abs(uv.x - 0.5) < tick_width) ||
                  (abs(uv.x - 0.75) < tick_width) ||
                  (abs(uv.x - 1.0) < tick_width);
    let is_tick_mark = is_tick && (local_y < 12.0 || local_y >= HEATMAP_SCALE_BAR_HEIGHT - 12.0);

    if is_top_border || is_bottom_border || is_tick_mark {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    return vec4<f32>(heatmap_color(uv.x), 1.0);
}
