//! Fullscreen quad visualization shader.
//!
//! Renders heatmap overlay blended with scene color based on selected performance metric.

struct VisualizeParams {
    mode: u32,              // PerfOverlayMode as u32
    num_tiles_x: u32,
    num_tiles_y: u32,
    screen_width: u32,
    screen_height: u32,
    opacity: f32,           // Blend factor (0.0-1.0)
    heatmap_scale: f32,     // Max value for normalization
    _pad0: u32,
    _pad1: u32,
}

struct TileMetrics {
    pass_overdraw_max: u32,
    light_count: u32,
    complexity_avg: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: VisualizeParams;
@group(0) @binding(1) var<storage, read> pass_overdraw: array<u32>;
@group(0) @binding(2) var<storage, read> tile_metrics: array<TileMetrics>;
@group(0) @binding(3) var scene_color: texture_2d<f32>;
@group(0) @binding(4) var scene_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

/// Fullscreen triangle vertex shader.
@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOut {
    var out: VertexOut;

    // Fullscreen triangle (covers entire viewport with one triangle)
    let x = f32((vertex_idx << 1u) & 2u);
    let y = f32(vertex_idx & 2u);

    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);

    return out;
}

/// Turbo colormap (blue → cyan → green → yellow → red).
///
/// Provides perceptually uniform color gradient for heatmaps.
fn heatmap_color(value: f32) -> vec3<f32> {
    let t = clamp(value, 0.0, 1.0);

    if t < 0.25 {
        // Blue → Cyan
        return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), t * 4.0);
    } else if t < 0.5 {
        // Cyan → Green
        return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), (t - 0.25) * 4.0);
    } else if t < 0.75 {
        // Green → Yellow
        return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), (t - 0.5) * 4.0);
    } else {
        // Yellow → Red
        return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), (t - 0.75) * 4.0);
    }
}

/// Fragment shader: heatmap overlay blended with scene.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Early discard when disabled
    if params.mode == 0u {
        discard;
    }

    // Sample scene color
    let scene = textureSample(scene_color, scene_sampler, in.uv).rgb;

    // Get tile index from UV
    let pixel_x = u32(clamp(in.uv.x * f32(params.screen_width), 0.0, f32(params.screen_width - 1u)));
    let pixel_y = u32(clamp(in.uv.y * f32(params.screen_height), 0.0, f32(params.screen_height - 1u)));
    let pixel_idx = pixel_y * params.screen_width + pixel_x;

    var overlay_color = vec3<f32>(0.0);

    if params.mode == 1u {
        // Pass-to-pass overdraw: direct per-pixel counter
        let raw_count = pass_overdraw[pixel_idx];
        let normalized = min(f32(raw_count) / params.heatmap_scale, 1.0);
        overlay_color = heatmap_color(normalized);
        return vec4<f32>(overlay_color, 1.0);
    }

    let tile_x = u32(in.uv.x * f32(params.num_tiles_x));
    let tile_y = u32(in.uv.y * f32(params.num_tiles_y));
    let tile_idx = tile_y * params.num_tiles_x + tile_x;
    let metrics = tile_metrics[tile_idx];

    if params.mode == 2u {
        // Shader complexity (0-200 scale)
        let normalized = f32(metrics.complexity_avg) / 200.0;
        overlay_color = heatmap_color(normalized);
    } else if params.mode == 3u {
        // Tile light count (0-64 scale)
        let normalized = f32(metrics.light_count) / 64.0;
        overlay_color = heatmap_color(normalized);
    }

    // Blend overlay with scene for non-overdraw modes
    let blended = mix(scene, overlay_color, params.opacity);
    return vec4<f32>(blended, 1.0);
}
