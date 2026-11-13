//! Fullscreen quad visualization shader.
//!
//! Renders raw heatmap data for performance metrics, with the frame's heatmap
//! colours (`helio_frame_api::diagnostics::HEATMAP_SNIPPET`).
//!use frame_heatmap

struct VisualizeParams {
    mode: u32,              // PerfOverlayMode as u32
    num_tiles_x: u32,
    num_tiles_y: u32,
    internal_width: u32,    // Buffer dimensions (internal resolution)
    internal_height: u32,
    display_width: u32,     // Target dimensions (display resolution)
    display_height: u32,
    heatmap_scale: f32,     // Max value for normalization
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

struct TileMetrics {
    pass_overdraw_max: u32,
    light_count: u32,
}

@group(0) @binding(0) var<uniform> params: VisualizeParams;
@group(0) @binding(1) var<storage, read> pass_overdraw: array<u32>;
@group(0) @binding(2) var<storage, read> tile_metrics: array<TileMetrics>;

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

/// Fragment shader: raw heatmap data visualization.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Early discard when disabled
    if params.mode == 0u {
        discard;
    }

    // Map display UV to internal buffer coordinates
    // Display is at full resolution (e.g., 1920×1080)
    // Internal buffers are at render resolution (e.g., 1440×810)
    let internal_x = u32(clamp(in.uv.x * f32(params.internal_width), 0.0, f32(params.internal_width - 1u)));
    let internal_y = u32(clamp(in.uv.y * f32(params.internal_height), 0.0, f32(params.internal_height - 1u)));
    let pixel_idx = internal_y * params.internal_width + internal_x;

    let scale_bar = heatmap_scale_bar(
        in.uv,
        vec2<f32>(f32(params.display_width), f32(params.display_height)),
    );
    if scale_bar.w > 0.0 {
        return scale_bar;
    }

    var overlay_color = vec3<f32>(0.0);

    if params.mode == 1u {
        // Pass-to-pass overdraw: direct per-pixel counter
        let raw_count = pass_overdraw[pixel_idx];
        let normalized = min(f32(raw_count) / params.heatmap_scale, 1.0);
        overlay_color = heatmap_color(normalized);
        return vec4<f32>(overlay_color, 1.0);
    }

    // Calculate exact tile coordinates from internal resolution pixel coordinates
    // This ensures pixel-perfect mapping instead of approximation from UV
    let tile_x = internal_x / 16u;  // TILE_SIZE = 16
    let tile_y = internal_y / 16u;
    let tile_idx = tile_y * params.num_tiles_x + tile_x;
    let metrics = tile_metrics[tile_idx];

    if params.mode == 3u {
        // Tile light count (0-64 scale)
        let normalized = f32(metrics.light_count) / 64.0;
        overlay_color = heatmap_color(normalized);
        return vec4<f32>(overlay_color, 1.0);
    }

    // Fallback: black
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}
