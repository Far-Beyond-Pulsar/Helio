//! Pass-to-pass color overdraw analysis compute shader.
//!
//! Compares current color buffer with previous pass's snapshot to detect
//! when passes overwrite each other's pixels (wasted fragment shader work).

struct ColorCompareParams {
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: ColorCompareParams;
@group(0) @binding(1) var color_prev: texture_2d<f32>;
@group(0) @binding(2) var color_current: texture_2d<f32>;
@group(0) @binding(3) var<storage, read_write> pass_overdraw: array<atomic<u32>>;

/// Analyzes pass-to-pass overdraw by comparing color values.
///
/// Detects when a fragment shader wrote a pixel, only for that pixel to be
/// overwritten by a later rendering pass (wasted GPU work).
///
/// Dispatch: (width/16, height/16, 1) with workgroup_size(16, 16, 1)
@compute @workgroup_size(16, 16, 1)
fn analyze_color_overdraw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = gid.xy;
    if coord.x >= params.screen_width || coord.y >= params.screen_height {
        return;
    }

    let color_before = textureLoad(color_prev, coord, 0);
    let color_after = textureLoad(color_current, coord, 0);

    // Calculate color delta (linear RGB distance)
    let delta = color_after.rgb - color_before.rgb;
    let distance = length(delta);

    // If color changed significantly, a pass overwrote this pixel
    // Threshold: 0.01 in linear space (~2.5/255 perceptual difference)
    if distance > 0.01 {
        let pixel_idx = coord.y * params.screen_width + coord.x;
        atomicAdd(&pass_overdraw[pixel_idx], 1u);
    }
}
