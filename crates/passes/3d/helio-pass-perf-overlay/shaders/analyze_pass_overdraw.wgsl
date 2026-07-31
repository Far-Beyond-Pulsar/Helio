//! Pass-to-pass overdraw analysis compute shader.
//!
//! Compares current depth buffer with previous frame's snapshot to detect
//! when passes overwrite each other's pixels.

struct DepthCompareParams {
    screen_width: u32,
    screen_height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: DepthCompareParams;
@group(0) @binding(1) var depth_prev: texture_depth_2d;
@group(0) @binding(2) var depth_current: texture_depth_2d;
@group(0) @binding(3) var<storage, read_write> pass_overdraw: array<atomic<u32>>;

/// Analyzes pass-to-pass overdraw by comparing depth values.
///
/// Dispatch: (width/16, height/16, 1) with workgroup_size(16, 16, 1)
@compute @workgroup_size(16, 16, 1)
fn analyze_pass_overdraw(@builtin(global_invocation_id) gid: vec3<u32>) {
    let coord = gid.xy;
    if coord.x >= params.screen_width || coord.y >= params.screen_height {
        return;
    }

    let depth_before = textureLoad(depth_prev, coord, 0);
    let depth_after = textureLoad(depth_current, coord, 0);

    // If depth changed and current is valid, this pass overwrote a previous pass
    if abs(depth_after - depth_before) > 0.0001 && depth_after < 1.0 {
        let pixel_idx = coord.y * params.screen_width + coord.x;
        atomicAdd(&pass_overdraw[pixel_idx], 1u);
    }
}
