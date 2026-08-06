//!use helio_prelude
// ── DOF CoC Pre-Pass (Compute) ──────────────────────────────────────────────
//
// Reads depth at full resolution, computes circle-of-confusion per pixel,
// writes half-resolution CoC buffer.
//
// Outputs:
//   coc_tex  — R32Float, half-resolution CoC buffer

const WG_X: u32 = 16u;
const WG_Y: u32 = 16u;

// Mirror of the CPU-side GpuCameraUniforms — same layout as postprocess.wgsl.
struct CameraUniforms {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

// DOF block of GpuPostProcessUniforms, bound at byte offset 224.
// Field names match the Rust struct for clarity.
struct DofUniforms {
    dof_focal_distance:     f32,
    dof_focal_region:       f32,
    dof_aperture_shape:     f32,
    dof_aperture_rotation:  f32,
    dof_near_transition:    f32,
    dof_far_transition:     f32,
    dof_max_bokeh_size:     f32,
    dof_sensor_diagonal:    f32,
}

@group(0) @binding(0) var<uniform> dof: DofUniforms;
@group(0) @binding(1) var<storage, read> cameras: array<CameraUniforms, CAMERA_SLOTS>;
@group(0) @binding(2) var depth_tex: texture_depth_2d;
@group(0) @binding(3) var coc_tex: texture_storage_2d<r32float, write>;

fn linearize_depth(raw: f32) -> f32 {
    return -cameras[0].proj[3][2] / (raw * 2.0 - 1.0 + cameras[0].proj[2][2]);
}

fn compute_coc(linear_depth: f32) -> f32 {
    let focal_dist = dof.dof_focal_distance;
    let focal_region = dof.dof_focal_region;
    let near_blur = max(focal_dist - focal_region - linear_depth, 0.0)
        / max(dof.dof_near_transition, 0.001);
    let far_blur = max(linear_depth - (focal_dist + focal_region), 0.0)
        / max(dof.dof_far_transition, 0.001);
    let coc = max(near_blur, far_blur) * dof.dof_sensor_diagonal * 0.02;
    return clamp(coc, 0.0, dof.dof_max_bokeh_size);
}

@compute @workgroup_size(WG_X, WG_Y)
fn cs_coc_prepass(@builtin(global_invocation_id) gid: vec3<u32>) {
    let depth_dims = textureDimensions(depth_tex);
    let half_w = (depth_dims.x + 1u) / 2u;
    let half_h = (depth_dims.y + 1u) / 2u;

    if (gid.x >= half_w || gid.y >= half_h) { return; }

    // Full-res pixel position (center of the half-res cell)
    let px = gid.x * 2u;
    let py = gid.y * 2u;

    // Sample depth at full resolution (top-left of 2x2 quad)
    let raw_depth = textureLoad(depth_tex, vec2<i32>(i32(px), i32(py)), 0);
    let linear_depth = linearize_depth(raw_depth);
    let coc = compute_coc(linear_depth);

    textureStore(coc_tex, vec2<i32>(gid.xy), vec4<f32>(coc, 0.0, 0.0, 0.0));
}
