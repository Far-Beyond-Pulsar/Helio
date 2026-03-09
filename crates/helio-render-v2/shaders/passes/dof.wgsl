//! Depth-of-Field (Bokeh DoF) pass.
//!
//! Single-pass circular CoC + Gaussian separable blur.
//!
//! Circle of Confusion (CoC) radius in pixels:
//!   CoC = |f^2 / (N × (focus_dist - f)) × (depth - focus_dist) / depth|
//!   simplified to:  CoC = aperture × |depth - focus_dist| / depth
//!
//! We use a separable approximation (horizontal + vertical pass) to keep
//! the cost low.  Near and far fields are blurred together — acceptable for
//! real-time use.  The blur radius is clamped to max_blur_px pixels.

struct DoFUniforms {
    focus_dist:  f32,   // world-space distance of the sharpest point
    aperture:    f32,   // larger = more blur
    max_blur_px: f32,   // hard limit on blur radius in pixels
    direction:   u32,   // 0 = horizontal, 1 = vertical
}

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var linear_samp: sampler;
@group(0) @binding(3) var <uniform> dof: DoFUniforms;

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    _pad:          f32,
    view_proj_inv: mat4x4<f32>,
}
@group(0) @binding(4) var <uniform> camera: Camera;

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)      uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    var pos = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    var uvs = array<vec2<f32>, 3>(vec2(0.0,1.0),   vec2(2.0,1.0),  vec2(0.0,-1.0));
    var o: VSOut; o.clip_pos = vec4<f32>(pos[vi], 0.0, 1.0); o.uv = uvs[vi]; return o;
}

fn linearize_depth(d: f32, near: f32, far: f32) -> f32 {
    return (near * far) / (far - d * (far - near));
}

// Gaussian kernel weight
fn gauss(x: f32, sigma: f32) -> f32 {
    return exp(-0.5 * x * x / (sigma * sigma + 0.0001));
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let pix    = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));
    let dim    = vec2<f32>(textureDimensions(scene_color));

    // Depth → approximate view-space linear depth using scene AABB heuristic
    // (proper near/far aren't passed; use a reasonable approximation)
    let raw_depth   = textureLoad(scene_depth, pix, 0);
    let ndc_z       = raw_depth;
    // Reconstruct view-space depth via inv-proj (w-component gives linear depth)
    let ndc_xy      = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    let clip_h      = camera.view_proj_inv * vec4<f32>(ndc_xy, ndc_z, 1.0);
    let view_pos    = clip_h.xyz / clip_h.w;
    let view_depth  = -view_pos.z;   // positive = in front of camera

    // CoC radius in pixels
    let coc_norm    = dof.aperture * abs(view_depth - dof.focus_dist) / max(view_depth, 0.0001);
    let coc_px      = clamp(coc_norm * dim.y, 0.0, dof.max_blur_px);

    if coc_px < 0.5 {
        return textureSample(scene_color, linear_samp, in.uv);
    }

    // Separable Gaussian blur with radius = coc_px pixels
    let sigma    = coc_px * 0.5;
    let samples  = i32(ceil(coc_px * 2.0));    // ±coc_px samples
    var col_sum  = vec3<f32>(0.0);
    var w_sum    = 0.0;

    for (var k = -samples; k <= samples; k++) {
        let offset = select(
            vec2<f32>(f32(k) / dim.x, 0.0),
            vec2<f32>(0.0, f32(k) / dim.y),
            dof.direction != 0u
        );
        let samp_uv = clamp(in.uv + offset, vec2<f32>(0.001), vec2<f32>(0.999));
        let w       = gauss(f32(k), sigma);
        col_sum    += textureSample(scene_color, linear_samp, samp_uv).rgb * w;
        w_sum      += w;
    }

    return vec4<f32>(col_sum / max(w_sum, 0.0001), 1.0);
}
