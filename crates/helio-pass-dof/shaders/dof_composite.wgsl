//!use helio_prelude
// ── DOF Composite Pass (Fullscreen Triangle, Full-Res) ──────────────────────
//
// Blends the half-resolution near/far blur buffers with the full-resolution
// in-focus image. Uses CoC-aware bilinear upsampling to prevent colour bleeding
// across the focal plane.
//
// Reads:   scene_colour (full res, in-focus), near_blur (half res), far_blur (half res), coc_tex (half res)
// Writes:  output (full res)

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

struct DofUniforms {
    focal_distance:     f32,
    focal_region:       f32,
    aperture_shape:     f32,
    aperture_rotation:  f32,
    near_transition:    f32,
    far_transition:     f32,
    max_bokeh_size:     f32,
    sensor_diagonal:    f32,
    enabled:            u32,
    _pad:               f32,
    blend_weight:       f32,
    _pad2:              f32,
}

@group(0) @binding(0) var<uniform> dof: DofUniforms;
@group(0) @binding(1) var src_tex: texture_2d<f32>;
@group(0) @binding(2) var near_blur: texture_2d<f32>;
@group(0) @binding(3) var far_blur: texture_2d<f32>;
@group(0) @binding(4) var coc_tex: texture_2d<f32>;
@group(0) @binding(5) var linear_samp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vi: u32) -> VOut {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return VOut(
        vec4<f32>(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0),
        vec2<f32>(x, y),
    );
}

fn coc_aware_sample(tex: texture_2d<f32>, half_uv: vec2<f32>, coc_tex: texture_2d<f32>, own_coc: f32) -> vec4<f32> {
    let half_dims = vec2<f32>(textureDimensions(tex));
    let fcoord = half_uv * half_dims - 0.5;
    let base = vec2<i32>(i32(floor(fcoord.x)), i32(floor(fcoord.y)));
    let frac = fcoord - vec2<f32>(base);

    var result = vec4<f32>(0.0);
    var total_w = 0.0;

    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            let coord = base + vec2<i32>(dx, dy);
            let clamped = clamp(coord, vec2<i32>(0), vec2<i32>(i32(half_dims.x) - 1, i32(half_dims.y) - 1));
            let sample_coc = textureLoad(coc_tex, clamped, 0).r;
            let coc_diff = abs(sample_coc - own_coc);
            let coc_weight = exp(-coc_diff * coc_diff * 0.5);
            let bilinear_w = (f32(1 - dx) - frac.x * f32(1 - 2 * dx))
                           * (f32(1 - dy) - frac.y * f32(1 - 2 * dy));
            let w = bilinear_w * coc_weight;
            let s = textureLoad(tex, clamped, 0);
            result += s * w;
            total_w += w;
        }
    }

    if total_w > 0.0 { result /= total_w; }
    return result;
}

@fragment
fn fs_composite(in: VOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src_tex));
    let uv = in.uv;

    let sharp = textureSampleLevel(src_tex, linear_samp, uv, 0.0).rgb;

    if dof.aperture_shape < 0.0 {
        return vec4<f32>(sharp, 1.0);
    }

    let half_dims = vec2<f32>(textureDimensions(near_blur));
    let half_uv = uv * half_dims / dims;

    let coc_coord = vec2<i32>(i32(half_uv.x), i32(half_uv.y));
    let coc = textureLoad(coc_tex, clamp(coc_coord, vec2<i32>(0), vec2<i32>(i32(half_dims.x) - 1, i32(half_dims.y) - 1)), 0).r;

    if coc < 0.5 {
        return vec4<f32>(sharp, 1.0);
    }

    let near = coc_aware_sample(near_blur, half_uv, coc_tex, coc);
    let far = coc_aware_sample(far_blur, half_uv, coc_tex, coc);

    let blend = clamp(coc / dof.max_bokeh_size, 0.0, 1.0);
    let blurred = mix(near.rgb, far.rgb, 0.5);

    return vec4<f32>(mix(sharp, blurred, blend), 1.0);
}
