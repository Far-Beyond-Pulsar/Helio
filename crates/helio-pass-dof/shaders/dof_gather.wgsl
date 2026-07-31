//!use helio_prelude
// ── DOF Gather Pass (Compute, Half-Resolution) ──────────────────────────────
//
// For each half-resolution pixel, gathers samples from the full-res scene colour
// within the CoC radius. Uses a Poisson disc sample pattern weighted by the
// bokeh aperture shape texture.
//
// Reads:   scene_colour (full res), coc_tex (half res), bokeh_shape (array)
// Writes:  near_blur (half res RGBA16F), far_blur (half res RGBA16F)
//
// When aperture_shape == 0 (Gaussian mode), the gather uses a circular
// Gaussian-weighted kernel matching the old fallback path.

const WG_X: u32 = 8u;
const WG_Y: u32 = 8u;
const POISSON_SAMPLES: u32 = 32u;
const BOKEH_SLICE_OFFSET: u32 = 3u; // slice 0 = 3 blades

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
@group(0) @binding(1) var<storage, read> cameras: array<CameraUniforms, 2>;
@group(0) @binding(2) var src_tex: texture_2d<f32>;
@group(0) @binding(3) var coc_tex: texture_2d<f32>;
@group(0) @binding(4) var bokeh_tex: texture_2d_array<f32>;
@group(0) @binding(5) var linear_samp: sampler;
@group(0) @binding(6) var near_blur: texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var far_blur: texture_storage_2d<rgba16float, write>;

// Pre-computed Poisson disc samples (32 samples in [-1, 1]^2)
// Generated with a Poisson disc sampling algorithm; the pattern is fixed
// to ensure temporal stability (no per-frame jitter).
fn poisson_sample(i: u32) -> vec2<f32> {
    let samples = array<vec2<f32>, 32>(
        vec2<f32>( 0.390,  0.258),
        vec2<f32>( 0.118,  0.682),
        vec2<f32>(-0.518,  0.399),
        vec2<f32>(-0.779, -0.135),
        vec2<f32>(-0.350, -0.476),
        vec2<f32>(-0.078, -0.818),
        vec2<f32>( 0.466, -0.607),
        vec2<f32>( 0.795, -0.128),
        vec2<f32>( 0.692,  0.561),
        vec2<f32>( 0.041,  0.938),
        vec2<f32>(-0.740,  0.613),
        vec2<f32>(-0.956,  0.210),
        vec2<f32>(-0.650, -0.674),
        vec2<f32>(-0.502, -0.862),
        vec2<f32>( 0.166, -0.960),
        vec2<f32>( 0.862, -0.470),
        vec2<f32>( 0.927,  0.263),
        vec2<f32>( 0.561,  0.826),
        vec2<f32>(-0.155,  0.920),
        vec2<f32>(-0.473,  0.879),
        vec2<f32>(-0.898, -0.394),
        vec2<f32>(-0.838, -0.465),
        vec2<f32>(-0.169, -0.980),
        vec2<f32>( 0.375, -0.921),
        vec2<f32>( 0.766, -0.618),
        vec2<f32>( 0.989,  0.078),
        vec2<f32>( 0.252,  0.952),
        vec2<f32>(-0.719,  0.684),
        vec2<f32>(-0.964, -0.122),
        vec2<f32>(-0.185, -0.614),
        vec2<f32>( 0.546,  0.098),
        vec2<f32>( 0.850,  0.385),
    );
    return samples[i];
}

fn sample_bokeh_weight(offset: vec2<f32>, blade_count: u32) -> f32 {
    let bokeh_uv = offset * 0.5 + 0.5;
    if blade_count >= 3u && blade_count <= 11u {
        let slice = blade_count - BOKEH_SLICE_OFFSET;
        return textureSampleLevel(bokeh_tex, linear_samp, bokeh_uv, i32(slice), 0.0).r;
    }
    let len2 = dot(offset, offset);
    return exp(-len2 * 4.0);
}

fn gather_blur(uv: vec2<f32>, coc_radius: f32, blade_count: u32, src_dims: vec2<f32>) -> vec3<f32> {
    var accumulated = vec3<f32>(0.0);
    var total_weight = 0.0;
    let radius = clamp(coc_radius, 0.5, dof.dof_max_bokeh_size);
    let scale = radius / 32.0;

    for (var i = 0u; i < POISSON_SAMPLES; i++) {
        let poisson = poisson_sample(i);
        let offset_ndc = poisson * scale;
        let offset_uv = offset_ndc * (1.0 / src_dims);
        let tap_uv = uv + offset_uv;

        let clamped = clamp(tap_uv, vec2<f32>(0.0), vec2<f32>(1.0));
        let tap = textureSampleLevel(src_tex, linear_samp, clamped, 0.0).rgb;
        let w = sample_bokeh_weight(poisson, blade_count);
        accumulated += tap * w;
        total_weight += w;
    }

    if total_weight > 0.0 {
        return accumulated / total_weight;
    }
    return textureSampleLevel(src_tex, linear_samp, uv, 0.0).rgb;
}

@compute @workgroup_size(WG_X, WG_Y)
fn cs_gather(@builtin(global_invocation_id) gid: vec3<u32>) {
    let src_dims = vec2<f32>(textureDimensions(src_tex));
    let coc_dims = textureDimensions(coc_tex);

    if (gid.x >= coc_dims.x || gid.y >= coc_dims.y) { return; }

    // Half-resolution UV
    let half_uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(coc_dims);
    // Corresponding full-resolution UV
    let full_uv = (vec2<f32>(gid.xy) * 2.0 + 0.5) / src_dims;

    let coc = textureLoad(coc_tex, vec2<i32>(gid.xy), 0).r;
    let blade_count = u32(max(dof.dof_aperture_shape, 0.0));

    if coc < 0.5 {
        let col = textureSampleLevel(src_tex, linear_samp, full_uv, 0.0).rgb;
        textureStore(near_blur, vec2<i32>(gid.xy), vec4<f32>(col, 0.0));
        textureStore(far_blur, vec2<i32>(gid.xy), vec4<f32>(col, 0.0));
        return;
    }

    let col_near = gather_blur(full_uv, coc, blade_count, src_dims);
    textureStore(near_blur, vec2<i32>(gid.xy), vec4<f32>(col_near, coc));
    textureStore(far_blur, vec2<i32>(gid.xy), vec4<f32>(col_near, coc));
}
