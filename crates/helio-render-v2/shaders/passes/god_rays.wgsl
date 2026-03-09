//! God-ray / crepuscular-ray pass.
//!
//! Radial blur outward from the sun's screen-space position, attenuated by
//! the depth buffer (occluded regions don't contribute light shafts).
//!
//! The pass runs AFTER the deferred lighting pass and composites additively
//! onto the HDR scene-colour buffer via wgpu additive blend.

struct GodRayUniforms {
    sun_screen_pos:  vec2<f32>,   // sun projected to [0..1] UV
    density:         f32,          // sample step density (0.8–1.0 typical)
    decay:           f32,          // intensity decay per step (0.93–0.98)
    weight:          f32,          // overall intensity
    exposure:        f32,          // scaling factor
    num_samples:     u32,          // 64–128
    _pad:            f32,
}

@group(0) @binding(0) var scene_color: texture_2d<f32>;
@group(0) @binding(1) var scene_depth: texture_depth_2d;
@group(0) @binding(2) var linear_samp: sampler;
@group(0) @binding(3) var <uniform> u: GodRayUniforms;

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

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let uv  = in.uv;
    let dim = vec2<f32>(textureDimensions(scene_depth));

    // Direction from current screen-pixel toward the sun
    let delta       = (u.sun_screen_pos - uv) * (u.density / f32(u.num_samples));
    var step_uv     = uv;
    var illumination_decay = 1.0;
    var result      = vec3<f32>(0.0);

    for (var i = 0u; i < u.num_samples; i++) {
        step_uv    += delta;
        // Clamp UV to avoid wrap-around artefacts
        let s_uv    = clamp(step_uv, vec2<f32>(0.001), vec2<f32>(0.999));
        // Read depth: sky (depth ≈ 1.0) contributes; geometry occludes
        let pix     = vec2<i32>(vec2<i32>(s_uv * dim));
        let depth   = textureLoad(scene_depth, pix, 0);
        let sky_fac = select(0.0, 1.0, depth >= 0.9999);   // only pure sky pixels
        let samp    = textureSample(scene_color, linear_samp, s_uv).rgb;
        result     += samp * illumination_decay * sky_fac;
        illumination_decay *= u.decay;
    }

    result *= u.exposure * u.weight / f32(u.num_samples);
    return vec4<f32>(result, 1.0);
}
