// taa.wgsl — Temporal Anti-Aliasing with velocity reprojection + history blend.

@group(0) @binding(0) var current_tex:  texture_2d<f32>;
@group(0) @binding(1) var history_tex:  texture_2d<f32>;
@group(0) @binding(2) var velocity_tex: texture_2d<f32>;
@group(0) @binding(3) var smp:          sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

// Luminance.
fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// Neighborhood AABB clipping of history color.
fn clip_aabb(aabb_min: vec3<f32>, aabb_max: vec3<f32>, q: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let p_clip = 0.5 * (aabb_max + aabb_min);
    let e_clip = 0.5 * (aabb_max - aabb_min) + 0.0001;
    let v_clip = q - p_clip;
    let v_unit = v_clip / e_clip;
    let a_unit = abs(v_unit);
    let ma_unit = max(a_unit.x, max(a_unit.y, a_unit.z));
    if ma_unit > 1.0 {
        return p_clip + v_clip / ma_unit;
    }
    return q;
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims     = vec2<f32>(textureDimensions(current_tex));
    let inv_dims = 1.0 / dims;
    let uv = frag_coord.xy * inv_dims;

    // Sample velocity for reprojection.
    let velocity = textureSample(velocity_tex, smp, uv).rg;
    let prev_uv  = uv - velocity;

    // Current sample.
    let current = textureSample(current_tex, smp, uv).rgb;

    // Neighborhood min/max for history clipping.
    var color_min = current;
    var color_max = current;
    for (var dx = -1; dx <= 1; dx++) {
        for (var dy = -1; dy <= 1; dy++) {
            let offset = vec2<f32>(f32(dx), f32(dy)) * inv_dims;
            let nb = textureSample(current_tex, smp, uv + offset).rgb;
            color_min = min(color_min, nb);
            color_max = max(color_max, nb);
        }
    }

    // History sample with AABB clip.
    var history = textureSample(history_tex, smp, prev_uv).rgb;
    history = clip_aabb(color_min, color_max, history, current);

    // Blend factor: higher = more history (ghosting vs. noise tradeoff).
    let blend_factor = 0.1;

    // Luminance-weighted to reduce flicker.
    let luma_curr = luma(current);
    let luma_hist = luma(history);
    let luma_w = 1.0 / (1.0 + luma_hist);  // weight history less if it's brighter
    let final_w = mix(blend_factor, blend_factor * luma_w, 0.5);

    let out = mix(history, current, final_w);
    return vec4<f32>(out, 1.0);
}
