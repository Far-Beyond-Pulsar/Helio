// fxaa.wgsl — FXAA 3.11 quality preset (Timothy Lottes).
// Reads pre-aa texture; writes to swapchain surface.

@group(0) @binding(0) var input_tex: texture_2d<f32>;
@group(0) @binding(1) var smp:       sampler;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

const FXAA_EDGE_THRESHOLD:     f32 = 0.166;
const FXAA_EDGE_THRESHOLD_MIN: f32 = 0.0833;
const FXAA_SUBPIX_TRIM:        f32 = 0.0;
const FXAA_SUBPIX_CAP:         f32 = 0.75;
const FXAA_SEARCH_STEPS:       i32 = 16;

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims     = vec2<f32>(textureDimensions(input_tex));
    let inv_dims = 1.0 / dims;
    let uv = frag_coord.xy * inv_dims;

    let rgbNW = textureSample(input_tex, smp, uv + vec2<f32>(-1.0, -1.0) * inv_dims).rgb;
    let rgbNE = textureSample(input_tex, smp, uv + vec2<f32>( 1.0, -1.0) * inv_dims).rgb;
    let rgbSW = textureSample(input_tex, smp, uv + vec2<f32>(-1.0,  1.0) * inv_dims).rgb;
    let rgbSE = textureSample(input_tex, smp, uv + vec2<f32>( 1.0,  1.0) * inv_dims).rgb;
    let rgbM  = textureSample(input_tex, smp, uv).rgb;

    let lumaNW = luma(rgbNW);
    let lumaNE = luma(rgbNE);
    let lumaSW = luma(rgbSW);
    let lumaSE = luma(rgbSE);
    let lumaM  = luma(rgbM);

    let lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    let lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    let lumaRange = lumaMax - lumaMin;
    if lumaRange < max(FXAA_EDGE_THRESHOLD_MIN, lumaMax * FXAA_EDGE_THRESHOLD) {
        return vec4<f32>(rgbM, 1.0);
    }

    // Edge direction.
    let lumaS  = luma(textureSample(input_tex, smp, uv + vec2<f32>( 0.0,  1.0) * inv_dims).rgb);
    let lumaN  = luma(textureSample(input_tex, smp, uv + vec2<f32>( 0.0, -1.0) * inv_dims).rgb);
    let lumaE  = luma(textureSample(input_tex, smp, uv + vec2<f32>( 1.0,  0.0) * inv_dims).rgb);
    let lumaW  = luma(textureSample(input_tex, smp, uv + vec2<f32>(-1.0,  0.0) * inv_dims).rgb);

    let edgeH = abs(-2.0*lumaW + lumaNW + lumaSW) + abs(-2.0*lumaM + lumaN + lumaS) * 2.0 + abs(-2.0*lumaE + lumaNE + lumaSE);
    let edgeV = abs(-2.0*lumaN + lumaNW + lumaNE) + abs(-2.0*lumaM + lumaW + lumaE) * 2.0 + abs(-2.0*lumaS + lumaSW + lumaSE);

    let is_horz = edgeH >= edgeV;
    var step_dir = select(inv_dims.x, inv_dims.y, is_horz);

    let luma1 = select(lumaW, lumaN, is_horz);
    let luma2 = select(lumaE, lumaS, is_horz);
    let grad1  = abs(luma1 - lumaM);
    let grad2  = abs(luma2 - lumaM);
    let steepest = select(false, true, grad1 >= grad2);
    if !steepest { step_dir = -step_dir; }

    var uv_blend = uv;
    if is_horz {
        uv_blend.y += step_dir * 0.5;
    } else {
        uv_blend.x += step_dir * 0.5;
    }

    var uv1 = uv_blend;
    var uv2 = uv_blend;
    let search_step = select(vec2<f32>(0.0, inv_dims.y), vec2<f32>(inv_dims.x, 0.0), is_horz);
    uv1 -= search_step;
    uv2 += search_step;

    let luma_end_ref = lumaM + (select(luma1, luma2, steepest) - lumaM) * 0.5;

    var done1 = false;
    var done2 = false;
    var luma_end1 = 0.0f;
    var luma_end2 = 0.0f;

    for (var i = 0; i < FXAA_SEARCH_STEPS; i++) {
        if !done1 { luma_end1 = luma(textureSample(input_tex, smp, uv1).rgb); }
        if !done2 { luma_end2 = luma(textureSample(input_tex, smp, uv2).rgb); }
        done1 = abs(luma_end1 - luma_end_ref) >= lumaRange * 0.25;
        done2 = abs(luma_end2 - luma_end_ref) >= lumaRange * 0.25;
        if done1 && done2 { break; }
        if !done1 { uv1 -= search_step; }
        if !done2 { uv2 += search_step; }
    }

    let dist1 = select(uv.y - uv1.y, uv.x - uv1.x, is_horz);
    let dist2 = select(uv2.y - uv.y, uv2.x - uv.x, is_horz);
    let span_len = dist1 + dist2;
    let sub_pix_offset = select(dist2, dist1, dist1 < dist2) / span_len - 0.5;
    let sub_pix_offset_nn = sub_pix_offset * 2.0;
    let sub_pix_blend = max(
        0.0,
        (abs(sub_pix_offset_nn) - FXAA_SUBPIX_TRIM) / (1.0 - FXAA_SUBPIX_TRIM)
    );
    let sub_pix_blend_clamped = min(sub_pix_blend, FXAA_SUBPIX_CAP);

    var uv_final = uv;
    if is_horz {
        uv_final.y += sub_pix_offset_nn * 0.5 * inv_dims.y;
    } else {
        uv_final.x += sub_pix_offset_nn * 0.5 * inv_dims.x;
    }
    uv_final = uv_blend + search_step * sub_pix_offset_nn * 0.5;

    let blended = textureSample(input_tex, smp, uv_final).rgb;
    return vec4<f32>(mix(rgbM, blended, sub_pix_blend_clamped), 1.0);
}
