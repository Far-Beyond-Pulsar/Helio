// LUT builder (the post settings' `GpuPostProcessUniforms` come from
// `post_settings.wgsl`, inserted after this line by `with_post_settings`).

@group(0) @binding(0) var<uniform> postprocess: GpuPostProcessUniforms;
@group(0) @binding(1) var lut_output: texture_storage_3d<rgba16float, write>;

override lut_size: u32 = 16u;

fn hue_shift_rgb(c: vec3<f32>, shift_deg: f32) -> vec3<f32> {
    if abs(shift_deg) < 0.001 { return c; }
    let angle = shift_deg * 3.14159265 / 180.0;
    let cos_a = cos(angle);
    let sin_a = sin(angle);
    let m = mat3x3<f32>(
        vec3<f32>(0.213, 0.213 - 0.213 * cos_a + 0.144 * sin_a, 0.213 - 0.213 * cos_a - 0.756 * sin_a),
        vec3<f32>(0.715, 0.715 - 0.715 * cos_a - 0.283 * sin_a, 0.715 - 0.715 * cos_a + 0.416 * sin_a),
        vec3<f32>(0.072, 0.072 - 0.072 * cos_a + 0.860 * sin_a, 0.072 - 0.072 * cos_a - 0.461 * sin_a),
    );
    return m * c;
}

fn apply_lift_gamma_gain(c: vec3<f32>) -> vec3<f32> {
    let lift  = postprocess.lift_color;
    let gamma = postprocess.gamma_color;
    let gain  = postprocess.gain_color;
    var result = c * (vec3<f32>(1.0) - lift) + lift;
    result = pow(max(result, vec3<f32>(0.0)), gamma + vec3<f32>(1.0));
    result = result * gain;
    return result;
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if any(gid >= vec3<u32>(lut_size)) { return; }
    let coord = (vec3<f32>(gid) + 0.5) / f32(lut_size);
    var graded = hue_shift_rgb(coord, postprocess.hue_shift);
    graded = apply_lift_gamma_gain(graded);
    textureStore(lut_output, vec3<i32>(gid), vec4<f32>(graded, 1.0));
}
