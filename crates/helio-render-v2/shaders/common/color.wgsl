//! Color space conversions

/// Linear to sRGB conversion
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    let cutoff = linear < vec3(0.0031308);
    let higher = vec3(1.055) * pow(linear, vec3(1.0 / 2.4)) - vec3(0.055);
    let lower = linear * vec3(12.92);
    return select(higher, lower, cutoff);
}

/// sRGB to linear conversion
fn srgb_to_linear(srgb: vec3<f32>) -> vec3<f32> {
    let cutoff = srgb < vec3(0.04045);
    let higher = pow((srgb + vec3(0.055)) / vec3(1.055), vec3(2.4));
    let lower = srgb / vec3(12.92);
    return select(higher, lower, cutoff);
}

/// ACES filmic tone mapping
fn tone_map_aces(color: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

/// Reinhard tone mapping
fn tone_map_reinhard(color: vec3<f32>) -> vec3<f32> {
    return color / (color + vec3(1.0));
}

/// Uncharted 2 tone mapping
fn tone_map_uncharted2_partial(x: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

fn tone_map_uncharted2(color: vec3<f32>) -> vec3<f32> {
    let exposure_bias = 2.0;
    let curr = tone_map_uncharted2_partial(color * exposure_bias);
    let W = vec3(11.2);
    let white_scale = vec3(1.0) / tone_map_uncharted2_partial(W);
    return curr * white_scale;
}

/// Apply exposure
fn apply_exposure(color: vec3<f32>, exposure: f32) -> vec3<f32> {
    return color * exp2(exposure);
}
