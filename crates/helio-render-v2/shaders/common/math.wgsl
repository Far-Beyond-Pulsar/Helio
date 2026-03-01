//! Common math utilities

const PI: f32 = 3.14159265359;
const INV_PI: f32 = 0.31830988618;
const EPSILON: f32 = 1e-6;

/// Saturate (clamp to [0, 1])
fn saturate(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

/// Saturate vec3
fn saturate_vec3(v: vec3<f32>) -> vec3<f32> {
    return clamp(v, vec3(0.0), vec3(1.0));
}

/// Safe normalize (returns zero vector if length is too small)
fn safe_normalize(v: vec3<f32>) -> vec3<f32> {
    let len_sq = dot(v, v);
    if (len_sq < EPSILON) {
        return vec3(0.0);
    }
    return v / sqrt(len_sq);
}

/// Pow5 (x^5) - common in graphics
fn pow5(x: f32) -> f32 {
    let x2 = x * x;
    return x2 * x2 * x;
}

/// Luminance from RGB
fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

/// Remap value from one range to another
fn remap(value: f32, from_min: f32, from_max: f32, to_min: f32, to_max: f32) -> f32 {
    let t = (value - from_min) / (from_max - from_min);
    return mix(to_min, to_max, t);
}
