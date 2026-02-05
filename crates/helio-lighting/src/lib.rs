// Helio Lighting - Modular shader functions for global illumination
// Based directly on blade-graphics ray-query example
// Provides composable WGSL functions that integrate with existing pipelines

/// GI shader functions that can be included in any WGSL shader
/// These are helper functions from blade's ray-query example
pub fn gi_functions_wgsl() -> &'static str {
    r#"
// Quaternion rotation (from blade ray-query example)
fn qrot(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

// Random unit vector for ray jittering
fn random_unit_vector(seed: u32, index: u32) -> vec3<f32> {
    let s = f32(seed * 747796405u + index * 2891336453u);
    let t = s * 0.00000000023283064365386962890625;
    let phi = t * 6.283185307179586;
    let cos_theta = 2.0 * fract(t) - 1.0;
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3<f32>(
        cos(phi) * sin_theta,
        sin(phi) * sin_theta,
        cos_theta
    );
}
"#
}

/// Minimal GI parameters that match blade's approach
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
pub struct GIConfig {
    pub intensity: f32,
    pub max_bounces: u32,
    pub num_samples: u32,
    pub _pad: u32,
}
