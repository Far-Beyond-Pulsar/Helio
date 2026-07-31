//! Material profiling shader.
//!
//! Renders test patches with specific material properties and measures GPU execution time.
//! This mimics the deferred lighting shader to get accurate timing for different materials.

struct MaterialParams {
    roughness: f32,
    metallic: f32,
    num_lights: u32,
    num_shadow_lights: u32,
}

struct LightData {
    position: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    intensity: f32,
    direction: vec3<f32>,
    _pad1: f32,
}

@group(0) @binding(0) var<uniform> params: MaterialParams;
@group(0) @binding(1) var<storage, read> lights: array<LightData>;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

/// Fullscreen triangle vertex shader.
@vertex
fn vs_main(@builtin(vertex_index) vertex_idx: u32) -> VertexOut {
    var out: VertexOut;
    
    // Fullscreen triangle
    let x = f32((vertex_idx << 1u) & 2u);
    let y = f32(vertex_idx & 2u);
    
    out.position = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(x, 1.0 - y);
    
    return out;
}

/// GGX/Trowbridge-Reitz normal distribution function.
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265359 * denom * denom);
}

/// Schlick-GGX geometry attenuation.
fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / (n_dot_v * (1.0 - k) + k);
}

/// Smith's geometry shadowing-masking function.
fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let ggx2 = geometry_schlick_ggx(n_dot_v, roughness);
    let ggx1 = geometry_schlick_ggx(n_dot_l, roughness);
    return ggx1 * ggx2;
}

/// Fresnel-Schlick approximation.
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

/// Cook-Torrance BRDF (metallic workflow).
fn cook_torrance_brdf(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    roughness: f32,
    metallic: f32,
    base_color: vec3<f32>,
) -> vec3<f32> {
    let h = normalize(v + l);
    
    let n_dot_v = max(dot(n, v), 0.001);
    let n_dot_l = max(dot(n, l), 0.001);
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);
    
    // F0 for dielectrics is 0.04, for metals use base color
    let f0 = mix(vec3(0.04), base_color, metallic);
    
    // Cook-Torrance specular BRDF
    let ndf = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);
    
    let numerator = ndf * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l;
    let specular = numerator / max(denominator, 0.001);
    
    // Diffuse (Lambertian)
    let k_d = (vec3(1.0) - f) * (1.0 - metallic);
    let diffuse = k_d * base_color / 3.14159265359;
    
    return (diffuse + specular) * n_dot_l;
}

/// Fragment shader: render test patch with full BRDF calculations.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // Fixed surface properties for test patch
    let base_color = vec3<f32>(0.8, 0.8, 0.8);
    let normal = vec3<f32>(0.0, 0.0, 1.0);
    let view_dir = normalize(vec3<f32>(0.0, 0.0, 1.0));
    let world_pos = vec3<f32>(in.uv.x, in.uv.y, 0.0);
    
    var total_light = vec3<f32>(0.0);
    
    // Accumulate lighting from all lights (mimic deferred lighting pass)
    for (var i = 0u; i < params.num_lights; i++) {
        let light = lights[i];
        
        // Light direction and attenuation
        let light_dir = normalize(light.position - world_pos);
        let distance = length(light.position - world_pos);
        let attenuation = 1.0 / (distance * distance + 1.0);
        
        // BRDF calculation
        let brdf = cook_torrance_brdf(
            normal,
            view_dir,
            light_dir,
            params.roughness,
            params.metallic,
            base_color
        );
        
        total_light += brdf * light.color * light.intensity * attenuation;
    }
    
    // Shadow sampling for shadow-casting lights (mimic PCF sampling)
    for (var i = 0u; i < params.num_shadow_lights; i++) {
        // Simulate 4 shadow samples per light (like deferred lighting)
        var shadow_factor = 0.0;
        for (var s = 0u; s < 4u; s++) {
            // Dummy shadow comparison (we just want timing, not correctness)
            let offset = vec2<f32>(f32(s) * 0.01, f32(s) * 0.01);
            let sample_pos = world_pos.xy + offset;
            shadow_factor += step(0.5, fract(sample_pos.x + sample_pos.y));
        }
        shadow_factor /= 4.0;
        total_light *= shadow_factor;
    }
    
    // Ambient IBL approximation
    let ambient = vec3<f32>(0.03) * base_color;
    total_light += ambient;
    
    return vec4<f32>(total_light, 1.0);
}
