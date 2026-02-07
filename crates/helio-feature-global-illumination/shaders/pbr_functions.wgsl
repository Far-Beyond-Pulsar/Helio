// PBR (Physically Based Rendering) functions
// Based on the principled BRDF with Cook-Torrance microfacet model
// Supports both Beckmann and GGX distributions

const PI: f32 = 3.14159265359;

// ===== Fresnel =====
fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn fresnel_schlick_roughness(cos_theta: f32, f0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let one_minus_roughness = vec3<f32>(1.0 - roughness);
    return f0 + (max(one_minus_roughness, f0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// ===== Normal Distribution Functions =====

// Beckmann distribution
fn distribution_beckmann(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;

    let n_dot_h2 = n_dot_h * n_dot_h;
    let tan_theta = (1.0 - n_dot_h2) / max(n_dot_h2, 0.0001);

    let denom = PI * a2 * n_dot_h2 * n_dot_h2;
    return exp(-tan_theta / a2) / max(denom, 0.0001);
}

// GGX / Trowbridge-Reitz distribution
fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h2 = n_dot_h * n_dot_h;

    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    let denom2 = PI * denom * denom;

    return a2 / max(denom2, 0.0001);
}

// ===== Geometry Functions =====

// Cook-Torrance geometry function
fn geometry_cook_torrance(n_dot_l: f32, n_dot_v: f32, n_dot_h: f32, v_dot_h: f32) -> f32 {
    let g1 = (2.0 * n_dot_h * n_dot_v) / max(v_dot_h, 0.0001);
    let g2 = (2.0 * n_dot_h * n_dot_l) / max(v_dot_h, 0.0001);
    return min(1.0, min(g1, g2));
}

// GGX Smith geometry function
fn geometry_smith_ggx(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;

    let ggx_v = n_dot_v / max(sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v), 0.0001);
    let ggx_l = n_dot_l / max(sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l), 0.0001);

    return ggx_v * ggx_l;
}

// ===== Diffuse BRDF =====

// Lambertian diffuse
fn diffuse_lambert(albedo: vec3<f32>) -> vec3<f32> {
    return albedo / PI;
}

// Oren-Nayar diffuse (accounts for roughness)
fn diffuse_oren_nayar(
    albedo: vec3<f32>,
    roughness: f32,
    n_dot_v: f32,
    n_dot_l: f32,
    v: vec3<f32>,
    l: vec3<f32>,
    n: vec3<f32>
) -> vec3<f32> {
    let roughness2 = roughness * roughness;

    let a = 1.0 - 0.5 * roughness2 / (roughness2 + 0.57);
    let b = 0.45 * roughness2 / (roughness2 + 0.09);

    let theta_i = acos(n_dot_l);
    let theta_r = acos(n_dot_v);

    let alpha = max(theta_i, theta_r);
    let beta = min(theta_i, theta_r);

    // Project V and L onto the plane
    let v_proj = normalize(v - n * n_dot_v);
    let l_proj = normalize(l - n * n_dot_l);
    let cos_phi_diff = max(dot(v_proj, l_proj), 0.0);

    let c = b * max(0.0, cos_phi_diff) * sin(alpha) * tan(beta);

    return (albedo / PI) * (a + c);
}

// ===== Cook-Torrance BRDF =====

fn brdf_cook_torrance(
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>
) -> vec3<f32> {
    let h = normalize(v + l);

    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let v_dot_h = max(dot(v, h), 0.0);

    // Early exit if surface is facing away
    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        return vec3<f32>(0.0);
    }

    // Calculate F0 (surface reflection at zero incidence)
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Fresnel term
    let f = fresnel_schlick(v_dot_h, f0);

    // Normal distribution
    let d = distribution_ggx(n_dot_h, roughness);

    // Geometry term
    let g = geometry_smith_ggx(n_dot_v, n_dot_l, roughness);

    // Specular component
    let numerator = d * g * f;
    let denominator = 4.0 * n_dot_v * n_dot_l;
    let specular = numerator / max(denominator, 0.0001);

    // Diffuse component (energy conserving)
    let k_d = (1.0 - f) * (1.0 - metallic);
    let diffuse = k_d * diffuse_oren_nayar(
        albedo, roughness, n_dot_v, n_dot_l, v, l, n
    );

    return (diffuse + specular) * n_dot_l;
}

// ===== PBR Material Evaluation =====

struct PBRMaterial {
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
    emissive: vec3<f32>,
}

fn evaluate_pbr_material(
    material: PBRMaterial,
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    light_dir: vec3<f32>,
    light_color: vec3<f32>
) -> vec3<f32> {
    // Direct lighting using Cook-Torrance BRDF
    let direct_light = brdf_cook_torrance(
        material.albedo,
        material.metallic,
        material.roughness,
        normal,
        view_dir,
        light_dir
    ) * light_color;

    return direct_light;
}

// ===== Ambient/Environment Lighting =====

fn evaluate_ambient_pbr(
    material: PBRMaterial,
    normal: vec3<f32>,
    view_dir: vec3<f32>,
    ambient_color: vec3<f32>
) -> vec3<f32> {
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let f0 = mix(vec3<f32>(0.04), material.albedo, material.metallic);
    let f = fresnel_schlick_roughness(n_dot_v, f0, material.roughness);

    let k_d = (1.0 - f) * (1.0 - material.metallic);
    let diffuse = k_d * material.albedo * ambient_color;

    return diffuse;
}
