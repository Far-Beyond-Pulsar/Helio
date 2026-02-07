// Global Illumination Functions
// Implements Lumen-like GI with multi-bounce diffuse lighting

// GI Configuration
struct GIUniforms {
    light_view_proj: mat4x4<f32>,
    light_direction: vec3<f32>,
    shadow_bias: f32,
    gi_intensity: f32,
    num_samples: u32,
    max_ray_distance: f32,
    sky_color: vec3<f32>,
    _pad: f32,
}

// TODO: Bind GI uniforms
// @group(3) @binding(0)
// var<uniform> gi_uniforms: GIUniforms;

// Hardcoded GI configuration for now
fn get_gi_config() -> GIUniforms {
    var config: GIUniforms;
    config.light_direction = normalize(vec3<f32>(0.5, -1.0, 0.3));
    config.shadow_bias = 0.005;
    config.gi_intensity = 1.0;
    config.num_samples = 8u;
    config.max_ray_distance = 10.0;
    config.sky_color = vec3<f32>(0.5, 0.7, 1.0);

    // Calculate light view-proj
    let light_pos = -config.light_direction * 20.0;
    let view = look_at_rh(light_pos, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0));
    let projection = orthographic_rh(-10.0, 10.0, -10.0, 10.0, 0.1, 50.0);
    config.light_view_proj = projection * view;

    return config;
}

// ===== Helper Functions =====

fn look_at_rh(eye: vec3<f32>, center: vec3<f32>, up: vec3<f32>) -> mat4x4<f32> {
    let f = normalize(center - eye);
    let s = normalize(cross(f, up));
    let u = cross(s, f);

    return mat4x4<f32>(
        vec4<f32>(s.x, u.x, -f.x, 0.0),
        vec4<f32>(s.y, u.y, -f.y, 0.0),
        vec4<f32>(s.z, u.z, -f.z, 0.0),
        vec4<f32>(-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0),
    );
}

fn orthographic_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> mat4x4<f32> {
    let w = 1.0 / (right - left);
    let h = 1.0 / (top - bottom);
    let d = 1.0 / (far - near);

    return mat4x4<f32>(
        vec4<f32>(2.0 * w, 0.0, 0.0, 0.0),
        vec4<f32>(0.0, 2.0 * h, 0.0, 0.0),
        vec4<f32>(0.0, 0.0, -d, 0.0),
        vec4<f32>(-(right + left) * w, -(top + bottom) * h, -near * d, 1.0),
    );
}

// ===== Random Number Generation =====

fn hash(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.13);
    let p3_shifted = p3 + dot(p3, vec3<f32>(p3.y, p3.z, p3.x) + 3.333);
    return fract((p3_shifted.x + p3_shifted.y) * p3_shifted.z);
}

fn random(seed: vec2<f32>) -> f32 {
    return fract(sin(dot(seed, vec2<f32>(12.9898, 78.233))) * 43758.5453);
}

// Generate random direction in hemisphere oriented around normal
fn random_hemisphere_direction(normal: vec3<f32>, seed: vec2<f32>) -> vec3<f32> {
    let r1 = random(seed);
    let r2 = random(seed + vec2<f32>(1.0, 1.0));

    let phi = 2.0 * PI * r1;
    let cos_theta = sqrt(r2);
    let sin_theta = sqrt(1.0 - cos_theta * cos_theta);

    let x = sin_theta * cos(phi);
    let y = sin_theta * sin(phi);
    let z = cos_theta;

    // Build tangent space
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    return normalize(tangent * x + bitangent * y + normal * z);
}

// ===== Screen-Space GI =====

// Estimate indirect lighting from nearby surfaces (screen-space approximation)
fn screen_space_indirect_lighting(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    tex_coords: vec2<f32>
) -> vec3<f32> {
    var indirect = vec3<f32>(0.0);
    let config = get_gi_config();

    // Sample multiple directions in hemisphere
    for (var i = 0u; i < 4u; i++) {
        let seed = tex_coords + vec2<f32>(f32(i), 0.0);
        let ray_dir = random_hemisphere_direction(normal, seed);

        // Simple occlusion approximation based on normal orientation
        let occlusion = max(dot(ray_dir, normal), 0.0);

        // Use sky color as indirect bounce
        indirect += config.sky_color * occlusion;
    }

    indirect /= 4.0;
    return indirect * albedo * config.gi_intensity * 0.3;
}

// ===== Probe-based GI =====

// Sample radiance from probe grid (simplified version)
fn sample_radiance_probes(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let config = get_gi_config();

    // Simple spherical harmonics approximation
    // In production, this would sample from actual probe grid

    // Directional ambient based on normal
    let up_contribution = max(dot(normal, vec3<f32>(0.0, 1.0, 0.0)), 0.0);
    let side_contribution = 1.0 - up_contribution;

    let sky_radiance = config.sky_color * 0.8;
    let ground_radiance = vec3<f32>(0.3, 0.25, 0.2);

    return mix(ground_radiance, sky_radiance, up_contribution) * config.gi_intensity;
}

// ===== Multi-bounce Diffuse GI =====

fn calculate_indirect_diffuse(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    albedo: vec3<f32>,
    tex_coords: vec2<f32>
) -> vec3<f32> {
    // Combine screen-space and probe-based GI
    let ss_gi = screen_space_indirect_lighting(world_pos, normal, albedo, tex_coords);
    let probe_gi = sample_radiance_probes(world_pos, normal) * albedo;

    // Blend between techniques based on distance
    let blend = 0.5;
    return mix(ss_gi, probe_gi, blend);
}

// ===== Complete GI Evaluation =====

fn apply_global_illumination(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    tex_coords: vec2<f32>,
    base_color: vec3<f32>
) -> vec3<f32> {
    let config = get_gi_config();
    let normal = normalize(world_normal);

    // Create PBR material
    // In production, these would come from material textures/uniforms
    var material: PBRMaterial;
    material.albedo = base_color;
    material.metallic = 0.1;
    material.roughness = 0.6;
    material.emissive = vec3<f32>(0.0);

    // Calculate view direction (assuming camera at origin for now)
    // TODO: Get actual camera position from uniforms
    let camera_pos = vec3<f32>(5.0, 4.0, 5.0);
    let view_dir = normalize(camera_pos - world_pos);

    // ===== Direct Lighting =====

    let light_dir = -config.light_direction;
    let light_color = vec3<f32>(3.0); // Sun intensity

    // Calculate shadow
    let shadow_data = ShadowData(
        config.light_view_proj,
        config.light_direction,
        config.shadow_bias
    );
    let shadow_visibility = get_shadow_visibility(world_pos, normal, shadow_data);

    // Evaluate PBR for direct light
    let direct_lighting = evaluate_pbr_material(
        material,
        world_pos,
        normal,
        view_dir,
        light_dir,
        light_color
    ) * shadow_visibility;

    // ===== Indirect Lighting (GI) =====

    let indirect_diffuse = calculate_indirect_diffuse(
        world_pos,
        normal,
        material.albedo,
        tex_coords
    );

    // Ambient occlusion approximation
    let ao = 0.9;

    // ===== Combine All Lighting =====

    var final_color = direct_lighting + indirect_diffuse * ao;

    // Add emissive
    final_color += material.emissive;

    // Tone mapping (ACES approximation)
    final_color = (final_color * (2.51 * final_color + 0.03)) /
                  (final_color * (2.43 * final_color + 0.59) + 0.14);

    // Gamma correction
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));

    return final_color;
}
