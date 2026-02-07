// Simplified Global Illumination Functions
// Works directly with base shader without requiring external uniforms

const PI: f32 = 3.14159265359;

// === Checkerboard pattern for materials ===
fn checkerboard_pattern(uv: vec2<f32>, scale: f32) -> f32 {
    let scaled_uv = uv * scale;
    let checker = floor(scaled_uv.x) + floor(scaled_uv.y);
    return fract(checker * 0.5) * 2.0;
}

// === Complete GI lighting function ===
fn apply_global_illumination(
    world_pos: vec3<f32>,
    world_normal: vec3<f32>,
    tex_coords: vec2<f32>,
    base_color: vec3<f32>
) -> vec3<f32> {
    let normal = normalize(world_normal);

    // Light setup
    let light_direction = normalize(vec3<f32>(0.5, -1.0, 0.3));
    let light_dir = -light_direction;
    let light_color = vec3<f32>(2.5);

    // View direction (camera orbits at (5,4,5) approximately)
    let view_dir = normalize(vec3<f32>(5.0, 4.0, 5.0) - world_pos);

    // Material with checkerboard pattern
    let checker = checkerboard_pattern(tex_coords, 4.0);
    let albedo = mix(
        vec3<f32>(0.8, 0.3, 0.3),  // Red
        vec3<f32>(0.3, 0.3, 0.8),  // Blue
        checker
    );
    let metallic = 0.0;
    let roughness = 0.5;

    // === PBR Direct Lighting ===
    let h = normalize(view_dir + light_dir);
    let n_dot_l = max(dot(normal, light_dir), 0.0);
    let n_dot_v = max(dot(normal, view_dir), 0.0);
    let n_dot_h = max(dot(normal, h), 0.0);
    let v_dot_h = max(dot(view_dir, h), 0.0);

    if (n_dot_l <= 0.0 || n_dot_v <= 0.0) {
        // Back-facing, return just ambient
        let sky = vec3<f32>(0.5, 0.7, 1.0) * 0.2;
        return pow(sky * albedo, vec3<f32>(1.0 / 2.2));
    }

    // Fresnel (Schlick)
    let f0 = mix(vec3<f32>(0.04), albedo, metallic);
    let fresnel = f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);

    // Distribution (GGX)
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    let distribution = a2 / (PI * denom * denom);

    // Geometry (Smith-GGX approximation)
    let k = (roughness + 1.0) * (roughness + 1.0) / 8.0;
    let g_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
    let g_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
    let geometry = g_v * g_l;

    // Specular BRDF
    let specular = (distribution * geometry * fresnel) / max(4.0 * n_dot_v * n_dot_l, 0.001);

    // Diffuse BRDF
    let k_d = (1.0 - fresnel) * (1.0 - metallic);
    let diffuse = k_d * albedo / PI;

    let direct_light = (diffuse + specular) * n_dot_l * light_color;

    // === Soft Shadows ===
    // Simple soft shadow based on world position
    let shadow_center = vec2<f32>(0.0, 0.0);
    let shadow_dist = length(world_pos.xz - shadow_center);
    let shadow_radius = 3.0;
    let shadow_soft = 1.5;
    let shadow = smoothstep(shadow_radius - shadow_soft, shadow_radius + shadow_soft, shadow_dist);
    let shadow_factor = mix(0.3, 1.0, shadow);

    // === Indirect Lighting (GI) ===
    let up_factor = dot(normal, vec3<f32>(0.0, 1.0, 0.0)) * 0.5 + 0.5;
    let sky_color = vec3<f32>(0.5, 0.7, 1.0);
    let ground_color = vec3<f32>(0.3, 0.25, 0.2);
    let ambient = mix(ground_color, sky_color, up_factor) * albedo * 0.4;

    // === Combine ===
    var final_color = direct_light * shadow_factor + ambient;

    // Tone mapping (Reinhard)
    final_color = final_color / (final_color + vec3<f32>(1.0));

    // Gamma correction
    final_color = pow(final_color, vec3<f32>(1.0 / 2.2));

    return final_color;
}
