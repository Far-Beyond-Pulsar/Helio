// High-quality realtime shadow mapping with PCF

// Shadow map texture and comparison sampler (bound automatically by ShaderData)
var shadow_map: texture_depth_2d;
var shadow_sampler: sampler_comparison;

// Shadow uniforms
struct ShadowData {
    light_view_proj: mat4x4<f32>,
    light_direction: vec3<f32>,
    shadow_bias: f32,
}

// Hardcoded shadow data matching the renderer's light setup
fn get_shadow_data() -> ShadowData {
    var data: ShadowData;
    // Light from top-right matching the lighting direction
    let light_dir = normalize(vec3<f32>(0.5, -1.0, 0.3));
    let light_pos = -light_dir * 20.0;

    // Create light view matrix
    let view = look_at_rh(light_pos, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0));

    // Orthographic projection for directional light
    let projection = orthographic_rh(-8.0, 8.0, -8.0, 8.0, 0.1, 40.0);

    data.light_view_proj = projection * view;
    data.light_direction = light_dir;
    data.shadow_bias = 0.005;

    return data;
}

// Helper: Look-at matrix
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

// Helper: Orthographic projection
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

// High-quality PCF shadow sampling with 5x5 kernel
// This provides smooth, realistic shadow edges
fn sample_shadow_pcf(shadow_coord: vec3<f32>, texel_size: vec2<f32>) -> f32 {
    var shadow = 0.0;
    let samples = 5;
    let half_samples = f32(samples) / 2.0;

    // 5x5 PCF kernel for smooth shadows
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_coord = shadow_coord.xy + offset;

            // Use hardware comparison sampling for efficiency
            shadow += textureSampleCompareLevel(
                shadow_map,
                shadow_sampler,
                sample_coord,
                shadow_coord.z
            );
        }
    }

    // Average the samples (25 samples for 5x5 kernel)
    return shadow / 25.0;
}

// Optimized PCF shadow sampling with 3x3 kernel
// Good balance between quality and performance
fn sample_shadow_pcf_3x3(shadow_coord: vec3<f32>, texel_size: vec2<f32>) -> f32 {
    var shadow = 0.0;

    // 3x3 PCF kernel
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_coord = shadow_coord.xy + offset;

            shadow += textureSampleCompareLevel(
                shadow_map,
                shadow_sampler,
                sample_coord,
                shadow_coord.z
            );
        }
    }

    return shadow / 9.0;
}

// Sample shadow visibility using PCF (1.0 = fully lit, 0.0 = fully shadowed)
fn sample_shadow_visibility(shadow_coord: vec3<f32>) -> f32 {
    // Get shadow map dimensions for texel size calculation
    let shadow_map_size = vec2<f32>(textureDimensions(shadow_map));
    let texel_size = 1.0 / shadow_map_size;

    // Use 3x3 PCF for good quality and performance
    // For even higher quality, use sample_shadow_pcf() instead
    return sample_shadow_pcf_3x3(shadow_coord, texel_size);
}

// Transform world position to shadow map space
fn world_to_shadow_coord(world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> vec3<f32> {
    let light_space = light_view_proj * vec4<f32>(world_pos, 1.0);
    var shadow_coord = light_space.xyz / light_space.w;

    // Transform to [0, 1] range for texture coordinates
    shadow_coord.x = shadow_coord.x * 0.5 + 0.5;
    shadow_coord.y = shadow_coord.y * -0.5 + 0.5;  // Flip Y for texture coords

    return shadow_coord;
}

// Apply shadow to color with production-quality bias to prevent shadow acne
fn apply_shadow(base_color: vec3<f32>, world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let shadow_data = get_shadow_data();

    let normal = normalize(world_normal);
    let light_dir = -shadow_data.light_direction;
    let ndotl = max(dot(normal, light_dir), 0.0);

    // Early out for surfaces facing away from light (can't receive shadows)
    if (ndotl < 0.01) {
        return base_color * 0.2;  // Fully in shadow
    }

    // Normal offset bias - offset sample point along normal to prevent acne
    // This is more effective than depth bias alone
    let normal_offset = 0.02;  // Adjust based on scene scale
    let offset_pos = world_pos + normal * normal_offset * (1.0 - ndotl);

    // Transform offset position to shadow space
    var shadow_coord = world_to_shadow_coord(offset_pos, shadow_data.light_view_proj);

    // Check if position is within shadow map bounds
    if (shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 || shadow_coord.y > 1.0 ||
        shadow_coord.z < 0.0 || shadow_coord.z > 1.0) {
        return base_color;  // Outside shadow map, fully lit
    }

    // Additional slope-scaled depth bias for extra protection
    // Use aggressive bias for steep angles
    let slope_bias = max(shadow_data.shadow_bias * (1.0 - ndotl), shadow_data.shadow_bias);
    shadow_coord.z -= slope_bias;

    // Sample shadow with PCF for smooth edges
    let visibility = sample_shadow_visibility(shadow_coord);

    // Apply shadow with smooth falloff (0.2 = 20% ambient in shadow)
    let shadow_factor = mix(0.2, 1.0, visibility);
    return base_color * shadow_factor;
}
