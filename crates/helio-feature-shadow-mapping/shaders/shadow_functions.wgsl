// Shadow mapping functions

// Shadow uniforms (will be bound by the feature)
struct ShadowData {
    light_view_proj: mat4x4<f32>,
    light_direction: vec3<f32>,
    shadow_bias: f32,
}

// TODO: This needs to be bound as a uniform - for now using constants
fn get_shadow_data() -> ShadowData {
    var data: ShadowData;
    // Light from top-right matching the lighting direction
    let light_dir = normalize(vec3<f32>(0.5, -1.0, 0.3));
    let light_pos = -light_dir * 20.0;

    // Create light view matrix
    let view = look_at_rh(light_pos, vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0));

    // Orthographic projection for directional light
    let projection = orthographic_rh(-15.0, 15.0, -15.0, 15.0, 0.1, 50.0);

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

// Sample shadow map with PCF (Percentage Closer Filtering)
fn sample_shadow_map(shadow_coord: vec3<f32>) -> f32 {
    // TODO: Actual shadow map texture sampling will be implemented
    // For now, return a simple falloff based on distance
    let center_dist = length(shadow_coord.xy - vec2<f32>(0.5));
    return smoothstep(0.7, 0.3, center_dist);
}

// Transform world position to shadow map space
fn world_to_shadow_coord(world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> vec3<f32> {
    let light_space = light_view_proj * vec4<f32>(world_pos, 1.0);
    var shadow_coord = light_space.xyz / light_space.w;

    // Transform to [0, 1] range
    shadow_coord.x = shadow_coord.x * 0.5 + 0.5;
    shadow_coord.y = shadow_coord.y * -0.5 + 0.5;  // Flip Y for texture coords

    return shadow_coord;
}

// Apply shadow to color
fn apply_shadow(base_color: vec3<f32>, world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    let shadow_data = get_shadow_data();

    // Transform to shadow space
    let shadow_coord = world_to_shadow_coord(world_pos, shadow_data.light_view_proj);

    // Check if in shadow map bounds
    if (shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 || shadow_coord.y > 1.0 ||
        shadow_coord.z < 0.0 || shadow_coord.z > 1.0) {
        return base_color;  // Outside shadow map, fully lit
    }

    // Bias based on surface angle to light
    let ndotl = dot(world_normal, -shadow_data.light_direction);
    let bias = max(shadow_data.shadow_bias * (1.0 - ndotl), shadow_data.shadow_bias * 0.1);

    // Sample shadow map
    let shadow = sample_shadow_map(shadow_coord - vec3<f32>(0.0, 0.0, bias));

    // Apply shadow (0.3 = minimum brightness in shadow)
    let shadow_factor = mix(0.3, 1.0, shadow);
    return base_color * shadow_factor;
}
