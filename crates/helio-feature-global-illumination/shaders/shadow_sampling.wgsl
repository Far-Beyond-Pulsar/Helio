// Shadow mapping with PCF (Percentage Closer Filtering)

// Shadow map texture and sampler (will be bound by the feature)
// TODO: These will be bound in the actual pipeline
// @group(2) @binding(0)
// var shadow_map: texture_depth_2d;
// @group(2) @binding(1)
// var shadow_sampler: sampler_comparison;

struct ShadowData {
    light_view_proj: mat4x4<f32>,
    light_direction: vec3<f32>,
    shadow_bias: f32,
}

// Transform world position to shadow map space
fn world_to_shadow_coords(world_pos: vec3<f32>, light_view_proj: mat4x4<f32>) -> vec3<f32> {
    let light_space = light_view_proj * vec4<f32>(world_pos, 1.0);
    var shadow_coords = light_space.xyz / light_space.w;

    // Transform to [0, 1] range for texture sampling
    shadow_coords.x = shadow_coords.x * 0.5 + 0.5;
    shadow_coords.y = shadow_coords.y * -0.5 + 0.5; // Flip Y
    // shadow_coords.z is already in [0, 1] for depth comparison

    return shadow_coords;
}

// PCF shadow sampling with adjustable kernel size
fn sample_shadow_pcf(shadow_coords: vec3<f32>, kernel_size: i32) -> f32 {
    // TODO: Actual texture sampling when shadow map is bound
    // For now, return a procedural shadow pattern

    // Check if in shadow map bounds
    if (shadow_coords.x < 0.0 || shadow_coords.x > 1.0 ||
        shadow_coords.y < 0.0 || shadow_coords.y > 1.0 ||
        shadow_coords.z < 0.0 || shadow_coords.z > 1.0) {
        return 1.0; // Outside shadow map, fully lit
    }

    // Procedural shadow (circular pattern) until texture binding implemented
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(shadow_coords.xy - center);
    let shadow_radius = 0.2;
    let shadow_softness = 0.15;

    // Soft circular shadow
    let shadow = smoothstep(shadow_radius - shadow_softness, shadow_radius + shadow_softness, dist);

    return shadow;

    /* Proper PCF implementation (when shadow map is bound):
    let texel_size = 1.0 / vec2<f32>(textureDimensions(shadow_map));
    var visibility = 0.0;
    var sample_count = 0.0;

    for (var x = -kernel_size; x <= kernel_size; x++) {
        for (var y = -kernel_size; y <= kernel_size; y++) {
            let offset = vec2<f32>(f32(x), f32(y)) * texel_size;
            let sample_coords = shadow_coords.xy + offset;

            visibility += textureSampleCompare(
                shadow_map,
                shadow_sampler,
                sample_coords,
                shadow_coords.z
            );
            sample_count += 1.0;
        }
    }

    return visibility / sample_count;
    */
}

// Calculate shadow with slope-based bias
fn calculate_shadow_with_bias(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    shadow_data: ShadowData
) -> f32 {
    // Transform to shadow space
    let shadow_coords = world_to_shadow_coords(world_pos, shadow_data.light_view_proj);

    // Calculate slope-scaled bias
    let n_dot_l = dot(normal, -shadow_data.light_direction);
    let bias = max(shadow_data.shadow_bias * (1.0 - n_dot_l), shadow_data.shadow_bias * 0.1);

    // Apply bias to depth coordinate
    var biased_coords = shadow_coords;
    biased_coords.z -= bias;

    // Sample shadow map with PCF
    return sample_shadow_pcf(biased_coords, 1);
}

// Soft shadow sampling with adjustable penumbra
fn calculate_soft_shadow(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    shadow_data: ShadowData,
    softness: f32
) -> f32 {
    let shadow_coords = world_to_shadow_coords(world_pos, shadow_data.light_view_proj);

    let n_dot_l = dot(normal, -shadow_data.light_direction);
    let bias = max(shadow_data.shadow_bias * (1.0 - n_dot_l), shadow_data.shadow_bias * 0.1);

    var biased_coords = shadow_coords;
    biased_coords.z -= bias;

    // Use larger PCF kernel for softer shadows
    let kernel_size = i32(softness * 3.0);
    return sample_shadow_pcf(biased_coords, kernel_size);
}

// Calculate shadow visibility (1.0 = fully lit, 0.0 = fully shadowed)
fn get_shadow_visibility(
    world_pos: vec3<f32>,
    normal: vec3<f32>,
    shadow_data: ShadowData
) -> f32 {
    return calculate_shadow_with_bias(world_pos, normal, shadow_data);
}
