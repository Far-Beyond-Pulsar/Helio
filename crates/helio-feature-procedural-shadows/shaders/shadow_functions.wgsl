// High-quality realtime multi-light shadow mapping with PCF

// Shadow map texture and comparison sampler (bound via ShaderData)
var shadow_map: texture_depth_2d;
var shadow_sampler: sampler_comparison;

// Light types
const LIGHT_TYPE_DIRECTIONAL: u32 = 0u;
const LIGHT_TYPE_POINT: u32 = 1u;
const LIGHT_TYPE_SPOT: u32 = 2u;
const LIGHT_TYPE_RECT: u32 = 3u;

// GPU representation of a light
struct GpuLight {
    light_type: u32,
    intensity: f32,
    radius: f32,
    _padding1: f32,
    
    position: vec3<f32>,
    inner_angle: f32,
    
    direction: vec3<f32>,
    outer_angle: f32,
    
    color: vec3<f32>,
    width: f32,
    
    light_view_proj: mat4x4<f32>,
    
    height: f32,
    _padding2: vec3<f32>,
}

// Shadow uniforms from CPU (bound via ShaderData)
struct ShadowUniforms {
    light_count: u32,
    _padding: vec3<u32>,
    lights: array<GpuLight, 8>,
}
var<uniform> shadow_uniforms: ShadowUniforms;

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

// Transform world position to shadow map space for a specific light
fn world_to_shadow_coord(world_pos: vec3<f32>, light: GpuLight) -> vec3<f32> {
    let light_space = light.light_view_proj * vec4<f32>(world_pos, 1.0);
    var shadow_coord = light_space.xyz / light_space.w;

    // Transform to [0, 1] range for texture coordinates
    shadow_coord.x = shadow_coord.x * 0.5 + 0.5;
    shadow_coord.y = shadow_coord.y * -0.5 + 0.5;  // Flip Y for texture coords

    return shadow_coord;
}

// Check if fragment is in shadow for a specific light
fn compute_light_shadow(light: GpuLight, light_index: u32, world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    // Only first light has shadow map for now (TODO: support multiple shadow maps via texture array)
    if (light_index > 0u) {
        return 1.0; // No shadows for lights beyond first
    }
    
    let normal = normalize(world_normal);
    
    // Calculate light direction based on type
    var light_dir: vec3<f32>;
    var distance_to_light: f32;
    
    if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {
        light_dir = -light.direction;
        distance_to_light = 0.0; // Infinite distance
    } else {
        let to_light = light.position - world_pos;
        distance_to_light = length(to_light);
        light_dir = normalize(to_light);
        
        // Check if beyond attenuation radius
        if (light.radius > 0.0 && distance_to_light > light.radius) {
            return 1.0; // Fully lit (no shadow contribution)
        }
    }
    
    let ndotl = max(dot(normal, light_dir), 0.0);
    
    // Spotlight cone attenuation
    if (light.light_type == LIGHT_TYPE_SPOT) {
        let spot_dir = -light.direction;
        let theta = dot(normalize(world_pos - light.position), spot_dir);
        let inner_cutoff = cos(light.inner_angle);
        let outer_cutoff = cos(light.outer_angle);
        
        // Outside spotlight cone
        if (theta < outer_cutoff) {
            return 1.0; // Fully lit (no shadow contribution)
        }
        
        // Smooth falloff at cone edges
        let epsilon = inner_cutoff - outer_cutoff;
        let spot_intensity = clamp((theta - outer_cutoff) / epsilon, 0.0, 1.0);
        if (spot_intensity < 0.01) {
            return 1.0;
        }
    }
    
    // Smooth fade for surfaces facing away from light
    let face_fade = smoothstep(0.0, 0.15, ndotl);
    if (face_fade < 0.001) {
        return 0.2; // Fully in shadow
    }
    
    // Normal offset bias - offset sample point along normal to prevent acne
    let normal_offset = 0.02;
    let offset_pos = world_pos + normal * normal_offset * (1.0 - ndotl);
    
    // Transform offset position to shadow space
    var shadow_coord = world_to_shadow_coord(offset_pos, light);
    
    // Check if position is within shadow map bounds
    if (shadow_coord.x < 0.0 || shadow_coord.x > 1.0 ||
        shadow_coord.y < 0.0 || shadow_coord.y > 1.0 ||
        shadow_coord.z < 0.0 || shadow_coord.z > 1.0) {
        return 1.0; // Outside shadow map, fully lit
    }
    
    // Additional slope-scaled depth bias
    let shadow_bias = 0.005;
    let slope_bias = max(shadow_bias * (1.0 - ndotl), shadow_bias);
    shadow_coord.z -= slope_bias;
    
    // Sample shadow with PCF for smooth edges
    let visibility = sample_shadow_visibility(shadow_coord);
    
    // Apply shadow with smooth falloff
    let shadow_factor = mix(0.2, 1.0, visibility);
    return mix(0.2, shadow_factor, face_fade);
}

// Apply shadows from all lights
fn apply_shadow(base_color: vec3<f32>, world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    // If no lights, return base color with minimal ambient
    if (shadow_uniforms.light_count == 0u) {
        return base_color * 0.1;
    }
    
    var accumulated_light = vec3<f32>(0.0);
    let normal = normalize(world_normal);
    
    // For each light, accumulate lighting with shadows
    for (var i = 0u; i < shadow_uniforms.light_count && i < 8u; i++) {
        let light = shadow_uniforms.lights[i];
        
        // Calculate light direction
        var light_dir: vec3<f32>;
        var distance_to_light: f32;
        
        if (light.light_type == LIGHT_TYPE_DIRECTIONAL) {
            light_dir = -light.direction;
            distance_to_light = 0.0;
        } else {
            let to_light = light.position - world_pos;
            distance_to_light = length(to_light);
            light_dir = normalize(to_light);
        }
        
        // Basic diffuse lighting
        let ndotl = max(dot(normal, light_dir), 0.0);
        
        // Distance attenuation (inverse square with smoothing)
        var attenuation = 1.0;
        if (light.light_type != LIGHT_TYPE_DIRECTIONAL && light.radius > 0.0) {
            if (distance_to_light > light.radius) {
                continue; // Skip lights that are too far
            }
            // Smooth quadratic falloff
            let normalized_dist = distance_to_light / light.radius;
            attenuation = 1.0 - (normalized_dist * normalized_dist);
            attenuation = attenuation * attenuation; // Smoother falloff
        }
        
        // Spotlight cone attenuation
        if (light.light_type == LIGHT_TYPE_SPOT) {
            let spot_dir = -light.direction;
            let theta = dot(-light_dir, spot_dir);
            let inner_cutoff = cos(light.inner_angle);
            let outer_cutoff = cos(light.outer_angle);
            
            if (theta < outer_cutoff) {
                continue; // Outside spotlight cone
            }
            
            // Smooth transition between inner and outer cone
            let epsilon = inner_cutoff - outer_cutoff;
            let spot_intensity = clamp((theta - outer_cutoff) / epsilon, 0.0, 1.0);
            attenuation *= spot_intensity;
        }
        
        // Calculate shadow factor for this light (only first light has shadows for now)
        var shadow_factor = 1.0;
        if (i == 0u) {
            shadow_factor = compute_light_shadow(light, i, world_pos, world_normal);
        }
        
        // Accumulate light contribution
        let light_contribution = light.color * light.intensity * ndotl * attenuation * shadow_factor;
        accumulated_light += light_contribution;
    }
    
    // Add small ambient
    let ambient = vec3<f32>(0.05);
    
    // Apply accumulated lighting to base color
    return base_color * (accumulated_light + ambient);
}
