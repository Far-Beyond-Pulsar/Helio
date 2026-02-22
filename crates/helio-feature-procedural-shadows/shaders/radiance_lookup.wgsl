// Radiance Cascades - Lookup functions for main rendering
// Samples the radiance cascade texture to get GI for fragments

// Radiance cascade data (bound automatically by ShaderData)
var shadow_maps: texture_2d<f32>;
var shadow_sampler: sampler;

// Light configuration
struct GpuLightConfig {
    sun_direction: vec3<f32>,
    _pad0: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    sky_color: vec3<f32>,
    ambient_intensity: f32,
}
var<uniform> lighting: GpuLightConfig;

// Surface geometry helper (matching trace shader)
struct SurfaceGeometry {
    position: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    normal: vec3<f32>,
    size: vec2<f32>,
    resolution: vec2<u32>,
    uv_offset: vec2<u32>,
}

// Get surface geometry (same as in trace shader)
fn get_surface_for_lookup(surface_id: u32) -> SurfaceGeometry {
    var surf: SurfaceGeometry;

    // Floor
    if (surface_id == 0u) {
        surf.position = vec3<f32>(0.0, 0.0, 0.0);
        surf.tangent = vec3<f32>(1.0, 0.0, 0.0);
        surf.bitangent = vec3<f32>(0.0, 0.0, 1.0);
        surf.normal = vec3<f32>(0.0, 1.0, 0.0);
        surf.size = vec2<f32>(1.0, 1.0);
        surf.resolution = vec2<u32>(256u, 256u);
        surf.uv_offset = vec2<u32>(0u, 0u);
    }
    // Ceiling
    else if (surface_id == 1u) {
        surf.position = vec3<f32>(0.0, 0.5, 0.0);
        surf.tangent = vec3<f32>(1.0, 0.0, 0.0);
        surf.bitangent = vec3<f32>(0.0, 0.0, 1.0);
        surf.normal = vec3<f32>(0.0, -1.0, 0.0);
        surf.size = vec2<f32>(1.0, 1.0);
        surf.resolution = vec2<u32>(256u, 256u);
        surf.uv_offset = vec2<u32>(256u, 0u);
    }
    // Wall X+ (Red)
    else if (surface_id == 2u) {
        surf.position = vec3<f32>(0.0, 0.0, 0.0);
        surf.tangent = vec3<f32>(0.0, 1.0, 0.0);
        surf.bitangent = vec3<f32>(0.0, 0.0, 1.0);
        surf.normal = vec3<f32>(1.0, 0.0, 0.0);
        surf.size = vec2<f32>(0.5, 1.0);
        surf.resolution = vec2<u32>(128u, 256u);
        surf.uv_offset = vec2<u32>(512u, 0u);
    }
    // Wall X- (Green)
    else if (surface_id == 3u) {
        surf.position = vec3<f32>(1.0, 0.0, 0.0);
        surf.tangent = vec3<f32>(0.0, 1.0, 0.0);
        surf.bitangent = vec3<f32>(0.0, 0.0, 1.0);
        surf.normal = vec3<f32>(-1.0, 0.0, 0.0);
        surf.size = vec2<f32>(0.5, 1.0);
        surf.resolution = vec2<u32>(128u, 256u);
        surf.uv_offset = vec2<u32>(640u, 0u);
    }
    // Wall Z+
    else if (surface_id == 4u) {
        surf.position = vec3<f32>(0.0, 0.0, 0.0);
        surf.tangent = vec3<f32>(0.0, 1.0, 0.0);
        surf.bitangent = vec3<f32>(1.0, 0.0, 0.0);
        surf.normal = vec3<f32>(0.0, 0.0, 1.0);
        surf.size = vec2<f32>(0.5, 1.0);
        surf.resolution = vec2<u32>(128u, 256u);
        surf.uv_offset = vec2<u32>(768u, 0u);
    }
    // Wall Z-
    else if (surface_id == 5u) {
        surf.position = vec3<f32>(0.0, 0.0, 1.0);
        surf.tangent = vec3<f32>(0.0, 1.0, 0.0);
        surf.bitangent = vec3<f32>(1.0, 0.0, 0.0);
        surf.normal = vec3<f32>(0.0, 0.0, -1.0);
        surf.size = vec2<f32>(0.5, 1.0);
        surf.resolution = vec2<u32>(128u, 256u);
        surf.uv_offset = vec2<u32>(896u, 0u);
    }
    // Interior wall front
    else if (surface_id == 6u) {
        surf.position = vec3<f32>(0.0, 0.0, 0.47 - 0.00390625);
        surf.tangent = vec3<f32>(0.0, 1.0, 0.0);
        surf.bitangent = vec3<f32>(1.0, 0.0, 0.0);
        surf.normal = vec3<f32>(0.0, 0.0, -1.0);
        surf.size = vec2<f32>(0.5, 1.0);
        surf.resolution = vec2<u32>(128u, 256u);
        surf.uv_offset = vec2<u32>(0u, 1536u);
    }
    // Interior wall back
    else {
        surf.position = vec3<f32>(0.0, 0.0, 0.53 - 0.00390625);
        surf.tangent = vec3<f32>(0.0, 1.0, 0.0);
        surf.bitangent = vec3<f32>(1.0, 0.0, 0.0);
        surf.normal = vec3<f32>(0.0, 0.0, 1.0);
        surf.size = vec2<f32>(0.5, 1.0);
        surf.resolution = vec2<u32>(128u, 256u);
        surf.uv_offset = vec2<u32>(128u, 1536u);
    }

    return surf;
}

// Find which surface a world position belongs to
fn find_surface_for_position(world_pos: vec3<f32>, world_normal: vec3<f32>) -> u32 {
    let eps = 0.01;

    // Test floor
    if (abs(world_pos.y) < eps && dot(world_normal, vec3<f32>(0.0, 1.0, 0.0)) > 0.5) {
        return 0u;
    }

    // Test ceiling
    if (abs(world_pos.y - 0.5) < eps && dot(world_normal, vec3<f32>(0.0, -1.0, 0.0)) > 0.5) {
        return 1u;
    }

    // Test wall X+
    if (abs(world_pos.x) < eps && dot(world_normal, vec3<f32>(1.0, 0.0, 0.0)) > 0.5) {
        return 2u;
    }

    // Test wall X-
    if (abs(world_pos.x - 1.0) < eps && dot(world_normal, vec3<f32>(-1.0, 0.0, 0.0)) > 0.5) {
        return 3u;
    }

    // Test wall Z+
    if (abs(world_pos.z) < eps && dot(world_normal, vec3<f32>(0.0, 0.0, 1.0)) > 0.5) {
        return 4u;
    }

    // Test wall Z-
    if (abs(world_pos.z - 1.0) < eps && dot(world_normal, vec3<f32>(0.0, 0.0, -1.0)) > 0.5) {
        return 5u;
    }

    // Test interior wall front
    let interior_z1 = 0.47 - 0.00390625;
    if (abs(world_pos.z - interior_z1) < eps && dot(world_normal, vec3<f32>(0.0, 0.0, -1.0)) > 0.5) {
        return 6u;
    }

    // Test interior wall back
    let interior_z2 = 0.53 - 0.00390625;
    if (abs(world_pos.z - interior_z2) < eps && dot(world_normal, vec3<f32>(0.0, 0.0, 1.0)) > 0.5) {
        return 7u;
    }

    // Default to floor if no match
    return 0u;
}

// Sample radiance from cascade texture
fn sample_radiance_cascade(world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    // Find which surface this position is on
    let surface_id = find_surface_for_position(world_pos, world_normal);
    let surf = get_surface_for_lookup(surface_id);

    // Compute UV in surface space
    let local_pos = world_pos - surf.position;
    let u = dot(local_pos, surf.tangent) / surf.size.x;
    let v = dot(local_pos, surf.bitangent) / surf.size.y;

    // Clamp to valid range
    let uv_clamped = clamp(vec2<f32>(u, v), vec2<f32>(0.0), vec2<f32>(1.0));

    // Scale to finest cascade resolution (probe_size = 2, so resolution / 2)
    let cascade_res = vec2<f32>(surf.resolution) * 0.5;
    let texel_uv = uv_clamped * cascade_res;

    // Add offset for this surface
    let final_uv = (texel_uv + vec2<f32>(surf.uv_offset)) / vec2<f32>(1024.0, 2048.0);

    // Sample with bilinear filtering
    let radiance = textureSampleLevel(shadow_maps, shadow_sampler, final_uv, 0.0);

    // Accumulate all 4 samples from the 2x2 probe pattern (as in the example shader)
    let offset_x = vec2<f32>(cascade_res.x * 0.5, 0.0) / vec2<f32>(1024.0, 2048.0);
    let offset_y = vec2<f32>(0.0, cascade_res.y * 0.5) / vec2<f32>(1024.0, 2048.0);
    let offset_xy = offset_x + offset_y;

    let r1 = textureSampleLevel(shadow_maps, shadow_sampler, final_uv + offset_x, 0.0);
    let r2 = textureSampleLevel(shadow_maps, shadow_sampler, final_uv + offset_y, 0.0);
    let r3 = textureSampleLevel(shadow_maps, shadow_sampler, final_uv + offset_xy, 0.0);

    return radiance.rgb + r1.rgb + r2.rgb + r3.rgb;
}

// ACES filmic tone mapping
fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Linear to sRGB conversion
fn linear_to_srgb(linear: vec3<f32>) -> vec3<f32> {
    return pow(max(linear, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.2));
}

// Apply radiance cascades GI to fragment
fn apply_radiance_cascade(base_color: vec3<f32>, world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    // DEBUG: Return simple lighting to verify shader is running
    let sun_dir = normalize(lighting.sun_direction);
    let ndotl = max(dot(normalize(world_normal), sun_dir), 0.0);
    let lit_color = base_color * (ndotl * lighting.sun_color * lighting.sun_intensity +
                                   lighting.sky_color * lighting.ambient_intensity);

    // Apply tone mapping and gamma correction
    return linear_to_srgb(aces_tonemap(lit_color));

    // TODO: Enable actual cascade sampling once verified working
    /*
    // Check for emissive materials
    let material = get_material_for_fragment(world_pos, camera.position);
    if (material.emissive_strength > 0.0) {
        // Return bright emissive color
        return material.base_color.rgb * material.emissive_strength;
    }

    // Sample radiance from cascades
    let radiance = sample_radiance_cascade(world_pos, world_normal);

    // Multiply albedo by radiance
    let final_color = base_color * radiance;

    // Apply tone mapping and gamma correction
    return linear_to_srgb(aces_tonemap(final_color));
    */
}
