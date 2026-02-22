// Radiance Cascades - Lookup shader for sampling GI in main render pass
// Samples from volumetric probe grid to get indirect lighting

var shadow_maps: texture_2d<f32>;
var shadow_sampler: sampler;

struct GpuLightConfig {
    sun_direction: vec3<f32>,
    _pad0: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    sky_color: vec3<f32>,
    ambient_intensity: f32,
}
var<uniform> lighting: GpuLightConfig;

// Must match trace shader configuration
const PROBES_PER_AXIS: u32 = 128u;
const WORLD_MIN: vec3<f32> = vec3<f32>(-50.0, -10.0, -50.0);
const WORLD_MAX: vec3<f32> = vec3<f32>(50.0, 30.0, 50.0);

// Convert world position to probe UV coordinates
fn world_pos_to_probe_uv(world_pos: vec3<f32>) -> vec2<f32> {
    let world_size = WORLD_MAX - WORLD_MIN;
    let local_pos = (world_pos - WORLD_MIN) / world_size;

    // Clamp to valid range
    let clamped = clamp(local_pos, vec3<f32>(0.0), vec3<f32>(1.0));

    // Map XZ to UV (ignore Y for now as probes are stratified by cascade level)
    return vec2<f32>(clamped.x, clamped.z);
}

// Sample radiance from the probe grid with bilinear interpolation
fn sample_radiance_cascade(world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    // Get UV for this world position
    let probe_uv = world_pos_to_probe_uv(world_pos);

    // Scale UV to probe grid region in texture (first 128x128 of 1024x2048)
    let texture_uv = probe_uv * (f32(PROBES_PER_AXIS) / 1024.0);

    // Sample with bilinear filtering for smooth interpolation between probes
    let radiance = textureSampleLevel(shadow_maps, shadow_sampler, texture_uv, 0.0).rgb;

    return radiance;
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
    let normal = normalize(world_normal);

    // Direct sun lighting
    let sun_dir = normalize(-lighting.sun_direction);
    let ndotl = max(dot(normal, sun_dir), 0.0);
    let direct_light = lighting.sun_color * lighting.sun_intensity * ndotl;

    // Sample indirect lighting from radiance cascades
    let indirect_light = sample_radiance_cascade(world_pos, world_normal);

    // Combine direct and indirect lighting
    let total_light = direct_light + indirect_light;

    // Apply to albedo
    let lit_color = base_color * total_light;

    // Tone mapping and gamma correction
    return linear_to_srgb(aces_tonemap(lit_color));
}
