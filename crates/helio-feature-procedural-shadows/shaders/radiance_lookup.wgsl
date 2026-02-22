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

// Sample radiance from the probe grid with trilinear interpolation
fn sample_radiance_cascade(world_pos: vec3<f32>, world_normal: vec3<f32>) -> vec3<f32> {
    // Get UV for this world position
    let probe_uv = world_pos_to_probe_uv(world_pos);

    // Map to probe grid coordinates
    let probe_coord = probe_uv * f32(PROBES_PER_AXIS);
    let probe_idx_base = vec2<i32>(probe_coord);
    let blend = fract(probe_coord);

    // Sample 4 neighboring probes for better interpolation
    let uv_00 = (vec2<f32>(probe_idx_base) + 0.5) / f32(PROBES_PER_AXIS);
    let uv_10 = (vec2<f32>(probe_idx_base + vec2<i32>(1, 0)) + 0.5) / f32(PROBES_PER_AXIS);
    let uv_01 = (vec2<f32>(probe_idx_base + vec2<i32>(0, 1)) + 0.5) / f32(PROBES_PER_AXIS);
    let uv_11 = (vec2<f32>(probe_idx_base + vec2<i32>(1, 1)) + 0.5) / f32(PROBES_PER_AXIS);

    // Convert to texture space
    let tex_scale = f32(PROBES_PER_AXIS) / 1024.0;
    let r00 = textureSampleLevel(shadow_maps, shadow_sampler, uv_00 * tex_scale, 0.0).rgb;
    let r10 = textureSampleLevel(shadow_maps, shadow_sampler, uv_10 * tex_scale, 0.0).rgb;
    let r01 = textureSampleLevel(shadow_maps, shadow_sampler, uv_01 * tex_scale, 0.0).rgb;
    let r11 = textureSampleLevel(shadow_maps, shadow_sampler, uv_11 * tex_scale, 0.0).rgb;

    // Bilinear interpolation
    let r0 = mix(r00, r10, blend.x);
    let r1 = mix(r01, r11, blend.x);
    let radiance = mix(r0, r1, blend.y);

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
    // Sample lighting from radiance cascades
    // All light comes from emissive objects in the scene - no hardcoded lighting
    let radiance = sample_radiance_cascade(world_pos, world_normal);

    // Apply to albedo
    let lit_color = base_color * radiance;

    // Tone mapping and gamma correction
    return linear_to_srgb(aces_tonemap(lit_color));
}
