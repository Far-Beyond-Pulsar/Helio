//! Water caustics generation compute shader.
//!
//! Generates caustic patterns by simulating light refraction through
//! animated water surfaces using Gerstner waves.

/// Caustics generation parameters
struct CausticsParams {
    time: f32,
    world_scale: f32,
    world_offset: vec2<f32>,
}

/// GPU water volume descriptor (matches libhelio::GpuWaterVolume)
struct GpuWaterVolume {
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
    wave_params: vec4<f32>,        // amplitude, frequency, speed, steepness
    wave_direction: vec4<f32>,     // xy direction + padding
    water_color: vec4<f32>,        // rgb + foam_threshold
    extinction: vec4<f32>,         // rgb absorption + foam_amount
    reflection_refraction: vec4<f32>, // strengths + fresnel_power
    caustics_params: vec4<f32>,    // enabled, intensity, scale, speed
    fog_params: vec4<f32>,         // density, god_rays + padding
    _pad0: vec4<f32>,
    _pad1: vec4<f32>,
    _pad2: vec4<f32>,
    _pad3: vec4<f32>,
    _pad4: vec4<f32>,
    _pad5: vec4<f32>,
    _pad6: vec4<f32>,
}

// Bindings
@group(0) @binding(0) var<uniform> params: CausticsParams;
@group(0) @binding(1) var<storage, read> water_volumes: array<GpuWaterVolume>;
@group(0) @binding(2) var output_texture: texture_storage_2d<rgba16float, write>;

/// Gerstner wave height calculation.
///
/// Returns wave height at given position and time.
fn gerstner_wave_height(pos: vec2<f32>, time: f32, vol: GpuWaterVolume) -> f32 {
    let amplitude = vol.wave_params.x;
    let frequency = vol.wave_params.y;
    let speed = vol.wave_params.z;
    let steepness = vol.wave_params.w;
    let direction = normalize(vol.wave_direction.xy);

    let k = frequency;
    let a = amplitude;
    let d = direction;
    let q = steepness / (k * a * 4.0); // Steepness factor

    let phase = k * dot(d, pos) - speed * time;
    return a * sin(phase);
}

/// Multi-octave Gerstner waves for realistic water surface.
///
/// Combines 4 octaves with decreasing amplitude and varying frequency.
fn gerstner_waves_multi(pos: vec2<f32>, time: f32, vol: GpuWaterVolume) -> f32 {
    let amplitude = vol.wave_params.x;
    let frequency = vol.wave_params.y;
    let speed = vol.wave_params.z;
    let direction = normalize(vol.wave_direction.xy);

    // First octave (primary wave)
    let h1 = gerstner_wave_height(pos, time, vol);

    // Second octave (smaller, faster)
    var vol2 = vol;
    vol2.wave_params.x = amplitude * 0.6; // Smaller amplitude
    vol2.wave_params.y = frequency * 1.7; // Higher frequency
    vol2.wave_params.z = speed * 1.3;     // Faster
    let h2 = gerstner_wave_height(pos * 1.2, time, vol2);

    // Third octave (detail waves)
    var vol3 = vol;
    vol3.wave_params.x = amplitude * 0.35;
    vol3.wave_params.y = frequency * 2.3;
    vol3.wave_params.z = speed * 1.5;
    let h3 = gerstner_wave_height(pos * 0.8, time, vol3);

    // Fourth octave (fine detail)
    var vol4 = vol;
    vol4.wave_params.x = amplitude * 0.2;
    vol4.wave_params.y = frequency * 3.1;
    vol4.wave_params.z = speed * 1.8;
    let h4 = gerstner_wave_height(pos * 1.5, time, vol4);

    return h1 + h2 + h3 + h4;
}

/// Calculate surface normal from Gerstner wave derivatives.
///
/// Uses finite differences to compute normal from height field.
fn calculate_normal(pos: vec2<f32>, time: f32, vol: GpuWaterVolume) -> vec3<f32> {
    let eps = 0.1; // Finite difference epsilon

    let h_center = gerstner_waves_multi(pos, time, vol);
    let h_x = gerstner_waves_multi(pos + vec2<f32>(eps, 0.0), time, vol);
    let h_z = gerstner_waves_multi(pos + vec2<f32>(0.0, eps), time, vol);

    // Tangent vectors
    let dx = vec3<f32>(eps, h_x - h_center, 0.0);
    let dz = vec3<f32>(0.0, h_z - h_center, eps);

    // Normal = cross product
    return normalize(cross(dx, dz));
}

/// Generate caustic intensity at given world XZ position.
///
/// Algorithm:
/// 1. Sample water surface height at this position
/// 2. Calculate surface normal
/// 3. Refract light ray (sun direction) through surface
/// 4. Trace to bottom plane
/// 5. Measure light convergence using finite differences
fn generate_caustic(world_xz: vec2<f32>, time: f32, vol: GpuWaterVolume) -> f32 {
    // Get water surface position
    let surface_y = vol.bounds_max.w + gerstner_waves_multi(world_xz, time, vol);
    let surface_pos = vec3<f32>(world_xz.x, surface_y, world_xz.y);

    // Calculate surface normal
    let normal = calculate_normal(world_xz, time, vol);

    // Sun direction (pointing from sun toward surface)
    let sun_dir = normalize(vec3<f32>(0.3, -1.0, 0.2));

    // Refract sun ray through water surface (IOR = 1.33 for water)
    let eta = 1.0 / 1.33; // Air to water
    let refracted = refract(sun_dir, normal, eta);

    // Trace to bottom plane
    let bottom_y = vol.bounds_min.y;
    let t = (bottom_y - surface_y) / refracted.y;
    let hit_pos = surface_pos + refracted * t;

    // Measure light convergence using finite differences
    // Sample neighboring positions to see how much light focuses/diverges
    let eps = 0.1;

    // Sample neighbors
    let h_x1 = vol.bounds_max.w + gerstner_waves_multi(world_xz + vec2<f32>(eps, 0.0), time, vol);
    let h_x2 = vol.bounds_max.w + gerstner_waves_multi(world_xz - vec2<f32>(eps, 0.0), time, vol);
    let h_z1 = vol.bounds_max.w + gerstner_waves_multi(world_xz + vec2<f32>(0.0, eps), time, vol);
    let h_z2 = vol.bounds_max.w + gerstner_waves_multi(world_xz - vec2<f32>(0.0, eps), time, vol);

    let n_x1 = calculate_normal(world_xz + vec2<f32>(eps, 0.0), time, vol);
    let n_x2 = calculate_normal(world_xz - vec2<f32>(eps, 0.0), time, vol);
    let n_z1 = calculate_normal(world_xz + vec2<f32>(0.0, eps), time, vol);
    let n_z2 = calculate_normal(world_xz - vec2<f32>(0.0, eps), time, vol);

    let r_x1 = refract(sun_dir, n_x1, eta);
    let r_x2 = refract(sun_dir, n_x2, eta);
    let r_z1 = refract(sun_dir, n_z1, eta);
    let r_z2 = refract(sun_dir, n_z2, eta);

    let t_x1 = (bottom_y - h_x1) / r_x1.y;
    let t_x2 = (bottom_y - h_x2) / r_x2.y;
    let t_z1 = (bottom_y - h_z1) / r_z1.y;
    let t_z2 = (bottom_y - h_z2) / r_z2.y;

    let hit_x1 = vec3<f32>(world_xz.x + eps, h_x1, world_xz.y) + r_x1 * t_x1;
    let hit_x2 = vec3<f32>(world_xz.x - eps, h_x2, world_xz.y) + r_x2 * t_x2;
    let hit_z1 = vec3<f32>(world_xz.x, h_z1, world_xz.y + eps) + r_z1 * t_z1;
    let hit_z2 = vec3<f32>(world_xz.x, h_z2, world_xz.y - eps) + r_z2 * t_z2;

    // Calculate area of light footprint
    let area = length(hit_x1 - hit_x2) * length(hit_z1 - hit_z2);
    let expected_area = eps * eps * 4.0; // 2eps × 2eps

    // Caustic intensity = inverse of area (light convergence)
    let intensity = expected_area / max(area, 0.001);

    return clamp(intensity, 0.0, 10.0);
}

/// Main compute shader entry point.
///
/// Each thread generates one texel of the caustics texture.
@compute @workgroup_size(32, 32, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let texel = vec2<i32>(gid.xy);
    let uv = vec2<f32>(gid.xy) / 512.0;

    // Map UV to world XZ coordinates
    let world_xz = uv * params.world_scale + params.world_offset;

    // Generate caustic intensity
    var intensity = 0.0;

    if arrayLength(&water_volumes) > 0u {
        let vol = water_volumes[0]; // Use first water volume for now

        // Only generate caustics if enabled
        if vol.caustics_params.x > 0.5 {
            intensity = generate_caustic(world_xz, params.time, vol);
            intensity *= vol.caustics_params.y; // Apply intensity multiplier
        }
    }

    // Write to output texture
    textureStore(output_texture, texel, vec4<f32>(intensity));
}
