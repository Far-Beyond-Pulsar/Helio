// Volumetric Radiance Cascades - Production Implementation
// Implements hierarchical radiance probes in world space for dynamic GI
//
// Algorithm: Radiance Cascades (Alexander Sannikov)
// - Multiple cascade levels with increasing probe spacing (2^N meters)
// - Probes placed on 3D volumetric grid in world space
// - Each probe traces rays and accumulates radiance
// - Coarser cascades merge data from finer cascades
// - Runtime: O(N * R) where N = probes, R = rays per probe

const PI: f32 = 3.141592653;
const INV_PI: f32 = 0.318309886;

struct GpuLightConfig {
    sun_direction: vec3<f32>,
    _pad0: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    sky_color: vec3<f32>,
    ambient_intensity: f32,
}
var<uniform> light_config: GpuLightConfig;

struct SceneUniforms {
    time: f32,
    cascade_index: u32,
    _pad0: f32,
    _pad1: f32,
}
var<uniform> scene: SceneUniforms;

var cascade_texture: texture_storage_2d<rgba16float, write>;
var prev_cascade_texture: texture_2d<f32>;
var prev_cascade_sampler: sampler;

// Probe grid configuration
// Total probes: 128x128 = 16,384 probes
// Texture layout: 1024x2048 can store 256x512 probes (131,072 probes)
// We use 128x128 grid = 16,384 probes arranged as 128x128 in texture
const PROBES_PER_AXIS: u32 = 128u;
const PROBES_TOTAL: u32 = PROBES_PER_AXIS * PROBES_PER_AXIS;

// World space bounds for probe grid
const WORLD_MIN: vec3<f32> = vec3<f32>(-50.0, -10.0, -50.0);
const WORLD_MAX: vec3<f32> = vec3<f32>(50.0, 30.0, 50.0);

// Ray tracing configuration
const NUM_RAYS_PER_PROBE: u32 = 32u;
const MAX_RAY_DISTANCE: f32 = 100.0;

// Get probe spacing for current cascade level
fn get_probe_spacing(cascade_level: u32) -> f32 {
    // Cascade 0: 2m, Cascade 1: 4m, Cascade 2: 8m, etc.
    return f32(1u << (cascade_level + 1u));
}

// Convert probe 2D index to 3D world position
fn probe_index_to_world_pos(probe_idx: vec2<u32>, cascade_level: u32) -> vec3<f32> {
    let spacing = get_probe_spacing(cascade_level);
    let world_size = WORLD_MAX - WORLD_MIN;

    // Map probe index to world position
    let t = vec2<f32>(probe_idx) / f32(PROBES_PER_AXIS);

    // Y is determined by cascade level (vertical stratification)
    let y_layer = f32(cascade_level) / 5.0;

    return WORLD_MIN + vec3<f32>(
        t.x * world_size.x,
        mix(WORLD_MIN.y, WORLD_MAX.y, y_layer),
        t.y * world_size.z
    );
}

// Convert world position to nearest probe index
fn world_pos_to_probe_index(world_pos: vec3<f32>, cascade_level: u32) -> vec2<i32> {
    let world_size = WORLD_MAX - WORLD_MIN;
    let local_pos = (world_pos - WORLD_MIN) / world_size;

    let probe_idx = vec2<i32>(
        i32(local_pos.x * f32(PROBES_PER_AXIS)),
        i32(local_pos.z * f32(PROBES_PER_AXIS))
    );

    return clamp(probe_idx, vec2<i32>(0), vec2<i32>(i32(PROBES_PER_AXIS) - 1));
}

// Fibonacci sphere sampling for uniform hemisphere distribution
fn fibonacci_hemisphere(i: u32, n: u32, normal: vec3<f32>) -> vec3<f32> {
    let phi = 2.0 * PI * f32(i) / 1.618033988749;
    let cos_theta = 1.0 - f32(i) / f32(n);
    let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));

    // Direction in +Y hemisphere
    let local_dir = vec3<f32>(
        cos(phi) * sin_theta,
        cos_theta,
        sin(phi) * sin_theta
    );

    // Build tangent space from normal
    let up = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.999);
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    // Transform to world space
    return normalize(tangent * local_dir.x + normal * local_dir.y + bitangent * local_dir.z);
}

// Hash for pseudo-random numbers
fn hash22(p: vec2<f32>) -> vec2<f32> {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

// Evaluate procedural sky
fn evaluate_sky(direction: vec3<f32>) -> vec3<f32> {
    let sun_dir = normalize(-light_config.sun_direction);
    let sun_dot = max(dot(direction, sun_dir), 0.0);

    // Sun disk
    let sun_intensity = pow(sun_dot, 256.0) * 100.0;
    let sun_contribution = light_config.sun_color * sun_intensity;

    // Atmospheric scattering approximation
    let horizon_factor = pow(1.0 - abs(direction.y), 4.0);
    let sky_gradient = mix(
        light_config.sky_color,
        light_config.sky_color * vec3<f32>(1.0, 0.8, 0.6),
        horizon_factor
    );

    // Sun glow
    let sun_glow = pow(sun_dot, 8.0) * light_config.sun_color * 2.0;

    let sky_radiance = (sky_gradient + sun_glow) * light_config.ambient_intensity * 2.0;

    return sun_contribution + sky_radiance;
}

// Sample radiance from previous (finer) cascade
fn sample_previous_cascade(world_pos: vec3<f32>, direction: vec3<f32>, prev_level: u32) -> vec3<f32> {
    // Get probe index for world position in previous cascade
    let probe_idx = world_pos_to_probe_index(world_pos, prev_level);

    // Convert to UV coordinates
    let uv = (vec2<f32>(probe_idx) + 0.5) / f32(PROBES_PER_AXIS);

    // Sample previous cascade
    return textureSampleLevel(prev_cascade_texture, prev_cascade_sampler, uv, 0.0).rgb;
}

// Trace ray through scene
// TODO: Integrate with actual scene geometry via:
//  - Depth buffer ray marching
//  - BVH acceleration structure
//  - Signed distance fields
//  - Voxel grid
// For now, returns environment lighting
fn trace_scene_ray(origin: vec3<f32>, direction: vec3<f32>, max_dist: f32) -> vec3<f32> {
    // Simple ground plane intersection
    if (direction.y < -0.01) {
        let t = -origin.y / direction.y;
        if (t > 0.0 && t < max_dist) {
            let hit_pos = origin + direction * t;

            // Checkerboard pattern
            let checker = select(0.3, 0.7,
                (i32(floor(hit_pos.x)) + i32(floor(hit_pos.z))) % 2 == 0
            );

            // Simple diffuse shading from sun
            let sun_dir = normalize(-light_config.sun_direction);
            let ndotl = max(dot(vec3<f32>(0.0, 1.0, 0.0), sun_dir), 0.0);

            return vec3<f32>(checker) * light_config.sun_color * light_config.sun_intensity * ndotl;
        }
    }

    // No hit - return sky
    return evaluate_sky(direction);
}

// Compute radiance for a probe by tracing rays
fn compute_probe_radiance(probe_pos: vec3<f32>, cascade_level: u32) -> vec3<f32> {
    var accumulated_radiance = vec3<f32>(0.0);
    let probe_normal = vec3<f32>(0.0, 1.0, 0.0); // Upward facing for now

    // Trace multiple rays in hemisphere
    for (var ray_idx = 0u; ray_idx < NUM_RAYS_PER_PROBE; ray_idx++) {
        let ray_dir = fibonacci_hemisphere(ray_idx, NUM_RAYS_PER_PROBE, probe_normal);

        var radiance: vec3<f32>;

        if (cascade_level == 0u) {
            // Finest cascade - trace actual scene
            radiance = trace_scene_ray(probe_pos, ray_dir, MAX_RAY_DISTANCE);
        } else {
            // Coarser cascades - sample from finer cascade
            let sample_pos = probe_pos + ray_dir * get_probe_spacing(cascade_level - 1u);
            radiance = sample_previous_cascade(sample_pos, ray_dir, cascade_level - 1u);

            // Fallback to direct sampling if needed
            if (length(radiance) < 0.001) {
                radiance = trace_scene_ray(probe_pos, ray_dir, MAX_RAY_DISTANCE);
            }
        }

        // Cosine weighting for diffuse surfaces
        let ndotl = max(dot(probe_normal, ray_dir), 0.0);
        accumulated_radiance += radiance * ndotl;
    }

    // Monte Carlo integration: divide by N and multiply by 2π (hemisphere integral)
    // The cosine weighting and hemisphere integral combine to give us 2π / N
    accumulated_radiance *= (2.0 * PI) / f32(NUM_RAYS_PER_PROBE);

    return accumulated_radiance;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let texture_dims = textureDimensions(cascade_texture);

    // Check bounds
    if (global_id.x >= texture_dims.x || global_id.y >= texture_dims.y) {
        return;
    }

    // Map to probe index (texture is 1024x2048, we use first 128x128 for probes)
    let probe_idx = global_id.xy;

    if (probe_idx.x >= PROBES_PER_AXIS || probe_idx.y >= PROBES_PER_AXIS) {
        // Outside probe grid - clear to black
        textureStore(cascade_texture, global_id.xy, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    // Get probe world position
    let probe_pos = probe_index_to_world_pos(probe_idx, scene.cascade_index);

    // Compute radiance for this probe
    let radiance = compute_probe_radiance(probe_pos, scene.cascade_index);

    // Store result
    textureStore(cascade_texture, global_id.xy, vec4<f32>(radiance, 1.0));
}
