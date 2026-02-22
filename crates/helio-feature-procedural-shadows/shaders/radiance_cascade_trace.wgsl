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
var scene_depth: texture_depth_2d;
var scene_color: texture_2d<f32>;
var linear_sampler: sampler;

struct Camera {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad: f32,
}
var<uniform> camera: Camera;

struct GpuLight {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
}
var<uniform> lights: array<GpuLight, 16>;
var<uniform> num_lights: u32;

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

// No hardcoded lighting - all light comes from the scene
fn evaluate_environment(direction: vec3<f32>) -> vec3<f32> {
    // Return black - no environment lighting
    // All light must come from emissive objects in the scene
    return vec3<f32>(0.0);
}

// Check if ray intersects with a sphere light
fn ray_sphere_intersection(origin: vec3<f32>, direction: vec3<f32>, center: vec3<f32>, radius: f32) -> f32 {
    let oc = origin - center;
    let a = dot(direction, direction);
    let b = 2.0 * dot(oc, direction);
    let c = dot(oc, oc) - radius * radius;
    let discriminant = b * b - 4.0 * a * c;

    if (discriminant < 0.0) {
        return -1.0; // No hit
    }

    let t = (-b - sqrt(discriminant)) / (2.0 * a);
    return select(-1.0, t, t > 0.001); // Return t if positive, else -1
}

// Check if ray hits any lights and return radiance
fn check_light_intersections(origin: vec3<f32>, direction: vec3<f32>) -> vec3<f32> {
    var closest_t = 999999.0;
    var hit_radiance = vec3<f32>(0.0);

    for (var i = 0u; i < num_lights; i++) {
        let light = lights[i];
        let t = ray_sphere_intersection(origin, direction, light.position, light.radius);

        if (t > 0.0 && t < closest_t) {
            closest_t = t;
            hit_radiance = light.color * light.intensity;
        }
    }

    return hit_radiance;
}

// Sample radiance from cascade 0 (finest) at world position with bilinear interpolation
fn sample_cascade_0(world_pos: vec3<f32>) -> vec3<f32> {
    // Convert world position to probe UV coordinates
    let world_size = WORLD_MAX - WORLD_MIN;
    let local_pos = (world_pos - WORLD_MIN) / world_size;
    let clamped = clamp(local_pos, vec3<f32>(0.0), vec3<f32>(1.0));

    // Map XZ to probe grid coordinates
    let probe_coord = vec2<f32>(clamped.x, clamped.z) * f32(PROBES_PER_AXIS);
    let probe_idx_base = vec2<i32>(probe_coord);
    let blend = fract(probe_coord);

    // Sample 4 neighboring probes for bilinear interpolation
    let uv_00 = (vec2<f32>(probe_idx_base) + 0.5) / f32(PROBES_PER_AXIS);
    let uv_10 = (vec2<f32>(probe_idx_base + vec2<i32>(1, 0)) + 0.5) / f32(PROBES_PER_AXIS);
    let uv_01 = (vec2<f32>(probe_idx_base + vec2<i32>(0, 1)) + 0.5) / f32(PROBES_PER_AXIS);
    let uv_11 = (vec2<f32>(probe_idx_base + vec2<i32>(1, 1)) + 0.5) / f32(PROBES_PER_AXIS);

    // Sample cascade 0 texture at 4 corners
    let r00 = textureSampleLevel(prev_cascade_texture, prev_cascade_sampler, uv_00, 0.0).rgb;
    let r10 = textureSampleLevel(prev_cascade_texture, prev_cascade_sampler, uv_10, 0.0).rgb;
    let r01 = textureSampleLevel(prev_cascade_texture, prev_cascade_sampler, uv_01, 0.0).rgb;
    let r11 = textureSampleLevel(prev_cascade_texture, prev_cascade_sampler, uv_11, 0.0).rgb;

    // Bilinear interpolation
    let r0 = mix(r00, r10, blend.x);
    let r1 = mix(r01, r11, blend.x);
    return mix(r0, r1, blend.y);
}

// Test if there's a clear path to a light (shadow test)
fn test_light_visibility(origin: vec3<f32>, light_pos: vec3<f32>) -> bool {
    let to_light = light_pos - origin;
    let light_dist = length(to_light);
    let light_dir = to_light / light_dist;

    // March through depth buffer to check occlusion
    let depth_size = textureDimensions(scene_depth);
    let max_steps = 32u;
    let step_size = light_dist / f32(max_steps);

    var current_pos = origin + light_dir * 0.1; // Offset to avoid self-intersection

    for (var i = 0u; i < max_steps; i++) {
        // Project to screen space
        let clip_pos = camera.view_proj * vec4<f32>(current_pos, 1.0);
        let ndc = clip_pos.xyz / clip_pos.w;
        let screen_uv = ndc.xy * 0.5 + 0.5;

        // Check if on screen
        if (screen_uv.x >= 0.0 && screen_uv.x <= 1.0 && screen_uv.y >= 0.0 && screen_uv.y <= 1.0) {
            let texel_coord = vec2<i32>(screen_uv * vec2<f32>(depth_size));
            let depth_sample = textureLoad(scene_depth, texel_coord, 0);

            // If ray is behind surface, light is occluded
            if (ndc.z > depth_sample + 0.001) {
                return false;
            }
        }

        current_pos += light_dir * step_size;

        // If we've reached the light, path is clear
        if (length(current_pos - origin) >= light_dist * 0.95) {
            break;
        }
    }

    return true;
}

// Reconstruct world position from depth buffer
fn reconstruct_world_pos(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world_pos = camera.inv_view_proj * ndc;
    return world_pos.xyz / world_pos.w;
}

// Estimate surface normal from depth buffer using screen-space derivatives
fn estimate_normal(screen_uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let depth_size = textureDimensions(scene_depth);
    let texel_size = 1.0 / vec2<f32>(depth_size);

    // Sample neighboring depths
    let d_right = textureLoad(scene_depth, vec2<i32>((screen_uv + vec2<f32>(texel_size.x, 0.0)) * vec2<f32>(depth_size)), 0);
    let d_up = textureLoad(scene_depth, vec2<i32>((screen_uv + vec2<f32>(0.0, texel_size.y)) * vec2<f32>(depth_size)), 0);

    // Reconstruct world positions
    let p_center = reconstruct_world_pos(screen_uv, depth);
    let p_right = reconstruct_world_pos(screen_uv + vec2<f32>(texel_size.x, 0.0), d_right);
    let p_up = reconstruct_world_pos(screen_uv + vec2<f32>(0.0, texel_size.y), d_up);

    // Compute normal from cross product
    let dx = p_right - p_center;
    let dy = p_up - p_center;
    return normalize(cross(dx, dy));
}

// Screen-space ray marching through depth buffer
fn trace_scene_ray(origin: vec3<f32>, direction: vec3<f32>, max_dist: f32) -> vec3<f32> {
    // First check if ray hits any lights directly
    let light_radiance = check_light_intersections(origin, direction);
    if (length(light_radiance) > 0.0) {
        return light_radiance;
    }

    // Then check scene geometry via depth buffer
    let depth_size = textureDimensions(scene_depth);
    let max_steps = 64u;
    let step_size = max_dist / f32(max_steps);

    var current_pos = origin;

    for (var i = 0u; i < max_steps; i++) {
        current_pos += direction * step_size;

        // Project to screen space
        let clip_pos = camera.view_proj * vec4<f32>(current_pos, 1.0);
        let ndc = clip_pos.xyz / clip_pos.w;
        let screen_uv = ndc.xy * 0.5 + 0.5;

        // Check if on screen
        if (screen_uv.x < 0.0 || screen_uv.x > 1.0 || screen_uv.y < 0.0 || screen_uv.y > 1.0) {
            break;
        }

        // Convert UV to texel coordinates for depth load
        let texel_coord = vec2<i32>(screen_uv * vec2<f32>(depth_size));

        // Load depth directly
        let depth_sample = textureLoad(scene_depth, texel_coord, 0);

        // Check if ray hit surface
        if (ndc.z > depth_sample) {
            // Reconstruct hit position and normal
            let hit_pos = reconstruct_world_pos(screen_uv, depth_sample);
            let hit_normal = estimate_normal(screen_uv, depth_sample);

            // Sample material albedo at hit point
            let albedo = textureSampleLevel(scene_color, linear_sampler, screen_uv, 0.0).rgb;

            // Sample indirect light from cascade 0 (finest level)
            var total_radiance = sample_cascade_0(hit_pos);

            // Add direct lighting from all lights with shadow testing
            for (var light_idx = 0u; light_idx < num_lights; light_idx++) {
                let light = lights[light_idx];
                let to_light = light.position - hit_pos;
                let light_dist = length(to_light);
                let light_dir = to_light / light_dist;

                // Only add light if surface faces the light
                let n_dot_l = max(dot(hit_normal, light_dir), 0.0);
                if (n_dot_l > 0.0) {
                    // Test visibility (shadow)
                    if (test_light_visibility(hit_pos, light.position)) {
                        // Add direct light contribution with inverse square falloff
                        let attenuation = 1.0 / (light_dist * light_dist + 1.0);
                        total_radiance += light.color * light.intensity * attenuation * n_dot_l;
                    }
                }
            }

            // Multiply by albedo (material color)
            return albedo * total_radiance;
        }
    }

    // No hit - return black (no environment lighting)
    return evaluate_environment(direction);
}

// Compute radiance for a probe by tracing rays
fn compute_probe_radiance(probe_pos: vec3<f32>, cascade_level: u32) -> vec3<f32> {
    var accumulated_radiance = vec3<f32>(0.0);
    let probe_normal = vec3<f32>(0.0, 1.0, 0.0); // Upward facing for now

    // Trace multiple rays in hemisphere
    for (var ray_idx = 0u; ray_idx < NUM_RAYS_PER_PROBE; ray_idx++) {
        let ray_dir = fibonacci_hemisphere(ray_idx, NUM_RAYS_PER_PROBE, probe_normal);

        var radiance: vec3<f32>;

        // Trace the scene
        radiance = trace_scene_ray(probe_pos, ray_dir, MAX_RAY_DISTANCE);

        // Cosine weighting for diffuse surfaces
        let ndotl = max(dot(probe_normal, ray_dir), 0.0);
        accumulated_radiance += radiance * ndotl;
    }

    // Monte Carlo integration with cosine weighting
    // Integral over hemisphere: π for diffuse, divided by number of samples
    accumulated_radiance *= PI / f32(NUM_RAYS_PER_PROBE);

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
