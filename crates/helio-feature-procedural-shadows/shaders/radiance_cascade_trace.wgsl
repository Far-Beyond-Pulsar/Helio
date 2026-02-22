// Radiance Cascades - Ray tracing compute shader
// Traces rays from surface-based probes and builds cascaded radiance fields for full GI

// Constants
const PI: f32 = 3.141592653;
const INV_PI: f32 = 0.318309886;
const MAX_RAY_DIST: f32 = 100.0;
const EPSILON: f32 = 0.001;

// Light configuration
struct LightConfig {
    sun_direction: vec3<f32>,
    _pad0: f32,
    sun_color: vec3<f32>,
    sun_intensity: f32,
    sky_color: vec3<f32>,
    ambient_intensity: f32,
}
var<uniform> light_config: LightConfig;

// Radiance cascade output texture
var cascade_texture: texture_storage_2d<rgba16float, write>;

// Previous cascade texture for merging (read from higher cascade)
var prev_cascade_texture: texture_2d<f32>;
var prev_cascade_sampler: sampler;

// Scene uniforms (for dynamic scene data)
struct SceneUniforms {
    time: f32,
    cascade_index: u32,  // Which cascade level we're computing (0-5)
    _pad0: f32,
    _pad1: f32,
}
var<uniform> scene: SceneUniforms;

// Hit record for ray tracing
struct HitRecord {
    hit: bool,
    t: f32,
    position: vec3<f32>,
    normal: vec3<f32>,
    uv: vec2<f32>,
    albedo: vec3<f32>,
    emissive: vec3<f32>,
    is_mirror: bool,
}

// Surface geometry definition
struct SurfaceGeometry {
    position: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    normal: vec3<f32>,
    size: vec2<f32>,          // Physical size in world units
    resolution: vec2<u32>,    // Probe grid resolution
    uv_offset: vec2<u32>,     // Offset in texture
}

// Get hardcoded surface geometries (matching the shader example scene)
fn get_surface_geometry(surface_id: u32) -> SurfaceGeometry {
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
    else if (surface_id == 7u) {
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

// Ray-quad intersection
fn intersect_quad(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    position: vec3<f32>,
    tangent: vec3<f32>,
    bitangent: vec3<f32>,
    normal: vec3<f32>,
    size: vec2<f32>
) -> HitRecord {
    var hit: HitRecord;
    hit.hit = false;

    let nor_dot = dot(normal, ray_dir);
    let p_dot = dot(normal, ray_origin - position);

    if (sign(nor_dot * p_dot) >= -0.5) {
        return hit;
    }

    let t = -p_dot / nor_dot;
    if (t < EPSILON) {
        return hit;
    }

    let hit_point = ray_origin + ray_dir * t;
    let local_point = hit_point - position;
    let u = dot(local_point, tangent);
    let v = dot(local_point, bitangent);

    if (u >= 0.0 && u <= size.x && v >= 0.0 && v <= size.y) {
        hit.hit = true;
        hit.t = t;
        hit.position = hit_point;
        hit.normal = normal;
        hit.uv = vec2<f32>(u / size.x, v / size.y);
    }

    return hit;
}

// Ray-sphere intersection
fn intersect_sphere(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    radius: f32
) -> HitRecord {
    var hit: HitRecord;
    hit.hit = false;

    let oc = ray_origin - center;
    let a = dot(oc, oc) - radius * radius;
    let b = 2.0 * dot(oc, ray_dir);
    let discriminant = b * b * 0.25 - a;

    if (dot(oc, ray_dir) < 0.0 && discriminant > 0.0) {
        let t = -b * 0.5 - sqrt(discriminant);
        if (t > EPSILON) {
            hit.hit = true;
            hit.t = t;
            hit.position = ray_origin + ray_dir * t;
            hit.normal = normalize(hit.position - center);
        }
    }

    return hit;
}

// Ray-box intersection
fn intersect_box(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    half_size: vec3<f32>
) -> HitRecord {
    var hit: HitRecord;
    hit.hit = false;

    let inv_dir = 1.0 / ray_dir;
    let t_min = (center - half_size - ray_origin) * inv_dir;
    let t_max = (center + half_size - ray_origin) * inv_dir;

    let t1 = min(t_min, t_max);
    let t2 = max(t_min, t_max);

    let t_near = max(max(t1.x, t1.y), t1.z);
    let t_far = min(min(t2.x, t2.y), t2.z);

    if (t_near < t_far && t_far > EPSILON) {
        let t = max(t_near, EPSILON);
        hit.hit = true;
        hit.t = t;
        hit.position = ray_origin + ray_dir * t;

        // Calculate normal
        let local_pos = hit.position - center;
        let d = abs(local_pos / half_size);
        if (d.x > d.y && d.x > d.z) {
            hit.normal = vec3<f32>(sign(local_pos.x), 0.0, 0.0);
        } else if (d.y > d.z) {
            hit.normal = vec3<f32>(0.0, sign(local_pos.y), 0.0);
        } else {
            hit.normal = vec3<f32>(0.0, 0.0, sign(local_pos.z));
        }
    }

    return hit;
}

// Check if position is inside interior wall holes
fn is_interior_intersection(pos: vec3<f32>) -> bool {
    // Circle cutout at (0.5, 0)
    if (length(pos.xy - vec2<f32>(0.5, 0.0)) < 0.25) {
        return true;
    }
    // Circle cutout at (0.87, 0.25)
    if (length(pos.xy - vec2<f32>(0.87, 0.25)) < 0.12) {
        return true;
    }
    return false;
}

// Trace ray through scene
fn trace_ray(origin: vec3<f32>, direction: vec3<f32>, max_t: f32) -> HitRecord {
    var closest_hit: HitRecord;
    closest_hit.hit = false;
    closest_hit.t = max_t;

    // Test all surfaces
    for (var i = 0u; i < 8u; i++) {
        let surf = get_surface_geometry(i);
        var hit = intersect_quad(origin, direction, surf.position, surf.tangent,
                                 surf.bitangent, surf.normal, surf.size);

        if (hit.hit && hit.t < closest_hit.t) {
            // Skip interior wall holes
            if (i >= 6u && is_interior_intersection(hit.position)) {
                continue;
            }

            hit.albedo = vec3<f32>(0.9);  // Default white

            // Color walls
            if (i == 2u) {
                hit.albedo = vec3<f32>(0.9, 0.1, 0.1);  // Red wall
            } else if (i == 3u) {
                hit.albedo = vec3<f32>(0.05, 0.95, 0.1);  // Green wall
            } else if (i >= 6u) {
                hit.albedo = vec3<f32>(0.99);  // Interior walls
            }

            hit.emissive = vec3<f32>(0.0);
            hit.is_mirror = false;
            closest_hit = hit;
        }
    }

    // Test mirror sphere
    let sphere_hit = intersect_sphere(origin, direction, vec3<f32>(0.15, 0.1005, 0.3), 0.1);
    if (sphere_hit.hit && sphere_hit.t < closest_hit.t) {
        closest_hit = sphere_hit;
        closest_hit.albedo = vec3<f32>(1.0);
        closest_hit.emissive = vec3<f32>(0.0);
        closest_hit.is_mirror = true;
    }

    // Test mirror box
    let box_hit = intersect_box(origin, direction, vec3<f32>(0.86, 0.14, 0.86), vec3<f32>(0.08));
    if (box_hit.hit && box_hit.t < closest_hit.t) {
        closest_hit = box_hit;
        closest_hit.albedo = vec3<f32>(1.0);
        closest_hit.emissive = vec3<f32>(0.0);
        closest_hit.is_mirror = true;
    }

    // Procedural rotating object (example from shader)
    let nt = 1.0 + scene.time * 0.2;
    let obj_center = vec3<f32>(
        0.21 + (sin(nt) * 0.5 + 0.5) * 0.58,
        0.5,
        0.21 + (cos(nt) * 0.5 + 0.5) * 0.58
    );

    // This object is just for visual interest - simplified version

    return closest_hit;
}

// Sample previous cascade for indirect lighting
fn sample_prev_cascade(
    surf: SurfaceGeometry,
    probe_pos: vec3<f32>,
    normal: vec3<f32>
) -> vec3<f32> {
    // For the coarsest cascade, return sky/ambient
    if (scene.cascade_index >= 4u) {
        return light_config.sky_color * light_config.ambient_intensity;
    }

    // Sample from previous (coarser) cascade
    // This is a simplified version - the full implementation would do proper weighted sampling

    // Find UV in texture based on probe position
    let local_pos = probe_pos - surf.position;
    let u = dot(local_pos, surf.tangent) / surf.size.x;
    let v = dot(local_pos, surf.bitangent) / surf.size.y;

    // Offset for previous cascade (stored at resolution/2)
    let cascade_res = vec2<f32>(surf.resolution) / f32(1u << (scene.cascade_index + 1u));
    let uv = (vec2<f32>(u, v) * cascade_res + vec2<f32>(surf.uv_offset)) / vec2<f32>(surf.resolution);

    // Sample previous cascade texture
    let radiance = textureSampleLevel(prev_cascade_texture, prev_cascade_sampler, uv, 0.0);

    return radiance.rgb;
}

// Compute probe position and ray direction from UV coordinates
struct ProbeRay {
    probe_pos: vec3<f32>,
    ray_dir: vec3<f32>,
    valid: bool,
}

fn compute_probe_ray(surf: SurfaceGeometry, uv: vec2<u32>) -> ProbeRay {
    var result: ProbeRay;
    result.valid = false;

    let cascade_idx = scene.cascade_index;
    let probe_size = f32(1u << (cascade_idx + 1u));  // 2, 4, 8, 16, 32
    let probe_positions = vec2<f32>(surf.resolution) / probe_size;

    // Get probe grid position
    let mod_uv = vec2<f32>(uv % surf.resolution);
    let probe_uv_idx = mod_uv / probe_positions;
    let probe_indices = vec2<u32>(probe_uv_idx);

    // Compute probe position in world space
    let probe_grid_pos = (vec2<f32>(probe_indices) + 0.5) * probe_size / vec2<f32>(surf.resolution);
    result.probe_pos = surf.position +
                      surf.tangent * probe_grid_pos.x * surf.size.x +
                      surf.bitangent * probe_grid_pos.y * surf.size.y +
                      surf.normal * EPSILON;

    // Compute ray direction (hemispherical distribution)
    let ray_uv = vec2<f32>(uv % u32(probe_size));
    let probe_rel = ray_uv - probe_size * 0.5;

    // Theta: angle from normal
    let probe_theta_i = max(abs(probe_rel.x), abs(probe_rel.y));
    let probe_theta = probe_theta_i / probe_size * PI;

    // Phi: azimuthal angle
    var probe_phi = 0.0;
    let half_size = probe_size * 0.5;

    if (probe_rel.x + 0.5 > probe_theta_i && probe_rel.y - 0.5 > -probe_theta_i) {
        probe_phi = probe_rel.x - probe_rel.y;
    } else if (probe_rel.y - 0.5 < -probe_theta_i && probe_rel.x - 0.5 > -probe_theta_i) {
        probe_phi = probe_theta_i * 2.0 - probe_rel.y - probe_rel.x;
    } else if (probe_rel.x - 0.5 < -probe_theta_i && probe_rel.y + 0.5 < probe_theta_i) {
        probe_phi = probe_theta_i * 4.0 - probe_rel.x + probe_rel.y;
    } else if (probe_rel.y + 0.5 > probe_theta_i && probe_rel.x + 0.5 < probe_theta_i) {
        probe_phi = probe_theta_i * 8.0 - (probe_rel.y - probe_rel.x);
    }

    probe_phi = probe_phi * PI * 2.0 / (4.0 + 8.0 * floor(probe_theta_i));

    // Convert spherical to Cartesian in tangent space
    let local_dir = vec3<f32>(
        sin(probe_phi) * sin(probe_theta),
        cos(probe_phi) * sin(probe_theta),
        cos(probe_theta)
    );

    // Transform to world space
    result.ray_dir = normalize(
        surf.tangent * local_dir.x +
        surf.bitangent * local_dir.y +
        surf.normal * local_dir.z
    );

    result.valid = true;
    return result;
}

// Main compute shader
@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let uv = global_id.xy;
    let tex_size = textureDimensions(cascade_texture);

    if (uv.x >= tex_size.x || uv.y >= tex_size.y) {
        return;
    }

    // Determine which surface this pixel belongs to
    var surface_id = 0u;
    var found_surface = false;

    for (var i = 0u; i < 8u; i++) {
        let surf = get_surface_geometry(i);
        let offset = surf.uv_offset;
        let cascade_res = surf.resolution / (1u << (scene.cascade_index + 1u));

        if (uv.x >= offset.x && uv.x < offset.x + cascade_res.x &&
            uv.y >= offset.y && uv.y < offset.y + cascade_res.y) {
            surface_id = i;
            found_surface = true;
            break;
        }
    }

    if (!found_surface) {
        // Outside all surfaces - write black
        textureStore(cascade_texture, vec2<i32>(uv), vec4<f32>(0.0));
        return;
    }

    let surf = get_surface_geometry(surface_id);
    let probe_ray = compute_probe_ray(surf, uv - surf.uv_offset);

    if (!probe_ray.valid) {
        textureStore(cascade_texture, vec2<i32>(uv), vec4<f32>(0.0));
        return;
    }

    // Trace ray
    let cascade_size = f32(1u << (scene.cascade_index + 1u));
    let max_ray_dist = (1.0 / 64.0) * cascade_size * 2.0;

    var hit = trace_ray(probe_ray.probe_pos, probe_ray.ray_dir, max_ray_dist);

    var radiance = vec3<f32>(0.0);
    var distance = -1.0;

    if (hit.hit) {
        distance = hit.t;

        if (hit.is_mirror) {
            // Simple mirror reflection - just reflect and trace one more ray
            let reflect_dir = reflect(probe_ray.ray_dir, hit.normal);
            let reflect_hit = trace_ray(hit.position + hit.normal * EPSILON, reflect_dir, MAX_RAY_DIST);

            if (reflect_hit.hit) {
                if (!reflect_hit.is_mirror && length(reflect_hit.emissive) == 0.0) {
                    // Sample indirect lighting for reflected hit (non-emissive surface)
                    let indirect = sample_prev_cascade(surf, reflect_hit.position, reflect_hit.normal);

                    // Add direct sun lighting
                    if (dot(reflect_hit.normal, light_config.sun_direction) > 0.0) {
                        let shadow_ray = trace_ray(
                            reflect_hit.position + reflect_hit.normal * EPSILON,
                            light_config.sun_direction,
                            MAX_RAY_DIST
                        );

                        if (!shadow_ray.hit) {
                            radiance += light_config.sun_color * light_config.sun_intensity *
                                       dot(reflect_hit.normal, light_config.sun_direction);
                        }
                    }

                    radiance += indirect;
                    radiance *= reflect_hit.albedo;
                }
            } else {
                radiance = light_config.sky_color;
            }
        } else if (length(hit.emissive) > 0.0) {
            // Emissive surface
            radiance = hit.emissive;
        } else {
            // Regular diffuse surface
            // Sample indirect lighting from previous cascade
            let indirect = sample_prev_cascade(surf, hit.position, hit.normal);

            // Add direct sun lighting
            if (dot(hit.normal, light_config.sun_direction) > 0.0) {
                let shadow_ray = trace_ray(
                    hit.position + hit.normal * EPSILON,
                    light_config.sun_direction,
                    MAX_RAY_DIST
                );

                if (!shadow_ray.hit) {
                    radiance += light_config.sun_color * light_config.sun_intensity *
                               dot(hit.normal, light_config.sun_direction);
                }
            }

            radiance += indirect;
            radiance *= hit.albedo;
        }
    } else {
        // Sky
        radiance = light_config.sky_color;
        distance = -1.0;
    }

    // Apply hemisphere normalization and BRDF (diffuse cosine)
    let probe_rel = vec2<f32>((uv - surf.uv_offset) % u32(cascade_size)) - cascade_size * 0.5;
    let probe_theta_i = max(abs(probe_rel.x), abs(probe_rel.y));
    let probe_theta = probe_theta_i / cascade_size * PI;

    // Hemisphere solid angle weighting
    let theta_step = PI / cascade_size;
    let solid_angle = (cos(probe_theta - theta_step) - cos(probe_theta + theta_step)) /
                     (4.0 + 8.0 * floor(probe_theta_i));

    radiance *= solid_angle;
    radiance *= cos(probe_theta);  // Diffuse BRDF (Lambertian)

    // Merge with previous cascade if not the finest
    if (scene.cascade_index > 0u) {
        let interp_min_dist = (1.0 / 256.0) * cascade_size * 1.5;
        let interp_max_interval = interp_min_dist;

        if (distance > 0.0) {
            let blend = 1.0 - clamp((distance - interp_min_dist) / interp_max_interval, 0.0, 1.0);
            let prev = sample_prev_cascade(surf, probe_ray.probe_pos, surf.normal);
            radiance = mix(prev, radiance, blend);
        }
    }

    // Write result
    textureStore(cascade_texture, vec2<i32>(uv), vec4<f32>(radiance, distance));
}
