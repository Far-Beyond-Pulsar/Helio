// Volumetric water rendering — replaces the box-mesh volume walls with a
// screen-space raymarch. For each pixel we compute the view ray's intersection
// with the water volume AABB, then raymarch through the interior accumulating
// Beer-Lambert absorption and in-scattering.
//
// Bindings (identical to the surface shaders for layout compatibility)
//   0  camera           uniform
//   1  water_volumes    storage read
//   2  water_sim        texture_2d<f32>  (RGBA16F: R=height, B/A=normal.xz)
//   3  water_samp       sampler
//   4  caustics_tex     texture_2d<f32>
//   5  shared_samp      sampler
//   6  scene_color      texture_2d<f32>
//   7  viewport         uniform vec4f
//   8  depth_texture    texture_depth_2d
//   9  depth_sampler    sampler
//   10 gbuffer_normal   texture_2d<f32>

struct Camera {
    view:           mat4x4f,
    proj:           mat4x4f,
    view_proj:      mat4x4f,
    inv_view_proj:  mat4x4f,
    position_near:  vec4f,
    forward_far:    vec4f,
    jitter_frame:   vec4f,
    prev_view_proj: mat4x4f,
}

struct WaterVolume {
    bounds_min:            vec4f,
    bounds_max:            vec4f,  // w = surface_height
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,
    extinction:            vec4f,
    reflection_refraction: vec4f,
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,
    pad1: vec4f, pad2: vec4f, pad3: vec4f,
}

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var<storage, read> volumes:       array<WaterVolume>;
@group(0) @binding(2) var water_sim:     texture_2d<f32>;
@group(0) @binding(3) var water_samp:    sampler;
@group(0) @binding(4) var caustics_tex:  texture_2d<f32>;
@group(0) @binding(5) var shared_samp:   sampler;
@group(0) @binding(6) var scene_color:   texture_2d<f32>;
@group(0) @binding(7) var<uniform>       viewport:      vec4f;
@group(0) @binding(8) var depth_texture:   texture_depth_2d;
@group(0) @binding(9) var depth_sampler:   sampler;
@group(0) @binding(10) var gbuffer_normal: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.position = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2f(x, y);
    return out;
}

fn reconstruct_world_pos(uv: vec2f, depth: f32) -> vec3f {
    let ndc_xy = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let world_h = camera.inv_view_proj * vec4f(ndc_xy, depth, 1.0);
    return world_h.xyz / world_h.w;
}

// Ray-AABB intersection. Returns (t_entry, t_exit, hit_flag).
fn ray_aabb_intersect(origin: vec3f, dir: vec3f, bmin: vec3f, bmax: vec3f) -> vec3f {
    let inv_dir = vec3f(1.0) / dir;
    let t1 = (bmin - origin) * inv_dir;
    let t2 = (bmax - origin) * inv_dir;
    let tmin = min(t1, t2);
    let tmax = max(t1, t2);
    let t_entry = max(max(tmin.x, tmin.y), tmin.z);
    let t_exit = min(min(tmax.x, tmax.y), tmax.z);

    var result = vec3f(t_entry, t_exit, 0.0);
    if t_exit > max(t_entry, 0.0) {
        result.z = 1.0;
    }
    return result;
}

fn world_to_sim_uv(world_xz: vec2f, bmin: vec3f, bmax: vec3f) -> vec2f {
    return vec2f(
        (world_xz.x - bmin.x) / (bmax.x - bmin.x),
        (world_xz.y - bmin.z) / (bmax.z - bmin.z),
    );
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol = volumes[0];
    let bmin = vol.bounds_min.xyz;
    let bmax = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;
    let cam_pos = camera.position_near.xyz;
    let extinction = vol.extinction.rgb;
    let water_color = vol.water_color.rgb;

    // Reconstruct world position at the scene surface
    let depth = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0);
    let world_pos = reconstruct_world_pos(in.uv, depth);
    let view_dir = normalize(world_pos - cam_pos);
    let t_surface = distance(world_pos, cam_pos);

    // Sample the heightfield at the camera's XZ to get the displaced surface height
    let cam_sim_uv = world_to_sim_uv(cam_pos.xz, bmin, bmax);
    let heightfield = textureSampleLevel(water_sim, water_samp, cam_sim_uv, 0.0);
    let surface_at_cam = surface_h + heightfield.r * (surface_h - bmin.y);

    // Build the water volume AABB. The top varies with the heightfield across
    // the surface, but for the AABB we use a single value at the camera's XZ.
    // For pixels far from the camera, we sample the heightfield per-pixel below.
    let vol_min = vec3f(bmin.x, bmin.y, bmin.z);

    // Sample heightfield at the pixel's world position for the top of the volume
    let pixel_sim_uv = world_to_sim_uv(world_pos.xz, bmin, bmax);
    let pixel_hf = textureSampleLevel(water_sim, water_samp, pixel_sim_uv, 0.0);
    let surface_at_pixel = surface_h + pixel_hf.r * (surface_h - bmin.y);
    let vol_max = vec3f(bmax.x, surface_at_pixel, bmax.z);

    // Ray-AABB intersection
    let hit = ray_aabb_intersect(cam_pos, view_dir, vol_min, vol_max);
    let t_entry_raw = hit.x;
    let t_exit = hit.y;
    let is_hit = hit.z > 0.5;

    if !is_hit {
        discard;
    }

    // If the entry point is behind the scene surface, the volume is occluded
    if t_entry_raw > t_surface + 0.001 {
        discard;
    }

    // Clamp exit to scene surface (the ray stops at whatever is in front)
    let t_end = min(t_exit, t_surface);
    let t_start = max(t_entry_raw, 0.0);

    if t_end <= t_start + 0.0001 {
        discard;
    }

    // Total distance the ray travels through water
    let total_dist = t_end - t_start;

    // Beer-Lambert absorption over the full path
    let absorption = exp(-extinction * total_dist);

    // Single-scattering in-scattering approximation:
    // L = L_bg * T + L_water * (1 - T)
    // where T = exp(-sigma * d) is the transmittance
    var scattering = water_color * (1.0 - absorption);

    // Caustic contribution: sample caustics texture at the entry point's XZ
    let entry_pos = cam_pos + view_dir * t_start;
    let entry_sim_uv = world_to_sim_uv(entry_pos.xz, bmin, bmax);
    let caustic_sample = textureSampleLevel(caustics_tex, shared_samp, entry_sim_uv, 0.0);
    let caustic_strength = vol.caustics_params.y * vol.caustics_params.x;
    scattering += caustic_sample.rgb * caustic_strength * 0.5 * (1.0 - absorption);

    // Sample the background scene at this pixel
    let background = textureSampleLevel(scene_color, shared_samp, in.uv, 0.0).rgb;

    // Composite: background attenuated by water + in-scattering
    let result = background * absorption + scattering;

    // Fresnel-like view-angle effect for the water wall appearance
    let view_angle = abs(dot(normalize(vec3f(0.0, 1.0, 0.0)), -view_dir));
    let fresnel_wall = mix(0.6, 1.0, pow(1.0 - view_angle, 3.0));

    // Distance fade — far-away water volume fades out (avoids hard cutoffs)
    let dist_fade = smoothstep(80.0, 20.0, t_start);

    return vec4f(result * fresnel_wall, dist_fade);
}
