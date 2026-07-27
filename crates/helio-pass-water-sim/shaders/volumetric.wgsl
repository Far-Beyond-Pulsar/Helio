// Volumetric water rendering — screen-space raymarch through the water volume.
//
// Unlike the old box-mesh approach, this marches per-pixel through the actual
// water volume bounded by the heightfield surface on top, the floor (bmin.y)
// on the bottom, and the AABB sides.  At each step we sample the heightfield
// to know whether we are inside water, accumulating Beer-Lambert absorption,
// multi-scattered in-scattering, and caustic lighting.
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
    water_color:           vec4f,  // rgb = base colour
    extinction:            vec4f,  // rgb = Beer-Lambert absorption per metre
    reflection_refraction: vec4f,  // x = refl_strength, y = refr_strength
    caustics_params:       vec4f,  // x = enabled, y = intensity, z = scale
    fog_params:            vec4f,  // x = density (scattering coefficient), y = god_rays
    sim_params:            vec4f,  // x = ior, y = caustic_intensity
    shadow_params:         vec4f,  // x = rim_light, y = hitbox_shadow, z = ambient_occlusion
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

fn sample_surface(world_xz: vec2f, bmin: vec3f, bmax: vec3f, surface_h: f32, wave_amp: f32) -> f32 {
    let uv = world_to_sim_uv(world_xz, bmin, bmax);
    if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
        return surface_h;
    }
    let hf = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    return surface_h + hf.r * wave_amp;
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let gg = g * g;
    return (1.0 - gg) / (4.0 * 3.14159265 * pow(1.0 + gg - 2.0 * g * cos_theta, 1.5));
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol = volumes[0];
    let bmin = vol.bounds_min.xyz;
    let bmax = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;
    let cam_pos = camera.position_near.xyz;

    let wave_amp = max(surface_h - bmin.y, 0.1) * 0.15;
    let extinction = max(vol.extinction.rgb, vec3f(0.001));
    let water_color = max(vol.water_color.rgb, vec3f(0.001));
    let scatter_density = max(vol.fog_params.x, 0.001);
    let rim_strength = vol.shadow_params.x;
    let sun_dir = normalize(vol.sun_direction.xyz);
    let caustic_enabled = vol.caustics_params.x > 0.5;
    let caustic_intensity = vol.caustics_params.y;

    // Scene surface depth and world position.
    let depth = textureSampleLevel(depth_texture, depth_sampler, in.uv, 0);
    let world_pos = reconstruct_world_pos(in.uv, depth);
    let view_dir = normalize(world_pos - cam_pos);
    let t_surface = distance(world_pos, cam_pos);

    // AABB for the water volume (generous top — the heightfield will gate it).
    let vol_min = vec3f(bmin.x, bmin.y, bmin.z);
    let vol_max = vec3f(bmax.x, max(surface_h, bmin.y + 0.5), bmax.z);

    let hit = ray_aabb_intersect(cam_pos, view_dir, vol_min, vol_max);
    let t_entry_box = hit.x;
    let t_exit_box = hit.y;
    if hit.z < 0.5 { discard; }
    if t_entry_box > t_surface + 0.001 { discard; }

    let t_start = max(t_entry_box, 0.0);
    let t_end = min(t_exit_box, t_surface);
    if t_end <= t_start + 0.001 { discard; }

    let total_dist = t_end - t_start;
    let num_steps = min(64u, max(8u, u32(total_dist * 6.0)));
    let step_size = total_dist / f32(num_steps);

    var transmittance = vec3f(1.0);
    var scatter = vec3f(0.0);
    var water_depth = 0.0;
    var inside_count = 0u;

    for (var i = 0u; i < num_steps; i++) {
        let t = t_start + (f32(i) + 0.5) * step_size;
        let sample_pos = cam_pos + view_dir * t;

        // Heightfield at this sample point — the actual water surface.
        let surf = sample_surface(sample_pos.xz, bmin, bmax, surface_h, wave_amp);

        // Skip if above the surface (ray passes through waves).
        if sample_pos.y > surf {
            continue;
        }

        inside_count++;
        let depth_below = surf - sample_pos.y;
        water_depth += step_size;

        // Beer-Lambert absorption per step.
        let step_abs = extinction * step_size;
        let step_trans = exp(-step_abs);
        transmittance *= step_trans;

        // In-scattering: sun light scattered by suspended particles.
        let cos_theta = dot(-view_dir, sun_dir);
        let phase = henyey_greenstein(cos_theta, 0.6);
        let incoming_light = vec3f(1.0, 0.95, 0.85) * 0.5;
        let contrib = incoming_light * water_color * scatter_density * phase * step_size;
        scatter += contrib * transmittance;
    }

    if inside_count == 0u { discard; }

    // Total Beer-Lambert over the entire water path.
    let total_absorption = exp(-extinction * water_depth);

    // Fresnel factor at the water surface entry.
    let cos_view = abs(dot(view_dir, vec3f(0.0, 1.0, 0.0)));
    let fresnel = mix(0.2, 1.0, pow(1.0 - cos_view, 4.0));

    // Caustic glow on the volume interior.
    var caustic = vec3f(0.0);
    if caustic_enabled {
        let entry_xz = cam_pos.xz + view_dir.xz * t_start;
        let cuv = world_to_sim_uv(entry_xz, bmin, bmax);
        let cval = textureSampleLevel(caustics_tex, shared_samp, cuv, 0.0).rgb;
        caustic = cval * caustic_intensity * 0.25 * (1.0 - total_absorption);
    }

    // Rim glow — scattered light near the surface.
    let rim = rim_strength * exp(-water_depth * 0.3) * 0.3;
    let rim_glow = water_color * rim * (1.0 - total_absorption);

    // Background scene, Beer-Lambert attenuated.
    let bg = textureSampleLevel(scene_color, shared_samp, in.uv, 0.0).rgb;
    let color = bg * total_absorption + scatter + caustic + rim_glow;

    // Alpha fade — smooth at volume boundaries so walls are invisible.
    // Fade top: how far below the entry surface we start.
    let entry_surf = sample_surface(
        (cam_pos.xz + view_dir.xz * t_start), bmin, bmax, surface_h, wave_amp);
    let below_surf_at_entry = entry_surf - (cam_pos.y + view_dir.y * t_start);
    let alpha_top = smoothstep(0.0, 1.5, max(0.0, below_surf_at_entry));

    // Fade bottom: approach to floor.
    let exit_y = cam_pos.y + view_dir.y * t_end;
    let dist_to_floor = exit_y - bmin.y;
    let alpha_bot = smoothstep(0.0, 1.0, max(0.0, dist_to_floor));

    // Fade sides: distance from entry point to the AABB side walls.
    let entry_x = cam_pos.x + view_dir.x * t_start;
    let entry_z = cam_pos.z + view_dir.z * t_start;
    let dx = min(entry_x - bmin.x, bmax.x - entry_x);
    let dz = min(entry_z - bmin.z, bmax.z - entry_z);
    let edge_dist = min(dx, dz);
    let alpha_side = smoothstep(0.0, 2.0, max(0.0, edge_dist));

    // Distance fade — far away water fades to avoid hard cutoffs.
    let alpha_dist = smoothstep(80.0, 10.0, t_start);

    let alpha = clamp(fresnel * alpha_top * alpha_bot * alpha_side * alpha_dist, 0.0, 0.92);
    if alpha < 0.005 { discard; }

    return vec4f(color, alpha);
}
