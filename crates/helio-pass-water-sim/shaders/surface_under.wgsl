// Water surface — underwater view.
//
// Vertex:   Identical to surface_above (displaces grid by heightfield).
// Fragment: Looking UP through the water surface from below.
//           Refracted ray escaping upward → procedural sky (Snell's window).
//           Reflected ray bouncing back down → real scene via screen-space sample.
//           No pool geometry — works with any scene.

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
    bounds_max:            vec4f,  // w=surface_height
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,
    extinction:            vec4f,
    reflection_refraction: vec4f,  // y=distortion_strength
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,  // x=ior, z=fresnelMin
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,  // x=enable, y=max_steps, z=step_size, w=thickness
    pad1: vec4f, pad2: vec4f, pad3: vec4f,
}

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(1) var<storage, read> volumes:     array<WaterVolume>;
@group(0) @binding(2) var water_sim:     texture_2d<f32>;
@group(0) @binding(3) var water_samp:    sampler;
@group(0) @binding(4) var caustics_tex:  texture_2d<f32>;
@group(0) @binding(5) var shared_samp:   sampler;
@group(0) @binding(6) var scene_color:   texture_2d<f32>;
@group(0) @binding(7) var<uniform>       viewport:    vec4f;
@group(0) @binding(8) var depth_texture:   texture_2d<f32>;
@group(0) @binding(9) var depth_sampler:   sampler;
@group(0) @binding(10) var gbuffer_normal: texture_2d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) simPos: vec3f,
}

fn simToWorld(sim: vec3f, bmin: vec3f, bmax: vec3f, surface_h: f32) -> vec3f {
    let d = surface_h - bmin.y;
    return vec3f(
        bmin.x + (sim.x * 0.5 + 0.5) * (bmax.x - bmin.x),
        surface_h + sim.y * d,
        bmin.z + (sim.z * 0.5 + 0.5) * (bmax.z - bmin.z),
    );
}

fn worldToSim(world: vec3f, bmin: vec3f, bmax: vec3f, surface_h: f32) -> vec3f {
    let d = surface_h - bmin.y;
    return vec3f(
        (world.x - bmin.x) / (bmax.x - bmin.x) * 2.0 - 1.0,
        (world.y - surface_h) / d,
        (world.z - bmin.z) / (bmax.z - bmin.z) * 2.0 - 1.0,
    );
}

// ── SSR Helper Functions ──────────────────────────────────────────────────────

// Reconstruct world position from screen UV + depth
fn reconstruct_world_pos(uv: vec2f, depth: f32) -> vec3f {
    let ndc_xy = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let world_h = camera.inv_view_proj * vec4f(ndc_xy, depth, 1.0);
    return world_h.xyz / world_h.w;
}

// Screen-space ray march along reflected direction
fn trace_ssr(
    ray_origin: vec3f,      // World space
    ray_dir: vec3f,         // World space, normalized
    screen_uv: vec2f,       // Starting UV [0,1]
    max_steps: u32,         // Quality control (16-64)
    step_size: f32,         // World-space step size
    thickness: f32,         // Depth comparison threshold
) -> vec3f {
    var hit_color = vec3f(0.0);
    var hit = false;

    // March in world space
    for (var i = 0u; i < max_steps && !hit; i++) {
        let t = f32(i) * step_size;
        let sample_world = ray_origin + ray_dir * t;

        // Project to screen space
        let sample_clip = camera.view_proj * vec4f(sample_world, 1.0);
        let sample_ndc = sample_clip.xyz / sample_clip.w;
        let sample_uv = vec2f(
            sample_ndc.x * 0.5 + 0.5,
            1.0 - (sample_ndc.y * 0.5 + 0.5)
        );

        // Out of screen bounds - miss
        if any(sample_uv < vec2f(0.0)) || any(sample_uv > vec2f(1.0)) {
            break;
        }

        // Sample scene depth at ray position
        let scene_depth = textureSampleLevel(depth_texture, depth_sampler, sample_uv, 0.0).r;

        // Reconstruct world position of scene geometry at this UV
        let scene_world = reconstruct_world_pos(sample_uv, scene_depth);
        let scene_dist = length(scene_world - ray_origin);
        let ray_dist = t;

        // Check if ray intersects geometry (within thickness tolerance)
        if abs(scene_dist - ray_dist) < thickness {
            // Hit! Sample scene color at this position
            hit_color = textureSampleLevel(scene_color, shared_samp, sample_uv, 0.0).rgb;
            hit = true;
        }
    }

    return hit_color;
}

@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol       = volumes[0];
    let bmin      = vol.bounds_min.xyz;
    let bmax      = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;

    let uv   = position.xy * 0.5 + 0.5;
    let info = textureSampleLevel(water_sim, water_samp, uv, 0.0);

    var simPos   = vec3f(position.x, info.r, position.y);
    let worldPos = simToWorld(simPos, bmin, bmax, surface_h);

    var out: VertexOutput;
    out.position = camera.view_proj * vec4f(worldPos, 1.0);
    out.simPos   = simPos;
    return out;
}

fn sky_color(ray: vec3f, light_dir: vec3f) -> vec3f {
    let up      = clamp(ray.y, 0.0, 1.0);
    let horizon = vec3f(0.80, 0.90, 1.00);
    let zenith  = vec3f(0.10, 0.30, 0.80);
    let sky     = mix(horizon, zenith, up * up);
    let spec    = pow(max(0.0, dot(normalize(light_dir), ray)), 5000.0);
    return sky + vec3f(spec) * vec3f(10.0, 8.0, 6.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {


    let vol       = volumes[0];
    let IOR_AIR   = 1.0;
    let IOR_WATER = vol.sim_params.x;
    let light_dir = normalize(vol.sun_direction.xyz);

    // ── Normal — 5-iteration refinement, then FLIP for underwater (reference) ─
    var uv   = in.simPos.xz * 0.5 + 0.5;
    var info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    for (var i = 0; i < 5; i++) {
        uv  += info.ba * 0.005;
        info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    }
    let ba     = vec2f(info.b, info.a);
    let normal = -vec3f(info.b, sqrt(max(0.0, 1.0 - dot(ba, ba))), info.a);

    // ── Camera ray in sim space ───────────────────────────────────────────────
    let surface_h = vol.bounds_max.w;
    let eye_sim   = worldToSim(camera.position_near.xyz,
                                vol.bounds_min.xyz, vol.bounds_max.xyz, surface_h);
    let incoming  = normalize(in.simPos - eye_sim);

    // ── Fresnel (same formula as above-water pass) ────────────────────────────
    let fresnel = mix(vol.sim_params.z, 1.0,
                      pow(1.0 - max(0.0, dot(normal, -incoming)), 3.0));

    // ── Refracted ray (water → air) — Snell's window ─────────────────────────
    let refracted_ray = refract(incoming, normal, IOR_WATER / IOR_AIR);
    var above_color: vec3f;
    if length(refracted_ray) > 0.5 {
        // Ray escapes upward through the surface — show sky
        above_color = sky_color(refracted_ray, light_dir);
    }
    // else: total internal reflection — above_color stays black (only reflected)

    // ── Reflected ray — SSR for real underwater scene ─────────────────────────
    let reflected_ray = reflect(incoming, normal);
    let screen_uv     = in.position.xy * viewport.zw;
    var below_color   = vec3f(0.0);

    // Check if SSR is enabled for this water volume
    let ssr_enabled = vol.ssr_params.x > 0.5;

    if ssr_enabled {
        // Convert sim-space to world-space for SSR
        let surface_h = vol.bounds_max.w;
        let world_pos = simToWorld(in.simPos, vol.bounds_min.xyz, vol.bounds_max.xyz, surface_h);

        // Extract SSR parameters from volume
        let max_steps = u32(vol.ssr_params.y);
        let step_size = vol.ssr_params.z;
        let thickness = vol.ssr_params.w;

        // Trace SSR - returns (0,0,0) if no hit
        below_color = trace_ssr(world_pos, reflected_ray, screen_uv, max_steps, step_size, thickness);
    }

    // Fallback to screen-space distortion if SSR disabled or missed
    if dot(below_color, below_color) < 0.001 {
        let distort_str = vol.reflection_refraction.y;
        let reflect_uv  = clamp(screen_uv + normal.xz * distort_str,
                                vec2f(0.001), vec2f(0.999));
        below_color = textureSampleLevel(scene_color, shared_samp, reflect_uv, 0.0).rgb;
    }

    // Apply underwater color tint
    below_color *= vec3f(0.4, 0.9, 1.0);  // UNDERwaterColor tint (reference)

    // ── Blend — exact reference formula for underwater view ───────────────────
    let final_color = mix(below_color, above_color,
                          (1.0 - fresnel) * length(refracted_ray));
    return vec4f(final_color, 1.0);
}
