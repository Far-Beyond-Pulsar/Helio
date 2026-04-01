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
    pad0: vec4f, pad1: vec4f, pad2: vec4f, pad3: vec4f,
}

@group(0) @binding(0) var<uniform>       camera:      Camera;
@group(0) @binding(1) var<storage, read> volumes:     array<WaterVolume>;
@group(0) @binding(2) var water_sim:     texture_2d<f32>;
@group(0) @binding(3) var water_samp:    sampler;
@group(0) @binding(4) var caustics_tex:  texture_2d<f32>;
@group(0) @binding(5) var shared_samp:   sampler;
@group(0) @binding(6) var scene_color:   texture_2d<f32>;
@group(0) @binding(7) var<uniform>       viewport:    vec4f;
@group(0) @binding(8) var depth_tex:     texture_depth_2d;

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
    // Manual depth test (same as surface_above).
    let depth_dims  = vec2f(textureDimensions(depth_tex));
    let depth_coord = vec2<i32>(in.position.xy * depth_dims * viewport.zw);
    let scene_depth = textureLoad(depth_tex, depth_coord, 0);
    if in.position.z >= scene_depth { discard; }

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

    // ── Reflected ray — sample real underwater scene ──────────────────────────
    let reflected_ray = reflect(incoming, normal);
    let screen_uv     = in.position.xy * viewport.zw;
    let distort_str   = vol.reflection_refraction.y;
    let reflect_uv    = clamp(screen_uv + normal.xz * distort_str,
                              vec2f(0.001), vec2f(0.999));
    let below_color   = textureSampleLevel(scene_color, shared_samp, reflect_uv, 0.0).rgb
                        * vec3f(0.4, 0.9, 1.0);  // UNDERwaterColor tint (reference)

    // ── Blend — exact reference formula for underwater view ───────────────────
    let final_color = mix(below_color, above_color,
                          (1.0 - fresnel) * length(refracted_ray));
    return vec4f(final_color, 1.0);
}
