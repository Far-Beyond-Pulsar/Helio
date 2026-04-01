// Water surface — above-water view.
//
// Vertex:   Displaces the 128×128 grid by the sim heightfield (identical to reference).
// Fragment: Screen-space refraction of the real opaque scene behind/below the surface,
//           Fresnel-blended with a procedural sky reflection + sun specular.
//           No pool geometry — integrates cleanly with any scene.
//
// Bindings
//   0  camera           uniform
//   1  water_volumes    storage read
//   2  water_sim        texture_2d<f32>  (RGBA16F: R=height B/A=normal.xz)
//   3  water_samp       sampler          (linear, repeat  — for sim)
//   4  caustics_tex     texture_2d<f32>  (unused here, kept for layout compatibility)
//   5  shared_samp      sampler          (linear, clamp   — for scene color)
//   6  scene_color      texture_2d<f32>  (opaque scene rendered before this pass)
//   7  viewport         uniform vec4f    (xy=px size, zw=1/size)

struct Camera {
    view:           mat4x4f,
    proj:           mat4x4f,
    view_proj:      mat4x4f,
    inv_view_proj:  mat4x4f,
    position_near:  vec4f,  // xyz=eye world pos, w=near
    forward_far:    vec4f,
    jitter_frame:   vec4f,
    prev_view_proj: mat4x4f,
}

struct WaterVolume {
    bounds_min:            vec4f,  // xyz=min corner
    bounds_max:            vec4f,  // xyz=max corner, w=surface_height
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,  // xyz=tint applied to refracted scene (Beer–Lambert)
    extinction:            vec4f,
    reflection_refraction: vec4f,  // x=reflection_str, y=refraction_distortion_strength
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,  // x=ior, y=causticIntensity, z=fresnelMin, w=density
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

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) simPos: vec3f,
}

// ── Coordinate helpers (identical to previous port) ───────────────────────────

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

// ── Vertex shader (identical to reference surface.vert) ───────────────────────

@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol      = volumes[0];
    let bmin     = vol.bounds_min.xyz;
    let bmax     = vol.bounds_max.xyz;
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

// ── Procedural sky + sun for reflected rays ───────────────────────────────────
// Matches the reference's sky appearance without requiring a cubemap.

fn sky_color(ray: vec3f, light_dir: vec3f) -> vec3f {
    let up     = clamp(ray.y, 0.0, 1.0);
    let horizon = vec3f(0.80, 0.90, 1.00);
    let zenith  = vec3f(0.10, 0.30, 0.80);
    let sky     = mix(horizon, zenith, up * up);
    // Sun disc — same exponent and colour as the reference (5000, 10/8/6)
    let spec    = pow(max(0.0, dot(normalize(light_dir), ray)), 5000.0);
    return sky + vec3f(spec) * vec3f(10.0, 8.0, 6.0);
}

// ── Fragment shader ───────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {


    let vol       = volumes[0];
    let IOR_AIR   = 1.0;
    let IOR_WATER = vol.sim_params.x;
    let light_dir = normalize(vol.sun_direction.xyz);

    // ── Normal from sim — 5-iteration UV refinement (identical to reference) ──
    var uv   = in.simPos.xz * 0.5 + 0.5;
    var info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    for (var i = 0; i < 5; i++) {
        uv  += info.ba * 0.005;
        info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    }
    let ba     = vec2f(info.b, info.a);
    let normal = vec3f(info.b, sqrt(max(0.0, 1.0 - dot(ba, ba))), info.a);

    // ── Camera ray in sim space (identical to reference) ──────────────────────
    let surface_h = vol.bounds_max.w;
    let eye_sim   = worldToSim(camera.position_near.xyz,
                                vol.bounds_min.xyz, vol.bounds_max.xyz, surface_h);
    let incoming  = normalize(in.simPos - eye_sim);

    // ── Fresnel (identical to reference formula) ──────────────────────────────
    let fresnel = mix(vol.sim_params.z, 1.0,
                      pow(1.0 - max(0.0, dot(normal, -incoming)), 3.0));

    // ── Screen-space refraction ───────────────────────────────────────────────
    // Perturb screen UV by the water normal's horizontal components to
    // distort the real scene behind/below the surface.
    let screen_uv   = in.position.xy * viewport.zw;
    let refract_str = vol.reflection_refraction.y;
    let refract_uv  = clamp(screen_uv + normal.xz * refract_str,
                            vec2f(0.001), vec2f(0.999));
    var refracted   = textureSampleLevel(scene_color, shared_samp, refract_uv, 0.0).rgb;

    // Beer–Lambert water colour absorption
    refracted *= vol.water_color.rgb;

    // ── Reflection: real sky + sun specular ───────────────────────────────────
    let reflected_ray = reflect(incoming, normal);
    let reflected     = sky_color(reflected_ray, light_dir);

    // ── Fresnel blend ─────────────────────────────────────────────────────────
    return vec4f(mix(refracted, reflected, fresnel), 1.0);
}

