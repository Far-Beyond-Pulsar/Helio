// Water surface — underwater view.
// Exact port of webgpu-water surface.vert + surface-under.frag.wgsl.
//
// Identical vertex shader to surface_above. Fragment shader flips the normal
// and swaps IOR ratio for the underwater perspective.

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
    bounds_min:            vec4f,  // xyz=min, w=unused
    bounds_max:            vec4f,  // xyz=max, w=surface_height
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,
    extinction:            vec4f,
    reflection_refraction: vec4f,
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,  // x=ior, y=causticIntensity, z=fresnelMin, w=density
    shadow_params:         vec4f,  // x=rim, y=hitbox, z=ao
    sun_direction:         vec4f,
    pad0: vec4f, pad1: vec4f, pad2: vec4f, pad3: vec4f,
}

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var<storage, read> water_volumes: array<WaterVolume>;
@group(0) @binding(2) var water_sim:     texture_2d<f32>;
@group(0) @binding(3) var water_samp:    sampler;
@group(0) @binding(4) var caustics_tex:  texture_2d<f32>;
@group(0) @binding(5) var caustics_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) worldPos: vec3f,  // sim-space position
}

// ── Coordinate helpers (identical to surface_above) ───────────────────────

fn simToWorld(sim: vec3f, bmin: vec3f, bmax: vec3f, surface_h: f32) -> vec3f {
    let pool_depth = surface_h - bmin.y;
    return vec3f(
        bmin.x + (sim.x * 0.5 + 0.5) * (bmax.x - bmin.x),
        surface_h + sim.y * pool_depth,
        bmin.z + (sim.z * 0.5 + 0.5) * (bmax.z - bmin.z),
    );
}

fn worldToSim(world: vec3f, bmin: vec3f, bmax: vec3f, surface_h: f32) -> vec3f {
    let pool_depth = surface_h - bmin.y;
    return vec3f(
        (world.x - bmin.x) / (bmax.x - bmin.x) * 2.0 - 1.0,
        (world.y - surface_h) / pool_depth,
        (world.z - bmin.z) / (bmax.z - bmin.z) * 2.0 - 1.0,
    );
}

// ── Vertex shader (identical to surface_above) ────────────────────────────
@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol       = water_volumes[0];
    let bmin      = vol.bounds_min.xyz;
    let bmax      = vol.bounds_max.xyz;
    let surface_h = vol.bounds_max.w;

    let uv   = position.xy * 0.5 + 0.5;
    let info = textureSampleLevel(water_sim, water_samp, uv, 0.0);

    var simPos = vec3f(position.x, info.r, position.y);
    let worldPos = simToWorld(simPos, bmin, bmax, surface_h);

    var out: VertexOutput;
    out.position = camera.view_proj * vec4f(worldPos, 1.0);
    out.worldPos = simPos;
    return out;
}

// ── Fragment helpers ──────────────────────────────────────────────────────

fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
    let tMin  = (cubeMin - origin) / ray;
    let tMax  = (cubeMax - origin) / ray;
    let t1    = min(tMin, tMax);
    let t2    = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar  = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
}

fn tile_color(uv: vec2f) -> vec3f {
    let t     = fract(uv * 8.0);
    let grout = max(step(0.93, t.x), step(0.93, t.y));
    return mix(vec3f(0.82, 0.84, 0.86), vec3f(0.58, 0.60, 0.62), grout);
}

fn getWallColor(point: vec3f, IOR_WATER: f32, poolHeight: f32, lightDir: vec3f) -> vec3f {
    var wallColor: vec3f;
    var normal = vec3f(0.0, 1.0, 0.0);
    if abs(point.x) > 0.999 {
        wallColor = tile_color(point.yz * 0.5 + vec2f(1.0, 0.5));
        normal    = vec3f(-point.x, 0.0, 0.0);
    } else if abs(point.z) > 0.999 {
        wallColor = tile_color(point.yx * 0.5 + vec2f(1.0, 0.5));
        normal    = vec3f(0.0, 0.0, -point.z);
    } else {
        wallColor = tile_color(point.xz * 0.5 + 0.5);
    }

    var scale          = 0.5 / max(0.001, length(point));
    let refractedLight = -refract(-lightDir, vec3f(0.0, 1.0, 0.0), 1.0 / IOR_WATER);
    let diffuse        = max(0.0, dot(refractedLight, normal));

    let info = textureSampleLevel(water_sim, water_samp, point.xz * 0.5 + 0.5, 0.0);
    if point.y < info.r {
        let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
        let caustic   = textureSampleLevel(caustics_tex, caustics_samp, causticUV, 0.0);
        scale += diffuse * caustic.r * 2.0 * caustic.g;
    } else {
        let t = intersectCube(point, refractedLight,
                              vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
        let s = 1.0 / (1.0 + exp(
            -200.0 / (1.0 + 10.0 * (t.y - t.x)) *
            (point.y + refractedLight.y * t.y - 2.0 / 12.0)
        ));
        scale += diffuse * s * 0.5;
    }
    return wallColor * scale;
}

fn getSurfaceRayColor(origin: vec3f, ray: vec3f, waterColor: vec3f, IOR_WATER: f32,
                      poolHeight: f32, lightDir: vec3f, pool_rim_sim_y: f32) -> vec3f {
    if ray.y < 0.0 {
        let t     = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
        let color = getWallColor(origin + ray * t.y, IOR_WATER, poolHeight, lightDir);
        return color * waterColor;
    } else {
        let t   = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
        let hit = origin + ray * t.y;
        if hit.y < pool_rim_sim_y {
            return getWallColor(hit, IOR_WATER, poolHeight, lightDir);
        } else {
            let sunDir = normalize(lightDir);
            let spec   = pow(max(0.0, dot(sunDir, ray)), 5000.0);
            let sky    = mix(vec3f(0.7, 0.8, 1.0), vec3f(0.2, 0.4, 0.8), max(0.0, ray.y));
            return sky + vec3f(spec) * vec3f(10.0, 8.0, 6.0);
        }
    }
}

// ── Fragment shader (exact port of surface-under.frag.wgsl) ──────────────
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol        = water_volumes[0];
    let IOR_AIR    = 1.0;
    let IOR_WATER  = vol.sim_params.x;
    let poolHeight = 1.0;
    let lightDir   = normalize(vol.sun_direction.xyz);

    let pool_depth     = vol.bounds_max.w - vol.bounds_min.y;
    let pool_rim_sim_y = (vol.bounds_max.y - vol.bounds_max.w) / pool_depth;

    var uv   = in.worldPos.xz * 0.5 + 0.5;
    var info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    for (var i = 0; i < 5; i++) {
        uv  += info.ba * 0.005;
        info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    }

    let ba = vec2f(info.b, info.a);
    // UNDERWATER: flip normal (exact reference: normal = -normal)
    let normal = -vec3f(info.b, sqrt(max(0.0, 1.0 - dot(ba, ba))), info.a);

    let surface_h = vol.bounds_max.w;
    let eyeSim    = worldToSim(camera.position_near.xyz, vol.bounds_min.xyz, vol.bounds_max.xyz, surface_h);

    let incomingRay  = normalize(in.worldPos - eyeSim);
    let reflectedRay = reflect(incomingRay, normal);
    // Underwater: IOR ratio flipped (water → air)
    let refractedRay = refract(incomingRay, normal, IOR_WATER / IOR_AIR);
    let fresnel      = mix(vol.sim_params.z, 1.0, pow(1.0 - max(0.0, dot(normal, -incomingRay)), 3.0));

    let UNDERwaterColor = vec3f(0.4, 0.9, 1.0);

    let reflectedColor = getSurfaceRayColor(in.worldPos, reflectedRay, UNDERwaterColor,
                                            IOR_WATER, poolHeight, lightDir, pool_rim_sim_y);
    // Refracted color: reference uses vec3f(1.0) tint * vec3f(0.8, 1.0, 1.1)
    let refractedColor = getSurfaceRayColor(in.worldPos, refractedRay, vec3f(1.0),
                                            IOR_WATER, poolHeight, lightDir, pool_rim_sim_y)
                       * vec3f(0.8, 1.0, 1.1);

    // Exact reference blend for underwater view
    let finalColor = mix(reflectedColor, refractedColor, (1.0 - fresnel) * length(refractedRay));
    return vec4f(finalColor, 1.0);
}
