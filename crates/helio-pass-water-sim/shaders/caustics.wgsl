// Caustics projection — exact port of webgpu-water caustics.vert/frag.wgsl
//
// Projects refracted light rays through the displaced water surface onto the
// pool floor, accumulating caustic intensity with additive blending.
// R channel = caustic intensity (area ratio * causticIntensity)
// G channel = sphere shadow (always 1.0 — no sphere in Helio)

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

@group(0) @binding(0) var<storage, read> water_volumes: array<WaterVolume>;
@group(0) @binding(1) var water_sim: texture_2d<f32>;
@group(0) @binding(2) var water_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) oldPos: vec3f,
    @location(1) newPos: vec3f,
    @location(2) ray:    vec3f,
}

fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
    let tMin = (cubeMin - origin) / ray;
    let tMax = (cubeMax - origin) / ray;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar  = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
}

// Projects a ray from the water surface to the pool floor (exactly as reference)
fn project(origin: vec3f, ray: vec3f, refractedLight: vec3f) -> vec3f {
    let poolHeight = 1.0;
    var point = origin;
    let tcube = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
    point += ray * tcube.y;
    let tplane = (-point.y - 1.0) / refractedLight.y;
    return point + refractedLight * tplane;
}

@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol       = water_volumes[0];
    let IOR_AIR   = 1.0;
    let IOR_WATER = vol.sim_params.x;

    let uv   = position.xy * 0.5 + 0.5;
    let info = textureSampleLevel(water_sim, water_samp, uv, 0.0);

    // Reconstruct normal (scaled 0.5 for stability, matching reference)
    let ba     = info.ba * 0.5;
    let normal = vec3f(ba.x, sqrt(max(0.0, 1.0 - dot(ba, ba))), ba.y);

    let lightDir       = normalize(vol.sun_direction.xyz);
    let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    let ray            = refract(-lightDir, normal, IOR_AIR / IOR_WATER);

    // XY surface grid → XZ sim plane (Y=0 is the undisplaced water surface)
    let pos    = vec3f(position.x, 0.0, position.y);
    let oldPos = project(pos, refractedLight, refractedLight);
    let newPos = project(pos + vec3f(0.0, info.r, 0.0), ray, refractedLight);

    // Project to caustics texture NDC (0.75 scale matches reference)
    let projectedPos = 0.75 * (newPos.xz - newPos.y * refractedLight.xz / refractedLight.y);

    var out: VertexOutput;
    out.position = vec4f(projectedPos.x, -projectedPos.y, 0.0, 1.0);
    out.oldPos   = oldPos;
    out.newPos   = newPos;
    out.ray      = ray;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol       = water_volumes[0];
    let IOR_AIR   = 1.0;
    let IOR_WATER = vol.sim_params.x;

    // Area-ratio caustic intensity (reference technique)
    let oldArea = length(dpdx(in.oldPos)) * length(dpdy(in.oldPos));
    let newArea = length(dpdx(in.newPos)) * length(dpdy(in.newPos));
    var intensity = oldArea / newArea * vol.sim_params.y;

    let lightDir       = normalize(vol.sun_direction.xyz);
    let refractedLight = refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);

    // Rim shadow at pool edges (exact reference formula)
    let poolHeight = 1.0;
    let t = intersectCube(in.newPos, -refractedLight,
                          vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
    let rimShadow = 1.0 / (1.0 + exp(
        -200.0 / (1.0 + 10.0 * (t.y - t.x)) *
        (in.newPos.y - refractedLight.y * t.y - 2.0 / 12.0)
    ));
    intensity *= mix(1.0, rimShadow, vol.shadow_params.x);

    // R = caustic intensity, G = sphere shadow (1.0 = no sphere/fully lit)
    return vec4f(intensity, 1.0, 0.0, 1.0);
}
