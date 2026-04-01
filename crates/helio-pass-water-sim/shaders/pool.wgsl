// Pool walls and floor — exact port of webgpu-water pool.vert/frag.wgsl
//
// The pool geometry is a unit cube. The vertex shader applies the same Y
// transform as the reference to map cube vertices to pool depth. All
// fragment-stage lighting is computed in sim space, matching the reference
// exactly. A procedural tile replaces the reference's watertiles.jpg texture.

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
    @location(0) simPos: vec3f,
}

// ── Coordinate helpers ─────────────────────────────────────────────────────
//
// In sim space:  pool floor = Y -1,  water surface = Y 0,
//                pool rim   = Y pool_rim_sim_y (computed from world bounds)
// World ↔ sim mapping anchored at surface_height = sim Y 0.

fn simToWorld(sim: vec3f, bmin: vec3f, bmax: vec3f, surface_h: f32) -> vec3f {
    let pool_depth = surface_h - bmin.y;
    return vec3f(
        bmin.x + (sim.x * 0.5 + 0.5) * (bmax.x - bmin.x),
        surface_h + sim.y * pool_depth,
        bmin.z + (sim.z * 0.5 + 0.5) * (bmax.z - bmin.z),
    );
}

// ── Procedural pool tile (replaces watertiles.jpg) ─────────────────────────
fn tile_color(uv: vec2f) -> vec3f {
    let t     = fract(uv * 8.0);
    let grout = max(step(0.93, t.x), step(0.93, t.y));
    return mix(vec3f(0.82, 0.84, 0.86), vec3f(0.58, 0.60, 0.62), grout);
}

// ── Ray-box intersection (exact reference) ─────────────────────────────────
fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
    let tMin  = (cubeMin - origin) / ray;
    let tMax  = (cubeMax - origin) / ray;
    let t1    = min(tMin, tMax);
    let t2    = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar  = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
}

// ── Vertex shader ──────────────────────────────────────────────────────────
@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol        = water_volumes[0];
    let surface_h  = vol.bounds_max.w;
    let pool_depth = surface_h - vol.bounds_min.y;
    let pool_above = vol.bounds_max.y - surface_h;

    // Generalised reference Y transform:
    //   mesh Y=-1 → sim rim (pool_above/pool_depth)
    //   mesh Y=+1 → sim -1 (floor)
    // Matches (1-y)*(7/12)-1 when pool_above/pool_depth = 2/12.
    let pool_rim_sim_y = pool_above / pool_depth;
    var simPos = position;
    simPos.y = pool_rim_sim_y + (position.y + 1.0) * 0.5 * (-1.0 - pool_rim_sim_y);

    let worldPos = simToWorld(simPos, vol.bounds_min.xyz, vol.bounds_max.xyz, surface_h);

    var out: VertexOutput;
    out.position = camera.view_proj * vec4f(worldPos, 1.0);
    out.simPos   = simPos;
    return out;
}

// ── Fragment shader ────────────────────────────────────────────────────────
@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol       = water_volumes[0];
    let IOR_AIR   = 1.0;
    let IOR_WATER = vol.sim_params.x;
    let poolHeight = 1.0;
    let point = in.simPos;

    // Tile colour — mirrors reference tileTexture sampling exactly
    var wallColor: vec3f;
    if abs(point.x) > 0.999 {
        wallColor = tile_color(point.yz * 0.5 + vec2f(1.0, 0.5));
    } else if abs(point.z) > 0.999 {
        wallColor = tile_color(point.yx * 0.5 + vec2f(1.0, 0.5));
    } else {
        wallColor = tile_color(point.xz * 0.5 + 0.5);
    }

    // Surface normal per face
    var normal = vec3f(0.0, 1.0, 0.0);
    if abs(point.x) > 0.999 { normal = vec3f(-point.x, 0.0, 0.0); }
    else if abs(point.z) > 0.999 { normal = vec3f(0.0, 0.0, -point.z); }

    // Ambient occlusion (reference: scale = 0.5 / length(point))
    var scale = 0.5 / max(0.001, length(point));

    // Refracted sunlight direction (Snell's law, identical to reference)
    let lightDir       = normalize(vol.sun_direction.xyz);
    let refractedLight = -refract(-lightDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    let diffuse        = max(0.0, dot(refractedLight, normal));

    let waterInfo = textureSampleLevel(water_sim, water_samp, point.xz * 0.5 + 0.5, 0.0);

    if point.y < waterInfo.r {
        // Underwater: caustic lighting (exact reference)
        let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
        let caustic   = textureSampleLevel(caustics_tex, caustics_samp, causticUV, 0.0);

        var intensity    = caustic.r;
        var sphereShadow = caustic.g;  // G=1 since we have no sphere

        // Fill black void outside caustic mesh when rim shadow is off
        if vol.shadow_params.x < 0.5 && intensity < 0.001 {
            intensity    = 0.2;
            sphereShadow = 1.0;
        }
        scale += diffuse * intensity * 2.0 * sphereShadow;
    } else {
        // Above water: rim shadow (exact reference)
        let t = intersectCube(point, refractedLight,
                              vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
        let shadowFactor = 1.0 / (1.0 + exp(
            -200.0 / (1.0 + 10.0 * (t.y - t.x)) *
            (point.y + refractedLight.y * t.y - 2.0 / 12.0)
        ));
        scale += diffuse * mix(1.0, shadowFactor, vol.shadow_params.x) * 0.5;
    }

    var finalColor = wallColor * scale;

    // Underwater tint (exact reference UNDERwaterColor * 1.2)
    if point.y < waterInfo.r {
        finalColor *= vec3f(0.4, 0.9, 1.0) * 1.2;
    }

    return vec4f(finalColor, 1.0);
}
