// Pool walls and floor shader.
//
// Renders the inside of the pool box in world space. The vertex shader
// maps unit-cube positions to pool geometry using the reference depth
// transform. The fragment shader computes diffuse lighting + caustics.

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
    bounds_max:            vec4f,
    wave_params:           vec4f,
    wave_direction:        vec4f,
    water_color:           vec4f,   // rgb=pool wall/floor color, a=foam threshold
    extinction:            vec4f,
    reflection_refraction: vec4f,
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,   // x=ior, y=causticIntensity, z=fresnelMin, w=density
    shadow_params:         vec4f,   // x=rim shadow factor
    sun_direction:         vec4f,
    pad0: vec4f, pad1: vec4f, pad2: vec4f, pad3: vec4f,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> water_volumes: array<WaterVolume>;
@group(0) @binding(2) var water_sim: texture_2d<f32>;
@group(0) @binding(3) var water_samp: sampler;
@group(0) @binding(4) var caustics_tex: texture_2d<f32>;
@group(0) @binding(5) var caustics_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) simPos: vec3f,
}

fn simToWorld(sim: vec3f, bmin: vec3f, bmax: vec3f) -> vec3f {
    return bmin + (sim * 0.5 + 0.5) * (bmax - bmin);
}

@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol = water_volumes[0];

    // Map unit cube Y to pool depth: floor at y=-1, rim at y=2/12
    var simPos = position;
    simPos.y = (1.0 - position.y) * (7.0 / 12.0) - 1.0;

    let worldPos = simToWorld(simPos, vol.bounds_min.xyz, vol.bounds_max.xyz);

    var out: VertexOutput;
    out.position = camera.view_proj * vec4f(worldPos, 1.0);
    out.simPos = simPos;
    return out;
}

// === Fragment helpers ===

fn intersectCube(origin: vec3f, ray: vec3f, cubeMin: vec3f, cubeMax: vec3f) -> vec2f {
    let tMin = (cubeMin - origin) / ray;
    let tMax = (cubeMax - origin) / ray;
    let t1 = min(tMin, tMax);
    let t2 = max(tMin, tMax);
    let tNear = max(max(t1.x, t1.y), t1.z);
    let tFar = min(min(t2.x, t2.y), t2.z);
    return vec2f(tNear, tFar);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol = water_volumes[0];
    let IOR_AIR = 1.0;
    let IOR_WATER = vol.sim_params.x;
    let poolHeight = 1.0;

    let point = in.simPos;
    let wallColor = vol.water_color.rgb * 0.8;

    var normal = vec3f(0.0, 1.0, 0.0);
    if abs(point.x) > 0.999 {
        normal = vec3f(-point.x, 0.0, 0.0);
    } else if abs(point.z) > 0.999 {
        normal = vec3f(0.0, 0.0, -point.z);
    }

    var scale = 0.5;
    scale = scale / max(0.001, length(point));

    let sunDir = normalize(vol.sun_direction.xyz);
    let refractedLight = -refract(-sunDir, vec3f(0.0, 1.0, 0.0), IOR_AIR / IOR_WATER);
    let diffuse = max(0.0, dot(refractedLight, normal));

    let waterInfo = textureSampleLevel(water_sim, water_samp, point.xz * 0.5 + 0.5, 0.0);

    if point.y < waterInfo.r {
        let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
        let caustic = textureSampleLevel(caustics_tex, caustics_samp, causticUV, 0.0);
        var intensity = caustic.r;
        var sphereShadow = caustic.g;
        // Fallback when shadow is disabled and caustics are weak
        if vol.shadow_params.x < 0.5 && intensity < 0.001 {
            intensity = 0.2;
            sphereShadow = 1.0;
        }
        scale = scale + diffuse * intensity * 2.0 * sphereShadow;
    } else {
        let t = intersectCube(point, refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
        let s = 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
        scale = scale + diffuse * mix(1.0, s, vol.shadow_params.x) * 0.5;
    }

    var finalColor = wallColor * scale;
    // Tint submerged areas with water color
    if point.y < waterInfo.r {
        finalColor = finalColor * vec3f(0.4, 0.9, 1.0) * 1.2;
    }
    return vec4f(finalColor, 1.0);
}
