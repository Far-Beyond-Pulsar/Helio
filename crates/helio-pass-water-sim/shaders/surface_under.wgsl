// Water surface shader — underwater view (looking up at the surface).
//
// Identical vertex shader to surface_above.wgsl. Fragment shader uses
// flipped normals and underwater color tinting.

struct Camera {
    view:           mat4x4f,
    proj:           mat4x4f,
    view_proj:      mat4x4f,
    inv_view_proj:  mat4x4f,
    position_near:  vec4f,   // xyz=eye world position, w=near plane
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

    let uv = position.xy * 0.5 + 0.5;
    let info = textureSampleLevel(water_sim, water_samp, uv, 0.0);

    var simPos = vec3f(position.x, 0.0, position.y);
    simPos.y = info.r;

    var worldPos = simToWorld(simPos, vol.bounds_min.xyz, vol.bounds_max.xyz);
    worldPos.y = vol.bounds_max.w + info.r * 0.1;

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

fn getSkyColor(ray: vec3f, sunDir: vec3f) -> vec3f {
    let t = max(0.0, ray.y);
    let sky = mix(vec3f(0.7, 0.8, 1.0), vec3f(0.2, 0.4, 0.8), t);
    let spec = pow(max(0.0, dot(ray, normalize(sunDir))), 5000.0);
    return sky + vec3f(spec) * vec3f(10.0, 8.0, 6.0);
}

fn getWallColor(point: vec3f, IOR_WATER: f32, poolHeight: f32, sunDir: vec3f, baseColor: vec3f) -> vec3f {
    var normal = vec3f(0.0, 1.0, 0.0);
    if abs(point.x) > 0.999 {
        normal = vec3f(-point.x, 0.0, 0.0);
    } else if abs(point.z) > 0.999 {
        normal = vec3f(0.0, 0.0, -point.z);
    }

    var scale = 0.5;
    scale = scale / max(0.001, length(point));

    let refractedLight = -refract(-sunDir, vec3f(0.0, 1.0, 0.0), 1.0 / IOR_WATER);
    let diffuse = max(0.0, dot(refractedLight, normal));

    let waterInfo = textureSampleLevel(water_sim, water_samp, point.xz * 0.5 + 0.5, 0.0);
    if point.y < waterInfo.r {
        let causticUV = 0.75 * (point.xz - point.y * refractedLight.xz / refractedLight.y) * 0.5 + 0.5;
        let caustic = textureSampleLevel(caustics_tex, caustics_samp, causticUV, 0.0);
        scale = scale + diffuse * caustic.r * 2.0 * caustic.g;
    } else {
        let t = intersectCube(point, refractedLight, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
        let s = 1.0 / (1.0 + exp(-200.0 / (1.0 + 10.0 * (t.y - t.x)) * (point.y + refractedLight.y * t.y - 2.0 / 12.0)));
        scale = scale + diffuse * s * 0.5;
    }
    return baseColor * scale;
}

fn getSurfaceRayColor(origin: vec3f, ray: vec3f, waterColor: vec3f, IOR_WATER: f32, poolHeight: f32, sunDir: vec3f, wallBase: vec3f) -> vec3f {
    if ray.y < 0.0 {
        let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
        var color = getWallColor(origin + ray * t.y, IOR_WATER, poolHeight, sunDir, wallBase);
        color = color * waterColor;
        return color;
    } else {
        let t = intersectCube(origin, ray, vec3f(-1.0, -poolHeight, -1.0), vec3f(1.0, 2.0, 1.0));
        let hit = origin + ray * t.y;
        if hit.y < 2.0 / 12.0 {
            return getWallColor(hit, IOR_WATER, poolHeight, sunDir, wallBase);
        } else {
            return getSkyColor(ray, sunDir);
        }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol = water_volumes[0];
    let IOR_WATER = vol.sim_params.x;
    let IOR_AIR = 1.0;
    let poolHeight = 1.0;
    let sunDir = normalize(vol.sun_direction.xyz);

    var uv = in.simPos.xz * 0.5 + 0.5;
    var info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    for (var i = 0; i < 5; i = i + 1) {
        uv = uv + info.ba * 0.005;
        info = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    }

    let ba = vec2f(info.b, info.a);
    // Flip normal for underwater view
    let normal = -vec3f(info.b, sqrt(max(0.0, 1.0 - dot(ba, ba))), info.a);

    let bmin = vol.bounds_min.xyz;
    let bmax = vol.bounds_max.xyz;
    let eyeSim = (camera.position_near.xyz - bmin) / (bmax - bmin) * 2.0 - 1.0;

    let incomingRay = normalize(in.simPos - eyeSim);
    let reflectedRay = reflect(incomingRay, normal);
    let refractedRay = refract(incomingRay, normal, IOR_WATER / IOR_AIR);
    let fresnel = mix(vol.sim_params.z, 1.0, pow(1.0 - max(0.0, dot(normal, -incomingRay)), 3.0));

    let UNDERwaterColor = vec3f(0.4, 0.9, 1.0);
    let wallBase = vol.water_color.rgb;

    let reflectedColor = getSurfaceRayColor(in.simPos, reflectedRay, UNDERwaterColor, IOR_WATER, poolHeight, sunDir, wallBase);
    let refractedColor = getSurfaceRayColor(in.simPos, refractedRay, vec3f(1.0, 1.0, 1.0), IOR_WATER, poolHeight, sunDir, wallBase) * vec3f(0.8, 1.0, 1.1);

    let finalColor = mix(reflectedColor, refractedColor, (1.0 - fresnel) * length(refractedRay));
    return vec4f(finalColor, 1.0);
}
