struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    view_proj_inv: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
};

struct Sky {
    sun_direction: vec3<f32>, sun_intensity: f32,
    rayleigh_scatter: vec3<f32>, rayleigh_h_scale: f32,
    mie_scatter: f32, mie_h_scale: f32, mie_g: f32, sun_disk_cos: f32,
    earth_radius: f32, atm_radius: f32, exposure: f32, clouds_enabled: u32,
    cloud_coverage: f32, cloud_density: f32, cloud_base: f32, cloud_top: f32,
    cloud_wind_x: f32, cloud_wind_z: f32, cloud_speed: f32, time_sky: f32,
    skylight_intensity: f32, cloud_mode: u32, cloud_quality: u32, cloud_resolution: u32,
};

@group(0) @binding(0) var<storage, read> cameras: array<Camera>;
@group(0) @binding(1) var<uniform> sky: Sky;
@group(0) @binding(2) var output: texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var output_data: texture_storage_2d<rgba16float, write>;

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.3183099 + 0.1);
    q *= 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p); let f = fract(p); let u = f * f * (3.0 - 2.0 * f);
    let x0 = mix(hash3(i), hash3(i + vec3<f32>(1, 0, 0)), u.x);
    let x1 = mix(hash3(i + vec3<f32>(0, 1, 0)), hash3(i + vec3<f32>(1, 1, 0)), u.x);
    let x2 = mix(hash3(i + vec3<f32>(0, 0, 1)), hash3(i + vec3<f32>(1, 0, 1)), u.x);
    let x3 = mix(hash3(i + vec3<f32>(0, 1, 1)), hash3(i + vec3<f32>(1, 1, 1)), u.x);
    return mix(mix(x0, x1, u.y), mix(x2, x3, u.y), u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var q = p; var value = 0.0; var amplitude = 0.5;
    for (var i = 0u; i < 4u; i++) {
        value += noise3(q) * amplitude;
        q = q * 2.02 + vec3<f32>(17.1, 31.7, 11.3);
        amplitude *= 0.5;
    }
    return value;
}

fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let k = 0.07957747;
    return k * (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cos_theta, 1.5);
}

fn cloud_gradient(h: f32, cloud_type: f32) -> f32 {
    // Schneider/Horizon cloud profiles: stratus, stratocumulus and cumulus
    // are continuously blended from the weather field rather than using one
    // identical vertical cutoff for every column.
    let stratus = vec4<f32>(0.02, 0.05, 0.09, 0.11);
    let stratocumulus = vec4<f32>(0.02, 0.20, 0.48, 0.625);
    let cumulus = vec4<f32>(0.01, 0.0625, 0.78, 1.0);
    let s = 1.0 - clamp(cloud_type * 2.0, 0.0, 1.0);
    let sc = 1.0 - abs(cloud_type - 0.5) * 2.0;
    let c = clamp(cloud_type - 0.5, 0.0, 1.0) * 2.0;
    let g = stratus * s + stratocumulus * sc + cumulus * c;
    return smoothstep(g.x, g.y, h) - smoothstep(g.z, g.w, h);
}

fn cloud_density(p: vec3<f32>, far_amount: f32, view_dir: vec3<f32>) -> f32 {
    let h = clamp((p.y - sky.cloud_base) / max(sky.cloud_top - sky.cloud_base, 1.0), 0.0, 1.0);
    let wind = vec2<f32>(sky.cloud_wind_x, sky.cloud_wind_z)
        * sky.cloud_speed * sky.time_sky * 260.0;
    let q = p + vec3<f32>(wind.x, 0.0, wind.y);
    let horizontal = q.xz * 0.010;
    let cloud_type = clamp(fbm(vec3<f32>(horizontal * 0.22, 0.0)), 0.0, 1.0);
    let local_domain = vec3<f32>(horizontal * 0.62, h * 2.8);
    // A direction-space macro field is used only at long distance. It keeps
    // the horizon coherent when a flat local slab projects thousands of world
    // units into a handful of screen pixels.
    let horizon_domain = vec3<f32>(view_dir.xz * 3.2, h * 2.2);
    let macro_domain = mix(local_domain, horizon_domain, far_amount * 0.82);
    let base_noise = fbm(macro_domain);
    let detail_noise = fbm(vec3<f32>(horizontal * 2.4, h * 8.0 + 3.7));
    let profile = cloud_gradient(h, cloud_type);
    let coverage = clamp(sky.cloud_coverage * (0.62 + cloud_type * 0.55), 0.05, 1.0);
    let billow = mix(base_noise, 1.0 - base_noise, clamp(h * 3.5, 0.0, 1.0));
    let shaped = smoothstep(1.0 - coverage, 1.0, billow * 0.78 + detail_noise * 0.22);
    // At long ray distances, suppress high-frequency erosion and bias toward
    // the broad cloud body. This is the same far-proxy idea used by the
    // reference renderer: distant clouds must read as coherent masses rather
    // than noisy pixel-sized fragments.
    let far_body = smoothstep(1.0 - coverage, 1.0, base_noise);
    let stable_shape = mix(shaped, far_body, far_amount * 0.72);
    let erosion = noise3(vec3<f32>(horizontal * 5.0, h * 15.0)) * 0.18 * (1.0 - far_amount);
    return clamp(stable_shape * profile - erosion * stable_shape, 0.0, 1.0);
}

fn intersect_slab(ro: vec3<f32>, rd: vec3<f32>) -> vec2<f32> {
    if (abs(rd.y) < 0.0001) {
        if (ro.y < sky.cloud_base || ro.y > sky.cloud_top) { return vec2<f32>(1e6, -1.0); }
        return vec2<f32>(0.0, 6000.0);
    }
    let a = (sky.cloud_base - ro.y) / rd.y;
    let b = (sky.cloud_top - ro.y) / rd.y;
    return vec2<f32>(max(min(a, b), 0.0), min(max(a, b), 6000.0));
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let size = textureDimensions(output);
    if (gid.x >= size.x || gid.y >= size.y) { return; }
    let uv = (vec2<f32>(gid.xy) + vec2<f32>(0.5)) / vec2<f32>(size);
    let clip = vec4<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);
    let camera = cameras[0];
    let world = camera.view_proj_inv * clip;
    let ro = camera.position_near.xyz;
    let rd = normalize(world.xyz / world.w - ro);
    let hit = intersect_slab(ro, rd);
    if (hit.y <= hit.x) {
        textureStore(output, vec2<i32>(gid.xy), vec4<f32>(0.0));
        textureStore(output_data, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    // Quality controls samples per pixel while the resolution tier controls
    // the number of pixels. This keeps the default quarter-res path cheap but
    // gives Ultra enough depth samples to avoid a visibly flat slab.
    // The Godot demo relies on baked mipmapped 3D textures. Helio's fallback
    // path currently evaluates procedural noise inline, so its public quality
    // tiers use equivalent visual budgets without multiplying that cost.
    let count = min(32u, 12u + sky.cloud_quality * 7u);
    let step_len = (hit.y - hit.x) / f32(count);
    let jitter = hash3(ro + rd * hit.x);
    let light_dir = normalize(sky.sun_direction);
    let phase = max(
        max(henyey_greenstein(dot(light_dir, rd), 0.60),
            henyey_greenstein(dot(light_dir, rd), 0.40 - 1.40 * light_dir.y)),
        henyey_greenstein(dot(light_dir, rd), -0.20));
    var transmittance = 1.0; var light = vec3<f32>(0.0);
    var density_sum = 0.0; var depth_sum = 0.0;
    for (var i = 0u; i < 64u; i++) {
        if (i >= count || transmittance < 0.02) { break; }
        let t = hit.x + (f32(i) + 0.5 + jitter * 0.8) * step_len;
        let p = ro + rd * t;
        let h = clamp((p.y - sky.cloud_base) / max(sky.cloud_top - sky.cloud_base, 1.0), 0.0, 1.0);
        let far_amount = smoothstep(700.0, 3200.0, t);
        let d = cloud_density(p, far_amount, rd);
        let extinction = d * sky.cloud_density * step_len * 0.035;
        let sample_alpha = 1.0 - exp(-extinction);
        var sun_transmittance = 1.0;
        if (d > 0.01 && far_amount < 0.82) {
            let sun_step = max((sky.cloud_top - sky.cloud_base) / 18.0, 2.0);
            let shadow_steps = min(4u, 2u + sky.cloud_quality);
            for (var s = 1u; s <= 4u; s++) {
                let shadow_p = p + light_dir * (f32(s) * sun_step);
                let shadow_d = cloud_density(shadow_p, 0.65, light_dir);
                sun_transmittance *= exp(-shadow_d * sky.cloud_density * sun_step * 0.045);
                if (sun_transmittance < 0.03) { break; }
                if (s >= shadow_steps) { break; }
            }
        }
        let ambient = mix(vec3<f32>(0.24, 0.30, 0.42), vec3<f32>(0.72, 0.80, 0.92), h);
        let sun = mix(vec3<f32>(0.55, 0.62, 0.76), vec3<f32>(1.0, 0.92, 0.78), h);
        let close_radiance = ambient + sun * sun_transmittance * phase * 2.2;
        let far_radiance = ambient + sun * (0.55 + 0.25 * sun_transmittance) * (0.72 + phase * 0.45);
        let radiance = mix(close_radiance, far_radiance, far_amount);
        light += transmittance * radiance * sample_alpha;
        transmittance *= 1.0 - sample_alpha;
        density_sum += d;
        depth_sum += d * t;
    }
    let horizon_stability = smoothstep(0.003, 0.025, abs(rd.y));
    let alpha = clamp(1.0 - transmittance, 0.0, 0.97) * horizon_stability;
    textureStore(output, vec2<i32>(gid.xy), vec4<f32>(light, alpha));
    let representative_depth = select(0.0, clamp((depth_sum / max(density_sum, 0.0001)) / 6000.0, 0.0, 1.0), density_sum > 0.0001);
    textureStore(output_data, vec2<i32>(gid.xy), vec4<f32>(representative_depth, alpha, 0.0, 0.0));
}
