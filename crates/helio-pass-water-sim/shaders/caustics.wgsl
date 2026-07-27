//!use helio_prelude
//
// Caustics projection.
//
// Refracts sunlight through the displaced surface and accumulates where the
// rays land on the volume floor, additively blended. Intensity is the ratio
// between the area a bundle of rays covered before refraction and after — where
// the wave focuses light, the bundle shrinks and the ratio spikes.
//
// The projection is driven by the volume's real bounds. It used to be hardcoded
// to the reference demo's +/-1 pool with a 0.75 magic scale, which meant the
// output only lined up with a volume that happened to match that pool. See #146.
//
// Output: R = caustic intensity, covering the volume footprint in [0,1] UV, so
// a consumer can look it up with the same world-XZ-to-UV mapping the surface
// shader uses.

struct WaterVolume {
    bounds_min:            vec4f,
    bounds_max:            vec4f,  // w = surface_height
    wave_params:           vec4f,  // x = wave_amplitude
    wave_direction:        vec4f,
    water_color:           vec4f,
    extinction:            vec4f,
    reflection_refraction: vec4f,
    caustics_params:       vec4f,  // x = enabled, y = intensity
    fog_params:            vec4f,
    sim_params:            vec4f,  // x = ior, y = caustic_intensity
    shadow_params:         vec4f,  // x = rim
    sun_direction:         vec4f,
    ssr_params:            vec4f,
    sim_dynamics:          vec4f,
    wind_params:           vec4f,
    _pad:                  vec4f,
}

@group(0) @binding(0) var<storage, read> water_volumes: array<WaterVolume>;
@group(0) @binding(1) var water_sim:  texture_2d<f32>;
@group(0) @binding(2) var water_samp: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       flat_hit: vec3f,
    @location(1)       wave_hit: vec3f,
}

// Mirrors surface.wgsl — see the note there. These must stay in step.
fn water_wave_amplitude(vol: WaterVolume) -> f32 {
    let rest     = vol.bounds_max.w;
    let headroom = min(rest - vol.bounds_min.y, vol.bounds_max.y - rest);
    return clamp(vol.wave_params.x, 0.0, max(headroom, 0.0));
}

/// Where a ray from `origin` along `dir` meets the volume floor.
fn floor_hit(origin: vec3f, dir: vec3f, floor_y: f32) -> vec3f {
    // A ray that is not descending never reaches the floor; park it at the
    // origin so the area ratio degenerates to 1 rather than to infinity.
    if dir.y > -1e-4 {
        return vec3f(origin.x, floor_y, origin.z);
    }
    return origin + dir * ((floor_y - origin.y) / dir.y);
}

@vertex
fn vs_main(@location(0) position: vec3f) -> VertexOutput {
    let vol = water_volumes[0];

    let uv     = position.xy * 0.5 + 0.5;
    let info   = textureSampleLevel(water_sim, water_samp, uv, 0.0);
    let extent = max(vol.bounds_max.xz - vol.bounds_min.xz, vec2f(1e-4));
    let amp    = water_wave_amplitude(vol);

    let xz      = mix(vol.bounds_min.xz, vol.bounds_max.xz, uv);
    let rest_y  = vol.bounds_max.w;
    let wave_y  = rest_y + info.r * amp;
    let floor_y = vol.bounds_min.y;

    // Surface normal, slopes rescaled into world units (as in surface.wgsl).
    let ba     = vec2f(info.b, info.a);
    let ny     = sqrt(max(1.0 - dot(ba, ba), 1e-6));
    let slope  = (ba / ny) * (amp / extent);
    let normal = normalize(vec3f(slope.x, 1.0, slope.y));

    let light_dir = normalize(vol.sun_direction.xyz);
    let eta       = 1.0 / max(vol.sim_params.x, 1.0);

    // Light travels along -light_dir. Refract it through the flat surface and
    // through the actual wave; the difference between where the two land is the
    // focusing that produces caustics.
    let flat_refract = refract(-light_dir, vec3f(0.0, 1.0, 0.0), eta);
    let wave_refract = refract(-light_dir, normal, eta);

    let flat_hit = floor_hit(vec3f(xz.x, rest_y, xz.y), flat_refract, floor_y);
    let wave_hit = floor_hit(vec3f(xz.x, wave_y, xz.y), wave_refract, floor_y);

    // Rasterize at the refracted landing point, in volume-footprint UV.
    let out_uv = (wave_hit.xz - vol.bounds_min.xz) / extent;

    var out: VertexOutput;
    out.position = vec4f(helio_uv_to_ndc(out_uv), 0.0, 1.0);
    out.flat_hit = flat_hit;
    out.wave_hit = wave_hit;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol = water_volumes[0];

    // Area ratio: how much the ray bundle contracted between the flat and the
    // displaced surface.
    let flat_area = length(dpdx(in.flat_hit)) * length(dpdy(in.flat_hit));
    let wave_area = length(dpdx(in.wave_hit)) * length(dpdy(in.wave_hit));

    // A vanishing wave_area is a caustic singularity; clamp so it stays finite.
    let ratio = clamp(flat_area / max(wave_area, 1e-6), 0.0, 8.0);

    // Relative to flat water, not absolute. A ratio of exactly 1 means the wave
    // neither focused nor spread the bundle, so it must contribute nothing —
    // emitting `ratio` directly would lay a uniform DC wash over the whole
    // floor and drown the pattern it is supposed to show. Focusing goes
    // positive, spreading goes negative, which is what makes caustics read as
    // bright lines between dimmer gaps rather than as an overall brightening.
    let intensity = (ratio - 1.0) * max(vol.sim_params.y, 0.0);
    return vec4f(clamp(intensity, -1.0, 6.0), 1.0, 0.0, 1.0);
}
