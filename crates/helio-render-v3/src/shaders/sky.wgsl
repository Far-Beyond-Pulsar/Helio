// sky.wgsl — fullscreen sky rendering from precomputed LUT.

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,
};

struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec3<f32>,
    csm_split_count:   u32,
    rc_world_min:      vec3<f32>,
    _pad0:             u32,
};

@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> globals: Globals;
@group(0) @binding(2) var          sky_lut: texture_2d<f32>;
@group(0) @binding(3) var          lut_smp: sampler;

// Fullscreen triangle.
@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    return vec4<f32>(pos[vi], 0.9999, 1.0);  // depth=0.9999 so it loses depth test vs geometry
}

const PI: f32 = 3.14159265358979;

// Reconstruct view ray from NDC.
fn clip_to_world_dir(ndc: vec2<f32>) -> vec3<f32> {
    let clip = vec4<f32>(ndc, 0.0, 1.0);
    let eye  = camera.view_proj_inv * clip;
    return normalize(eye.xyz / eye.w - camera.position);
}

// Fixed sun direction derived from globals.ambient_color as a proxy (or const for now).
// A proper implementation would read sun direction from a uniform.
fn sun_direction() -> vec3<f32> {
    return normalize(vec3<f32>(0.3, 0.8, 0.1));
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(sky_lut));
    let screen_uv = frag.xy / vec2<f32>(textureDimensions(sky_lut));

    // Reconstruct ray.
    let ndc = frag.xy / vec2<f32>(f32(textureNumLevels(sky_lut)), 1.0); // workaround: use actual dims
    // Simple: map frag to NDC via viewport size in the LUT texture dims (scaled by pass).
    let ray = clip_to_world_dir(frag.xy / dims * 2.0 - 1.0);

    let sun_dir = sun_direction();

    // LUT UV: (cos_view_zenith+1)*0.5, (cos_sun_zenith+1)*0.5
    let cos_view = clamp(ray.y, -1.0, 1.0);
    let cos_sun  = clamp(sun_dir.y, -1.0, 1.0);
    let lut_uv   = vec2<f32>((cos_view + 1.0) * 0.5, (cos_sun + 1.0) * 0.5);

    let scatter  = textureSample(sky_lut, lut_smp, lut_uv).rgb;

    // Sun disc.
    let cos_sun_angle = dot(ray, sun_dir);
    let sun_disc = smoothstep(0.9997, 1.0, cos_sun_angle) * 5.0;
    let sun_color = vec3<f32>(1.0, 0.95, 0.85) * sun_disc;

    let sky = scatter + sun_color;

    // ACES-ish tonemap (same as deferred_lighting).
    let mapped = sky / (sky + vec3<f32>(0.187)) * 1.035;

    return vec4<f32>(mapped, 1.0);
}
