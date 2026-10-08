//!use atmosphere
// Atmosphere composite over the lit scene (`pre_aa`), blended as
// dst * alpha + src: sky pixels (no geometry) are replaced by the sky and the
// sun's disk (alpha 0); geometry keeps its light attenuated by the air in
// front of it plus the light the air scatters towards the eye (aerial
// perspective, alpha = transmittance). Without an atmosphere it writes
// (0, 1): the scene is unchanged.
//
// Inside the atmosphere the sky comes from the sky-view LUT and the air in
// front of geometry from the aerial-perspective volume. From space both
// would spend their resolution on empty space around a thin shell (and the
// volume would not reach the planet), so each pixel marches its own ray
// through the shell instead.

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> frame: AtmosphereFrame;
@group(0) @binding(2) var transmittance_lut: texture_2d<f32>;
@group(0) @binding(3) var sky_view_lut: texture_2d<f32>;
@group(0) @binding(4) var aerial_lut: texture_3d<f32>;
@group(0) @binding(5) var depth_texture: texture_depth_2d;
@group(0) @binding(6) var lut_sampler: sampler;
@group(0) @binding(7) var multi_scattering_lut: texture_2d<f32>;

const SPACE_STEPS: f32 = 32.0;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) v: u32) -> VertexOut {
    let p = vec2<f32>(f32((v << 1u) & 2u), f32(v & 2u));
    var out: VertexOut;
    out.position = vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(p.x, 1.0 - p.y);
    return out;
}

fn transmittance(r: f32, mu: f32) -> vec3<f32> {
    return textureSampleLevel(transmittance_lut, lut_sampler, transmittance_uv(frame.params, r, mu), 0.0).rgb;
}

// The air along `d` from the camera up to `limit` km, marched per pixel;
// with no limit (sky pixels) it includes the sunlit ground it may end on.
fn march_from_space(eye: vec3<f32>, d: vec3<f32>, limit: f32) -> Scattering {
    let p = frame.params;
    let segment = atmosphere_segment(p, eye, d);
    let end = min(segment.y, limit);
    var out: Scattering;
    out.radiance = vec3<f32>(0.0);
    out.transmittance = vec3<f32>(1.0);
    if end <= segment.x { return out; }
    out = atmosphere_integrate(p, frame.sun.xyz, transmittance_lut, multi_scattering_lut, lut_sampler,
        eye, d, segment.x, end, SPACE_STEPS, false);
    if segment.z > 0.5 && limit >= segment.y {
        let n = normalize(eye + d * segment.y);
        let mu_s = dot(n, frame.sun.xyz);
        out.radiance += out.transmittance
            * atmosphere_sun_transmittance_lut(p, transmittance_lut, lut_sampler, p.bottom_radius, mu_s)
            * max(mu_s, 0.0) * p.ground_albedo / ATMOSPHERE_PI;
    }
    return out;
}

@fragment
fn fs_composite(in: VertexOut) -> @location(0) vec4<f32> {
    if frame.planet.w < 0.5 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    let camera = cameras[0];
    let pixel = vec2<i32>(in.position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    let ndc = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
    // A point at mid depth gives the direction even when the far plane is
    // at infinity (w = 0 there).
    let mid = camera.inv_view_proj * vec4<f32>(ndc, 0.5, 1.0);
    let d = normalize(mid.xyz / mid.w - camera.position_near.xyz);
    let sun = frame.sun_illuminance.rgb;
    let eye = frame.eye.xyz;
    let r = length(eye);
    let in_space = r > frame.params.top_radius;
    if depth >= 1.0 {
        var radiance: vec3<f32>;
        // The air between the eye and the sun's disk along `d`.
        var through = vec3<f32>(1.0);
        if in_space {
            let air = march_from_space(eye, d, 3.0e38);
            radiance = air.radiance * sun;
            through = air.transmittance;
        } else {
            through = transmittance(r, dot(eye / r, d));
            let angles = sky_view_direction_angles(frame, d);
            radiance = textureSampleLevel(sky_view_lut, lut_sampler,
                sky_view_uv(frame.params, r, angles.x, angles.y), 0.0).rgb * sun;
        }
        // The sun's disk, through the air in front of it, unless the planet
        // hides it.
        let cos_sun = dot(d, frame.sun.xyz);
        if cos_sun > frame.sun.w && ray_sphere(eye, d, frame.params.bottom_radius).x < 0.0 {
            let solid_angle = 2.0 * ATMOSPHERE_PI * (1.0 - frame.sun.w);
            let edge = clamp((cos_sun - frame.sun.w) / max(1.0 - frame.sun.w, 1e-7), 0.0, 1.0);
            let limb = 0.4 + 0.6 * sqrt(edge);
            radiance += sun / solid_angle * limb * through;
        }
        return vec4<f32>(radiance, 0.0);
    }
    let point = camera.inv_view_proj * vec4<f32>(ndc, depth, 1.0);
    let distance = length(point.xyz / point.w - camera.position_near.xyz) * 0.001;
    if in_space {
        let air = march_from_space(eye, d, distance);
        return vec4<f32>(air.radiance * sun, dot(air.transmittance, vec3<f32>(1.0 / 3.0)));
    }
    let range = frame.sun_illuminance.w;
    // Slices hold values at their centres; the first fades in from the eye.
    let slice = aerial_slice(distance, range) * AERIAL_SLICES - 0.5;
    var aerial = textureSampleLevel(aerial_lut, lut_sampler,
        vec3<f32>(in.uv, (max(slice, 0.0) + 0.5) / AERIAL_SLICES), 0.0);
    if slice < 0.0 {
        aerial = mix(vec4<f32>(0.0, 0.0, 0.0, 1.0), aerial, clamp(slice + 1.0, 0.0, 1.0));
    }
    return vec4<f32>(aerial.rgb * sun, aerial.a);
}
