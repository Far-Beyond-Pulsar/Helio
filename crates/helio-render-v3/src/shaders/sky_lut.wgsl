// sky_lut.wgsl — Precomputed atmospheric scattering LUT (Hillaire 2020 approximation).
// Output: 256×64 Rgba16Float — transmittance + in-scatter radiance for each
//         (view_zenith_cos, sun_zenith_cos) pair.
//
// Dispatch: (256/8, 64/8, 1) = (32, 8, 1) workgroups.

override LUT_WIDTH:  u32 = 256u;
override LUT_HEIGHT: u32 = 64u;

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

@group(0) @binding(0) var<uniform> globals:    Globals;
@group(0) @binding(1) var          lut_output: texture_storage_2d<rgba16float, write>;

const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;

// Atmosphere constants (Rayleigh + Mie, Earth-like).
const EARTH_R: f32   = 6371000.0;
const ATM_R: f32     = 6471000.0;
const HR: f32        = 8000.0;   // Rayleigh scale height (m)
const HM: f32        = 1200.0;   // Mie scale height (m)
const MIE_G: f32     = 0.8;      // Mie phase asymmetry

const BETA_R: vec3<f32> = vec3<f32>(5.8e-6, 13.5e-6, 33.1e-6);   // Rayleigh scattering
const BETA_M: vec3<f32> = vec3<f32>(21e-6);                        // Mie scattering

// Intersect ray with sphere; returns (t_near, t_far) or negative if miss.
fn intersect_sphere(orig: vec3<f32>, dir: vec3<f32>, radius: f32) -> vec2<f32> {
    let a = dot(dir, dir);
    let b = 2.0 * dot(orig, dir);
    let c = dot(orig, orig) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 { return vec2<f32>(-1.0); }
    let sq = sqrt(disc);
    return vec2<f32>((-b - sq) / (2.0 * a), (-b + sq) / (2.0 * a));
}

fn phase_rayleigh(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn phase_mie(cos_theta: f32) -> f32 {
    let g2 = MIE_G * MIE_G;
    return (3.0 / (8.0 * PI)) * ((1.0 - g2) * (1.0 + cos_theta * cos_theta)) /
        ((2.0 + g2) * pow(1.0 + g2 - 2.0 * MIE_G * cos_theta, 1.5));
}

// Numerically integrate transmittance + in-scatter along ray in atmosphere.
fn compute_lut_sample(cos_view_zenith: f32, cos_sun_zenith: f32) -> vec4<f32> {
    // View ray from sea level looking up at cos_view_zenith angle.
    let origin  = vec3<f32>(0.0, EARTH_R + 1.0, 0.0);
    let view_dir = vec3<f32>(sqrt(1.0 - cos_view_zenith * cos_view_zenith), cos_view_zenith, 0.0);
    let sun_dir  = vec3<f32>(sqrt(1.0 - cos_sun_zenith  * cos_sun_zenith),  cos_sun_zenith,  0.0);

    // Find atmosphere-sphere intersection.
    let atm_hit = intersect_sphere(origin, view_dir, ATM_R);
    let t_min   = max(atm_hit.x, 0.0);
    let t_max   = atm_hit.y;
    if t_max <= 0.0 { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }

    let NUM_STEPS: i32 = 16;
    let step_size = (t_max - t_min) / f32(NUM_STEPS);

    var transmittance = vec3<f32>(1.0);
    var in_scatter_r  = vec3<f32>(0.0);
    var in_scatter_m  = vec3<f32>(0.0);

    var t = t_min + step_size * 0.5;
    for (var i = 0; i < NUM_STEPS; i++) {
        let pos  = origin + view_dir * t;
        let h    = length(pos) - EARTH_R;
        let hr   = exp(-h / HR);
        let hm   = exp(-h / HM);

        let dt    = step_size;
        let ext_r = BETA_R * hr * dt;
        let ext_m = BETA_M * hm * dt;
        transmittance *= exp(-(ext_r + ext_m));

        // Approximate sun visibility (single-scattering, no shadow rays for LUT).
        let cos_theta = dot(view_dir, sun_dir);
        in_scatter_r  += transmittance * BETA_R * hr * phase_rayleigh(cos_theta) * dt;
        in_scatter_m  += transmittance * BETA_M * hm * phase_mie(cos_theta)      * dt;

        t += step_size;
    }

    let scatter = (in_scatter_r + in_scatter_m) * vec3<f32>(20.0, 18.0, 16.0); // sun irradiance approx
    // Pack: RGB = in-scatter, A = average transmittance luma.
    let trans_luma = dot(transmittance, vec3<f32>(0.2126, 0.7152, 0.0722));
    return vec4<f32>(scatter, trans_luma);
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= LUT_WIDTH || gid.y >= LUT_HEIGHT { return; }

    let u = (f32(gid.x) + 0.5) / f32(LUT_WIDTH);
    let v = (f32(gid.y) + 0.5) / f32(LUT_HEIGHT);

    // Map UV to (view_zenith_cos, sun_zenith_cos).
    let cos_view_zenith = u * 2.0 - 1.0;   // -1..1
    let cos_sun_zenith  = v * 2.0 - 1.0;   // -1..1

    let result = compute_lut_sample(cos_view_zenith, cos_sun_zenith);
    textureStore(lut_output, vec2<i32>(i32(gid.x), i32(gid.y)), result);
}
