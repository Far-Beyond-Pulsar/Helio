// Physically based atmosphere (Hillaire 2020, "A Scalable and Production
// Ready Sky and Atmosphere Rendering Technique"), shared by the atmosphere
// LUT kernels and the composite. Distances are kilometres, positions are
// relative to the planet centre unless named otherwise.

// `AtmosphereComponent` (SceneDB row, packed): coefficients per kilometre.
struct AtmosphereParams {
    center: vec3<f32>,
    // 0: ground at the world origin (planet centre straight below it),
    // 1: planet centre at `center` (world metres).
    placement: u32,
    rayleigh_scattering: vec3<f32>,
    rayleigh_scale_height: f32,
    mie_scattering: vec3<f32>,
    mie_scale_height: f32,
    mie_absorption: vec3<f32>,
    mie_g: f32,
    ozone_absorption: vec3<f32>,
    ozone_center: f32,
    ground_albedo: vec3<f32>,
    ozone_width: f32,
    bottom_radius: f32,
    top_radius: f32,
    sun_angular_radius: f32,
    enabled: u32,
}

// Per frame, resolved on the GPU (`resolve`).
struct AtmosphereFrame {
    // Planet centre relative to the camera (km); w: 1 when an atmosphere is
    // active.
    planet: vec4<f32>,
    // Direction towards the sun; w: cos of its angular radius.
    sun: vec4<f32>,
    // The sun's illuminance (the directional light's colour x intensity);
    // w: distance (km) the aerial-perspective volume reaches.
    sun_illuminance: vec4<f32>,
    // Camera relative to the planet centre (km); w: unused.
    eye: vec4<f32>,
    params: AtmosphereParams,
    // Diffuse sky irradiance / pi as L2 spherical harmonics (rgb): a
    // Lambertian surface of albedo a with normal n receives a * E(n).
    irradiance: array<vec4<f32>, 9>,
}

const ATMOSPHERE_PI: f32 = 3.14159265358979;
const TRANSMITTANCE_SIZE: vec2<f32> = vec2<f32>(256.0, 64.0);
const MULTI_SCATTERING_SIZE: f32 = 32.0;
const SKY_VIEW_SIZE: vec2<f32> = vec2<f32>(192.0, 108.0);
const AERIAL_SLICES: f32 = 32.0;

struct Medium {
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
    scattering: vec3<f32>,
    extinction: vec3<f32>,
}

fn medium_at(p: AtmosphereParams, height: f32) -> Medium {
    let h = max(height, 0.0);
    let rayleigh_density = exp(-h / p.rayleigh_scale_height);
    let mie_density = exp(-h / p.mie_scale_height);
    let ozone_density = max(0.0, 1.0 - abs(h - p.ozone_center) / max(0.5 * p.ozone_width, 1e-3));
    var m: Medium;
    m.rayleigh = p.rayleigh_scattering * rayleigh_density;
    m.mie = p.mie_scattering * mie_density;
    m.scattering = m.rayleigh + m.mie;
    m.extinction = m.scattering + p.mie_absorption * mie_density + p.ozone_absorption * ozone_density;
    return m;
}

fn phase_rayleigh(cos_theta: f32) -> f32 {
    return 3.0 / (16.0 * ATMOSPHERE_PI) * (1.0 + cos_theta * cos_theta);
}

// Cornette-Shanks.
fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let k = 3.0 / (8.0 * ATMOSPHERE_PI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + cos_theta * cos_theta) / pow(max(1.0 + g2 - 2.0 * g * cos_theta, 1e-5), 1.5);
}

// Distance along `d` from `o` to the sphere of radius `r` around the
// origin: x the near hit, y the far one (both negative: missed).
fn ray_sphere(o: vec3<f32>, d: vec3<f32>, r: f32) -> vec2<f32> {
    let b = dot(o, d);
    // Perpendicular form: exact for rays far from the sphere.
    let perpendicular = o - b * d;
    let disc = r * r - dot(perpendicular, perpendicular);
    if disc < 0.0 { return vec2<f32>(-1.0); }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

// Bruneton's transmittance parameterisation: (r, mu) <-> uv.
fn transmittance_uv(p: AtmosphereParams, r: f32, mu: f32) -> vec2<f32> {
    let big_h = sqrt(max(p.top_radius * p.top_radius - p.bottom_radius * p.bottom_radius, 0.0));
    let rho = sqrt(max(r * r - p.bottom_radius * p.bottom_radius, 0.0));
    let disc = r * r * (mu * mu - 1.0) + p.top_radius * p.top_radius;
    let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    let d_min = p.top_radius - r;
    let d_max = rho + big_h;
    return vec2<f32>((d - d_min) / max(d_max - d_min, 1e-5), rho / max(big_h, 1e-5));
}

fn transmittance_r_mu(p: AtmosphereParams, uv: vec2<f32>) -> vec2<f32> {
    let big_h = sqrt(max(p.top_radius * p.top_radius - p.bottom_radius * p.bottom_radius, 0.0));
    let rho = big_h * uv.y;
    let r = sqrt(rho * rho + p.bottom_radius * p.bottom_radius);
    let d_min = p.top_radius - r;
    let d_max = rho + big_h;
    let d = d_min + uv.x * (d_max - d_min);
    var mu = 1.0;
    if d > 0.0 { mu = (big_h * big_h - rho * rho - d * d) / (2.0 * r * d); }
    return vec2<f32>(r, clamp(mu, -1.0, 1.0));
}

// Whether the ray from radius `r` at cosine `mu` towards the zenith hits
// the ground.
fn hits_ground(p: AtmosphereParams, r: f32, mu: f32) -> bool {
    return mu < 0.0 && r * r * (mu * mu - 1.0) + p.bottom_radius * p.bottom_radius >= 0.0;
}

// Local frame of the sky-view LUT at the camera: y up, x towards the sun's
// azimuth.
fn sky_view_basis(up: vec3<f32>, sun: vec3<f32>) -> mat3x3<f32> {
    var x = sun - up * dot(sun, up);
    if dot(x, x) < 1e-8 {
        x = cross(select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(up.y) > 0.9), up);
    }
    x = normalize(x);
    return mat3x3<f32>(x, up, cross(x, up));
}

// Zenith angle of the geometric horizon seen from radius `r`.
fn horizon_zenith(p: AtmosphereParams, r: f32) -> f32 {
    let s = sqrt(max(r * r - p.bottom_radius * p.bottom_radius, 0.0)) / max(r, 1e-5);
    return ATMOSPHERE_PI - acos(clamp(s, -1.0, 1.0));
}

// Sky-view LUT: rows concentrate at the horizon, columns near the sun's
// azimuth (the sky is symmetric about it).
fn sky_view_uv(p: AtmosphereParams, r: f32, zenith: f32, azimuth: f32) -> vec2<f32> {
    let horizon = horizon_zenith(p, r);
    var v: f32;
    if zenith < horizon {
        v = 0.5 * (1.0 - sqrt(max(1.0 - zenith / horizon, 0.0)));
    } else {
        v = 0.5 + 0.5 * sqrt(clamp((zenith - horizon) / max(ATMOSPHERE_PI - horizon, 1e-5), 0.0, 1.0));
    }
    return vec2<f32>(sqrt(clamp(azimuth / ATMOSPHERE_PI, 0.0, 1.0)), v);
}

fn sky_view_angles(p: AtmosphereParams, r: f32, uv: vec2<f32>) -> vec2<f32> {
    let horizon = horizon_zenith(p, r);
    var zenith: f32;
    if uv.y < 0.5 {
        let c = 1.0 - 2.0 * uv.y;
        zenith = horizon * (1.0 - c * c);
    } else {
        let c = 2.0 * uv.y - 1.0;
        zenith = horizon + c * c * (ATMOSPHERE_PI - horizon);
    }
    return vec2<f32>(zenith, uv.x * uv.x * ATMOSPHERE_PI);
}

// Zenith and sun-relative azimuth of world direction `d` seen from the
// camera.
fn sky_view_direction_angles(frame: AtmosphereFrame, d: vec3<f32>) -> vec2<f32> {
    let up = normalize(frame.eye.xyz);
    let basis = sky_view_basis(up, frame.sun.xyz);
    let local = transpose(basis) * d;
    return vec2<f32>(acos(clamp(local.y, -1.0, 1.0)), abs(atan2(local.z, local.x)));
}

// Aerial-perspective slice coordinate (0..1) of a distance (km): slices are
// spaced quadratically up to `range`.
fn aerial_slice(distance: f32, range: f32) -> f32 {
    return sqrt(clamp(distance / max(range, 1e-5), 0.0, 1.0));
}

fn aerial_distance(slice: f32, range: f32) -> f32 {
    return slice * slice * range;
}

// Transmittance from a point `from_camera` (km, relative to the camera)
// towards the direction `towards`, integrated in a few steps rather than read
// from the transmittance LUT: passes at their sampled-texture budget (the
// deferred lighting pass) light with it. Fades to zero as the planet's limb
// covers a sun of the frame's angular radius. One where no atmosphere is
// active.
fn atmosphere_transmittance_towards(frame: AtmosphereFrame, from_camera: vec3<f32>, towards: vec3<f32>) -> vec3<f32> {
    if frame.planet.w < 0.5 { return vec3<f32>(1.0); }
    let p = frame.params;
    let pos = from_camera - frame.planet.xyz;
    // Points below the planet's surface (valleys, caves) are lit as if on it.
    let r = max(length(pos), p.bottom_radius);
    let up = normalize(pos);
    let o = up * r;
    let mu = dot(up, towards);
    let mu_horizon = -sqrt(max(1.0 - (p.bottom_radius / r) * (p.bottom_radius / r), 0.0));
    let sun_sin = sqrt(max(1.0 - frame.sun.w * frame.sun.w, 0.0));
    let visible = smoothstep(mu_horizon - sun_sin, mu_horizon + sun_sin, mu);
    if visible <= 0.0 { return vec3<f32>(0.0); }
    let top = ray_sphere(o, towards, p.top_radius);
    let start = max(top.x, 0.0);
    let length_through = max(top.y - start, 0.0);
    if length_through <= 0.0 { return vec3<f32>(visible); }
    // Steps spaced quadratically: the air is densest where the path starts.
    let steps = 12u;
    var optical = vec3<f32>(0.0);
    for (var i = 0u; i < steps; i++) {
        let u0 = f32(i) / f32(steps);
        let u1 = f32(i + 1u) / f32(steps);
        let t = start + length_through * 0.25 * (u0 + u1) * (u0 + u1);
        let dt = length_through * (u1 * u1 - u0 * u0);
        let height = length(o + towards * t) - p.bottom_radius;
        optical += medium_at(p, height).extinction * dt;
    }
    return exp(-optical) * visible;
}

// Diffuse sky irradiance / pi at the camera for a surface facing `n`: a
// Lambertian surface of albedo a reflects a * this.
fn atmosphere_sky_irradiance(frame: AtmosphereFrame, n: vec3<f32>) -> vec3<f32> {
    let sh = frame.irradiance;
    let e = sh[0].rgb * 0.282095
        + sh[1].rgb * (0.488603 * n.y)
        + sh[2].rgb * (0.488603 * n.z)
        + sh[3].rgb * (0.488603 * n.x)
        + sh[4].rgb * (1.092548 * n.x * n.y)
        + sh[5].rgb * (1.092548 * n.y * n.z)
        + sh[6].rgb * (0.315392 * (3.0 * n.z * n.z - 1.0))
        + sh[7].rgb * (1.092548 * n.x * n.z)
        + sh[8].rgb * (0.546274 * (n.x * n.x - n.y * n.y));
    return max(e, vec3<f32>(0.0));
}

// The segment of a ray inside the atmosphere's shell, cut at the ground:
// x start, y end, z 1 when it ends on the ground.
fn atmosphere_segment(p: AtmosphereParams, o: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let top = ray_sphere(o, d, p.top_radius);
    if top.y <= 0.0 { return vec3<f32>(0.0); }
    let start = max(top.x, 0.0);
    var end = top.y;
    let ground = ray_sphere(o, d, p.bottom_radius);
    var hit = 0.0;
    if ground.x > 0.0 && ground.x < end { end = ground.x; hit = 1.0; }
    return vec3<f32>(start, max(end, start), hit);
}

// Transmittance to the sun from radius `r` at sun cosine `mu_s`, read from
// the transmittance LUT; zero behind the planet.
fn atmosphere_sun_transmittance_lut(p: AtmosphereParams, lut: texture_2d<f32>, s: sampler, r: f32, mu_s: f32) -> vec3<f32> {
    if hits_ground(p, r, mu_s) { return vec3<f32>(0.0); }
    return textureSampleLevel(lut, s, transmittance_uv(p, r, mu_s), 0.0).rgb;
}

fn atmosphere_multi_scattering_lut(p: AtmosphereParams, lut: texture_2d<f32>, s: sampler, r: f32, mu_s: f32) -> vec3<f32> {
    let uv = vec2<f32>(mu_s * 0.5 + 0.5, (r - p.bottom_radius) / (p.top_radius - p.bottom_radius));
    return textureSampleLevel(lut, s, clamp(uv, vec2<f32>(0.0), vec2<f32>(1.0)), 0.0).rgb;
}

// In-scattered radiance (unit sun) and transmittance along `d` from `o`
// over [t0, t1]: single scattering towards `sun` plus Hillaire's multiple
// scattering. Steps are spaced quadratically from t0 (rays leaving the
// ground, where the air is densest) or evenly (rays through the shell from
// space, densest at their middle).
struct Scattering {
    radiance: vec3<f32>,
    transmittance: vec3<f32>,
}

fn atmosphere_integrate(p: AtmosphereParams, sun: vec3<f32>, transmittance_lut: texture_2d<f32>,
    multi_scattering_lut: texture_2d<f32>, s: sampler, o: vec3<f32>, d: vec3<f32>,
    t0: f32, t1: f32, steps: f32, quadratic: bool) -> Scattering {
    let cos_theta = dot(d, sun);
    let rayleigh_phase = phase_rayleigh(cos_theta);
    let mie_phase = phase_mie(cos_theta, p.mie_g);
    var out: Scattering;
    out.radiance = vec3<f32>(0.0);
    out.transmittance = vec3<f32>(1.0);
    var t_prev = t0;
    for (var i = 0.0; i < steps; i += 1.0) {
        var f = (i + 1.0) / steps;
        if quadratic { f = f * f; }
        let t = t0 + (t1 - t0) * f;
        let dt = t - t_prev;
        let x = o + d * (t_prev + 0.5 * dt);
        t_prev = t;
        let rx = length(x);
        let m = medium_at(p, rx - p.bottom_radius);
        let mu_s = dot(x / rx, sun);
        let lit = atmosphere_sun_transmittance_lut(p, transmittance_lut, s, rx, mu_s);
        let source = (m.rayleigh * rayleigh_phase + m.mie * mie_phase) * lit
            + m.scattering * atmosphere_multi_scattering_lut(p, multi_scattering_lut, s, rx, mu_s);
        let step_t = exp(-m.extinction * dt);
        let integral = (vec3<f32>(1.0) - step_t) / max(m.extinction, vec3<f32>(1e-7));
        out.radiance += out.transmittance * source * integral;
        out.transmittance *= step_t;
    }
    return out;
}
