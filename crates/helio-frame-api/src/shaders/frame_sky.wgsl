// Physically based atmosphere (Hillaire 2020, "A Scalable and Production
// Ready Sky and Atmosphere Rendering Technique"), shared by the atmosphere
// LUT kernels and the composite. Distances are kilometres, positions are
// relative to the planet centre unless named otherwise.

// The atmosphere's participating medium (the frame's `sky` contract):
// Rayleigh, Mie and ozone, coefficients per kilometre, and where the planet
// is. What the sky draws of its own (clouds, stars, a panorama) is the sky
// plugin's (`sky_layers.wgsl` in `helio-pass-sky`).
struct SkyMedium {
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
struct SkyFrame {
    // Planet centre relative to the camera (km); w: 1 when an atmosphere is
    // active.
    planet: vec4<f32>,
    // Reference direction for the sky-view LUT's angular coordinate system;
    // w: cos of the atmosphere's default emitter radius.
    sun: vec4<f32>,
    // First light's legacy illuminance (rgb); all current sky lighting reads
    // directional_lights below. w: aerial-perspective range in km.
    sun_illuminance: vec4<f32>,
    // Camera relative to the planet centre (km); w: unused.
    eye: vec4<f32>,
    params: SkyMedium,
    // Diffuse sky irradiance / pi as L2 spherical harmonics (rgb): a
    // Lambertian surface of albedo a with normal n receives a * E(n).
    // Seen from the eye; surfaces take their own sky from the sky table
    // (`sky_table`).
    irradiance: array<vec4<f32>, 9>,
    // Up to 32 directional lights, each stored as direction, radiance,
    // then emitter metadata (radius, brightness, texture kind, enabled).
    directional_lights: array<vec4<f32>, 96>,
    directional_light_count: vec4<u32>,
}

// The sky anywhere in the atmosphere, for a unit sun, by altitude
// (SKY_TABLE_R rows, by the square root of the height) and sun cosine
// (SKY_TABLE_MU columns from SKY_TABLE_MU_LO to 1, dense about the horizon,
// where the sky changes fastest with the sun: `sky_table_column_mu`), in
// each entry's local frame
// (y up, x towards the sun's azimuth; the sky is symmetric about the sun's
// vertical plane). Two tables share these entries:
//
// - `sky_radiance` (`sky_radiance_kernel`): the sky's radiance by
//   direction, what a ray that meets nothing takes (`sky_radiance_eval`).
//   A 3D texture: entry `ir * SKY_TABLE_MU + im` is slice `ir * MU + im`, a
//   SKY_RADIANCE_AZIMUTH x SKY_RADIANCE_ZENITH map, rows as the sky-view
//   LUT's at the row's radius (`sky_radiance_uvw`), computed as the
//   sky-view LUT is (`sky_integrate`, 32 steps, the lit ground below the
//   horizon).
// - `sky_table` (`sky_table_kernel`, from `sky_radiance`): irradiance / pi
//   on a surface, the six L2 harmonics symmetric about the sun's plane,
//   cosine-convolved, SKY_TABLE_COEFFS vec4s an entry (`sky_table_eval`).
const SKY_TABLE_R: u32 = 16u;
const SKY_TABLE_MU: u32 = 40u;
const SKY_TABLE_MU_LO: f32 = -0.4;
// Columns below the horizon: column SKY_TABLE_MU_BELOW is a sun on it.
const SKY_TABLE_MU_BELOW: u32 = 16u;
const SKY_TABLE_COEFFS: u32 = 6u;
// The irradiance table's vec4s (a reader binding it as a uniform sizes its
// array by this).
const SKY_TABLE_VEC4S: u32 = SKY_TABLE_R * SKY_TABLE_MU * SKY_TABLE_COEFFS;
const SKY_RADIANCE_AZIMUTH: u32 = 32u;
const SKY_RADIANCE_ZENITH: u32 = 64u;

// Height (km) of the sky tables' lowest row (an eye stands above it): the
// ground seen just below the horizon is as far as the horizon, which grows
// as the square root of the eye's height, so an eye below the lowest row
// sees it nearer than any row does.
const SKY_TABLE_LOWEST: f32 = 0.001;

// Radius of the sky tables' row `ir`: quadratic in the height from
// SKY_TABLE_LOWEST to the top of the air.
fn sky_table_row_radius(p: SkyMedium, ir: u32) -> f32 {
    let x = f32(ir) / f32(SKY_TABLE_R - 1u);
    return p.bottom_radius + SKY_TABLE_LOWEST
        + x * x * (p.top_radius - p.bottom_radius - SKY_TABLE_LOWEST);
}

// Sun cosine of the sky tables' column `im`: quadratic away from the
// horizon on either side (the sun's path through the air, and so the sky,
// changes fastest as it nears the horizon).
fn sky_table_column_mu(im: u32) -> f32 {
    if im < SKY_TABLE_MU_BELOW {
        let x = f32(SKY_TABLE_MU_BELOW - im) / f32(SKY_TABLE_MU_BELOW);
        return SKY_TABLE_MU_LO * x * x;
    }
    let x = f32(im - SKY_TABLE_MU_BELOW) / f32(SKY_TABLE_MU - 1u - SKY_TABLE_MU_BELOW);
    return x * x;
}

// The fractional row (0..SKY_TABLE_R - 1) of radius `r`
// (`sky_table_row_radius`'s inverse).
fn sky_table_row(p: SkyMedium, r: f32) -> f32 {
    let span = p.top_radius - p.bottom_radius - SKY_TABLE_LOWEST;
    let x = clamp((r - p.bottom_radius - SKY_TABLE_LOWEST) / span, 0.0, 1.0);
    return sqrt(x) * f32(SKY_TABLE_R - 1u);
}

// The fractional column (0..SKY_TABLE_MU - 1) of sun cosine `mu_s`
// (`sky_table_column_mu`'s inverse).
fn sky_table_column(mu_s: f32) -> f32 {
    let below = f32(SKY_TABLE_MU_BELOW);
    if mu_s < 0.0 {
        return below * (1.0 - sqrt(clamp(mu_s / SKY_TABLE_MU_LO, 0.0, 1.0)));
    }
    return below + f32(SKY_TABLE_MU - 1u - SKY_TABLE_MU_BELOW) * sqrt(min(mu_s, 1.0));
}

// A sky table entry's local frame for a sun `sun` seen at `up`: x towards
// the sun's azimuth, y up, z = x cross y.
fn sky_table_frame(up: vec3<f32>, sun: vec3<f32>) -> mat3x3<f32> {
    var x = sun - up * dot(up, sun);
    if dot(x, x) < 1e-8 { x = cross(up, select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(up.x) > 0.9)); }
    x = normalize(x);
    return mat3x3<f32>(x, up, cross(x, up));
}

// The radiance table's texture coordinate (xy in the entry's map, z the
// slice's centre) of local direction `l` (y up, x towards the sun) at radius
// `r` in entry `entry` (`ir * SKY_TABLE_MU`, plus a fractional column).
fn sky_radiance_uvw(p: SkyMedium, r: f32, l: vec3<f32>, entry: f32) -> vec3<f32> {
    let u = sqrt(clamp(abs(atan2(l.z, l.x)) / SKY_PI, 0.0, 1.0));
    return vec3<f32>(u, sky_zenith_v(p, r, acos(clamp(l.y, -1.0, 1.0))),
        (entry + 0.5) / f32(SKY_TABLE_R * SKY_TABLE_MU));
}

// The direction (y up, x towards the sun) of texel coordinate `uv` of a
// radiance table entry's map at radius `r`: one side of the sun's vertical
// plane (an entry holds one sun, about whose plane its sky is symmetric),
// columns concentrated near the sun's azimuth, rows at the horizon.
fn sky_radiance_direction(p: SkyMedium, r: f32, uv: vec2<f32>) -> vec3<f32> {
    let zenith = sky_v_zenith(p, r, uv.y);
    let azimuth = uv.x * uv.x * SKY_PI;
    return vec3<f32>(sin(zenith) * cos(azimuth), cos(zenith), sin(zenith) * sin(azimuth));
}

const SKY_PI: f32 = 3.14159265358979;
const TRANSMITTANCE_SIZE: vec2<f32> = vec2<f32>(256.0, 64.0);
const MULTI_SCATTERING_SIZE: f32 = 32.0;
const SKY_VIEW_SIZE: vec2<f32> = vec2<f32>(256.0, 108.0);
const AERIAL_SLICES: f32 = 32.0;

struct Medium {
    rayleigh: vec3<f32>,
    mie: vec3<f32>,
    scattering: vec3<f32>,
    extinction: vec3<f32>,
}

fn medium_at(p: SkyMedium, height: f32) -> Medium {
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
    return 3.0 / (16.0 * SKY_PI) * (1.0 + cos_theta * cos_theta);
}

// Cornette-Shanks.
fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let k = 3.0 / (8.0 * SKY_PI) * (1.0 - g2) / (2.0 + g2);
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
    // The root nearer zero from the roots' product (|o|^2 - r^2, its sign
    // exact from |o| - r), not as the difference of two near-equal
    // distances: a point metres above an Earth-sized sphere (radii in km)
    // lost the near root's sign to rounding and saw through the ground.
    let l = length(o);
    let c = (l - r) * (l + r);
    if b < 0.0 {
        let far = s - b;
        return vec2<f32>(c / far, far);
    }
    let near = -b - s;
    return vec2<f32>(near, select(c / near, -b + s, near == 0.0));
}

// Bruneton's transmittance parameterisation: (r, mu) <-> uv.
fn transmittance_uv(p: SkyMedium, r: f32, mu: f32) -> vec2<f32> {
    let big_h = sqrt(max(p.top_radius * p.top_radius - p.bottom_radius * p.bottom_radius, 0.0));
    let rho = sqrt(max(r * r - p.bottom_radius * p.bottom_radius, 0.0));
    let disc = r * r * (mu * mu - 1.0) + p.top_radius * p.top_radius;
    let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    let d_min = p.top_radius - r;
    let d_max = rho + big_h;
    return vec2<f32>((d - d_min) / max(d_max - d_min, 1e-5), rho / max(big_h, 1e-5));
}

fn transmittance_r_mu(p: SkyMedium, uv: vec2<f32>) -> vec2<f32> {
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
fn hits_ground(p: SkyMedium, r: f32, mu: f32) -> bool {
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
fn horizon_zenith(p: SkyMedium, r: f32) -> f32 {
    let s = sqrt(max(r * r - p.bottom_radius * p.bottom_radius, 0.0)) / max(r, 1e-5);
    return SKY_PI - acos(clamp(s, -1.0, 1.0));
}

// The sky maps' rows: concentrated at the horizon (seen from radius `r`),
// above it from v 0 (the zenith) to 0.5, below it to 1 (the nadir).
fn sky_zenith_v(p: SkyMedium, r: f32, zenith: f32) -> f32 {
    let horizon = horizon_zenith(p, r);
    if zenith < horizon {
        return 0.5 * (1.0 - sqrt(max(1.0 - zenith / horizon, 0.0)));
    }
    return 0.5 + 0.5 * sqrt(clamp((zenith - horizon) / max(SKY_PI - horizon, 1e-5), 0.0, 1.0));
}

fn sky_v_zenith(p: SkyMedium, r: f32, v: f32) -> f32 {
    let horizon = horizon_zenith(p, r);
    if v < 0.5 {
        let c = 1.0 - 2.0 * v;
        return horizon * (1.0 - c * c);
    }
    let c = 2.0 * v - 1.0;
    return horizon + c * c * (SKY_PI - horizon);
}

// Sky-view LUT: rows concentrate at the horizon, columns near the first
// sun's azimuth, on both sides of it (u 0.5): with more than one sun the sky
// is not symmetric about the first one's vertical plane.
fn sky_view_uv(p: SkyMedium, r: f32, zenith: f32, azimuth: f32) -> vec2<f32> {
    let u = 0.5 + 0.5 * sign(azimuth) * sqrt(clamp(abs(azimuth) / SKY_PI, 0.0, 1.0));
    return vec2<f32>(u, sky_zenith_v(p, r, zenith));
}

fn sky_view_angles(p: SkyMedium, r: f32, uv: vec2<f32>) -> vec2<f32> {
    let c = 2.0 * uv.x - 1.0;
    return vec2<f32>(sky_v_zenith(p, r, uv.y), sign(c) * c * c * SKY_PI);
}

// Zenith and signed azimuth from the first sun of world direction `d` seen
// from the camera.
fn sky_view_direction_angles(frame: SkyFrame, d: vec3<f32>) -> vec2<f32> {
    let up = normalize(frame.eye.xyz);
    let basis = sky_view_basis(up, frame.sun.xyz);
    let local = transpose(basis) * d;
    return vec2<f32>(acos(clamp(local.y, -1.0, 1.0)), atan2(local.z, local.x));
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
fn sky_transmittance_towards(frame: SkyFrame, from_camera: vec3<f32>, towards: vec3<f32>) -> vec3<f32> {
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
fn sky_irradiance(frame: SkyFrame, n: vec3<f32>) -> vec3<f32> {
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

// The segment of a ray inside the atmosphere's shell: x start, y end, z 1
// when it ends on the planet's analytic ground. Only rays that meet no
// geometry (`to_ground`) end there: real terrain may lie below that sphere
// (valleys under sea level), and the air reaches down to it.
fn sky_segment(p: SkyMedium, o: vec3<f32>, d: vec3<f32>, to_ground: bool) -> vec3<f32> {
    let top = ray_sphere(o, d, p.top_radius);
    if top.y <= 0.0 { return vec3<f32>(0.0); }
    let start = max(top.x, 0.0);
    var end = top.y;
    var hit = 0.0;
    if to_ground {
        let ground = ray_sphere(o, d, p.bottom_radius);
        if ground.x > 0.0 && ground.x < end { end = ground.x; hit = 1.0; }
    }
    return vec3<f32>(start, max(end, start), hit);
}

// Transmittance to the sun from radius `r` at sun cosine `mu_s`, read from
// the transmittance LUT; zero behind the planet.
fn sky_sun_transmittance_lut(p: SkyMedium, lut: texture_2d<f32>, s: sampler, r: f32, mu_s: f32) -> vec3<f32> {
    if hits_ground(p, r, mu_s) { return vec3<f32>(0.0); }
    return textureSampleLevel(lut, s, transmittance_uv(p, r, mu_s), 0.0).rgb;
}

fn sky_multi_scattering_lut(p: SkyMedium, lut: texture_2d<f32>, s: sampler, r: f32, mu_s: f32) -> vec3<f32> {
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

fn sky_integrate(p: SkyMedium, sun: vec3<f32>, transmittance_lut: texture_2d<f32>,
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
        let lit = sky_sun_transmittance_lut(p, transmittance_lut, s, rx, mu_s);
        let source = (m.rayleigh * rayleigh_phase + m.mie * mie_phase) * lit
            + m.scattering * sky_multi_scattering_lut(p, multi_scattering_lut, s, rx, mu_s);
        let step_t = exp(-m.extinction * dt);
        let integral = (vec3<f32>(1.0) - step_t) / max(m.extinction, vec3<f32>(1e-7));
        out.radiance += out.transmittance * source * integral;
        out.transmittance *= step_t;
    }
    return out;
}

// View ray through `ndc` from the camera's rotation and perspective
// projection alone, never its position: a camera far from the world's origin
// (absolute frames on a planet) loses every bit of a direction rebuilt as
// `inv_view_proj * ndc - position`. World space, scaled to one unit of view
// depth (its length times the prelude's `helio_view_depth` is a distance).
fn sky_view_ray(view: mat4x4<f32>, proj: mat4x4<f32>, ndc: vec2<f32>) -> vec3<f32> {
    let v = vec3<f32>((ndc.x + proj[2][0]) / proj[0][0], (ndc.y + proj[2][1]) / proj[1][1], -1.0);
    let rotation = mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz);
    return transpose(rotation) * v;
}
