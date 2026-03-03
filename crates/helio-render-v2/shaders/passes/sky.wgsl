// Sky pass – Nishita single-scatter atmospheric model + FBM volumetric clouds
//
// Bind groups:
//   group(0)  binding(0)  Camera        (view_proj, position, time, view_proj_inv)
//   group(1)  binding(0)  SkyUniforms

// ──────────────────────────────────────────────────────────────────────────────
// Structs
// ──────────────────────────────────────────────────────────────────────────────

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,
}

// Layout must match SkyUniform in renderer.rs (112 bytes, 16-byte aligned)
struct SkyUniforms {
    sun_direction:     vec3<f32>,  // toward sun (normalised)
    sun_intensity:     f32,
    rayleigh_scatter:  vec3<f32>,  // per-wavelength (km⁻¹)
    rayleigh_h_scale:  f32,        // scale height / atm thickness
    mie_scatter:       f32,
    mie_h_scale:       f32,
    mie_g:             f32,        // HG asymmetry factor
    sun_disk_cos:      f32,        // cos(sun angular radius)
    earth_radius:      f32,        // km
    atm_radius:        f32,        // km
    exposure:          f32,
    clouds_enabled:    u32,
    cloud_coverage:    f32,
    cloud_density:     f32,
    cloud_base:        f32,        // world units
    cloud_top:         f32,
    cloud_wind_x:      f32,
    cloud_wind_z:      f32,
    cloud_speed:       f32,
    time_sky:          f32,        // elapsed time (seconds)
    skylight_intensity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

// ──────────────────────────────────────────────────────────────────────────────
// Bind groups
// ──────────────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> sky:    SkyUniforms;

// ──────────────────────────────────────────────────────────────────────────────
// Vertex shader – emit full-screen triangle covering the far plane
// ──────────────────────────────────────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       ndc_xy:        vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    // Three vertices that form a triangle covering [-1,1]² NDC space
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let xy = positions[vid];
    var out: VertexOutput;
    out.clip_position = vec4<f32>(xy, 1.0, 1.0);   // z=1 → far plane
    out.ndc_xy        = xy;
    return out;
}

// ──────────────────────────────────────────────────────────────────────────────
// Constants
// ──────────────────────────────────────────────────────────────────────────────

const PI: f32 = 3.14159265358979323846;

// Atmosphere integration samples
const ATMO_STEPS:  u32 = 16u;
const DEPTH_STEPS: u32 = 4u;

// Cloud ray-march steps
const CLOUD_STEPS: u32 = 32u;

// ──────────────────────────────────────────────────────────────────────────────
// Math helpers
// ──────────────────────────────────────────────────────────────────────────────

// Ray–sphere intersection. Returns (t_near, t_far). Both negative = miss.
fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> vec2<f32> {
    let b   = dot(ro, rd);
    let c   = dot(ro, ro) - r * r;
    let disc = b * b - c;
    if disc < 0.0 { return vec2<f32>(-1.0, -1.0); }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

// Rayleigh phase
fn phase_rayleigh(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

// Henyey-Greenstein Mie phase
fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (3.0 * (1.0 - g2)) / (8.0 * PI * (2.0 + g2)) *
           ((1.0 + cos_theta * cos_theta) / pow(max(denom, 1e-5), 1.5));
}

// Integrate optical depth (Rayleigh + Mie) along a ray of given length
fn optical_depth(ro: vec3<f32>, rd: vec3<f32>, ray_len: f32) -> vec2<f32> {
    var dr = 0.0;
    var dm = 0.0;
    let ds = ray_len / f32(DEPTH_STEPS);
    var t = ds * 0.5;
    for (var i = 0u; i < DEPTH_STEPS; i++) {
        let p = ro + rd * t;
        let h = max(length(p) - sky.earth_radius, 0.0);
        dr += exp(-h / (sky.atm_radius - sky.earth_radius) / sky.rayleigh_h_scale) * ds;
        dm += exp(-h / (sky.atm_radius - sky.earth_radius) / sky.mie_h_scale)      * ds;
        t  += ds;
    }
    return vec2<f32>(dr, dm);
}

// ──────────────────────────────────────────────────────────────────────────────
// Atmospheric scattering
// ──────────────────────────────────────────────────────────────────────────────

fn atmosphere(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    let atm_hit = ray_sphere(ro, rd, sky.atm_radius);
    if atm_hit.y < 0.0 { return vec3<f32>(0.0); }

    let t_start = max(atm_hit.x, 0.0);
    let t_end   = atm_hit.y;
    let seg_len = t_end - t_start;
    let ds      = seg_len / f32(ATMO_STEPS);

    let cos_theta = dot(rd, sky.sun_direction);
    let pr        = phase_rayleigh(cos_theta);
    let pm        = phase_mie(cos_theta, sky.mie_g);

    var scatter_r = vec3<f32>(0.0);
    var scatter_m = vec3<f32>(0.0);
    var t         = t_start + ds * 0.5;

    for (var i = 0u; i < ATMO_STEPS; i++) {
        let p = ro + rd * t;
        let h = max(length(p) - sky.earth_radius, 0.0);
        let atm_thickness = sky.atm_radius - sky.earth_radius;

        let density_r = exp(-h / (atm_thickness * sky.rayleigh_h_scale));
        let density_m = exp(-h / (atm_thickness * sky.mie_h_scale));

        // Is this point in Earth's shadow?
        let earth_hit = ray_sphere(p, sky.sun_direction, sky.earth_radius);
        if earth_hit.x < 0.0 || earth_hit.y < 0.0 {
            // Optical depth from camera to this point
            let depth_cam = optical_depth(ro, rd, t);
            // Optical depth from this point to the top of atmosphere toward sun
            let sun_atm = ray_sphere(p, sky.sun_direction, sky.atm_radius);
            let sun_dist = max(sun_atm.y, 0.0);
            let depth_sun = optical_depth(p, sky.sun_direction, sun_dist);

            let tau_r = sky.rayleigh_scatter * (depth_cam.x + depth_sun.x);
            let tau_m = sky.mie_scatter * 1.11 * (depth_cam.y + depth_sun.y);
            let transmit = exp(-(tau_r + vec3<f32>(tau_m)));

            scatter_r += density_r * transmit * ds;
            scatter_m += density_m * transmit * ds;
        }
        t += ds;
    }

    return sky.sun_intensity * (
        pr * sky.rayleigh_scatter * scatter_r +
        pm * sky.mie_scatter      * scatter_m
    );
}

// ──────────────────────────────────────────────────────────────────────────────
// Procedural cloud noise (FBM, no textures needed)
// ──────────────────────────────────────────────────────────────────────────────

fn hash3(p: vec3<f32>) -> f32 {
    var q = fract(p * 0.3183099 + 0.1);
    q *= 17.0;
    return fract(q.x * q.y * q.z * (q.x + q.y + q.z));
}

fn noise3(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(mix(hash3(i + vec3<f32>(0,0,0)), hash3(i + vec3<f32>(1,0,0)), u.x),
            mix(hash3(i + vec3<f32>(0,1,0)), hash3(i + vec3<f32>(1,1,0)), u.x), u.y),
        mix(mix(hash3(i + vec3<f32>(0,0,1)), hash3(i + vec3<f32>(1,0,1)), u.x),
            mix(hash3(i + vec3<f32>(0,1,1)), hash3(i + vec3<f32>(1,1,1)), u.x), u.y),
        u.z);
}

fn fbm(p: vec3<f32>) -> f32 {
    var v = 0.0;
    var a = 0.5;
    var q = p;
    for (var i = 0u; i < 5u; i++) {
        v += a * noise3(q);
        q  = q * 2.03 + vec3<f32>(31.1, 17.7, 43.3);
        a *= 0.5;
    }
    return v;
}

fn cloud_density(world_p: vec3<f32>) -> f32 {
    if world_p.y < sky.cloud_base || world_p.y > sky.cloud_top { return 0.0; }
    let height_frac = (world_p.y - sky.cloud_base) / (sky.cloud_top - sky.cloud_base);
    let vshape = smoothstep(0.0, 0.1, height_frac) * smoothstep(1.0, 0.6, height_frac);

    let wind = vec3<f32>(sky.cloud_wind_x, 0.0, sky.cloud_wind_z) * sky.cloud_speed * sky.time_sky;
    let sp   = (world_p + wind) * 0.0006;   // scale noise to world

    let base_noise = fbm(sp);
    let detail     = fbm(sp * 3.7 + vec3<f32>(5.2, 1.3, 2.7)) * 0.35;
    let combined   = base_noise + detail;

    let d = max(0.0, combined - (1.0 - sky.cloud_coverage)) * sky.cloud_density;
    return d * vshape;
}

// ──────────────────────────────────────────────────────────────────────────────
// Volumetric cloud ray-march
// Returns (inscattered colour, transmittance)
// ──────────────────────────────────────────────────────────────────────────────

fn trace_clouds(ro: vec3<f32>, rd: vec3<f32>, bg_col: vec3<f32>) -> vec3<f32> {
    if sky.clouds_enabled == 0u { return bg_col; }

    // Find entry/exit heights in cloud slab (approximate for near-horizontal rays)
    let base_h = sky.cloud_base;
    let top_h  = sky.cloud_top;

    // Entry: the t at which the ray reaches base_h from below (rd.y > 0) or top (rd.y < 0)
    var t0 = 0.0;
    var t1 = 0.0;
    if abs(rd.y) < 1e-4 {
        // Ray is nearly horizontal – check if camera is in the cloud band
        if ro.y >= base_h && ro.y <= top_h {
            t0 = 0.0;
            t1 = 10000.0;
        } else {
            return bg_col;
        }
    } else {
        let ta = (base_h - ro.y) / rd.y;
        let tb = (top_h  - ro.y) / rd.y;
        t0 = max(min(ta, tb), 0.0);
        t1 = max(ta, tb);
        if t1 < 0.0 { return bg_col; }
    }

    let seg   = min(t1 - t0, 20000.0);  // cap march length
    let step  = seg / f32(CLOUD_STEPS);
    var t     = t0 + step * 0.5;
    var transmit = 1.0;
    var accum    = vec3<f32>(0.0);

    // Simple cloud colour: sunlit white mixed with sky horizon
    let cloud_col = vec3<f32>(0.95, 0.96, 1.0);

    for (var i = 0u; i < CLOUD_STEPS; i++) {
        if transmit < 0.01 { break; }
        let p  = ro + rd * t;
        let dn = cloud_density(p);
        if dn > 0.0 {
            // Single-scattering approximation: Beer-Lambert + Henyey-Greenstein
            let sigma = dn * 0.05;
            let dT    = exp(-sigma * step);

            // Light march toward sun (cheap: 4 steps)
            var sun_od = 0.0;
            let sun_step = 500.0 / 4.0;
            for (var j = 0u; j < 4u; j++) {
                let sp = p + sky.sun_direction * (sun_step * (f32(j) + 0.5));
                sun_od += cloud_density(sp) * 0.05 * sun_step;
            }
            let sun_transmit = exp(-sun_od);

            let phase = phase_mie(dot(rd, sky.sun_direction), 0.6);
            let lit   = cloud_col * sky.sun_intensity * sun_transmit * phase * 0.15;

            accum    += transmit * (1.0 - dT) * lit;
            transmit *= dT;
        }
        t += step;
    }

    return bg_col * transmit + accum;
}

// ──────────────────────────────────────────────────────────────────────────────
// Tone mapping
// ──────────────────────────────────────────────────────────────────────────────

fn aces_approx(v: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((v * (a * v + b)) / (v * (c * v + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ──────────────────────────────────────────────────────────────────────────────
// Fragment shader
// ──────────────────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Reconstruct world-space ray direction from the inverse VP matrix
    let clip  = vec4<f32>(in.ndc_xy, 1.0, 1.0);
    let world = camera.view_proj_inv * clip;
    let ray_dir = normalize(world.xyz / world.w - camera.position);

    // Camera sits at the surface of the Earth in atmospheric coordinates
    let cam_atm = vec3<f32>(0.0, sky.earth_radius + 0.001, 0.0);

    // If the ray points into the Earth, output ground colour
    let earth_hit = ray_sphere(cam_atm, ray_dir, sky.earth_radius);
    if earth_hit.x > 0.0 && ray_dir.y < 0.0 {
        let ground = vec3<f32>(0.12, 0.10, 0.09);
        return vec4<f32>(aces_approx(ground * sky.exposure), 1.0);
    }

    // Atmosphere
    var sky_col = atmosphere(cam_atm, ray_dir);

    // Sun disc
    let cos_a = dot(ray_dir, sky.sun_direction);
    if cos_a > sky.sun_disk_cos {
        let t = smoothstep(sky.sun_disk_cos, sky.sun_disk_cos + 0.0002, cos_a);
        sky_col += t * vec3<f32>(1.5, 1.3, 0.9) * sky.sun_intensity * 0.1;
    }

    // Volumetric clouds (composited over atmosphere)
    sky_col = trace_clouds(camera.position, ray_dir, sky_col);

    // Tone map and gamma-correct
    var final_col = aces_approx(sky_col * sky.exposure);
    final_col = pow(final_col, vec3<f32>(1.0 / 2.2));

    return vec4<f32>(final_col, 1.0);
}
