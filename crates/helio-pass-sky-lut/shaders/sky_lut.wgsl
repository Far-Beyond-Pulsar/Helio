// sky_lut.wgsl – Sky-View LUT generation pass (Hillaire 2020)
//
// Renders Nishita single-scatter atmosphere into a 192×108 Rgba16Float panoramic
// texture.  The main SkyPass samples this LUT instead of running the atmosphere
// ray-march per screen-pixel, giving ~46× cost reduction at 1280×720.
//
// Panoramic layout:
//   u = azimuth / (2π) + 0.5            ∈ [0, 1]   (wraps)
//   v = sin(elevation) * 0.5 + 0.5      ∈ [0, 1]   (sin-mapping, better horizon res)

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,
}

struct SkyUniforms {
    sun_direction:     vec3<f32>,
    sun_intensity:     f32,
    rayleigh_scatter:  vec3<f32>,
    rayleigh_h_scale:  f32,
    mie_scatter:       f32,
    mie_h_scale:       f32,
    mie_g:             f32,
    sun_disk_cos:      f32,
    earth_radius:      f32,
    atm_radius:        f32,
    exposure:          f32,
    clouds_enabled:    u32,
    cloud_coverage:    f32,
    cloud_density:     f32,
    cloud_base:        f32,
    cloud_top:         f32,
    cloud_wind_x:      f32,
    cloud_wind_z:      f32,
    cloud_speed:       f32,
    time_sky:          f32,
    skylight_intensity: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var<uniform> sky:    SkyUniforms;

// ── Vertex: full-screen triangle ─────────────────────────────────────────────

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       uv:       vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VertexOutput {
    let pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    let xy = pos[vid];
    var out: VertexOutput;
    out.clip_pos = vec4<f32>(xy, 0.0, 1.0);
    out.uv       = xy * 0.5 + 0.5; // [0,1]
    return out;
}

// ── Constants & helpers ────────────────────────────────────────────────────────

const PI:          f32 = 3.14159265358979;
const ATMO_STEPS:  u32 = 16u;
const DEPTH_STEPS: u32 = 4u;

fn ray_sphere(ro: vec3<f32>, rd: vec3<f32>, r: f32) -> vec2<f32> {
    let b    = dot(ro, rd);
    let c    = dot(ro, ro) - r * r;
    let disc = b * b - c;
    if disc < 0.0 { return vec2<f32>(-1.0, -1.0); }
    let s = sqrt(disc);
    return vec2<f32>(-b - s, -b + s);
}

fn phase_rayleigh(cos_theta: f32) -> f32 {
    return (3.0 / (16.0 * PI)) * (1.0 + cos_theta * cos_theta);
}

fn phase_mie(cos_theta: f32, g: f32) -> f32 {
    let g2    = g * g;
    let denom = 1.0 + g2 - 2.0 * g * cos_theta;
    return (3.0 * (1.0 - g2)) / (8.0 * PI * (2.0 + g2))
         * ((1.0 + cos_theta * cos_theta) / pow(max(denom, 1e-5), 1.5));
}

fn optical_depth(ro: vec3<f32>, rd: vec3<f32>, ray_len: f32) -> vec2<f32> {
    var dr = 0.0; var dm = 0.0;
    let ds = ray_len / f32(DEPTH_STEPS);
    var t  = ds * 0.5;
    for (var i = 0u; i < DEPTH_STEPS; i++) {
        let p  = ro + rd * t;
        let h  = max(length(p) - sky.earth_radius, 0.0);
        let th = sky.atm_radius - sky.earth_radius;
        dr += exp(-h / (th * sky.rayleigh_h_scale)) * ds;
        dm += exp(-h / (th * sky.mie_h_scale))      * ds;
        t  += ds;
    }
    return vec2<f32>(dr, dm);
}

fn atmosphere(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    let atm_hit = ray_sphere(ro, rd, sky.atm_radius);
    if atm_hit.y < 0.0 { return vec3<f32>(0.0); }

    let t_start   = max(atm_hit.x, 0.0);
    let seg_len   = atm_hit.y - t_start;
    let ds        = seg_len / f32(ATMO_STEPS);
    let cos_theta = dot(rd, sky.sun_direction);
    let pr        = phase_rayleigh(cos_theta);
    let pm        = phase_mie(cos_theta, sky.mie_g);

    var scatter_r = vec3<f32>(0.0);
    var scatter_m = vec3<f32>(0.0);
    var t         = t_start + ds * 0.5;

    for (var i = 0u; i < ATMO_STEPS; i++) {
        let p  = ro + rd * t;
        let h  = max(length(p) - sky.earth_radius, 0.0);
        let th = sky.atm_radius - sky.earth_radius;

        let density_r = exp(-h / (th * sky.rayleigh_h_scale));
        let density_m = exp(-h / (th * sky.mie_h_scale));

        let earth_hit = ray_sphere(p, sky.sun_direction, sky.earth_radius);
        if earth_hit.x < 0.0 || earth_hit.y < 0.0 {
            let depth_cam = optical_depth(ro, rd, t);
            let sun_atm   = ray_sphere(p, sky.sun_direction, sky.atm_radius);
            let depth_sun = optical_depth(p, sky.sun_direction, max(sun_atm.y, 0.0));
            let tau_r     = sky.rayleigh_scatter * (depth_cam.x + depth_sun.x);
            let tau_m     = sky.mie_scatter * 1.11 * (depth_cam.y + depth_sun.y);
            let transmit  = exp(-(tau_r + vec3<f32>(tau_m)));
            scatter_r    += density_r * transmit * ds;
            scatter_m    += density_m * transmit * ds;
        }
        t += ds;
    }

    return sky.sun_intensity * (
        pr * sky.rayleigh_scatter * scatter_r +
        pm * sky.mie_scatter      * scatter_m
    );
}

// ── Fragment: one LUT texel = one sky direction ────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = in.uv; // [0,1]²

    // Decode direction from panoramic UV
    //   u → azimuth: 0..1 maps to -π..π
    //   v → elevation via inverse-sin: 0..1 maps to -π/2..π/2
    let azimuth   = (uv.x - 0.5) * 2.0 * PI;
    let sin_elev  = uv.y * 2.0 - 1.0;        // [-1, 1]
    let cos_elev  = sqrt(max(1.0 - sin_elev * sin_elev, 0.0));
    let ray_dir   = vec3<f32>(
        cos_elev * cos(azimuth),
        sin_elev,
        cos_elev * sin(azimuth),
    );

    let cam_atm = vec3<f32>(0.0, sky.earth_radius + 0.001, 0.0);

    // Below horizon: keep colour from horizon moving smoothly to night.
    // This ensures the whole lower hemisphere keeps sunset gradation.
    var out_col = atmosphere(cam_atm, ray_dir);
    if sin_elev < 0.0 {
        let horizon_dir = vec3<f32>(cos(azimuth), 0.0, sin(azimuth));
        let horizon_col = atmosphere(cam_atm, horizon_dir);
        let falloff = clamp(-sin_elev, 0.0, 1.0);
        let night = vec3<f32>(0.01, 0.005, 0.002);
        let shifted = mix(horizon_col, night, pow(falloff, 1.8));
        out_col = mix(out_col, shifted, 0.4);
    }

    // Store pre-exposed HDR value; main pass tone-maps on read
    return vec4<f32>(out_col, 1.0);
}
