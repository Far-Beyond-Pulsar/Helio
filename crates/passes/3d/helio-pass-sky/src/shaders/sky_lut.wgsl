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
    planet_observer: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(1) @binding(0) var<storage, read> sky_rows: array<SkyUniforms>;
var<private> sky: SkyUniforms;

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

// Concentrate lookup rows around the geometric atmospheric limb. A uniform
// global-Y panorama magnifies a thin orbital atmosphere into a broad halo.
fn planet_sky_basis()->mat3x3<f32> {
    let up=normalize(sky.planet_observer.xyz);
    let reference=select(vec3<f32>(0.0,1.0,0.0),vec3<f32>(0.0,0.0,1.0),abs(up.y)>0.99);
    let east=normalize(cross(reference,up));
    return mat3x3<f32>(east,up,cross(east,up));
}
fn planet_horizon_mu()->f32 {
    let ratio=min(1.0,sky.earth_radius/max(length(sky.planet_observer.xyz),sky.earth_radius));
    return -sqrt(max(0.0,1.0-ratio*ratio));
}
fn planet_lut_direction(uv_up:vec2<f32>)->vec3<f32> {
    let horizon=planet_horizon_mu();let q=uv_up.y*2.0-1.0;
    // Preserve exact polar endpoints; subtracting horizon can leave a one-ulp
    // residual that expands into a visible angular error through sqrt(1-mu^2).
    let mapped_mu=horizon+select(-(horizon+1.0),1.0-horizon,q>=0.0)*q*q;
    let mu=select(mapped_mu,sign(q),abs(q)>=1.0);
    let azimuth=(uv_up.x-0.5)*2.0*PI;
    let tangent=sqrt(max(0.0,1.0-mu*mu));
    return planet_sky_basis()*vec3<f32>(tangent*cos(azimuth),mu,tangent*sin(azimuth));
}
fn planet_lut_uv(rd:vec3<f32>)->vec2<f32> {
    let local=transpose(planet_sky_basis())*rd;
    let horizon=planet_horizon_mu();let delta=clamp(local.y,-1.0,1.0)-horizon;
    let span=select(horizon+1.0,1.0-horizon,delta>=0.0);
    let mapped_q=select(-1.0,1.0,delta>=0.0)*sqrt(clamp(abs(delta)/max(span,1e-8),0.0,1.0));
    let q=select(mapped_q,sign(local.y),local.x==0.0 && local.z==0.0);
    return vec2<f32>(atan2(local.z,local.x)/(2.0*PI)+0.5,0.5-q*0.5);
}

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
    var t_end = atm_hit.y;
    let surface_hit = ray_sphere(ro, rd, sky.earth_radius);
    if surface_hit.x > 0.0 { t_end = min(t_end, surface_hit.x); }
    let seg_len   = max(0.0, t_end - t_start);
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
            // Integrate only the atmospheric segment, excluding orbital vacuum.
            let depth_cam = optical_depth(ro + rd * t_start, rd, t - t_start);
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
fn atmosphere_observer()->vec3<f32> {
    return select(vec3<f32>(0.0, sky.earth_radius + 0.001, 0.0),sky.planet_observer.xyz,sky.planet_observer.w>0.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    sky = sky_rows[0];
    let uv = in.uv; // [0,1]²

    // Decode direction from panoramic UV
    //   u → azimuth: 0..1 maps to -π..π
    //   v → elevation via inverse-sin: 0..1 maps to -π/2..π/2
    let azimuth   = (uv.x - 0.5) * 2.0 * PI;
    let sin_elev  = uv.y * 2.0 - 1.0;        // [-1, 1]
    let cos_elev  = sqrt(max(1.0 - sin_elev * sin_elev, 0.0));
    var ray_dir   = vec3<f32>(
        cos_elev * cos(azimuth),
        sin_elev,
        cos_elev * sin(azimuth),
    );

    if sky.planet_observer.w>0.0 {ray_dir=planet_lut_direction(uv);}
    let cam_atm = atmosphere_observer();

    // Below horizon: keep colour from horizon moving smoothly to night.
    // This ensures the whole lower hemisphere keeps sunset gradation.
    var out_col = atmosphere(cam_atm, ray_dir);
    if sky.planet_observer.w==0.0 && sin_elev < 0.0 {
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
