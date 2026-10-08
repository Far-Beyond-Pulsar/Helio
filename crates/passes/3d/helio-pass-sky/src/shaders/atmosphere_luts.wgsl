//!use atmosphere
// Atmosphere LUT kernels (see atmosphere_common.wgsl). Every kernel reads
// the frame `resolve` wrote; all leave their outputs untouched while no
// atmosphere is active.

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

struct Resolve {
    // The frame's world origin (camera-relative frames), split into an f32
    // and its remainder so planet centres far from it stay exact.
    origin_hi: vec4<f32>,
    origin_lo: vec4<f32>,
}

@group(0) @binding(0) var<uniform> resolve_input: Resolve;
@group(0) @binding(1) var<storage, read> atmospheres: array<AtmosphereParams>;
// `scene_lights`: 128-byte rows (8 vec4s).
@group(0) @binding(2) var<storage, read> lights: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(4) var<storage, read_write> frame: AtmosphereFrame;
@group(0) @binding(5) var lut_sampler: sampler;

@group(1) @binding(0) var transmittance_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(1) var transmittance_lut: texture_2d<f32>;
@group(1) @binding(2) var multi_scattering_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(3) var multi_scattering_lut: texture_2d<f32>;
@group(1) @binding(4) var sky_view_out: texture_storage_2d<rgba16float, write>;
@group(1) @binding(5) var sky_view_lut: texture_2d<f32>;
@group(1) @binding(6) var aerial_out: texture_storage_3d<rgba16float, write>;

const LIGHT_ROW_VEC4S: u32 = 8u;

// Picks the first enabled atmosphere and the first directional light, and
// places the planet relative to the camera.
@compute @workgroup_size(1)
fn resolve() {
    frame.planet = vec4<f32>(0.0);
    var found = false;
    var params: AtmosphereParams;
    for (var i = 0u; i < arrayLength(&atmospheres); i++) {
        let a = atmospheres[i];
        if a.enabled != 0u && a.top_radius > a.bottom_radius && a.bottom_radius > 0.0 {
            params = a;
            found = true;
            break;
        }
    }
    if !found { return; }
    // Sun: the first directional light with any intensity (direction_outer
    // is the direction the light travels).
    var sun = vec3<f32>(0.0, 1.0, 0.0);
    var illuminance = vec3<f32>(0.0);
    let rows = arrayLength(&lights) / LIGHT_ROW_VEC4S;
    for (var i = 0u; i < rows; i++) {
        let base = i * LIGHT_ROW_VEC4S;
        let colour = lights[base + 2u];
        let light_type = bitcast<u32>(lights[base + 3u].y);
        if light_type == 0u && colour.w > 0.0 {
            sun = -normalize(lights[base + 1u].xyz);
            illuminance = colour.rgb * colour.w;
            break;
        }
    }
    var center = params.center;
    if params.placement == 0u { center = vec3<f32>(0.0, -params.bottom_radius * 1000.0, 0.0); }
    let camera = cameras[0].position_near.xyz;
    // Metres; the large terms cancel first.
    let relative = (center - resolve_input.origin_hi.xyz) - resolve_input.origin_lo.xyz - camera;
    let planet = relative * 0.001;
    let eye = -planet;
    let r = length(eye);
    // Aerial perspective reaches the horizon (and the far side of the
    // atmosphere's shell seen past it), at least 32 km.
    let horizon = sqrt(max(r * r - params.bottom_radius * params.bottom_radius, 0.0));
    let shell = sqrt(max(params.top_radius * params.top_radius - params.bottom_radius * params.bottom_radius, 0.0));
    let range = clamp(horizon + shell, 32.0, 4000.0);
    frame.planet = vec4<f32>(planet, 1.0);
    frame.sun = vec4<f32>(sun, cos(params.sun_angular_radius));
    frame.sun_illuminance = vec4<f32>(illuminance, range);
    frame.eye = vec4<f32>(eye, 0.0);
    frame.params = params;
}

fn atmosphere_active() -> bool {
    return frame.planet.w > 0.5;
}

// The shared integrators, bound to this pass's LUTs.
fn sun_transmittance(r: f32, mu_s: f32) -> vec3<f32> {
    return atmosphere_sun_transmittance_lut(frame.params, transmittance_lut, lut_sampler, r, mu_s);
}

fn integrate(o: vec3<f32>, d: vec3<f32>, t0: f32, t1: f32, steps: f32) -> Scattering {
    return atmosphere_integrate(frame.params, frame.sun.xyz, transmittance_lut, multi_scattering_lut,
        lut_sampler, o, d, t0, t1, steps, true);
}

@compute @workgroup_size(8, 8)
fn transmittance_lut_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    if !atmosphere_active() || any(vec2<f32>(id.xy) >= TRANSMITTANCE_SIZE) { return; }
    let p = frame.params;
    let uv = (vec2<f32>(id.xy) + 0.5) / TRANSMITTANCE_SIZE;
    let r_mu = transmittance_r_mu(p, uv);
    let o = vec3<f32>(0.0, r_mu.x, 0.0);
    let d = vec3<f32>(sqrt(max(1.0 - r_mu.y * r_mu.y, 0.0)), r_mu.y, 0.0);
    let length_top = max(ray_sphere(o, d, p.top_radius).y, 0.0);
    let steps = 40.0;
    let dt = length_top / steps;
    var depth = vec3<f32>(0.0);
    for (var i = 0.0; i < steps; i += 1.0) {
        let x = o + d * (i + 0.5) * dt;
        depth += medium_at(p, length(x) - p.bottom_radius).extinction * dt;
    }
    textureStore(transmittance_out, vec2<i32>(id.xy), vec4<f32>(exp(-depth), 1.0));
}

// Second-order scattering transfer (Hillaire's psi_ms): the light reaching
// a point after any number of isotropic bounces, for a unit sun.
@compute @workgroup_size(8, 8)
fn multi_scattering_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    if !atmosphere_active() || any(vec2<f32>(id.xy) >= vec2<f32>(MULTI_SCATTERING_SIZE)) { return; }
    let p = frame.params;
    let uv = (vec2<f32>(id.xy) + 0.5) / MULTI_SCATTERING_SIZE;
    let mu_s = uv.x * 2.0 - 1.0;
    let r = p.bottom_radius + clamp(uv.y, 0.002, 0.998) * (p.top_radius - p.bottom_radius);
    let o = vec3<f32>(0.0, r, 0.0);
    let sun = vec3<f32>(sqrt(max(1.0 - mu_s * mu_s, 0.0)), mu_s, 0.0);
    let directions = 64u;
    var second = vec3<f32>(0.0);
    var transfer = vec3<f32>(0.0);
    for (var n = 0u; n < directions; n++) {
        // Fibonacci sphere.
        let z = 1.0 - (f32(n) + 0.5) * 2.0 / f32(directions);
        let a = f32(n) * 2.39996323;
        let s = sqrt(max(1.0 - z * z, 0.0));
        let d = vec3<f32>(s * cos(a), z, s * sin(a));
        let ground = ray_sphere(o, d, p.bottom_radius);
        let top = ray_sphere(o, d, p.top_radius);
        var t_max = max(top.y, 0.0);
        let hit = ground.x > 0.0;
        if hit { t_max = ground.x; }
        let steps = 20.0;
        let dt = t_max / steps;
        var throughput = vec3<f32>(1.0);
        var light = vec3<f32>(0.0);
        var bounce = vec3<f32>(0.0);
        for (var i = 0.0; i < steps; i += 1.0) {
            let x = o + d * (i + 0.5) * dt;
            let rx = length(x);
            let m = medium_at(p, rx - p.bottom_radius);
            let step_t = exp(-m.extinction * dt);
            let integral = (vec3<f32>(1.0) - step_t) / max(m.extinction, vec3<f32>(1e-7));
            let lit = sun_transmittance(rx, dot(x / rx, sun));
            light += throughput * m.scattering * lit / (4.0 * ATMOSPHERE_PI) * integral;
            bounce += throughput * m.scattering * integral;
            throughput *= step_t;
        }
        if hit {
            let x = o + d * t_max;
            let n_ground = normalize(x);
            let lit = sun_transmittance(p.bottom_radius, dot(n_ground, sun));
            light += throughput * lit * max(dot(n_ground, sun), 0.0) * p.ground_albedo / ATMOSPHERE_PI;
        }
        // Uniform phase over the sphere: each direction weighs 1 / count.
        second += light / f32(directions);
        transfer += bounce / f32(directions);
    }
    let psi = second / max(vec3<f32>(1.0) - transfer, vec3<f32>(1e-3));
    textureStore(multi_scattering_out, vec2<i32>(id.xy), vec4<f32>(psi, 1.0));
}

@compute @workgroup_size(8, 8)
fn sky_view_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    if !atmosphere_active() || any(vec2<f32>(id.xy) >= SKY_VIEW_SIZE) { return; }
    let p = frame.params;
    let eye = frame.eye.xyz;
    let r = length(eye);
    let uv = (vec2<f32>(id.xy) + 0.5) / SKY_VIEW_SIZE;
    let angles = sky_view_angles(p, r, uv);
    let basis = sky_view_basis(eye / r, frame.sun.xyz);
    let d = basis * vec3<f32>(sin(angles.x) * cos(angles.y), cos(angles.x), sin(angles.x) * sin(angles.y));
    let segment = atmosphere_segment(frame.params, eye, d, true);
    var radiance = vec3<f32>(0.0);
    if segment.y > segment.x {
        let s = integrate(eye, d, segment.x, segment.y, 32.0);
        radiance = s.radiance;
        if segment.z > 0.5 {
            // The ground seen through the atmosphere (where no geometry
            // draws it): sunlit Lambertian albedo.
            let x = eye + d * segment.y;
            let n = normalize(x);
            let mu_s = dot(n, frame.sun.xyz);
            radiance += s.transmittance * sun_transmittance(p.bottom_radius, mu_s) * max(mu_s, 0.0) * p.ground_albedo / ATMOSPHERE_PI;
        }
    }
    textureStore(sky_view_out, vec2<i32>(id.xy), vec4<f32>(radiance, 1.0));
}

// Aerial perspective: per froxel column, in-scattered radiance (unit sun)
// and mean transmittance from the camera to each slice's distance.
@compute @workgroup_size(8, 8)
fn aerial_kernel(@builtin(global_invocation_id) id: vec3<u32>) {
    if !atmosphere_active() || any(vec2<f32>(id.xy) >= vec2<f32>(AERIAL_SLICES)) { return; }
    let camera = cameras[0];
    let uv = (vec2<f32>(id.xy) + 0.5) / AERIAL_SLICES;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    let world = camera.inv_view_proj * ndc;
    let d = normalize(world.xyz / world.w - camera.position_near.xyz);
    let eye = frame.eye.xyz;
    let range = frame.sun_illuminance.w;
    // Geometry the volume is sampled for may lie below the analytic ground.
    let segment = atmosphere_segment(frame.params, eye, d, false);
    var radiance = vec3<f32>(0.0);
    var throughput = vec3<f32>(1.0);
    var t_prev = 0.0;
    for (var k = 0u; k < u32(AERIAL_SLICES); k++) {
        let t = aerial_distance((f32(k) + 0.5) / AERIAL_SLICES, range);
        // Only the part of [t_prev, t] inside the atmosphere scatters.
        let a = clamp(t_prev, segment.x, max(segment.y, segment.x));
        let b = clamp(t, segment.x, max(segment.y, segment.x));
        if b > a {
            let s = integrate(eye, d, a, b, 4.0);
            radiance += throughput * s.radiance;
            throughput *= s.transmittance;
        }
        t_prev = t;
        let mean = dot(throughput, vec3<f32>(1.0 / 3.0));
        textureStore(aerial_out, vec3<i32>(vec2<i32>(id.xy), i32(k)), vec4<f32>(radiance, mean));
    }
}

// Sky irradiance (diffuse ambient) projected onto L2 spherical harmonics
// from the sky-view LUT, convolved with the cosine lobe and divided by pi.
var<workgroup> partial: array<array<vec3<f32>, 9>, 64>;

fn sky_radiance(d: vec3<f32>) -> vec3<f32> {
    let r = length(frame.eye.xyz);
    let angles = sky_view_direction_angles(frame, d);
    let uv = sky_view_uv(frame.params, r, angles.x, angles.y);
    return textureSampleLevel(sky_view_lut, lut_sampler, uv, 0.0).rgb;
}

@compute @workgroup_size(64)
fn irradiance_kernel(@builtin(local_invocation_index) li: u32) {
    // No early return: the barrier below needs uniform control flow.
    let samples = 16u;
    let total = 64u * samples;
    var sh: array<vec3<f32>, 9>;
    for (var c = 0u; c < 9u; c++) { sh[c] = vec3<f32>(0.0); }
    for (var s = 0u; s < samples; s++) {
        let n = li * samples + s;
        let z = 1.0 - (f32(n) + 0.5) * 2.0 / f32(total);
        let a = f32(n) * 2.39996323;
        let q = sqrt(max(1.0 - z * z, 0.0));
        let d = vec3<f32>(q * cos(a), z, q * sin(a));
        let l = sky_radiance(d);
        sh[0] += l * 0.282095;
        sh[1] += l * 0.488603 * d.y;
        sh[2] += l * 0.488603 * d.z;
        sh[3] += l * 0.488603 * d.x;
        sh[4] += l * 1.092548 * d.x * d.y;
        sh[5] += l * 1.092548 * d.y * d.z;
        sh[6] += l * 0.315392 * (3.0 * d.z * d.z - 1.0);
        sh[7] += l * 1.092548 * d.x * d.z;
        sh[8] += l * 0.546274 * (d.x * d.x - d.y * d.y);
    }
    partial[li] = sh;
    workgroupBarrier();
    if li != 0u || !atmosphere_active() { return; }
    let weight = 4.0 * ATMOSPHERE_PI / f32(total);
    // Cosine-lobe convolution (Ramamoorthi & Hanrahan), then / pi.
    let band = array<f32, 9>(1.0, 2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0, 0.25, 0.25, 0.25, 0.25, 0.25);
    for (var c = 0u; c < 9u; c++) {
        var sum = vec3<f32>(0.0);
        for (var t = 0u; t < 64u; t++) { sum += partial[t][c]; }
        frame.irradiance[c] = vec4<f32>(sum * weight * band[c] * frame.sun_illuminance.rgb, 0.0);
    }
}
