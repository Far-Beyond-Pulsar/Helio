//!use atmosphere
//!use gi_common
//!use gi_sample
// Probe updates (README.md): the sun found, last frame's rays integrated
// into their probes, probes that came into the clipmap reset, this frame's
// rays written for the tracers.

@group(0) @binding(0) var<uniform> frame_gi: GiFrame;
@group(0) @binding(1) var<storage, read_write> gi_probes: array<u32>;
@group(0) @binding(2) var<storage, read_write> gi_rays: array<GiRay>;
@group(0) @binding(3) var<storage, read> gi_hits: array<GiHit>;
// Two update lists (GI_MAX_UPDATES probes each: this frame's and last
// frame's, by `counts.w`), then the probes to reset.
@group(0) @binding(4) var<storage, read> gi_lists: array<u32>;
// The sun: towards it (xyz), its illuminance at the eye (rgb).
@group(0) @binding(5) var<storage, read_write> gi_sun: array<vec4<f32>, 2>;
// Helio's lights (`GpuLight`, 128 bytes).
struct SceneLight {
    position_range: vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index: u32,
    light_type: u32,
    inner_angle: f32,
    flags: u32,
    rest: array<vec4<u32>, 4>,
}
@group(0) @binding(6) var<storage, read> scene_lights: array<SceneLight>;
@group(0) @binding(7) var<uniform> atmosphere: AtmosphereFrame;
@group(0) @binding(8) var<uniform> atmosphere_sky_table: array<vec4<f32>, 2304>;

const PI: f32 = 3.14159265;

// The sun: the atmosphere's, through the air to the eye; without one, the
// scene's first directional light.
@compute @workgroup_size(1)
fn gi_sun_main() {
    if atmosphere.planet.w > 0.5 {
        let towards = atmosphere.sun.xyz;
        gi_sun[0] = vec4<f32>(towards, 0.0);
        gi_sun[1] = vec4<f32>(atmosphere.sun_illuminance.rgb * atmosphere_transmittance_towards(atmosphere, vec3<f32>(0.0), towards), 0.0);
        return;
    }
    gi_sun[0] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    gi_sun[1] = vec4<f32>(0.0);
    for (var i = 0u; i < arrayLength(&scene_lights); i++) {
        let light = scene_lights[i];
        if light.light_type == 0u && light.color_intensity.w > 0.0 && dot(light.direction_outer.xyz, light.direction_outer.xyz) > 0.0 {
            gi_sun[0] = vec4<f32>(-normalize(light.direction_outer.xyz), 0.0);
            gi_sun[1] = vec4<f32>(light.color_intensity.rgb * light.color_intensity.w, 0.0);
            return;
        }
    }
}

// The sky's radiance along unit direction `d` from eye-relative point
// `from_camera` (km): the atmosphere's sky table there (irradiance / pi in
// L2 harmonics, cosine-convolved) back to radiance, band by band; without
// an atmosphere, the flat sky and ground.
fn sky_radiance(from_camera: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if atmosphere.planet.w < 0.5 { return select(frame_gi.ground.rgb, frame_gi.sky.rgb, d.y >= 0.0); }
    let bottom = atmosphere.params.bottom_radius;
    let top = atmosphere.params.top_radius;
    let pos = from_camera - atmosphere.planet.xyz;
    let r = max(length(pos), bottom);
    let up = normalize(pos);
    let sun = atmosphere.sun.xyz;
    let mu_s = dot(up, sun);
    var x = sun - up * mu_s;
    if dot(x, x) < 1e-8 { x = cross(up, select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 1.0), abs(up.x) > 0.9)); }
    x = normalize(x);
    let z = cross(x, up);
    let l = vec3<f32>(dot(d, x), dot(d, up), dot(d, z));
    let height = clamp((r - bottom) / (top - bottom), 0.0, 1.0);
    let fr = sqrt(height) * f32(SKY_TABLE_R - 1u);
    let fm = clamp((mu_s - SKY_TABLE_MU_LO) / (1.0 - SKY_TABLE_MU_LO), 0.0, 1.0) * f32(SKY_TABLE_MU - 1u);
    let r0 = min(u32(fr), SKY_TABLE_R - 2u);
    let m0 = min(u32(fm), SKY_TABLE_MU - 2u);
    let w = vec2<f32>(fr - f32(r0), fm - f32(m0));
    // The basis over each band's convolution (1, 2/3, 1/4).
    let basis = array<f32, 6>(0.282095, 0.488603 * l.y * 1.5, 0.488603 * l.x * 1.5, 1.092548 * l.x * l.y * 4.0, 0.315392 * (3.0 * l.z * l.z - 1.0) * 4.0, 0.546274 * (l.x * l.x - l.y * l.y) * 4.0);
    var e = vec3<f32>(0.0);
    for (var k = 0u; k < SKY_TABLE_COEFFS; k++) {
        let a = (r0 * SKY_TABLE_MU + m0) * SKY_TABLE_COEFFS + k;
        let b = a + SKY_TABLE_COEFFS;
        let c = a + SKY_TABLE_MU * SKY_TABLE_COEFFS;
        let dd = c + SKY_TABLE_COEFFS;
        let v = mix(mix(atmosphere_sky_table[a].rgb, atmosphere_sky_table[b].rgb, w.y), mix(atmosphere_sky_table[c].rgb, atmosphere_sky_table[dd].rgb, w.y), w.x);
        e += v * basis[k];
    }
    return max(e, vec3<f32>(0.0)) * atmosphere.sun_illuminance.rgb;
}

// Probes that came into the clipmap (after `gi_integrate`, which may have
// written the probe that was there): no light, no distances, no updates.
@compute @workgroup_size(64)
fn gi_reset(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame_gi.counts.z { return; }
    let at = gi_lists[2u * GI_MAX_UPDATES + id.x] * GI_PROBE_WORDS;
    for (var w = 0u; w < GI_PROBE_WORDS; w++) { gi_probes[at + w] = 0u; }
    gi_probes[at + GI_STATE_WORD] = GI_ACTIVE;
}

var<workgroup> w_radiance: array<vec3<f32>, 64>;
var<workgroup> w_dir: array<vec3<f32>, 64>;
var<workgroup> w_t: array<f32, 64>;
var<workgroup> w_inside: atomic<u32>;

// Last frame's probes, a workgroup each, a lane a ray: each ray's radiance
// (the sky, or its hit's: albedo times the sun's light and the probes'
// there; nothing from a back face), projected on L1 harmonics and blended
// into the probe; its distances into the probe's moments (a lane a texel).
// A probe whose rays mostly meet back faces is inside the ground.
@compute @workgroup_size(64)
fn gi_integrate(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if wg.x >= frame_gi.counts.y { return; }
    let index = gi_lists[(1u - frame_gi.counts.w) * GI_MAX_UPDATES + wg.x];
    let probe = gi_probe_position(index);
    let spacing = frame_gi.levels[gi_probe_cell(index).w].origin.w;
    // Moments past a few spacings tell nothing more.
    let far = 3.5 * spacing;
    let dir = gi_ray_dir(li, frame_gi.prev_rotation);
    let hit = gi_hits[wg.x * GI_RAYS + li];
    if li == 0u { atomicStore(&w_inside, 0u); }
    workgroupBarrier();
    var radiance = vec3<f32>(0.0);
    var t = far;
    if hit.t > 0.0 {
        t = min(hit.t, far);
        let n = gi_unpack_unit(hit.normal);
        if dot(n, dir) > 0.0 {
            atomicAdd(&w_inside, 1u);
        } else {
            // Lambertian on the side the ray meets; a surface that lets
            // light through adds that share of its far side's light.
            let albedo = unpack4x8unorm(hit.albedo);
            let at = probe + dir * hit.t;
            let front = gi_sun[1].rgb * albedo.a * max(dot(n, gi_sun[0].xyz), 0.0) / PI + gi_irradiance(at, n).rgb;
            let through = f32((hit.light >> 8u) & 255u) / 255.0;
            var back = vec3<f32>(0.0);
            if through > 0.0 {
                let seen = f32((hit.light >> 16u) & 255u) / 255.0;
                back = gi_sun[1].rgb * seen * max(dot(-n, gi_sun[0].xyz), 0.0) / PI + gi_irradiance(at, -n).rgb;
            }
            // And the light it gives off.
            let glow = f32(hit.light >> 24u) / 16.0;
            radiance = albedo.rgb * (front + through * back + glow);
        }
    } else {
        radiance = sky_radiance(probe * 0.001, dir);
    }
    w_radiance[li] = radiance;
    w_dir[li] = dir;
    w_t[li] = t;
    workgroupBarrier();
    let at = index * GI_PROBE_WORDS;
    let state = gi_probes[at + GI_STATE_WORD];
    let updates = state >> 8u;
    // Moments: a lane a texel, its nearby rays' distances.
    {
        let texel = vec2<f32>(f32(li % GI_MOMENTS_SIDE), f32(li / GI_MOMENTS_SIDE));
        let d = gi_oct_decode((texel + 0.5) / f32(GI_MOMENTS_SIDE));
        var sum = vec2<f32>(0.0);
        var weights = 0.0;
        for (var k = 0u; k < GI_RAYS; k++) {
            let w = pow(max(dot(d, w_dir[k]), 0.0), 16.0);
            sum += vec2<f32>(w_t[k], w_t[k] * w_t[k]) * w;
            weights += w;
        }
        if weights > 1e-6 {
            let fresh = sum / weights;
            let old = gi_half2(gi_probes[at + GI_MOMENTS_WORD + li]);
            let alpha = select(max(1.0 / f32(updates + 1u), 0.05), 1.0, updates == 0u);
            gi_probes[at + GI_MOMENTS_WORD + li] = pack2x16float(mix(old, fresh, alpha));
        }
    }
    if li != 0u { return; }
    // Irradiance / pi: radiance on L1 harmonics, convolved (bands 1, 2/3).
    var sh = array<vec3<f32>, 4>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    for (var k = 0u; k < GI_RAYS; k++) {
        let b = gi_sh_basis(w_dir[k]);
        sh[0] += w_radiance[k] * b.x;
        sh[1] += w_radiance[k] * b.y;
        sh[2] += w_radiance[k] * b.z;
        sh[3] += w_radiance[k] * b.w;
    }
    // Radiance's coefficients by Monte Carlo over the sphere; irradiance /
    // pi's are theirs times 1 and 2/3 (uniform radiance 1 gives 1).
    let scale = 4.0 * PI / f32(GI_RAYS);
    sh[0] *= scale;
    sh[1] *= scale * (2.0 / 3.0);
    sh[2] *= scale * (2.0 / 3.0);
    sh[3] *= scale * (2.0 / 3.0);
    let alpha = select(max(1.0 / f32(updates + 1u), 0.03), 1.0, updates == 0u);
    // Words: (r0 g0) (b0 r1) (g1 b1) (r2 g2) (b2 r3) (g3 b3).
    let fresh = array<vec2<f32>, 6>(vec2<f32>(sh[0].r, sh[0].g), vec2<f32>(sh[0].b, sh[1].r), vec2<f32>(sh[1].g, sh[1].b), vec2<f32>(sh[2].r, sh[2].g), vec2<f32>(sh[2].b, sh[3].r), vec2<f32>(sh[3].g, sh[3].b));
    for (var w = 0u; w < 6u; w++) {
        gi_probes[at + w] = pack2x16float(mix(gi_half2(gi_probes[at + w]), fresh[w], alpha));
    }
    let inside = atomicLoad(&w_inside) * 4u > GI_RAYS;
    gi_probes[at + GI_STATE_WORD] = (min(updates + 1u, 255u) << 8u) | select(GI_ACTIVE, 0u, inside);
}

// This frame's rays: from each probe to update, its set turned by this
// frame's rotation.
@compute @workgroup_size(64)
fn gi_rays_main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= frame_gi.counts.x * GI_RAYS { return; }
    let index = gi_lists[frame_gi.counts.w * GI_MAX_UPDATES + id.x / GI_RAYS];
    gi_rays[id.x] = GiRay(vec4<f32>(gi_probe_position(index), 1e30), gi_ray_dir(id.x % GI_RAYS, frame_gi.rotation), index);
}
