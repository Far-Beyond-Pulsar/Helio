//!use helio_prelude
//
// Underwater participating medium.
//
// Runs fullscreen when the camera is inside a water volume. Every pixel is
// integrated against the distance the view ray travels through water, so near
// objects stay readable and far ones dissolve into the medium — which is the
// difference between being submerged and having a filter laid over the image.
//
// This replaces a depth-independent screen filter (lens warp + chromatic
// aberration + flat tint + vignette). Those touches are kept but scaled back,
// because the depth cue now does the work. See #145.
//
// Output goes to a scratch texture with alpha = 1; the Rust side blits it back
// over water_output.
//
// Bindings
//   0  cameras       storage  array<Camera, CAMERA_SLOTS> (prelude layout)
//   1  volumes       storage  array<WaterVolume>
//   2  scene_tex     texture  water_output bound as source
//   3  scene_samp    sampler  linear clamp
//   4  depth_texture texture_depth_2d
//   5  depth_samp    sampler  nearest clamp
//   6  water_sim     texture  RGBA16F heightfield
//   7  water_samp    sampler  linear

struct WaterVolume {
    bounds_min:            vec4f,
    bounds_max:            vec4f,  // w = surface_height
    wave_params:           vec4f,  // x = wave_amplitude
    wave_direction:        vec4f,
    water_color:           vec4f,  // xyz = medium colour
    extinction:            vec4f,  // xyz = absorption per metre
    reflection_refraction: vec4f,
    caustics_params:       vec4f,
    fog_params:            vec4f,  // x = density, y = god_rays_intensity
    sim_params:            vec4f,
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,
    sim_dynamics:          vec4f,
    wind_params:           vec4f,
    _pad:                  vec4f,
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, CAMERA_SLOTS>;
@group(0) @binding(1) var<storage, read> volumes:    array<WaterVolume>;
@group(0) @binding(2) var scene_tex:     texture_2d<f32>;
@group(0) @binding(3) var scene_samp:    sampler;
@group(0) @binding(4) var depth_texture: texture_depth_2d;
@group(0) @binding(5) var depth_samp:    sampler;
@group(0) @binding(6) var water_sim:     texture_2d_array<f32>;
@group(0) @binding(7) var water_samp:    sampler;
@group(0) @binding(8) var caustics_tex:  texture_2d<f32>;

/// Vertical distance over which entering the water ramps in, in metres. Without
/// this the effect switches on the instant the camera crosses the plane, which
/// pops hard exactly when the camera sits at the waterline with waves rolling
/// over it.
const SUBMERGE_BLEND: f32 = 0.35;

const GOD_RAY_SAMPLES: i32 = 24;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0)       uv:       vec2f,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    var out: VertexOutput;
    out.position = vec4f(x * 2.0 - 1.0, 1.0 - y * 2.0, 0.0, 1.0);
    out.uv = vec2f(x, y);
    return out;
}

// ── Volume geometry ──────────────────────────────────────────────────────────
// Mirrors the helpers in surface.wgsl. WGSL has no include mechanism, and the
// prelude is engine-wide rather than water-specific, so these are duplicated
// rather than shared. They must stay in step with surface.wgsl.

fn water_wave_amplitude(vol: WaterVolume) -> f32 {
    let rest     = vol.bounds_max.w;
    let headroom = min(rest - vol.bounds_min.y, vol.bounds_max.y - rest);
    return clamp(vol.wave_params.x, 0.0, max(headroom, 0.0));
}

/// Return the sim texture array layer for a volume index and cascade.
fn sim_layer(vol_index: u32, cascade: u32) -> i32 {
    return i32(vol_index * 3u + cascade);
}

/// Displaced surface height above a world XZ position.
fn water_surface_at(world_xz: vec2f, vol: WaterVolume, vol_idx: u32) -> f32 {
    let extent = max(vol.bounds_max.xz - vol.bounds_min.xz, vec2f(1e-4));
    let uv     = (world_xz - vol.bounds_min.xz) / extent;
    if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
        return vol.bounds_max.w;
    }
    let h = textureSampleLevel(water_sim, water_samp, uv, vol_idx * 3u, 0.0).r;
    return vol.bounds_max.w + h * water_wave_amplitude(vol);
}

// ── Noise (lens wobble) ──────────────────────────────────────────────────────

fn hash2(p: vec2f) -> vec2f {
    let k = vec2f(127.1, 311.7);
    let s = sin(dot(p, k)) * 43758.5453;
    return fract(vec2f(s, s + 0.618));
}

fn smooth_noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = dot(hash2(i),                   f - vec2f(0.0, 0.0));
    let b = dot(hash2(i + vec2f(1.0, 0.0)), f - vec2f(1.0, 0.0));
    let c = dot(hash2(i + vec2f(0.0, 1.0)), f - vec2f(0.0, 1.0));
    let d = dot(hash2(i + vec2f(1.0, 1.0)), f - vec2f(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 0.5 + 0.5;
}

fn water_distortion(uv: vec2f, t: f32) -> vec2f {
    let scale = 3.0;
    let nx = smooth_noise(uv * scale + vec2f( t * 0.11, t * 0.07)) - 0.5;
    let ny = smooth_noise(uv * scale + vec2f(-t * 0.09, t * 0.13)) - 0.5;
    return vec2f(nx, ny);
}

// ── God rays ─────────────────────────────────────────────────────────────────

/// Screen-space light shafts toward the sun.
///
/// Marches from the pixel toward the sun's projected position, accumulating
/// where the depth buffer says nothing is occluding — i.e. where light has a
/// clear path down through the water. Cheap compared to marching the shadow
/// map, and underwater shafts are diffuse enough that the difference does not
/// read.
fn god_rays(uv: vec2f, vol: WaterVolume) -> f32 {
    let intensity = vol.fog_params.y;
    if intensity <= 1e-3 {
        return 0.0;
    }

    // Project a point far along the sun direction. Behind the camera: no shafts.
    let sun_world = cameras[0].position_near.xyz + normalize(vol.sun_direction.xyz) * 10000.0;
    let sun_clip  = cameras[0].view_proj * vec4f(sun_world, 1.0);
    if sun_clip.w <= 0.0 {
        return 0.0;
    }
    let sun_uv = helio_ndc_to_uv(sun_clip.xy / sun_clip.w);

    let delta = (sun_uv - uv) / f32(GOD_RAY_SAMPLES);
    var pos   = uv;
    var accum = 0.0;
    var decay = 1.0;

    for (var i = 0; i < GOD_RAY_SAMPLES; i++) {
        pos += delta;
        if any(pos < vec2f(0.0)) || any(pos > vec2f(1.0)) { break; }
        // Unoccluded (far) samples carry light; geometry blocks it.
        let d = textureSampleLevel(depth_texture, depth_samp, pos, 0);
        accum += step(0.9995, d) * decay;
        decay *= 0.96;
    }

    // Fade with angular distance so the effect concentrates around the sun.
    let falloff = 1.0 - smoothstep(0.0, 0.9, length(sun_uv - uv));
    return accum / f32(GOD_RAY_SAMPLES) * intensity * falloff;
}

// ── Fragment ─────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let cam_pos = cameras[0].position_near.xyz;

    // ── Which volume is the camera in, and how far under? ────────────────────
    var vol_idx: i32 = -1;
    var submersion = 0.0;
    for (var i = 0u; i < arrayLength(&volumes); i++) {
        let v = volumes[i];
        if cam_pos.x < v.bounds_min.x || cam_pos.x > v.bounds_max.x { continue; }
        if cam_pos.z < v.bounds_min.z || cam_pos.z > v.bounds_max.z { continue; }
        if cam_pos.y < v.bounds_min.y { continue; }

        // Against the DISPLACED surface, not the flat plane — otherwise the
        // test is wrong precisely when the camera is at the waterline.
        let surf = water_surface_at(cam_pos.xz, v, i);
        let depth_under = surf - cam_pos.y;
        if depth_under <= -SUBMERGE_BLEND { continue; }

        submersion = smoothstep(-SUBMERGE_BLEND, SUBMERGE_BLEND, depth_under);
        vol_idx = i32(i);
        break;
    }

    let scene_raw = textureSampleLevel(scene_tex, scene_samp, in.uv, 0.0);
    if vol_idx < 0 || submersion <= 0.0 {
        return scene_raw;
    }

    let vol       = volumes[u32(vol_idx)];
    let surface_h = water_surface_at(cam_pos.xz, vol, u32(vol_idx));
    let cam_depth = max(surface_h - cam_pos.y, 0.0);

    // ── Lens wobble + chromatic aberration ───────────────────────────────────
    // Kept, but well below the previous magnitudes: the depth-based extinction
    // below now carries the underwater read, and these become distracting once
    // it does.
    let t         = cameras[0].jitter_frame.z * 0.016 * max(vol.wave_params.z, 0.1);
    let dist_raw  = water_distortion(in.uv, t) * 0.7
                  + water_distortion(in.uv * 2.1 + vec2f(0.37, 0.71), t * 0.6) * 0.3;
    let dist_str  = clamp(water_wave_amplitude(vol) * (0.05 + cam_depth * 0.01), 0.002, 0.06);
    let warp_uv   = in.uv + dist_raw * dist_str;

    let ca     = clamp(cam_depth * 0.0015, 0.0002, 0.004);
    let radial = in.uv - 0.5;
    let uv_r   = clamp(warp_uv + radial * ca,       vec2f(0.001), vec2f(0.999));
    let uv_g   = clamp(warp_uv,                     vec2f(0.001), vec2f(0.999));
    let uv_b   = clamp(warp_uv - radial * ca * 1.4, vec2f(0.001), vec2f(0.999));

    let scene = vec3f(
        textureSampleLevel(scene_tex, scene_samp, uv_r, 0.0).r,
        textureSampleLevel(scene_tex, scene_samp, uv_g, 0.0).g,
        textureSampleLevel(scene_tex, scene_samp, uv_b, 0.0).b,
    );

    // ── Distance through the medium ──────────────────────────────────────────
    let depth = textureSampleLevel(depth_texture, depth_samp, uv_g, 0);
    let far   = cameras[0].forward_far.w;

    var dist: f32;
    var ray_dir: vec3f;
    var lit = scene;
    if depth >= 1.0 {
        // Nothing there: the ray runs until the medium has fully absorbed it.
        ray_dir = normalize(helio_world_from_depth(cameras[0].view_proj_inv, uv_g, 0.5) - cam_pos);
        dist    = far;
    } else {
        let world_pos = helio_world_from_depth(cameras[0].view_proj_inv, uv_g, depth);
        ray_dir = normalize(world_pos - cam_pos);
        dist    = distance(world_pos, cam_pos);

        // Caustics on submerged geometry, seen directly rather than through
        // the surface. Same projection the surface shader uses.
        if vol.caustics_params.x > 0.5 {
            let extent = max(vol.bounds_max.xz - vol.bounds_min.xz, vec2f(1e-4));
            let cuv    = (world_pos.xz - vol.bounds_min.xz) / extent;
            let below  = water_surface_at(world_pos.xz, vol, u32(vol_idx)) - world_pos.y;
            if below > 0.0 && all(cuv >= vec2f(0.0)) && all(cuv <= vec2f(1.0)) {
                let caustic = textureSampleLevel(caustics_tex, water_samp, cuv, 0.0).r;
                lit += vec3f(caustic) * exp(-below * 0.12);
            }
        }
    }

    // A ray heading upward leaves the water at the surface; it should only
    // accumulate medium up to that crossing.
    if ray_dir.y > 1e-4 {
        let t_surface = (surface_h - cam_pos.y) / ray_dir.y;
        if t_surface > 0.0 {
            dist = min(dist, t_surface);
        }
    }

    // ── Beer-Lambert toward the in-scattered medium colour ───────────────────
    let extinction = max(vol.extinction.rgb, vec3f(1e-4));
    let transmit   = exp(-extinction * dist);

    let phase    = helio_hg_phase(dot(ray_dir, normalize(vol.sun_direction.xyz)), 0.35)
                 * 4.0 * HELIO_PI;
    let inscatter = max(vol.water_color.rgb, vec3f(0.02, 0.10, 0.28))
                  * clamp(phase, 0.4, 2.0);

    var color = lit * transmit + inscatter * (1.0 - transmit);

    // ── God rays ─────────────────────────────────────────────────────────────
    color += inscatter * god_rays(uv_g, vol);

    // ── Vignette ─────────────────────────────────────────────────────────────
    let d        = in.uv - 0.5;
    let vignette = 1.0 - dot(d, d) * 3.2;
    let vig_str  = clamp(cam_depth * 0.06, 0.0, 0.45);
    color *= mix(1.0, max(vignette, 0.0), vig_str);

    // Ramp the whole effect in across the waterline.
    return vec4f(mix(scene_raw.rgb, color, submersion), 1.0);
}
