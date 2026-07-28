//!use helio_prelude
//!use helio_hiz
//
// Water surface — above-water and underwater views.
//
// One vertex stage displaces the grid by the sim heightfield; two fragment
// entry points shade it depending on which side the camera is on.
//
// `fs_above` integrates the medium analytically: the depth buffer gives the
// slant path length from the surface to whatever is behind it, and everything
// else — Beer-Lambert absorption, in-scatter, refraction magnitude — is derived
// from that one number. This is why there is no separate fullscreen raymarch
// any more; the march computed the same integral and was then overwritten by
// this pass (see #139).
//
// Bindings
//   0  camera           uniform  Camera (prelude layout)
//   1  water_volumes    storage read
//   2  water_sim        texture_2d<f32>  (RGBA16F: R=height, G=velocity, B/A=normal.xz)
//   3  water_samp       sampler          (linear, repeat  — for sim)
//   4  caustics_tex     texture_2d<f32>  (unused here; consumed by #146)
//   5  shared_samp      sampler          (linear, clamp   — for scene colour)
//   6  scene_color      texture_2d<f32>  (opaque scene rendered before this pass)
//   7  viewport         uniform vec4f    (xy=px size, zw=1/size)
//   8  depth_texture    texture_depth_2d (copy of the scene depth buffer)
//   9  depth_sampler    sampler          (nearest, clamp)
//   10 gbuffer_normal   texture_2d<f32>  (unused here)

struct WaterVolume {
    bounds_min:            vec4f,  // xyz = min corner
    bounds_max:            vec4f,  // xyz = max corner, w = surface_height
    wave_params:           vec4f,  // x = wave_amplitude (metres), y = freq, z = speed, w = steepness
    wave_direction:        vec4f,
    water_color:           vec4f,  // xyz = medium colour, w = foam_threshold
    extinction:            vec4f,  // xyz = absorption per metre, w = foam_amount
    reflection_refraction: vec4f,  // x = reflection_str, y = refraction_str, z = fresnel_power
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,  // x = ior, y = caustic_intensity, z = fresnel_min, w = density
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,  // x = enable, y = max_steps, z = step_size, w = thickness
    sim_dynamics:          vec4f,
    wind_params:           vec4f,
    _pad:                  vec4f,
}

@group(0) @binding(0) var<uniform>       camera:         Camera;
@group(0) @binding(1) var<storage, read> volumes:        array<WaterVolume>;
@group(0) @binding(2) var water_sim:      texture_2d_array<f32>;
@group(0) @binding(3) var water_samp:     sampler;
@group(0) @binding(4) var caustics_tex:   texture_2d<f32>;
@group(0) @binding(5) var shared_samp:    sampler;
@group(0) @binding(6) var scene_color:    texture_2d<f32>;
@group(0) @binding(7) var<uniform>        viewport:      vec4f;
@group(0) @binding(8) var depth_texture:  texture_depth_2d;
@group(0) @binding(9) var depth_sampler:  sampler;
@group(0) @binding(10) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(11) var hiz_min:        texture_2d<f32>;

const IOR_AIR: f32 = 1.0;

const CASCADE_PATCH_SIZES: array<f32, 3> = array(30.0, 90.0, 270.0);
const CASCADE_AMPLITUDE_WEIGHTS: array<f32, 3> = array(0.6, 0.3, 0.1);

/// Ceiling on refractive displacement, in metres. Long grazing paths would
/// otherwise ask for an offset large enough to smear unrelated parts of the
/// screen into the water.
const REFRACTION_MAX_OFFSET: f32 = 2.0;

struct VertexOutput {
    @builtin(position) position:  vec4f,
    @location(0)       world_pos: vec3f,
    @location(1)       sim_uv:    vec2f,
    @location(2)       face_type: f32,
    @location(3) @interpolate(flat) vol_index: u32,
}

// ── Volume geometry ──────────────────────────────────────────────────────────

/// Wave amplitude in world units.
///
/// Comes from the descriptor (`wave_amplitude`), NOT from the volume's vertical
/// extent — deepening a volume must not make its waves taller. Clamped to the
/// headroom on either side of the rest height so the displaced surface can never
/// leave the volume bounds.
fn water_wave_amplitude(vol: WaterVolume) -> f32 {
    let rest     = vol.bounds_max.w;
    let headroom = min(rest - vol.bounds_min.y, vol.bounds_max.y - rest);
    return clamp(vol.wave_params.x, 0.0, max(headroom, 0.0));
}

/// Normalized sim height → world Y. The single point of truth for where the
/// surface is; every consumer must go through this.
fn water_surface_height(sim_h: f32, vol: WaterVolume) -> f32 {
    return vol.bounds_max.w + sim_h * water_wave_amplitude(vol);
}

fn sim_layer(vol_index: u32, cascade: u32) -> u32 {
    return vol_index * 3u + cascade;
}

/// Sample a single cascade's height at a tiled UV.
fn sample_cascade_height(cascade: u32, uv: vec2f, vol_index: u32) -> f32 {
    return textureSampleLevel(water_sim, water_samp, uv, sim_layer(vol_index, cascade), 0.0).r;
}

/// Sample a single cascade's stored normal BA at a tiled UV.
fn sample_cascade_normal_ba(cascade: u32, uv: vec2f, vol_index: u32) -> vec2f {
    return textureSampleLevel(water_sim, water_samp, uv, sim_layer(vol_index, cascade), 0.0).ba;
}

/// Weighted sum of all cascade heights at a world XZ position.
/// Each cascade tiles: uv = fract(world_xz / patch_size).
fn water_cascade_height_sum(world_xz: vec2f, vol_index: u32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < 3u; i++) {
        let uv = fract(world_xz / CASCADE_PATCH_SIZES[i]);
        sum += sample_cascade_height(i, uv, vol_index) * CASCADE_AMPLITUDE_WEIGHTS[i];
    }
    return sum;
}

/// World-space surface normal from all cascades, computed by summing
/// per-cascade slopes (reconstructed from the stored normal) weighted by
/// each cascade's amplitude fraction, then normalizing.
fn water_total_normal(world_xz: vec2f, vol: WaterVolume, vol_index: u32) -> vec3f {
    var total_slope = vec2f(0.0);
    for (var i = 0u; i < 3u; i++) {
        let uv = fract(world_xz / CASCADE_PATCH_SIZES[i]);
        let ba = sample_cascade_normal_ba(i, uv, vol_index);
        let ny = sqrt(max(1.0 - dot(ba, ba), 1e-6));
        let amp = water_wave_amplitude(vol) * CASCADE_AMPLITUDE_WEIGHTS[i];
        total_slope += (ba / ny) * (amp / CASCADE_PATCH_SIZES[i]);
    }
    return normalize(vec3f(total_slope.x, 1.0, total_slope.y));
}

/// Grid UV in [0,1] → world XZ across the volume footprint.
fn water_sim_uv_to_world_xz(uv: vec2f, vol: WaterVolume) -> vec2f {
    return mix(vol.bounds_min.xz, vol.bounds_max.xz, uv);
}

/// World XZ → grid UV in [0,1].
fn water_world_xz_to_sim_uv(world_xz: vec2f, vol: WaterVolume) -> vec2f {
    let extent = max(vol.bounds_max.xz - vol.bounds_min.xz, vec2f(1e-4));
    return (world_xz - vol.bounds_min.xz) / extent;
}

/// World-space surface normal at a grid UV.
///
/// The sim's normal pass writes the heightfield normal into B/A in *normalized
/// sim space*: slopes are `dh/duv` with the raw (unscaled) height. Those slopes
/// have to be rescaled by `amplitude / footprint` before they mean anything in
/// world units — without it the normal does not respond to `wave_amplitude` and
/// the shading disagrees with the geometry the vertex stage produced.
fn water_normal(sim_uv: vec2f, vol: WaterVolume, vol_index: u32) -> vec3f {
    let layer = sim_layer(vol_index, 0u);
    let info = textureSampleLevel(water_sim, water_samp, sim_uv, layer, 0.0);
    let ba   = vec2f(info.b, info.a);
    let ny   = sqrt(max(1.0 - dot(ba, ba), 1e-6));

    let extent = max(vol.bounds_max.xz - vol.bounds_min.xz, vec2f(1e-4));
    let slope  = (ba / ny) * (water_wave_amplitude(vol) / extent);

    return normalize(vec3f(slope.x, 1.0, slope.y));
}

// ── Medium ───────────────────────────────────────────────────────────────────

/// How far a view ray travels through water before it hits something.
///
/// `min(distance to the opaque scene, distance to the volume floor)`, capped at
/// the volume diagonal so a sky sample or a grazing ray cannot produce an
/// unbounded path. This is the number the whole medium model is built on.
fn water_path_length(
    screen_uv:   vec2f,
    surface_pos: vec3f,
    view_dir:    vec3f,
    vol:         WaterVolume,
) -> f32 {
    let max_path = length(vol.bounds_max.xyz - vol.bounds_min.xyz);

    // Distance to the opaque scene behind the surface.
    var t_scene = max_path;
    let depth = textureSampleLevel(depth_texture, depth_sampler, screen_uv, 0);
    if depth < 1.0 {
        let scene_pos = helio_world_from_depth(camera.view_proj_inv, screen_uv, depth);
        t_scene = distance(scene_pos, surface_pos);
    }

    // Distance to the volume floor along the same ray.
    var t_floor = max_path;
    if view_dir.y < -1e-4 {
        t_floor = (vol.bounds_min.y - surface_pos.y) / view_dir.y;
    }

    return clamp(min(t_scene, t_floor), 0.0, max_path);
}

/// Displaced surface height above an arbitrary world XZ (not just the shaded
/// vertex), for working out how deep a submerged point sits.
fn water_surface_at(world_xz: vec2f, vol: WaterVolume, vol_index: u32) -> f32 {
    let uv = water_world_xz_to_sim_uv(world_xz, vol);
    if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
        return vol.bounds_max.w;
    }
    let h_sum = water_cascade_height_sum(world_xz, vol_index);
    return water_surface_height(h_sum, vol);
}

/// Caustic light landing on a submerged point.
///
/// The caustics pass rasterizes into the volume footprint, so the lookup is the
/// same world-XZ-to-UV mapping used everywhere else. Attenuated by how deep the
/// receiver sits (the pattern washes out as the column deepens) and by how much
/// the receiving surface faces up.
fn water_caustics(scene_pos: vec3f, vol: WaterVolume, screen_uv: vec2f, vol_index: u32) -> vec3f {
    if vol.caustics_params.x <= 0.5 {
        return vec3f(0.0);
    }

    let uv = water_world_xz_to_sim_uv(scene_pos.xz, vol);
    if any(uv < vec2f(0.0)) || any(uv > vec2f(1.0)) {
        return vec3f(0.0);
    }

    // Receivers above the waterline get nothing.
    let depth_below = water_surface_at(scene_pos.xz, vol, vol_index) - scene_pos.y;
    if depth_below <= 0.0 {
        return vec3f(0.0);
    }

    // Intensity was already applied by the projection pass; this only shapes it.
    let caustic = textureSampleLevel(caustics_tex, shared_samp, uv, 0.0).r;

    let recv_n = helio_gbuffer_normal(
        textureSampleLevel(gbuffer_normal, shared_samp, screen_uv, 0.0).xyz);
    let facing = clamp(recv_n.y, 0.0, 1.0);

    return vec3f(caustic) * facing * exp(-depth_below * 0.12);
}

/// Beer-Lambert transmittance over a path.
fn water_transmittance(path: f32, vol: WaterVolume) -> vec3f {
    return exp(-max(vol.extinction.rgb, vec3f(1e-4)) * path);
}

/// Schlick-style Fresnel, using the descriptor's `fresnel_min` and
/// `fresnel_power` rather than a hardcoded exponent.
fn water_fresnel(normal: vec3f, view_dir: vec3f, vol: WaterVolume) -> f32 {
    let cos_theta = max(0.0, dot(normal, -view_dir));
    let power     = max(vol.reflection_refraction.z, 1.0);
    return mix(vol.sim_params.z, 1.0, pow(1.0 - cos_theta, power));
}

/// Light scattered toward the eye by the medium itself.
///
/// The HG term is normalized so 1.0 is isotropic — it brightens the water when
/// looking toward the sun and darkens it looking away, which is what makes the
/// medium read as lit rather than as a flat tint.
fn water_inscatter(view_dir: vec3f, sun_dir: vec3f, vol: WaterVolume) -> vec3f {
    let phase = helio_hg_phase(dot(view_dir, sun_dir), 0.35) * 4.0 * HELIO_PI;
    return vol.water_color.rgb * clamp(phase, 0.4, 2.0);
}

// ── Foam ─────────────────────────────────────────────────────────────────────

fn hash21(p: vec2f) -> f32 {
    var h = fract(p * vec2f(0.1031, 0.1030));
    h += dot(h, h.yx + 33.33);
    return fract((h.x + h.y) * h.x);
}

fn value_noise(p: vec2f) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(
        mix(hash21(i),                  hash21(i + vec2f(1.0, 0.0)), u.x),
        mix(hash21(i + vec2f(0.0, 1.0)), hash21(i + vec2f(1.0, 1.0)), u.x),
        u.y,
    );
}

/// How much the water has softened into the shoreline: 0 at the contact line,
/// 1 once there is a useful depth of water underneath.
///
/// The colour already resolves correctly on its own — at zero path length the
/// transmittance is 1 and the medium returns the scene behind it untouched.
/// What produced the hard line was the Fresnel *reflection*, which persisted at
/// full strength right up to the contact. Fading it is what removes the edge,
/// and it keeps the surface opaque so no alpha blending is needed.
fn water_shore_factor(path: f32, vol: WaterVolume) -> f32 {
    let fade = max(water_wave_amplitude(vol) * 3.0, 0.5);
    return smoothstep(0.0, fade, path);
}

/// Foam coverage in [0,1] from two sources, driven by `foam_threshold`
/// (`water_color.w`) and scaled by `foam_amount` (`extinction.w`).
fn water_foam(path: f32, world_xz: vec2f, normal: vec3f, vol: WaterVolume) -> f32 {
    let amp = max(water_wave_amplitude(vol), 1e-3);
    let t   = camera.jitter_frame.z * 0.016;

    // Two layers scrolling against each other, so the band churns rather than
    // sitting still as a uniform ring.
    let n = value_noise(world_xz * 0.35 + vec2f( t * 0.20, -t * 0.13)) * 0.65
          + value_noise(world_xz * 1.10 + vec2f(-t * 0.31,  t * 0.17)) * 0.35;

    // Contact foam. The band scales with wave size — a bigger swell washes
    // further up the shore — and the noise makes its edge irregular.
    let band       = max(amp * 2.5, 0.35);
    let shore      = 1.0 - smoothstep(0.0, band, path);
    let shore_foam = smoothstep(0.25, 0.85, shore * (0.55 + n * 0.75));

    // Crest foam, from the horizontal slope magnitude. Note this stays subtle
    // until the sim gains resolution (#148): at 0.78 m per texel the steepest
    // representable slope is well below the default `foam_threshold`.
    let threshold  = max(vol.water_color.w, 1e-3);
    let steepness  = length(normal.xz) / max(normal.y, 1e-3);
    let crest_foam = smoothstep(threshold, threshold * 1.6, steepness) * (0.5 + n * 0.5);

    return clamp(max(shore_foam, crest_foam) * vol.extinction.w, 0.0, 1.0);
}

// ── Screen-space reflection ──────────────────────────────────────────────────
// Marches the shared `hiz_min` pyramid via `helio_hiz_march` — the same
// traversal helio-pass-ssr uses, rather than a second implementation living
// here (#147).
//
// The pyramid is built from the opaque depth buffer, which does not contain the
// water surface, so the ray cannot self-intersect and water does not reflect
// water. That is the expected limitation of this approach.

const SSR_MAX_DIST:    f32 = 100.0;
const SSR_NORMAL_BIAS: f32 = 0.002;
const SSR_START_LEVEL: i32 = 2;
const SSR_MAX_LEVEL:   i32 = 8;

struct SsrResult {
    color:      vec3f,
    confidence: f32,
}

fn trace_ssr(origin: vec3f, dir: vec3f, normal: vec3f, vol: WaterVolume) -> SsrResult {
    var result: SsrResult;
    result.color      = vec3f(0.0);
    result.confidence = 0.0;

    let near = camera.position_near.w;
    let far  = camera.forward_far.w;

    // Build the ray in view space so it can be clipped against the near plane
    // before projection — a ray crossing w = 0 projects to nonsense.
    var start_view  = (camera.view * vec4f(origin, 1.0)).xyz;
    let dir_view    = normalize((camera.view * vec4f(dir, 0.0)).xyz);
    let normal_view = (camera.view * vec4f(normal, 0.0)).xyz;
    start_view += normal_view * (-start_view.z * SSR_NORMAL_BIAS);

    var ray_len = SSR_MAX_DIST;
    if start_view.z + dir_view.z * ray_len > -near {
        ray_len = (-near - start_view.z) / dir_view.z;
    }
    if ray_len <= 0.0 {
        return result;
    }
    let end_view = start_view + dir_view * ray_len;

    let clip0 = camera.proj * vec4f(start_view, 1.0);
    let clip1 = camera.proj * vec4f(end_view, 1.0);
    if clip0.w <= 0.0 || clip1.w <= 0.0 {
        return result;
    }
    let p0 = vec3f(helio_ndc_to_uv(clip0.xy / clip0.w), clip0.z / clip0.w);
    let p1 = vec3f(helio_ndc_to_uv(clip1.xy / clip1.w), clip1.z / clip1.w);

    let march = helio_hiz_march(
        hiz_min, p0, p1,
        SSR_START_LEVEL, SSR_MAX_LEVEL,
        u32(clamp(vol.ssr_params.y, 8.0, 128.0)),
    );
    if !march.hit {
        return result;
    }

    let hit_uv = march.pos.xy;

    // Thickness: the pyramid says the ray reached this surface, but a hit far
    // behind it is the ray having tunnelled past a thin object.
    let thickness = max(vol.ssr_params.w, 1e-3);
    let ray_z     = helio_view_depth(march.pos.z, near, far);
    let scene_z   = helio_view_depth(
        textureSampleLevel(depth_texture, depth_sampler, hit_uv, 0), near, far);
    if ray_z > scene_z * (1.0 + thickness) {
        return result;
    }

    // Fade at the screen border and as the ray runs out of its budget, so
    // reflections dissolve instead of cutting off.
    let border    = min(min(hit_uv.x, 1.0 - hit_uv.x), min(hit_uv.y, 1.0 - hit_uv.y));
    let edge_fade = smoothstep(0.0, 0.1, border);

    let travelled  = length(hit_uv - p0.xy) / max(length(p1.xy - p0.xy), 1e-6);
    let dist_fade  = 1.0 - smoothstep(0.6, 1.0, travelled);

    result.color      = textureSampleLevel(scene_color, shared_samp, hit_uv, 0.0).rgb;
    result.confidence = clamp(edge_fade * dist_fade, 0.0, 1.0);
    return result;
}

// ── Sky ──────────────────────────────────────────────────────────────────────

fn sky_color(ray: vec3f, light_dir: vec3f) -> vec3f {
    let up      = clamp(ray.y, 0.0, 1.0);
    let horizon = vec3f(0.80, 0.90, 1.00);
    let zenith  = vec3f(0.10, 0.30, 0.80);
    let sky     = mix(horizon, zenith, up * up);
    let spec    = pow(max(0.0, dot(normalize(light_dir), ray)), 5000.0);
    return sky + vec3f(spec) * vec3f(10.0, 8.0, 6.0);
}

/// Reflected radiance for a ray leaving the surface. Sky is the baseline; a
/// screen-space hit refines it. A miss must never resolve to black.
fn water_reflection(
    world_pos: vec3f,
    ray:       vec3f,
    normal:    vec3f,
    light_dir: vec3f,
    vol:       WaterVolume,
) -> vec3f {
    let sky = sky_color(ray, light_dir);

    if vol.ssr_params.x <= 0.5 {
        return sky;
    }

    // Blend by confidence rather than switching, so a partial hit near the
    // screen edge dissolves into sky instead of popping.
    let hit = trace_ssr(world_pos, ray, normal, vol);
    return mix(sky, hit.color, hit.confidence);
}

// ── Vertex ───────────────────────────────────────────────────────────────────

@vertex
fn vs_main(@location(0) position: vec4f, @builtin(instance_index) instance_idx: u32) -> VertexOutput {
    let vol = volumes[instance_idx];
    let face_type = position.w;

    var out: VertexOutput;
    out.face_type = face_type;
    out.vol_index = instance_idx;

    if face_type < 0.5 {
        // Top face — static grid in normalized [-1,1], mapped to world XZ
        let uv  = position.xy * 0.5 + 0.5;
        let xz  = water_sim_uv_to_world_xz(uv, vol);
        let h_sum = water_cascade_height_sum(xz, instance_idx);
        let world = vec3f(xz.x, water_surface_height(h_sum, vol), xz.y);
        out.position  = camera.view_proj * vec4f(world, 1.0);
        out.world_pos = world;
        out.sim_uv    = uv;
    } else {
        // Side or bottom face — uniform AABB mapping.
        // Top-edge side vertices (uv.y ≈ 1.0) follow the displaced water surface
        // so the side walls match the wave height instead of the flat AABB cap.
        let uv = position.xyz * 0.5 + 0.5;
        let xz = mix(vol.bounds_min.xz, vol.bounds_max.xz, uv.xz);
        var world_y: f32;
        if uv.y > 0.999 {
            let h_sum = water_cascade_height_sum(xz, instance_idx);
            world_y = water_surface_height(h_sum, vol);
        } else {
            world_y = mix(vol.bounds_min.y, vol.bounds_max.y, uv.y);
        }
        let world = vec3f(xz.x, world_y, xz.y);
        out.position  = camera.view_proj * vec4f(world, 1.0);
        out.world_pos = world;
        out.sim_uv    = vec2f(0.0);
    }
    return out;
}

// ── Fragment: above water ────────────────────────────────────────────────────

fn fs_above(in: VertexOutput) -> vec4f {
    let vol       = volumes[in.vol_index];
    let light_dir = normalize(vol.sun_direction.xyz);

    let normal    = water_total_normal(in.world_pos.xz, vol, in.vol_index);
    let view_dir  = normalize(in.world_pos - camera.position_near.xyz);
    let screen_uv = in.position.xy * viewport.zw;

    // Path through the medium straight down the view ray, before refraction.
    // Used to size the refraction offset; the offset then changes which pixel
    // we are looking through, so absorption is re-evaluated against that.
    let path_direct = water_path_length(screen_uv, in.world_pos, view_dir, vol);
    let vi = in.vol_index;

    // ── Refraction ───────────────────────────────────────────────────────────
    // Work out the lateral displacement in WORLD units first, then project it
    // to screen. A ray entering a tilted surface deviates in proportion to the
    // tilt and to how strongly the medium bends it, and that deviation
    // accumulates over the distance it travels — so shallow water barely
    // distorts and deep water distorts a lot, which a constant screen-space
    // offset cannot express.
    let view_pos = camera.view * vec4f(in.world_pos, 1.0);
    let view_z   = max(-view_pos.z, camera.position_near.w);
    let n_view   = (camera.view * vec4f(normal, 0.0)).xyz;

    let bend = 1.0 - 1.0 / max(vol.sim_params.x, 1.0);
    let world_offset = clamp(
        n_view.xy * path_direct * bend * vol.reflection_refraction.y,
        vec2f(-REFRACTION_MAX_OFFSET),
        vec2f(REFRACTION_MAX_OFFSET),
    );

    // World offset at `view_z` → screen offset. `proj[0][0]` / `proj[1][1]` carry
    // the focal length and aspect, so dividing by view depth makes the
    // distortion perspective-correct: it no longer grows as the camera walks
    // away. View-space Y is up and UV Y is down, hence the flip.
    let focal  = vec2f(camera.proj[0][0], camera.proj[1][1]);
    let offset = vec2f(world_offset.x, -world_offset.y) * focal * 0.5 / view_z;

    var refract_uv = clamp(screen_uv + offset, vec2f(0.001), vec2f(0.999));

    // Foreground rejection: if the offset landed on something nearer than the
    // water surface — a rock breaking the waterline, a hull, a character — that
    // object is in front of the water and must not be dragged into it.
    let refr_depth = textureSampleLevel(depth_texture, depth_sampler, refract_uv, 0);
    let refr_z     = helio_view_depth(refr_depth, camera.position_near.w, camera.forward_far.w);
    var path        = path_direct;
    var scene_depth = refr_depth;
    if refr_z < view_z {
        refract_uv  = screen_uv;
        scene_depth = textureSampleLevel(depth_texture, depth_sampler, screen_uv, 0);
    } else {
        path = water_path_length(refract_uv, in.world_pos, view_dir, vol);
    }

    var refracted = textureSampleLevel(scene_color, shared_samp, refract_uv, 0.0).rgb;

    // Caustics land on the submerged surface being looked at through the water,
    // so they are added before absorption attenuates the whole lot.
    if scene_depth < 1.0 {
        let scene_pos = helio_world_from_depth(camera.view_proj_inv, refract_uv, scene_depth);
        refracted += water_caustics(scene_pos, vol, refract_uv, vi);
    }

    // ── Medium ───────────────────────────────────────────────────────────────
    let transmittance = water_transmittance(path, vol);
    let inscatter     = water_inscatter(view_dir, light_dir, vol);
    let through_water = refracted * transmittance + inscatter * (1.0 - transmittance);

    // ── Reflection ───────────────────────────────────────────────────────────
    let reflected_ray = reflect(view_dir, normal);
    let reflected     = water_reflection(in.world_pos, reflected_ray, normal, light_dir, vol);

    // Fade the reflection into the shoreline, otherwise the contact with
    // geometry reads as a hard analytic line.
    let fresnel = water_fresnel(normal, view_dir, vol) * water_shore_factor(path, vol);
    var color   = mix(through_water, reflected, fresnel);

    // ── Foam ─────────────────────────────────────────────────────────────────
    let foam = water_foam(path, in.world_pos.xz, normal, vol);
    color = mix(color, vec3f(0.92, 0.96, 1.00), foam);

    return vec4f(color, 1.0);
}

// ── Fragment: underwater ─────────────────────────────────────────────────────
// Looking UP at the surface from below. The refracted ray escapes into the sky
// through Snell's window; outside that cone the surface is a mirror onto the
// submerged scene.

fn fs_under(in: VertexOutput) -> vec4f {
    let vol       = volumes[in.vol_index];
    let light_dir = normalize(vol.sun_direction.xyz);
    let ior_water = max(vol.sim_params.x, 1.0);

    let normal    = -water_total_normal(in.world_pos.xz, vol, in.vol_index);
    let view_dir  = normalize(in.world_pos - camera.position_near.xyz);
    let screen_uv = in.position.xy * viewport.zw;

    let fresnel = water_fresnel(normal, view_dir, vol);

    // Water → air. A zero-length result is total internal reflection.
    let refracted_ray = refract(view_dir, normal, ior_water / IOR_AIR);
    var above_color   = vec3f(0.0);
    let escapes       = length(refracted_ray) > 0.5;
    if escapes {
        above_color = sky_color(refracted_ray, light_dir);
    }

    // Reflected back down onto the submerged scene.
    let reflected_ray = reflect(view_dir, normal);
    var below_color   = vec3f(0.0);

    var confidence = 0.0;
    if vol.ssr_params.x > 0.5 {
        let hit = trace_ssr(in.world_pos, reflected_ray, normal, vol);
        below_color = hit.color;
        confidence  = hit.confidence;
    }
    if confidence < 0.999 {
        // Screen-space approximation of the submerged scene. Distortion is in
        // view space for the same reason as the above-water path.
        let n_view = (camera.view * vec4f(normal, 0.0)).xyz;
        let reflect_uv = clamp(
            screen_uv + vec2f(n_view.x, -n_view.y) * vol.reflection_refraction.y,
            vec2f(0.001),
            vec2f(0.999),
        );
        let fallback = textureSampleLevel(scene_color, shared_samp, reflect_uv, 0.0).rgb;
        below_color = mix(fallback, below_color, confidence);
    }

    // The medium's own colour, from the descriptor rather than a hardcoded tint.
    // Full depth-based underwater extinction is #145.
    below_color *= max(vol.water_color.rgb, vec3f(0.02));

    let escape_factor = select(0.0, 1.0, escapes);
    return vec4f(mix(below_color, above_color, (1.0 - fresnel) * escape_factor), 1.0);
}

// ── Combined entry point ─────────────────────────────────────────────────────

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let vol       = volumes[in.vol_index];
    let light_dir = normalize(vol.sun_direction.xyz);
    let view_dir  = normalize(in.world_pos - camera.position_near.xyz);
    let screen_uv = in.position.xy * viewport.zw;

    if in.face_type >= 0.5 {
        // Side or bottom face: simple medium integration without refraction,
        // reflection, or foam — just depth-based extinction.
        let path = water_path_length(screen_uv, in.world_pos, view_dir, vol);
        let transmittance = water_transmittance(path, vol);
        let inscatter     = water_inscatter(view_dir, light_dir, vol);
        let scene = textureSampleLevel(scene_color, shared_samp, screen_uv, 0.0).rgb;
        let color = scene * transmittance + inscatter * (1.0 - transmittance);
        return vec4f(color, 1.0);
    }

    // Top face: above or underwater based on camera height.
    let cam_above = camera.position_near.y > in.world_pos.y;
    if cam_above {
        return fs_above(in);
    } else {
        return fs_under(in);
    }
}
