//!use helio_prelude
//
// Volumetric fog accumulation.
//
// Ray-marches from the camera to the depth-buffer surface, accumulating
// in-scattered light and Beer-Lambert extinction into a quarter-resolution
// target. The post-process uber shader composites the result:
//
//     color = color * fog.a + fog.rgb
//
// so rgb is in-scattered radiance (already premultiplied by the transmittance in
// front of it) and a is the transmittance to the surface. Both are scene-linear:
// the composite happens before exposure and tonemapping, because in-scattering is
// radiance like any other and has to be tonemapped *with* the scene, not painted
// on afterwards.

// ── Fog config ──────────────────────────────────────────────────────────────
// Byte-identical to libhelio::GpuFogUniforms (64 bytes), which is itself the tail
// block of GpuPostProcessUniforms. The pass copies it out rather than mirroring
// all 368 bytes of the post-process struct here.

struct FogUniforms {
    fog_enabled:               u32,
    fog_mode:                  u32,
    fog_density:               f32,
    fog_height_falloff:        f32,
    fog_start_distance:        f32,
    fog_max_distance:          f32,
    fog_height:                f32,
    fog_scattering_anisotropy: f32,
    fog_color:                 vec3<f32>,
    _pad_fog_color:            f32,
    fog_emissive:              vec3<f32>,
    _pad_fog_emissive:         f32,
}

const FOG_MODE_UNIFORM: u32 = 0u;
const FOG_MODE_HEIGHT: u32  = 1u;

// ── Lights ──────────────────────────────────────────────────────────────────
// Mirror of libhelio::GpuLight (96 bytes). See that struct's doc comment for the
// full list of shaders that must be edited together.

struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
    god_rays_enabled:  u32,
    god_rays_density:  f32,
    god_rays_weight:   f32,
    god_rays_decay:    f32,
    god_rays_exposure: f32,
    _pad2_0:           u32,
    _pad2_1:           u32,
    _pad2_2:           u32,
}

struct LightMatrix { mat: mat4x4<f32> }

struct FogGlobals {
    /// Globals.csm_splits — cascade boundaries for directional shadow lookup.
    csm_splits:  vec4<f32>,
    light_count: u32,
    frame:       u32,
    /// Ray-march steps per pixel.
    steps:       u32,
    _pad:        u32,
}

const LIGHT_DIRECTIONAL: u32 = 0u;
const LIGHT_POINT:       u32 = 1u;
const LIGHT_SPOT:        u32 = 2u;

const NO_SHADOW: u32 = 4294967295u;

@group(0) @binding(0) var<uniform>       camera:          Camera;
@group(0) @binding(1) var<uniform>       fog:             FogUniforms;
@group(0) @binding(2) var<uniform>       fog_globals:     FogGlobals;
@group(0) @binding(3) var<storage, read> lights:          array<GpuLight>;
@group(0) @binding(4) var<storage, read> shadow_matrices: array<LightMatrix>;
@group(0) @binding(5) var                depth_tex:       texture_depth_2d;
@group(0) @binding(6) var                shadow_atlas:    texture_depth_2d_array;
@group(0) @binding(7) var                shadow_samp:     sampler_comparison;
@group(0) @binding(8) var                fog_out:         texture_storage_2d<rgba16float, write>;

// ── Density ─────────────────────────────────────────────────────────────────

/// Media density at a world position.
fn density_at(p: vec3<f32>) -> f32 {
    if fog.fog_mode == FOG_MODE_HEIGHT {
        // Full density at or below fog_height, decaying exponentially above it.
        // max(h, 0) rather than h: without the clamp, a point far below the base
        // gets exp(+large) and the march blows up to an opaque wall.
        let h = p.y - fog.fog_height;
        return fog.fog_density * exp(-max(h, 0.0) * fog.fog_height_falloff);
    }
    return fog.fog_density;
}

// ── Shadowing ───────────────────────────────────────────────────────────────

/// Fraction of `light_idx` reaching `p`. 1.0 = fully lit.
///
/// A single comparison tap, not the PCF/PCSS kernel deferred lighting uses: this
/// runs once per light per march step, and the result is integrated over ~64 steps
/// and then bilinearly upscaled 4x, which smooths the aliasing that a single tap
/// would show on a surface.
fn shaft_visibility(light_idx: u32, p: vec3<f32>) -> f32 {
    let light = lights[light_idx];
    if light.shadow_index == NO_SHADOW { return 1.0; }

    var layer = light.shadow_index;

    if light.light_type == LIGHT_DIRECTIONAL {
        // Cascade chosen by distance from the camera to the *sample*, matching how
        // deferred lighting picks one for a surface point.
        let dist = length(p - camera.position_near.xyz);
        let sel = helio_csm_select(dist, fog_globals.csm_splits);
        layer = light.shadow_index + sel.cascade_a;
    } else if light.light_type == LIGHT_POINT {
        // Point lights need a cube face; that lookup is not shared yet, so they
        // contribute unshadowed rather than sampling the wrong atlas layer.
        return 1.0;
    }

    let proj = helio_shadow_project(shadow_matrices[layer].mat, p);
    // Outside the map or behind the light: lit, not shadowed. Returning 0.0 here
    // would ring the fog volume with a black shell wherever the cascade ends.
    if !proj.valid { return 1.0; }

    return textureSampleCompareLevel(shadow_atlas, shadow_samp, proj.uv, layer, proj.depth);
}

// ── Light evaluation ────────────────────────────────────────────────────────

/// In-scattered radiance from one light at `p`, for a ray travelling `ray_dir`.
///
/// `step_idx` drives `god_rays_decay`, which attenuates per march step: values
/// below 1.0 shorten the shaft. It, `god_rays_weight` and `god_rays_exposure` come
/// from the radial-blur god-ray technique and have no physical meaning in a
/// raymarch — they are kept as artistic multipliers so existing light setups
/// author the same way.
fn inscatter_from_light(light_idx: u32, p: vec3<f32>, ray_dir: vec3<f32>, step_idx: u32) -> vec3<f32> {
    let light = lights[light_idx];

    // Opt-in per light: in-scattering costs a shadow tap per march step, so a
    // scene's fill lights should not pay for it silently.
    if light.god_rays_enabled == 0u { return vec3<f32>(0.0); }

    var to_light: vec3<f32>;
    var atten = 1.0;

    if light.light_type == LIGHT_DIRECTIONAL {
        to_light = normalize(-light.direction_outer.xyz);
    } else {
        let delta = light.position_range.xyz - p;
        let dist = length(delta);
        let range = max(light.position_range.w, 1e-4);
        if dist > range { return vec3<f32>(0.0); }
        to_light = delta / max(dist, 1e-6);

        // Inverse-square with a windowed range cutoff, so the contribution reaches
        // exactly zero at the range boundary instead of popping.
        let window = clamp(1.0 - pow(dist / range, 4.0), 0.0, 1.0);
        atten = (window * window) / max(dist * dist, 1e-4);

        if light.light_type == LIGHT_SPOT {
            let cd = dot(-to_light, normalize(light.direction_outer.xyz));
            let outer = light.direction_outer.w;
            let inner = light.inner_angle;
            let spot = clamp((cd - outer) / max(inner - outer, 1e-4), 0.0, 1.0);
            atten *= spot * spot;
        }
    }

    // cos = 1 when the ray points straight at the light, so g > 0 peaks looking
    // into the sun — the forward-scattering halo.
    let phase = helio_hg_phase(dot(ray_dir, to_light), fog.fog_scattering_anisotropy);
    let vis = shaft_visibility(light_idx, p);
    let decay = pow(clamp(light.god_rays_decay, 0.0, 1.0), f32(step_idx));

    let radiance = light.color_intensity.rgb * light.color_intensity.w;
    return radiance * atten * phase * vis * decay
        * light.god_rays_weight * light.god_rays_exposure * light.god_rays_density;
}

// ── Accumulation ────────────────────────────────────────────────────────────

/// Interleaved-gradient noise — dithers the march start so the step pattern reads
/// as noise rather than as concentric bands, which the 4x upscale then blurs out.
fn ign(pixel: vec2<f32>, frame: u32) -> f32 {
    let f = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(f, vec2<f32>(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8, 1)
fn cs_fog(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(fog_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let coord = vec2<i32>(gid.xy);

    // Identity for the composite: no in-scattering, full transmittance.
    if fog.fog_enabled == 0u || fog.fog_density <= 0.0 {
        textureStore(fog_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims);

    // depth_tex is full internal resolution while this target is a quarter of it,
    // so sample by UV rather than reusing gid. This takes one depth per fog texel;
    // a texel straddling a silhouette picks one side of it, which the upscale then
    // feathers.
    let depth_dims = textureDimensions(depth_tex);
    let depth_coord = vec2<i32>(clamp(
        uv * vec2<f32>(depth_dims),
        vec2<f32>(0.0),
        vec2<f32>(depth_dims) - vec2<f32>(1.0),
    ));
    let depth = textureLoad(depth_tex, depth_coord, 0);

    let cam_pos = camera.position_near.xyz;
    let world = helio_world_from_depth(camera.view_proj_inv, uv, depth);

    let to_frag = world - cam_pos;
    let frag_dist = length(to_frag);
    if frag_dist < 1e-4 {
        textureStore(fog_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }
    let ray_dir = to_frag / frag_dist;

    // Depth 1.0 is the far plane (sky): march the full fog range rather than to a
    // reconstructed far-plane point, which would make fog depth swing with the
    // projection.
    var end = select(min(frag_dist, fog.fog_max_distance), fog.fog_max_distance, depth >= 1.0);
    let start = min(fog.fog_start_distance, end);
    let span = end - start;
    if span <= 0.0 {
        textureStore(fog_out, coord, vec4<f32>(0.0, 0.0, 0.0, 1.0));
        return;
    }

    let steps = max(fog_globals.steps, 1u);
    let seg = span / f32(steps);
    let jitter = ign(vec2<f32>(gid.xy), fog_globals.frame);

    var transmittance = 1.0;
    var scattered = vec3<f32>(0.0);

    for (var i = 0u; i < steps; i++) {
        let t = start + (f32(i) + jitter) * seg;
        let p = cam_pos + ray_dir * t;

        let density = density_at(p);
        if density > 0.0 {
            let sigma = density * seg;
            let step_transmittance = exp(-sigma);

            // Emissive is media self-illumination, so it is not tinted by fog_color
            // (which is the albedo the in-scattered light bounces off).
            var radiance = vec3<f32>(0.0);
            for (var li = 0u; li < fog_globals.light_count; li++) {
                radiance += inscatter_from_light(li, p, ray_dir, i);
            }
            radiance = radiance * fog.fog_color + fog.fog_emissive;

            // Energy-conserving integration: the light scattered in this segment is
            // what the segment absorbs, (1 - exp(-sigma)), attenuated by everything
            // already in front of it. Accumulating `radiance * sigma` instead
            // over-brightens as density rises, because it has no saturation term.
            scattered += transmittance * (1.0 - step_transmittance) * radiance;
            transmittance *= step_transmittance;

            // Nothing behind this can contribute meaningfully.
            if transmittance < 0.003 { break; }
        }
    }

    textureStore(fog_out, coord, vec4<f32>(scattered, transmittance));
}
