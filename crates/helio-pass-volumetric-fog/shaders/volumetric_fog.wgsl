//!use helio_prelude
//
// Volumetric fog — froxel grid (Hillaire, "Physically Based and Unified Volumetric
// Rendering in Frostbite", SIGGRAPH 2015).
//
// Two compute passes over a view-space 3D grid, rather than a raymarch per pixel:
//
//   cs_inject     — one thread per froxel: density, one shadow tap per light,
//                   temporally blended against the reprojected previous frame.
//   cs_integrate  — one thread per (x,y) column: marches z once, turning the
//                   per-froxel scattering/extinction into accumulated
//                   in-scattering + transmittance.
//
// The composite in postprocess.wgsl is then a single trilinear 3D fetch at the
// pixel's depth:
//
//     color = color * fog.a + fog.rgb
//
// Why this shape:
//   - Cost is decoupled from screen resolution. 160x90x64 = ~920k froxels lit
//     once each, against 1280x720x64 = ~59M samples for the per-pixel march.
//   - The trilinear fetch filters in x, y *and* depth, so there is no blocky
//     upsample and no bilateral filter to write.
//   - Temporal reprojection carries samples across frames, which is what makes
//     one shadow tap per froxel enough. Without it this would need many more.

// ── Fog config ──────────────────────────────────────────────────────────────
// Byte-identical to libhelio::GpuFogUniforms (64 bytes), the tail block of
// GpuPostProcessUniforms. Copied out rather than mirroring all 368 bytes here.

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
    /// 0 on the first frame or after a camera cut: ignore history.
    history_valid: u32,
    /// Weight of the current frame in the temporal blend. Lower = steadier, but
    /// slower to react to lights and shadows moving.
    temporal_blend: f32,
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
@group(0) @binding(5) var                shadow_atlas:    texture_depth_2d_array;
@group(0) @binding(6) var                shadow_samp:     sampler_comparison;
/// Previous frame's scattering grid, for temporal reprojection.
@group(0) @binding(7) var                scatter_history: texture_3d<f32>;
@group(0) @binding(8) var                linear_samp:     sampler;
/// rgb = in-scattered radiance * density, a = extinction.
@group(0) @binding(9) var                scatter_out:     texture_storage_3d<rgba16float, write>;

// Integration reads the scattering grid written above and writes the accumulated
// result. Separate group so the two dispatches can swap only what differs.
@group(1) @binding(0) var scatter_in:     texture_3d<f32>;
/// rgb = accumulated in-scattering, a = transmittance.
@group(1) @binding(1) var integrated_out: texture_storage_3d<rgba16float, write>;

// ── Density ─────────────────────────────────────────────────────────────────

fn density_at(p: vec3<f32>) -> f32 {
    if fog.fog_mode == FOG_MODE_HEIGHT {
        // Full density at or below fog_height, decaying exponentially above it.
        // max(h, 0) rather than h: without the clamp a point far below the base
        // gets exp(+large) and the grid fills with an opaque wall.
        let h = p.y - fog.fog_height;
        return fog.fog_density * exp(-max(h, 0.0) * fog.fog_height_falloff);
    }
    return fog.fog_density;
}

// ── Froxel <-> world ────────────────────────────────────────────────────────

/// World position at the centre of a froxel, given normalized grid coords.
///
/// `slice_norm` maps through the prelude's exponential distribution, so this is
/// the exact inverse of what the composite does to find a slice from a depth.
fn froxel_world_pos(uv: vec2<f32>, slice_norm: f32) -> vec3<f32> {
    let ndc = helio_uv_to_ndc(uv);

    // Ray through this pixel: unproject the near and far plane points. Cheaper
    // schemes exist, but this one cannot disagree with the depth reconstruction
    // the rest of the engine does.
    let p_near = camera.view_proj_inv * vec4<f32>(ndc, 0.0, 1.0);
    let p_far  = camera.view_proj_inv * vec4<f32>(ndc, 1.0, 1.0);
    let wn = p_near.xyz / p_near.w;
    let wf = p_far.xyz / p_far.w;
    let dir = normalize(wf - wn);

    let view_depth = helio_froxel_view_depth_from_slice(slice_norm, fog.fog_max_distance);

    // Slices are planes of constant *view depth*, not spheres of constant radius,
    // so the radial distance along this ray is view_depth / cos(angle to forward).
    // Skipping this bows the grid toward the camera at the screen edges.
    let fwd = normalize(camera.forward_far.xyz);
    let cos_a = max(dot(dir, fwd), 1e-4);
    return camera.position_near.xyz + dir * (view_depth / cos_a);
}

// ── Shadowing ───────────────────────────────────────────────────────────────

/// Fraction of `light_idx` reaching `p`. 1.0 = fully lit.
///
/// One comparison tap, not the PCF/PCSS kernel deferred lighting uses. That is
/// affordable because it runs once per froxel rather than once per pixel per
/// step, and the temporal blend averages the result across frames.
fn shaft_visibility(light_idx: u32, p: vec3<f32>) -> f32 {
    let light = lights[light_idx];
    if light.shadow_index == NO_SHADOW { return 1.0; }

    var layer = light.shadow_index;

    if light.light_type == LIGHT_DIRECTIONAL {
        let dist = length(p - camera.position_near.xyz);
        let sel = helio_csm_select(dist, fog_globals.csm_splits);
        layer = light.shadow_index + sel.cascade_a;
    } else if light.light_type == LIGHT_POINT {
        // Point lights need a cube-face lookup, which is not shared yet, so they
        // light the fog unshadowed rather than sampling the wrong atlas layer.
        return 1.0;
    }

    let proj = helio_shadow_project(shadow_matrices[layer].mat, p);
    // Outside the map or behind the light: lit, not shadowed. Returning 0.0 would
    // ring the fog with a black shell wherever the cascade ends.
    if !proj.valid { return 1.0; }

    return textureSampleCompareLevel(shadow_atlas, shadow_samp, proj.uv, layer, proj.depth);
}

// ── Light evaluation ────────────────────────────────────────────────────────

/// In-scattered radiance from one light at `p`, for a view ray `ray_dir`.
///
/// `god_rays_weight` / `god_rays_exposure` / `god_rays_density` come from the
/// radial-blur god-ray technique and have no physical meaning here; they are kept
/// as artistic multipliers so existing light setups author the same way.
/// `god_rays_decay` is not applied — it attenuated per raymarch step, and a froxel
/// has no step index. Distance falloff comes from the medium instead.
fn inscatter_from_light(light_idx: u32, p: vec3<f32>, ray_dir: vec3<f32>) -> vec3<f32> {
    let light = lights[light_idx];

    // Opt-in per light: each one costs a shadow tap per froxel.
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

        // Inverse-square with a windowed cutoff, so the contribution reaches zero
        // exactly at the range boundary instead of popping.
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

    // cos = 1 looking straight at the light, so g > 0 peaks into the sun.
    let phase = helio_hg_phase(dot(ray_dir, to_light), fog.fog_scattering_anisotropy);
    let vis = shaft_visibility(light_idx, p);

    let radiance = light.color_intensity.rgb * light.color_intensity.w;
    return radiance * atten * phase * vis
        * light.god_rays_weight * light.god_rays_exposure * light.god_rays_density;
}

// ── Temporal reprojection ───────────────────────────────────────────────────

/// Previous frame's scattering at world position `p`, or `none` if it reprojects
/// off-grid.
///
/// Returns w < 0 to signal "no history" — a froxel that was off-screen or behind
/// the camera last frame has nothing to blend with, and reusing a clamped edge
/// sample there smears fog across the screen edges as the camera turns.
fn sample_history(p: vec3<f32>) -> vec4<f32> {
    let prev_clip = camera.prev_view_proj * vec4<f32>(p, 1.0);
    // For the engine's perspective matrix, clip.w is the positive view depth.
    if prev_clip.w <= HELIO_FROXEL_NEAR { return vec4<f32>(0.0, 0.0, 0.0, -1.0); }

    let prev_ndc = prev_clip.xyz / prev_clip.w;
    let prev_uv = helio_ndc_to_uv(prev_ndc.xy);
    if any(prev_uv < vec2<f32>(0.0)) || any(prev_uv > vec2<f32>(1.0)) {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    let prev_slice = helio_froxel_slice_from_view_depth(prev_clip.w, fog.fog_max_distance);
    if prev_slice < 0.0 || prev_slice > 1.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    return textureSampleLevel(scatter_history, linear_samp, vec3<f32>(prev_uv, prev_slice), 0.0);
}

// ── Injection ───────────────────────────────────────────────────────────────

/// Interleaved-gradient noise, used to jitter the sample point within its froxel.
/// Combined with the temporal blend this turns slice banding into noise that
/// averages out across frames.
fn ign(pixel: vec2<f32>, frame: u32) -> f32 {
    let f = pixel + 5.588238 * f32(frame % 64u);
    return fract(52.9829189 * fract(dot(f, vec2<f32>(0.06711056, 0.00583715))));
}

@compute @workgroup_size(8, 8, 1)
fn cs_inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(scatter_out);
    if any(gid >= dims) { return; }

    let coord = vec3<i32>(gid);

    if fog.fog_enabled == 0u || fog.fog_density <= 0.0 {
        textureStore(scatter_out, coord, vec4<f32>(0.0));
        return;
    }

    // Jitter within the froxel, varying per frame — the temporal blend then
    // averages many positions and the fixed slice boundaries stop being visible.
    let j = ign(vec2<f32>(gid.xy), fog_globals.frame);
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims.xy);
    let slice_norm = (f32(gid.z) + j) / f32(dims.z);

    let p = froxel_world_pos(uv, slice_norm);
    let ray_dir = normalize(p - camera.position_near.xyz);

    let density = density_at(p);

    var scattering = vec3<f32>(0.0);
    if density > 0.0 {
        var radiance = vec3<f32>(0.0);
        for (var li = 0u; li < fog_globals.light_count; li++) {
            radiance += inscatter_from_light(li, p, ray_dir);
        }
        // fog_color is the medium's albedo — what in-scattered light bounces off.
        // Emissive is self-illumination and is not tinted by it.
        scattering = (radiance * fog.fog_color + fog.fog_emissive) * density;
    }

    var result = vec4<f32>(scattering, density);

    if fog_globals.history_valid != 0u {
        let hist = sample_history(p);
        if hist.w >= 0.0 {
            result = mix(hist, result, clamp(fog_globals.temporal_blend, 0.0, 1.0));
        }
    }

    textureStore(scatter_out, coord, result);
}

// ── Integration ─────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn cs_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(integrated_out);
    if gid.x >= dims.x || gid.y >= dims.y { return; }

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(dims.xy);

    // Slice planes are constant view depth, so a ray at the screen edge travels
    // further between two slices than one down the centre. Without this the fog
    // thins toward the corners.
    let ndc = helio_uv_to_ndc(uv);
    let p_near = camera.view_proj_inv * vec4<f32>(ndc, 0.0, 1.0);
    let p_far  = camera.view_proj_inv * vec4<f32>(ndc, 1.0, 1.0);
    let dir = normalize(p_far.xyz / p_far.w - p_near.xyz / p_near.w);
    let cos_a = max(dot(dir, normalize(camera.forward_far.xyz)), 1e-4);

    var accum = vec3<f32>(0.0);
    var transmittance = 1.0;
    var prev_depth = HELIO_FROXEL_NEAR;

    for (var z = 0u; z < dims.z; z++) {
        let slice_norm = (f32(z) + 1.0) / f32(dims.z);
        let depth = helio_froxel_view_depth_from_slice(slice_norm, fog.fog_max_distance);
        let step_len = max(depth - prev_depth, 0.0) / cos_a;
        prev_depth = depth;

        let s = textureLoad(scatter_in, vec3<i32>(i32(gid.x), i32(gid.y), i32(z)), 0);
        let scattering = s.rgb;
        let sigma_t = max(s.a, 0.0);

        let step_transmittance = exp(-sigma_t * step_len);

        // Hillaire's analytic slice integral: the exact integral of scattering
        // over a segment of constant density, rather than a point sample scaled
        // by step length. Point-sampling over-brightens as density rises, because
        // it has no saturation term.
        let s_int = (scattering - scattering * step_transmittance) / max(sigma_t, 1e-5);

        accum += transmittance * s_int;
        transmittance *= step_transmittance;

        textureStore(
            integrated_out,
            vec3<i32>(i32(gid.x), i32(gid.y), i32(z)),
            vec4<f32>(accum, transmittance),
        );
    }
}
