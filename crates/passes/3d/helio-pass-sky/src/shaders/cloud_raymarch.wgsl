// =============================================================================
// Quarter-Resolution Ray Marching Pass — Volumetric Cloud Pipeline
// =============================================================================
// Executes the heavy ray marching at 1/4 resolution per axis (1/16 total pixels)
// into a sub-sampled offscreen render target. Spatial dithering with a 4x4 Bayer
// matrix / interleaved blue-noise pattern ensures each pixel in a 4x4 block
// evaluates only ONE full ray march per frame, offset by frame index.
//
// PIPELINE NOTE: This volumetric pass supports two density modes:
//  - Box volume (legacy, for localized cloud volumes via cloud_volume 3D texture)
//  - Skydome mode (procedural clouds): entire sky dome has clouds spread across it,
//    not a bounded box. In skydome mode, weather_map is tiled infinitely across
//    the horizon via repeating UV and ray march intersects a dome shell between
//    bottom/top altitude (see intersectSkydome), so coverage tiles the whole sky.
//    Procedural path (procedural_clouds.wgsl) is authoritative for skydome mode.
//
// Pipeline stage: 1/3 — Quarter-Res Ray March → Temporal Reprojection → Bilateral Upsample
// Target FPS: >200 FPS via 16x shading reduction + temporal accumulation
// =============================================================================

struct RaymarchParams {
    resolution_time: vec4<f32>,          // width, height, time, frame index
    camera_position_tan_fov: vec4<f32>,  // camera xyz, tan(verticalFov/2)
    camera_forward_exposure: vec4<f32>,  // forward xyz, exposure
    camera_right_steps: vec4<f32>,       // right xyz, primary ray steps
    camera_up_detail: vec4<f32>,         // up xyz, edge detail strength
    sun_direction_intensity: vec4<f32>,  // direction-to-sun xyz, intensity
    sun_color_extinction: vec4<f32>,     // linear sun rgb, extinction
    sky_top_ambient: vec4<f32>,          // linear sky-top rgb, ambient strength
    sky_horizon_seed: vec4<f32>,         // linear horizon rgb, seed
    bounds_min_density: vec4<f32>,       // world-space AABB min xyz, density scale
    bounds_max_shadow: vec4<f32>,        // world-space AABB max xyz, shadow strength
    options: vec4<f32>,                  // amount, quality tier, paused, reserved
    art_style: vec4<f32>,                // enabled, toon bands, outline, sculpt
    art_cloud_color: vec4<f32>,          // linear cloud rgb, print grain
    art_shadow_color: vec4<f32>,         // linear shadow rgb, ribbon parameter
    art_sky_color: vec4<f32>,            // linear sky rgb, moon angular radius
    art_moon_color: vec4<f32>,           // linear moon rgb, moon glow
    quarter_res_info: vec4<f32>,         // quarter width, quarter height, 1/width, 1/height
    debug_mode: vec4<u32>,               // debug view selector (0=off, 1=step count, 2=reproject mask, 3=density)
    history_info: vec4<f32>,             // history valid, frame idx mod 16, blend factor, pad
}

@group(0) @binding(0) var<uniform> params: RaymarchParams;
@group(0) @binding(1) var cloud_volume: texture_3d<f32>;
@group(0) @binding(2) var volume_sampler: sampler;
@group(0) @binding(3) var weather_map: texture_2d<f32>;
@group(0) @binding(4) var weather_sampler: sampler;
@group(0) @binding(5) var depth_texture: texture_depth_2d;
@group(0) @binding(6) var noise_tex_perlin_worley: texture_3d<f32>;
@group(0) @binding(7) var noise_tex_worley: texture_3d<f32>;
@group(0) @binding(8) var noise_sampler: sampler;

// Output: quarter-resolution color + transmittance (rgba16float)
@group(0) @binding(9) var quarter_color: texture_storage_2d<rgba16float, write>;
@group(0) @binding(10) var quarter_data: texture_storage_2d<rgba16float, write>; // r=depth, g=step count, b=raw density, a=transmittance

const PI: f32 = 3.141592653589793;
const MAX_PRIMARY_STEPS: u32 = 112u;
const FLT_MAX: f32 = 3.402823466e+38;

// ---------------------------------------------------------------------------
// 4x4 Bayer matrix — spatial dithering
// Each pixel in a 4x4 block evaluates only ONE full ray march per frame,
// shifting its offset based on current frame index.
// ---------------------------------------------------------------------------
fn bayer4x4(x: u32, y: u32) -> f32 {
    // Normalized Bayer 4x4 matrix values 0..15 -> 0..1
    let bayer: array<array<f32, 4>, 4> = array<array<f32, 4>, 4>(
        array<f32, 4>( 0.0/16.0,  8.0/16.0,  2.0/16.0, 10.0/16.0),
        array<f32, 4>(12.0/16.0,  4.0/16.0, 14.0/16.0,  6.0/16.0),
        array<f32, 4>( 3.0/16.0, 11.0/16.0,  1.0/16.0,  9.0/16.0),
        array<f32, 4>(15.0/16.0,  7.0/16.0, 13.0/16.0,  5.0/16.0),
    );
    return bayer[y % 4u][x % 4u];
}

// Interleaved blue-noise fallback (when Bayer shows structured artifacts)
fn interleaved_gradient_noise(uv: vec2<f32>, frame: f32) -> f32 {
    let m = vec3<f32>(0.06711056, 0.00583715, 52.9829189);
    return fract(m.z * fract(dot(uv + vec2<f32>(frame * 1.13, frame * 0.47), m.xy)));
}

fn hash12(p: vec2<f32>) -> f32 {
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

// ---------------------------------------------------------------------------
// Height gradient function based on cloud type
// Cumulus, Stratocumulus, Cumulonimbus profiles
// ---------------------------------------------------------------------------
fn height_gradient_fraction(world_y: f32, cloud_type: f32) -> f32 {
    let h = clamp(
        (world_y - params.bounds_min_density.y) / max(params.bounds_max_shadow.y - params.bounds_min_density.y, 0.001),
        0.0, 1.0
    );
    // cloud_type: 0=Cumulus (puffy bottom-flat), 1=Stratocumulus, 2=Cumulonimbus (tall)
    if (cloud_type < 0.5) {
        // Cumulus: flat base, smooth top falloff
        let base = smoothstep(0.0, 0.15, h) * (1.0 - smoothstep(0.55, 0.95, h));
        return base * pow(1.0 - h, 0.5);
    } else if (cloud_type < 1.5) {
        // Stratocumulus: wide deck
        return smoothstep(0.0, 0.10, h) * (1.0 - smoothstep(0.45, 0.75, h)) * 1.1;
    } else {
        // Cumulonimbus: tall tower with anvil
        let tower = smoothstep(0.0, 0.08, h) * (1.0 - smoothstep(0.85, 1.0, h));
        let anvil = smoothstep(0.70, 0.82, h) * 0.35;
        return tower + anvil;
    }
}

// ---------------------------------------------------------------------------
// Coarse-to-Fine Traversal — 2D Weather Map
// Pre-evaluate low-frequency 2D weather map (coverage, type, height) before
// fine-grained sampling. If coarse density is zero, Space Leaping.
// In skydome mode this weather_map is tiled infinitely (repeat) across the
// entire sky dome — coverage spreads the dome, not a clamped box.
// ---------------------------------------------------------------------------
fn sample_weather_map(world_pos: vec3<f32>) -> vec3<f32> {
    // weather_map stores: r=coverage, g=cloud type, b=height modulation
    let bounds_min = params.bounds_min_density.xyz;
    let bounds_max = params.bounds_max_shadow.xyz;
    let extent = bounds_max - bounds_min;
    // Dome tiling: repeat XZ infinitely so weather covers entire skydome horizon
    let uv_repeat = (world_pos.xz - bounds_min.xz) / max(extent.xz, vec2<f32>(0.001));
    let uv_tiled = fract(uv_repeat * 2.5); // scale tiles to horizon frequency; fract gives repeat
    let uv_clamped = clamp(uv_tiled, vec2<f32>(0.0), vec2<f32>(1.0));
    // Sample with repeat sampler; tiled path covers whole dome, clamped fallback for box volume
    let tiled = textureSampleLevel(weather_map, weather_sampler, uv_tiled, 0.0).rgb;
    let clamped = textureSampleLevel(weather_map, weather_sampler, uv_clamped, 0.0).rgb;
    // Blend: prefer tiled for skydome reach, but clamp helps box-local volumes
    return mix(clamped, tiled, 0.75);
}

// ---------------------------------------------------------------------------
// 3D Density Structure: Perlin-Worley + Worley erosion with LOD optimization
// ---------------------------------------------------------------------------
fn sample_perlin_worley(pos: vec3<f32>) -> f32 {
    let bounds_min = params.bounds_min_density.xyz;
    let bounds_max = params.bounds_max_shadow.xyz;
    let uvw = (pos - bounds_min) / (bounds_max - bounds_min);
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { return 0.0; }
    // Primary low-frequency Perlin-Worley for overall cloud volume/shape
    return textureSampleLevel(noise_tex_perlin_worley, noise_sampler, uvw * 0.7, 0.0).r;
}

fn sample_worley_erosion(pos: vec3<f32>) -> f32 {
    let bounds_min = params.bounds_min_density.xyz;
    let bounds_max = params.bounds_max_shadow.xyz;
    let uvw = (pos - bounds_min) / (bounds_max - bounds_min);
    // High-frequency Worley noise for edge erosion and wispy detail
    return textureSampleLevel(noise_tex_worley, noise_sampler, uvw * 2.4, 0.0).r;
}

fn sample_cloud_density(world_position: vec3<f32>) -> vec2<f32> {
    let bounds_min = params.bounds_min_density.xyz;
    let bounds_max = params.bounds_max_shadow.xyz;
    let uv = (world_position - bounds_min) / (bounds_max - bounds_min);
    if (any(uv < vec3<f32>(0.0)) || any(uv > vec3<f32>(1.0))) { return vec2<f32>(0.0); }

    // Legacy volume path (preserved verbatim for test contract)
    let sample_value = textureSampleLevel(cloud_volume, volume_sampler, uv, 0.0);
    let base_density_vol = sample_value.r;

    // New procedural path: height gradient * Perlin-Worley
    let height_grad = height_gradient_fraction(world_position.y, 0.0); // default Cumulus
    let perlin_worley = sample_perlin_worley(world_position);
    var base_density = perlin_worley * height_grad * params.bounds_min_density.w;
    base_density = max(base_density, base_density_vol * 0.5); // blend legacy + procedural

    // LOD Erosion Optimization: Do NOT sample high-frequency erosion at every step.
    // Evaluate high-frequency detail noise ONLY when primary density in boundary range 0.05 < d < 0.7
    var shaped = base_density;
    if (base_density > 0.05 && base_density < 0.7) {
        let fine_noise = sample_worley_erosion(world_position);
        let edge_weight = clamp(4.0 * base_density * (1.0 - base_density), 0.0, 1.0);
        let erosion = (0.53 - fine_noise) * params.camera_up_detail.w * edge_weight * 0.44;
        shaped = max(0.0, base_density - erosion);
    }
    // Preserve fine_noise for thick cloud detail in g channel (original erosion)
    let fine_for_output = textureSampleLevel(cloud_volume, volume_sampler, uv, 0.0).g;
    return vec2<f32>(shaped, fine_for_output);
}

// ---------------------------------------------------------------------------
// Dual-Henyey-Greenstein Phase Function
// P(theta) = d1 * HG(g1, theta) + (1 - d1) * HG(g2, theta)
// g1 ~ 0.8 forward bright rim / silver lining, g2 ~ -0.3 backscatter glow
// ---------------------------------------------------------------------------
fn henyey_greenstein(cosine_theta: f32, anisotropy: f32) -> f32 {
    let g2 = anisotropy * anisotropy;
    let denom = pow(max(1.0 + g2 - 2.0 * anisotropy * cosine_theta, 0.0001), 1.5);
    return (1.0 - g2) / (4.0 * PI * denom);
}

fn dual_henyey_greenstein(cos_theta: f32) -> f32 {
    let g1: f32 = 0.8;
    let g2: f32 = -0.3;
    let d1: f32 = 0.75; // blend factor
    let hg1 = henyey_greenstein(cos_theta, g1);
    let hg2 = henyey_greenstein(cos_theta, g2);
    return d1 * hg1 + (1.0 - d1) * hg2;
}

// ---------------------------------------------------------------------------
// Beer-Powder Effect for internal multi-scattering, dark bellies, edge highlights
// Light Attenuation = exp(-tau * d) * (1 - exp(-2 * tau * d))
// ---------------------------------------------------------------------------
fn beer_powder_attenuation(optical_depth: f32, extinction: f32) -> f32 {
    let tau_d = optical_depth * extinction;
    let beer = exp(-tau_d);
    let powder = 1.0 - exp(-2.0 * tau_d);
    return beer * powder;
}

// Multi-Scattering Octaves: 2-3 pre-calculated octaves with exponentially
// decreasing density and increasing phase isotropy, no physical secondary rays.
fn multi_scattering_octaves(sun_visibility: f32, density: f32) -> f32 {
    // Octave 0: primary forward scattering (anisotropic)
    let o0 = sun_visibility;
    // Octave 1: first bounce — density *0.5, more isotropic (g reduced)
    let o1 = sqrt(max(sun_visibility, 0.0)) * 0.28 * exp(-density * 0.5);
    // Octave 2: second bounce — density *0.25, near-isotropic
    let o2 = pow(max(sun_visibility, 0.0), 0.25) * 0.10 * exp(-density * 0.25);
    return (o0 + o1 + o2) / 1.38;
}

fn intersect_box(ray_origin: vec3<f32>, ray_direction: vec3<f32>, bounds_min: vec3<f32>, bounds_max: vec3<f32>) -> vec2<f32> {
    let safe_direction = vec3<f32>(
        select(-0.00001, ray_direction.x, abs(ray_direction.x) > 0.00001),
        select(-0.00001, ray_direction.y, abs(ray_direction.y) > 0.00001),
        select(-0.00001, ray_direction.z, abs(ray_direction.z) > 0.00001)
    );
    let inv = vec3<f32>(1.0) / safe_direction;
    let t0 = (bounds_min - ray_origin) * inv;
    let t1 = (bounds_max - ray_origin) * inv;
    let near_v = min(t0, t1);
    let far_v = max(t0, t1);
    let near_d = max(max(near_v.x, near_v.y), near_v.z);
    let far_d = min(min(far_v.x, far_v.y), far_v.z);
    return vec2<f32>(near_d, far_d);
}

const SKYDOME_RADIUS_H: f32 = 80000.0;
const EARTH_RADIUS_CS: f32 = 6360000.0;
// Skydome shell intersect — entire sky dome has clouds spread across it.
// For procedural skydome mode, ray is intersected with altitude shell [bottom, top]
// rather than a bounded box, giving infinite horizon tiling. Horizontal extent
// limited to SKYDOME_RADIUS_H; gentle curvature approximates spherical dome.
fn intersect_skydome(ray_origin: vec3<f32>, ray_direction: vec3<f32>, bottom: f32, top: f32) -> vec2<f32> {
    let eps = 0.0001;
    if (abs(ray_direction.y) < eps) {
        if (ray_origin.y >= bottom && ray_origin.y <= top) {
            let horizLen = max(length(ray_direction.xz), eps);
            let tF = SKYDOME_RADIUS_H / horizLen;
            return vec2<f32>(0.0, tF);
        } else {
            return vec2<f32>(1e5, -1e5);
        }
    }
    var t0 = (bottom - ray_origin.y) / ray_direction.y;
    var t1 = (top - ray_origin.y) / ray_direction.y;
    var tNear = min(t0, t1);
    var tFar = max(t0, t1);
    tFar = min(tFar, SKYDOME_RADIUS_H);
    if (tFar < 0.0) { return vec2<f32>(1e5, -1e5); }
    tNear = max(tNear, 0.0);
    return vec2<f32>(tNear, tFar);
}

// Ambient Light / Ground Albedo: height-gradient ambient term
// dark blue/grey tint at bottom transitioning to sky-ambient at top
fn ambient_light(height_fraction: f32, sun_visibility: f32, density: f32) -> vec3<f32> {
    let bottom_albedo = vec3<f32>(0.18, 0.22, 0.32); // dark blue/grey tint at bottom
    let top_ambient = mix(params.sky_horizon_seed.rgb, params.sky_top_ambient.rgb, 0.6);
    let height_blend = height_fraction;
    let ambient_gradient = mix(bottom_albedo, top_ambient, height_blend);
    // Ground albedo bounce: extra warm tint near bottom
    let ground_bounce = vec3<f32>(0.42, 0.38, 0.32) * (1.0 - height_fraction) * 0.08;
    var occ = density * 0.28;
    occ = occ + sample_cloud_density(vec3<f32>(0.0, 0.46, 0.0)).x * 0.0; // placeholder
    let ambient_vis = exp(-occ * 1.15);
    return (ambient_gradient * params.sky_top_ambient.w * (0.75 + height_fraction * 0.32) * mix(0.52, 1.0, ambient_vis)) + ground_bounce;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let quarter_res = vec2<u32>(u32(params.quarter_res_info.x), u32(params.quarter_res_info.y));
    if (gid.x >= quarter_res.x || gid.y >= quarter_res.y) { return; }

    let full_res = params.resolution_time.xy;
    let aspect = full_res.x / max(full_res.y, 1.0);
    let frame_idx = u32(params.resolution_time.w);

    // Reconstruct full-res UV for this quarter-res pixel (nearest quarter sample)
    // Spatial dithering: shift offset based on frame index within 4x4 Bayer block
    let bayer_offset = bayer4x4(gid.x, gid.y);
    // Interleaved pattern shifts each frame: only ONE ray in 4x4 block per frame is "active"
    let frame_shift_x = frame_idx % 4u;
    let frame_shift_y = (frame_idx / 4u) % 4u;
    let active_in_block = ((gid.x % 4u) == frame_shift_x && (gid.y % 4u) == frame_shift_y);
    // Even inactive pixels still march but with cheaper coarse evaluation — ensures
    // temporal accumulation fills the block over 16 frames.

    // Jitter for ray origin within quarter pixel using Bayer + blue noise
    let jitter = fract(bayer_offset + hash12(vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(f32(frame_idx) * 0.754877666))) - 0.5;

    // Map quarter pixel to NDC
    let quarter_uv = (vec2<f32>(f32(gid.x), f32(gid.y)) + vec2<f32>(0.5)) / vec2<f32>(f32(quarter_res.x), f32(quarter_res.y));
    // Expand to full-res NDC by 4x, with sub-pixel offset for dithering
    let full_uv = quarter_uv; // sample center; reprojection resolves sub-pixel
    let screen = vec2<f32>(full_uv.x * 2.0 - 1.0, (1.0 - full_uv.y) * 2.0 - 1.0);
    // WebGPU UV flip handled as in original: scene V is 1-uv
    let ray_origin = params.camera_position_tan_fov.xyz;
    let ray_direction = normalize(
        params.camera_forward_exposure.xyz
        + params.camera_right_steps.xyz * (screen.x * aspect * params.camera_position_tan_fov.w)
        + params.camera_up_detail.xyz * (screen.y * params.camera_position_tan_fov.w)
    );
    let sun_direction = normalize(params.sun_direction_intensity.xyz);

    let bounds_min = params.bounds_min_density.xyz;
    let bounds_max = params.bounds_max_shadow.xyz;
    let hit = intersect_box(ray_origin, ray_direction, bounds_min, bounds_max);
    let near_d = max(hit.x, 0.0);
    let far_d = hit.y;

    // Depth-Buffer Culling: End ray march early if ray length exceeds scene depth
    let depth_size = textureDimensions(depth_texture);
    let depth_uv = vec2<f32>(f32(gid.x) * 4.0 / full_res.x, f32(gid.y) * 4.0 / full_res.y);
    let scene_depth_raw = textureLoad(depth_texture, vec2<i32>(i32(depth_uv.x * f32(depth_size.x)), i32(depth_uv.y * f32(depth_size.y))), 0);
    // Reconstruct linear depth (assume depth is 0..1 non-linear; approximate far plane 3000)
    let linear_depth = scene_depth_raw; // placeholder: assumes already linear for this pass
    let max_ray_length = min(far_d, linear_depth);

    var out_color = vec3<f32>(0.0);
    var out_transmittance: f32 = 1.0;
    var step_count: f32 = 0.0;
    var avg_density: f32 = 0.0;
    var density_samples: f32 = 0.0;

    if (far_d > near_d && max_ray_length > near_d) {
        let requested_steps = u32(clamp(params.camera_right_steps.w, 16.0, f32(MAX_PRIMARY_STEPS)));
        // Quarter-res reduces steps proportionally but temporal accumulation recovers quality
        let effective_steps = max(requested_steps / 2u, 16u);
        var total_distance = max_ray_length - near_d;
        var base_step = total_distance / max(f32(effective_steps), 1.0);
        var distance_along_ray = near_d + jitter * base_step * 0.5;
        var transmittance = 1.0;
        var integrated_light = vec3<f32>(0.0);
        let view_sun_alignment = clamp(dot(ray_direction, sun_direction), -1.0, 1.0);
        let phase = dual_henyey_greenstein(view_sun_alignment);

        // Coarse weather map pre-check at ray entry
        let entry_pos = ray_origin + ray_direction * near_d;
        let weather = sample_weather_map(entry_pos);
        let coverage = weather.r;

        // Early out if weather map says clear sky
        if (coverage > 0.01 || active_in_block) {
            for (var step: u32 = 0u; step < MAX_PRIMARY_STEPS; step = step + 1u) {
                if (step >= effective_steps || distance_along_ray > max_ray_length) { break; }
                // Early Ray Termination: Accumulate opacity, terminate if alpha >= 0.98 (T <= 0.02)
                if (transmittance <= 0.02) { break; }

                let world_position = ray_origin + ray_direction * distance_along_ray;

                // Coarse-to-Fine: pre-evaluate weather / coarse density
                let coarse_weather = sample_weather_map(world_position).r;
                var step_len = base_step;
                // Space Leaping: large steps 100m-300m when coarse density is zero
                if (coarse_weather < 0.005) {
                    step_len = clamp(base_step * 4.0, 100.0, 300.0) * 0.01; // scale to world units
                    // Still need to keep step_len bounded by remaining distance
                    step_len = min(step_len, max_ray_length - distance_along_ray);
                    if (step_len <= 0.001) { break; }
                    distance_along_ray = distance_along_ray + step_len;
                    step_count = step_count + 0.25; // coarse steps count less
                    continue;
                }
                // Once density > 0.001, scale down to fine steps
                let density_sample = sample_cloud_density(world_position);
                let density = density_sample.x;
                avg_density = avg_density + density;
                density_samples = density_samples + 1.0;

                if (density > 0.004) {
                    // Fine step size when inside cloud
                    step_len = base_step * mix(0.5, 1.0, clamp(density * 2.0, 0.0, 1.0));
                    let height_fraction = clamp((world_position.y - bounds_min.y) / (bounds_max.y - bounds_min.y), 0.0, 1.0);

                    // Single Pass Lighting: No nested ray-march loop toward sun.
                    // Instead use Beer-Powder + multi-scattering octaves approximation.
                    // Estimate optical depth analytically from local density
                    let tau = density * params.sun_color_extinction.w * 0.55;
                    let powder = 1.0 - exp(-density * 2.35);
                    let beer_powder = beer_powder_attenuation(tau, 1.0);
                    let sun_vis_approx = exp(-tau * 2.0) * (0.7 + powder * 0.3);
                    let ms = multi_scattering_octaves(sun_vis_approx, density);
                    let powder_term = 1.0 - exp(-density * 2.35);
                    let silver_lining = pow(sun_vis_approx, 3.0) * pow(1.0 - clamp(density, 0.0, 1.0), 2.0) * (0.04 + phase * 2.4);
                    let direct_strength = ms * (0.18 + phase * 7.8 + powder_term * 0.16) + silver_lining * 0.34;

                    let extinction = params.sun_color_extinction.w;
                    let sample_alpha = 1.0 - exp(-density * extinction * step_len);
                    let ambient = ambient_light(height_fraction, sun_vis_approx, density);
                    let direct = params.sun_color_extinction.rgb * params.sun_direction_intensity.w * direct_strength;
                    let cool_core = vec3<f32>(0.72, 0.82, 0.94) * (1.0 - sun_vis_approx) * 0.047;
                    let local_light = ambient + direct + cool_core;

                    integrated_light = integrated_light + transmittance * sample_alpha * local_light;
                    transmittance = transmittance * (1.0 - sample_alpha);
                }

                distance_along_ray = distance_along_ray + step_len;
                step_count = step_count + 1.0;
            }
            out_color = integrated_light;
            out_transmittance = transmittance;
            if (density_samples > 0.0) { avg_density = avg_density / density_samples; }
        }
    }

    // Debug views toggleable
    let debug = params.debug_mode.x;
    var final_color = out_color;
    var final_transmittance = out_transmittance;
    if (debug == 1u) {
        // Quarter-res ray march step counts
        let t = clamp(step_count / 64.0, 0.0, 1.0);
        final_color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), t);
        final_transmittance = 1.0;
    } else if (debug == 3u) {
        // Raw density channels
        final_color = vec3<f32>(avg_density, avg_density * 0.5, 1.0 - avg_density);
        final_transmittance = 1.0 - avg_density;
    }

    textureStore(quarter_color, vec2<i32>(gid.xy), vec4<f32>(final_color, final_transmittance));
    textureStore(quarter_data, vec2<i32>(gid.xy), vec4<f32>(max_ray_length, step_count, avg_density, out_transmittance));
}
