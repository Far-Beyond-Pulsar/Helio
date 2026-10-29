//!use helio_prelude
//!use water_shore_surface
//!use helio_hiz
//!use water_query
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
//   0  cameras          storage  array<Camera, 2> (prelude layout)
//   1  water_volumes    storage read
//   2  ocean_displacement texture_2d_array<f32> (four Ultra-Water cascade fields)
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
    water_color:           vec4f,  // xyz = medium colour, w = Jacobian foam bias
    extinction:            vec4f,  // xyz = absorption per metre, w = Jacobian foam scale
    reflection_refraction: vec4f,  // x = reflection_str, y = refraction_str, z = fresnel_power, w = foam brightness
    caustics_params:       vec4f,
    fog_params:            vec4f,
    sim_params:            vec4f,  // x = ior, y = caustic_intensity, z = fresnel_min, w = density
    shadow_params:         vec4f,
    sun_direction:         vec4f,
    ssr_params:            vec4f,  // x = enable, y = max_steps, z = step_size, w = thickness
    sim_dynamics:          vec4f,
    wind_params:           vec4f,
    world_phase:           vec4f,  // xy = the frame's world origin XZ mod 270 m (GpuWaterVolume::world_phase)
}

@group(0) @binding(0) var<storage, read> cameras: array<Camera, 2>;
@group(0) @binding(1) var<storage, read> volumes:        array<WaterVolume>;
@group(0) @binding(2) var ocean_displacement: texture_2d_array<f32>;
@group(0) @binding(3) var water_samp:     sampler;
@group(0) @binding(4) var caustics_tex:   texture_2d<f32>;
@group(0) @binding(5) var shared_samp:    sampler;
@group(0) @binding(6) var scene_color:    texture_2d<f32>;
@group(0) @binding(7) var<uniform>        viewport:      vec4f;
@group(0) @binding(8) var depth_texture:  texture_depth_2d;
@group(0) @binding(9) var depth_sampler:  sampler;
@group(0) @binding(10) var gbuffer_normal: texture_2d<f32>;
@group(0) @binding(11) var hiz_nearest:    texture_2d<f32>;
@group(0) @binding(12) var ocean_slope_moments: texture_2d_array<f32>;
@group(0) @binding(13) var water_interaction: texture_2d_array<f32>;
struct OceanPhase { patches: array<vec4f, 4> }
@group(0) @binding(14) var<uniform> ocean_phase: OceanPhase;
@group(0) @binding(17) var foam_material_tex: texture_2d<f32>;
@group(0) @binding(18) var foam_material_samp: sampler;
@group(0) @binding(19) var<storage, read> reflection_cameras: array<Camera, 1>;
@group(0) @binding(20) var planar_reflection_tex: texture_2d<f32>;
@group(0) @binding(23) var caustics_deep_tex: texture_2d<f32>;
@group(0) @binding(24) var caustics_samp: sampler;

const IOR_AIR: f32 = 1.0;

const OCEAN_PATCH_SIZES: array<f32, 4> = array(600.0, 85.0, 12.0, 2.1);
const OCEAN_GRID_ANGULAR: u32 = 384u;

/// Ceiling on refractive displacement, in metres. Long grazing paths would
/// otherwise ask for an offset large enough to smear unrelated parts of the
/// screen into the water.
const REFRACTION_MAX_OFFSET: f32 = 2.0;

struct VertexOutput {
    @builtin(position) position:  vec4f,
    @location(0)       world_pos: vec3f,
    @location(1) @interpolate(flat) vol_index: u32,
    @location(2) cell_size: f32,
}

// ── Volume geometry ──────────────────────────────────────────────────────────

/// Wave amplitude in world units.
///
/// Comes from the descriptor (`wave_amplitude`), NOT from the volume's vertical
/// extent — deepening a volume must not make its waves taller. Clamped to the
/// headroom on either side of the rest height so the displaced surface can never
/// leave the volume bounds.
/// Normalized sim height → world Y. The single point of truth for where the
/// surface is; every consumer must go through this.
fn water_surface_height(sim_h: f32, vol: WaterVolume) -> f32 {
    // The FFT output is already expressed in metres, as is Ultra-Water's field.
    return vol.bounds_max.w + sim_h;
}

fn sim_layer(vol_index: u32, cascade: u32) -> u32 {
    return vol_index * 3u + cascade;
}

/// Convert frame-relative coordinates into each FFT patch's world phase. CPU
/// preparation reduces the f64 world origin independently by each patch length,
/// so camera-relative rebases cannot jump any of the four spectra.
fn water_tiling_xz(world_xz: vec2f, cascade: u32) -> vec2f {
    return world_xz + ocean_phase.patches[cascade].xy;
}

/// Sum displacement from the three geometric cascades. The 2.1 m capillary
/// cascade supplies slope variance only; Ultra-Water intentionally excludes it
/// from geometry and its Jacobian foam. Then add localized interactions.
fn water_cascade_displacement(world_xz: vec2f, cell_size: f32) -> vec3f {
    return water_geometric_displacement(world_xz, cell_size, false);
}

fn water_cascade_height_sum(world_xz: vec2f, vol_index: u32, cell_size: f32) -> f32 {
    let vol = volumes[vol_index];
    return water_surface_height_offset(world_xz, vol, vol_index, cell_size, false);
}

/// World-space surface normal from all cascades, computed by summing
/// per-cascade slopes (reconstructed from the stored normal) weighted by
/// each cascade's amplitude fraction, then normalizing.
fn water_total_normal(world_xz: vec2f, vol: WaterVolume, vol_index: u32, cell_size: f32) -> vec3f {
    var total_slope = vec2f(0.0);
    let shore_damp = shore_ocean_damp(world_xz);
    for (var i = 0u; i < 4u; i++) {
        let patch_size = OCEAN_PATCH_SIZES[i];
        let uv = fract(water_tiling_xz(world_xz, i) / patch_size);
        let lod = max(log2(max(cell_size, 1e-4) * 256.0 / patch_size) + 1.0, 0.0);
        total_slope += textureSampleLevel(ocean_slope_moments, water_samp, uv, i, lod).xy
            * shore_damp[i];
    }
    let interaction_uv = water_world_xz_to_sim_uv(world_xz, vol);
    if all(interaction_uv >= vec2f(0.0)) && all(interaction_uv <= vec2f(1.0)) {
        let interaction = textureSampleLevel(
            water_interaction, water_samp, interaction_uv, sim_layer(vol_index, 0u), 0.0);
        let ny = sqrt(max(1.0 - dot(interaction.ba, interaction.ba), 1e-6));
        let extent = max(vol.bounds_max.xz - vol.bounds_min.xz, vec2f(1e-4));
        total_slope += (interaction.ba / ny) * (water_wave_amplitude(vol) / extent);
    }
    total_slope += shore_wave(world_xz).slope;
    return normalize(vec3f(total_slope.x, 1.0, total_slope.y));
}

/// World XZ → grid UV in [0,1].
fn water_world_xz_to_sim_uv(world_xz: vec2f, vol: WaterVolume) -> vec2f {
    let extent = max(vol.bounds_max.xz - vol.bounds_min.xz, vec2f(1e-4));
    return (world_xz - vol.bounds_min.xz) / extent;
}

// ── Medium ───────────────────────────────────────────────────────────────────

/// How far a view ray travels through water before it hits something.
///
/// Distance to opaque scene depth, capped by an extinction-based open-water
/// distance. The legacy volume AABB does not bound a continuous ocean.
fn water_path_length(
    screen_uv:   vec2f,
    surface_pos: vec3f,
    vol:         WaterVolume,
) -> f32 {
    // This is a continuous ocean, so the old water-volume AABB cannot act as
    // its floor or path-length limit. Use opaque scene depth when available;
    // open-water rays use a bounded path long enough to saturate the medium.
    let weakest_extinction = max(min(vol.extinction.r, min(vol.extinction.g, vol.extinction.b)), 1e-4);
    let max_path = clamp(6.0 / weakest_extinction, 100.0, 10000.0);

    // Distance to the opaque scene behind the surface.
    var t_scene = max_path;
    let depth = textureSampleLevel(depth_texture, depth_sampler, screen_uv, 0);
    if !helio_is_sky(depth) {
        let scene_pos = helio_world_from_depth(cameras[0].view_proj_inv, screen_uv, depth);
        t_scene = distance(scene_pos, surface_pos);
    }

    return clamp(t_scene, 0.0, max_path);
}

/// Displaced surface height above an arbitrary world XZ (not just the shaded
/// vertex), for working out how deep a submerged point sits.
fn water_surface_at(world_xz: vec2f, vol: WaterVolume, vol_index: u32) -> f32 {
    return water_surface_height_at(world_xz, vol, vol_index, 1.0, false, 4u);
}

/// Caustic light landing on a submerged point.
///
/// Sample the shared 12 m FFT caustics fields at the receiver's world position.
/// Near and deep targets have isolated repeat/mip chains, so their depth slices
/// filter independently. LOD follows the projected receiver footprint.
fn water_caustics(scene_pos: vec3f, vol: WaterVolume, screen_uv: vec2f, vol_index: u32) -> vec3f {
    if vol.caustics_params.x <= 0.5 {
        return vec3f(0.0);
    }

    let depth_below = water_surface_at(scene_pos.xz, vol, vol_index) - scene_pos.y;
    if depth_below <= 0.0 {
        return vec3f(0.0);
    }

    let tile_uv = fract((scene_pos.xz + ocean_phase.patches[2].xy) / OCEAN_PATCH_SIZES[2]);
    let depth = clamp(depth_below, 1.5, 10.0);
    let footprint = max(length(dpdx(scene_pos.xz)), length(dpdy(scene_pos.xz)));
    let near_lod = clamp(log2(max(footprint * 512.0 / 12.0, 1.0)), 0.0, 9.0);
    let deep_lod = clamp(log2(max(footprint * 256.0 / 12.0, 1.0)), 0.0, 8.0);
    var caustic: f32;
    if depth <= 4.0 {
        let shallow = textureSampleLevel(caustics_tex, caustics_samp,
            tile_uv, near_lod).r;
        let mid = textureSampleLevel(caustics_deep_tex, caustics_samp,
            tile_uv, deep_lod).r;
        caustic = mix(shallow, mid, smoothstep(1.5, 4.0, depth));
    } else {
        let mid = textureSampleLevel(caustics_deep_tex, caustics_samp,
            tile_uv, deep_lod).r;
        let deep = textureSampleLevel(caustics_deep_tex, caustics_samp,
            tile_uv, deep_lod).g;
        caustic = mix(mid, deep, smoothstep(4.0, 10.0, depth));
    }
    caustic = max(caustic - 1.0, 0.0) * max(vol.caustics_params.y, 0.0);

    let recv_n = helio_gbuffer_normal(
        textureSampleLevel(gbuffer_normal, shared_samp, screen_uv, 0.0).xyz);
    let facing = clamp(recv_n.y, 0.0, 1.0);

    return vec3f(caustic) * facing * exp(-depth_below * 0.12);
}

/// Beer-Lambert transmittance over a path.
fn water_transmittance(path: f32, vol: WaterVolume) -> vec3f {
    return exp(-max(vol.extinction.rgb, vec3f(1e-4)) * path);
}

/// Ultra-Water's roughness-aware Fresnel (Lagarde): unresolved slope variance
/// lowers F90 in the far field, avoiding an unnaturally white horizon.
fn water_fresnel(normal: vec3f, view_dir: vec3f, alpha: f32) -> f32 {
    let no_v = max(dot(normal, -view_dir), 0.0);
    let f0 = 0.02;
    let f90 = max(1.0 - 0.7 * sqrt(clamp(alpha, 0.0, 1.0)), f0);
    return f0 + (f90 - f0) * pow(1.0 - no_v, 5.0);
}

/// Beckmann-style slope variance encoded alongside the FFT slopes. The
/// cascade mips already filter unresolved waves; subtracting the local mean
/// slope leaves variance suitable for specular anti-aliasing.
fn ocean_slope_variance(world_xz: vec2f, cell_size: f32, damp: vec4f) -> vec2f {
    var variance = vec2f(0.0);
    for (var i = 0u; i < 4u; i++) {
        let patch_size = OCEAN_PATCH_SIZES[i];
        let uv = fract(water_tiling_xz(world_xz, i) / patch_size);
        let lod = max(log2(max(cell_size, 1e-4) * 256.0 / patch_size) + 1.0, 0.0);
        let moments = textureSampleLevel(ocean_slope_moments, water_samp, uv, i, lod);
        let slope = moments.xy * damp[i];
        let second = moments.zw * damp[i] * damp[i];
        variance += max(second - slope * slope, vec2f(0.0));
    }
    return variance;
}

/// GGX sun lobe with Smith's height-correlated visibility term. Alpha includes
/// the FFT slope variance and geometric-normal variance across the pixel.
fn poisson_facet_count(lambda: f32, random_value: f32) -> f32 {
    let u = clamp(random_value, 1e-4, 0.9999);
    if lambda > 6.0 {
        return max(0.0, lambda + sqrt(lambda) * log(u / (1.0 - u)) / 1.702);
    }
    var probability = exp(-lambda);
    var cumulative = probability;
    var count = 0.0;
    for (var k = 0u; k < 14u; k++) {
        if u <= cumulative { break; }
        count += 1.0;
        probability *= lambda / count;
        cumulative += probability;
    }
    return count;
}

fn water_sun_specular(
    normal: vec3f,
    view_dir: vec3f,
    light_dir: vec3f,
    alpha: f32,
    world_xz: vec2f,
    camera_distance: f32,
) -> vec3f {
    let no_v = max(dot(normal, -view_dir), 1e-4);
    let no_l = max(dot(normal, light_dir), 0.0);
    if no_l <= 0.0 { return vec3f(0.0); }
    let half_vec = normalize(light_dir - view_dir);
    let no_h = max(dot(normal, half_vec), 0.0);
    let vo_h = max(dot(-view_dir, half_vec), 0.0);
    let a = max(alpha, 0.015);
    let a2 = a * a;
    let d0 = no_h * no_h * (a2 - 1.0) + 1.0;
    let distribution = a2 / (HELIO_PI * d0 * d0);
    let visibility = 0.5 / (no_l * sqrt(no_v * no_v * (1.0 - a2) + a2)
        + no_v * sqrt(no_l * no_l * (1.0 - a2) + a2) + 1e-5);
    let fresnel = 0.02 + 0.98 * pow(1.0 - vo_h, 5.0);
    var specular = vec3f(distribution * visibility * fresnel * no_l * HELIO_PI);

    // Finite-facet glints: a Poisson draw estimates the number of sun-mirroring
    // microfacets in each pixel footprint. Dividing by the expected count
    // preserves the smooth GGX lobe's energy while letting sparse facets
    // sparkle at distance. Two adjacent spatial LODs and two temporal draws
    // crossfade to avoid grid and frame-boundary popping.
    let footprint = max(max(length(dpdx(world_xz)), length(dpdy(world_xz))), 1e-3);
    if dot(specular, vec3f(1.0)) > 0.02 {
        let facet_density = 2.0e4;
        let screen_lod = log2(footprint) + 1.0;
        let lod0 = floor(screen_lod);
        let lod_blend = fract(screen_lod);
        let expected_per_area = facet_density * distribution * no_h * 6.8e-5
                              / (4.0 * max(vo_h, 0.05));
        var multiplier = 0.0;
        for (var lod_offset = 0u; lod_offset < 2u; lod_offset++) {
            let lod = lod0 + f32(lod_offset);
            let cell_size = exp2(lod);
            let cell = floor(world_xz / cell_size);
            let lambda = expected_per_area * cell_size * cell_size;
            let temporal = cameras[0].jitter_frame.z * 0.048
                         + hash21(cell + vec2f(lod * 17.17)) * 64.0;
            let tick0 = floor(temporal);
            let tick_blend = smoothstep(0.0, 1.0, fract(temporal));
            let count0 = poisson_facet_count(lambda, hash21(cell * 1.31 + vec2f(tick0 * 3.7)));
            let count1 = poisson_facet_count(lambda, hash21(cell * 1.31 + vec2f((tick0 + 1.0) * 3.7)));
            let sampled_count = mix(count0, count1, tick_blend);
            let spatial_weight = select(1.0 - lod_blend, lod_blend, lod_offset == 1u);
            multiplier += spatial_weight * min(sampled_count / max(lambda, 1e-4), 300.0);
        }
        let glint_strength = 0.6 * smoothstep(4.0, 30.0, camera_distance);
        specular *= mix(1.0, multiplier, glint_strength);
    }
    return specular;
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

/// Foam coverage from Ultra-Water's combined damped Jacobian field plus
/// independent shoreline contact foam.
fn water_foam(
    path: f32,
    world_xz: vec2f,
    vol: WaterVolume,
    cell_size: f32,
) -> f32 {
    let amp = max(water_wave_amplitude(vol), 1e-3);
    let t   = cameras[0].jitter_frame.z * 0.016;

    // Two layers scrolling against each other, so the band churns rather than
    // sitting still as a uniform ring. In the waves' tiling space, so the
    // pattern follows the world rather than a camera-relative frame's origin.
    let tiling_xz = water_tiling_xz(world_xz, 0u);
    let n = value_noise(tiling_xz * 0.35 + vec2f( t * 0.20, -t * 0.13)) * 0.65
          + value_noise(tiling_xz * 1.10 + vec2f(-t * 0.31,  t * 0.17)) * 0.35;

    // Contact foam. The band scales with wave size — a bigger swell washes
    // further up the shore — and the noise makes its edge irregular.
    let band       = max(amp * 2.5, 0.35);
    let shore      = 1.0 - smoothstep(0.0, band, path);
    let shore_foam = smoothstep(0.25, 0.85, shore * (0.55 + n * 0.75));

    var jacobian = 0.0;
    let cascade_damp = shore_ocean_damp(world_xz);
    // Ultra-Water aggregates each cascade around the neutral Jacobian value
    // 1.0, weighting it by the same depth damping as its geometric spectrum.
    // Cascade 3 is capillary-only: retain it in the normal/roughness path,
    // but exclude it from this three-cascade foam source.
    for (var cascade = 0u; cascade < 3u; cascade++) {
        let patch_size = OCEAN_PATCH_SIZES[cascade];
        let uv = fract(water_tiling_xz(world_xz, cascade) / patch_size);
        let lod = max(log2(max(cell_size, 1e-4) * 256.0 / patch_size) + 1.0, 0.0);
        let band_jacobian = textureSampleLevel(ocean_displacement, water_samp, uv, cascade, lod).w;
        jacobian += mix(1.0, band_jacobian, cascade_damp[cascade]);
    }
    jacobian -= 2.0;
    let spectral_foam = clamp((vol.water_color.w - jacobian) * vol.extinction.w, 0.0, 1.0);
    return max(shore_foam, spectral_foam);
}

struct FoamMaterial {
    coverage: f32,
    billow: f32,
    gradient: vec2f,
}

/// Erode sparse/aged foam into the tileable lace pattern and recover the
/// prebaked cauliflower billow and slope detail. Dense fresh foam fills in;
/// weaker remnants break into foam filaments and open water.
fn water_foam_material(
    world_xz: vec2f,
    source_strength: f32,
    shore: ShoreWave,
) -> FoamMaterial {
    let t = cameras[0].jitter_frame.z * 0.016;
    let flow = vec2f(0.11, -0.075) * t;
    let primary = textureSample(foam_material_tex, foam_material_samp, (world_xz + flow) * 0.11);
    let broad = textureSample(foam_material_tex, foam_material_samp, (world_xz - flow * 0.43) * 0.031 + vec2f(0.37));
    let pattern = primary.r * 0.65 + broad.r * 0.35;
    // A spatially coherent, advected foam-age field gives fresh crests a
    // filled-in appearance and progressively erodes older remnants into the
    // baked lace filaments. The field is stable in each FFT patch's world
    // tiling coordinate, so camera-origin rebases do not restart foam aging.
    let age_noise = value_noise(water_tiling_xz(world_xz, 1u) * 0.045
        + vec2f(t * 0.035, -t * 0.027));
    let freshness = 1.0 - smoothstep(0.34, 0.91, age_noise);
    let aged_strength = source_strength * mix(0.42, 1.0, freshness);
    var coverage = smoothstep(1.05 - aged_strength, 1.3 - aged_strength, pattern)
                  * sqrt(max(aged_strength, 0.0));
    var billow = primary.g;
    var gradient = (primary.ba - vec2f(0.5)) * 64.0 * 0.11;

    // Fresh breaker foam rides shoreward with the bore. Crossfading two
    // travelling phases avoids a hard reset every 2.5 seconds.
    let breaker = clamp(shore.foam, 0.0, 1.0);
    if breaker > 0.0 {
        let flow_time = 2.5;
        let speed = shore.first_train.z / max(shore.first_train.y, 1e-3);
        let velocity = shore.direction * speed;
        let f0 = fract(shore_params.state.y / flow_time);
        let f1 = fract(f0 + 0.5);
        let blend = abs(2.0 * f0 - 1.0);
        let b0 = textureSample(
            foam_material_tex,
            foam_material_samp,
            (world_xz - velocity * (f0 * flow_time)) * 0.15,
        );
        let b1 = textureSample(
            foam_material_tex,
            foam_material_samp,
            (world_xz - velocity * (f1 * flow_time)) * 0.15 + vec2f(0.43),
        );
        let shore_billow = mix(b0.g, b1.g, blend);
        let shore_pattern = mix(b0.r, b1.r, blend) * 0.3
            + shore_billow * 0.45
            + textureSample(
                foam_material_tex,
                foam_material_samp,
                world_xz * 0.023 + vec2f(0.5),
            ).r * 0.25;
        let shore_coverage = smoothstep(
            1.0 - breaker * 1.1,
            1.2 - breaker * 1.1,
            shore_pattern,
        ) * sqrt(breaker);
        if shore_coverage >= coverage {
            billow = shore_billow;
            gradient = (mix(b0.ba, b1.ba, blend) - vec2f(0.5)) * 64.0 * 0.15;
        }
        coverage = max(coverage, shore_coverage);
    }

    // The breaker leaves a slower, thinner lace trail. Positive phase wrapping
    // is handled when the residual source is formed in shore_wave; this flow
    // uses fract so negative times also remain in [0, 1).
    let residual = clamp(shore.residual, 0.0, 1.0);
    if residual > 0.0 {
        let flow_time = 8.0;
        let f0 = fract(shore_params.state.y / flow_time);
        let f1 = fract(f0 + 0.5);
        let blend = abs(2.0 * f0 - 1.0);
        let drift = shore.direction * 2.8;
        let r0 = textureSample(
            foam_material_tex,
            foam_material_samp,
            (world_xz - drift * f0) * 0.06,
        ).r;
        let r1 = textureSample(
            foam_material_tex,
            foam_material_samp,
            (world_xz - drift * f1) * 0.06 + vec2f(0.5),
        ).r;
        let residual_pattern = mix(r0, r1, blend);
        let residual_coverage = smoothstep(
            1.2 - residual,
            1.35 - residual,
            residual_pattern,
        ) * residual;
        if residual_coverage >= coverage {
            billow = 0.5;
            gradient = vec2f(0.0);
        }
        coverage = max(coverage, residual_coverage);
    }

    return FoamMaterial(coverage, billow, gradient);
}

// ── Screen-space reflection ──────────────────────────────────────────────────
// Marches the shared `hiz_nearest` pyramid via `helio_hiz_march` — the same
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

    let near = cameras[0].position_near.w;

    // Build the ray in view space so it can be clipped against the near plane
    // before projection — a ray crossing w = 0 projects to nonsense.
    var start_view  = (cameras[0].view * vec4f(origin, 1.0)).xyz;
    let dir_view    = normalize((cameras[0].view * vec4f(dir, 0.0)).xyz);
    let normal_view = (cameras[0].view * vec4f(normal, 0.0)).xyz;
    start_view += normal_view * (-start_view.z * SSR_NORMAL_BIAS);

    var ray_len = SSR_MAX_DIST;
    if start_view.z + dir_view.z * ray_len > -near {
        ray_len = (-near - start_view.z) / dir_view.z;
    }
    if ray_len <= 0.0 {
        return result;
    }
    let end_view = start_view + dir_view * ray_len;

    let clip0 = cameras[0].proj * vec4f(start_view, 1.0);
    let clip1 = cameras[0].proj * vec4f(end_view, 1.0);
    if clip0.w <= 0.0 || clip1.w <= 0.0 {
        return result;
    }
    let p0 = vec3f(helio_ndc_to_uv(clip0.xy / clip0.w), clip0.z / clip0.w);
    let p1 = vec3f(helio_ndc_to_uv(clip1.xy / clip1.w), clip1.z / clip1.w);

    let march = helio_hiz_march(
        hiz_nearest, p0, p1,
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
    let ray_z     = helio_view_depth(cameras[0].proj, march.pos.z);
    let scene_z   = helio_view_depth(
        cameras[0].proj, textureSampleLevel(depth_texture, depth_sampler, hit_uv, 0));
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

    // The mirrored capture is invalid underwater (or before the first above-
    // water frame). Do not sample a stale prior view; retain the analytic sky
    // baseline and let screen-space hits refine it when enabled.
    let capture_valid = bitcast<u32>(reflection_cameras[0].jitter_frame.w) == 0x57415445u;
    if !capture_valid {
        if vol.ssr_params.x <= 0.5 {
            return sky;
        }
        let hit = trace_ssr(world_pos, ray, normal, vol);
        return mix(sky, hit.color, hit.confidence);
    }

    let capture_clip = reflection_cameras[0].view_proj * vec4f(world_pos, 1.0);
    if capture_clip.w <= 1e-5 {
        return sky;
    }
    let capture_ndc = capture_clip.xyz / capture_clip.w;
    let capture_uv = vec2f(capture_ndc.x * 0.5 + 0.5, 0.5 - capture_ndc.y * 0.5);
    if any(capture_uv < vec2f(0.0)) || any(capture_uv > vec2f(1.0))
        || capture_ndc.z < 0.0 || capture_ndc.z > 1.0 {
        return sky;
    }
    // The prefiltered half-resolution capture softens reflected detail in
    // proportion to the resolved spectral slope, while preserving sharp calm
    // water and keeping high-frequency FFT chop stable.
    let reflection_lod = clamp(length(normal.xz) * 1.7, 0.0, 8.0);
    let planar = textureSampleLevel(planar_reflection_tex, shared_samp, capture_uv, reflection_lod).rgb;

    if vol.ssr_params.x <= 0.5 {
        return planar;
    }

    // Blend by confidence rather than switching, so a partial hit near the
    // screen edge dissolves into sky instead of popping.
    let hit = trace_ssr(world_pos, ray, normal, vol);
    return mix(planar, hit.color, hit.confidence);
}

// ── Vertex ───────────────────────────────────────────────────────────────────

@vertex
fn vs_main(
    @location(0) position: vec4f,
    @builtin(vertex_index) vertex_idx: u32,
    @builtin(instance_index) instance_idx: u32,
) -> VertexOutput {
    let vol = volumes[instance_idx];

    var out: VertexOutput;
    out.vol_index = instance_idx;

    // Polar grid centred on the camera. Its inner rings expand with camera
    // height; the far skirt stays fixed so distant ocean coverage is stable.
    let radius = length(position.xy);
    let camera_height = cameras[0].position_near.y - vol.bounds_max.w;
    let half_density = camera_height > 60.0;
    var local_xz = position.xy;
    // Ultra-Water rebuilds the polar mesh at half angular resolution above
    // 60 m. Collapse every odd angular column onto its preceding even
    // column here, preserving the same camera-height transition while
    // keeping one static vertex/index allocation. Degenerate triangles
    // vanish and each surviving angular cell spans the 192-column grid.
    if half_density && vertex_idx > 0u {
        let angle_index = ((vertex_idx - 1u) % OCEAN_GRID_ANGULAR) & ~1u;
        let angle = f32(angle_index) * (2.0 * HELIO_PI / f32(OCEAN_GRID_ANGULAR));
        local_xz = radius * vec2f(cos(angle), sin(angle));
    }
    let grid_scale = max(1.0, abs(camera_height) * 0.5);
    let scale = select(grid_scale, 1.0, radius > 6000.0);
    let density_scale = select(1.0, 2.0, half_density);
    let xz = cameras[0].position_near.xz + local_xz * scale;
    let cell_size = max(position.z, 1.0 / 16.0) * scale * density_scale;
    let displacement = water_cascade_displacement(xz, cell_size);
    let h_sum = water_cascade_height_sum(xz, instance_idx, cell_size);
    let world = vec3f(
        xz.x + displacement.x,
        vol.bounds_max.w + h_sum,
        xz.y + displacement.z,
    );
    out.position  = cameras[0].view_proj * vec4f(world, 1.0);
    out.world_pos = world;
    out.cell_size = cell_size;
    return out;
}

// ── Fragment: above water ────────────────────────────────────────────────────

fn fs_above(in: VertexOutput) -> vec4f {
    let vol       = volumes[in.vol_index];
    let light_dir = normalize(vol.sun_direction.xyz);

    let normal    = water_total_normal(in.world_pos.xz, vol, in.vol_index, in.cell_size);
    let view_dir  = normalize(in.world_pos - cameras[0].position_near.xyz);
    let screen_uv = in.position.xy * viewport.zw;
    let cascade_damp = shore_ocean_damp(in.world_pos.xz);
    let slope_variance = ocean_slope_variance(in.world_pos.xz, in.cell_size, cascade_damp);
    let normal_dx = dpdx(normal);
    let normal_dy = dpdy(normal);
    let geometric_variance = min(0.25 * (dot(normal_dx, normal_dx) + dot(normal_dy, normal_dy)), 0.18);
    let alpha = sqrt(max(0.02 * 0.02 + slope_variance.x + slope_variance.y + geometric_variance, 1e-4));

    // Path through the medium straight down the view ray, before refraction.
    // Used to size the refraction offset; the offset then changes which pixel
    // we are looking through, so absorption is re-evaluated against that.
    let path_direct = water_path_length(screen_uv, in.world_pos, vol);
    let vi = in.vol_index;

    // ── Refraction ───────────────────────────────────────────────────────────
    // Work out the lateral displacement in WORLD units first, then project it
    // to screen. A ray entering a tilted surface deviates in proportion to the
    // tilt and to how strongly the medium bends it, and that deviation
    // accumulates over the distance it travels — so shallow water barely
    // distorts and deep water distorts a lot, which a constant screen-space
    // offset cannot express.
    let view_pos = cameras[0].view * vec4f(in.world_pos, 1.0);
    let view_z   = max(-view_pos.z, cameras[0].position_near.w);
    let n_view   = (cameras[0].view * vec4f(normal, 0.0)).xyz;

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
    let focal  = vec2f(cameras[0].proj[0][0], cameras[0].proj[1][1]);
    let offset = vec2f(world_offset.x, -world_offset.y) * focal * 0.5 / view_z;

    var refract_uv = clamp(screen_uv + offset, vec2f(0.001), vec2f(0.999));

    // Foreground rejection: if the offset landed on something nearer than the
    // water surface — a rock breaking the waterline, a hull, a character — that
    // object is in front of the water and must not be dragged into it.
    let refr_depth = textureSampleLevel(depth_texture, depth_sampler, refract_uv, 0);
    let refr_z     = helio_view_depth(cameras[0].proj, refr_depth);
    // Smoothly reject offset samples that landed on foreground geometry. This
    // avoids a hard refractive outline at rocks, hulls, and shoreline edges.
    let valid_refract = smoothstep(view_z - 0.3, view_z + 0.3, refr_z);
    refract_uv = mix(screen_uv, refract_uv, valid_refract);
    let scene_depth = textureSampleLevel(depth_texture, depth_sampler, refract_uv, 0);
    let path_offset = water_path_length(refract_uv, in.world_pos, vol);
    let path = mix(path_direct, path_offset, valid_refract);

    var refracted = textureSampleLevel(scene_color, shared_samp, refract_uv, 0.0).rgb;

    // Caustics land on the submerged surface being looked at through the water,
    // so they are added before absorption attenuates the whole lot.
    if !helio_is_sky(scene_depth) {
        let scene_pos = helio_world_from_depth(cameras[0].view_proj_inv, refract_uv, scene_depth);
        refracted += water_caustics(scene_pos, vol, refract_uv, vi);
    }

    // ── Medium ───────────────────────────────────────────────────────────────
    let transmittance = water_transmittance(path, vol);
    let inscatter     = water_inscatter(view_dir, light_dir, vol);
    // A restrained crest lift uses the existing medium color; no unrelated
    // caustics or legacy volume parameter is repurposed as an SSS control.
    let crest_scatter = vol.water_color.rgb * max(dot(normal, light_dir), 0.0) * 0.08;
    let through_water = refracted * transmittance
        + (inscatter + crest_scatter) * (1.0 - transmittance);

    // ── Reflection ───────────────────────────────────────────────────────────
    let reflected_ray = reflect(view_dir, normal);
    let reflected     = water_reflection(in.world_pos, reflected_ray, normal, light_dir, vol);

    // Fade the reflection into the shoreline, otherwise the contact with
    // geometry reads as a hard analytic line.
    let fresnel = water_fresnel(normal, view_dir, alpha) * water_shore_factor(path, vol);
    var color   = mix(through_water, reflected, fresnel)
                + water_sun_specular(
                    normal, view_dir, light_dir, alpha, in.world_pos.xz,
                    distance(in.world_pos, cameras[0].position_near.xyz));

    // ── Foam ─────────────────────────────────────────────────────────────────
    let shore = shore_wave(in.world_pos.xz);
    let foam_source = water_foam(path, in.world_pos.xz, vol, in.cell_size);
    let foam = water_foam_material(in.world_pos.xz, foam_source, shore);
    let foam_normal = normalize(vec3f(-foam.gradient.x, 1.0, -foam.gradient.y));
    let foam_light = 0.78 + 0.22 * max(dot(foam_normal, light_dir), 0.0);
    // Zero is the legacy value of this formerly-unused lane; interpret it as
    // Ultra-Water's default brightness so older volume rows remain visible.
    let foam_brightness = select(0.85, vol.reflection_refraction.w, vol.reflection_refraction.w > 0.0);
    let foam_color = vec3f(0.92, 0.96, 1.00) * foam_brightness * foam_light * (0.94 + foam.billow * 0.12);
    color = mix(color, foam_color, foam.coverage);

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

    let normal    = -water_total_normal(in.world_pos.xz, vol, in.vol_index, in.cell_size);
    let view_dir  = normalize(in.world_pos - cameras[0].position_near.xyz);
    let screen_uv = in.position.xy * viewport.zw;

    let cascade_damp = shore_ocean_damp(in.world_pos.xz);
    let slope_variance = ocean_slope_variance(in.world_pos.xz, in.cell_size, cascade_damp);
    let normal_dx = dpdx(normal);
    let normal_dy = dpdy(normal);
    let geometric_variance = min(0.25 * (dot(normal_dx, normal_dx) + dot(normal_dy, normal_dy)), 0.18);
    let alpha = sqrt(max(0.02 * 0.02 + slope_variance.x + slope_variance.y + geometric_variance, 1e-4));
    let fresnel = water_fresnel(normal, view_dir, alpha);

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
        let n_view = (cameras[0].view * vec4f(normal, 0.0)).xyz;
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
    // Continuous ocean: above or underwater based on camera height.
    let cam_above = cameras[0].position_near.y > in.world_pos.y;
    if cam_above {
        return fs_above(in);
    } else {
        return fs_under(in);
    }
}
