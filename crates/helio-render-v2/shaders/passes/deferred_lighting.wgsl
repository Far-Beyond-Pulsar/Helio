//! Deferred lighting pass.
//!
//! Runs as a fullscreen triangle (no vertex buffer) over the G-buffer written
//! by the gbuffer pass.  Performs the full Cook-Torrance PBR evaluation,
//! PCF shadow sampling, Radiance-Cascades GI, environment IBL and tonemapping
//! in a single screen-space draw — O(pixels) instead of O(pixels × lights).
//!
//! Feature override constants injected by PipelineCache:
//!   override ENABLE_LIGHTING:   bool = false;
//!   override LIGHT_COUNT:       u32  = 0u;
//!   override ENABLE_SHADOWS:    bool = false;
//!   override MAX_SHADOW_LIGHTS: u32  = 0u;
//!   override ENABLE_BLOOM:      bool = false;
//!   override BLOOM_INTENSITY:   f32  = 0.3;
//!   override BLOOM_THRESHOLD:   f32  = 1.0;

// ── Uniforms ──────────────────────────────────────────────────────────────────

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,   // offset 80 – used to reconstruct world pos from depth
}

struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec4<f32>,
    rc_world_min:      vec4<f32>,
    rc_world_max:      vec4<f32>,
    csm_splits:        vec4<f32>,
}

struct GpuLight {
    position:   vec3<f32>,
    light_type: f32,
    direction:  vec3<f32>,   // prenormalized
    range:      f32,
    color:      vec3<f32>,
    intensity:  f32,
    cos_inner:  f32,
    cos_outer:  f32,
    _pad:       vec2<f32>,
}

struct LightMatrix { mat: mat4x4<f32> }

@group(0) @binding(0) var <uniform> camera:  Camera;
@group(0) @binding(1) var <uniform> globals: Globals;

// Group 1 – G-buffer inputs (read-only, textureLoad)
@group(1) @binding(0) var gbuf_albedo:   texture_2d<f32>;       // Rgba8Unorm   albedo.rgb + alpha
@group(1) @binding(1) var gbuf_normal:   texture_2d<f32>;       // Rgba16Float  world-space normal
@group(1) @binding(2) var gbuf_orm:      texture_2d<f32>;       // Rgba8Unorm   AO, roughness, metallic
@group(1) @binding(3) var gbuf_emissive: texture_2d<f32>;       // Rgba16Float  pre-multiplied emissive
@group(1) @binding(4) var gbuf_depth:    texture_depth_2d;      // Depth32Float

// Group 2 – lights, shadows, environment (same as forward geometry pass)
@group(2) @binding(0) var <storage, read> lights:          array<GpuLight>;
@group(2) @binding(1) var shadow_atlas:   texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;
@group(2) @binding(3) var env_cube:       texture_cube<f32>;
@group(2) @binding(4) var <storage, read> shadow_matrices: array<LightMatrix>;
@group(2) @binding(5) var rc_cascade0:    texture_2d<f32>;
@group(2) @binding(6) var env_sampler:    sampler;
// cluster bindings removed - GPU-driven architecture

// Cluster constants removed - GPU-driven architecture

// ── Fullscreen-triangle vertex shader ────────────────────────────────────────

struct VSOut {
    @builtin(position) clip_pos: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VSOut {
    // Three vertices covering the entire NDC square.
    // No vertex buffer required — just draw(3, 1, 0, 0).
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 3.0, -1.0),
        vec2<f32>(-1.0,  3.0),
    );
    var out: VSOut;
    out.clip_pos = vec4<f32>(pos[vi], 0.0, 1.0);
    return out;
}

// ── Shadow helpers ────────────────────────────────────────────────────────────

const ATLAS_SIZE: f32 = 2048.0;

// 4-tap rotated Poisson disk
var<private> POISSON_DISK: array<vec2<f32>, 4> = array<vec2<f32>, 4>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
);

fn point_light_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(0u, 1u, dir.x < 0.0);
    } else if a.y >= a.x && a.y >= a.z {
        return select(2u, 3u, dir.y < 0.0);
    } else {
        return select(4u, 5u, dir.z < 0.0);
    }
}

// Helper: Sample shadow from a specific cascade layer with PCF
fn sample_cascade_shadow(layer: u32, depth_bias: f32, cascade_scale: f32, world_pos: vec3<f32>) -> f32 {
    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }

    let ndc       = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0))
       || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let biased_depth = ndc.z - depth_bias;
    let filter_radius = (2.0 / ATLAS_SIZE) * cascade_scale;
    var lit_sum = 0.0;
    for (var i = 0; i < 4; i++) {
        let offset = POISSON_DISK[i] * filter_radius;
        lit_sum += textureSampleCompareLevel(
            shadow_atlas, shadow_sampler,
            shadow_uv + offset,
            i32(layer),
            biased_depth,
        );
    }
    return lit_sum * 0.25;
}

// World-space normal offset scale for directional shadow casters.
// Pulls the shadow-lookup point away from the surface along the normal,
// eliminating self-shadowing (acne) without the peter-panning that a
// constant depth bias introduces (UE5 / TheRealMJP technique).
const SHADOW_NORMAL_OFFSET: f32 = 0.03;  // metres; tune with scene scale

fn shadow_factor(light_idx: u32, world_pos: vec3<f32>, N: vec3<f32>) -> f32 {
    if !ENABLE_SHADOWS { return 1.0; }
    if light_idx >= MAX_SHADOW_LIGHTS { return 1.0; }

    let light = lights[light_idx];
    var layer: u32;
    if light.light_type > 0.5 && light.light_type < 1.5 {
        // Point light — no normal offset; use a small residual depth bias.
        let to_frag = world_pos - light.position;
        layer = light_idx * 6u + point_light_face(to_frag);
        return sample_cascade_shadow(layer, 0.0002, 1.0, world_pos);
    } else if light.light_type < 0.5 {
        // Directional light CSM with normal-offset bias.
        // The offset is largest at grazing angles (NdotL ≈ 0), where acne is worst.
        let L     = normalize(-light.direction);
        let NdotL = clamp(dot(N, L), 0.0, 1.0);
        // Normal-offset: proportional to sin(theta) = sqrt(1 - NdotL²).
        // Larger offset for surfaces nearly perpendicular to the light.
        let sin_theta    = sqrt(max(1.0 - NdotL * NdotL, 0.0));
        let biased_pos   = world_pos + N * (SHADOW_NORMAL_OFFSET * sin_theta);

        let dist   = length(world_pos - camera.position);
        let splits = globals.csm_splits;

        var cascade_a = 3u;
        var cascade_b = 3u;
        var blend     = 0.0;
        const BLEND_ZONE = 0.1;

        if dist < splits.x * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 0u;
        } else if dist < splits.x * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 0u; cascade_b = 1u;
            blend = smoothstep(splits.x*(1.0-BLEND_ZONE/2.0), splits.x*(1.0+BLEND_ZONE/2.0), dist);
        } else if dist < splits.y * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 1u;
        } else if dist < splits.y * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 1u; cascade_b = 2u;
            blend = smoothstep(splits.y*(1.0-BLEND_ZONE/2.0), splits.y*(1.0+BLEND_ZONE/2.0), dist);
        } else if dist < splits.z * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 2u;
        } else if dist < splits.z * (1.0 + BLEND_ZONE / 2.0) {
            cascade_a = 2u; cascade_b = 3u;
            blend = smoothstep(splits.z*(1.0-BLEND_ZONE/2.0), splits.z*(1.0+BLEND_ZONE/2.0), dist);
        } else {
            cascade_a = 3u;
        }

        // Tiny residual depth bias only — the normal offset handles most of the work.
        let depth_bias = 0.00002 * (1.0 + f32(cascade_a) * 1.5);
        let layer_a    = light_idx * 6u + cascade_a;

        // PCSS for directional: soft penumbra varies with blocker distance.
        let shadow_a = sample_cascade_shadow_pcss(layer_a, depth_bias, biased_pos);
        if blend <= 0.001 { return shadow_a; }
        let layer_b  = light_idx * 6u + cascade_b;
        let shadow_b = sample_cascade_shadow_pcss(layer_b, depth_bias, biased_pos);
        return mix(shadow_a, shadow_b, blend);
    } else {
        layer = light_idx * 6u;
        return sample_cascade_shadow(layer, 0.00015, 1.0, world_pos);
    }
}

// ── BRDF helpers ─────────────────────────────────────────────────────────────

const PI: f32 = 3.14159265359;

fn pow5(x: f32) -> f32 { let x2 = x * x; return x2 * x2 * x; }

fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a    = roughness * roughness;
    let a2   = a * a;
    let NdH  = max(dot(N, H), 0.0);
    let denom = NdH * NdH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k + 0.0001);
}

fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdV = max(dot(N, V), 0.0);
    let NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, roughness) * geometry_schlick_ggx(NdL, roughness);
}

fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

fn fresnel_schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let one_minus_r = vec3<f32>(1.0 - roughness);
    return F0 + (max(one_minus_r, F0) - F0) * pow5(clamp(1.0 - cos_theta, 0.0, 1.0));
}

// Evaluate one direct light with the full Cook-Torrance BRDF.
// `sf` is the shadow factor (0=shadowed, 1=lit), computed at the call site.
fn pbr_direct_light(
    light:     GpuLight,
    world_pos: vec3<f32>,
    N:         vec3<f32>,
    V:         vec3<f32>,
    F0:        vec3<f32>,
    albedo:    vec3<f32>,
    roughness: f32,
    metallic:  f32,
    sf:        f32,
) -> vec3<f32> {
    var L:        vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type < 0.5 {
        L        = normalize(-light.direction);
        radiance = light.color * light.intensity;
    } else {
        let to_light = light.position - world_pos;
        let dist     = length(to_light);
        if dist > light.range { return vec3<f32>(0.0); }
        L = to_light / dist;
        let ratio   = dist / light.range;
        let falloff = max(0.0, 1.0 - ratio * ratio);
        var atten   = falloff * falloff;
        if light.light_type > 1.5 {
            let cos_a = dot(-L, light.direction);
            atten    *= smoothstep(light.cos_outer, light.cos_inner, cos_a);
        }
        radiance = light.color * light.intensity * atten;
    }

    let NdL = max(dot(N, L), 0.0);
    if NdL == 0.0 { return vec3<f32>(0.0); }

    if all(radiance < vec3<f32>(0.002)) { return vec3<f32>(0.0); }

    let H        = normalize(V + L);
    let D        = distribution_ggx(N, H, roughness);
    let G        = geometry_smith(N, V, L, roughness);
    let F        = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let kD       = (1.0 - F) * (1.0 - metallic);
    let specular = D * G * F / (4.0 * max(dot(N, V), 0.0) * NdL + 0.0001);

    return (kD * albedo / PI + specular) * radiance * NdL * sf;
}

// ── Radiance Cascades GI ──────────────────────────────────────────────────────

const RC_PROBE_DIM: u32 = 16u;
const RC_DIR_DIM:   u32 = 4u;

fn rc_oct_decode(uv: vec2<f32>) -> vec3<f32> {
    let f  = uv * 2.0 - 1.0;
    let af = abs(f);
    let l  = af.x + af.y;
    var n: vec3<f32>;
    if l > 1.0 {
        let sx = select(-1.0, 1.0, f.x >= 0.0);
        let sz = select(-1.0, 1.0, f.y >= 0.0);
        n = vec3<f32>((1.0 - af.y) * sx, 1.0 - l, (1.0 - af.x) * sz);
    } else {
        n = vec3<f32>(f.x, 1.0 - l, f.y);
    }
    return normalize(n);
}

fn rc_corner_irradiance_precomp(
    px: u32, py: u32, pz: u32,
    cos_weights: array<f32, 16>,
) -> vec3<f32> {
    let dim = RC_PROBE_DIM - 1u;
    let cpx = min(px, dim); let cpy = min(py, dim); let cpz = min(pz, dim);
    var irr  = vec3<f32>(0.0);
    var wsum = 0.0;
    var idx  = 0u;
    for (var ddx: u32 = 0u; ddx < RC_DIR_DIM; ddx++) {
        for (var ddy: u32 = 0u; ddy < RC_DIR_DIM; ddy++) {
            let cos_w = cos_weights[idx];
            if cos_w > 0.001 {
                let atlas_x = i32(cpx * RC_DIR_DIM + ddx);
                let atlas_y = i32((cpy * RC_PROBE_DIM + cpz) * RC_DIR_DIM + ddy);
                irr  += textureLoad(rc_cascade0, vec2<i32>(atlas_x, atlas_y), 0).rgb * cos_w;
                wsum += cos_w;
            }
            idx++;
        }
    }
    return irr / max(wsum, 0.001);
}

fn sample_rc_irradiance(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let world_min  = globals.rc_world_min.xyz;
    let world_max  = globals.rc_world_max.xyz;
    let world_size = world_max - world_min;
    if world_size.x <= 0.0 || world_size.y <= 0.0 || world_size.z <= 0.0 {
        return vec3<f32>(0.0);
    }

    let t            = (world_pos - world_min) / world_size;
    let fade_margin  = 0.05;
    let fade         = smoothstep(vec3<f32>(0.0), vec3<f32>(fade_margin), t)
                     * smoothstep(vec3<f32>(1.0), vec3<f32>(1.0 - fade_margin), t);
    let volume_weight = fade.x * fade.y * fade.z;
    if volume_weight <= 0.0 { return vec3<f32>(0.0); }

    // Precompute per-direction cosine weights ONCE, shared across all 8 trilinear corners.
    var cos_weights: array<f32, 16>;
    var idx = 0u;
    for (var ddx: u32 = 0u; ddx < RC_DIR_DIM; ddx++) {
        for (var ddy: u32 = 0u; ddy < RC_DIR_DIM; ddy++) {
            let dir_uv = (vec2<f32>(f32(ddx), f32(ddy)) + 0.5) / f32(RC_DIR_DIM);
            cos_weights[idx] = max(0.0, dot(normal, rc_oct_decode(dir_uv)));
            idx++;
        }
    }

    let cell_size = world_size / f32(RC_PROBE_DIM);
    let probe_f   = (world_pos - world_min) / cell_size - 0.5;
    let pf        = clamp(probe_f, vec3<f32>(0.0), vec3<f32>(f32(RC_PROBE_DIM) - 1.0));
    let pi        = vec3<u32>(u32(pf.x), u32(pf.y), u32(pf.z));
    let frc       = fract(pf);

    let c000 = rc_corner_irradiance_precomp(pi.x,      pi.y,      pi.z,      cos_weights);
    let c001 = rc_corner_irradiance_precomp(pi.x,      pi.y,      pi.z + 1u, cos_weights);
    let c010 = rc_corner_irradiance_precomp(pi.x,      pi.y + 1u, pi.z,      cos_weights);
    let c011 = rc_corner_irradiance_precomp(pi.x,      pi.y + 1u, pi.z + 1u, cos_weights);
    let c100 = rc_corner_irradiance_precomp(pi.x + 1u, pi.y,      pi.z,      cos_weights);
    let c101 = rc_corner_irradiance_precomp(pi.x + 1u, pi.y,      pi.z + 1u, cos_weights);
    let c110 = rc_corner_irradiance_precomp(pi.x + 1u, pi.y + 1u, pi.z,      cos_weights);
    let c111 = rc_corner_irradiance_precomp(pi.x + 1u, pi.y + 1u, pi.z + 1u, cos_weights);

    let c0 = mix(mix(c000, c001, frc.z), mix(c010, c011, frc.z), frc.y);
    let c1 = mix(mix(c100, c101, frc.z), mix(c110, c111, frc.z), frc.y);
    return mix(c0, c1, frc.x) * volume_weight;
}

// ── Contact shadows ───────────────────────────────────────────────────────────

fn contact_shadow(world_pos: vec3<f32>, light_dir: vec3<f32>, screen_size: vec2<f32>) -> f32 {
    // Short screen-space ray march from the surface toward the light.
    // 12 steps, maximum 0.5m range.  Occludes ONLY when another surface is hit
    // within that range — fills the small shadow gap at the base of casters that
    // CSM cascade 0 misses due to limited resolution.
    let step_count   = 12;
    let max_distance = 0.5;
    let step_size    = max_distance / f32(step_count);
    var ray_pos      = world_pos + light_dir * 0.02;   // small bias away from surface

    for (var i = 0; i < step_count; i++) {
        ray_pos += light_dir * step_size;

        // Project ray sample into clip space.
        let clip = camera.view_proj * vec4<f32>(ray_pos, 1.0);
        if clip.w <= 0.0 { break; }
        let ndc = clip.xyz / clip.w;
        if abs(ndc.x) > 1.0 || abs(ndc.y) > 1.0 { break; }

        // Convert to texture coords.
        let uv  = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
        let pix = vec2<i32>(i32(uv.x * screen_size.x), i32(uv.y * screen_size.y));
        let scene_depth = textureLoad(gbuf_depth, pix, 0);

        // Reconstruct scene world-pos at this 2-D sample.
        let ndc2 = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
        let h    = camera.view_proj_inv * vec4<f32>(ndc2, scene_depth, 1.0);
        let p    = h.xyz / h.w;

        // If the scene surface is between the original point and the ray sample
        // (and behind it in the light direction), we're in shadow.
        let dist_scene = dot(p - world_pos, light_dir);
        let dist_ray   = dot(ray_pos - world_pos, light_dir);
        if dist_scene > 0.01 && dist_scene < dist_ray + 0.1 {
            return 0.0;   // occluded
        }
    }
    return 1.0;
}

// ── PCSS helpers ──────────────────────────────────────────────────────────────

const PCSS_LIGHT_SIZE:   f32 = 0.04;   // world-space half-width of the sun disc
const PCSS_BLOCKER_SAMPLES: i32 = 16;
const PCSS_PCF_SAMPLES:     i32 = 16;

// Poisson samples (16-tap, used for both blocker search and PCF)
var<private> POISSON16: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2(-0.94201624, -0.39906216),
    vec2( 0.94558609, -0.76890725),
    vec2(-0.09418410, -0.92938870),
    vec2( 0.34495938,  0.29387760),
    vec2(-0.91588581,  0.45771432),
    vec2(-0.81544232, -0.87912464),
    vec2(-0.38277543,  0.27676845),
    vec2( 0.97484398,  0.75648379),
    vec2( 0.44323325, -0.97511554),
    vec2( 0.53742981, -0.47373420),
    vec2(-0.26496911, -0.41893023),
    vec2( 0.79197514,  0.19090188),
    vec2(-0.24188840,  0.99706507),
    vec2(-0.81409955,  0.91437590),
    vec2( 0.19984126,  0.78641367),
    vec2( 0.14383161, -0.14100790),
);

// Average blocker depth search within `search_radius` (UV space)
// Uses textureLoad for raw depth values (needed for PCSS penumbra estimation)
fn pcss_blocker_search(layer: u32, shadow_uv: vec2<f32>, receiver_depth: f32, search_radius: f32) -> vec2<f32> {
    var blocker_sum   = 0.0;
    var blocker_count = 0.0;
    let atlas_sz = i32(ATLAS_SIZE);
    for (var i = 0; i < PCSS_BLOCKER_SAMPLES; i++) {
        let samp_uv = shadow_uv + POISSON16[i] * search_radius;
        let pix     = vec2<i32>(
            clamp(i32(samp_uv.x * ATLAS_SIZE), 0, atlas_sz - 1),
            clamp(i32(samp_uv.y * ATLAS_SIZE), 0, atlas_sz - 1),
        );
        let sample_depth = textureLoad(shadow_atlas, pix, i32(layer), 0);
        if sample_depth < receiver_depth - 0.0001 {
            blocker_sum   += sample_depth;
            blocker_count += 1.0;
        }
    }
    return vec2<f32>(blocker_sum, blocker_count);
}

// PCSS soft-shadow with variable penumbra based on blocker distance
fn sample_cascade_shadow_pcss(layer: u32, depth_bias: f32, world_pos: vec3<f32>) -> f32 {
    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }
    let ndc       = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);
    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0))
       || ndc.z < 0.0 || ndc.z > 1.0 { return 1.0; }

    let biased_depth  = ndc.z - depth_bias;
    let search_radius = PCSS_LIGHT_SIZE / ATLAS_SIZE;

    let blocker = pcss_blocker_search(layer, shadow_uv, biased_depth, search_radius);
    if blocker.y == 0.0 { return 1.0; }   // no blockers → fully lit

    let avg_blocker = blocker.x / blocker.y;
    // Penumbra radius in UV space: larger when blocker is farther from receiver
    let penumbra     = max((biased_depth - avg_blocker) / avg_blocker, 0.0) * PCSS_LIGHT_SIZE;
    let filter_radius = clamp(penumbra / ATLAS_SIZE, 0.5 / ATLAS_SIZE, 8.0 / ATLAS_SIZE);

    var lit_sum = 0.0;
    for (var i = 0; i < PCSS_PCF_SAMPLES; i++) {
        let offset = POISSON16[i] * filter_radius;
        lit_sum += textureSampleCompareLevel(
            shadow_atlas, shadow_sampler,
            shadow_uv + offset,
            i32(layer),
            biased_depth,
        );
    }
    return lit_sum / f32(PCSS_PCF_SAMPLES);
}

// ── UE5-quality IBL helpers ───────────────────────────────────────────────────

// Lazarov 2013 analytic DFG split-sum approximation.
// Returns vec2(AB.x, AB.y) where:  specular_ibl = F0 * AB.x + AB.y
// Matches the precomputed GF LUT from UE4 paper (Karis 2013) to ≈1%.
// Source: "Getting More Physical in Call of Duty: Black Ops II", Lazarov 2013.
fn dfg_approx(roughness: f32, NdV: f32) -> vec2<f32> {
    let c0   = vec4<f32>(-1.0, -0.0275, -0.572,  0.022);
    let c1   = vec4<f32>( 1.0,  0.0425,  1.04,  -0.04);
    let r    = roughness * c0 + c1;
    let a004 = min(r.x * r.x, exp2(-9.28 * NdV)) * r.x + r.y;
    return max(vec2<f32>(-1.04, 1.04) * a004 + r.zw, vec2<f32>(0.0));
}

// Specular occlusion from AO (Lagarde & de Rousiers 2014, "Moving Frostbite to PBR").
// Reduces indirect specular in occluded regions; rougher surfaces attenuate more.
fn specular_occlusion(NdV: f32, ao: f32, roughness: f32) -> f32 {
    return clamp(pow(NdV + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao, 0.0, 1.0);
}

// ── Fragment entry ────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let pix = vec2<i32>(i32(in.clip_pos.x), i32(in.clip_pos.y));

    // ── Depth guard: sky areas (depth=1) are already in the target → discard ──
    let depth = textureLoad(gbuf_depth, pix, 0);
    if depth >= 1.0 { discard; }

    // ── Read G-buffer ─────────────────────────────────────────────────────────
    let albedo_a  = textureLoad(gbuf_albedo,   pix, 0);
    let normal_r  = textureLoad(gbuf_normal,   pix, 0);
    let orm_r     = textureLoad(gbuf_orm,      pix, 0);
    let emissive  = textureLoad(gbuf_emissive, pix, 0).rgb;

    let albedo   = albedo_a.rgb;
    let alpha    = albedo_a.a;
    let N        = normalize(normal_r.xyz);
    let ao       = orm_r.r;
    // Clamp to [0.045, 1.0]: prevents NaN in GGX at roughness=0 (UE5 minimum).
    let roughness = clamp(orm_r.g, 0.045, 1.0);
    let metallic  = orm_r.b;

    // ── Specular anti-aliasing: widen roughness from normal-map high-freq ──────
    // Screen-space normal gradient → roughness variance ("LEAN mapping" approx).
    // Prevents specular fireflies on curved/normal-mapped surfaces.
    let nx = dpdx(N);
    let ny = dpdy(N);
    let roughness_aa = min(1.0, roughness + sqrt(dot(nx, nx) + dot(ny, ny)) * 0.45);

    // ── Reconstruct world position from depth ─────────────────────────────────
    let screen_size = vec2<f32>(textureDimensions(gbuf_albedo));
    let uv_01       = in.clip_pos.xy / screen_size;
    let ndc_xy      = vec2<f32>(uv_01.x * 2.0 - 1.0, 1.0 - uv_01.y * 2.0);
    let world_h     = camera.view_proj_inv * vec4<f32>(ndc_xy, depth, 1.0);
    let world_pos   = world_h.xyz / world_h.w;

    // ── PBR setup ─────────────────────────────────────────────────────────────
    // F0: 4% reflectance for dielectrics, albedo for metals (UE4 convention).
    let F0  = mix(vec3<f32>(0.04), albedo, metallic);
    let V   = normalize(camera.position - world_pos);
    let NdV = max(dot(N, V), 0.0001);

    // Lazarov DFG split-sum (same table used for both IBL and multiscattering).
    let dfg = dfg_approx(roughness_aa, NdV);

    // ── Direct lighting ───────────────────────────────────────────────────────
    var Lo = vec3<f32>(0.0);
    if ENABLE_LIGHTING {
        for (var i = 0u; i < globals.light_count; i++) {
            var sf = shadow_factor(i, world_pos, N);
            // Fill CSM cascade 0 gap with screen-space contact shadows.
            if lights[i].light_type < 0.5 && sf > 0.01 {
                let sun_dir = normalize(-lights[i].direction);
                sf *= contact_shadow(world_pos, sun_dir, screen_size);
            }
            Lo += pbr_direct_light(lights[i], world_pos, N, V, F0, albedo, roughness_aa, metallic, sf);
        }
    }

    // ── Indirect diffuse (RC radiance cascades) ───────────────────────────────
    let rc_irr = sample_rc_irradiance(world_pos, N);
    // Fresnel at grazing angle for IBL (roughness-aware Schlick).
    let F_ibl  = fresnel_schlick_roughness(NdV, F0, roughness_aa);
    // Energy-conserving kD: specular + diffuse can't exceed 1, metals have no diffuse.
    let kD_ibl   = (1.0 - F_ibl) * (1.0 - metallic);
    let diff_ind = kD_ibl * rc_irr * albedo;

    // ── Indirect specular: env cubemap + UE4 split-sum + multi-scatter ────────
    let R       = reflect(-V, N);
    // Map roughness to mip level: 0 = mirror, 8 = very rough.
    let env_mip    = roughness_aa * 8.0;
    let env_sample = textureSampleLevel(env_cube, env_sampler, R, env_mip).rgb;

    // Single-scatter specular from split-sum: F0 * AB.x + AB.y
    let Fss = F0 * dfg.x + dfg.y;

    // Multi-scatter energy compensation (Turquin 2019 / Karis 2013):
    // Standard single-scatter GGX loses energy at high roughness.
    // Add back the missing energy as a tinted diffuse-like bounce.
    let Ess  = dfg.x + dfg.y;                           // single-scatter albedo (F0=white approx)
    let Ems  = max(1.0 - Ess, 0.0);                     // missing fraction
    let Favg = F0 + (1.0 - F0) / 21.0;                  // hemisphere-average Fresnel (Turquin)
    let Fms  = Favg * Ems / max(1.0 - Favg * Ems, vec3<f32>(0.001));  // multi-bounce tint

    let spec_occ = specular_occlusion(NdV, ao, roughness_aa);
    let spec_ind = (Fss + Fms) * env_sample * spec_occ;

    // ── Hemisphere ambient fallback blended with RC irradiance ────────────────
    let sky_color    = globals.ambient_color.rgb * globals.ambient_intensity;
    let ground_color = sky_color * 0.15;
    let hemi         = mix(ground_color, sky_color, N.y * 0.5 + 0.5) * albedo;
    let rc_weight    = clamp(length(rc_irr) * 4.0, 0.0, 1.0);
    let amb_diffuse  = mix(hemi, diff_ind, rc_weight);

    // ── Final combine (pure HDR linear; tonemapping in post-process) ──────────
    // AO occludes indirect diffuse; specular already occluded via spec_occ.
    let indirect = amb_diffuse * ao + spec_ind;
    let color    = Lo + indirect + emissive;

    return vec4<f32>(color, alpha);
}
