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

const ENABLE_LIGHTING: bool = true;
const ENABLE_SHADOWS: bool = true;
const ENABLE_BLOOM: bool = false;
const MAX_SHADOW_LIGHTS: u32 = 42u;
const BLOOM_INTENSITY: f32 = 0.3;
const BLOOM_THRESHOLD: f32 = 1.0;

struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    view_proj_inv:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
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
    debug_mode:        u32,
    _pad0:             u32,
    num_tiles_x:       u32,
    _pad2:             u32,
}

/// GpuLight (64 bytes, matches libhelio::GpuLight)
struct GpuLight {
    position_range:  vec4<f32>,  // xyz = position, w = range
    direction_outer: vec4<f32>,  // xyz = direction, w = spot outer cos angle
    color_intensity: vec4<f32>,  // xyz = color, w = intensity
    shadow_index:    u32,        // -1u32 if no shadow
    light_type:      u32,        // LightType enum (0=directional, 1=point, 2=spot)
    inner_angle:     f32,        // spot inner cos angle
    _pad:            u32,
}

struct LightMatrix { mat: mat4x4<f32> }

/// Per-cascade shadow configuration (16 bytes, matches libhelio::CascadeConfig)
struct CascadeConfig {
    split_distance:   f32,  // Far plane distance (meters)
    depth_bias:       f32,  // Base depth bias
    filter_radius:    f32,  // PCF filter radius (texels)
    pcss_light_size:  f32,  // PCSS light size (meters, 0.0 = disable)
}

/// Global shadow configuration (96 bytes, matches libhelio::ShadowConfig)
struct ShadowConfig {
    cascades:             array<CascadeConfig, 4>,  // 64 bytes
    enable_pcss:          u32,                      // Global PCSS toggle
    pcss_blocker_samples: u32,                      // Blocker search samples
    pcss_filter_samples:  u32,                      // PCSS filter samples
    pcf_sample_count:     u32,                      // Standard PCF sample count (4/8/12/16)
}

@group(0) @binding(0) var <uniform> camera:        Camera;
@group(0) @binding(1) var <uniform> globals:       Globals;
@group(0) @binding(7) var <uniform> shadow_config: ShadowConfig;

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
@group(2) @binding(7) var shadow_depth_sampler: sampler;  // Non-comparison sampler for PCSS blocker search

// Group 3 – tiled light culling results (written by LightCullPass each frame)
const TILE_SIZE:          u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 64u;
@group(3) @binding(0) var<storage, read> tile_light_lists:  array<u32>;
@group(3) @binding(1) var<storage, read> tile_light_counts: array<u32>;
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

const ATLAS_SIZE: f32 = 1024.0;

// Vogel disk sampling - blue-noise-like spiral pattern for high-quality PCF
fn vogel_disk_sample(sample_idx: u32, sample_count: u32, theta: f32) -> vec2<f32> {
    let GOLDEN_ANGLE = 2.39996323;  // 2π / φ² (golden angle in radians)
    let r = sqrt(f32(sample_idx) + 0.5) / sqrt(f32(sample_count));
    let angle = f32(sample_idx) * GOLDEN_ANGLE + theta;
    return vec2<f32>(cos(angle), sin(angle)) * r;
}

// Per-pixel hash for PCF rotation (reduces banding artifacts)
fn hash22(p: vec2<f32>) -> f32 {
    let p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    let d = dot(p3, vec3<f32>(p3.y + 33.33, p3.z + 33.33, p3.x + 33.33));
    return fract((p3.x + p3.y) * d);
}

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

// Normal-offset bias constant (world-space units).
// Shifts the shadow query point along the surface normal before projecting into
// light space, eliminating self-shadowing without any visible surface gap.
// This is the same technique used by UE4 ("Normal Shadow Bias") and Unity HDRP.
const NORMAL_OFFSET_SCALE: f32 = 0.05;

// High-quality PCF shadow sampling with Vogel disk pattern.
// world_pos must already have normal-offset applied (call shadow_factor, not this directly).
fn sample_cascade_shadow(
    layer: u32,
    cascade_scale: f32,
    world_pos: vec3<f32>,
    frag_coord: vec2<f32>,
    frame: u32
) -> f32 {
    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }

    let ndc       = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0))
       || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let biased_depth = ndc.z;
    let filter_radius = (2.0 / ATLAS_SIZE) * cascade_scale;

    // Per-pixel rotation to break up banding (stable with frame counter)
    let theta = hash22(frag_coord + vec2<f32>(f32(frame))) * 6.28318530718;

    let pcf_count = shadow_config.pcf_sample_count;
    var lit_sum = 0.0;
    for (var i = 0u; i < pcf_count; i++) {
        let offset = vogel_disk_sample(i, pcf_count, theta) * filter_radius;
        lit_sum += textureSampleCompareLevel(
            shadow_atlas, shadow_sampler,
            shadow_uv + offset,
            i32(layer),
            biased_depth,
        );
    }
    return lit_sum / f32(pcf_count);
}

// ── PCSS (Contact-Hardening Shadows) ─────────────────────────────────────────

// Step 1: Blocker search - find average occluder depth in light-space
fn pcss_blocker_search(
    layer: u32,
    shadow_uv: vec2<f32>,
    receiver_depth: f32,
    search_radius: f32,
    blocker_samples: u32,
    theta: f32
) -> vec2<f32> {  // Returns (avg_blocker_depth, num_blockers)
    var blocker_sum = 0.0;
    var blocker_count = 0.0;

    for (var i = 0u; i < blocker_samples; i++) {
        let offset = vogel_disk_sample(i, blocker_samples, theta) * search_radius;
        let sample_uv = shadow_uv + offset;

        // Convert UV to pixel coordinates for textureLoad (no filtering needed for blocker search)
        let pixel_coord = vec2<i32>(sample_uv * ATLAS_SIZE);

        // Bounds check to prevent out-of-range access
        if any(pixel_coord < vec2<i32>(0)) || any(pixel_coord >= vec2<i32>(i32(ATLAS_SIZE))) {
            continue;
        }

        // Sample actual depth value (not comparison) for blocker detection
        let occluder_depth = textureLoad(shadow_atlas, pixel_coord, i32(layer), 0);

        if occluder_depth < receiver_depth - 0.0001 {  // Is blocker
            blocker_sum += occluder_depth;
            blocker_count += 1.0;
        }
    }

    if blocker_count < 0.5 {
        return vec2<f32>(0.0, 0.0);  // Fully lit (no blockers found)
    }

    return vec2<f32>(blocker_sum / blocker_count, blocker_count);
}

// Step 2: Compute penumbra size based on blocker-receiver distance
fn pcss_penumbra_size(
    receiver_depth: f32,
    avg_blocker_depth: f32,
    light_size: f32
) -> f32 {
    // Classic PCSS formula: penumbra_width = (d_receiver - d_blocker) / d_blocker * light_width
    // Contact shadows (blocker_depth ≈ receiver_depth) → small penumbra (sharp)
    // Distant shadows (receiver_depth >> blocker_depth) → large penumbra (soft)
    return (receiver_depth - avg_blocker_depth) / max(avg_blocker_depth, 0.001) * light_size;
}

// Step 3: Full PCSS shadow sampling (blocker search + variable-kernel PCF).
// world_pos must already have normal-offset applied (call shadow_factor, not this directly).
fn sample_cascade_shadow_pcss(
    layer: u32,
    cascade_idx: u32,
    world_pos: vec3<f32>,
    frag_coord: vec2<f32>,
    frame: u32
) -> f32 {
    let config = shadow_config.cascades[cascade_idx];
    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }

    let ndc = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0))
       || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let receiver_depth = ndc.z;
    let theta = hash22(frag_coord + vec2<f32>(f32(frame))) * 6.28318530718;

    // Step 1: Blocker search (average occluder depth)
    let search_radius = config.pcss_light_size / ATLAS_SIZE;
    let blocker = pcss_blocker_search(layer, shadow_uv, receiver_depth, search_radius,
                                       shadow_config.pcss_blocker_samples, theta);

    if blocker.y < 0.5 {
        return 1.0;  // No blockers - fully lit (early exit optimization)
    }

    // Step 2: Compute penumbra size (distance-based filter width)
    let penumbra = pcss_penumbra_size(receiver_depth, blocker.x, config.pcss_light_size);
    let filter_radius = clamp(penumbra / ATLAS_SIZE,
                                config.filter_radius / ATLAS_SIZE,
                                config.filter_radius * 3.0 / ATLAS_SIZE);

    // Step 3: Variable-kernel PCF (filter size scales with penumbra)
    var lit_sum = 0.0;
    for (var i = 0u; i < shadow_config.pcss_filter_samples; i++) {
        let offset = vogel_disk_sample(i, shadow_config.pcss_filter_samples, theta) * filter_radius;
        lit_sum += textureSampleCompareLevel(
            shadow_atlas, shadow_sampler,
            shadow_uv + offset,
            i32(layer),
            receiver_depth
        );
    }

    return lit_sum / f32(shadow_config.pcss_filter_samples);
}

fn shadow_factor(light_idx: u32, world_pos: vec3<f32>, N: vec3<f32>, frag_coord: vec2<f32>, frame: u32) -> f32 {
    if !ENABLE_SHADOWS { return 1.0; }
    if light_idx >= MAX_SHADOW_LIGHTS { return 1.0; }

    let light = lights[light_idx];

    // Check if this light actually casts shadows (shadow_index != u32::MAX)
    if light.shadow_index == 4294967295u { return 1.0; }

    // Normal-offset: shift the world-space query point along the surface normal
    // toward the light before projecting.  This eliminates self-shadowing caused
    // by floating-point depth quantization, without the visible gap from a
    // constant depth-offset.  Scale by (1 - NdotL) so face-on surfaces (no
    // self-shadow risk) get near-zero offset while grazing surfaces get the full
    // amount — exactly matching the UE4 / Unity HDRP normal-bias approach.
    var light_dir: vec3<f32>;
    if light.light_type == 0u {
        light_dir = normalize(-light.direction_outer.xyz);
    } else {
        light_dir = normalize(light.position_range.xyz - world_pos);
    }
    let NdotL         = max(dot(N, light_dir), 0.0);
    let normal_offset = N * NORMAL_OFFSET_SCALE * (1.0 - NdotL);
    let biased_pos    = world_pos + normal_offset;

    var layer: u32;
    if light.light_type > 0u && light.light_type < 2u {  // Point light (type 1)
        let to_frag = biased_pos - light.position_range.xyz;
        layer = light.shadow_index + point_light_face(to_frag);
        return sample_cascade_shadow(layer, 1.0, biased_pos, frag_coord, frame);
    } else if light.light_type == 0u {  // Directional light (type 0)
        let dist = length(world_pos - camera.position_near.xyz);
        let splits = globals.csm_splits;
        
        // Determine cascades and blend factor
        var cascade_a = 3u;
        var cascade_b = 3u;
        var blend = 0.0;
        
        const BLEND_ZONE = 0.1;  // 10% blend zone around boundaries
        
        if dist < splits.x * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 0u;
        } else if dist < splits.x * (1.0 + BLEND_ZONE / 2.0) {
            // Blend zone between cascade 0 and 1
            cascade_a = 0u;
            cascade_b = 1u;
            blend = smoothstep(
                splits.x * (1.0 - BLEND_ZONE / 2.0),
                splits.x * (1.0 + BLEND_ZONE / 2.0),
                dist
            );
        } else if dist < splits.y * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 1u;
        } else if dist < splits.y * (1.0 + BLEND_ZONE / 2.0) {
            // Blend zone between cascade 1 and 2
            cascade_a = 1u;
            cascade_b = 2u;
            blend = smoothstep(
                splits.y * (1.0 - BLEND_ZONE / 2.0),
                splits.y * (1.0 + BLEND_ZONE / 2.0),
                dist
            );
        } else if dist < splits.z * (1.0 - BLEND_ZONE / 2.0) {
            cascade_a = 2u;
        } else if dist < splits.z * (1.0 + BLEND_ZONE / 2.0) {
            // Blend zone between cascade 2 and 3
            cascade_a = 2u;
            cascade_b = 3u;
            blend = smoothstep(
                splits.z * (1.0 - BLEND_ZONE / 2.0),
                splits.z * (1.0 + BLEND_ZONE / 2.0),
                dist
            );
        } else {
            cascade_a = 3u;
        }
        
        // Use PCSS if enabled and light size is non-zero for this cascade
        let use_pcss = shadow_config.enable_pcss != 0u && shadow_config.cascades[cascade_a].pcss_light_size > 0.0;

        let layer_a = light.shadow_index + cascade_a;
        var shadow_a: f32;
        if use_pcss {
            shadow_a = sample_cascade_shadow_pcss(layer_a, cascade_a, biased_pos, frag_coord, frame);
        } else {
            let cascade_scale_a = 1.0 + f32(cascade_a) * 1.5;
            shadow_a = sample_cascade_shadow(layer_a, cascade_scale_a, biased_pos, frag_coord, frame);
        }

        // If no blending needed, return immediately
        if blend <= 0.001 { return shadow_a; }

        // Blend between cascades if needed
        if cascade_b != cascade_a && blend > 0.001 {
            let use_pcss_b = shadow_config.enable_pcss != 0u && shadow_config.cascades[cascade_b].pcss_light_size > 0.0;
            let layer_b = light.shadow_index + cascade_b;
            var shadow_b: f32;
            if use_pcss_b {
                shadow_b = sample_cascade_shadow_pcss(layer_b, cascade_b, biased_pos, frag_coord, frame);
            } else {
                let cascade_scale_b = 1.0 + f32(cascade_b) * 1.5;
                shadow_b = sample_cascade_shadow(layer_b, cascade_scale_b, biased_pos, frag_coord, frame);
            }
            return mix(shadow_a, shadow_b, blend);
        }

        return shadow_a;
    } else {
        // Spot light (type 2)
        layer = light.shadow_index;
        return sample_cascade_shadow(layer, 1.0, biased_pos, frag_coord, frame);
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

    if light.light_type == 0u {  // Directional light
        L        = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {  // Point or spot light
        let to_light = light.position_range.xyz - world_pos;
        let dist     = length(to_light);
        if dist > light.position_range.w { return vec3<f32>(0.0); }
        L = to_light / dist;
        let ratio   = dist / light.position_range.w;
        let falloff = max(0.0, 1.0 - ratio * ratio);
        var atten   = falloff * falloff;
        if light.light_type == 2u {  // Spot light
            let cos_a = dot(-L, light.direction_outer.xyz);
            atten    *= smoothstep(light.direction_outer.w, light.inner_angle, cos_a);
        }
        radiance = light.color_intensity.xyz * light.color_intensity.w * atten;
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

// ── Tonemapping & bloom ───────────────────────────────────────────────────────

fn luminance(c: vec3<f32>) -> f32 { return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)); }

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

fn apply_bloom(color: vec3<f32>) -> vec3<f32> {
    if !ENABLE_BLOOM { return color; }
    let lum    = luminance(color);
    let excess = max(lum - BLOOM_THRESHOLD, 0.0);
    return color + color * (excess * BLOOM_INTENSITY);
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
    let emissive_r = textureLoad(gbuf_emissive, pix, 0);
    let emissive  = emissive_r.rgb;

    let albedo    = albedo_a.rgb;
    let alpha     = albedo_a.a;
    let N         = normalize(normal_r.xyz);
    let ao        = orm_r.r;
    let roughness = orm_r.g;
    let metallic  = orm_r.b;

    // ── Debug mode: bypass lighting ───────────────────────────────────────────
    // Mode 1 (UV Grid) and Mode 2 (Texture Direct) show raw colors without lighting
    // Mode 3 (Lit without normal mapping) goes through normal lighting
    // Mode 4 (G-buffer readback test) shows albedo read from G-buffer without lighting
    // Mode 5 (World normals) remaps N from [-1,1] → [0,1] as RGB (R=+X, G=+Y, B=+Z)
    // Mode 20 (VG triangle debug, F3): per-face colour written into albedo by vg_gbuffer.wgsl
    if globals.debug_mode == 1u || globals.debug_mode == 2u || globals.debug_mode == 4u
    || globals.debug_mode == 20u {
        return vec4<f32>(albedo, alpha);
    }
    if globals.debug_mode == 5u {
        return vec4<f32>(N * 0.5 + 0.5, 1.0);
    }

    // ── Reconstruct world position from depth + inv_view_proj ────────────────
    // clip_pos.xy is in viewport space (0→width, 0→height, y↓).
    // Convert to NDC: x ∈ [-1,1], y ∈ [1,-1] (wgpu NDC y+ = up, viewport y+ = down).
    let screen_size = vec2<f32>(textureDimensions(gbuf_albedo));
    let uv_01       = in.clip_pos.xy / screen_size;
    let ndc_xy      = vec2<f32>(uv_01.x * 2.0 - 1.0, 1.0 - uv_01.y * 2.0);
    let world_h     = camera.view_proj_inv * vec4<f32>(ndc_xy, depth, 1.0);
    let world_pos   = world_h.xyz / world_h.w;

    // ── Debug mode 10: shadow factor heatmap ──────────────────────────────────
    // Shows shadow_factor() per light averaged across all lights.
    // White = fully lit, black = fully occluded.
    // Useful for verifying shadow atlas is filled and matrices are correct.
    if globals.debug_mode == 10u {
        var shadow_sum = 0.0;
        for (var i = 0u; i < globals.light_count; i++) {
            shadow_sum += shadow_factor(i, world_pos, N, in.clip_pos.xy, globals.frame);
        }
        let sf = shadow_sum / max(f32(globals.light_count), 1.0);
        return vec4<f32>(sf, sf, sf, 1.0);
    }

    // ── Debug mode 11: light-space projection for first light face 0 ─────────
    // Orange gradient = pixel is inside the light frustum, depth = ndc.z.
    // Dark blue = pixel is outside the frustum (w<=0 or uv out of [0,1]).
    // Use this to verify shadow matrices are computed by ShadowMatrixPass.
    if globals.debug_mode == 11u && globals.light_count > 0u {
        let lc  = shadow_matrices[0u].mat * vec4<f32>(world_pos, 1.0);
        if lc.w > 0.001 {
            let ndc3 = lc.xyz / lc.w;
            let uv   = vec2<f32>(ndc3.x * 0.5 + 0.5, -ndc3.y * 0.5 + 0.5);
            if all(uv >= vec2<f32>(0.0)) && all(uv <= vec2<f32>(1.0))
                    && ndc3.z >= 0.0 && ndc3.z <= 1.0 {
                return vec4<f32>(ndc3.z, ndc3.z * 0.3, 0.0, 1.0);
            }
        }
        return vec4<f32>(0.0, 0.0, 0.2, 1.0);
    }

    // ── PBR setup ─────────────────────────────────────────────────────────────
    let F0  = clamp(vec3<f32>(normal_r.w, orm_r.a, emissive_r.a), vec3<f32>(0.0), vec3<f32>(0.999));
    let V   = normalize(camera.position_near.xyz - world_pos);
    let NdV = max(dot(N, V), 0.0);

    // ── Direct lighting ────────────────────────────────────────────────────────
    // GPU-driven: iterate all visible lights (already culled on CPU by distance).
    // Shadow factor affects ONLY direct lighting (Lo).  Ambient / indirect light
    // is handled separately — shadow maps do not occlude it (that is AO's job).
    var Lo = vec3<f32>(0.0);
    if ENABLE_LIGHTING {
        let tile_x = u32(in.clip_pos.x) / TILE_SIZE;
        let tile_y = u32(in.clip_pos.y) / TILE_SIZE;
        let tile_idx = tile_y * globals.num_tiles_x + tile_x;
        let tile_light_count = tile_light_counts[tile_idx];
        for (var i = 0u; i < tile_light_count; i++) {
            let light_idx = tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i];
            let light = lights[light_idx];
            if light.light_type != 0u {
                let dist = length(light.position_range.xyz - world_pos);
                if dist > light.position_range.w { continue; }
            }
            let sf = shadow_factor(light_idx, world_pos, N, in.clip_pos.xy, globals.frame);
            Lo += pbr_direct_light(light, world_pos, N, V, F0, albedo, roughness, metallic, sf);
        }
    }

    // ── RC indirect diffuse ───────────────────────────────────────────────────
    let rc_irr   = sample_rc_irradiance(world_pos, N);
    let F_ibl    = fresnel_schlick_roughness(NdV, F0, roughness);
    let kD_ibl   = (1.0 - F_ibl) * (1.0 - metallic);
    let diff_ind = kD_ibl * rc_irr * albedo;

    // ── Indirect specular: environment cubemap ────────────────────────────────
    let R            = reflect(-V, N);
    let env_lod      = roughness * 8.0;  // approx mip from roughness (WebGPU: textureSample not allowed in non-uniform flow)
    let env_sample   = textureSampleLevel(env_cube, env_sampler, R, env_lod).rgb;
    let spec_scale   = 1.0 - roughness * roughness;
    let spec_ind     = F_ibl * env_sample * spec_scale;

    // ── INDIRECT LIGHTING ────────────────────────────────────────────────────
    // Hemisphere ambient is shadow-INDEPENDENT.  Shadow maps only affect direct
    // lighting (Lo above); ambient occlusion (ao from G-buffer ORM.r) handles
    // indirect-light occlusion instead.  This ensures shadowed areas still
    // receive fill light and are never pitch black.
    //
    // When RC GI is active it replaces the hemisphere fallback with physically-
    // based global illumination.  When inactive the hemisphere ambient is used.

    let sky_color      = globals.ambient_color.rgb * globals.ambient_intensity;
    let ground_color   = sky_color * 0.15;
    let hemi_t         = N.y * 0.5 + 0.5;
    let hemi           = mix(ground_color, sky_color, hemi_t) * albedo;

    // RC weight: 0 = no RC data, 1 = full RC coverage
    let rc_weight      = clamp(length(rc_irr) * 4.0, 0.0, 1.0);

    // Blend between hemisphere fallback and RC-based GI
    let ambient_fallback = mix(hemi, diff_ind, rc_weight);

    // ── Combine ───────────────────────────────────────────────────────────────
    let indirect  = (ambient_fallback + spec_ind) * ao;
    var color     = Lo + indirect;
    color        += emissive;               // emissive from G-buffer
    color         = apply_bloom(color);
    color         = aces_tonemap(color);

    return vec4<f32>(color, alpha);
}
