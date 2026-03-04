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

@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

// Group 1 – G-buffer inputs (read-only, textureLoad)
@group(1) @binding(0) var gbuf_albedo:   texture_2d<f32>;       // Rgba8Unorm   albedo.rgb + alpha
@group(1) @binding(1) var gbuf_normal:   texture_2d<f32>;       // Rgba16Float  world-space normal
@group(1) @binding(2) var gbuf_orm:      texture_2d<f32>;       // Rgba8Unorm   AO, roughness, metallic
@group(1) @binding(3) var gbuf_emissive: texture_2d<f32>;       // Rgba16Float  pre-multiplied emissive
@group(1) @binding(4) var gbuf_depth:    texture_depth_2d;      // Depth32Float

// Group 2 – lights, shadows, environment (same as forward geometry pass)
@group(2) @binding(0) var<storage, read> lights:          array<GpuLight>;
@group(2) @binding(1) var shadow_atlas:   texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;
@group(2) @binding(3) var env_cube:       texture_cube<f32>;
@group(2) @binding(4) var<storage, read> shadow_matrices: array<LightMatrix>;
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

fn shadow_factor(light_idx: u32, world_pos: vec3<f32>) -> f32 {
    if !ENABLE_SHADOWS { return 1.0; }
    if light_idx >= MAX_SHADOW_LIGHTS { return 1.0; }

    let light = lights[light_idx];
    var layer: u32;
    if light.light_type > 0.5 && light.light_type < 1.5 {
        let to_frag = world_pos - light.position;
        layer = light_idx * 6u + point_light_face(to_frag);
        let depth_bias   = 0.0008;
        let cascade_scale = 1.0;
        return sample_cascade_shadow(layer, depth_bias, cascade_scale, world_pos);
    } else if light.light_type < 0.5 {
        let dist = length(world_pos - camera.position);
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
        
        let depth_bias = 0.0003;
        let layer_a = light_idx * 6u + cascade_a;
        let cascade_scale_a = 1.0 + f32(cascade_a) * 1.5;
        let shadow_a = sample_cascade_shadow(layer_a, depth_bias, cascade_scale_a, world_pos);
        
        // If no blending needed, return immediately
        if blend <= 0.001 { return shadow_a; }
        if blend >= 0.999 && cascade_b != cascade_a {
            let layer_b = light_idx * 6u + cascade_b;
            let cascade_scale_b = 1.0 + f32(cascade_b) * 1.5;
            let shadow_b = sample_cascade_shadow(layer_b, depth_bias, cascade_scale_b, world_pos);
            return mix(shadow_a, shadow_b, blend);
        }
        
        // Blend between cascades
        if cascade_b != cascade_a {
            let layer_b = light_idx * 6u + cascade_b;
            let cascade_scale_b = 1.0 + f32(cascade_b) * 1.5;
            let shadow_b = sample_cascade_shadow(layer_b, depth_bias, cascade_scale_b, world_pos);
            return mix(shadow_a, shadow_b, blend);
        }
        
        return shadow_a;
    } else {
        layer = light_idx * 6u;
        let depth_bias   = 0.0003;
        let cascade_scale = 1.0;
        return sample_cascade_shadow(layer, depth_bias, cascade_scale, world_pos);
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
    let emissive  = textureLoad(gbuf_emissive, pix, 0).rgb;

    let albedo    = albedo_a.rgb;
    let alpha     = albedo_a.a;
    let N         = normalize(normal_r.xyz);
    let ao        = orm_r.r;
    let roughness = orm_r.g;
    let metallic  = orm_r.b;

    // ── Reconstruct world position from depth + inv_view_proj ────────────────
    // clip_pos.xy is in viewport space (0→width, 0→height, y↓).
    // Convert to NDC: x ∈ [-1,1], y ∈ [1,-1] (wgpu NDC y+ = up, viewport y+ = down).
    let screen_size = vec2<f32>(textureDimensions(gbuf_albedo));
    let uv_01       = in.clip_pos.xy / screen_size;
    let ndc_xy      = vec2<f32>(uv_01.x * 2.0 - 1.0, 1.0 - uv_01.y * 2.0);
    let world_h     = camera.view_proj_inv * vec4<f32>(ndc_xy, depth, 1.0);
    let world_pos   = world_h.xyz / world_h.w;

    // ── PBR setup ─────────────────────────────────────────────────────────────
    let F0  = mix(vec3<f32>(0.04), albedo, metallic);
    let V   = normalize(camera.position - world_pos);
    let NdV = max(dot(N, V), 0.0);

    // ── Direct lighting — shadow computed ONCE per light, reused for sky_occ ──
    // GPU-driven: iterate all visible lights (already culled on CPU by distance)
    var Lo            = vec3<f32>(0.0);
    var sky_occlusion = 1.0;
    if ENABLE_LIGHTING {
        for (var i = 0u; i < globals.light_count; i++) {
            let sf = shadow_factor(i, world_pos);
            if lights[i].light_type < 0.5 { sky_occlusion = sf; }
            
            // Distance-based fade: smooth fade-out as light goes beyond render distance
            let cam_to_light = lights[i].position - camera.position;
            let dist_to_light = length(cam_to_light);
            let light_range = max(lights[i].range, 0.001);
            
            // Fade zone: fully visible at 2.0x range, fade out toward 2.2x range
            let fade_in_dist = light_range * 2.0;
            let fade_out_dist = light_range * 2.2;
            let distance_fade = 1.0 - smoothstep(fade_in_dist, fade_out_dist, dist_to_light);
            
            let light_contrib = pbr_direct_light(lights[i], world_pos, N, V,
                                   F0, albedo, roughness, metallic, sf);
            Lo += light_contrib * distance_fade;
        }
    }

    // ── RC indirect diffuse ───────────────────────────────────────────────────
    let rc_irr   = sample_rc_irradiance(world_pos, N);
    let F_ibl    = fresnel_schlick_roughness(NdV, F0, roughness);
    let kD_ibl   = (1.0 - F_ibl) * (1.0 - metallic);
    let diff_ind = kD_ibl * rc_irr * albedo;

    // ── Indirect specular: environment cubemap ────────────────────────────────
    let R            = reflect(-V, N);
    let env_sample   = textureSample(env_cube, env_sampler, R).rgb;
    let spec_scale   = 1.0 - roughness * roughness;
    let spec_ind     = F_ibl * env_sample * spec_scale;

    // ── Hemisphere ambient fallback ───────────────────────────────────────────
    let sky_color    = globals.ambient_color.rgb * globals.ambient_intensity;
    let ground_color = sky_color * 0.15;
    let hemi_t       = N.y * 0.5 + 0.5;
    let hemi_ambient = mix(ground_color, sky_color, hemi_t) * albedo * sky_occlusion;

    let rc_weight       = clamp(length(rc_irr) * 4.0, 0.0, 1.0);
    let ambient_fallback = mix(hemi_ambient, diff_ind * sky_occlusion, rc_weight);

    // ── Combine ───────────────────────────────────────────────────────────────
    let indirect  = (ambient_fallback + spec_ind) * ao;
    var color     = Lo + indirect;
    color        += emissive;               // emissive from G-buffer
    color         = apply_bloom(color);
    color         = aces_tonemap(color);

    return vec4<f32>(color, alpha);
}
