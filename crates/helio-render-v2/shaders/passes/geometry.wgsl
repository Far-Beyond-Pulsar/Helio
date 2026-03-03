//! Main geometry pass shader – Cook-Torrance GGX PBR
//!
//! Feature override constants (injected by PipelineCache before the source):
//!
//!   override ENABLE_LIGHTING:   bool  = false;
//!   override LIGHT_COUNT:       u32   = 0u;
//!   override ENABLE_SHADOWS:    bool  = false;
//!   override MAX_SHADOW_LIGHTS: u32   = 0u;
//!   override ENABLE_BLOOM:      bool  = false;
//!   override BLOOM_INTENSITY:   f32   = 0.3;
//!   override BLOOM_THRESHOLD:   f32   = 1.0;

// ============================================================================
// Common Bindings (Group 0-2)
// ============================================================================

struct Camera {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
}

struct Globals {
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: vec4<f32>,
    rc_world_min: vec4<f32>,
    rc_world_max: vec4<f32>,
    /// World-space distance at each CSM cascade boundary (x=c0, y=c1, z=c2, w=c3).
    csm_splits: vec4<f32>,
}

// Material uniform – must match material::MaterialUniform (48 bytes).
struct Material {
    base_color:      vec4<f32>,   // offset  0
    metallic:        f32,          // offset 16
    roughness:       f32,          // offset 20
    emissive_factor: f32,          // offset 24
    ao:              f32,          // offset 28
    emissive_color:  vec3<f32>,   // offset 32  (alignment 16 — ok)
    _pad:            f32,          // offset 44
}

@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

@group(1) @binding(0) var<uniform> material:          Material;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_map:         texture_2d<f32>;
@group(1) @binding(3) var material_sampler:   sampler;
@group(1) @binding(4) var orm_texture:        texture_2d<f32>; // R=AO, G=roughness, B=metallic
@group(1) @binding(5) var emissive_texture:   texture_2d<f32>;

// ============================================================================
// Vertex Input/Output
// ============================================================================

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,   // Packed SNORM8x4
    @location(4) tangent:        u32,   // Packed SNORM8x4 (kept for buffer compat, TBN computed via derivatives)
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
}

// ============================================================================
// Vertex Shader
// ============================================================================

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position  = camera.view_proj * vec4<f32>(vertex.position, 1.0);
    out.world_position = vertex.position;
    out.world_normal   = normalize(decode_snorm8x4(vertex.normal));
    out.tex_coords     = vertex.tex_coords;
    return out;
}

// ============================================================================
// Lighting helpers
// ============================================================================

const PI: f32 = 3.14159265359;

fn saturate_f(x: f32) -> f32 { return clamp(x, 0.0, 1.0); }

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

struct GpuLight {
    position:    vec3<f32>,
    light_type:  f32,   // 0=directional  1=point  2=spot
    direction:   vec3<f32>,
    range:       f32,
    color:       vec3<f32>,
    intensity:   f32,
    inner_angle: f32,
    outer_angle: f32,
    _pad:        vec2<f32>,
}

@group(2) @binding(0) var<storage, read> lights:          array<GpuLight>;
@group(2) @binding(1) var shadow_atlas:   texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;
@group(2) @binding(3) var env_cube:       texture_cube<f32>;

struct LightMatrix { mat: mat4x4<f32> }
@group(2) @binding(4) var<storage, read> shadow_matrices: array<LightMatrix>;
@group(2) @binding(5) var rc_cascade0: texture_2d<f32>;

// ============================================================================
// Shadow helpers (unchanged from original)
// ============================================================================

const ATLAS_SIZE: f32 = 512.0;

var<private> POISSON_DISK: array<vec2<f32>, 16> = array<vec2<f32>, 16>(
    vec2<f32>(-0.94201624, -0.39906216),
    vec2<f32>( 0.94558609, -0.76890725),
    vec2<f32>(-0.09418410, -0.92938870),
    vec2<f32>( 0.34495938,  0.29387760),
    vec2<f32>(-0.91588581,  0.45771432),
    vec2<f32>(-0.81544232, -0.87912464),
    vec2<f32>(-0.38277543,  0.27676845),
    vec2<f32>( 0.97484398,  0.75648379),
    vec2<f32>( 0.44323325, -0.97511554),
    vec2<f32>( 0.53742981, -0.47373420),
    vec2<f32>(-0.26496911, -0.41893023),
    vec2<f32>( 0.79197514,  0.19090188),
    vec2<f32>(-0.24188840,  0.99706507),
    vec2<f32>(-0.81409955,  0.91437590),
    vec2<f32>( 0.19984126,  0.78641367),
    vec2<f32>( 0.14383161, -0.14100790),
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

fn shadow_factor(light_idx: u32, world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if !ENABLE_SHADOWS { return 1.0; }

    let light = lights[light_idx];
    var layer: u32;
    if light.light_type > 0.5 && light.light_type < 1.5 {
        // Point light: select cube-map face from world-to-light vector
        let to_frag = world_pos - light.position;
        layer = light_idx * 6u + point_light_face(to_frag);
    } else if light.light_type < 0.5 {
        // Directional light: choose CSM cascade by world-space distance from camera.
        // Slots 0-3 hold the four cascade matrices (see compute_directional_cascades).
        let dist = length(world_pos - camera.position);
        var cascade = 3u;
        if      dist < globals.csm_splits.x { cascade = 0u; }
        else if dist < globals.csm_splits.y { cascade = 1u; }
        else if dist < globals.csm_splits.z { cascade = 2u; }
        layer = light_idx * 6u + cascade;
    } else {
        // Spot light: single projection at slot 0
        layer = light_idx * 6u;
    }

    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    if light_clip.w <= 0.0 { return 1.0; }

    let ndc       = light_clip.xyz / light_clip.w;
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0))
       || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let depth = ndc.z;

    let filter_radius = 2.0 / ATLAS_SIZE;
    var lit_sum = 0.0;
    for (var i = 0; i < 16; i++) {
        let offset = POISSON_DISK[i] * filter_radius;
        lit_sum += textureSampleCompareLevel(
            shadow_atlas, shadow_sampler,
            shadow_uv + offset,
            i32(layer),
            depth,
        );
    }
    return lit_sum / 16.0;
}

// ============================================================================
// Cook-Torrance GGX PBR BRDF
// ============================================================================

// GGX normal distribution function
fn distribution_ggx(N: vec3<f32>, H: vec3<f32>, roughness: f32) -> f32 {
    let a    = roughness * roughness;
    let a2   = a * a;
    let NdH  = max(dot(N, H), 0.0);
    let NdH2 = NdH * NdH;
    let denom = NdH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

// Schlick-GGX geometry term (single direction)
fn geometry_schlick_ggx(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k + 0.0001);
}

// Smith combined geometry
fn geometry_smith(N: vec3<f32>, V: vec3<f32>, L: vec3<f32>, roughness: f32) -> f32 {
    let NdV = max(dot(N, V), 0.0);
    let NdL = max(dot(N, L), 0.0);
    return geometry_schlick_ggx(NdV, roughness) * geometry_schlick_ggx(NdL, roughness);
}

// Fresnel-Schlick
fn fresnel_schlick(cos_theta: f32, F0: vec3<f32>) -> vec3<f32> {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick with roughness (IBL)
fn fresnel_schlick_roughness(cos_theta: f32, F0: vec3<f32>, roughness: f32) -> vec3<f32> {
    let one_minus_r = vec3<f32>(1.0 - roughness);
    return F0 + (max(one_minus_r, F0) - F0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

// Evaluate one light using the full Cook-Torrance BRDF + PCF shadows
fn pbr_direct_light(
    light_idx: u32,
    light:     GpuLight,
    world_pos: vec3<f32>,
    N:         vec3<f32>,
    V:         vec3<f32>,
    F0:        vec3<f32>,
    albedo:    vec3<f32>,
    roughness: f32,
    metallic:  f32,
) -> vec3<f32> {
    var L: vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type < 0.5 {
        L        = normalize(-light.direction);
        radiance = light.color * light.intensity;
    } else {
        let to_light = light.position - world_pos;
        let dist     = length(to_light);
        if dist > light.range { return vec3<f32>(0.0); }
        L = to_light / dist;
        let ratio    = dist / light.range;
        let falloff  = max(0.0, 1.0 - ratio * ratio);
        var atten    = falloff * falloff;
        if light.light_type > 1.5 {
            let cos_a = dot(-L, normalize(light.direction));
            atten    *= smoothstep(cos(light.outer_angle), cos(light.inner_angle), cos_a);
        }
        radiance = light.color * light.intensity * atten;
    }

    let NdL = max(dot(N, L), 0.0);
    if NdL == 0.0 { return vec3<f32>(0.0); }

    let H   = normalize(V + L);
    let D   = distribution_ggx(N, H, roughness);
    let G   = geometry_smith(N, V, L, roughness);
    let F   = fresnel_schlick(max(dot(H, V), 0.0), F0);

    let kS      = F;
    let kD      = (1.0 - kS) * (1.0 - metallic);
    let specular = D * G * F / (4.0 * max(dot(N, V), 0.0) * NdL + 0.0001);

    let sf = shadow_factor(light_idx, world_pos, N);
    return (kD * albedo / PI + specular) * radiance * NdL * sf;
}

// ============================================================================
// Radiance Cascades GI (unchanged)
// ============================================================================

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

fn rc_corner_irradiance(px: u32, py: u32, pz: u32, normal: vec3<f32>) -> vec3<f32> {
    let dim = RC_PROBE_DIM - 1u;
    let cpx = min(px, dim); let cpy = min(py, dim); let cpz = min(pz, dim);
    var irr  = vec3<f32>(0.0);
    var wsum = 0.0;
    for (var ddx: u32 = 0u; ddx < RC_DIR_DIM; ddx++) {
        for (var ddy: u32 = 0u; ddy < RC_DIR_DIM; ddy++) {
            let dir_uv = (vec2<f32>(f32(ddx), f32(ddy)) + 0.5) / f32(RC_DIR_DIM);
            let dir    = rc_oct_decode(dir_uv);
            let cos_w  = max(0.0, dot(normal, dir));
            if cos_w > 0.001 {
                let atlas_x = i32(cpx * RC_DIR_DIM + ddx);
                let atlas_y = i32((cpy * RC_PROBE_DIM + cpz) * RC_DIR_DIM + ddy);
                let rad = textureLoad(rc_cascade0, vec2<i32>(atlas_x, atlas_y), 0).rgb;
                irr  += rad * cos_w;
                wsum += cos_w;
            }
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
    let cell_size   = world_size / f32(RC_PROBE_DIM);
    let probe_f     = (world_pos - world_min) / cell_size - 0.5;
    let probe_dim_f = f32(RC_PROBE_DIM) - 1.0;
    let pf  = clamp(probe_f, vec3<f32>(0.0), vec3<f32>(probe_dim_f));
    let pi  = vec3<u32>(u32(pf.x), u32(pf.y), u32(pf.z));
    let frc = fract(pf);

    let c000 = rc_corner_irradiance(pi.x,      pi.y,      pi.z,      normal);
    let c001 = rc_corner_irradiance(pi.x,      pi.y,      pi.z + 1u, normal);
    let c010 = rc_corner_irradiance(pi.x,      pi.y + 1u, pi.z,      normal);
    let c011 = rc_corner_irradiance(pi.x,      pi.y + 1u, pi.z + 1u, normal);
    let c100 = rc_corner_irradiance(pi.x + 1u, pi.y,      pi.z,      normal);
    let c101 = rc_corner_irradiance(pi.x + 1u, pi.y,      pi.z + 1u, normal);
    let c110 = rc_corner_irradiance(pi.x + 1u, pi.y + 1u, pi.z,      normal);
    let c111 = rc_corner_irradiance(pi.x + 1u, pi.y + 1u, pi.z + 1u, normal);

    let c00 = mix(c000, c001, frc.z);
    let c01 = mix(c010, c011, frc.z);
    let c10 = mix(c100, c101, frc.z);
    let c11 = mix(c110, c111, frc.z);
    let c0  = mix(c00, c01, frc.y);
    let c1  = mix(c10, c11, frc.y);
    return mix(c0, c1, frc.x);
}

// ============================================================================
// ACES filmic tonemapping (matches sky.wgsl)
// ============================================================================

fn aces_tonemap(x: vec3<f32>) -> vec3<f32> {
    // Narkowicz 2015 ACES approximation
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return saturate((x * (a * x + b)) / (x * (c * x + d) + e));
}

// ============================================================================
// Bloom helper (unchanged)
// ============================================================================

fn apply_bloom(color: vec3<f32>) -> vec3<f32> {
    if !ENABLE_BLOOM { return color; }
    let lum    = luminance(color);
    let excess = max(lum - BLOOM_THRESHOLD, 0.0);
    return color + color * (excess * BLOOM_INTENSITY);
}

// ============================================================================
// Fragment entry
// ============================================================================

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.tex_coords;

    // ── Base color ────────────────────────────────────────────────────────────
    let tex_sample = textureSample(base_color_texture, material_sampler, uv);
    let albedo     = material.base_color.rgb * tex_sample.rgb;
    let alpha      = material.base_color.a  * tex_sample.a;

    // ── Normal mapping – derivative-based TBN (Schuler/ShaderX5, used by Three.js) ──
    // Reconstructs the tangent frame entirely from screen-space position + UV
    // derivatives, so it is ALWAYS correct regardless of vertex tangent quality.
    let N_geom = normalize(input.world_normal);
    let q0     = dpdx(input.world_position);
    let q1     = dpdy(input.world_position);
    let st0    = dpdx(uv);
    let st1    = dpdy(uv);
    let q1perp = cross(q1, N_geom);
    let q0perp = cross(N_geom, q0);
    let T_deriv = q1perp * st0.x + q0perp * st1.x;
    let B_deriv = q1perp * st0.y + q0perp * st1.y;
    let det     = max(dot(T_deriv, T_deriv), dot(B_deriv, B_deriv));
    let scale   = select(0.0, inverseSqrt(det), det > 1e-10);
    let norm_ts = textureSample(normal_map, material_sampler, uv).rgb * 2.0 - 1.0;
    let N       = normalize(
        T_deriv * (norm_ts.x * scale) +
        B_deriv * (norm_ts.y * scale) +
        N_geom  *  norm_ts.z
    );

    // ── ORM texture (R=AO, G=roughness, B=metallic) ──────────────────────────
    let orm      = textureSample(orm_texture, material_sampler, uv).rgb;
    let ao       = material.ao       * orm.r;
    let roughness = clamp(material.roughness * orm.g, 0.04, 1.0);
    let metallic  = clamp(material.metallic  * orm.b, 0.0,  1.0);

    // ── PBR setup ─────────────────────────────────────────────────────────────
    // Dielectric F0 = 0.04; metallic surfaces use albedo as F0
    let F0   = mix(vec3<f32>(0.04), albedo, metallic);
    let V    = normalize(camera.position - input.world_position);
    let NdV  = max(dot(N, V), 0.0);

    // ── Direct lighting (Cook-Torrance) ───────────────────────────────────────
    var Lo = vec3<f32>(0.0);
    if ENABLE_LIGHTING {
        for (var i = 0u; i < globals.light_count; i++) {
            Lo += pbr_direct_light(i, lights[i], input.world_position, N, V,
                                   F0, albedo, roughness, metallic);
        }
    }

    // ── Sky occlusion: find first directional light, sample its cascade shadow ─
    // Ambient (skylight) comes from the open sky. If a fragment is occluded from
    // the sun's direction it is also occluded from the sky, so we modulate the
    // ambient by the directional shadow factor. Soft PCF makes this a smooth fade.
    var sky_occlusion = 1.0;
    if ENABLE_SHADOWS && ENABLE_LIGHTING {
        for (var i = 0u; i < globals.light_count; i++) {
            if lights[i].light_type < 0.5 {
                sky_occlusion = shadow_factor(i, input.world_position, N);
                break;
            }
        }
    }

    // ── Indirect diffuse: Radiance Cascades GI ────────────────────────────────
    let rc_irr    = sample_rc_irradiance(input.world_position, N);
    let F_ibl     = fresnel_schlick_roughness(NdV, F0, roughness);
    let kD_ibl    = (1.0 - F_ibl) * (1.0 - metallic);
    let diffuse_indirect = kD_ibl * rc_irr * albedo;

    // ── Indirect specular: environment cubemap ─────────────────────────────────
    // Reuse material_sampler (group 1 binding 3) to sample the env cube.
    // A roughness-based lod would require a pre-filtered env map; we approximate
    // by attenuating specular by roughness².
    let R           = reflect(-V, N);
    let env_sample  = textureSample(env_cube, material_sampler, R).rgb;
    let spec_scale  = (1.0 - roughness * roughness);
    let specular_indirect = F_ibl * env_sample * spec_scale;

    // ── Ambient fallback (active when neither lights nor RC are providing) ─────
    // Modulated by sky_occlusion: surfaces under roofs/geometry don't receive
    // the full sky ambient, matching the directional light's shadow coverage.
    let ambient_fallback = globals.ambient_color.rgb * globals.ambient_intensity * albedo * sky_occlusion;

    // ── Combine: direct + indirect (AO applied to indirect only) ─────────────
    let indirect = (diffuse_indirect * sky_occlusion + specular_indirect + ambient_fallback) * ao;
    var color    = Lo + indirect;

    // ── Emissive ──────────────────────────────────────────────────────────────
    let emissive_tex = textureSample(emissive_texture, material_sampler, uv).rgb;
    color += material.emissive_color * emissive_tex * material.emissive_factor;

    // ── Bloom ─────────────────────────────────────────────────────────────────
    color = apply_bloom(color);

    // ── ACES tonemapping (HDR → LDR, matches sky shader) ─────────────────────
    color = aces_tonemap(color);

    return vec4<f32>(color, alpha);
}

