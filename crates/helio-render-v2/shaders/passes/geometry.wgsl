//! Main geometry pass shader
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
}

struct Material {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emissive: f32,
    ao: f32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_map: texture_2d<f32>;
@group(1) @binding(3) var material_sampler: sampler;

// ============================================================================
// Vertex Input/Output
// ============================================================================

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords: vec2<f32>,
    @location(3) normal: u32,      // Packed as SNORM8x4
    @location(4) tangent: u32,     // Packed as SNORM8x4
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

// ============================================================================
// Vertex Shader
// ============================================================================

fn decode_normal(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    var output: VertexOutput;

    // Transform position to clip space
    let world_pos = vec4<f32>(vertex.position, 1.0);
    output.clip_position = camera.view_proj * world_pos;
    output.world_position = vertex.position;

    // Decode and transform normal
    let normal = decode_normal(vertex.normal);
    output.world_normal = normalize(normal);

    output.tex_coords = vertex.tex_coords;

    return output;
}

// ============================================================================
// Fragment Shader
// ============================================================================

const PI: f32 = 3.14159265359;

fn saturate_f(x: f32) -> f32 {
    return clamp(x, 0.0, 1.0);
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// ── Lighting helpers ────────────────────────────────────────────────────────

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

@group(2) @binding(0) var<storage, read> lights:        array<GpuLight>;
@group(2) @binding(1) var shadow_atlas:   texture_depth_2d_array;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;

// Binding 3: env_cube (IBL, declared in layout but not used here)

struct LightMatrix { mat: mat4x4<f32> }
@group(2) @binding(4) var<storage, read> shadow_matrices: array<LightMatrix>;

// ── Shadow helpers ───────────────────────────────────────────────────────────

/// PCF shadow lookup for the given light.
/// Returns 1.0 (fully lit) or 0.0 (fully shadowed), with soft edges.
fn shadow_factor(light_idx: u32, world_pos: vec3<f32>) -> f32 {
    if !ENABLE_SHADOWS {
        return 1.0;
    }

    let light_clip = shadow_matrices[light_idx].mat * vec4<f32>(world_pos, 1.0);
    // Reject geometry behind the light (w <= 0 means behind clip plane)
    if light_clip.w <= 0.0 {
        return 1.0;
    }

    // Perspective divide → NDC
    let ndc = light_clip.xyz / light_clip.w;

    // Convert NDC xy to texture UV: [-1,1] → [0,1]
    let shadow_uv = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);

    // Reject out-of-frustum fragments
    if any(shadow_uv < vec2<f32>(0.0, 0.0)) || any(shadow_uv > vec2<f32>(1.0, 1.0))
       || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    let depth     = ndc.z;
    let bias      = 0.005;
    let texel     = 1.0 / 1024.0;

    // 3×3 PCF kernel
    // shadow_sampler uses LessEqual compare: stored_depth <= (depth - bias) → 1.0 (in shadow)
    var shadow_sum = 0.0;
    for (var xi = -1; xi <= 1; xi++) {
        for (var yi = -1; yi <= 1; yi++) {
            let off = vec2<f32>(f32(xi) * texel, f32(yi) * texel);
            shadow_sum += textureSampleCompareLevel(
                shadow_atlas, shadow_sampler,
                shadow_uv + off,
                i32(light_idx),
                depth - bias,
            );
        }
    }
    // shadow_sum/9 = 1.0 when fully in shadow → invert to get attenuation factor
    return 1.0 - shadow_sum / 9.0;
}

// Evaluate one light contribution (Lambertian diffuse)
fn eval_light(light: GpuLight, world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var L: vec3<f32>;
    var attenuation: f32 = 1.0;

    if light.light_type < 0.5 {
        // Directional
        L = normalize(-light.direction);
    } else {
        // Point or spot
        let to_light = light.position - world_pos;
        let dist     = length(to_light);
        L            = to_light / dist;
        let falloff  = saturate_f(1.0 - (dist / light.range));
        attenuation  = falloff * falloff;

        if light.light_type > 1.5 {
            // Spot cone
            let cos_angle = dot(-L, normalize(light.direction));
            let spot      = smoothstep(cos(light.outer_angle), cos(light.inner_angle), cos_angle);
            attenuation  *= spot;
        }
    }

    let ndotl = max(dot(normal, L), 0.0);
    return light.color * light.intensity * ndotl * attenuation;
}

// Accumulate all active lights
fn calculate_lighting(world_pos: vec3<f32>, normal: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let ambient = globals.ambient_intensity * globals.ambient_color.rgb * base_color;

    if !ENABLE_LIGHTING {
        return ambient;
    }

    var diffuse = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < globals.light_count; i++) {
        let sf = shadow_factor(i, world_pos);
        diffuse += eval_light(lights[i], world_pos, normal) * base_color * sf;
    }
    return ambient + diffuse;
}

// ── Bloom helper ─────────────────────────────────────────────────────────────

fn apply_bloom(color: vec3<f32>) -> vec3<f32> {
    if !ENABLE_BLOOM {
        return color;
    }
    let lum    = luminance(color);
    let excess = max(lum - BLOOM_THRESHOLD, 0.0);
    return color + color * (excess * BLOOM_INTENSITY);
}

// ── Procedural helpers ────────────────────────────────────────────────────────

fn checkerboard(uv: vec2<f32>, scale: f32) -> f32 {
    let f = floor(uv * scale);
    return fract((f.x + f.y) * 0.5) * 2.0;
}

// ── Fragment entry ────────────────────────────────────────────────────────────

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Procedural checkerboard mixed with material base color
    let checker    = checkerboard(input.tex_coords, 8.0);
    let dark       = vec3<f32>(0.25, 0.25, 0.28);
    let light_col  = vec3<f32>(0.85, 0.85, 0.82);
    let proc_color = mix(dark, light_col, checker);
    let base_color = material.base_color.rgb * proc_color;

    var color = calculate_lighting(input.world_position, input.world_normal, base_color);

    // Emissive
    color += base_color * material.emissive;

    // Bloom (branch eliminated at compile time when disabled)
    color = apply_bloom(color);

    return vec4<f32>(color, material.base_color.a);
}
