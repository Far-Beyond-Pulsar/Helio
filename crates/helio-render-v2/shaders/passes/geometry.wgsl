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

/// Select a cube face index (0-5 = +X,-X,+Y,-Y,+Z,-Z) from a direction vector.
fn point_light_face(dir: vec3<f32>) -> u32 {
    let a = abs(dir);
    if a.x >= a.y && a.x >= a.z {
        return select(0u, 1u, dir.x < 0.0); // +X=0, -X=1
    } else if a.y >= a.x && a.y >= a.z {
        return select(2u, 3u, dir.y < 0.0); // +Y=2, -Y=3
    } else {
        return select(4u, 5u, dir.z < 0.0); // +Z=4, -Z=5
    }
}

/// PCF shadow factor for the given light.
/// Returns 1.0 (fully lit) … 0.0 (fully shadowed), with soft Poisson-disk filtering.
fn shadow_factor(light_idx: u32, world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if !ENABLE_SHADOWS {
        return 1.0;
    }

    let light = lights[light_idx];

    // Select atlas layer: point lights use the face matching dominant axis;
    // directional/spot always use face 0.
    var layer: u32;
    if light.light_type > 0.5 && light.light_type < 1.5 {
        // Point light — pick face from world-space direction to fragment
        let to_frag = world_pos - light.position;
        layer = light_idx * 6u + point_light_face(to_frag);
    } else {
        layer = light_idx * 6u; // face 0
    }

    let light_clip = shadow_matrices[layer].mat * vec4<f32>(world_pos, 1.0);
    // Reject geometry behind the light clip plane
    if light_clip.w <= 0.0 {
        return 1.0;
    }

    let ndc = light_clip.xyz / light_clip.w;
    // wgpu/Vulkan NDC: Y=-1 is top of screen, texture V=0 is top → flip Y
    let shadow_uv = vec2<f32>(ndc.x * 0.5 + 0.5, -ndc.y * 0.5 + 0.5);

    // Reject out-of-frustum fragments
    if any(shadow_uv < vec2<f32>(0.0)) || any(shadow_uv > vec2<f32>(1.0))
       || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }

    // Normal-based slope bias: less bias when light hits head-on, more at grazing
    var L_dir: vec3<f32>;
    if light.light_type < 0.5 {
        L_dir = normalize(-light.direction);
    } else {
        L_dir = normalize(light.position - world_pos);
    }
    let n_dot_l    = max(dot(world_normal, L_dir), 0.0);
    let slope_bias = mix(0.001, 0.0001, n_dot_l);
    let depth      = ndc.z - slope_bias;

    // 16-tap Poisson disk PCF
    // textureSampleCompareLevel with LessEqual returns 1.0 when fragment is lit
    // (stored_depth >= frag_depth → not occluded), 0.0 when in shadow
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
    // lit_sum/16 = fraction of samples that pass (1.0=fully lit, 0.0=fully shadowed)
    return lit_sum / 16.0;
}

// Evaluate one light's contribution (Blinn-Phong: diffuse + specular + shadow).
fn eval_light(light_idx: u32, light: GpuLight, world_pos: vec3<f32>, normal: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    var L: vec3<f32>;
    var attenuation: f32 = 1.0;

    if light.light_type < 0.5 {
        // Directional
        L = normalize(-light.direction);
    } else {
        // Point or spot — smooth quadratic falloff that reaches 0 at range
        let to_light = light.position - world_pos;
        let dist     = length(to_light);
        if dist > light.range { return vec3<f32>(0.0); }
        L            = to_light / dist;
        let ratio    = dist / light.range;
        let falloff  = max(0.0, 1.0 - ratio * ratio);
        attenuation  = falloff * falloff;  // smooth, no rescaling — intensity controls brightness

        if light.light_type > 1.5 {
            // Spot cone
            let cos_angle = dot(-L, normalize(light.direction));
            let spot      = smoothstep(cos(light.outer_angle), cos(light.inner_angle), cos_angle);
            attenuation  *= spot;
        }
    }

    let ndotl   = max(dot(normal, L), 0.0);
    let sf      = shadow_factor(light_idx, world_pos, normal);

    // Blinn-Phong specular
    let half_v  = normalize(L + view_dir);
    let spec    = pow(max(dot(normal, half_v), 0.0), 32.0) * 0.3;

    let lit     = light.color * light.intensity * attenuation * sf;
    return lit * (ndotl + spec);
}

// Accumulate all active lights
fn calculate_lighting(world_pos: vec3<f32>, normal: vec3<f32>, base_color: vec3<f32>, view_dir: vec3<f32>) -> vec3<f32> {
    let ambient = globals.ambient_intensity * globals.ambient_color.rgb * base_color;

    if !ENABLE_LIGHTING {
        return ambient;
    }

    var diffuse = vec3<f32>(0.0);
    for (var i: u32 = 0u; i < globals.light_count; i++) {
        diffuse += eval_light(i, lights[i], world_pos, normal, view_dir) * base_color;
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

    let view_dir = normalize(camera.position - input.world_position);
    var color = calculate_lighting(input.world_position, input.world_normal, base_color, view_dir);

    // Emissive
    color += base_color * material.emissive;

    // Bloom (branch eliminated at compile time when disabled)
    color = apply_bloom(color);

    return vec4<f32>(color, material.base_color.a);
}
