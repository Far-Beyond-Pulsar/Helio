//! G-buffer write pass.
//!
//! Rasterises scene geometry into four screen-sized textures:
//!   target 0 – albedo   (Rgba8Unorm)   : linear albedo.rgb + alpha
//!   target 1 – normal   (Rgba16Float)  : world-space normal XYZ  (W = 0)
//!   target 2 – orm      (Rgba8Unorm)   : AO, roughness, metallic (W = 0)
//!   target 3 – emissive (Rgba16Float)  : pre-multiplied emissive.rgb (W = 0)
//!
//! No lighting is performed here at all.  The deferred_lighting pass reads
//! these textures one full-screen draw later, running PBR × all lights once
//! per screen pixel rather than once per mesh fragment × light.

struct Camera {
    view_proj: mat4x4<f32>,
    position:  vec3<f32>,
    time:      f32,
}

struct Material {
    base_color:      vec4<f32>,
    metallic:        f32,
    roughness:       f32,
    emissive_factor: f32,
    ao:              f32,
    emissive_color:  vec3<f32>,
    alpha_cutoff:    f32,
}

@group(0) @binding(0) var <uniform> camera:   Camera;
@group(1) @binding(0) var <uniform> material: Material;
@group(1) @binding(1) var base_color_texture: texture_2d<f32>;
@group(1) @binding(2) var normal_map:         texture_2d<f32>;
@group(1) @binding(3) var material_sampler:   sampler;
@group(1) @binding(4) var orm_texture:        texture_2d<f32>;
@group(1) @binding(5) var emissive_texture:   texture_2d<f32>;

// ── Vertex ────────────────────────────────────────────────────────────────────

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

// GPU Scene: persistent per-object data (group 3, binding 0).
// Each instance carries only a 4-byte primitive_id; transforms and bounds
// come from the persistent storage buffer, uploaded once and dirty-tracked.
struct GpuPrimitive {
    transform:     mat4x4<f32>,   //  offset   0
    inv_trans_c0:  vec4<f32>,     //  offset  64
    inv_trans_c1:  vec4<f32>,     //  offset  80
    inv_trans_c2:  vec4<f32>,     //  offset  96
    bounds_center: vec3<f32>,     //  offset 112
    bounds_radius: f32,           //  offset 124
    material_id:   u32,           //  offset 128
    flags:         u32,           //  offset 132
    _pad:          vec2<u32>,     //  offset 136  →  total 144 bytes
}
@group(3) @binding(0) var<storage, read> gpu_primitives: array<GpuPrimitive>;

@vertex
fn vs_main(v: Vertex, @location(5) primitive_id: u32) -> VertexOutput {
    let prim       = gpu_primitives[primitive_id];
    let model      = prim.transform;
    let world_pos  = model * vec4<f32>(v.position, 1.0);
    // Correct normal transform via pre-computed inverse-transpose mat3x3.
    let normal_mat = mat3x3<f32>(prim.inv_trans_c0.xyz, prim.inv_trans_c1.xyz, prim.inv_trans_c2.xyz);
    var out: VertexOutput;
    out.clip_position  = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat * decode_snorm8x4(v.normal));
    out.tex_coords     = v.tex_coords;
    return out;
}

// ── Fragment ─────────────────────────────────────────────────────────────────

struct GBufferOutput {
    @location(0) albedo:   vec4<f32>,   // Rgba8Unorm
    @location(1) normal:   vec4<f32>,   // Rgba16Float
    @location(2) orm:      vec4<f32>,   // Rgba8Unorm
    @location(3) emissive: vec4<f32>,   // Rgba16Float
}

@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    let uv = input.tex_coords;

    // ── Base color ────────────────────────────────────────────────────────────
    let tex_sample = textureSample(base_color_texture, material_sampler, uv);
    let albedo     = material.base_color.rgb * tex_sample.rgb;
    let alpha      = material.base_color.a  * tex_sample.a;

    // Always drop fully transparent texels from PNGs even when alpha_cutoff is 0.
    if alpha <= 0.001 {
        discard;
    }

    // Deferred path transparency support: alpha cutout / masked materials.
    if alpha < material.alpha_cutoff {
        discard;
    }

    // ── Normal mapping – derivative-based TBN ────────────────────────────────
    let N_geom  = normalize(input.world_normal);
    let q0      = dpdx(input.world_position);
    let q1      = dpdy(input.world_position);
    let st0     = dpdx(uv);
    let st1     = dpdy(uv);
    let q1perp  = cross(q1, N_geom);
    let q0perp  = cross(N_geom, q0);
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

    // ── ORM ───────────────────────────────────────────────────────────────────
    let orm       = textureSample(orm_texture, material_sampler, uv).rgb;
    let ao        = material.ao       * orm.r;
    let roughness = clamp(material.roughness * orm.g, 0.04, 1.0);
    let metallic  = clamp(material.metallic  * orm.b, 0.0,  1.0);

    // ── Emissive (pre-multiplied into linear colour) ──────────────────────────
    let emissive_tex = textureSample(emissive_texture, material_sampler, uv).rgb;
    let emissive     = material.emissive_color * emissive_tex * material.emissive_factor;

    var out: GBufferOutput;
    out.albedo   = vec4<f32>(albedo,         alpha);
    out.normal   = vec4<f32>(N,              0.0);
    out.orm      = vec4<f32>(ao, roughness, metallic, 0.0);
    out.emissive = vec4<f32>(emissive,       0.0);
    return out;
}
