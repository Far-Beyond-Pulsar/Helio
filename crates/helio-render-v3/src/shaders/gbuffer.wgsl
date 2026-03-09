// gbuffer.wgsl — PBR attribute packing into 4 MRT.
// GBuffer layout:
//   target0 = Rgba8Unorm    — albedo (RGB) + mask_alpha (A)
//   target1 = Rgba16Float   — world-space normal (RG oct-encoded) + metallic (B) + roughness (A)
//   target2 = Rgba8Unorm    — occlusion (R) + custom/emissive_scale (G) + unused (BA)
//   target3 = Rgba16Float   — emissive (RGB) + 0 (A)

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,
};

struct Globals {
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec3<f32>,
    csm_split_count:   u32,
    rc_world_min:      vec3<f32>,
    _pad0:             u32,
};

// Layout must match Rust MaterialUniform exactly:
//   base_color[4]  metallic  roughness  emissive_factor  ao  emissive_color[3]  alpha_cutoff
struct MaterialUniform {
    base_color:      vec4<f32>,   // [0-15]
    metallic:        f32,         // [16]
    roughness:       f32,         // [20]
    emissive_factor: f32,         // [24]
    ao:              f32,         // [28]
    emissive_color:  vec3<f32>,   // [32-43]  (align 16; 32 is 16-aligned)
    alpha_cutoff:    f32,         // [44]
};

@group(0) @binding(0) var<uniform> camera:  Camera;
@group(0) @binding(1) var<uniform> globals: Globals;

@group(1) @binding(0) var<uniform> material:          MaterialUniform;
@group(1) @binding(1) var                albedo_tex:  texture_2d<f32>;
@group(1) @binding(2) var                normal_tex:  texture_2d<f32>;
@group(1) @binding(3) var                tex_sampler: sampler;
@group(1) @binding(4) var                orm_tex:     texture_2d<f32>;
@group(1) @binding(5) var                emissive_tex: texture_2d<f32>;

struct VertexIn {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) uv:             vec2<f32>,
    @location(3) packed_normal:  vec4<f32>,
    @location(4) packed_tangent: vec4<f32>,
    // Instance data
    @location(5) i0: vec4<f32>,
    @location(6) i1: vec4<f32>,
    @location(7) i2: vec4<f32>,
    @location(8) i3: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip_pos:   vec4<f32>,
    @location(0)       world_pos:  vec3<f32>,
    @location(1)       uv:         vec2<f32>,
    @location(2)       world_norm: vec3<f32>,
    @location(3)       world_tan:  vec3<f32>,
    @location(4)       bitangent_sign: f32,
};



@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    let model    = mat4x4<f32>(in.i0, in.i1, in.i2, in.i3);
    let norm_mat = mat3x3<f32>(in.i0.xyz, in.i1.xyz, in.i2.xyz); // upper-left 3×3

    let world_pos = model * vec4<f32>(in.position, 1.0);
    let norm      = normalize(in.packed_normal.xyz);
    let tan       = normalize(in.packed_tangent.xyz);

    var out: VertexOut;
    out.clip_pos        = camera.view_proj * world_pos;
    out.world_pos       = world_pos.xyz;
    out.uv              = in.uv;
    out.world_norm      = normalize(norm_mat * norm);
    out.world_tan       = normalize(norm_mat * tan);
    out.bitangent_sign  = in.bitangent_sign;
    return out;
}

// Octahedral encoding (2 components → keeps 16-bit float precision).
fn oct_encode(n: vec3<f32>) -> vec2<f32> {
    let p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z < 0.0 {
        return (1.0 - abs(p.yx)) * sign(p);
    }
    return p;
}

struct GbufOut {
    @location(0) albedo_mask: vec4<f32>,
    @location(1) norm_mr:     vec4<f32>,
    @location(2) ao_extra:    vec4<f32>,
    @location(3) emissive:    vec4<f32>,
};

@fragment
fn fs_opaque(in: VertexOut) -> GbufOut {
    let albedo_smp = textureSample(albedo_tex, tex_sampler, in.uv);
    let albedo     = albedo_smp.rgb * material.base_color.rgb;

    let norm_smp   = textureSample(normal_tex, tex_sampler, in.uv).rgb * 2.0 - 1.0;
    let bitangent  = cross(in.world_norm, in.world_tan) * in.bitangent_sign;
    let tbn        = mat3x3<f32>(in.world_tan, bitangent, in.world_norm);
    let world_norm = normalize(tbn * norm_smp);

    let orm_smp = textureSample(orm_tex, tex_sampler, in.uv);
    let ao       = orm_smp.r * material.ao;
    let rough    = orm_smp.g * material.roughness;
    let metal    = orm_smp.b * material.metallic;

    let emissive = textureSample(emissive_tex, tex_sampler, in.uv).rgb
                   * material.emissive_color * material.emissive_factor;

    var out: GbufOut;
    out.albedo_mask = vec4<f32>(albedo, 1.0);
    out.norm_mr     = vec4<f32>(oct_encode(world_norm) * 0.5 + 0.5, metal, rough);
    out.ao_extra    = vec4<f32>(ao, 0.0, 0.0, 0.0);
    out.emissive    = vec4<f32>(emissive, 0.0);
    return out;
}

@fragment
fn fs_masked(in: VertexOut) -> GbufOut {
    let albedo_smp = textureSample(albedo_tex, tex_sampler, in.uv);
    if albedo_smp.a < material.alpha_cutoff { discard; }

    let albedo = albedo_smp.rgb * material.base_color.rgb;

    let norm_smp  = textureSample(normal_tex, tex_sampler, in.uv).rgb * 2.0 - 1.0;
    let bitangent = cross(in.world_norm, in.world_tan) * in.bitangent_sign;
    let tbn       = mat3x3<f32>(in.world_tan, bitangent, in.world_norm);
    let world_norm = normalize(tbn * norm_smp);

    let orm_smp = textureSample(orm_tex, tex_sampler, in.uv);
    let ao       = orm_smp.r * material.ao;
    let rough    = orm_smp.g * material.roughness;
    let metal    = orm_smp.b * material.metallic;
    let emissive = textureSample(emissive_tex, tex_sampler, in.uv).rgb
                   * material.emissive_color * material.emissive_factor;

    var out: GbufOut;
    out.albedo_mask = vec4<f32>(albedo, albedo_smp.a);
    out.norm_mr     = vec4<f32>(oct_encode(world_norm) * 0.5 + 0.5, metal, rough);
    out.ao_extra    = vec4<f32>(ao, 0.0, 0.0, 0.0);
    out.emissive    = vec4<f32>(emissive, 0.0);
    return out;
}
