//! G-buffer write pass (GPU-driven).
//!
//! Rasterises scene geometry into four screen-sized textures:
//!   target 0 – albedo   (Rgba8Unorm)
//!   target 1 – normal   (Rgba16Float)
//!   target 2 – orm      (Rgba8Unorm)
//!   target 3 – emissive (Rgba16Float)
//!
//! Resolved F0 is packed into unused alpha channels:
//!   normal.a   = F0.r
//!   orm.a      = F0.g
//!   emissive.a = F0.b

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
    frame: u32,
    delta_time: f32,
    light_count: u32,
    ambient_intensity: f32,
    ambient_color: vec4<f32>,
    rc_world_min: vec4<f32>,
    rc_world_max: vec4<f32>,
    csm_splits: vec4<f32>,
    debug_mode: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU material (96 bytes, matches libhelio::GpuMaterial)
struct GpuMaterial {
    base_color:         vec4<f32>,
    emissive:           vec4<f32>,
    roughness_metallic: vec4<f32>,
    tex_base_color:     u32,
    tex_normal:         u32,
    tex_roughness:      u32,
    tex_emissive:       u32,
    tex_occlusion:      u32,
    workflow:           u32,
    flags:              u32,
    _pad:               u32,
}

/// Per-material texture metadata (224 bytes, matches helio::GpuMaterialTextures)
struct MaterialTextureSlot {
    texture_index: u32,
    uv_channel:    u32,
    _pad0:         u32,
    _pad1:         u32,
    offset_scale:  vec4<f32>,
    rotation:      vec4<f32>,
}

struct MaterialTextureData {
    base_color:         MaterialTextureSlot,
    normal:             MaterialTextureSlot,
    roughness_metallic: MaterialTextureSlot,
    emissive:           MaterialTextureSlot,
    occlusion:          MaterialTextureSlot,
    specular_color:     MaterialTextureSlot,
    specular_weight:    MaterialTextureSlot,
    params:             vec4<f32>,  // x=normal_scale, y=occlusion_strength, z=alpha_cutoff
}

/// Per-instance data (144 bytes). Must match `GpuInstanceData` in libhelio.
struct GpuInstanceData {
    transform:    mat4x4<f32>,  // offset   0  (64 bytes)
    normal_mat_0: vec4<f32>,    // offset  64  — row 0 of inv-transpose 3×3
    normal_mat_1: vec4<f32>,    // offset  80
    normal_mat_2: vec4<f32>,    // offset  96
    bounds:       vec4<f32>,    // offset 112
    mesh_id:      u32,          // offset 128
    material_id:  u32,          // offset 132
    flags:        u32,          // offset 136
    _pad:         u32,          // offset 140
}

@group(0) @binding(0) var<uniform>          camera:        Camera;
@group(0) @binding(1) var<uniform>          globals:       Globals;
@group(0) @binding(2) var<storage, read>    instance_data: array<GpuInstanceData>;

@group(1) @binding(0) var<storage, read>    materials:          array<GpuMaterial>;
@group(1) @binding(1) var<storage, read>    material_textures:  array<MaterialTextureData>;
@group(1) @binding(2) var                   scene_textures:     binding_array<texture_2d<f32>, 256>;
@group(1) @binding(3) var                   scene_samplers:     binding_array<sampler, 256>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) normal:         u32,
    @location(4) tangent:        u32,
}

struct VertexOutput {
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) world_tangent:  vec3<f32>,
    @location(4) bitangent_sign: f32,
    @location(5) material_id:    u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(v: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let inst       = instance_data[slot];
    let world_pos  = inst.transform * vec4<f32>(v.position, 1.0);

    // Normals transform by the inverse-transpose (stored in normal_mat).
    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );

    // Tangents are NOT normals — they transform by the regular upper-3×3 of
    // the model matrix (no inverse-transpose).  Extract it from column vectors.
    let model_mat3 = mat3x3<f32>(
        inst.transform[0].xyz,
        inst.transform[1].xyz,
        inst.transform[2].xyz,
    );

    var out: VertexOutput;
    out.clip_position  = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat  * decode_snorm8x4(v.normal));
    out.world_tangent  = normalize(model_mat3  * decode_snorm8x4(v.tangent));
    out.bitangent_sign = v.bitangent_sign;
    out.tex_coords     = v.tex_coords;
    out.material_id    = inst.material_id;
    return out;
}

// ── Fragment ─────────────────────────────────────────────────────────────────

struct GBufferOutput {
    @location(0) albedo:   vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) orm:      vec4<f32>,
    @location(3) emissive: vec4<f32>,
}

const NO_TEXTURE: u32 = 0xffffffffu;
const MATERIAL_WORKFLOW_METALLIC: u32 = 0u;
const MATERIAL_WORKFLOW_SPECULAR: u32 = 1u;

/// Select UV channel and apply texture transform
fn select_uv(slot: MaterialTextureSlot, base_uv: vec2<f32>) -> vec2<f32> {
    // TODO: support uv_channel when we have tex_coords1
    let scaled = base_uv * slot.offset_scale.zw;
    let s = slot.rotation.x;
    let c = slot.rotation.y;
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    );
    return rotated + slot.offset_scale.xy;
}

/// Sample texture from bindless array, or return fallback if NO_TEXTURE
fn sample_texture(slot: MaterialTextureSlot, base_uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if slot.texture_index == NO_TEXTURE {
        return fallback;
    }
    let uv = select_uv(slot, base_uv);
    return textureSample(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], uv);
}

fn resolve_specular_f0(
    material: GpuMaterial,
    material_tex: MaterialTextureData,
    albedo: vec3<f32>,
    metallic: f32,
    uv: vec2<f32>,
) -> vec3<f32> {
    if material.workflow == MATERIAL_WORKFLOW_SPECULAR {
        let specular_color = sample_texture(material_tex.specular_color, uv, vec4<f32>(1.0)).rgb;
        let specular_weight = sample_texture(material_tex.specular_weight, uv, vec4<f32>(1.0)).a;
        let ior = max(material.roughness_metallic.z, 1.0);
        let dielectric_f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
        return material.roughness_metallic.w * specular_weight * specular_color * dielectric_f0;
    }

    // Metallic workflow: F0 = mix(0.04, albedo, metallic)
    return clamp(
        mix(vec3<f32>(0.04), albedo, metallic),
        vec3<f32>(0.0),
        vec3<f32>(0.999),
    );
}

@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    let material = materials[input.material_id];
    let material_tex = material_textures[input.material_id];
    let uv = input.tex_coords;

    // DEBUG MODE 1: Show UVs as colors (R=U, G=V, helps verify UV layout)
    if globals.debug_mode == 1u {
        return GBufferOutput(
            vec4<f32>(uv.x, uv.y, 0.0, 1.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0)
        );
    }

    let tex_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));

    // DEBUG MODE 2: Show texture sample directly (bypass material multiply AND lighting)
    if globals.debug_mode == 2u {
        return GBufferOutput(
            vec4<f32>(tex_sample.rgb, 1.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
            vec4<f32>(0.0),
            vec4<f32>(0.0)
        );
    }

    // Common for modes 0 and 3
    let base_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));
    let albedo = material.base_color * base_sample;
    let alpha = albedo.a;

    if alpha <= 0.001 { discard; }
    if alpha < material_tex.params.z { discard; }  // alpha_cutoff in params.z

    let N_geom = normalize(input.world_normal);

    // DEBUG MODE 3: Use geometry normal only (skip normal mapping)
    var N: vec3<f32>;
    if globals.debug_mode == 3u {
        N = N_geom;
    } else {
        // NORMAL RENDERING (mode 0): vertex TBN normal mapping (MikkTSpace-compatible).
        //
        // Use the per-vertex tangent and bitangent sign that were computed by the
        // DCC tool (Maya / Blender / etc.) and stored in the FBX/glTF file.
        // Gram-Schmidt re-orthogonalization keeps T strictly perpendicular to N
        // after interpolation across the triangle, then B = cross(N, T) * sign.
        //
        // This replaces the previous screen-space derivative approach which
        // produced checkerboard artifacts at every UV island boundary because
        // dpdx/dpdy are undefined at triangle edges and discontinuous across seams.
        if material_tex.normal.texture_index != NO_TEXTURE {
            let T = normalize(input.world_tangent - dot(input.world_tangent, N_geom) * N_geom);
            let B = cross(N_geom, T) * input.bitangent_sign;
            var norm_ts = sample_texture(material_tex.normal, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - 1.0;
            norm_ts = vec3<f32>(norm_ts.x * material_tex.params.x, norm_ts.y * material_tex.params.x, norm_ts.z);  // normal_scale in params.x
            N = normalize(T * norm_ts.x + B * norm_ts.y + N_geom * norm_ts.z);
        } else {
            N = N_geom;
        }
    }

    let orm_sample = sample_texture(material_tex.roughness_metallic, uv, vec4<f32>(1.0));
    let occlusion_sample = sample_texture(material_tex.occlusion, uv, vec4<f32>(1.0));
    let emissive_sample = sample_texture(material_tex.emissive, uv, vec4<f32>(1.0));

    let ao = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;  // occlusion_strength in params.y
    let roughness = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    let metallic = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    let specular_f0 = resolve_specular_f0(material, material_tex, albedo.rgb, metallic, uv);

    let emissive = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    var out: GBufferOutput;
    out.albedo = vec4<f32>(albedo.rgb, alpha);
    out.normal = vec4<f32>(N, specular_f0.r);
    out.orm = vec4<f32>(ao, roughness, metallic, specular_f0.g);
    out.emissive = vec4<f32>(emissive, specular_f0.b);
    return out;
}
