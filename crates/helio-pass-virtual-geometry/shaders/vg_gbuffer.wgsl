// Virtual geometry G-buffer draw pass.
//
// Identical to gbuffer.wgsl but reads instance data from the VG-specific
// instance buffer.  The indirect draw's `first_instance` is the slot in the
// VG instance buffer written by the culling pass.
//
// This shader is compiled into `helio-pass-virtual-geometry` and draws only
// VG meshlets that survived the cull compute shader.

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
    _pad1:             u32,
    _pad2:             u32,
}

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
    params:             vec4<f32>,
}

struct GpuInstanceData {
    transform:    mat4x4<f32>,
    normal_mat_0: vec4<f32>,
    normal_mat_1: vec4<f32>,
    normal_mat_2: vec4<f32>,
    bounds:       vec4<f32>,
    mesh_id:      u32,
    material_id:  u32,
    flags:        u32,
    _pad:         u32,
}

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var<uniform>       globals:       Globals;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;

@group(1) @binding(0) var<storage, read> materials:          array<GpuMaterial>;
@group(1) @binding(1) var<storage, read> material_textures:  array<MaterialTextureData>;
@group(1) @binding(2) var                scene_textures:     binding_array<texture_2d<f32>, 256>;
@group(1) @binding(3) var                scene_samplers:     binding_array<sampler, 256>;

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
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(v.position, 1.0);

    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );
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

struct GBufferOutput {
    @location(0) albedo:   vec4<f32>,
    @location(1) normal:   vec4<f32>,
    @location(2) orm:      vec4<f32>,
    @location(3) emissive: vec4<f32>,
}

const NO_TEXTURE: u32 = 0xffffffffu;
const MATERIAL_WORKFLOW_METALLIC: u32 = 0u;
const MATERIAL_WORKFLOW_SPECULAR: u32 = 1u;

fn select_uv(slot: MaterialTextureSlot, base_uv: vec2<f32>) -> vec2<f32> {
    let scaled = base_uv * slot.offset_scale.zw;
    let s = slot.rotation.x;
    let c = slot.rotation.y;
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    );
    return rotated + slot.offset_scale.xy;
}

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
    return clamp(
        mix(vec3<f32>(0.04), albedo, metallic),
        vec3<f32>(0.0),
        vec3<f32>(0.999),
    );
}

@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    // ── Normal rendering ──────────────────────────────────────────────────────
    let material     = materials[input.material_id];
    let material_tex = material_textures[input.material_id];
    let uv = input.tex_coords;

    let base_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));
    let albedo      = material.base_color * base_sample;
    let alpha       = albedo.a;

    if alpha <= 0.001         { discard; }
    if alpha < material_tex.params.z { discard; }

    let N_geom = normalize(input.world_normal);
    var N: vec3<f32>;
    if material_tex.normal.texture_index != NO_TEXTURE {
        let T = normalize(input.world_tangent - dot(input.world_tangent, N_geom) * N_geom);
        let B = cross(N_geom, T) * input.bitangent_sign;
        var norm_ts = sample_texture(material_tex.normal, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - 1.0;
        norm_ts = vec3<f32>(norm_ts.x * material_tex.params.x, norm_ts.y * material_tex.params.x, norm_ts.z);
        N = normalize(T * norm_ts.x + B * norm_ts.y + N_geom * norm_ts.z);
    } else {
        N = N_geom;
    }

    let orm_sample      = sample_texture(material_tex.roughness_metallic, uv, vec4<f32>(1.0));
    let occlusion_sample = sample_texture(material_tex.occlusion, uv, vec4<f32>(1.0));
    let emissive_sample = sample_texture(material_tex.emissive, uv, vec4<f32>(1.0));

    let ao       = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;
    let roughness = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    let metallic  = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    let specular_f0 = resolve_specular_f0(material, material_tex, albedo.rgb, metallic, uv);
    let emissive  = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    var out: GBufferOutput;
    out.albedo  = vec4<f32>(albedo.rgb, alpha);
    out.normal  = vec4<f32>(N, specular_f0.r);
    out.orm     = vec4<f32>(ao, roughness, metallic, specular_f0.g);
    out.emissive = vec4<f32>(emissive, specular_f0.b);
    return out;
}

// ── VG triangle debug (mode 20): solid colour per triangle via primitive_index ─
// Lives in a separate entry point so the normal fs_main never pays the cost of
// primitive_index tracking even when debug is inactive.
@fragment
fn fs_debug(input: VertexOutput, @builtin(primitive_index) prim_id: u32) -> GBufferOutput {
    var h = prim_id * 2747636419u;
    h ^= h >> 16u;
    h *= 2654435769u;
    h ^= h >> 16u;
    let idx = h % 12u;

    var pal: array<vec3<f32>, 12>;
    pal[0]  = vec3<f32>(1.00, 0.18, 0.18);
    pal[1]  = vec3<f32>(1.00, 0.55, 0.00);
    pal[2]  = vec3<f32>(1.00, 0.90, 0.00);
    pal[3]  = vec3<f32>(0.35, 1.00, 0.10);
    pal[4]  = vec3<f32>(0.00, 0.90, 0.40);
    pal[5]  = vec3<f32>(0.00, 0.85, 1.00);
    pal[6]  = vec3<f32>(0.10, 0.40, 1.00);
    pal[7]  = vec3<f32>(0.55, 0.10, 1.00);
    pal[8]  = vec3<f32>(0.90, 0.10, 1.00);
    pal[9]  = vec3<f32>(1.00, 0.10, 0.60);
    pal[10] = vec3<f32>(0.00, 0.65, 0.65);
    pal[11] = vec3<f32>(1.00, 0.70, 0.10);

    let face_n = normalize(cross(dpdx(input.world_position), dpdy(input.world_position)));
    var out: GBufferOutput;
    out.albedo   = vec4<f32>(pal[idx], 1.0);
    out.normal   = vec4<f32>(face_n, 0.0);
    out.orm      = vec4<f32>(1.0, 0.9, 0.0, 0.0);
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return out;
}
