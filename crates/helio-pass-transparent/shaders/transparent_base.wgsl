enable wgpu_binding_array;

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
    material_class:     u32,
    class_params:       vec4<f32>,
}

struct MaterialTextureData {
    base_color:  MaterialTextureSlot,
    normal:      MaterialTextureSlot,
    roughness_metallic: MaterialTextureSlot,
    emissive:    MaterialTextureSlot,
    occlusion:   MaterialTextureSlot,
    specular_color:  MaterialTextureSlot,
    specular_weight: MaterialTextureSlot,
    params:      vec4<f32>,
}

struct MaterialTextureSlot {
    texture_index: u32,
    rotation:      vec2<f32>,
    offset_scale:  vec4<f32>,
}

struct GpuInstanceData {
    model:         mat4x4<f32>,
    normal_mat:    array<vec4<f32>, 3>,
    bounds:        vec4<f32>,
    prev_model:    mat4x4<f32>,
    mesh_id:       u32,
    material_id:   u32,
    flags:         u32,
    lightmap_index: u32,
}

// ── Bindings ────────────────────────────────────────────────────────
@group(0) @binding(0) var<uniform>          camera:        Camera;
@group(0) @binding(1) var<uniform>          globals:       Globals;
@group(0) @binding(2) var<storage, read>    instance_data: array<GpuInstanceData>;

@group(1) @binding(0) var<storage, read>    materials:         array<GpuMaterial>;
@group(1) @binding(1) var<storage, read>    material_textures: array<MaterialTextureData>;
@group(1) @binding(2) var                   scene_textures:    binding_array<texture_2d<f32>, 256>;
@group(1) @binding(3) var                   scene_samplers:    binding_array<sampler, 256>;

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
    @location(3) material_id:    u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(vertex: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let inst       = instance_data[slot];
    let world_pos  = inst.model * vec4<f32>(vertex.position, 1.0);
    let normal_mat = mat3x3<f32>(inst.normal_mat[0].xyz, inst.normal_mat[1].xyz, inst.normal_mat[2].xyz);
    var out: VertexOutput;
    out.clip_position  = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat * decode_snorm8x4(vertex.normal));
    out.tex_coords     = vertex.tex_coords;
    out.material_id    = inst.material_id;
    return out;
}

fn sample_texture(slot: MaterialTextureSlot, uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if slot.texture_index == 0xFFFFFFFFu { return fallback; }
    let scaled = uv * slot.offset_scale.zw;
    let c = slot.rotation.y;
    let s = slot.rotation.x;
    let rotated = vec2<f32>(scaled.x * c - scaled.y * s, scaled.x * s + scaled.y * c);
    return textureSample(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], rotated + slot.offset_scale.xy);
}

// Default transparent evaluation — simple ambient + normal shading
fn radiant_eval_transparent(material: GpuMaterial,
                            material_tex: MaterialTextureData,
                            input: VertexOutput) -> vec4<f32> {
    let uv = input.tex_coords;
    let base_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));
    let albedo = material.base_color * base_sample;
    let N = normalize(input.world_normal);
    let ambient = globals.ambient_color.rgb * globals.ambient_intensity;
    let normal_shade = N * 0.5 + 0.5;
    let color = albedo.rgb * (ambient + normal_shade * 0.3);
    return vec4<f32>(color, albedo.a * 0.5);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let material = materials[input.material_id];
    let material_tex = material_textures[input.material_id];

    // RADIANT_OVERRIDE_TRANSPARENT
    // RADIANT_OVERRIDE_END

    let result = radiant_eval_transparent(material, material_tex, input);
    return result;
}