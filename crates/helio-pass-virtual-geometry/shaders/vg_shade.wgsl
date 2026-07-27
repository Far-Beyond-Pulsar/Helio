// Shading compute shader (Phase 5: Software Rasterization).
//
// Full-screen compute dispatch.  Each thread reads the visibility buffer,
// reconstructs the triangle from the meshlet data, interpolates vertex
// attributes with the instance transform, samples materials, and writes
// the GBuffer.

enable wgpu_binding_array;
enable wgpu_int16;

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

struct MeshletEntry {
    center:               vec3<f32>,
    radius:               f32,
    cone_apex:            vec3<f32>,
    cone_cutoff:          f32,
    cone_axis:            vec3<f32>,
    lod_error:            f32,
    packed_counts:        u32,
    meshlet_index_offset: u32,
    meshlet_vertex_offset: u32,
    parent_cluster_id:    u32,
}

struct InstanceData {
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

struct GpuMeshletVertex {
    position:       vec3<f32>,
    bitangent_sign: f32,
    tex_coords0:    vec2<f32>,
    tex_coords1:    vec2<f32>,
    normal:         u32,
    tangent:        u32,
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

@group(0) @binding(0)  var<storage, read>     visibility_depth:    array<u32>;
@group(0) @binding(1)  var<storage, read>     visibility_data:     array<u32>;
@group(0) @binding(2)  var<storage, read>     visibility_instance: array<u32>;
@group(0) @binding(3)  var<storage, read>     meshlets:            array<MeshletEntry>;
@group(0) @binding(4)  var<storage, read>     meshlet_vertices:    array<GpuMeshletVertex>;
@group(0) @binding(5)  var<storage, read>     meshlet_indices:     array<u16>;
@group(0) @binding(6)  var<uniform>           camera:              Camera;
@group(0) @binding(7)  var<uniform>           globals:             Globals;
@group(0) @binding(8)  var<storage, read>     instances:           array<InstanceData>;

@group(1) @binding(0)  var<storage, read>                 materials:         array<GpuMaterial>;
@group(1) @binding(1)  var<storage, read>                 material_textures: array<MaterialTextureData>;
@group(1) @binding(2)  var                                scene_textures:    binding_array<texture_2d<f32>, 256>;
@group(1) @binding(3)  var                                scene_samplers:    binding_array<sampler, 256>;

@group(2) @binding(0)  var albedo_tex:    texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(1)  var normal_tex:    texture_storage_2d<rgba16float, write>;
@group(2) @binding(2)  var orm_tex:       texture_storage_2d<rgba8unorm, write>;
@group(2) @binding(3)  var emissive_tex:  texture_storage_2d<rgba16float, write>;
@group(2) @binding(4)  var lightmap_uv_tex: texture_storage_2d<rgba16float, write>;
@group(2) @binding(5)  var sss_tex:       texture_storage_2d<rgba16float, write>;
@group(2) @binding(6)  var extra_tex:     texture_storage_2d<rgba16float, write>;

const VIS_VALID_BIT: u32 = 2147483648u;
const NO_TEXTURE: u32 = 0xffffffffu;
const MATERIAL_WORKFLOW_METALLIC: u32 = 0u;
const MATERIAL_WORKFLOW_SPECULAR: u32 = 1u;

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

fn edge_function(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

fn select_uv(slot: MaterialTextureSlot, base_uv: vec2<f32>) -> vec2<f32> {
    let scaled = base_uv * slot.offset_scale.zw;
    let s = slot.rotation.x;
    let c = slot.rotation.y;
    let rotated = vec2<f32>(scaled.x * c - scaled.y * s, scaled.x * s + scaled.y * c);
    return rotated + slot.offset_scale.xy;
}

fn sample_texture(slot: MaterialTextureSlot, base_uv: vec2<f32>, fallback: vec4<f32>) -> vec4<f32> {
    if slot.texture_index == NO_TEXTURE { return fallback; }
    let uv = select_uv(slot, base_uv);
    return textureSampleLevel(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], uv, 0.0);
}

fn resolve_specular_f0(
    material: GpuMaterial, material_tex: MaterialTextureData,
    albedo: vec3<f32>, metallic: f32, uv: vec2<f32>,
) -> vec3<f32> {
    if material.workflow == MATERIAL_WORKFLOW_SPECULAR {
        let specular_color = sample_texture(material_tex.specular_color, uv, vec4<f32>(1.0)).rgb;
        let specular_weight = sample_texture(material_tex.specular_weight, uv, vec4<f32>(1.0)).a;
        let ior = max(material.roughness_metallic.z, 1.0);
        let dielectric_f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
        return material.roughness_metallic.w * specular_weight * specular_color * dielectric_f0;
    }
    return clamp(mix(vec3<f32>(0.04), albedo, metallic), vec3<f32>(0.0), vec3<f32>(0.999));
}

fn hash_color(idx: u32) -> vec3<f32> {
    var h = idx * 2747636419u;
    h ^= h >> 16u;
    h *= 2654435769u;
    h ^= h >> 16u;
    let i = h % 12u;
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
    return pal[i];
}

@compute @workgroup_size(8, 8, 1)
fn cs_shade(@builtin(global_invocation_id) id: vec3<u32>) {
    let pixel_x = id.x;
    let pixel_y = id.y;

    let screen_w = u32(textureDimensions(albedo_tex).x);
    let screen_h = u32(textureDimensions(albedo_tex).y);
    if pixel_x >= screen_w || pixel_y >= screen_h { return; }

    let pxl = vec2<i32>(i32(pixel_x), i32(pixel_y));
    let pixel_idx = pixel_y * screen_w + pixel_x;

    let packed = visibility_data[pixel_idx];
    if (packed & VIS_VALID_BIT) == 0u {
        // No geometry — clear GBuffer at this pixel
        textureStore(albedo_tex,   pxl, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(normal_tex,   pxl, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(orm_tex,      pxl, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(emissive_tex, pxl, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(lightmap_uv_tex, pxl, vec4<f32>(0.0, 0.0, 0.0, 0.0));
        textureStore(sss_tex,      pxl, vec4<f32>(0.0));
        textureStore(extra_tex,    pxl, vec4<f32>(0.0));
        return;
    }

    let meshlet_id = packed & 0x3FFFFFu;
    let triangle_id = packed >> 22u;
    let instance_id = visibility_instance[pixel_idx];

    if meshlet_id >= arrayLength(&meshlets) { return; }
    if instance_id >= arrayLength(&instances) { return; }

    let meshlet = meshlets[meshlet_id];
    let inst = instances[instance_id];

    let vert_offset = meshlet.meshlet_vertex_offset;
    let idx_offset = meshlet.meshlet_index_offset;

    let idx_base = idx_offset + triangle_id * 3u;
    if idx_base + 2u >= arrayLength(&meshlet_indices) { return; }

    let i0 = u32(meshlet_indices[idx_base]);
    let i1 = u32(meshlet_indices[idx_base + 1u]);
    let i2 = u32(meshlet_indices[idx_base + 2u]);

    if vert_offset + max(i0, max(i1, i2)) >= arrayLength(&meshlet_vertices) { return; }

    let v0 = meshlet_vertices[vert_offset + i0];
    let v1 = meshlet_vertices[vert_offset + i1];
    let v2 = meshlet_vertices[vert_offset + i2];

    // Transform to clip space
    let clip0 = camera.view_proj * (inst.transform * vec4<f32>(v0.position, 1.0));
    let clip1 = camera.view_proj * (inst.transform * vec4<f32>(v1.position, 1.0));
    let clip2 = camera.view_proj * (inst.transform * vec4<f32>(v2.position, 1.0));

    if clip0.w <= 0.0 || clip1.w <= 0.0 || clip2.w <= 0.0 { return; }

    let ndc0 = clip0.xyz / clip0.w;
    let ndc1 = clip1.xyz / clip1.w;
    let ndc2 = clip2.xyz / clip2.w;

    let w_f = f32(screen_w);
    let h_f = f32(screen_h);

    let s0 = vec2<f32>((ndc0.x * 0.5 + 0.5) * w_f, (ndc0.y * -0.5 + 0.5) * h_f);
    let s1 = vec2<f32>((ndc1.x * 0.5 + 0.5) * w_f, (ndc1.y * -0.5 + 0.5) * h_f);
    let s2 = vec2<f32>((ndc2.x * 0.5 + 0.5) * w_f, (ndc2.y * -0.5 + 0.5) * h_f);

    let p = vec2<f32>(f32(pixel_x) + 0.5, f32(pixel_y) + 0.5);

    let ew0 = edge_function(s1, s2, p);
    let ew1 = edge_function(s2, s0, p);
    let ew2 = edge_function(s0, s1, p);

    let area = ew0 + ew1 + ew2;
    if area <= 0.0 { return; }

    let bary_u = ew0 / area;
    let bary_v = ew1 / area;
    let bary_w = ew2 / area;

    // Perspective-correct interpolation for world position
    let w0_inv = 1.0 / clip0.w;
    let w1_inv = 1.0 / clip1.w;
    let w2_inv = 1.0 / clip2.w;

    let wp_inv = bary_u * w0_inv + bary_v * w1_inv + bary_w * w2_inv;
    if wp_inv <= 0.0 { return; }
    let wp = 1.0 / wp_inv;

    // World position
    let world_pos0 = (inst.transform * vec4<f32>(v0.position, 1.0)).xyz;
    let world_pos1 = (inst.transform * vec4<f32>(v1.position, 1.0)).xyz;
    let world_pos2 = (inst.transform * vec4<f32>(v2.position, 1.0)).xyz;

    let world_pos = (world_pos0 * bary_u * w0_inv + world_pos1 * bary_v * w1_inv + world_pos2 * bary_w * w2_inv) * wp;

    // Normals
    let normal_mat = mat3x3<f32>(inst.normal_mat_0.xyz, inst.normal_mat_1.xyz, inst.normal_mat_2.xyz);
    let model_mat3 = mat3x3<f32>(inst.transform[0].xyz, inst.transform[1].xyz, inst.transform[2].xyz);

    let n0 = normalize(normal_mat * decode_snorm8x4(v0.normal));
    let n1 = normalize(normal_mat * decode_snorm8x4(v1.normal));
    let n2 = normalize(normal_mat * decode_snorm8x4(v2.normal));
    let world_normal = normalize((n0 * bary_u * w0_inv + n1 * bary_v * w1_inv + n2 * bary_w * w2_inv) * wp);

    // Tangents
    let t0 = normalize(model_mat3 * decode_snorm8x4(v0.tangent));
    let t1 = normalize(model_mat3 * decode_snorm8x4(v1.tangent));
    let t2 = normalize(model_mat3 * decode_snorm8x4(v2.tangent));
    let world_tangent = normalize((t0 * bary_u * w0_inv + t1 * bary_v * w1_inv + t2 * bary_w * w2_inv) * wp);

    // Tex coords
    let uv = v0.tex_coords0 * bary_u + v1.tex_coords0 * bary_v + v2.tex_coords0 * bary_w;

    // Material
    let material_id = inst.material_id;
    let material = materials[material_id];
    let material_tex = material_textures[material_id];

    let base_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));
    let albedo = material.base_color * base_sample;
    let alpha = albedo.a;

    if alpha <= 0.001 { return; }

    let N_geom = world_normal;
    var N: vec3<f32>;
    if material_tex.normal.texture_index != NO_TEXTURE {
        let T = normalize(world_tangent - dot(world_tangent, N_geom) * N_geom);
        let B = cross(N_geom, T) * v0.bitangent_sign;
        var norm_ts = sample_texture(material_tex.normal, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - 1.0;
        norm_ts = vec3<f32>(norm_ts.x * material_tex.params.x, norm_ts.y * material_tex.params.x, norm_ts.z);
        N = normalize(T * norm_ts.x + B * norm_ts.y + N_geom * norm_ts.z);
    } else {
        N = N_geom;
    }

    let orm_sample       = sample_texture(material_tex.roughness_metallic, uv, vec4<f32>(1.0));
    let occlusion_sample = sample_texture(material_tex.occlusion, uv, vec4<f32>(1.0));
    let emissive_sample  = sample_texture(material_tex.emissive, uv, vec4<f32>(1.0));

    let ao        = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;
    let roughness = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    let metallic  = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    let specular_f0 = resolve_specular_f0(material, material_tex, albedo.rgb, metallic, uv);
    let emissive  = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    textureStore(albedo_tex,   pxl, vec4<f32>(albedo.rgb, alpha));
    textureStore(normal_tex,   pxl, vec4<f32>(N, specular_f0.r));
    textureStore(orm_tex,      pxl, vec4<f32>(ao, roughness, metallic, specular_f0.g));
    textureStore(emissive_tex, pxl, vec4<f32>(emissive, specular_f0.b));
    textureStore(lightmap_uv_tex, pxl, vec4<f32>(-2.0, -2.0, 0.0, 0.0));
    textureStore(sss_tex,      pxl, vec4<f32>(0.0));
    textureStore(extra_tex,    pxl, vec4<f32>(0.0));
}
