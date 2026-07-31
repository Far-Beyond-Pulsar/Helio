//!use pbr_eval

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
    frame:             u32,
    delta_time:        f32,
    light_count:       u32,
    ambient_intensity: f32,
    ambient_color:     vec4<f32>,
    num_tiles_x:       u32,
    num_tiles_y:       u32,
    screen_width:      f32,
    screen_height:     f32,
}

const TILE_SIZE: u32 = 16u;
const MAX_LIGHTS_PER_TILE: u32 = 64u;

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

struct GpuInstanceData {
    transform:    mat4x4<f32>,
    normal_mat_0: vec4<f32>,
    normal_mat_1: vec4<f32>,
    normal_mat_2: vec4<f32>,
    bounds:       vec4<f32>,
    prev_model:   mat4x4<f32>,
    mesh_id:      u32,
    material_id:  u32,
    flags:        u32,
    lightmap_index: u32,
}

struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
    god_rays_enabled:  u32,
    god_rays_density:  f32,
    god_rays_weight:   f32,
    god_rays_decay:    f32,
    god_rays_exposure: f32,
    flare_enabled:      u32,
    flare_type:         u32,
    flare_intensity:    f32,
    flare_scale:        f32,
    flare_tint_r:       f32,
    flare_tint_g:       f32,
    flare_tint_b:       f32,
    ies_profile_index:    i32,
    light_function_index: i32,
    ies_angle_scale:      f32,
    ies_angle_offset:     f32,
}

const FLAG_HAS_NORMAL_MAP: u32 = 1u << 3u;
const NO_TEXTURE: u32 = 0xffffffffu;
const MATERIAL_WORKFLOW_METALLIC: u32 = 0u;
const MATERIAL_WORKFLOW_SPECULAR: u32 = 1u;

@group(0) @binding(0) var<uniform>          camera:            Camera;
@group(0) @binding(1) var<uniform>          globals:           Globals;
@group(0) @binding(2) var<storage, read>    instance_data:     array<GpuInstanceData>;
@group(0) @binding(3) var<storage, read>    compacted_indices: array<u32>;
@group(0) @binding(4) var<storage, read>    lights:            array<GpuLight>;
@group(0) @binding(5) var<storage, read>    tile_light_lists:  array<u32>;
@group(0) @binding(6) var<storage, read>    tile_light_counts: array<u32>;

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
    @invariant @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) tex_coords:     vec2<f32>,
    @location(3) world_tangent:  vec3<f32>,
    @location(4) bitangent_sign: f32,
    @location(5) @interpolate(flat) material_id: u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(v: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let inst       = instance_data[compacted_indices[slot]];
    let world_pos  = inst.transform * vec4<f32>(v.position, 1.0);

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

struct SurfaceData {
    albedo:              vec4<f32>,
    normal:              vec3<f32>,
    ao:                  f32,
    roughness:           f32,
    metallic:            f32,
    specular_f0:         vec3<f32>,
    emissive:            vec3<f32>,
    alpha:               f32,
    flags:               u32,
    subsurface_color:    vec3<f32>,
    subsurface_radius:   f32,
    roughness_aniso_x:   f32,
    roughness_aniso_y:   f32,
    aniso_rotation:      f32,
}

fn evaluate_surface(material: GpuMaterial, material_tex: MaterialTextureData, input: VertexOutput) -> SurfaceData {
    let uv = input.tex_coords;
    let base_sample = sample_texture(material_tex.base_color, uv, vec4<f32>(1.0));
    let albedo = material.base_color * base_sample;
    let alpha = albedo.a;

    let N_geom = normalize(input.world_normal);

    var N: vec3<f32>;
    if (material.flags & FLAG_HAS_NORMAL_MAP) != 0u && material_tex.normal.texture_index != NO_TEXTURE {
        let T = normalize(input.world_tangent - dot(input.world_tangent, N_geom) * N_geom);
        let B = cross(N_geom, T) * input.bitangent_sign;
        var norm_ts = sample_texture(material_tex.normal, uv, vec4<f32>(0.5, 0.5, 1.0, 1.0)).rgb * 2.0 - 1.0;
        norm_ts = vec3<f32>(norm_ts.x * material_tex.params.x, norm_ts.y * material_tex.params.x, norm_ts.z);
        N = normalize(T * norm_ts.x + B * norm_ts.y + N_geom * norm_ts.z);
    } else {
        N = N_geom;
    }

    let orm_sample = sample_texture(material_tex.roughness_metallic, uv, vec4<f32>(1.0));
    let occlusion_sample = sample_texture(material_tex.occlusion, uv, vec4<f32>(1.0));
    let emissive_sample = sample_texture(material_tex.emissive, uv, vec4<f32>(1.0));

    var ao: f32 = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;
    var roughness: f32 = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    var metallic: f32 = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    var specular_f0: vec3<f32> = resolve_specular_f0(material, material_tex, albedo.rgb, metallic, uv);
    var emissive: vec3<f32> = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    return SurfaceData(albedo, N, ao, roughness, metallic, specular_f0, emissive, alpha,
                       0u, vec3<f32>(0.0), 0.0, 0.0, 0.0, 0.0);
}

fn radiant_eval_surface(material: GpuMaterial, material_tex: MaterialTextureData, input: VertexOutput) -> SurfaceData {
    var s = evaluate_surface(material, material_tex, input);

    var albedo: vec4<f32> = s.albedo;
    var N: vec3<f32> = s.normal;
    var ao: f32 = s.ao;
    var roughness: f32 = s.roughness;
    var metallic: f32 = s.metallic;
    var specular_f0: vec3<f32> = s.specular_f0;
    var emissive: vec3<f32> = s.emissive;
    var alpha: f32 = s.alpha;
    var surface_flags: u32 = s.flags;
    var subsurface_color: vec3<f32> = s.subsurface_color;
    var subsurface_radius: f32 = s.subsurface_radius;
    var roughness_aniso_x: f32 = s.roughness_aniso_x;
    var roughness_aniso_y: f32 = s.roughness_aniso_y;
    var aniso_rotation: f32 = s.aniso_rotation;

    // RADIANT_OVERRIDE_SURFACE
    // RADIANT_OVERRIDE_END

    s.albedo = albedo;
    s.normal = N;
    s.ao = ao;
    s.roughness = roughness;
    s.metallic = metallic;
    s.specular_f0 = specular_f0;
    s.emissive = emissive;
    s.alpha = alpha;
    s.flags = surface_flags;
    s.subsurface_color = subsurface_color;
    s.subsurface_radius = subsurface_radius;
    s.roughness_aniso_x = roughness_aniso_x;
    s.roughness_aniso_y = roughness_aniso_y;
    s.aniso_rotation = aniso_rotation;

    return s;
}

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
    is_anisotropic: bool,
    T:         vec3<f32>,
    ax:        f32,
    ay:        f32,
    has_subsurface: bool,
    subsurface_color: vec3<f32>,
) -> vec3<f32> {
    var L:        vec3<f32>;
    var radiance: vec3<f32>;

    if light.light_type == 0u {
        L        = normalize(-light.direction_outer.xyz);
        radiance = light.color_intensity.xyz * light.color_intensity.w;
    } else {
        let to_light = light.position_range.xyz - world_pos;
        let dist     = length(to_light);
        if dist > light.position_range.w { return vec3<f32>(0.0); }
        L = to_light / dist;
        var atten = 1.0 / (dist * dist + 0.0001);
        let normalized_dist = dist / light.position_range.w;
        atten *= max(0.0, 1.0 - normalized_dist * normalized_dist * normalized_dist * normalized_dist);
        if light.light_type == 2u {
            let cos_a = dot(-L, light.direction_outer.xyz);
            atten    *= smoothstep(light.direction_outer.w, light.inner_angle, cos_a);
        }
        radiance = light.color_intensity.xyz * light.color_intensity.w * atten;
    }

    let NdL = max(dot(N, L), 0.0);
    if NdL == 0.0 { return vec3<f32>(0.0); }

    let H = normalize(V + L);
    let F = fresnel_schlick(max(dot(H, V), 0.0), F0);
    let kD = (1.0 - F) * (1.0 - metallic);
    let NdV = max(dot(N, V), 0.0);

    var D = distribution_ggx(N, H, roughness);
    var G = geometry_smith(N, V, L, roughness);

    let specular = D * G * F / (4.0 * NdV * NdL + 0.0001);
    let diffuse_term = kD * albedo / PI;

    return (diffuse_term + specular) * radiance * NdL;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let material = materials[input.material_id];
    let material_tex = material_textures[input.material_id];

    let surface = radiant_eval_surface(material, material_tex, input);

    if surface.alpha <= 0.001 { discard; }

    let V = normalize(camera.position_near.xyz - input.world_position);
    let N = surface.normal;
    let NdV = max(dot(N, V), 0.0);
    let albedo = surface.albedo.rgb;
    let alpha = surface.alpha;
    let roughness = surface.roughness;
    let metallic = surface.metallic;
    let F0 = surface.specular_f0;
    let emissive = surface.emissive;
    let ao = surface.ao;

    var Lo = vec3<f32>(0.0);

    let tile_x = u32(input.clip_position.x) / TILE_SIZE;
    let tile_y = u32(input.clip_position.y) / TILE_SIZE;
    let tile_idx = tile_y * globals.num_tiles_x + tile_x;
    let tile_light_count = tile_light_counts[tile_idx];

    for (var i = 0u; i < tile_light_count; i++) {
        let light_idx = tile_light_lists[tile_idx * MAX_LIGHTS_PER_TILE + i];
        let light = lights[light_idx];
        if light.light_type != 0u {
            let dist = length(light.position_range.xyz - input.world_position);
            if dist > light.position_range.w { continue; }
        }
        Lo += pbr_direct_light(
            light, input.world_position, N, V, F0, albedo,
            roughness, metallic, 1.0, false,
            vec3<f32>(0.0), 0.0, 0.0, false, vec3<f32>(0.0),
        );
    }

    let ambient = globals.ambient_color.rgb * globals.ambient_intensity * albedo * ao;
    let color = Lo + ambient + emissive;

    return vec4<f32>(color, alpha);
}
