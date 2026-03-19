struct Camera {
    view:           mat4x4<f32>,
    proj:           mat4x4<f32>,
    view_proj:      mat4x4<f32>,
    inv_view_proj:  mat4x4<f32>,
    position_near:  vec4<f32>,
    forward_far:    vec4<f32>,
    jitter_frame:   vec4<f32>,
    prev_view_proj: mat4x4<f32>,
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

struct GpuLight {
    position_range:  vec4<f32>,
    direction_outer: vec4<f32>,
    color_intensity: vec4<f32>,
    shadow_index:    u32,
    light_type:      u32,
    inner_angle:     f32,
    _pad:            u32,
}

struct FrameUniforms {
    ambient_color: vec4<f32>,
    clear_color:   vec4<f32>,
    light_count:   u32,
    _pad0:         u32,
    _pad1:         u32,
    _pad2:         u32,
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

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> frame_uniforms: FrameUniforms;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;
@group(0) @binding(3) var<storage, read> materials: array<GpuMaterial>;
@group(0) @binding(4) var<storage, read> lights: array<GpuLight>;
@group(0) @binding(5) var<storage, read> material_textures: array<MaterialTextureData>;
@group(0) @binding(6) var scene_textures: binding_array<texture_2d<f32>, 256>;
@group(0) @binding(7) var scene_samplers: binding_array<sampler, 256>;

struct Vertex {
    @location(0) position:       vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords0:    vec2<f32>,
    @location(3) tex_coords1:    vec2<f32>,
    @location(4) normal:         u32,
    @location(5) tangent:        u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal:   vec3<f32>,
    @location(2) world_tangent:  vec3<f32>,
    @location(3) bitangent_sign: f32,
    @location(4) tex_coords0:    vec2<f32>,
    @location(5) tex_coords1:    vec2<f32>,
    @location(6) material_id:    u32,
}

const NO_TEXTURE: u32 = 0xffffffffu;
const WORKFLOW_METALLIC: u32 = 0u;
const WORKFLOW_SPECULAR: u32 = 1u;
const FLAG_ALPHA_BLEND: u32 = 1u << 1u;
const FLAG_ALPHA_TEST: u32 = 1u << 2u;

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(v: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let inst = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(v.position, 1.0);
    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );

    var out: VertexOutput;
    out.clip_position = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = normalize(normal_mat * decode_snorm8x4(v.normal));
    out.world_tangent = normalize(normal_mat * decode_snorm8x4(v.tangent));
    out.bitangent_sign = v.bitangent_sign;
    out.tex_coords0 = v.tex_coords0;
    out.tex_coords1 = v.tex_coords1;
    out.material_id = inst.material_id;
    return out;
}

fn select_uv(slot: MaterialTextureSlot, input: VertexOutput) -> vec2<f32> {
    let source_uv = select(input.tex_coords0, input.tex_coords1, slot.uv_channel != 0u);
    let scaled = source_uv * slot.offset_scale.zw;
    let s = slot.rotation.x;
    let c = slot.rotation.y;
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c,
    );
    return rotated + slot.offset_scale.xy;
}

fn sample_texture(slot: MaterialTextureSlot, input: VertexOutput, fallback: vec4<f32>) -> vec4<f32> {
    if slot.texture_index == NO_TEXTURE {
        return fallback;
    }
    let uv = select_uv(slot, input);
    return textureSample(scene_textures[slot.texture_index], scene_samplers[slot.texture_index], uv);
}

fn compute_base_reflectance(
    material: GpuMaterial,
    material_tex: MaterialTextureData,
    input: VertexOutput,
    base_color: vec3<f32>,
) -> vec3<f32> {
    if material.workflow == WORKFLOW_SPECULAR {
        let specular_color = sample_texture(material_tex.specular_color, input, vec4<f32>(1.0)).rgb;
        let specular_weight = sample_texture(material_tex.specular_weight, input, vec4<f32>(1.0)).a;
        let ior = max(material.roughness_metallic.z, 1.0);
        let dielectric_f0 = pow((ior - 1.0) / (ior + 1.0), 2.0);
        return material.roughness_metallic.w * specular_weight * specular_color * dielectric_f0;
    }
    let metallic = clamp(material.roughness_metallic.y * sample_texture(material_tex.roughness_metallic, input, vec4<f32>(1.0)).b, 0.0, 1.0);
    return mix(vec3<f32>(0.04), base_color, metallic);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(1.0 - cos_theta, 5.0);
}

fn distribution_ggx(n: vec3<f32>, h: vec3<f32>, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let n_dot_h = max(dot(n, h), 0.0);
    let n_dot_h2 = n_dot_h * n_dot_h;
    let denom = n_dot_h2 * (a2 - 1.0) + 1.0;
    return a2 / max(3.14159265 * denom * denom, 1.0e-4);
}

fn geometry_schlick_ggx(n_dot_v: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return n_dot_v / max(n_dot_v * (1.0 - k) + k, 1.0e-4);
}

fn geometry_smith(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let n_dot_v = max(dot(n, v), 0.0);
    let n_dot_l = max(dot(n, l), 0.0);
    return geometry_schlick_ggx(n_dot_v, roughness) * geometry_schlick_ggx(n_dot_l, roughness);
}

fn light_radiance(light: GpuLight, position: vec3<f32>) -> vec4<f32> {
    if light.light_type == 0u {
        return vec4<f32>(normalize(-light.direction_outer.xyz), light.color_intensity.w);
    }

    let to_light = light.position_range.xyz - position;
    let distance = max(length(to_light), 1.0e-4);
    if distance > light.position_range.w {
        return vec4<f32>(vec3<f32>(0.0), 0.0);
    }

    let direction = to_light / distance;
    var attenuation = pow(max(1.0 - distance / max(light.position_range.w, 1.0e-4), 0.0), 2.0);

    if light.light_type == 2u {
        let spot_dir = normalize(-light.direction_outer.xyz);
        let spot_cos = dot(spot_dir, direction);
        if spot_cos < light.direction_outer.w {
            return vec4<f32>(vec3<f32>(0.0), 0.0);
        }
        let cone = smoothstep(light.direction_outer.w, max(light.inner_angle, light.direction_outer.w), spot_cos);
        attenuation *= cone;
    }

    return vec4<f32>(direction, attenuation * light.color_intensity.w);
}

fn sample_normal_map(
    slot: MaterialTextureSlot,
    input: VertexOutput,
    world_normal: vec3<f32>,
    normal_scale: f32,
) -> vec3<f32> {
    if slot.texture_index == NO_TEXTURE {
        return world_normal;
    }

    let tangent = normalize(input.world_tangent);
    let bitangent = normalize(cross(world_normal, tangent)) * input.bitangent_sign;
    let tbn = mat3x3<f32>(tangent, bitangent, world_normal);
    var mapped = sample_texture(slot, input, vec4<f32>(0.5, 0.5, 1.0, 1.0)).xyz * 2.0 - 1.0;
    mapped.xy *= normal_scale;
    return normalize(tbn * mapped);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let material = materials[input.material_id];
    let material_tex = material_textures[input.material_id];

    let base_sample = sample_texture(material_tex.base_color, input, vec4<f32>(1.0));
    let orm_sample = sample_texture(material_tex.roughness_metallic, input, vec4<f32>(1.0));
    let emissive_sample = sample_texture(material_tex.emissive, input, vec4<f32>(1.0));
    let occlusion_sample = sample_texture(material_tex.occlusion, input, vec4<f32>(1.0));

    let base_color = material.base_color * base_sample;
    if (material.flags & FLAG_ALPHA_TEST) != 0u && base_color.a < material_tex.params.z {
        discard;
    }

    let metallic = clamp(material.roughness_metallic.y * orm_sample.b, 0.0, 1.0);
    let roughness = clamp(material.roughness_metallic.x * orm_sample.g, 0.045, 1.0);
    let occlusion = 1.0 + (occlusion_sample.r - 1.0) * material_tex.params.y;
    let emissive = material.emissive.rgb * material.emissive.w * emissive_sample.rgb;

    let view_dir = normalize(camera.position_near.xyz - input.world_position);
    let normal = sample_normal_map(
        material_tex.normal,
        input,
        normalize(input.world_normal),
        material_tex.params.x,
    );
    let f0 = compute_base_reflectance(material, material_tex, input, base_color.rgb);
    var diffuse_color = base_color.rgb * (1.0 - metallic);
    if material.workflow == WORKFLOW_SPECULAR {
        diffuse_color = base_color.rgb * (vec3<f32>(1.0) - clamp(f0, vec3<f32>(0.0), vec3<f32>(1.0)));
    }

    var lit = diffuse_color * frame_uniforms.ambient_color.rgb * frame_uniforms.ambient_color.w * occlusion;

    for (var i = 0u; i < frame_uniforms.light_count; i = i + 1u) {
        let light = lights[i];
        let light_state = light_radiance(light, input.world_position);
        let light_dir = light_state.xyz;
        let intensity = light_state.w;
        if intensity <= 0.0 {
            continue;
        }

        let half_vec = normalize(view_dir + light_dir);
        let n_dot_l = max(dot(normal, light_dir), 0.0);
        let n_dot_v = max(dot(normal, view_dir), 0.0);
        let h_dot_v = max(dot(half_vec, view_dir), 0.0);
        if n_dot_l <= 0.0 || n_dot_v <= 0.0 {
            continue;
        }

        let radiance = light.color_intensity.xyz * intensity;
        let fresnel = fresnel_schlick(h_dot_v, f0);
        let distribution = distribution_ggx(normal, half_vec, roughness);
        let geometry = geometry_smith(normal, view_dir, light_dir, roughness);
        let numerator = distribution * geometry * fresnel;
        let denominator = max(4.0 * n_dot_v * n_dot_l, 1.0e-4);
        let specular = numerator / denominator;
        let kd = vec3<f32>(1.0) - fresnel;
        lit += ((kd * diffuse_color / 3.14159265) + specular) * radiance * n_dot_l;
    }

    lit = lit * occlusion + emissive;
    let mapped = lit / (lit + vec3<f32>(1.0));
    let alpha = select(1.0, base_color.a, (material.flags & FLAG_ALPHA_BLEND) != 0u);
    return vec4<f32>(pow(mapped, vec3<f32>(1.0 / 2.2)), alpha);
}
