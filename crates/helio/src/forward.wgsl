struct Camera {
    view:          mat4x4<f32>,
    proj:          mat4x4<f32>,
    view_proj:     mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far:   vec4<f32>,
    jitter_frame:  vec4<f32>,
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

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> frame_uniforms: FrameUniforms;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;
@group(0) @binding(3) var<storage, read> materials: array<GpuMaterial>;
@group(0) @binding(4) var<storage, read> lights: array<GpuLight>;

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
    @location(2) material_id:    u32,
}

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
    out.material_id = inst.material_id;
    return out;
}

fn point_light_contrib(light: GpuLight, position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let to_light = light.position_range.xyz - position;
    let distance = max(length(to_light), 0.0001);
    if distance > light.position_range.w {
        return vec3<f32>(0.0);
    }
    let L = to_light / distance;
    let attenuation = pow(max(1.0 - distance / max(light.position_range.w, 0.0001), 0.0), 2.0);
    let ndotl = max(dot(normal, L), 0.0);
    return light.color_intensity.xyz * (light.color_intensity.w * attenuation * ndotl);
}

fn directional_light_contrib(light: GpuLight, normal: vec3<f32>) -> vec3<f32> {
    let L = normalize(-light.direction_outer.xyz);
    let ndotl = max(dot(normal, L), 0.0);
    return light.color_intensity.xyz * (light.color_intensity.w * ndotl);
}

fn spot_light_contrib(light: GpuLight, position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let to_light = light.position_range.xyz - position;
    let distance = max(length(to_light), 0.0001);
    if distance > light.position_range.w {
        return vec3<f32>(0.0);
    }
    let L = to_light / distance;
    let spot_dir = normalize(-light.direction_outer.xyz);
    let spot_cos = dot(spot_dir, L);
    if spot_cos < light.direction_outer.w {
        return vec3<f32>(0.0);
    }
    let cone = smoothstep(light.direction_outer.w, max(light.inner_angle, light.direction_outer.w), spot_cos);
    let attenuation = pow(max(1.0 - distance / max(light.position_range.w, 0.0001), 0.0), 2.0);
    let ndotl = max(dot(normal, L), 0.0);
    return light.color_intensity.xyz * (light.color_intensity.w * attenuation * cone * ndotl);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let material = materials[input.material_id];
    let normal = normalize(input.world_normal);

    var lit = material.base_color.rgb * frame_uniforms.ambient_color.rgb * frame_uniforms.ambient_color.w;
    for (var i = 0u; i < frame_uniforms.light_count; i = i + 1u) {
        let light = lights[i];
        if light.light_type == 0u {
            lit += material.base_color.rgb * directional_light_contrib(light, normal);
        } else if light.light_type == 1u {
            lit += material.base_color.rgb * point_light_contrib(light, input.world_position, normal);
        } else if light.light_type == 2u {
            lit += material.base_color.rgb * spot_light_contrib(light, input.world_position, normal);
        }
    }

    lit += material.emissive.rgb * material.emissive.w;
    let mapped = lit / (lit + vec3<f32>(1.0));
    return vec4<f32>(pow(mapped, vec3<f32>(1.0 / 2.2)), material.base_color.a);
}
