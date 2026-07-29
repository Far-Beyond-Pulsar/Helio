// Transparent pass base shader — shared shader used by the transparent pass.
// Custom transparent templates override `radiant_eval_transparent`.
// Camera struct matches the gbuffer's layout so position_near is correct.

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
}

struct GpuInstanceData {
    transform:     mat4x4<f32>,
    normal_mat_0:  vec4<f32>,
    normal_mat_1:  vec4<f32>,
    normal_mat_2:  vec4<f32>,
    bounds:        vec4<f32>,
    mesh_id:       u32,
    material_id:   u32,
    flags:         u32,
    _pad:          u32,
}

@group(0) @binding(0) var<uniform>       camera:        Camera;
@group(0) @binding(1) var<uniform>       globals:       Globals;
@group(0) @binding(2) var<storage, read> instance_data: array<GpuInstanceData>;

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
    @location(3) @interpolate(flat) material_id: u32,
}

fn decode_snorm8x4(packed: u32) -> vec3<f32> {
    return unpack4x8snorm(packed).xyz;
}

@vertex
fn vs_main(vertex: Vertex, @builtin(instance_index) slot: u32) -> VertexOutput {
    let inst      = instance_data[slot];
    let world_pos = inst.transform * vec4<f32>(vertex.position, 1.0);
    let normal_mat = mat3x3<f32>(
        inst.normal_mat_0.xyz,
        inst.normal_mat_1.xyz,
        inst.normal_mat_2.xyz,
    );
    var out: VertexOutput;
    out.clip_position  = camera.view_proj * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal   = normalize(normal_mat * decode_snorm8x4(vertex.normal));
    out.tex_coords     = vertex.tex_coords;
    out.material_id    = inst.material_id;
    return out;
}

fn radiant_eval_transparent(material_id: u32,
                            world_pos: vec3<f32>,
                            world_normal: vec3<f32>,
                            tex_coords: vec2<f32>) -> vec4<f32> {
    let ambient = globals.ambient_color.rgb * globals.ambient_intensity;
    let normal_shade = world_normal * 0.5 + 0.5;
    let color = ambient + normal_shade * 0.4;
    return vec4<f32>(color, 0.5);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // RADIANT_OVERRIDE_TRANSPARENT
    // RADIANT_OVERRIDE_END

    return radiant_eval_transparent(
        input.material_id,
        input.world_position,
        input.world_normal,
        input.tex_coords,
    );
}