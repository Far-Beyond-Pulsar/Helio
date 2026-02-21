// Shadow depth pass shader

struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
}
@group(0) @binding(0) var<uniform> shadow_uniforms: ShadowUniforms;

struct ObjectUniforms {
    model: mat4x4<f32>,
}
@group(0) @binding(1) var<uniform> object_uniforms: ObjectUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords: vec2<f32>,
    @location(3) normal: u32,
    @location(4) tangent: u32,
) -> VertexOutput {
    var output: VertexOutput;
    let world_pos = object_uniforms.model * vec4<f32>(position, 1.0);
    output.position = shadow_uniforms.light_view_proj * world_pos;
    return output;
}
