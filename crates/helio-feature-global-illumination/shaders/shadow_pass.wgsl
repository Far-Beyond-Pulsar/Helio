// Shadow depth pass shader for Global Illumination
// Renders geometry from light's POV to create shadow map

struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
}
var<uniform> shadow_uniforms: ShadowUniforms;

struct ObjectUniforms {
    model: mat4x4<f32>,
}
var<uniform> object_uniforms: ObjectUniforms;

struct Vertex {
    position: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    var output: VertexOutput;

    // Transform vertex to world space, then to light space
    let world_pos = object_uniforms.model * vec4<f32>(vertex.position, 1.0);
    output.position = shadow_uniforms.light_view_proj * world_pos;

    return output;
}

// No fragment shader needed - depth is written automatically
