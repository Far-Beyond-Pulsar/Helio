// Shadow depth pass shader
// Renders geometry from light's POV to create shadow map

// GPU representation of a light
struct GpuLight {
    light_type: u32,
    intensity: f32,
    radius: f32,
    _padding1: f32,
    
    position: vec3<f32>,
    inner_angle: f32,
    
    direction: vec3<f32>,
    outer_angle: f32,
    
    color: vec3<f32>,
    width: f32,
    
    light_view_proj: mat4x4<f32>,
    
    height: f32,
    _padding2: vec3<f32>,
}

struct ShadowUniforms {
    light_count: u32,
    _padding: vec3<u32>,
    lights: array<GpuLight, 8>,
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
    // Use first light's view projection (shadow pass renders one light at a time)
    let world_pos = object_uniforms.model * vec4<f32>(vertex.position, 1.0);
    output.position = shadow_uniforms.lights[0].light_view_proj * world_pos;

    return output;
}

// No fragment shader needed - depth is written automatically
