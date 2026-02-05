struct Camera {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad: f32,
};
var<uniform> camera: Camera;

struct Transform {
    model: mat4x4<f32>,
};
var<uniform> transform: Transform;

struct Vertex {
    position: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

fn decode_normal(raw: u32) -> vec3<f32> {
    return unpack4x8snorm(raw).xyz;
}

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    let world_pos = transform.model * vec4<f32>(vertex.position, 1.0);
    let world_normal = normalize((transform.model * vec4<f32>(decode_normal(vertex.normal), 0.0)).xyz);
    
    var output: VertexOutput;
    output.position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Base geometry pass - NO LIGHTING
    // Output flat color based on world normal (for visual debugging)
    let color = input.world_normal * 0.5 + 0.5;  // Map [-1,1] to [0,1]
    return vec4<f32>(color, 1.0);
}
