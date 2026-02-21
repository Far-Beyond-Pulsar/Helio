// Base Geometry Shader

struct Camera {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    time: f32,
};
@group(0) @binding(0) var<uniform> camera: Camera;

struct Transform {
    model: mat4x4<f32>,
};
@group(1) @binding(0) var<uniform> transform: Transform;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords: vec2<f32>,
    @location(3) normal: u32,
    @location(4) tangent: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
};

// INJECT_VERTEXPREAMBLE

fn decode_normal(raw: u32) -> vec3<f32> {
    return unpack4x8snorm(raw).xyz;
}

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    // INJECT_VERTEXMAIN

    let world_pos = transform.model * vec4<f32>(vertex.position, 1.0);
    let world_normal = normalize((transform.model * vec4<f32>(decode_normal(vertex.normal), 0.0)).xyz);

    var output: VertexOutput;
    output.position = camera.view_proj * world_pos;
    output.world_position = world_pos.xyz;
    output.world_normal = world_normal;
    output.tex_coords = vertex.tex_coords;

    // INJECT_VERTEXPOSTPROCESS

    return output;
}

// INJECT_FRAGMENTPREAMBLE

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    var final_color = vec3<f32>(0.8);

    // INJECT_FRAGMENTMAIN

    // INJECT_FRAGMENTCOLORCALCULATION

    // INJECT_FRAGMENTPOSTPROCESS

    return vec4<f32>(final_color, 1.0);
}
