// Simple forward rendering shader

struct CameraData {
    view_proj: mat4x4<f32>,
}
var<uniform> camera_data: CameraData;

struct ModelData {
    transform: mat4x4<f32>,
    color: vec4<f32>,
}
var<uniform> model_data: ModelData;

struct Vertex {
    position: vec3<f32>,
    normal: vec3<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
}

@vertex
fn vs_main(in: Vertex) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = model_data.transform * vec4<f32>(in.position, 1.0);
    out.position = camera_data.view_proj * world_pos;
    
    let normal_matrix = mat3x3<f32>(
        model_data.transform[0].xyz,
        model_data.transform[1].xyz,
        model_data.transform[2].xyz
    );
    out.world_normal = normalize(normal_matrix * in.normal);
    out.color = model_data.color;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(-0.5, -1.0, -0.3));
    let n_dot_l = max(dot(in.world_normal, -light_dir), 0.0);
    
    let ambient = 0.3;
    let diffuse = n_dot_l * 0.7;
    let lighting = ambient + diffuse;
    
    return vec4<f32>(in.color.rgb * lighting, in.color.a);
}
