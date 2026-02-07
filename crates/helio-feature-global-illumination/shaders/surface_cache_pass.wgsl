// Surface Cache Pass
// Stores geometry and material information for GI calculations

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad: f32,
}

var<uniform> camera: CameraUniforms;

struct TransformUniforms {
    model: mat4x4<f32>,
}

var<uniform> transform: TransformUniforms;

struct MaterialUniforms {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    emissive_strength: f32,
    _pad: f32,
}

var<uniform> material: MaterialUniforms;

// Vertex input
struct Vertex {
    position: vec3<f32>,
    bitangent_sign: f32,
    tex_coords: vec2<f32>,
    normal: u32,
    tangent: u32,
}

// Unpack normal from u32
fn unpack_normal(packed: u32) -> vec3<f32> {
    let x = f32((packed >> 0u) & 0xFFu) / 255.0 * 2.0 - 1.0;
    let y = f32((packed >> 8u) & 0xFFu) / 255.0 * 2.0 - 1.0;
    let z = f32((packed >> 16u) & 0xFFu) / 255.0 * 2.0 - 1.0;
    return normalize(vec3<f32>(x, y, z));
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
}

struct FragmentOutput {
    @location(0) position: vec4<f32>,    // World position
    @location(1) normal: vec4<f32>,      // World normal
    @location(2) albedo: vec4<f32>,      // Albedo color
    @location(3) material_props: vec4<f32>, // Metallic, Roughness, Emissive, AO
}

@vertex
fn vs_main(vertex: Vertex) -> VertexOutput {
    var output: VertexOutput;

    let world_pos = transform.model * vec4<f32>(vertex.position, 1.0);
    output.world_position = world_pos.xyz;
    output.clip_position = camera.view_proj * world_pos;

    let normal = unpack_normal(vertex.normal);
    let normal_matrix = mat3x3<f32>(
        transform.model[0].xyz,
        transform.model[1].xyz,
        transform.model[2].xyz
    );
    output.world_normal = normalize(normal_matrix * normal);
    output.tex_coords = vertex.tex_coords;

    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> FragmentOutput {
    var output: FragmentOutput;

    // Store world position
    output.position = vec4<f32>(input.world_position, 1.0);

    // Store world normal
    output.normal = vec4<f32>(normalize(input.world_normal), 1.0);

    // Store albedo
    output.albedo = material.base_color;

    // Store material properties
    output.material_props = vec4<f32>(
        material.metallic,
        material.roughness,
        material.emissive_strength,
        1.0 // Ambient occlusion (placeholder)
    );

    return output;
}
