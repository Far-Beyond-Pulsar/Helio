// Helio Rendering Engine - GBuffer Fill Shader
struct CameraData {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    projection: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    position: vec3<f32>,
    near_plane: f32,
    forward: vec3<f32>,
    far_plane: f32,
    right: vec3<f32>,
    fov_y: f32,
    up: vec3<f32>,
    aspect_ratio: f32,
    exposure: f32,
    aperture: f32,
    focus_distance: f32,
    frame_index: u32,
};

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) bitangent_sign: f32,
    @location(2) tex_coords: vec2<f32>,
    @location(3) normal_packed: u32,
    @location(4) tangent_packed: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tangent: vec3<f32>,
    @location(3) bitangent: vec3<f32>,
    @location(4) tex_coords: vec2<f32>,
    @location(5) prev_clip_position: vec4<f32>,
};

struct FragmentOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) material: vec4<f32>,
    @location(3) emissive: vec4<f32>,
    @location(4) velocity: vec2<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraData;
@group(1) @binding(0) var<uniform> model_matrix: mat4x4<f32>;

fn unpack_normal(packed: u32) -> vec3<f32> {
    let x = f32(packed & 0x3FFu) / 1023.0;
    let y = f32((packed >> 10u) & 0x3FFu) / 1023.0;
    let z = f32((packed >> 20u) & 0x3FFu) / 1023.0;
    return normalize(vec3<f32>(x, y, z) * 2.0 - 1.0);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    let world_pos = model_matrix * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    
    let normal = unpack_normal(in.normal_packed);
    let tangent_vec = unpack_normal(in.tangent_packed);
    
    let normal_mat = mat3x3<f32>(
        model_matrix[0].xyz,
        model_matrix[1].xyz,
        model_matrix[2].xyz
    );
    
    out.normal = normalize(normal_mat * normal);
    out.tangent = normalize(normal_mat * tangent_vec);
    out.bitangent = cross(out.normal, out.tangent) * in.bitangent_sign;
    out.tex_coords = in.tex_coords;
    
    out.prev_clip_position = camera.prev_view_proj * world_pos;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    out.albedo = vec4<f32>(0.8, 0.8, 0.8, 1.0);
    
    out.normal = vec4<f32>(in.normal * 0.5 + 0.5, 1.0);
    
    out.material = vec4<f32>(0.0, 0.5, 0.0, 1.0);
    
    out.emissive = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    
    let current_ndc = in.clip_position.xy / in.clip_position.w;
    let prev_ndc = in.prev_clip_position.xy / in.prev_clip_position.w;
    out.velocity = (current_ndc - prev_ndc) * 0.5;
    
    return out;
}
