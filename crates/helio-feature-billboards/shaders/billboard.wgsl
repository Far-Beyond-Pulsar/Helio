// Billboard vertex shader - creates camera-facing quads

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
}

struct BillboardInstance {
    world_position: vec3<f32>,
    scale: vec2<f32>,
}

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(1) @binding(0) var<uniform> billboard: BillboardInstance;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Extract camera basis vectors from view matrix
    let view_inv = transpose(mat3x3<f32>(
        camera.view_proj[0].xyz,
        camera.view_proj[1].xyz,
        camera.view_proj[2].xyz,
    ));
    
    let camera_right = view_inv[0];
    let camera_up = view_inv[1];
    
    // Build billboard quad facing camera
    let world_pos = billboard.world_position
        + camera_right * in.position.x * billboard.scale.x
        + camera_up * in.position.y * billboard.scale.y;
    
    out.position = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.tex_coords = in.tex_coords;
    
    return out;
}

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var tex_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(tex, tex_sampler, in.tex_coords);
    
    // Discard fully transparent pixels
    if (color.a < 0.01) {
        discard;
    }
    
    return color;
}
