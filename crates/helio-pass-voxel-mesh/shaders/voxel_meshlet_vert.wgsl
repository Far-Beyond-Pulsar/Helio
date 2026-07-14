// Vertex shader for voxel meshlets.
// Reads position (+ material) from vertex buffer 0, pre-computed normal
// from vertex buffer 1 (written by the extract compute shader).

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

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) @interpolate(flat) material: u32,
    @location(1) world_pos: vec3<f32>,
    @location(2) world_normal: vec3<f32>,
}

struct VertexInput {
    @location(0) data: vec4<f32>,
    @location(1) normal: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = camera.view_proj * vec4(v.data.xyz, 1.0);
    out.material = u32(v.data.w);
    out.world_pos = v.data.xyz;
    out.world_normal = normalize(v.normal.xyz);
    return out;
}
