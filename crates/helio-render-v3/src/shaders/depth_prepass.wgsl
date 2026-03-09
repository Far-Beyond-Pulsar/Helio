// depth_prepass.wgsl — writes depth only, no fragment output.

struct Camera {
    view_proj:     mat4x4<f32>,
    position:      vec3<f32>,
    time:          f32,
    view_proj_inv: mat4x4<f32>,
};

struct InstanceData {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
    col3: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexIn {
    @location(0) position:  vec3<f32>,
    @location(1) _bitangent_sign: f32,
    @location(2) uv:        vec2<f32>,
    @location(3) _normal:   vec4<f32>,
    @location(4) _tangent:  vec4<f32>,
    // Instance data (mat4 split into 4 vec4s)
    @location(5) i0: vec4<f32>,
    @location(6) i1: vec4<f32>,
    @location(7) i2: vec4<f32>,
    @location(8) i3: vec4<f32>,
};

@vertex
fn vs_main(in: VertexIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(in.i0, in.i1, in.i2, in.i3);
    let world_pos = model * vec4<f32>(in.position, 1.0);
    return camera.view_proj * world_pos;
}
