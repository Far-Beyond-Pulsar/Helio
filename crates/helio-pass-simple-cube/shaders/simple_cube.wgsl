// Simple cube pass WGSL shader.
//
// Renders a single hardcoded cube with per-face colors at full brightness
// (no lighting calculation — the color IS the final lit value).

// Minimal camera view — only the fields we need (view_proj is at byte 128
// of the real CameraUniform, which is larger; reading a prefix is safe).
struct Camera {
    view:      mat4x4<f32>,
    proj:      mat4x4<f32>,
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) color:    vec3<f32>,
}

struct VertexOutput {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       color:    vec3<f32>,
}

@vertex
fn vs_main(v: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_pos = camera.view_proj * vec4(v.position, 1.0);
    out.color    = v.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4(in.color, 1.0);
}
