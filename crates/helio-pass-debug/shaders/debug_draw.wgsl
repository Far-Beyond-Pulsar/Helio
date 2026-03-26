struct DebugCamera {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> debug_camera: DebugCamera;
@group(0) @binding(1) var depth_texture: texture_depth_2d;
@group(0) @binding(2) var depth_sampler: sampler_comparison;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) depth: f32,
}

@vertex
fn vs_main(input: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip_position = debug_camera.view_proj * vec4<f32>(input.position, 1.0);
    out.color = input.color;
    let ndc = out.clip_position.xy / out.clip_position.w;
    out.uv = ndc * vec2<f32>(0.5, 0.5) + vec2<f32>(0.5, 0.5);
    out.depth = out.clip_position.z / out.clip_position.w;
    return out;
}

@fragment
fn fs_main(input: VertexOut) -> @location(0) vec4<f32> {
    return input.color;
}

@fragment
fn fs_main_depth(input: VertexOut) -> @location(0) vec4<f32> {
    let depth_hit = textureSampleCompare(depth_texture, depth_sampler, input.uv, input.depth);
    if depth_hit < 0.5 {
        discard;
    }
    return input.color;
}

