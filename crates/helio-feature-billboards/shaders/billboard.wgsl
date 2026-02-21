// Billboard shader

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
    _pad: f32,
}
@group(0) @binding(0) var<uniform> camera: CameraUniforms;

struct BillboardInstance {
    world_position: vec3<f32>,
    _pad1: f32,
    scale: vec2<f32>,
    screen_scale: u32,
    _pad2: u32,
}
@group(1) @binding(0) var<uniform> billboard: BillboardInstance;

@group(2) @binding(0) var tex: texture_2d<f32>;
@group(2) @binding(1) var tex_sampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
) -> VertexOutput {
    var out: VertexOutput;

    let view_inv = transpose(mat3x3<f32>(
        camera.view_proj[0].xyz,
        camera.view_proj[1].xyz,
        camera.view_proj[2].xyz,
    ));
    let camera_right = normalize(view_inv[0]);
    let camera_up    = normalize(view_inv[1]);

    var effective_scale = billboard.scale;
    if billboard.screen_scale != 0u {
        let dist = length(camera.position - billboard.world_position);
        effective_scale = effective_scale * dist;
    }

    let world_pos = billboard.world_position
        + camera_right * position.x * effective_scale.x
        + camera_up    * position.y * effective_scale.y;

    out.position   = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.tex_coords = tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(tex, tex_sampler, in.tex_coords);
    if color.a < 0.01 { discard; }
    return color;
}
