// Billboard shader - renders camera-facing quads with a texture.
// blade-graphics uses name-based binding: variable names must match
// the field names in the Rust ShaderData structs. No @group/@binding needed.

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    position: vec3<f32>,
}

struct BillboardInstance {
    world_position: vec3<f32>,
    scale: vec2<f32>,
    // When non-zero, scale is multiplied by camera distance so the billboard
    // keeps a constant apparent size on screen regardless of depth.
    screen_scale: u32,
}

var<uniform> camera: CameraUniforms;
var<uniform> billboard: BillboardInstance;
var tex: texture_2d<f32>;
var tex_sampler: sampler;

struct VertexInput {
    position: vec3<f32>,
    tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    // Extract camera right and up vectors from the view matrix.
    // The top-left 3x3 of view_proj is rotation*projection; we transpose
    // to get the inverse view rotation (camera right/up in world space).
    let view_inv = transpose(mat3x3<f32>(
        camera.view_proj[0].xyz,
        camera.view_proj[1].xyz,
        camera.view_proj[2].xyz,
    ));

    let camera_right = normalize(view_inv[0]);
    let camera_up    = normalize(view_inv[1]);

    // When screen_scale is set, multiply the world-space scale by the camera
    // distance so the billboard subtends the same angle (pixel size) at any depth.
    var effective_scale = billboard.scale;
    if billboard.screen_scale != 0u {
        let dist = length(camera.position - billboard.world_position);
        effective_scale = effective_scale * dist;
    }

    // Build a world-space position that faces the camera
    let world_pos = billboard.world_position
        + camera_right * in.position.x * effective_scale.x
        + camera_up    * in.position.y * effective_scale.y;

    out.position   = camera.view_proj * vec4<f32>(world_pos, 1.0);
    out.tex_coords = in.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = textureSample(tex, tex_sampler, in.tex_coords);
    if color.a < 0.01 {
        discard;
    }
    return color;
}
