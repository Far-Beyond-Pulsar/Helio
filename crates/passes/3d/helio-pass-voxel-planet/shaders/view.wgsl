// Camera, packed surface samples and pixel helpers shared by shading and GBuffer output.

struct Camera {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    position_near: vec4<f32>,
    forward_far: vec4<f32>,
    jitter_frame: vec4<f32>,
    prev_view_proj: mat4x4<f32>,
}

// Shaded terrain sample: distance, albedo+ao, octahedral normal, material/flags.
struct Surface {
    t: f32,
    albedo_ao: u32,
    normal: u32,
    flags: u32,
}

// Private camera with its eye at zero; the precise eye lives in `frame`.
// Orientation, projection and jitter match the shared scene camera, so depth
// and screen-space velocity can be published directly into its GBuffer.
@group(1) @binding(0) var<uniform> camera: Camera;

fn pixel_index(p: vec2<u32>) -> u32 {
    return p.x + p.y * u32(frame.screen.x);
}

fn pixel_ray(p: vec2<f32>) -> vec3<f32> {
    let uv = p / frame.screen.xy;
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    let world = camera.inv_view_proj * ndc;
    return normalize(world.xyz / world.w - camera.position_near.xyz);
}

fn oct_encode(n: vec3<f32>) -> u32 {
    var p = n.xy / (abs(n.x) + abs(n.y) + abs(n.z));
    if n.z < 0.0 {
        p = (1.0 - abs(p.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    }
    return pack2x16snorm(p);
}

fn oct_decode(v: u32) -> vec3<f32> {
    let p = unpack2x16snorm(v);
    var n = vec3<f32>(p, 1.0 - abs(p.x) - abs(p.y));
    if n.z < 0.0 {
        n = vec3<f32>((1.0 - abs(n.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), n.xy >= vec2<f32>(0.0)), n.z);
    }
    return normalize(n);
}

