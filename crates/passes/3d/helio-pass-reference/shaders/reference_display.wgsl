// The reference's image on screen (`helio_pass_reference`): each pixel of
// the scene colour takes the mean of the paths of the image pixel it covers
// (the image may be smaller: the reference's resolution scale), the whole
// image or (`view.mode.x` 1) its indirect diffuse part. And the snapshot: the
// scene colour the real-time lighting wrote, at the image's pixels, for a
// host that compares the two.

struct DisplayView {
    // The image's width and height; the scene colour's.
    size: vec4<u32>,
    // x: 0 the image, 1 its indirect diffuse.
    mode: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> image: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> image_diffuse: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> view: DisplayView;

struct VsOut {
    @builtin(position) position: vec4<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    let p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -1.0), vec2<f32>(3.0, -1.0), vec2<f32>(-1.0, 3.0));
    return VsOut(vec4<f32>(p[i], 0.0, 1.0));
}

// The image pixel holding scene colour pixel `xy`.
fn image_pixel(xy: vec2<u32>) -> u32 {
    let at = min(xy * view.size.xy / max(view.size.zw, vec2<u32>(1u)), view.size.xy - 1u);
    return at.y * view.size.x + at.x;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let i = image_pixel(vec2<u32>(in.position.xy));
    let sum = select(image[i], image_diffuse[i], view.mode.x == 1u);
    return vec4<f32>(sum.rgb / max(sum.w, 1.0), 1.0);
}

@group(0) @binding(3) var scene_color: texture_2d<f32>;
@group(0) @binding(4) var<storage, read_write> snapshot: array<vec4<f32>>;

// The scene colour at the centre of each image pixel.
@compute @workgroup_size(64)
fn reference_snapshot(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= view.size.x * view.size.y { return; }
    let xy = vec2<u32>(id.x % view.size.x, id.x / view.size.x);
    let at = min((2u * xy + 1u) * view.size.zw / (2u * view.size.xy), view.size.zw - 1u);
    snapshot[id.x] = textureLoad(scene_color, vec2<i32>(at), 0);
}
