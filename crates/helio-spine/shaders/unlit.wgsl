// The lighting model's stand-in: every covered pixel of the G-buffer as
// its albedo plus its emission, over the background.

@group(0) @binding(0) var gbuf_albedo: texture_2d<f32>;
@group(0) @binding(1) var gbuf_emissive: texture_2d<f32>;
@group(0) @binding(2) var gbuf_depth: texture_depth_2d;

// The depth the frame clears to (reversed Z: the far plane is 0).
const DEPTH_CLEAR: f32 = 0.0;

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let pixel = vec2<i32>(position.xy);
    if textureLoad(gbuf_depth, pixel, 0) == DEPTH_CLEAR {
        discard;
    }
    let albedo = textureLoad(gbuf_albedo, pixel, 0).rgb;
    let emissive = textureLoad(gbuf_emissive, pixel, 0).rgb;
    return vec4<f32>(albedo + emissive, 1.0);
}
