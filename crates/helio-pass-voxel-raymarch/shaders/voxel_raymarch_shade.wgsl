// Fullscreen triangle shader that samples the ray march storage textures
// and writes to the color attachment.

@group(0) @binding(0) var color_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOut {
    // Fullscreen triangle: covers NDC [-1,1] in x/y, z=1, w=1
    let uv = vec2<f32>(
        f32((vi << 1u) & 2u),
        f32(vi & 2u),
    );
    return VertexOut(
        vec4<f32>(uv * 2.0 - 1.0, 1.0, 1.0),
        uv,
    );
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // The fullscreen-triangle uv has v=0 at the bottom of the screen (NDC
    // convention), but the compute pass wrote texel row 0 as the top of the
    // screen — flip v here or the image comes out upside down.
    let texel_uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    let col = textureLoad(color_tex, vec2<i32>(texel_uv * vec2<f32>(textureDimensions(color_tex))), 0);
    if col.a == 0.0 {
        discard;
    }
    return col;
}
