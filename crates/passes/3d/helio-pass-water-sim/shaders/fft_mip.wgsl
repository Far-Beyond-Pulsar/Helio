// fft_mip.wgsl — box-filter one mip level of both ocean outputs from the level
// above. Filtering the slope texture's first and second moments is what gives
// the surface its roughness at distance.

@group(0) @binding(0) var src_disp: texture_2d<f32>;
@group(0) @binding(1) var src_slope: texture_2d<f32>;

struct Out {
    @location(0) disp: vec4<f32>,
    @location(1) slope: vec4<f32>,
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> Out {
    let q = vec2<i32>(pos.xy) * 2;
    var o: Out;
    o.disp = 0.25 * (textureLoad(src_disp, q, 0) + textureLoad(src_disp, q + vec2<i32>(1, 0), 0)
                   + textureLoad(src_disp, q + vec2<i32>(0, 1), 0) + textureLoad(src_disp, q + vec2<i32>(1, 1), 0));
    o.slope = 0.25 * (textureLoad(src_slope, q, 0) + textureLoad(src_slope, q + vec2<i32>(1, 0), 0)
                    + textureLoad(src_slope, q + vec2<i32>(0, 1), 0) + textureLoad(src_slope, q + vec2<i32>(1, 1), 0));
    return o;
}
