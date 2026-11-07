// Reading an indirect-specular layer (`helio_frame_api::lighting::
// INDIRECT_SPECULAR_LAYERS`) at a lighting pixel, whatever the layer's
// resolution (SSR traces at half the internal resolution, planar reflections
// at full): the 2x2 bilinear reconstruction of its texels. A layer's rgb is
// straight (unassociated with its coverage a), so each tap is premultiplied
// before the filter: a hit beside an empty texel keeps its energy instead of
// fading twice.

// The layer at `uv` (0..1 over the lighting target): rgb premultiplied by
// coverage, a the coverage. Composite as `spec * (1 - a) + rgb * F`.
fn indirect_specular_at(layer: texture_2d<f32>, uv: vec2<f32>) -> vec4<f32> {
    let dims = textureDimensions(layer);
    let at = uv * vec2<f32>(dims) - vec2<f32>(0.5);
    let base = vec2<i32>(floor(at));
    let f = fract(at);
    let last = vec2<i32>(dims) - vec2<i32>(1);
    let h00 = textureLoad(layer, clamp(base, vec2<i32>(0), last), 0);
    let h10 = textureLoad(layer, clamp(base + vec2<i32>(1, 0), vec2<i32>(0), last), 0);
    let h01 = textureLoad(layer, clamp(base + vec2<i32>(0, 1), vec2<i32>(0), last), 0);
    let h11 = textureLoad(layer, clamp(base + vec2<i32>(1, 1), vec2<i32>(0), last), 0);
    let rgb = mix(
        mix(h00.rgb * h00.a, h10.rgb * h10.a, f.x),
        mix(h01.rgb * h01.a, h11.rgb * h11.a, f.x),
        f.y,
    );
    let a = mix(mix(h00.a, h10.a, f.x), mix(h01.a, h11.a, f.x), f.y);
    return vec4<f32>(max(rgb, vec3<f32>(0.0)), a);
}
