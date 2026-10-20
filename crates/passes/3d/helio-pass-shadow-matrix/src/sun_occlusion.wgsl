// The sun's occlusion by terrain that does not draw into the shadow maps
// (`sun_occlusion` module docs): a clipmap of orthographic depth maps along
// the sun, each texel the distance from its level's near plane to the first
// occluder (0: none). Bind `sun_occlusion_frame` (a `SunOcclusionFrame`)
// and `sun_occlusion_maps` (texture_2d_array<f32>, R32Float, a layer a
// level).

struct SunOcclusionFrame {
    // The levels' axes: right, up, and forward (from the sun, unit); w of
    // `right` the number of levels (0: no occlusion published), w of
    // `forward` the side in texels.
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    // Per level: its near plane's centre (camera-relative) and half extent.
    levels: array<vec4<f32>, 4>,
}

// The share of the sun reaching camera-relative point `p`: 0 behind
// terrain, else 1, from the finest level holding it (a 2x2 percentage
// filter); 1 outside every level.
fn sun_occlusion(p: vec3<f32>) -> f32 {
    let count = u32(sun_occlusion_frame.right.w);
    let side = sun_occlusion_frame.forward.w;
    for (var level = 0u; level < min(count, 4u); level++) {
        let l = sun_occlusion_frame.levels[level];
        let d = p - l.xyz;
        let uv = vec2<f32>(dot(d, sun_occlusion_frame.right.xyz), -dot(d, sun_occlusion_frame.up.xyz)) / l.w * 0.5 + 0.5;
        if any(uv < vec2<f32>(0.02)) || any(uv > vec2<f32>(0.98)) { continue; }
        let t = dot(d, sun_occlusion_frame.forward.xyz);
        // A texel and a half of slope bias: a texel spans 2 l.w / side.
        let bias = 3.0 * l.w / side;
        let at = uv * side - 0.5;
        let base = vec2<i32>(floor(at));
        let f = at - vec2<f32>(base);
        var lit = vec4<f32>(0.0);
        for (var k = 0; k < 4; k++) {
            let q = clamp(base + vec2<i32>(k & 1, k >> 1), vec2<i32>(0), vec2<i32>(i32(side) - 1));
            let stored = textureLoad(sun_occlusion_maps, q, i32(level), 0).r;
            lit[k] = select(0.0, 1.0, stored <= 0.0 || t <= stored + bias);
        }
        return mix(mix(lit.x, lit.y, f.x), mix(lit.z, lit.w, f.x), f.y);
    }
    return 1.0;
}
