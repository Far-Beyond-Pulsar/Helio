// The sun's occlusion by terrain that does not draw into the shadow maps
// (`sun_occlusion` module docs): a clipmap of orthographic depth maps along
// the sun, each texel the distance from its level's near plane to the first
// occluder (0: none). Bind `sun_occlusion_frame` (a `SunOcclusionFrame`)
// and `sun_occlusion_maps` (texture_2d_array<f32>, R32Float, a layer a
// level).

// A level: its near plane's centre (camera-relative; w its half extent, 0
// not drawn) and the axes it was drawn with: right, up, and forward (from
// the sun, unit).
struct SunOcclusionLevel {
    centre: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
}

struct SunOcclusionFrame {
    // x: the number of levels (0: no occlusion published), y: the side in
    // texels.
    info: vec4<f32>,
    levels: array<SunOcclusionLevel, 4>,
}

// The share of the sun reaching camera-relative point `p`: 0 behind
// terrain, else 1, from the finest level holding it (a 2x2 percentage
// filter); 1 outside every level.
fn sun_occlusion(p: vec3<f32>) -> f32 {
    return sun_occlusion_levels(p, 4u).x;
}

// `sun_occlusion` from the first `levels` levels only: the share, and 1 if
// one of them held the point (else 0, the share 1).
fn sun_occlusion_levels(p: vec3<f32>, levels: u32) -> vec2<f32> {
    let count = min(u32(sun_occlusion_frame.info.x), levels);
    let side = sun_occlusion_frame.info.y;
    for (var level = 0u; level < min(count, 4u); level++) {
        let l = sun_occlusion_frame.levels[level];
        let half = l.centre.w;
        if half <= 0.0 { continue; }
        let d = p - l.centre.xyz;
        let uv = vec2<f32>(dot(d, l.right.xyz), -dot(d, l.up.xyz)) / half * 0.5 + 0.5;
        if any(uv < vec2<f32>(0.02)) || any(uv > vec2<f32>(0.98)) { continue; }
        let t = dot(d, l.forward.xyz);
        // A texel and a half of slope bias: a texel spans 2 half / side.
        let bias = 3.0 * half / side;
        let at = uv * side - 0.5;
        let base = vec2<i32>(floor(at));
        let f = at - vec2<f32>(base);
        var lit = vec4<f32>(0.0);
        for (var k = 0; k < 4; k++) {
            let q = clamp(base + vec2<i32>(k & 1, k >> 1), vec2<i32>(0), vec2<i32>(i32(side) - 1));
            let stored = textureLoad(sun_occlusion_maps, q, i32(level), 0).r;
            lit[k] = select(0.0, 1.0, stored <= 0.0 || t <= stored + bias);
        }
        return vec2<f32>(mix(mix(lit.x, lit.y, f.x), mix(lit.z, lit.w, f.x), f.y), 1.0);
    }
    return vec2<f32>(1.0, 0.0);
}
