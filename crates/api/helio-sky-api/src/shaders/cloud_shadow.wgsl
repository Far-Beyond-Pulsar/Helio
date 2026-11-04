// The clouds' shadow (`clouds` module docs): orthographic maps along the
// sun round the camera, each texel what the clouds on its line towards the
// sun take from the light, as Hillaire's Beer shadow map: where the clouds
// begin along the light, their mean extinction, their whole optical depth,
// and the cirrus sheet's. Two cascades side by side in one texture: the
// near one (the billows marched) and the far one, out to the horizon (the
// weather's cover). Bind `cloud_shadow_frame` (a `CloudShadowFrame`
// uniform) and `cloud_shadow_map` (texture_2d<f32>, Rgba16Float).

struct CloudShadowFrame {
    // The maps' axes: right (w: 1 when the map is published, 0: no clouds),
    // up and the light's direction (from the sun; w: each cascade's side in
    // texels).
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    // The near and the far cascade: centre relative to the camera (km), w
    // half the extent (km). The far one is the texture's right half.
    near: vec4<f32>,
    far: vec4<f32>,
}

// One cascade's light at `d` (km, relative to the camera), its texels from
// column `column`: x the share let through, y how far inside the cascade
// `d` is (its uv's distance to the edge; not positive outside).
fn cloud_shadow_cascade(d: vec3<f32>, cascade: vec4<f32>, column: i32) -> vec2<f32> {
    let side = cloud_shadow_frame.forward.w;
    let rel = d - cascade.xyz;
    let uv = vec2<f32>(dot(rel, cloud_shadow_frame.right.xyz), -dot(rel, cloud_shadow_frame.up.xyz)) / cascade.w * 0.5 + 0.5;
    let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    if edge <= 0.0 { return vec2<f32>(1.0, edge); }
    let z = dot(rel, cloud_shadow_frame.forward.xyz);
    let at = uv * side - 0.5;
    let base = vec2<i32>(floor(at));
    let f = at - floor(at);
    var lit = vec4<f32>(1.0);
    for (var k = 0; k < 4; k++) {
        let q = clamp(base + vec2<i32>(k & 1, k >> 1), vec2<i32>(0), vec2<i32>(i32(side) - 1));
        let m = textureLoad(cloud_shadow_map, q + vec2<i32>(column, 0), 0);
        lit[k] = exp(-clamp((z - m.x) * m.y, 0.0, m.z) - m.w);
    }
    return vec2<f32>(mix(mix(lit.x, lit.y, f.x), mix(lit.z, lit.w, f.x), f.y), edge);
}

// The share of the sun the clouds let through to camera-relative point `p`
// (metres): 1 without clouds or outside the maps (the far one fades out at
// its edge, the near one into the far).
fn cloud_shadow(p: vec3<f32>) -> f32 {
    if cloud_shadow_frame.right.w < 0.5 { return 1.0; }
    let d = p * 0.001;
    let near = cloud_shadow_cascade(d, cloud_shadow_frame.near, 0);
    let inside = clamp(near.y * 16.0, 0.0, 1.0);
    if inside >= 1.0 { return near.x; }
    let far = cloud_shadow_cascade(d, cloud_shadow_frame.far, i32(cloud_shadow_frame.forward.w));
    return mix(mix(1.0, far.x, clamp(far.y * 16.0, 0.0, 1.0)), near.x, inside);
}
