// The clouds' shadow (`clouds` module docs): an orthographic map along the
// sun around the camera, each texel what the clouds on its line towards the
// sun take from the light, as Hillaire's Beer shadow map: where the clouds
// begin along the light, their mean extinction, and their whole optical
// depth. Bind `cloud_shadow_frame` (a `CloudShadowFrame` uniform) and
// `cloud_shadow_map` (texture_2d<f32>, Rgba16Float).

struct CloudShadowFrame {
    // The map's axes: right (w: 1 when the map is published, 0: no clouds),
    // up (w: half the map's extent, km) and the light's direction (from the
    // sun; w: the map's side in texels).
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    // The map's centre relative to the camera (km); w unused.
    centre: vec4<f32>,
}

// The share of the sun the clouds let through to camera-relative point `p`
// (metres): 1 without clouds or outside the map (it fades out at its edge).
fn cloud_shadow(p: vec3<f32>) -> f32 {
    if cloud_shadow_frame.right.w < 0.5 { return 1.0; }
    let half = cloud_shadow_frame.up.w;
    let side = cloud_shadow_frame.forward.w;
    let d = p * 0.001 - cloud_shadow_frame.centre.xyz;
    let uv = vec2<f32>(dot(d, cloud_shadow_frame.right.xyz), -dot(d, cloud_shadow_frame.up.xyz)) / half * 0.5 + 0.5;
    let edge = min(min(uv.x, uv.y), min(1.0 - uv.x, 1.0 - uv.y));
    if edge <= 0.0 { return 1.0; }
    let z = dot(d, cloud_shadow_frame.forward.xyz);
    let at = uv * side - 0.5;
    let base = vec2<i32>(floor(at));
    let f = at - floor(at);
    var lit = vec4<f32>(1.0);
    for (var k = 0; k < 4; k++) {
        let q = clamp(base + vec2<i32>(k & 1, k >> 1), vec2<i32>(0), vec2<i32>(i32(side) - 1));
        let m = textureLoad(cloud_shadow_map, q, 0);
        lit[k] = exp(-clamp((z - m.x) * m.y, 0.0, m.z));
    }
    let shadow = mix(mix(lit.x, lit.y, f.x), mix(lit.z, lit.w, f.x), f.y);
    return mix(1.0, shadow, clamp(edge * 16.0, 0.0, 1.0));
}
