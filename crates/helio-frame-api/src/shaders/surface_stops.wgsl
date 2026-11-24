// The surface stops (`helio_frame_api::history::SURFACE_STOPS_SNIPPET`,
// `//!use frame_surface_stops`): whether two pixels' points lie on one
// surface and whether the frame's depth shows an occluder standing between
// them, the one test the denoiser's weights and the pixels' gather's reuse
// both take. Points and normals are in any one space a shader chooses (view
// or frame space: distances and angles are the same).
//
// A shader that walks between two pixels (`surface_stops_seen`) defines
// `fn surface_stops_point(q: vec2<i32>) -> vec4<f32>`: the point its depth
// holds at pixel `q` in that space, and in w a pixel's footprint there
// (0 or less: the sky, nothing).

// The normal stop: a sample's irradiance holds the hemisphere about the
// normal it was traced at; seen from a normal tilted by t, a share
// (1 - cos t) / 2 of the cosine-weighted hemisphere was never sampled. The
// weight falls to 0 where that share reaches 1/8 (t = 41 degrees).
const SURFACE_STOPS_NORMAL_SHARE: f32 = 0.125;
// The plane stop: the chord between two points of a surface whose normals
// differ by the normal stop's 41 degrees leaves the tangent plane by
// tan(41 / 2 degrees) = 0.378 of its length; a step (a silhouette, a ledge)
// by a whole length or more. The same bound says how far a surface between
// two such points may come out of their planes (`surface_stops_seen`).
const SURFACE_STOPS_PLANE_SLOPE: f32 = 0.378;

// How surely point `q` (unit normal `nq`) lies on the surface of point `p`
// (unit normal `np`, a pixel `footprint` wide there): the normal and plane
// stops, each falling linearly to 0 at its bound.
fn surface_stops_weight(p: vec3<f32>, np: vec3<f32>, q: vec3<f32>, nq: vec3<f32>, footprint: f32) -> f32 {
    let w_normal = saturate(1.0 - (1.0 - dot(np, nq)) / (2.0 * SURFACE_STOPS_NORMAL_SHARE));
    let plane = abs(dot(np, q - p));
    let w_plane = saturate(1.0 - plane / (SURFACE_STOPS_PLANE_SLOPE * distance(p, q) + footprint));
    return w_normal * w_plane;
}

// Whether the frame's depth shows nothing between pixels `a` and `b` (their
// points `xa`, `xb` and unit normals `na`, `nb`) that stands out of both
// receivers' surfaces: no pixel on the screen's segment between them holds a
// point in front of both tangent planes by more than the plane stop accepts
// of a neighbour (SURFACE_STOPS_PLANE_SLOPE of its distance and a pixel's
// footprint). Two points of one floor either side of a ledge, a thin wall's
// or a pillar's foot are close, on one plane and share a normal, so the
// surface weight passes them; what tells them apart is the occluder
// standing between, which hides from each the light the other receives.
// Symmetric in a and b; reads only the shader's depth
// (`surface_stops_point`).
fn surface_stops_seen(a: vec2<i32>, b: vec2<i32>, xa: vec3<f32>, na: vec3<f32>, xb: vec3<f32>, nb: vec3<f32>) -> bool {
    let d = b - a;
    let steps = max(abs(d.x), abs(d.y));
    for (var k = 1; k < steps; k++) {
        let q = a + vec2<i32>(round(vec2<f32>(d) * (f32(k) / f32(steps))));
        let o = surface_stops_point(q);
        // The sky is behind everything.
        if o.w <= 0.0 { continue; }
        let above_a = dot(na, o.xyz - xa) - SURFACE_STOPS_PLANE_SLOPE * distance(o.xyz, xa);
        let above_b = dot(nb, o.xyz - xb) - SURFACE_STOPS_PLANE_SLOPE * distance(o.xyz, xb);
        if min(above_a, above_b) > o.w { return false; }
    }
    return true;
}
