// The surface stops' depth walk
// (`helio_frame_api::history::SURFACE_SEEN_SNIPPET`, `//!use
// frame_surface_seen`, with `frame_surface_stops`): whether the frame's depth
// shows an occluder standing between two pixels. A shader that opts in
// defines `fn surface_stops_point(q: vec2<i32>) -> vec4<f32>`: the point its
// depth holds at pixel `q` in the surface stops' space, and in w a pixel's
// footprint there (0 or less: the sky, nothing).

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
