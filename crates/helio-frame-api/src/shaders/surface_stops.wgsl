// The surface stops (`helio_frame_api::history::SURFACE_STOPS_SNIPPET`,
// `//!use frame_surface_stops`): whether two points lie on one surface, the
// one test the denoiser's weights, the pixels' gather's reuse and the
// radiance cache's reads (a reader on a cell's surface) take. Points and
// normals are in any one space a shader chooses (view or frame space:
// distances and angles are the same). Whether the frame's depth shows an
// occluder between two pixels is `frame_surface_seen`
// (`surface_stops_seen.wgsl`).

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
// (unit normal `np`; `footprint` how far off its plane it may be however
// close: a pixel's footprint, a cell's point's neighbourhood): the normal
// and plane stops, each falling linearly to 0 at its bound.
fn surface_stops_weight(p: vec3<f32>, np: vec3<f32>, q: vec3<f32>, nq: vec3<f32>, footprint: f32) -> f32 {
    let w_normal = saturate(1.0 - (1.0 - dot(np, nq)) / (2.0 * SURFACE_STOPS_NORMAL_SHARE));
    let plane = abs(dot(np, q - p));
    let w_plane = saturate(1.0 - plane / (SURFACE_STOPS_PLANE_SLOPE * distance(p, q) + footprint));
    return w_normal * w_plane;
}
