// Direct light at a point (`helio_frame_api::rays::light_samples`,
// `//!use frame_light_samples`, with `//!use frame_ray_stream` for its
// packings): a consumer writes a batch of points (`LightQuery`), each kind
// of light that registered a sampler offers one sample of its lights at each
// point, and the row ends with one sample (`LightSample`) a point, chosen
// among the samplers' offers in proportion to what they bring. The consumer
// turns a sample into an occlusion ray toward the light and weights it by its
// own BSDF.
//
// The estimate of the direct light a point gets from every light:
//   bsdf(wo, light_sample_dir(s)) * cos(n, dir) * light_sample_radiance(s) * visibility
// where visibility is what its occlusion ray toward the light lets through
// (`light_sample_distance(s)` long). For multiple importance sampling
// against the consumer's own BSDF sample, `light_sample_pdf(s)` is the
// solid-angle density of the direction (the choices among lights and
// samplers included); 0 marks a delta light (a point, a spot, a
// directional), which a BSDF sample cannot hit: the light sample's weight is
// then 1.

// A point asking for direct light, in the frame's space. `width`: the
// footprint the point stands for (`scene_ray_footprint` of the ray that
// found it), the size of light a sampler may resolve it at. `normal`: the
// side it is lit from (`scene_pack_unit`; a sampler offers no light behind
// it unless the query is `LIGHT_QUERY_ANY_SIDE`: a point in a medium, a thin
// two-sided leaf). `random`: two uniform numbers for where on a light (16
// bits each, low first); `pick`: a uniform number for the choices among
// lights and samplers (`light_query_uniform`). `space`: bits 0..16 the
// coordinate space the point is in (`SceneRay::space`'s encoding: a sampler
// offers only lights that space sees, `light_query_space`), bit 31
// `LIGHT_QUERY_ANY_SIDE`. 32 bytes.
struct LightQuery {
    position: vec3<f32>,
    width: f32,
    normal: u32,
    random: u32,
    pick: u32,
    space: u32,
}

// `LightQuery::space` bit: the point is lit from every side.
const LIGHT_QUERY_ANY_SIDE: u32 = 0x80000000u;

// The query's coordinate space (`SceneRay::space`'s encoding).
fn light_query_space(q: LightQuery) -> u32 {
    return q.space & 0xffffu;
}

// Whether a light authored in space slot `slot` lights query `q`'s point.
fn light_query_sees_space(q: LightQuery, slot: u32) -> bool {
    let space = light_query_space(q);
    return space == SCENE_SPACE_ANY || space == slot + 1u;
}

// Whether a light toward unit `dir` is on a side query `q` is lit from.
fn light_query_faces(q: LightQuery, dir: vec3<f32>) -> bool {
    return (q.space & LIGHT_QUERY_ANY_SIDE) != 0u || dot(scene_unpack_unit(q.normal), dir) > 0.0;
}

// One light sample at a point: toward the light (`dir`, `scene_pack_unit`)
// and how far (`distance`; a directional light: `LIGHT_SAMPLE_FAR`), the
// incident radiance over the sample's probability (`radiance`, as the
// sampler that chose it offered it: read it with `light_sample_radiance`),
// the solid-angle density of `dir` within that sampler (`pdf`; 0: a delta
// light), the sum of the offers' weights (`weight`; 0: no sample) and the
// light (`light`: the sampler's number in bits 28..32, its own light id
// below; `LIGHT_SAMPLE_NONE`: none). 32 bytes.
struct LightSample {
    radiance: vec3<f32>,
    distance: f32,
    dir: u32,
    pdf: f32,
    weight: f32,
    light: u32,
}

const LIGHT_SAMPLE_NONE: u32 = 0xffffffffu;
// How far a directional light is.
const LIGHT_SAMPLE_FAR: f32 = 1e30;

fn light_luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn light_hash(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

// Uniform number `k` of query `q` in [0, 1) for a choice (`k`: a sampler's
// number, then its own choices), from its `pick`.
fn light_query_uniform(q: LightQuery, k: u32) -> f32 {
    let bits = select(q.pick, light_hash(q.pick ^ (k * 0x9e3779b9u)), k != 0u);
    return f32(bits >> 8u) / 16777216.0;
}

// The query's two uniform numbers for where on a light.
fn light_query_position_uniforms(q: LightQuery) -> vec2<f32> {
    return (vec2<f32>(f32(q.random & 0xffffu), f32(q.random >> 16u)) + 0.5) / 65536.0;
}

// A query for the point `position` (frame space) of footprint `width`, lit
// from the side of `normal` (`any_side`: from every side), with uniform
// numbers `random` and `pick`, in coordinate space `space`.
fn light_query_point(position: vec3<f32>, width: f32, normal: vec3<f32>, any_side: bool, random: vec2<f32>, pick: u32, space: u32) -> LightQuery {
    let r = vec2<u32>(clamp(random, vec2<f32>(0.0), vec2<f32>(1.0)) * 65535.0 + 0.5);
    return LightQuery(position, width, scene_pack_unit(normal), r.x | (r.y << 16u), pick, (space & 0xffffu) | select(0u, LIGHT_QUERY_ANY_SIDE, any_side));
}

// A sampler's offer: `radiance` the incident radiance over its probability
// within the sampler, toward unit `dir`, `distance` away, `pdf` its density
// (0: delta), `light` the sampler's own id of it.
fn light_sample_of(radiance: vec3<f32>, dir: vec3<f32>, distance: f32, pdf: f32, sampler: u32, light: u32) -> LightSample {
    return LightSample(radiance, distance, scene_pack_unit(dir), pdf, 0.0, (sampler << 28u) | (light & 0x0fffffffu));
}

// The row `held` with sampler `sampler`'s offer `offer` merged in, by
// weighted reservoir sampling over the samplers' offers (weight: the offer's
// luminance), with query `q`'s uniform for that sampler. Each sampler calls
// it once a point; the row stays one sample, chosen in proportion to what
// each sampler brings, and `light_sample_radiance` divides by that choice.
fn light_sample_offer(held: LightSample, offer: LightSample, q: LightQuery, sampler: u32) -> LightSample {
    let w = light_luminance(offer.radiance);
    if !(w > 0.0) { return held; }
    let sum = held.weight + w;
    var out = held;
    if light_query_uniform(q, sampler) * sum < w {
        out = offer;
    }
    out.weight = sum;
    return out;
}

// Whether the row holds a sample.
fn light_sample_found(s: LightSample) -> bool {
    return s.weight > 0.0;
}

fn light_sample_dir(s: LightSample) -> vec3<f32> {
    return scene_unpack_unit(s.dir);
}

fn light_sample_distance(s: LightSample) -> f32 {
    return s.distance;
}

// The incident radiance over the probability of the whole choice (the
// light within its sampler, then the sampler among the offers): what the
// estimate above multiplies.
fn light_sample_radiance(s: LightSample) -> vec3<f32> {
    let w = light_luminance(s.radiance);
    return select(vec3<f32>(0.0), s.radiance * (s.weight / max(w, 1e-30)), s.weight > 0.0 && w > 0.0);
}

// The solid-angle density of the sample's direction, the choices included
// (for multiple importance sampling); 0 for a delta light.
fn light_sample_pdf(s: LightSample) -> f32 {
    let w = light_luminance(s.radiance);
    return select(0.0, s.pdf * w / s.weight, s.weight > 0.0 && w > 0.0);
}
