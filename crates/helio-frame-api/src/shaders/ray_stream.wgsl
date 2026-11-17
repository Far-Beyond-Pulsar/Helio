// A ray stream's rows (`helio_frame_api::rays::stream`,
// `//!use frame_ray_stream`): the rays a consumer writes, the hits the
// kinds' tracers answer with, and the surfaces at those hits. Positions and
// directions are in the frame's space (relative to the frame's origin, as the
// TLAS and the G-buffer are).

// Invocations of a tracer's workgroup: a stream's header holds the dispatch
// for workgroups this size (`RAY_STREAM_GROUP` in Rust).
const RAY_STREAM_GROUP: u32 = 64u;

// A stream's header: the indirect dispatch that covers its rays (x groups
// of `RAY_STREAM_GROUP`, then 1, 1), and how many rays it holds this frame.
struct RayStreamHeader {
    groups: vec3<u32>,
    count: u32,
}

// A ray: its origin and the most distance it travels (0 or less: no ray in
// this row), its unit direction, and `cone`: its spread (the width of its
// footprint a metre out, a half float in bits 0..16) and the caster mask it
// sees (bits 16..24, `RAY_MASK_*`).
struct SceneRay {
    origin: vec3<f32>,
    t_max: f32,
    dir: vec3<f32>,
    cone: u32,
}

// Casters that never move, casters that may, and every caster.
const RAY_MASK_STATIC: u32 = 1u;
const RAY_MASK_MOVABLE: u32 = 2u;
const RAY_MASK_ALL: u32 = 0xffu;

fn scene_ray_cone(spread: f32, mask: u32) -> u32 {
    return (pack2x16float(vec2<f32>(spread, 0.0)) & 0xffffu) | ((mask & 0xffu) << 16u);
}

fn scene_ray_spread(ray: SceneRay) -> f32 {
    return unpack2x16float(ray.cone & 0xffffu).x;
}

fn scene_ray_mask(ray: SceneRay) -> u32 {
    return (ray.cone >> 16u) & 0xffu;
}

// A hit, one a ray. Of a nearest-hit stream: the distance to the nearest
// surface any tracer found (0: none yet); in `a`, the tracer that found it
// (bits 28..32, 1-based), whether the ray met the surface from its inside
// (`SCENE_HIT_INSIDE`) and 27 bits of the tracer's own; `b` and `c` are the
// tracer's own (what it needs to find the surface again: a triangle and its
// barycentrics, a voxel and its face). Of an occlusion stream: `t` above 0
// once nothing gets through, and in `b` the share of light stopped so far
// (`scene_occlusion_through`).
struct SceneHit {
    t: f32,
    a: u32,
    b: u32,
    c: u32,
}

const SCENE_HIT_INSIDE: u32 = 0x08000000u;
const SCENE_HIT_PAYLOAD: u32 = 0x07ffffffu;

fn scene_hit(tracer: u32, t: f32, inside: bool, payload: u32, b: u32, c: u32) -> SceneHit {
    return SceneHit(t, (tracer << 28u) | select(0u, SCENE_HIT_INSIDE, inside) | (payload & SCENE_HIT_PAYLOAD), b, c);
}

fn scene_hit_tracer(hit: SceneHit) -> u32 {
    return hit.a >> 28u;
}

fn scene_hit_inside(hit: SceneHit) -> bool {
    return (hit.a & SCENE_HIT_INSIDE) != 0u;
}

fn scene_hit_payload(hit: SceneHit) -> u32 {
    return hit.a & SCENE_HIT_PAYLOAD;
}

// The distance a tracer still has to search along a ray whose row holds
// `hit`: up to the nearest hit so far, or the ray's whole length.
fn scene_hit_reach(ray: SceneRay, hit: SceneHit) -> f32 {
    return select(ray.t_max, min(hit.t, ray.t_max), hit.t > 0.0);
}

// What an occlusion stream's row lets through so far, per channel (a
// cleared row: everything).
fn scene_occlusion_through(hit: SceneHit) -> vec3<f32> {
    let stopped = vec3<f32>(f32(hit.b & 1023u), f32((hit.b >> 10u) & 1023u), f32((hit.b >> 20u) & 1023u)) / 1023.0;
    return 1.0 - stopped;
}

// The row of an occlusion stream that lets `through` pass: a tracer
// multiplies what the row let through by what its own geometry does and
// writes this.
fn scene_occlusion(through: vec3<f32>) -> SceneHit {
    let stopped = vec3<u32>(clamp(1.0 - through, vec3<f32>(0.0), vec3<f32>(1.0)) * 1023.0 + 0.5);
    let blocked = all(stopped == vec3<u32>(1023u));
    return SceneHit(select(0.0, 1.0, blocked), 0u, stopped.x | (stopped.y << 10u) | (stopped.z << 20u), 0u);
}

// The surface at a nearest hit, written by the tracer that found it: the
// shading normal and the face's (both on the side the ray met, packed by
// `scene_pack_unit`), base colour (RGB8) and metalness (alpha), `params`
// (roughness in bits 0..8, the share of light the surface lets through from
// its far side, tinted by its base colour, in bits 8..16: leaves), and the
// radiance it emits (`scene_pack_radiance`).
struct SceneSurface {
    normal: u32,
    face: u32,
    base_color: u32,
    params: u32,
    emissive: u32,
}

fn scene_surface(normal: vec3<f32>, face: vec3<f32>, base_color: vec3<f32>, metallic: f32, roughness: f32, transmission: f32, emissive: vec3<f32>) -> SceneSurface {
    return SceneSurface(
        scene_pack_unit(normal),
        scene_pack_unit(face),
        pack4x8unorm(vec4<f32>(base_color, metallic)),
        pack4x8unorm(vec4<f32>(roughness, transmission, 0.0, 0.0)),
        scene_pack_radiance(emissive),
    );
}

fn scene_surface_base_color(surface: SceneSurface) -> vec3<f32> {
    return unpack4x8unorm(surface.base_color).rgb;
}

fn scene_surface_metallic(surface: SceneSurface) -> f32 {
    return unpack4x8unorm(surface.base_color).a;
}

fn scene_surface_roughness(surface: SceneSurface) -> f32 {
    return unpack4x8unorm(surface.params).x;
}

fn scene_surface_transmission(surface: SceneSurface) -> f32 {
    return unpack4x8unorm(surface.params).y;
}

// A unit vector in 32 bits (octahedral, 16 bits an axis).
fn scene_pack_unit(n: vec3<f32>) -> u32 {
    var p = n.xy / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
    if n.z < 0.0 {
        p = (1.0 - abs(p.yx)) * select(vec2<f32>(-1.0), vec2<f32>(1.0), p >= vec2<f32>(0.0));
    }
    let q = vec2<u32>(clamp(p * 0.5 + 0.5, vec2<f32>(0.0), vec2<f32>(1.0)) * 65535.0 + 0.5);
    return q.x | (q.y << 16u);
}

fn scene_unpack_unit(p: u32) -> vec3<f32> {
    let e = vec2<f32>(f32(p & 0xffffu), f32(p >> 16u)) / 65535.0 * 2.0 - 1.0;
    var n = vec3<f32>(e, 1.0 - abs(e.x) - abs(e.y));
    let fold = max(-n.z, 0.0);
    n.x += select(fold, -fold, n.x >= 0.0);
    n.y += select(fold, -fold, n.y >= 0.0);
    return normalize(n);
}

// Linear radiance in 32 bits (RGB9E5: 9 bits a channel, a shared 5-bit
// exponent), up to 65408.
fn scene_pack_radiance(c: vec3<f32>) -> u32 {
    let v = clamp(c, vec3<f32>(0.0), vec3<f32>(65408.0));
    let most = max(v.r, max(v.g, v.b));
    var e = clamp(floor(log2(max(most, 1e-10))) + 1.0, -15.0, 16.0);
    // The largest channel rounding up to 512 takes the next exponent.
    if floor(most * exp2(9.0 - e) + 0.5) >= 512.0 {
        e += 1.0;
    }
    let q = vec3<u32>(floor(v * exp2(9.0 - e) + 0.5));
    return q.r | (q.g << 9u) | (q.b << 18u) | (u32(e + 15.0) << 27u);
}

fn scene_unpack_radiance(p: u32) -> vec3<f32> {
    let scale = exp2(f32(p >> 27u) - 24.0);
    return vec3<f32>(f32(p & 511u), f32((p >> 9u) & 511u), f32((p >> 18u) & 511u)) * scale;
}
