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
// this row), its unit direction, `cone`, and the coordinate space it is in.
// A tracer traces from the origin itself, from distance 0: the consumer
// starts a ray off the surface it leaves.
//
// `cone` is the ray's cone (Akenine-Moller et al., "Texture Level of Detail
// Strategies for Real-Time Ray Tracing", Ray Tracing Gems): its footprint is
// `width` across at the origin and grows by `spread` a metre, so a tracer
// reads textures and tests cut-outs at the detail the ray can resolve
// (`scene_ray_footprint`). Bits 0..16: the spread, a half float. Bits
// 16..24: the caster mask the ray sees (`RAY_MASK_*`). Bits 24..32: the
// width, 0 for none, else `2^((bits - 160) / 8)` metres (a micrometre to
// kilometres in steps of 9%). A camera or probe ray has no width; a ray
// that leaves a hit starts as wide as the ray that found the hit was there.
//
// `space` is the authored coordinate space the ray travels in, encoded as a
// camera's (`Camera.view_info.w`): the space's slot plus one, or
// `SCENE_SPACE_ANY` (0) for a ray that sees every space, as an unrestricted
// camera does. A tracer answers only with geometry of that space
// (`scene_ray_sees_space`). A ray serving a view takes the view's
// (`cameras[0].view_info.w`); a ray leaving a hit keeps its parent's, or the
// hit's content space where a remap put the hit in another
// (`helio_frame_api::lighting::VIEW_REMAP_SPACES`). 48 bytes: 12 bytes past
// `space` are the row's alignment, and stay zero.
struct SceneRay {
    origin: vec3<f32>,
    t_max: f32,
    dir: vec3<f32>,
    cone: u32,
    space: u32,
}

// `SceneRay::space` of a ray that sees every coordinate space.
const SCENE_SPACE_ANY: u32 = 0u;

// `SceneRay::space` of a ray in authored space slot `slot` (0: the world).
fn scene_space_of_slot(slot: u32) -> u32 {
    return slot + 1u;
}

// Whether `ray` sees geometry authored in space slot `slot`
// (`helio_camera_sees_space` for rays).
fn scene_ray_sees_space(ray: SceneRay, slot: u32) -> bool {
    return ray.space == SCENE_SPACE_ANY || ray.space == slot + 1u;
}

// Casters that never move, casters that may, and every caster.
const RAY_MASK_STATIC: u32 = 1u;
const RAY_MASK_MOVABLE: u32 = 2u;
const RAY_MASK_ALL: u32 = 0xffu;

fn scene_ray_cone(width: f32, spread: f32, mask: u32) -> u32 {
    let w = select(0u, u32(clamp(round(log2(max(width, 1e-30)) * 8.0 + 160.0), 1.0, 255.0)), width > 0.0);
    return (pack2x16float(vec2<f32>(spread, 0.0)) & 0xffffu) | ((mask & 0xffu) << 16u) | (w << 24u);
}

fn scene_ray_spread(ray: SceneRay) -> f32 {
    return unpack2x16float(ray.cone & 0xffffu).x;
}

fn scene_ray_width(ray: SceneRay) -> f32 {
    let w = ray.cone >> 24u;
    return select(0.0, exp2((f32(w) - 160.0) / 8.0), w != 0u);
}

// How wide the ray's footprint is `t` along it.
fn scene_ray_footprint(ray: SceneRay, t: f32) -> f32 {
    return scene_ray_width(ray) + scene_ray_spread(ray) * t;
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

// The row of a nearest-hit stream for a hit `t` along its ray, found by
// `tracer`.
fn scene_nearest(tracer: u32, t: f32, inside: bool, payload: u32, b: u32, c: u32) -> SceneHit {
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

// The surface at a nearest hit is a `SceneSurface` row
// (`//!use frame_ray_surface`, with `//!use frame_surface`).

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
