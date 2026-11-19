//!use frame_ray_stream
//!use frame_ray_surface
//!use frame_surface
//!use frame_bsdf
//!use frame_light_samples
//!use frame_sky
//!use sky_table
// The reference path tracer (`helio_pass_reference`): a wavefront path
// tracer over the frame's ray streams and light samples, knowing no kind.
// A frame traces one path for each pixel of a window of the image (the
// paths budget), stage by stage: camera rays from the eye (one run), then
// at every depth the nearest hits and their surfaces (the kinds' tracers),
// the hit's emission (or the sky on a miss), one light sample (the frame's
// light samplers) with its shadow ray, multiple importance sampling of that
// sample against the BSDF, and the BSDF's sample for the next ray, live
// paths compacted into the next stage's stream. No cache, no denoiser, no
// clamp: each pixel's paths are summed in 32-bit floats over frames.

// The frame (`GpuFrame` in Rust, 144 bytes).
struct ReferenceFrame {
    // Clip to frame space, without the camera's jitter.
    clip_to_frame: mat4x4<f32>,
    // The eye (frame space); w: a pixel's cone spread a metre out.
    eye: vec4<f32>,
    // The image's width, height and pixels; the first pixel of this frame's
    // window.
    size: vec4<u32>,
    // Paths this frame; the frame's seed; the rays' coordinate space
    // (`SceneRay::space`); the last depth (stages - 1).
    info: vec4<u32>,
    // Without an atmosphere: the sky's radiance above the horizon (rgb) and
    // below it. w: the depth Russian roulette starts at (sky.w).
    sky: vec4<f32>,
    ground: vec4<f32>,
}

// A path (48 bytes): what reaches the eye from here on is multiplied by
// `throughput`; `radiance` what reached it so far; `diffuse` the part that
// came by a first bounce off a diffuse lobe (`REFERENCE_DIFFUSE_FIRST`);
// `pdf` the density of its last BSDF sample.
struct Path {
    throughput: vec3<f32>,
    pixel: u32,
    radiance: vec3<f32>,
    flags: u32,
    diffuse: vec3<f32>,
    pdf: f32,
}

// A light query's path: the throughput at its hit and the stream row of
// the hit (32 bytes: scalar padding, a `vec3` would align to 16 and make
// it 48).
struct Owner {
    throughput: vec3<f32>,
    path: u32,
    row: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const REFERENCE_DIFFUSE_FIRST: u32 = 1u;
// How far a ray goes: past the far side of an Earth-sized planet.
const REFERENCE_REACH: f32 = 1e9;

@group(0) @binding(0) var<uniform> frame: ReferenceFrame;
@group(0) @binding(1) var<storage, read_write> paths: array<Path>;
@group(0) @binding(2) var<storage, read_write> cur_header: array<atomic<u32>, 4>;
@group(0) @binding(3) var<storage, read_write> cur_rays: array<SceneRay>;
@group(0) @binding(4) var<storage, read_write> cur_hits: array<SceneHit>;
@group(0) @binding(5) var<storage, read> cur_surfaces: array<SceneSurface>;
@group(0) @binding(6) var<storage, read_write> cur_rows: array<u32>;
@group(0) @binding(7) var<storage, read_write> next_header: array<atomic<u32>, 4>;
@group(0) @binding(8) var<storage, read_write> next_rays: array<SceneRay>;
@group(0) @binding(9) var<storage, read_write> next_hits: array<SceneHit>;
@group(0) @binding(10) var<storage, read_write> next_rows: array<u32>;
@group(0) @binding(11) var<storage, read_write> batch_header: array<atomic<u32>, 4>;
@group(0) @binding(12) var<storage, read_write> queries: array<LightQuery>;
@group(0) @binding(13) var<storage, read> samples: array<LightSample>;
@group(0) @binding(14) var<storage, read_write> owners: array<Owner>;
@group(0) @binding(15) var<storage, read_write> shadow_header: array<atomic<u32>, 4>;
@group(0) @binding(16) var<storage, read_write> shadow_rays: array<SceneRay>;
@group(0) @binding(17) var<storage, read_write> shadow_hits: array<SceneHit>;
// A shadow ray's light (rgb, as bits) and its path: the path is not
// stored as a float's bits, which a denormal flush would zero.
@group(0) @binding(18) var<storage, read_write> pending: array<vec4<u32>>;
@group(0) @binding(19) var<storage, read_write> image: array<vec4<f32>>;
@group(0) @binding(20) var<storage, read_write> image_diffuse: array<vec4<f32>>;
@group(0) @binding(21) var<uniform> sky_frame: SkyFrame;
@group(0) @binding(22) var<uniform> sky_table: array<vec4<f32>, 2304>;
@group(0) @binding(23) var<storage, read_write> stats: array<atomic<u32>>;
// x: this stage's depth.
@group(0) @binding(24) var<uniform> stage: vec4<u32>;

override REFERENCE_STAT: u32 = 0u;

// A workgroup's rows of a compacted stream, taken with one global atomic:
// its lanes count into these, one lane reserves the block.
var<workgroup> w_count: array<atomic<u32>, 2>;
var<workgroup> w_base: array<u32, 2>;

fn hash(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

// Uniform number `dim` of pixel `pixel`'s path this frame at depth `depth`,
// in [0, 1).
fn random(pixel: u32, depth: u32, dim: u32) -> f32 {
    let h = hash(hash(hash(pixel) ^ frame.info.y) ^ (depth * 16u + dim));
    return f32(h >> 8u) / 16777216.0;
}

fn luminance(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
}

// How far a ray leaving frame point `p` starts off the surface a ray of
// footprint `width` found there: a few ulps of f32 at the point, 2 mm, and a
// hundredth of the footprint (a tracer resolves geometry about as fine as
// the footprint, a terrain at a level of cells about that wide, and takes an
// origin within a thousandth of a cell of a solid face for an inside).
fn ray_offset(p: vec3<f32>, width: f32) -> f32 {
    return 0.002 + 4e-6 * max(abs(p.x), max(abs(p.y), abs(p.z))) + 0.01 * width;
}

// The sky's radiance along unit `d` from frame point `p`: the
// atmosphere's sky table there (irradiance / pi in L2 harmonics) back to
// radiance band by band, as GI's probe rays take it; without an
// atmosphere, the flat sky and ground. The sun's disc is not in it: the sun
// is a light the samplers sample (a delta light no BSDF sample meets), so it
// is counted once.
fn sky_radiance(p: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    if sky_frame.planet.w < 0.5 { return select(frame.ground.rgb, frame.sky.rgb, d.y >= 0.0); }
    return sky_table_eval((p - frame.eye.xyz) * 0.001, d, vec3<f32>(1.0, 1.5, 4.0));
}

// `c` reaching the eye along path `path` at depth `depth`.
fn add(path: ptr<function, Path>, c: vec3<f32>, depth: u32) {
    (*path).radiance += c;
    if depth >= 1u && ((*path).flags & REFERENCE_DIFFUSE_FIRST) != 0u {
        (*path).diffuse += c;
    }
}

// The camera's rays: a path a pixel of this frame's window, from the eye
// through a uniformly jittered point of the pixel. Every ray leaves the eye
// (the stream's one run).
@compute @workgroup_size(64)
fn reference_camera(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x;
    if p >= frame.info.x { return; }
    let pixel = (frame.size.w + p) % frame.size.z;
    let xy = vec2<f32>(f32(pixel % frame.size.x), f32(pixel / frame.size.x));
    let uv = (xy + vec2<f32>(random(pixel, 0u, 0u), random(pixel, 0u, 1u))) / vec2<f32>(frame.size.xy);
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    // Reversed depth: the near plane at 1.
    let near = frame.clip_to_frame * vec4<f32>(ndc, 1.0, 1.0);
    let mid = frame.clip_to_frame * vec4<f32>(ndc, 0.5, 1.0);
    let dir = normalize(mid.xyz / mid.w - near.xyz / near.w);
    paths[p] = Path(vec3<f32>(1.0), pixel, vec3<f32>(0.0), 0u, vec3<f32>(0.0), 0.0);
    cur_rows[p] = p;
    let cone = scene_cone_in_space(scene_ray_cone(0.0, frame.eye.w, RAY_MASK_ALL), frame.info.z);
    cur_rays[p] = SceneRay(frame.eye.xyz, REFERENCE_REACH, dir, cone);
    cur_hits[p] = SceneHit(0.0, 0u, 0u, 0u);
}

// A stage's hits: emission (the sky on a miss), a light query a hit, and
// the BSDF's sample for the next ray (Russian roulette from depth
// `frame.sky.w`), compacted into the next stage's stream.
@compute @workgroup_size(64)
fn reference_shade(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li == 0u {
        atomicStore(&w_count[0], 0u);
        atomicStore(&w_count[1], 0u);
    }
    workgroupBarrier();
    let r = id.x;
    let depth = stage.x;
    let live = r < atomicLoad(&cur_header[3]);
    var path: Path;
    var p = 0u;
    var query: LightQuery;
    var owner: Owner;
    var next: SceneRay;
    var want_query = false;
    var want_next = false;
    if live {
        p = cur_rows[r];
        path = paths[p];
        let ray = cur_rays[r];
        let hit = cur_hits[r];
        if hit.t <= 0.0 {
            add(&path, path.throughput * sky_radiance(ray.origin, ray.dir), depth);
        } else if !scene_hit_inside(hit) {
            // (The inside of something solid gives no light.)
            let row = cur_surfaces[r];
            let face = scene_surface_face(row);
            let wo = -ray.dir;
            // A two-sided surface met from behind is lit as its front seen
            // from that side, as a drawer turns it by `front_facing`.
            let s = bsdf_facing(scene_surface_unpack(row), face, wo);
            let at = ray.origin + ray.dir * hit.t;
            // Emission, found by the BSDF's sample: no sampler offers
            // emissive surfaces (emission stays emission at hits), so its
            // weight is 1 and it is counted once.
            add(&path, path.throughput * s.emissive, depth);
            let footprint = scene_ray_footprint(ray, hit.t);
            // One light sample at the hit (every kind's samplers'), lit from
            // both sides where light passes through it.
            let two_sided = (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u;
            let pixel = path.pixel;
            let u = vec2<f32>(random(pixel, depth, 2u), random(pixel, depth, 3u));
            query = light_query_point(at, footprint, s.normal, two_sided, u, hash(hash(pixel ^ frame.info.y) ^ (depth * 0x9e3779b9u)), scene_ray_space(ray));
            owner = Owner(path.throughput, p, r, 0u, 0u, 0u);
            want_query = true;
            if depth < frame.info.w {
                let b = bsdf_sample(s, wo, vec3<f32>(random(pixel, depth, 4u), random(pixel, depth, 5u), random(pixel, depth, 6u)));
                let side_o = dot(face, wo);
                let side_i = dot(face, b.wi);
                // A reflection stays on the side it came from, light through
                // a thin surface crosses it: a sample past the face its lobe
                // allows is lost (the shading normal's leak).
                let through = b.lobe == BSDF_LOBE_THROUGH;
                let kept = b.pdf > 0.0 && any(b.weight > vec3<f32>(0.0)) && ((side_o * side_i < 0.0) == through);
                var throughput = path.throughput * b.weight;
                var survive = 1.0;
                if f32(depth + 1u) >= frame.sky.w {
                    survive = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 1.0);
                }
                if kept && random(pixel, depth, 7u) < survive {
                    throughput /= survive;
                    if depth == 0u && (b.lobe == BSDF_LOBE_DIFFUSE || through) {
                        path.flags |= REFERENCE_DIFFUSE_FIRST;
                    }
                    path.throughput = throughput;
                    path.pdf = b.pdf;
                    // The ray's cone (Akenine-Moller et al., "Improved
                    // Shader and Texture Level of Detail Using Ray Cones",
                    // RTG II, 2021): it leaves the hit as a point (the
                    // reference samples a point of the pixel, not its
                    // footprint) and spreads by the lobe it was drawn from:
                    // twice a GGX lobe's alpha (the tangent of its width)
                    // for a specular sample, a diffuse lobe's 90-degree cone
                    // for the others.
                    let alpha = bsdf_alphas(s);
                    let lobe = select(2.0, 2.0 * sqrt(alpha.x * alpha.y), b.lobe == BSDF_LOBE_SPECULAR);
                    let cone = scene_cone_in_space(scene_ray_cone(0.0, scene_ray_spread(ray) + lobe, RAY_MASK_ALL), scene_ray_space(ray));
                    next = SceneRay(at + face * (sign(side_i) * ray_offset(at, footprint)), REFERENCE_REACH, b.wi, cone);
                    want_next = true;
                }
            }
        }
    }
    var q_slot = 0u;
    var n_slot = 0u;
    if want_query { q_slot = atomicAdd(&w_count[0], 1u); }
    if want_next { n_slot = atomicAdd(&w_count[1], 1u); }
    workgroupBarrier();
    if li == 0u {
        let queries_here = atomicLoad(&w_count[0]);
        let rays_here = atomicLoad(&w_count[1]);
        w_base[0] = select(0u, atomicAdd(&batch_header[3], queries_here), queries_here > 0u);
        w_base[1] = select(0u, atomicAdd(&next_header[3], rays_here), rays_here > 0u);
    }
    workgroupBarrier();
    if want_query {
        queries[w_base[0] + q_slot] = query;
        owners[w_base[0] + q_slot] = owner;
    }
    if want_next {
        let n = w_base[1] + n_slot;
        next_rows[n] = p;
        next_rays[n] = next;
        next_hits[n] = SceneHit(0.0, 0u, 0u, 0u);
    }
    if live { paths[p] = path; }
}

// The next stage's and the light batch's dispatches, from their counts; the
// rays into the frame's statistics.
@compute @workgroup_size(1)
fn reference_shade_headers() {
    let next = atomicLoad(&next_header[3]);
    atomicStore(&next_header[0], (next + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP);
    atomicStore(&next_header[1], 1u);
    atomicStore(&next_header[2], 1u);
    let queries = atomicLoad(&batch_header[3]);
    atomicStore(&batch_header[0], (queries + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP);
    atomicStore(&batch_header[1], 1u);
    atomicStore(&batch_header[2], 1u);
    atomicAdd(&stats[REFERENCE_STAT], next);
}

// Each light sample's light off its hit, weighted against the BSDF's
// sample of the same direction (the power heuristic; a delta light's
// weight is 1), waiting on a shadow ray toward the light, a workgroup's
// rows together.
@compute @workgroup_size(64)
fn reference_shadow(@builtin(global_invocation_id) id: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if li == 0u { atomicStore(&w_count[0], 0u); }
    workgroupBarrier();
    let q = id.x;
    var want = false;
    var shadow: SceneRay;
    var light = vec4<u32>(0u);
    if q < atomicLoad(&batch_header[3]) {
        let sample = samples[q];
        if light_sample_found(sample) {
            let o = owners[q];
            let ray = cur_rays[o.row];
            let hit = cur_hits[o.row];
            let row = cur_surfaces[o.row];
            let face = scene_surface_face(row);
            let wo = -ray.dir;
            let s = bsdf_facing(scene_surface_unpack(row), face, wo);
            let at = ray.origin + ray.dir * hit.t;
            let dir = light_sample_dir(sample);
            let side_o = dot(face, wo);
            let side_i = dot(face, dir);
            let crosses = side_o * side_i < 0.0;
            let light_pdf = light_sample_pdf(sample);
            var weight = 1.0;
            if light_pdf > 0.0 {
                let bsdf_pdf = bsdf_pdf(s, wo, dir);
                weight = light_pdf * light_pdf / (light_pdf * light_pdf + bsdf_pdf * bsdf_pdf);
            }
            let c = o.throughput * bsdf_eval(s, wo, dir) * light_sample_radiance(sample) * weight;
            if luminance(c) > 0.0 && (!crosses || (s.flags & SURFACE_FLAG_TWO_SIDED) != 0u) {
                let footprint = scene_ray_footprint(ray, hit.t);
                let off = ray_offset(at, footprint);
                let distance = light_sample_distance(sample);
                let reach = select(max(distance - 2.0 * off, 1e-4), REFERENCE_REACH, distance >= LIGHT_SAMPLE_FAR);
                // A point's visibility of the light: no width, no spread.
                // (As wide as the camera's footprint at the hit, shadow
                // rays leaving voxel_craft's meadow were all stopped: the
                // terrain came out black; with zero width, or without the
                // meadow's foliage, they are not. rendering.md section 4.)
                let cone = scene_cone_in_space(scene_ray_cone(0.0, 0.0, RAY_MASK_ALL), scene_ray_space(ray));
                shadow = SceneRay(at + face * (sign(side_i) * off), reach, dir, cone);
                light = vec4<u32>(bitcast<vec3<u32>>(c), o.path);
                want = true;
            }
        }
    }
    var slot = 0u;
    if want { slot = atomicAdd(&w_count[0], 1u); }
    workgroupBarrier();
    if li == 0u {
        let here = atomicLoad(&w_count[0]);
        w_base[0] = select(0u, atomicAdd(&shadow_header[3], here), here > 0u);
    }
    workgroupBarrier();
    if want {
        let n = w_base[0] + slot;
        shadow_rays[n] = shadow;
        shadow_hits[n] = SceneHit(0.0, 0u, 0u, 0u);
        pending[n] = light;
    }
}

@compute @workgroup_size(1)
fn reference_shadow_header() {
    let count = atomicLoad(&shadow_header[3]);
    atomicStore(&shadow_header[0], (count + RAY_STREAM_GROUP - 1u) / RAY_STREAM_GROUP);
    atomicStore(&shadow_header[1], 1u);
    atomicStore(&shadow_header[2], 1u);
    atomicAdd(&stats[REFERENCE_STAT], count);
}

// What each shadow ray let through of its light (a path has one at most a
// stage).
@compute @workgroup_size(64)
fn reference_resolve(@builtin(global_invocation_id) id: vec3<u32>) {
    let n = id.x;
    if n >= atomicLoad(&shadow_header[3]) { return; }
    let c = pending[n];
    let p = c.w;
    var path = paths[p];
    add(&path, bitcast<vec3<f32>>(c.xyz) * scene_occlusion_through(shadow_hits[n]), stage.x);
    paths[p] = path;
}

// The frame's paths into the image: each pixel's sum and its count of
// paths, in 32-bit floats. A path that came back not finite (a fault
// upstream) is counted as no light, so one cannot erase a pixel's sum.
@compute @workgroup_size(64)
fn reference_accumulate(@builtin(global_invocation_id) id: vec3<u32>) {
    let p = id.x;
    if p >= frame.info.x { return; }
    let path = paths[p];
    let finite = all(abs(path.radiance) < vec3<f32>(3e38)) && all(abs(path.diffuse) < vec3<f32>(3e38));
    let radiance = select(vec3<f32>(0.0), path.radiance, finite);
    let diffuse = select(vec3<f32>(0.0), path.diffuse, finite);
    image[path.pixel] += vec4<f32>(radiance, 1.0);
    image_diffuse[path.pixel] += vec4<f32>(diffuse, 1.0);
}
