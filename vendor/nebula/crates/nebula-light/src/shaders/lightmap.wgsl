// ── Nebula lightmap baker — GPU path tracer ──────────────────────────────────
//
// This is a reference Monte Carlo path tracer.  It is intentionally readable
// over micro-optimised — production quality comes from more samples, not from
// shader tricks.  Replace with a hardware ray-tracing pipeline on devices that
// support it.

struct Params {
    resolution:        u32,
    samples_per_texel: u32,
    bounce_count:      u32,
    num_lights:        u32,
    num_meshes:        u32,
    num_indices:       u32,
    max_ray_dist:      f32,
    frame_seed:        u32,
}

struct Vertex {
    pos:    vec3<f32>,
    _pad0:  f32,
    normal: vec3<f32>,
    _pad1:  f32,
    uv:     vec2<f32>,
    lm_uv:  vec2<f32>,
}

struct MeshInfo {
    index_offset:  u32,
    index_count:   u32,
    vertex_offset: u32,
    material_id:   u32,
    transform:     mat4x4<f32>,
}

struct GpuLight {
    pos_range:       vec4<f32>,
    dir_outer:       vec4<f32>,
    color_intensity: vec4<f32>,
    kind:            u32,
    inner_angle:     f32,
    _pad:            vec2<u32>,
}

@group(0) @binding(0) var<uniform>         params:     Params;
@group(0) @binding(1) var<storage, read>   vertices:   array<Vertex>;
@group(0) @binding(2) var<storage, read>   indices:    array<u32>;
@group(0) @binding(3) var<storage, read>   meshes:     array<MeshInfo>;
@group(0) @binding(4) var<storage, read>   lights:     array<GpuLight>;
@group(0) @binding(5) var                  out_lm:     texture_storage_2d<rgba32float, write>;

// ── PCG random ───────────────────────────────────────────────────────────────

var<private> rng_state: u32;

fn pcg_next() -> u32 {
    rng_state = rng_state * 747796405u + 2891336453u;
    let word = ((rng_state >> ((rng_state >> 28u) + 4u)) ^ rng_state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f() -> f32 { return f32(pcg_next()) * (1.0 / 4294967296.0); }

fn rand_hemisphere(n: vec3<f32>) -> vec3<f32> {
    let u1 = rand_f();
    let u2 = rand_f();
    let r   = sqrt(1.0 - u1 * u1);
    let phi = 6.28318530718 * u2;
    // Build local ONB
    var t  = vec3<f32>(1.0, 0.0, 0.0);
    if abs(n.x) > 0.9 { t = vec3<f32>(0.0, 1.0, 0.0); }
    let b  = normalize(cross(n, t));
    let tt = cross(b, n);
    return normalize(r * cos(phi) * tt + r * sin(phi) * b + u1 * n);
}

// ── Möller–Trumbore ray–triangle intersection ─────────────────────────────────

fn ray_triangle(
    ro: vec3<f32>, rd: vec3<f32>,
    v0: vec3<f32>, v1: vec3<f32>, v2: vec3<f32>,
) -> f32 {
    let e1  = v1 - v0;
    let e2  = v2 - v0;
    let h   = cross(rd, e2);
    let det = dot(e1, h);
    if abs(det) < 1e-7 { return -1.0; }
    let inv = 1.0 / det;
    let s   = ro - v0;
    let u   = inv * dot(s, h);
    if u < 0.0 || u > 1.0 { return -1.0; }
    let q   = cross(s, e1);
    let v   = inv * dot(rd, q);
    if v < 0.0 || (u + v) > 1.0 { return -1.0; }
    let t   = inv * dot(e2, q);
    if t > 1e-4 { return t; }
    return -1.0;
}

// ── Scene ray march ───────────────────────────────────────────────────────────

fn scene_hit(ro: vec3<f32>, rd: vec3<f32>, max_t: f32) -> f32 {
    var closest = max_t;
    for (var mi = 0u; mi < params.num_meshes; mi++) {
        let mesh = meshes[mi];
        for (var ii = mesh.index_offset; ii < mesh.index_offset + mesh.index_count; ii += 3u) {
            let i0 = indices[ii];
            let i1 = indices[ii + 1u];
            let i2 = indices[ii + 2u];
            let p0 = (mesh.transform * vec4<f32>(vertices[i0].pos, 1.0)).xyz;
            let p1 = (mesh.transform * vec4<f32>(vertices[i1].pos, 1.0)).xyz;
            let p2 = (mesh.transform * vec4<f32>(vertices[i2].pos, 1.0)).xyz;
            let t  = ray_triangle(ro, rd, p0, p1, p2);
            if t > 0.0 && t < closest { closest = t; }
        }
    }
    return closest;
}

// ── Light contribution ────────────────────────────────────────────────────────

fn eval_direct(pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var lo = vec3<f32>(0.0);
    for (var li = 0u; li < params.num_lights; li++) {
        let lgt = lights[li];
        var l_dir = vec3<f32>(0.0);
        var l_dist = params.max_ray_dist;
        var l_radiance = lgt.color_intensity.xyz * lgt.color_intensity.w;

        if lgt.kind == 0u { // directional
            l_dir  = -normalize(lgt.dir_outer.xyz);
            l_dist  = params.max_ray_dist;
        } else { // point / spot
            let to_light = lgt.pos_range.xyz - pos;
            l_dist = length(to_light);
            if l_dist > lgt.pos_range.w { continue; }
            l_dir  = to_light / l_dist;
            let atten = 1.0 / (l_dist * l_dist + 0.0001);
            l_radiance *= atten;
        }

        let ndotl = max(dot(n, l_dir), 0.0);
        if ndotl <= 0.0 { continue; }

        // Shadow ray
        let shadow_t = scene_hit(pos + n * 0.001, l_dir, l_dist - 0.001);
        if shadow_t < l_dist - 0.001 { continue; }

        lo += l_radiance * ndotl;
    }
    return lo;
}

// ── Texel world-space lookup ──────────────────────────────────────────────────

struct TexelInfo { pos: vec3<f32>, normal: vec3<f32>, valid: bool }

fn texel_world_pos(lm_uv: vec2<f32>) -> TexelInfo {
    // Brute-force: find the triangle that contains this LM UV.
    // Production would use a pre-built texel→triangle table.
    var best_t = TexelInfo(vec3<f32>(0.0), vec3<f32>(0.0, 1.0, 0.0), false);
    for (var mi = 0u; mi < params.num_meshes; mi++) {
        let mesh = meshes[mi];
        for (var ii = mesh.index_offset; ii < mesh.index_offset + mesh.index_count; ii += 3u) {
            let i0 = indices[ii]; let i1 = indices[ii+1u]; let i2 = indices[ii+2u];
            let uv0 = vertices[i0].lm_uv;
            let uv1 = vertices[i1].lm_uv;
            let uv2 = vertices[i2].lm_uv;
            // Barycentric coords in UV space
            let d1  = uv1 - uv0; let d2 = uv2 - uv0; let dp = lm_uv - uv0;
            let inv = 1.0 / (d1.x * d2.y - d1.y * d2.x);
            let u   = (dp.x * d2.y - dp.y * d2.x) * inv;
            let v   = (d1.x * dp.y - d1.y * dp.x) * inv;
            if u >= 0.0 && v >= 0.0 && (u + v) <= 1.0 {
                let w = 1.0 - u - v;
                let p0 = (mesh.transform * vec4<f32>(vertices[i0].pos, 1.0)).xyz;
                let p1 = (mesh.transform * vec4<f32>(vertices[i1].pos, 1.0)).xyz;
                let p2 = (mesh.transform * vec4<f32>(vertices[i2].pos, 1.0)).xyz;
                let n0 = (mesh.transform * vec4<f32>(vertices[i0].normal, 0.0)).xyz;
                let n1 = (mesh.transform * vec4<f32>(vertices[i1].normal, 0.0)).xyz;
                let n2 = (mesh.transform * vec4<f32>(vertices[i2].normal, 0.0)).xyz;
                best_t.pos    = p0 * w + p1 * u + p2 * v;
                best_t.normal = normalize(n0 * w + n1 * u + n2 * v);
                best_t.valid  = true;
                return best_t;
            }
        }
    }
    return best_t;
}

// ── Main ──────────────────────────────────────────────────────────────────────

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let res = params.resolution;
    if gid.x >= res || gid.y >= res { return; }

    // Seed RNG per-texel
    rng_state = gid.x + gid.y * res + params.frame_seed * res * res;

    let lm_uv = (vec2<f32>(gid.xy) + 0.5) / f32(res);
    let texel  = texel_world_pos(lm_uv);

    if !texel.valid {
        textureStore(out_lm, vec2<i32>(gid.xy), vec4<f32>(0.0));
        return;
    }

    var accum = vec3<f32>(0.0);
    let spp   = params.samples_per_texel;

    for (var s = 0u; s < spp; s++) {
        // Direct illumination
        var radiance = eval_direct(texel.pos, texel.normal);

        // Indirect bounces
        var pos    = texel.pos;
        var normal = texel.normal;
        var throughput = vec3<f32>(1.0);

        for (var b = 0u; b < params.bounce_count; b++) {
            let dir = rand_hemisphere(normal);
            let t   = scene_hit(pos + normal * 0.001, dir, params.max_ray_dist);
            if t >= params.max_ray_dist { break; }

            // Move to bounce point (no material lookup → assume 50% grey diffuse)
            pos    = pos + dir * t;
            normal = dir; // approximate; a real baker would interpolate vertex normals
            throughput *= 0.5 * 3.14159; // Lambertian BRDF × pi
            radiance += throughput * eval_direct(pos, normal);
        }

        accum += radiance;
    }

    if spp > 0u { accum /= f32(spp); }
    textureStore(out_lm, vec2<i32>(gid.xy), vec4<f32>(accum, 1.0));
}
