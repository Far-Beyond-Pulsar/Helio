// Procedural Clouds - true 3D thickness baked per sample (0 extra fetches in volume path)
// Based on user's Blender node graph, with thickness = d*1.8
// Helpers (hash, noise, voronoi) + main

fn rot_u32(x: u32, k: u32) -> u32 { return (x << k) | (x >> (32u - k)); }
fn hash_uint4(kx: u32, ky: u32, kz: u32, kw: u32) -> u32 {
    var a = 0xdeadbeefu + (4u << 2u) + 13u; var b = a; var c = a; a += kx; b += ky; a -= c; a ^= rot_u32(c, 4u); c += b; b -= a; b ^= rot_u32(a, 6u); a += c; c -= b; c ^= rot_u32(b, 8u); b += a; a -= c; a ^= rot_u32(c, 16u); c += b; b -= a; b ^= rot_u32(a, 19u); a += c; c -= b; c ^= rot_u32(b, 4u); b += a; a += kz; b += kw; c ^= b; c -= rot_u32(b, 14u); a ^= c; a -= rot_u32(c, 11u); b ^= a; b -= rot_u32(a, 25u); c ^= b; c -= rot_u32(b, 16u); a ^= c; a -= rot_u32(c, 4u); b ^= a; b -= rot_u32(a, 14u); c ^= b; c -= rot_u32(b, 24u); return c;
}
fn hash_uint4_to_float(kx: u32, ky: u32, kz: u32, kw: u32) -> f32 { return f32(hash_uint4(kx, ky, kz, kw)) / f32(0xFFFFFFFFu); }
fn hash_vec4_to_vec4(k: vec4f) -> vec4f { return vec4f(hash_uint4_to_float(bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z), bitcast<u32>(k.w)), hash_uint4_to_float(bitcast<u32>(k.w), bitcast<u32>(k.x), bitcast<u32>(k.y), bitcast<u32>(k.z)), hash_uint4_to_float(bitcast<u32>(k.z), bitcast<u32>(k.w), bitcast<u32>(k.x), bitcast<u32>(k.y)), hash_uint4_to_float(bitcast<u32>(k.y), bitcast<u32>(k.z), bitcast<u32>(k.w), bitcast<u32>(k.x))); }
fn noise_fade(t: f32) -> f32 { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }
fn tri_mix(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32, x: f32, y: f32, z: f32) -> f32 { let x1 = 1.0 - x;let y1 = 1.0 - y;let z1 = 1.0 - z;return z1 * (y1 * (v0 * x1 + v1 * x) + y * (v2 * x1 + v3 * x)) + z * (y1 * (v4 * x1 + v5 * x) + y * (v6 * x1 + v7 * x)); }
fn quad_mix(v0: f32, v1: f32, v2: f32, v3: f32, v4: f32, v5: f32, v6: f32, v7: f32, v8: f32, v9: f32, v10: f32, v11: f32, v12: f32, v13: f32, v14: f32, v15: f32, x: f32, y: f32, z: f32, w: f32) -> f32 { return mix(tri_mix(v0, v1, v2, v3, v4, v5, v6, v7, x, y, z), tri_mix(v8, v9, v10, v11, v12, v13, v14, v15, x, y, z), w); }
fn noiseg_4d(h: u32, x: f32, y: f32, z: f32, w: f32) -> f32 { let hh = h & 31u;let u = select(x, y, hh >= 24u);let v = select(y, z, hh >= 16u);let s = select(z, w, hh >= 8u);let r = select(u, -u, (hh & 1u) != 0u);let rv = select(v, -v, (hh & 2u) != 0u);let rs = select(s, -s, (hh & 4u) != 0u);return r + rv + rs; }
fn perlin_noise_4d(p: vec4f) -> f32 {
    let pf = floor(p);let X = i32(pf.x);let Y = i32(pf.y);let Z = i32(pf.z);let W = i32(pf.w);
    let fx = p.x - pf.x;let fy = p.y - pf.y;let fz = p.z - pf.z;let fw = p.w - pf.w;
    let u = noise_fade(fx);let v = noise_fade(fy);let t = noise_fade(fz);let s = noise_fade(fw);
    return quad_mix(noiseg_4d(hash_uint4(u32(X), u32(Y), u32(Z), u32(W)), fx, fy, fz, fw), noiseg_4d(hash_uint4(u32(X + 1), u32(Y), u32(Z), u32(W)), fx - 1, fy, fz, fw), noiseg_4d(hash_uint4(u32(X), u32(Y + 1), u32(Z), u32(W)), fx, fy - 1, fz, fw), noiseg_4d(hash_uint4(u32(X + 1), u32(Y + 1), u32(Z), u32(W)), fx - 1, fy - 1, fz, fw), noiseg_4d(hash_uint4(u32(X), u32(Y), u32(Z + 1), u32(W)), fx, fy, fz - 1, fw), noiseg_4d(hash_uint4(u32(X + 1), u32(Y), u32(Z + 1), u32(W)), fx - 1, fy, fz - 1, fw), noiseg_4d(hash_uint4(u32(X), u32(Y + 1), u32(Z + 1), u32(W)), fx, fy - 1, fz - 1, fw), noiseg_4d(hash_uint4(u32(X + 1), u32(Y + 1), u32(Z + 1), u32(W)), fx - 1, fy - 1, fz - 1, fw), noiseg_4d(hash_uint4(u32(X), u32(Y), u32(Z), u32(W + 1)), fx, fy, fz, fw - 1), noiseg_4d(hash_uint4(u32(X + 1), u32(Y), u32(Z), u32(W + 1)), fx - 1, fy, fz, fw - 1), noiseg_4d(hash_uint4(u32(X), u32(Y + 1), u32(Z), u32(W + 1)), fx, fy - 1, fz, fw - 1), noiseg_4d(hash_uint4(u32(X + 1), u32(Y + 1), u32(Z), u32(W + 1)), fx - 1, fy - 1, fz, fw - 1), noiseg_4d(hash_uint4(u32(X), u32(Y), u32(Z + 1), u32(W + 1)), fx, fy, fz - 1, fw - 1), noiseg_4d(hash_uint4(u32(X + 1), u32(Y), u32(Z + 1), u32(W + 1)), fx - 1, fy, fz - 1, fw - 1), noiseg_4d(hash_uint4(u32(X), u32(Y + 1), u32(Z + 1), u32(W + 1)), fx, fy - 1, fz - 1, fw - 1), noiseg_4d(hash_uint4(u32(X + 1), u32(Y + 1), u32(Z + 1), u32(W + 1)), fx - 1, fy - 1, fz - 1, fw - 1), u, v, t, s);
}
fn noise_fbm(p: vec4f, detail: f32, roughness: f32, lacunarity: f32, normalize: bool) -> f32 {
    var fscale = 1.0;var amp = 1.0;var maxamp = 0.0;var sum = 0.0;let d = i32(detail);
    for (var i = 0; i <= d; i++) { let t = perlin_noise_4d(fscale * p);sum += t * amp;maxamp += amp;amp *= roughness;fscale *= lacunarity; }
    let rmd = detail - floor(detail);if rmd != 0.0 { let t = perlin_noise_4d(fscale * p);let sum2 = sum + t * amp;return select(mix(sum, sum2, rmd), mix(0.5 + 0.5 * (sum / maxamp), 0.5 + 0.5 * (sum2 / (maxamp + amp)), rmd), normalize); }
    return select(sum, 0.5 + 0.5 * (sum / maxamp), normalize);
}
fn random_vec4_offset(seed: f32) -> vec4f { return hash_vec4_to_vec4(vec4f(seed, seed * 1.37, seed * 2.23, seed * 3.11)); }
fn node_noise_texture_4d_value(co: vec3f, w: f32, scale: f32, detail: f32, roughness: f32, lacunarity: f32, distortion: f32, normalize: f32) -> f32 {
    var p = vec4f(co, w) * scale;if distortion != 0.0 { p += vec4f(perlin_noise_4d(p + random_vec4_offset(0.0)) * distortion, perlin_noise_4d(p + random_vec4_offset(1.0)) * distortion, perlin_noise_4d(p + random_vec4_offset(2.0)) * distortion, perlin_noise_4d(p + random_vec4_offset(3.0)) * distortion); }
    return noise_fbm(p, detail, roughness, lacunarity, normalize != 0.0);
}
fn hash_pcg4d_i(v: vec4i) -> vec4i { var vv = v * 1664525 + 1013904223;vv.x += vv.y * vv.w;vv.y += vv.z * vv.x;vv.z += vv.x * vv.y;vv.w += vv.y * vv.z;vv = vv ^ (vv >> vec4u(16u));vv.x += vv.y * vv.w;vv.y += vv.z * vv.x;vv.z += vv.x * vv.y;vv.w += vv.y * vv.z;return vv; }
fn hash_int4_to_vec4(k: vec4i) -> vec4f { let h = hash_pcg4d_i(k);return vec4f(h & vec4i(0x7fffffff)) * (1.0 / f32(0x7fffffff)); }
fn hash_int4_to_vec3(k: vec4i) -> vec3f { return hash_int4_to_vec4(k).xyz; }
const SHD_VORONOI_EUCLIDEAN = 0;const SHD_VORONOI_F1 = 0;
struct VoronoiParams {
    scale: f32,
    detail: f32,
    roughness: f32,
    lacunarity: f32,
    smoothness: f32,
    exponent: f32,
    randomness: f32,
    max_distance: f32,
    normalize: bool,
    feature: i32,
    metric: i32};
struct VoronoiOutput {
    Distance: f32,
    Color: vec3f,
    Position: vec4f};
fn voronoi_distance(a: vec4f, b: vec4f, p: VoronoiParams) -> f32 { return distance(a, b); }
fn voronoi_f1(params: VoronoiParams, coord: vec4f) -> VoronoiOutput {
    let cellP = floor(coord);let local = coord - cellP;let cell = vec4i(cellP);
    var minD = 3.402823466e+38;var off = vec4i(0);var tpos = vec4f(0.0);
    for (var u = -1; u <= 1; u++) {
        for (var k = -1; k <= 1; k++) {
            for (var j = -1; j <= 1; j++) {
                for (var i = -1; i <= 1; i++) {
                    let o = vec4i(i, j, k, u);let pp = vec4f(o) + hash_int4_to_vec4(cell + o) * params.randomness;let d = voronoi_distance(pp, local, params);if d < minD { off = o;minD = d;tpos = pp; }
                }
            }
        }
    }
    var o: VoronoiOutput;o.Distance = minD;o.Color = hash_int4_to_vec3(cell + off);o.Position = tpos + cellP;return o;
}
fn fractal_voronoi_x_fx(params: VoronoiParams, coord: vec4f) -> VoronoiOutput {
    var amp = 1.0;var maxAmp = 0.0;var scale = 1.0;var Out: VoronoiOutput;Out.Distance = 0.0;Out.Color = vec3f(0.0);Out.Position = vec4f(0.0);
    let zero = params.detail == 0.0 || params.roughness == 0.0;let maxI = i32(ceil(params.detail));
    for (var i = 0; i <= maxI; i++) { let o = voronoi_f1(params, coord * scale);if zero { maxAmp = 1.0;Out = o;break; } else if f32(i) <= params.detail { maxAmp += amp;Out.Distance += o.Distance * amp;Out.Color += o.Color * amp;Out.Position = mix(Out.Position, o.Position / scale, amp);scale *= params.lacunarity;amp *= params.roughness; } else { let r = params.detail - floor(params.detail);if r != 0.0 { maxAmp = mix(maxAmp, maxAmp + amp, r);Out.Distance = mix(Out.Distance, Out.Distance + o.Distance * amp, r);Out.Color = mix(Out.Color, Out.Color + o.Color * amp, r);Out.Position = mix(Out.Position, mix(Out.Position, o.Position / scale, amp), r); } } }
    if params.normalize { Out.Distance /= maxAmp * params.max_distance;Out.Color /= maxAmp; }Out.Position /= params.scale;return Out;
}
fn node_tex_voronoi_f1_4d_distance(coord: vec3f, w: f32, scale: f32, detail: f32, roughness: f32, lacunarity: f32, smoothness: f32, exponent: f32, randomness: f32, metric: f32, normalize: f32) -> f32 {
    var p: VoronoiParams;p.feature = 0;p.metric = i32(metric);p.scale = scale;p.detail = clamp(detail, 0.0, 15.0);p.roughness = clamp(roughness, 0.0, 1.0);p.lacunarity = lacunarity;p.smoothness = clamp(smoothness / 2.0, 0.0, 0.5);p.exponent = exponent;p.randomness = clamp(randomness, 0.0, 1.0);p.max_distance = 0.0;p.normalize = normalize != 0.0;
    let ws = w * scale;let cs = coord * scale;p.max_distance = voronoi_distance(vec4f(0.0), vec4f(0.5 + 0.5 * p.randomness), p);let Out = fractal_voronoi_x_fx(p, vec4f(cs, ws));return Out.Distance;
}
fn mapRange(v: f32, f0: f32, f1: f32, t0: f32, t1: f32) -> f32 { if abs(f1 - f0) < 1e-5 { return t0; }let t = (v - f0) / (f1 - f0);return clamp(mix(t0, t1, t), min(t0, t1), max(t0, t1)); }
fn clamp01(v: f32) -> f32 { return clamp(v, 0.0, 1.0); }
fn vertical_band(y: f32, l0: f32, l1: f32, u1: f32, u0: f32) -> f32 { return smoothstep(l0, l1, y) * (1.0 - smoothstep(u1, u0, y)); }
fn ellipsoid_blob(p: vec3f, c: vec3f, s: vec3f) -> f32 { let d = length((p - c) * s);return 1.0 - smoothstep(0.19, 0.52, d); }
fn spiral_scroll(p: vec3f, c: vec3f, s: vec3f, turns: f32, phase: f32) -> f32 { let q = (p - c) * s;let r = length(q.xy);let a = atan2(q.y, q.x);let crest = cos(a + r * turns + phase) * 0.5 + 0.5;let tube = smoothstep(0.76, 0.985, crest);let rw = smoothstep(0.025, 0.10, r) * (1.0 - smoothstep(0.38, 0.64, r));let dw = exp(-abs(q.z) * 2.35);return tube * rw * dw; }

struct Camera {
    invViewProj: mat4x4f,
    position: vec3f,
    _pad: f32};
struct Params {
    time_pack: vec4f,
    alt_pack: vec4f,
    scale_pack: vec4f,
    extra_pack: vec4f,
    cache_pack: vec4f,
    bounds_pack: vec4f};
@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<uniform> params: Params;

fn cloudDensity(pos: vec3f) -> f32 {
    let tN = params.time_pack.x;let tV1 = params.time_pack.y;let tV2 = params.time_pack.z;let dens = params.time_pack.w;
    let lowAlt = params.alt_pack.x;let alt = params.alt_pack.y;let facM = params.alt_pack.z;let facD = params.alt_pack.w;
    let facS = params.scale_pack.x;let sAlt = params.scale_pack.y;let sN = params.scale_pack.z;let sV1 = params.scale_pack.w;
    let sV2 = params.extra_pack.x;let det = params.extra_pack.y;
    let obj = vec3f(pos.x, pos.z, pos.y);let zN = (pos.y - BOX_MIN.y) / (getBoxMax().y - BOX_MIN.y);let Z = 1.0 - clamp(zN, 0.0, 1.0);
    let altFrom = alt / 5.0;let altTo = 1.0 - lowAlt;let altRamp = mapRange(Z, 0.0, altFrom, altTo, 1.0);
    let nC = obj / sN;let s1N = node_noise_texture_4d_value(nC, tN, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let altMask = clamp01(altRamp * s1N);
    let v1C = obj / sV1;let v1d = node_tex_voronoi_f1_4d_distance(v1C, tV1, 5.0, det, 0.5, 3.0, 1.0, 0.5, 1.0, 0.0, 1.0);
    let v1m = mapRange(v1d, 0.0, 0.75, facM * -0.4, facM);let v1s = clamp01(v1m * 0.5);let s2 = clamp01(altMask + v1s);
    let v2C = obj / sV2;let v2d = node_tex_voronoi_f1_4d_distance(v2C, tV2, 2.0, det * 5.0, 0.75, 2.5, 1.0, 0.5, 1.0, 0.0, 1.0);
    let v2m = mapRange(v2d, 0.0, 1.0, facD * -0.25, facD);let s3 = clamp01(s2 + v2m);
    let cutFrom = alt * sAlt;let cut = mapRange(Z, cutFrom, 0.0, 0.0, 1.0);let shaped = clamp01(s3 - cut);let finalShaped = clamp01(shaped - (1.0 - facS));
    let falloff = mapRange(Z, 0.0, alt, 0.0, 1.0);let ds = dens * 5.0;return finalShaped * falloff * ds;
}
const BOX_MIN = vec3f(-18.0, 12.0, -18.0);
const BOX_MAX_XZ = 18.0;
fn getBoxMax() -> vec3f { return vec3f(BOX_MAX_XZ, params.bounds_pack.x, BOX_MAX_XZ); }
struct HitInfo {
    hit: bool,
    tNear: f32,
    tFar: f32};
fn intersectBox(ro: vec3f, rd: vec3f) -> HitInfo { let inv = 1.0 / rd;let t0 = (BOX_MIN - ro) * inv;let t1 = (getBoxMax() - ro) * inv;let tmin = min(t0, t1);let tmax = max(t0, t1);let tn = max(tmin.x, max(tmin.y, tmin.z));let tf = min(tmax.x, min(tmax.y, tmax.z));return HitInfo(tf >= max(tn, 0.0), tn, tf); }
const SUN_DIR = vec3f(0.189, 0.943, 0.283);const SUN_COLOR = vec3f(1.0, 1.0, 1.0);const AMBIENT = vec3f(0.26, 0.30, 0.42);const BG_COLOR = vec3f(0.045, 0.10, 0.18);
fn hgPhase(c: f32, g: f32) -> f32 { let g2 = g * g;return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * c, 1.5)); }
fn interleavedGradientNoise(uv: vec2f) -> f32 { let m = vec3f(0.06711056, 0.00583715, 52.9829189);return fract(m.z * fract(dot(uv, m.xy))); }
// Thickness-aware sampling: returns (density, thickness) - thin edges, thick core
fn sampleDensityThick(pos: vec3f) -> vec2f { let d = cloudDensity(pos);let thick = clamp(d * 1.2 - 0.1, 0.0, 1.0);return vec2f(d, thick); }
fn lightMarch(pos: vec3f) -> f32 { var s = 0.0;let steps = i32(params.cache_pack.y);let sz = 0.15;for (var i = 1; i <= steps; i++) { let p = pos + SUN_DIR * (f32(i) * sz);s += sampleDensityThick(p).x * sz; }return exp(-s * params.cache_pack.z); }
struct VSOut {
    @builtin(position)
    pos: vec4f,@location(0)
    uv: vec2f};
@vertexfn vs(@builtin(vertex_index)vi: u32) -> VSOut { let p = array<vec2f,3>(vec2f(-1, -1), vec2f(3, -1), vec2f(-1, 3));var o: VSOut;o.pos = vec4f(p[vi], 0, 1);o.uv = p[vi];return o; }
@fragmentfn fs(@builtin(position)fc: vec4f,@location(0) uv: vec2f) -> @location(0) vec4f {
    let skipLight = params.extra_pack.w > 0.5;let numSteps = i32(params.extra_pack.z);
    let wn = camera.invViewProj * vec4f(uv, 0, 1);let wf = camera.invViewProj * vec4f(uv, 1, 1);
    let ro = camera.position;let rd = normalize(wf.xyz / wf.w - wn.xyz / wn.w);
    let hit = intersectBox(ro, rd);
    let sky = mix(BG_COLOR, vec3f(0.1, 0.2, 0.4), clamp(rd.y * 0.5 + 0.5, 0.0, 1.0));
    let sunTheta = dot(rd, SUN_DIR);let finalSky = sky + pow(max(sunTheta, 0.0), 64.0) * SUN_COLOR * 0.8;
    var out = finalSky;
    if hit.hit {
        let t0 = max(hit.tNear, 0.0);let t1 = hit.tFar;let step = (t1 - t0) / f32(numSteps);
        let dither = interleavedGradientNoise(fc.xy);
        var pos = ro + rd * (t0 + step * dither);var trans = 1.0;var col = vec3f(0.0);
        let phase = mix(1.0, hgPhase(sunTheta, 0.45), 0.6);
        for (var i = 0; i < 64; i++) {
            if i >= numSteps { break; }let d2 = sampleDensityThick(pos);let d = d2.x;let thick = d2.y;
            if d > 0.015 {
                let effStep = step * (1.0 + thick * 0.35);
                let tr = exp(-d * effStep * 0.9);
                let sh = select(lightMarch(pos), 1.0, skipLight);
                let scat = sh * phase * (1.0 - exp(-d * (1.0 + thick * 0.25)));
                let lit = SUN_COLOR * scat * params.cache_pack.w + AMBIENT * 0.5;
                col += trans * (1.0 - tr) * lit;
                trans *= tr;
                if trans < 0.01 { break; }
            }
            pos += rd * step;
        }
        out = col + trans * finalSky;
    }
    out = out / (out + vec3f(1.0));out = pow(out, vec3f(1.0 / 2.2));return vec4f(out, 1.0);
}
