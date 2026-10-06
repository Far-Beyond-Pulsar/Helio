// The terrain-stack interpreter of `helio.terrain`; mirror of src/landform.rs.
const LANDFORM_WARP: u32 = 6u; // leading domain-warp slots (two per axis)

struct LandformOctave { shift: u32, amplitude: i32, seed: u32, kind: u32 } // kind | layer << 8
struct StackLayer { kind: u32, mask: u32, a: i32, b: i32 }
// `landform::PackedRule`: material, speck share, patch shift and threshold;
// height and slope ranges; depth and moisture ranges; erosion range, band
// period and parity.
struct MaterialRule { head: vec4<i32>, height_slope: vec4<i32>, depth_moisture: vec4<i32>, surface_bands: vec4<i32> }
struct TerrainConstants {
    header: vec4<i32>, // octave count, layer mm, soil depth (cells), seed
    levels: vec4<i32>, // moisture shift, continents layer + 1, snowline, low-basin height (mm)
    shape: vec4<i32>,  // ridge display layer + 1, steep slope (cells), ridge display flag, plane (+Y up)
    style: vec4<i32>,  // material style, Layered surface / soil / rock ids
    stack: vec4<i32>,  // layer count, erosion amplitude sum, domain radius, warp octaves
    materials: vec4<i32>, // rules style: rule count, fallback material
    layers: array<StackLayer, 8>,
    octaves: array<LandformOctave, 48>,
    rules: array<MaterialRule, 16>,
    ridge_suffix: array<vec4<i32>, 66>, // signed conditional suffix means
    // Caves and overhangs (`LandformVolume`): flags/region/depth, tunnel
    // and cavern shapes, overhangs, sizes (tunnel radius, cavern, cover, layer mm).
    volume: array<vec4<i32>, 4>,
}

// Octave kinds (`landform::CONTINENT` ...).
const LF_CONTINENT: u32 = 0u;
const LF_REGION: u32 = 1u;
const LF_RIDGE: u32 = 2u;
const LF_HILLS: u32 = 3u;
const LF_WARP: u32 = 4u;
const LF_EROSION: u32 = 8u;
const LF_CRATER: u32 = 9u;
const LF_BASIN: u32 = 10u;
// Layer kinds (`StackLayer::CONTINENTS` ...).
const LAYER_CONTINENTS: u32 = 2u;
const LAYER_MOUNTAINS: u32 = 3u;
const LAYER_HILLS: u32 = 4u;
const LAYER_CRATERS: u32 = 7u;
const LAYER_BASINS: u32 = 8u;
const LAYER_PLATEAU: u32 = 9u;

const SEED_CAVE_REGION: u32 = 0xA511E9B3u;
const SEED_TUNNEL_A: u32 = 0x63D83595u;
const SEED_TUNNEL_B: u32 = 0x2B1F4C7Au;
const SEED_CAVERN: u32 = 0x9E3779B1u;
const SEED_OVERHANG: u32 = 0x7F4A7C15u;
const SEED_OVERHANG_REGION: u32 = 0x4CF5AD43u;

fn landform_seed() -> u32 { return bitcast<u32>(terrain.header.w); }

// Tunnels and caverns resolved at `level` (`LandformVolume::caves_at`).
fn landform_caves_at(level: u32) -> vec2<bool> {
    let v = terrain.volume;
    if (v[0].x & 1) == 0 { return vec2<bool>(false); }
    let layer = v[3].w;
    // Covered caverns also need a cell within their cover (`caves_at`).
    let cavern = select(v[3].y, min(v[3].y, v[3].z), v[3].z > 0);
    return vec2<bool>((v[3].x >> level) >= layer, (cavern >> level) >= layer);
}

fn landform_cave_region(p: vec3<i32>) -> bool {
    let v = terrain.volume;
    return noise(p, u32(v[0].y), landform_seed() ^ SEED_CAVE_REGION) > v[0].z;
}

fn landform_overhang_amplitude(p: vec3<i32>, level: u32) -> i32 {
    let v = terrain.volume;
    if (v[0].x & 2) == 0 { return 0; }
    let n = noise(p, u32(v[2].z), landform_seed() ^ SEED_OVERHANG_REGION);
    let ramp = clamp((n - v[2].w) * 4, 0, NOISE_ONE);
    let a = scale_q16(ramp, v[2].x);
    if (a >> level) < 2 * v[3].w { return 0; }
    return a;
}

fn terrain_extent(p: vec3<i32>, level: u32) -> vec2<i32> {
    let v = terrain.volume;
    let layer = v[3].w;
    let caves = landform_caves_at(level);
    var below = 0;
    if (caves.x || caves.y) && landform_cave_region(p) {
        below = ((v[0].w / layer) >> level) + 2;
    }
    let a = landform_overhang_amplitude(p, level);
    var above = 0;
    if a > 0 {
        above = ((a / layer) >> level) + 2;
        below = max(below, above);
    }
    return vec2<i32>(below, above);
}

// mm per noise unit at lattice shift s: 2^s * 12.5 mm / 2^17 (`density`).
fn landform_noise_mm(excess: i32, shift: i32) -> i32 {
    return mul_shr_signed(excess, 25 << u32(shift), 18u);
}

// Signed distance of a cell to the surface, CSG of the surface, overhangs
// and caves (`LandformVolume::density`).
fn terrain_density(p: vec3<i32>, q: vec3<i32>, level: u32, top: i32, k: i32) -> i32 {
    let v = terrain.volume;
    let a = landform_overhang_amplitude(p, level);
    let caves = landform_caves_at(level);
    if a == 0 && !caves.x && !caves.y { return heightfield_density(top, k); }
    let layer = v[3].w;
    let cell = layer << level;
    let d = (k - top) * cell + cell / 2;
    var solid = k < top;
    var f = -d;
    if a > 0 {
        let s = scale_q16(noise(q, u32(v[2].y), landform_seed() ^ SEED_OVERHANG), a);
        solid = d < s;
        f = s - d;
    }
    if caves.x || caves.y {
        let region = noise(p, u32(v[0].y), landform_seed() ^ SEED_CAVE_REGION);
        let depth = (top - k) * cell - cell / 2;
        let outside = max(max(landform_noise_mm(v[0].z - region, v[0].y), d), depth - v[0].w);
        var walls = 0x7fffffff;
        var carved = false;
        if caves.x {
            let w = v[1].y;
            let na = abs(noise(q, u32(v[1].x), landform_seed() ^ SEED_TUNNEL_A));
            let nb = abs(noise(q, u32(v[1].x), landform_seed() ^ SEED_TUNNEL_B));
            walls = min(walls, landform_noise_mm(max(na, nb) - w, v[1].x));
            carved = carved || (na < w && nb < w);
        }
        if caves.y {
            let n = noise(q, u32(v[1].z), landform_seed() ^ SEED_CAVERN);
            walls = min(walls, max(landform_noise_mm(v[1].w - n, v[1].z), v[3].z - depth));
            carved = carved || (depth >= v[3].z && n > v[1].w);
        }
        f = min(f, max(outside, walls));
        if solid && k < top && region > v[0].z && depth <= v[0].w && carved { solid = false; }
    }
    let density = clamp(f, -(1 << 22u), 1 << 22u) * 256 / max(cell, 1);
    if solid { return max(density, 1); }
    return min(density, 0);
}

// Masks are always evaluated; other detail finer than about four level
// cells is omitted (`landform::resolved`).
fn landform_resolved(o: LandformOctave, level: u32) -> bool {
    let kind = o.kind & 0xffu;
    return kind <= LF_REGION || kind == LF_BASIN || o.shift >= level + 5u; // landform::RESOLVED_SHIFT
}

fn landform_layer_count() -> u32 { return min(u32(max(terrain.stack.x, 0)), 8u); }

override RIDGE_DISPLAY_GENERATION: bool = false;

fn ridge_suffix_knot(row: u32, knot: u32) -> i32 {
    let index = row * 33u + knot;
    return terrain.ridge_suffix[index >> 2u][index & 3u];
}

fn ridge_suffix_mean(row: u32, incoming: i32) -> i32 {
    let weight = clamp(incoming, FINE_ONE / 4, FINE_ONE - 1);
    let delta = weight - FINE_ONE / 4;
    // (.75 * 2^24)/32 = 3*2^17 exactly. Last interval reaches weight1.
    let step = 393216;
    let knot = min(u32(delta / step), 31u);
    let remainder = delta - i32(knot) * step;
    let fraction = (remainder << 7u) / 3;
    let lo = ridge_suffix_knot(row, knot);
    let hi = ridge_suffix_knot(row, knot + 1u);
    return lo + mul_fine(hi - lo, fraction);
}

fn ridge_display_support(shift: u32, effective_level: u32) -> i32 {
    if shift >= effective_level + 5u { return FINE_ONE; }
    if shift == effective_level + 4u { return FINE_ONE / 2; }
    return 0;
}

const LANDFORM_GRAD_SHIFT: u32 = 18u; // landform::GRAD_SHIFT
const LANDFORM_STRIPES: i32 = 2;

fn landform_per_span(d: i32, shift: u32) -> i32 {
    if shift >= LANDFORM_GRAD_SHIFT { return d >> (shift - LANDFORM_GRAD_SHIFT); }
    return d << (LANDFORM_GRAD_SHIFT - shift);
}

fn landform_per_span3(d: vec3<i32>, shift: u32) -> vec3<i32> {
    return vec3<i32>(landform_per_span(d.x, shift), landform_per_span(d.y, shift), landform_per_span(d.z, shift));
}

fn mul_fine3(v: vec3<i32>, b: i32) -> vec3<i32> {
    return vec3<i32>(mul_fine(v.x, b), mul_fine(v.y, b), mul_fine(v.z, b));
}

// Q24 weight of a clamp's derivative inside (lo, hi) (`soft_inside`).
fn soft_inside(x: i32, lo: i32, hi: i32) -> i32 {
    return min(clamp((x - lo) * 8, 0, FINE_ONE), clamp((hi - x) * 8, 0, FINE_ONE));
}

// Continents base (`continent_base`).
fn landform_base(c: i32, floor: i32, lowland: i32) -> i32 {
    if c < 0 {
        let t = min(-c, FINE_ONE);
        return mul_fine(floor, t) + mul_fine(lowland / 8, FINE_ONE - t);
    }
    return mul_fine(lowland, min(c * 2, FINE_ONE));
}

// Land weight and deep-sea fade (`land_masks`).
fn landform_land(c: i32) -> vec2<i32> {
    return vec2<i32>(clamp(c * 3, 0, FINE_ONE), clamp(FINE_ONE + c * 2, FINE_ONE / 8, FINE_ONE));
}

fn landform_masked(mask: u32, x: i32, lw: vec2<i32>) -> i32 {
    if mask == 1u { return mul_fine(x, lw.x); }
    if mask == 2u { return mul_fine(x, lw.y); }
    return x;
}

// Per-layer accumulators (`Accumulators`), reset by every height evaluation.
var<private> lf_value: array<i32, 8>;
var<private> lf_region: array<i32, 8>;
var<private> lf_weight: array<i32, 8>;
var<private> lf_dvalue: array<vec3<i32>, 8>;
var<private> lf_dregion: array<vec3<i32>, 8>;
var<private> lf_dweight: array<vec3<i32>, 8>;

// Continents value and gradient, FINE_ONE (all land) without them.
fn landform_continent() -> vec4<i32> {
    let ci = terrain.levels.y - 1;
    if ci < 0 { return vec4<i32>(FINE_ONE, 0, 0, 0); }
    let l = u32(ci) & 7u;
    return vec4<i32>(lf_value[l], lf_dvalue[l]);
}

// Steering gradient of the continents and mountains w.r.t. the warped point
// (`steering`).
fn landform_steering() -> vec3<i32> {
    let cd = landform_continent();
    let c = cd.x;
    let dc = cd.yzw;
    let lw = landform_land(c);
    let dland = mul_fine3(dc * 3, soft_inside(c * 3, 0, FINE_ONE));
    let dwet = mul_fine3(dc * 2, soft_inside(FINE_ONE + c * 2, FINE_ONE / 8, FINE_ONE));
    var g = vec3<i32>(0);
    for (var l = 0u; l < landform_layer_count(); l++) {
        let layer = terrain.layers[l];
        var x = 0;
        var dx = vec3<i32>(0);
        if layer.kind == LAYER_CONTINENTS {
            if c < 0 {
                dx = mul_fine3(mul_fine3(-dc, soft_inside(-c, 0, FINE_ONE)), layer.a - layer.b / 8);
            } else {
                dx = mul_fine3(mul_fine3(dc * 2, soft_inside(c * 2, 0, FINE_ONE)), layer.b);
            }
            x = landform_base(c, layer.a, layer.b);
        } else if layer.kind == LAYER_MOUNTAINS {
            let m = (lf_region[l] - (layer.a << 8u)) * 3;
            let region = clamp(m, 0, FINE_ONE);
            let dregion = mul_fine3(lf_dregion[l] * 3, soft_inside(m, 0, FINE_ONE));
            x = mul_fine(lf_value[l], region);
            dx = mul_fine3(lf_dvalue[l], region) + mul_fine3(dregion, lf_value[l]);
        } else {
            continue;
        }
        if layer.mask == 1u {
            g += mul_fine3(dx, lw.x) + mul_fine3(dland, x);
        } else if layer.mask == 2u {
            g += mul_fine3(dx, lw.y) + mul_fine3(dwet, x);
        } else {
            g += dx;
        }
    }
    return g;
}

fn trilerp_q16(v: array<i32, 8>, w: vec3<i32>) -> i32 {
    let y0 = lerp_q16(lerp_q16(v[0], v[1], w.x), lerp_q16(v[2], v[3], w.x), w.y);
    let y1 = lerp_q16(lerp_q16(v[4], v[5], w.x), lerp_q16(v[6], v[7], w.x), w.y);
    return lerp_q16(y0, y1, w.z);
}

fn landform_lattice_q16(v: i32, mask: i32, s: u32) -> i32 {
    if s >= 16u { return (v & mask) >> (s - 16u); }
    return (v & mask) << (16u - s);
}

// One erosion octave (`erosion_octave`): `.x` height, `.yzw` gradient.
fn landform_erosion(saturation_slope: i32, o: LandformOctave, p: vec3<i32>, g: vec3<i32>, up: vec3<i32>) -> vec4<i32> {
    let t = vec3<i32>(
        mul_shr_signed(up.y, g.z, 30u) - mul_shr_signed(up.z, g.y, 30u),
        mul_shr_signed(up.z, g.x, 30u) - mul_shr_signed(up.x, g.z, 30u),
        mul_shr_signed(up.x, g.y, 30u) - mul_shr_signed(up.y, g.x, 30u));
    let tn = unit_q30(t);
    if all(tn == vec3<i32>(0)) { return vec4<i32>(0); }
    let slope = u32(mul_shr_signed(t.x, tn.x, 30u) + mul_shr_signed(t.y, tn.y, 30u) + mul_shr_signed(t.z, tn.z, 30u));
    let saturation = u32(max(saturation_slope, 1));
    let strength = i32(((min(slope, saturation) >> 5u) << 16u) / max(saturation >> 5u, 1u));
    let s = o.shift;
    let c = p >> vec3<u32>(s);
    let mask = (1 << s) - 1;
    let w = vec3<i32>(fade(landform_lattice_q16(p.x, mask, s)), fade(landform_lattice_q16(p.y, mask, s)), fade(landform_lattice_q16(p.z, mask, s)));
    var cosv: array<i32, 8>;
    var sinv: array<i32, 8>;
    for (var index = 0; index < 8; index++) {
        let corner = c + vec3<i32>(index & 1, (index >> 1u) & 1, index >> 2u);
        let d = p - (corner << vec3<u32>(s));
        let along = mul_shr_signed(d.x, tn.x, s + 14u) + mul_shr_signed(d.y, tn.y, s + 14u) + mul_shr_signed(d.z, tn.z, s + 14u);
        let phase = along * LANDFORM_STRIPES + i32(hash3(corner.x, corner.y, corner.z, o.seed) & 0xffffu);
        cosv[index] = mul16(abs(sin_turns((phase >> 1u) + 16384)), 102944) - NOISE_ONE;
        sinv[index] = sin_turns(phase);
    }
    let value = scale_q16(mul16(trilerp_q16(cosv, w), strength), o.amplitude);
    let m = scale_q16(mul16(trilerp_q16(sinv, w), strength), o.amplitude);
    let magnitude = landform_per_span(-((m * 3217) >> 8u), s);
    return vec4<i32>(value, mul_shr_signed(magnitude, tn.x, 30u), mul_shr_signed(magnitude, tn.y, 30u), mul_shr_signed(magnitude, tn.z, 30u));
}

const LANDFORM_CRATER_RADIUS: i32 = 157286; // landform::CRATER_RADIUS_Q19
const LANDFORM_REACH: i32 = 67108864; // squared ejecta reach in crater radii (Q24)

// 2^60 / d for d in [2^29, 2^30) (`recip_q30`).
fn landform_recip_q30(d: u32) -> u32 {
    var y = 3031741621u - mul_shr(d, 2021161080u, 30u);
    for (var step = 0u; step < 4u; step++) {
        y = mul_shr(y, 2u * Q30 - mul_shr(d, y, 30u), 30u);
    }
    return y;
}

// r2 / rad2 in Q24 (`squared_ratio`).
fn landform_squared_ratio(r2: u32, rad2: u32) -> u32 {
    let bits = 32u - countLeadingZeros(rad2);
    return mul_shr(r2, landform_recip_q30(rad2 << (30u - bits)), 6u + bits);
}

// Polynomial smooth minimum over a width `k` (`smooth_min`).
fn landform_smooth_min(a: i32, b: i32, k: i32) -> i32 {
    let m = min(a, b);
    let gap = abs(a - b);
    if k <= 0 || gap >= k { return m; }
    let n = (k - gap) << 8u;
    let q = ((n / k) << 8u) + ((n % k) << 8u) / k;
    return m - scale_q16(mul16(q, q), k) / 4;
}

// One crater octave (`crater_octave`): height (mm), freshest ejecta.
fn landform_crater(layer: StackLayer, o: LandformOctave, p: vec3<i32>, up: vec3<i32>) -> vec2<i32> {
    let s = o.shift;
    let density = o.seed & 0xffffu;
    let radius_domain = u32(terrain.stack.z);
    let c0 = p >> vec3<u32>(s);
    var height = 0;
    var ejecta = 0u;
    for (var index = 0; index < 27; index++) {
        let cell = c0 + vec3<i32>(index % 3 - 1, (index / 3) % 3 - 1, index / 9 - 1);
        let a = hash3(cell.x, cell.y, cell.z, o.seed);
        if (a & 0xffffu) >= density { continue; }
        let b = hash3(cell.x, cell.y, cell.z, o.seed ^ 0x6C8E9CF5u);
        let jitter = vec3<i32>(i32(b & 1023u), i32((b >> 10u) & 1023u), i32((b >> 20u) & 1023u));
        let centre = (cell << vec3<u32>(s)) + (jitter << vec3<u32>(s - 10u));
        // Only centres within half a cell of the surface.
        if terrain.shape.w != 0 {
            if abs(centre.y) >= (1 << (s - 1u)) { continue; }
        } else {
            let ac = vec3<u32>(abs(centre));
            let len2 = mul_shr(ac.x, ac.x, 30u) + mul_shr(ac.y, ac.y, 30u) + mul_shr(ac.z, ac.z, 30u);
            let r2 = mul_shr(radius_domain, radius_domain, 30u);
            if u32(abs(i32(len2) - i32(r2))) >= (radius_domain >> (30u - s)) { continue; }
        }
        // Squared horizontal distance in Q28 lattice cells from Q(19 + e)
        // offsets, exact for coarse lattices (`crater_octave`).
        let e = select(0u, s - 19u, s >= 19u);
        var d = centre - p;
        if s < 19u { d = d << vec3<u32>(19u - s); }
        if max(max(abs(d.x), abs(d.y)), abs(d.z)) >= (1 << (19u + e)) { continue; }
        let along = mul_shr_signed(d.x, up.x, 20u + e) + mul_shr_signed(d.y, up.y, 20u + e) + mul_shr_signed(d.z, up.z, 20u + e);
        let ad = vec3<u32>(abs(d));
        let len2 = mul_shr(ad.x, ad.x, 10u + 2u * e) + mul_shr(ad.y, ad.y, 10u + 2u * e) + mul_shr(ad.z, ad.z, 10u + 2u * e);
        let aa = mul_shr(u32(abs(along)), u32(abs(along)), 30u);
        let r2 = select(0u, len2 - aa, len2 > aa);
        let size = 36045 + i32(((a >> 16u) * 29491u) >> 16u);
        let radius = max(mul16(LANDFORM_CRATER_RADIUS, size), 1);
        let rad2 = max(mul_shr(u32(radius), u32(radius), 10u), 1u);
        if r2 >= 4u * rad2 { continue; }
        let x2 = i32(landform_squared_ratio(r2, rad2));
        let depth = scale_q16(size, o.amplitude);
        let rim = scale_q16(layer.a, depth);
        let bowl = mul_fine(x2, depth + rim) - depth;
        let t = (LANDFORM_REACH - x2) / 3;
        let ejecta_height = mul_fine(mul_fine(mul_fine(t, t), t), rim);
        height += landform_smooth_min(bowl, ejecta_height, rim / 4);
        if (hash3(cell.x, cell.y, cell.z, o.seed ^ 0x1B873593u) & 0xffffu) < u32(layer.b) {
            ejecta = max(ejecta, 255u - u32((x2 >> 16u) * 255 / (LANDFORM_REACH >> 16u)));
        }
    }
    return vec2<i32>(height, i32(ejecta));
}

fn landform_noise(p: vec3<i32>, o: LandformOctave, gradient: bool) -> vec4<i32> {
    if gradient { return noise_fine_grad(p, o.shift, o.seed); }
    return vec4<i32>(noise_fine(p, o.shift, o.seed), 0, 0, 0);
}

// One ridge octave (noise `nd`) into layer `l`'s chain (`height_parts`),
// weighted by `gain`; `tracked` carries gradients.
fn landform_ridge(l: u32, o: LandformOctave, nd: vec4<i32>, tracked: bool, gain: i32) {
    let n = nd.x;
    let r = clamp(FINE_ONE - abs(n), 0, FINE_ONE - 1);
    let rr = mul_fine(r, r);
    let v = mul_fine(rr, lf_weight[l]);
    if tracked {
        // The crest's sign flip is softened over |n| < 1/4.
        let dr = mul_fine3(landform_per_span3(-nd.yzw, o.shift), clamp(n * 4, -FINE_ONE, FINE_ONE));
        let dv = mul_fine3(mul_fine3(dr, r) * 2, lf_weight[l]) + mul_fine3(lf_dweight[l], rr);
        lf_dweight[l] = mul_fine3(dv * 2, soft_inside(v * 2, FINE_ONE / 4, FINE_ONE - 1));
        lf_dvalue[l] += mul_fine3(dv, o.amplitude);
    }
    lf_weight[l] = clamp(v * 2, FINE_ONE / 4, FINE_ONE - 1);
    if gain == FINE_ONE {
        lf_value[l] += mul_fine(o.amplitude, v);
    } else {
        lf_value[l] += mul_fine(mul_fine(o.amplitude, v), gain);
    }
}

// Height and surface word (`height_parts`). The false mode preserves the
// canonical operation order. Only generation compiles the display
// capability; climate, shade and verify_field stay exact.
//
// Compilers inline every call: each octave reaches one fine-noise site and
// one ridge site, and generation evaluates this once per column
// (`terrain_column`). Three inlined copies took 22 s to compile.
fn terrain_parts_mode(p: vec3<i32>, level: u32, display: bool) -> vec2<i32> {
    let count = min(u32(max(terrain.header.x, 0)), 48u);
    // Erosion follows the larger terrain's slope (`height_parts`).
    var erosion = false;
    for (var index = LANDFORM_WARP; index < count; index++) {
        let o = terrain.octaves[index];
        if (o.kind & 0xffu) == LF_EROSION && landform_resolved(o, level) { erosion = true; }
    }
    var warp = vec3<i32>(0);
    var jx = vec3<i32>(0);
    var jy = vec3<i32>(0);
    var jz = vec3<i32>(0);
    let warps = min(u32(max(terrain.stack.w, 0)), LANDFORM_WARP);
    for (var index = 0u; index < warps; index++) {
        let o = terrain.octaves[index];
        let nd = landform_noise(p, o, erosion);
        let n = mul_fine(o.amplitude, nd.x);
        var jac = vec3<i32>(0);
        if erosion {
            let m = mul_fine3(nd.yzw, o.amplitude);
            if o.shift <= 24u { jac = m << vec3<u32>(24u - o.shift); } else { jac = m >> vec3<u32>(o.shift - 24u); }
        }
        let axis = min(o.kind - LF_WARP, 2u);
        if axis == 0u { warp.x += n; jx += jac; } else if axis == 1u { warp.y += n; jy += jac; } else { warp.z += n; jz += jac; }
    }
    let q = p + warp;
    var up = vec3<i32>(0, 1073741824, 0);
    if terrain.shape.w == 0 { up = unit_q30(p); }
    for (var l = 0u; l < 8u; l++) {
        lf_value[l] = 0;
        lf_region[l] = 0;
        lf_weight[l] = FINE_ONE - 1;
        lf_dvalue[l] = vec3<i32>(0);
        lf_dregion[l] = vec3<i32>(0);
        lf_dweight[l] = vec3<i32>(0);
    }
    var eroded = 0;
    var deroded = vec3<i32>(0);
    var ejecta = 0;
    var display_tag = 0xffffffffu;
    if terrain.shape.x > 0 { display_tag = (u32(terrain.shape.x - 1) << 8u) | LF_RIDGE; }
    var ridge_gain = FINE_ONE;
    var ridge_row = 0u;
    for (var index = LANDFORM_WARP; index < count; index++) {
        let o = terrain.octaves[index];
        let l = (o.kind >> 8u) & 7u;
        let kind = o.kind & 0xffu;
        var gain = FINE_ONE;
        var gradient = erosion;
        if RIDGE_DISPLAY_GENERATION && display && o.kind == display_tag {
            // Expand B[k](w)=(1-s)*F[k](w)+s*(A[k]*v+B[k+1](next(w))).
            // Repeated lattice shifts may have several partial octaves.
            var partial = ridge_gain;
            if ridge_gain != 0 {
                let support = ridge_display_support(o.shift, level);
                if support != FINE_ONE {
                    let mean_gain = mul_fine(ridge_gain, FINE_ONE - support);
                    lf_value[l] += mul_fine(ridge_suffix_mean(ridge_row, lf_weight[l]), mean_gain);
                    ridge_gain = mul_fine(ridge_gain, support);
                }
                partial = ridge_gain;
            }
            ridge_row += 1u;
            if partial == 0 { continue; }
            // Resolved erosion octaves precede every partial ridge: only
            // fully supported ridges carry gradients.
            gain = ridge_gain;
            gradient = erosion && ridge_gain == FINE_ONE;
        } else if !landform_resolved(o, level) {
            continue;
        }
        if kind <= LF_HILLS || kind == LF_BASIN {
            // The octave's one fine-noise evaluation (basins: unwarped).
            let nd = landform_noise(select(q, p, kind == LF_BASIN), o, gradient && kind <= LF_RIDGE);
            if kind == LF_RIDGE {
                landform_ridge(l, o, nd, gradient, gain);
            } else if kind <= LF_REGION {
                let v = mul_fine(nd.x, o.amplitude << 8u);
                let dv = mul_fine3(landform_per_span3(nd.yzw, o.shift), o.amplitude << 8u);
                if kind == LF_CONTINENT { lf_value[l] += v; lf_dvalue[l] += dv; } else { lf_region[l] += v; lf_dregion[l] += dv; }
            } else if kind == LF_HILLS {
                lf_value[l] += mul_fine(o.amplitude, nd.x);
            } else {
                lf_value[l] += nd.x;
            }
        } else if kind == LF_EROSION {
            let gq = landform_steering();
            // (I + J)^T: column j gathers every warp axis' dependence on p_j.
            let gp = gq + vec3<i32>(
                mul_fine(gq.x, jx.x) + mul_fine(gq.y, jy.x) + mul_fine(gq.z, jz.x),
                mul_fine(gq.x, jx.y) + mul_fine(gq.y, jy.y) + mul_fine(gq.z, jz.y),
                mul_fine(gq.x, jx.z) + mul_fine(gq.y, jy.z) + mul_fine(gq.z, jz.z));
            let e = landform_erosion(terrain.layers[l].a, o, p, gp + deroded, up);
            lf_value[l] += e.x;
            eroded += e.x;
            deroded += e.yzw;
        } else if kind == LF_CRATER {
            let c = landform_crater(terrain.layers[l], o, p, up);
            lf_value[l] += c.x;
            ejecta = max(ejecta, c.y);
        } else {
            lf_value[l] += scale_q16(noise(p, o.shift, o.seed), o.amplitude);
        }
    }
    // Compose the layers in stack order.
    let c = landform_continent().x;
    let lw = landform_land(c);
    var h = 0;
    var basin = 0;
    for (var l = 0u; l < landform_layer_count(); l++) {
        let layer = terrain.layers[l];
        var x = 0;
        if layer.kind == LAYER_CONTINENTS {
            x = landform_base(c, layer.a, layer.b);
        } else if layer.kind == LAYER_MOUNTAINS {
            x = mul_fine(lf_value[l], clamp((lf_region[l] - (layer.a << 8u)) * 3, 0, FINE_ONE));
        } else if layer.kind == LAYER_BASINS {
            // Basins flatten what lies below them in the stack by half and
            // sink it by their depth.
            let m = landform_masked(layer.mask, clamp((lf_value[l] - layer.b) * 4, 0, FINE_ONE), lw);
            basin = max(basin, m);
            h = mul_fine(h, FINE_ONE - m / 2) - mul_fine(layer.a, m);
            continue;
        } else if layer.kind == LAYER_PLATEAU {
            x = layer.a;
        } else if layer.kind >= LAYER_HILLS && layer.kind <= LAYER_CRATERS {
            x = lf_value[l];
        }
        h += landform_masked(layer.mask, x, lw);
    }
    var surface = 0;
    if terrain.style.x == 0 || terrain.style.x == 3 {
        surface = clamp(eroded * 127 / max(terrain.stack.y, 1), -127, 127) & 0xff;
    } else if terrain.style.x == 1 {
        surface = (ejecta >> 1u) | (select(0, 1, basin >= FINE_ONE / 2) << 7u);
    }
    landform_surface = surface;
    landform_surface_at = vec4<i32>(p, i32(level));
    return vec2<i32>(h, surface);
}

fn terrain_height(p: vec3<i32>, level: u32) -> i32 {
    return terrain_parts_mode(p, level, false).x;
}

// Surface word of the last height evaluated in this invocation: generation
// asks for it right after the column's height.
var<private> landform_surface: i32 = 0;
var<private> landform_surface_at: vec4<i32> = vec4<i32>(0x7fffffff);

// Surface word (`landform::surface`), by material style.
fn terrain_surface(p: vec3<i32>, level: u32, height: i32) -> u32 {
    if any(landform_surface_at != vec4<i32>(p, i32(level))) { _ = terrain_height(p, level); }
    return u32(landform_surface);
}

// Height and surface word of a column being generated (`display`: its
// coarse ridges keep their envelope).
fn terrain_column(p: vec3<i32>, level: u32, display: bool) -> vec2<i32> {
    return terrain_parts_mode(p, level, display && terrain.shape.z != 0);
}

fn landform_moisture(p: vec3<i32>) -> i32 {
    return (noise_fine(p, u32(max(terrain.levels.x, 1)), bitcast<u32>(terrain.header.w) ^ 0x51ED270Bu) + FINE_ONE) / 2;
}

// Strata altitude (mm): layers undulate +-8 m over ~100 m, so cuts through
// them never show flat rings (`strata` in landform.rs).
fn landform_strata(p: vec3<i32>, altitude: i32) -> i32 {
    return altitude + scale_q16(noise(p, 13u, bitcast<u32>(terrain.header.w) ^ 0x9B05688Cu), 8000);
}

// Material noise is defined in the fixed 1.25 cm domain unit.
// Four samples per lattice spacing retain contrast; below two samples the
// unresolved octave contributes its coverage instead of aliased class noise.
fn material_noise_support(shift: u32, pixel: f32) -> f32 {
    if pixel <= 0.0 { return 1.0; }
    let wavelength = 0.0125 * f32(1u << shift);
    return 1.0 - smoothstep(wavelength * 0.25, wavelength * 0.5, pixel);
}

// Empirical CDF of the project's integer noise on cube-face/plane slices:
// 1.2M samples, four seeds, three axes. The normalized two-octave sum agrees
// within 0.0035 coverage at these knots; this is an appearance approximation.
// Same measured noise distribution as snow, plus its piecewise-linear
// integral. This permits a box convolution without new noise samples.
const MATERIAL_NOISE_CDF = array<f32, 33>(0.00000000, 0.00000250, 0.00002000, 0.00006917, 0.00032333, 0.00097500, 0.00300250, 0.00770250, 0.01944750, 0.04330750, 0.07070667, 0.10610750, 0.16352167, 0.23430250, 0.31787500, 0.40675833, 0.50058917, 0.59367667, 0.68297667, 0.76579917, 0.83619500, 0.89390500, 0.92928833, 0.95633250, 0.98046750, 0.99220417, 0.99696000, 0.99898750, 0.99968417, 0.99994000, 0.99998250, 0.99999750, 1.00000000);
const MATERIAL_NOISE_AREA = array<f32, 33>(0.0000000000, 0.0000012500, 0.0000125000, 0.0000570850, 0.0002533350, 0.0009025000, 0.0028912500, 0.0082437500, 0.0218187500, 0.0531962500, 0.1102033350, 0.1986104200, 0.3334250050, 0.5323370900, 0.8084258400, 1.1707425050, 1.6244162550, 2.1715491750, 2.8098758450, 3.5342637650, 4.3352608500, 5.2003108500, 6.1119075150, 7.0547179300, 8.0231179300, 9.0094537650, 10.0040358500, 11.0020096000, 12.0013454350, 13.0011575200, 14.0011187700, 15.0011087700, 16.0011075200);

fn material_noise_integral(value: f32) -> f32 {
    let x = clamp(value / 4096.0 + 16.0, 0.0, 32.0);
    let i = min(u32(x), 31u);
    let t = x - f32(i);
    return 4096.0 * (MATERIAL_NOISE_AREA[i] + MATERIAL_NOISE_CDF[i] * t
        + 0.5 * (MATERIAL_NOISE_CDF[i + 1u] - MATERIAL_NOISE_CDF[i]) * t * t)
        + max(value - 65536.0, 0.0);
}

// Integral of the unresolved phase CDF, conditioned on natural rock exposure.
// The CDF is an appearance approximation; canonical integer IDs never use it.
fn rock_phase_integral(value: f32, noise_scale: f32, cutoff: f32, cut_cdf: f32, cutoff_integral: f32) -> f32 {
    let x = max(value / noise_scale, cutoff);
    return noise_scale * max(material_noise_integral(x) - cutoff_integral
        - cut_cdf * (x - cutoff), 0.0) / (1.0 - cut_cdf);
}

// Stable integration in the rare upper CDF tail. Its tiny probability cannot
// be recovered by subtracting the two large integrated CDF values in f32.
fn weathered_tail_interval(index: u32, cut: f32, lo: f32, hi: f32) -> vec2<f32> {
    let left = max(-65536.0 + f32(index) * 4096.0, cut);
    let right = -65536.0 + f32(index + 1u) * 4096.0;
    if left >= right { return vec2<f32>(0.0); }
    let density = MATERIAL_NOISE_CDF[index + 1u] - MATERIAL_NOISE_CDF[index];
    let start = max(left, lo);
    let end = min(right, hi);
    let linear = max(end - start, 0.0) * 0.5
        * (clamp((start - lo) / (hi - lo), 0.0, 1.0) + clamp((end - lo) / (hi - lo), 0.0, 1.0));
    let saturated = max(right - max(left, hi), 0.0);
    return density * vec2<f32>(linear + saturated, right - left);
}
// A restrained world-space coating on exposed natural rock, independent of
// geological IDs. Reuse the outcrop samples; unresolved noise contributes the
// conditional mean of a linear weathering ramp instead of altitude contours.
fn weathered_stone_coverage(resolved: f32, deviation: f32, cutoff: f32) -> f32 {
    let lower = -16384.0;
    let upper = 16384.0;
    var mean = clamp((resolved - lower) / (upper - lower), 0.0, 1.0);
    if deviation > 0.0 {
        let lo = (lower - resolved) / deviation;
        let hi = (upper - resolved) / deviation;
        if hi <= max(cutoff, -65536.0) { return 0.65; }
        if lo >= 65536.0 { return 0.35; }
        let cut = clamp(cutoff, -65536.0, 65536.0);
        if cut >= 49152.0 && cut < 65536.0 {
            let tail = weathered_tail_interval(28u, cut, lo, hi)
                + weathered_tail_interval(29u, cut, lo, hi)
                + weathered_tail_interval(30u, cut, lo, hi)
                + weathered_tail_interval(31u, cut, lo, hi);
            return 0.35 + 0.30 * clamp(tail.x / tail.y, 0.0, 1.0);
        }
        let cut_cdf = material_noise_cdf(cut);
        if cut_cdf < 1.0 {
            let cut_integral = material_noise_integral(cut);
            let area = rock_phase_integral(upper - resolved, deviation, cut, cut_cdf, cut_integral)
                - rock_phase_integral(lower - resolved, deviation, cut, cut_cdf, cut_integral);
            mean = clamp(1.0 - area / (upper - lower), 0.0, 1.0);
        }
    }
    return 0.35 + 0.30 * mean;
}

fn rock_box_cdf(value: f32, left: f32, width: f32,
    noise_scale: f32, cutoff: f32, cut_cdf: f32, cutoff_integral: f32) -> f32 {
    if noise_scale <= 0.0 { return clamp((value - left) / width, 0.0, 1.0); }
    let width_q16 = width / noise_scale;
    let hi = (value - left) / noise_scale;
    let lo = hi - width_q16;
    // Saturated endpoints need no CDF primitive, including wide boxes.
    if hi <= max(cutoff, -65536.0) { return 0.0; }
    if lo >= 65536.0 { return 1.0; }
    if width_q16 <= 4096.0 {
        // A narrow box meets at most one CDF knot. Integrate its local
        // trapezoids instead of cancelling two large cumulative areas.
        let a = max(lo, cutoff);
        let split = min((floor(a / 4096.0) + 1.0) * 4096.0, hi);
        let fa = max(material_noise_cdf(a) - cut_cdf, 0.0) / (1.0 - cut_cdf);
        let fs = max(material_noise_cdf(split) - cut_cdf, 0.0) / (1.0 - cut_cdf);
        let fh = max(material_noise_cdf(hi) - cut_cdf, 0.0) / (1.0 - cut_cdf);
        return clamp(0.5 * ((fa + fs) * (split - a) + (fs + fh) * (hi - split)) / width_q16, 0.0, 1.0);
    }
    return clamp((rock_phase_integral(value - left, noise_scale, cutoff, cut_cdf, cutoff_integral)
        - rock_phase_integral(value - left - width, noise_scale, cutoff, cut_cdf, cutoff_integral)) / width, 0.0, 1.0);
}

fn rock_band_coverage(phase: f32, span: f32, deviation: f32, cutoff: f32) -> f32 {
    let noise_scale = deviation * (3000.0 / 65536.0);
    if noise_scale <= 0.0 {
        if span <= 0.0 { return select(0.0, 1.0, fract(phase / 9000.0) < 0.5); }
        if span < 1.0 {
            // Centre on the nearest band boundary before adding a tiny span.
            // Avoid cancelling a sub-mm pixel against a 4.5 m global phase.
            let p = phase - floor(phase / 9000.0) * 9000.0;
            if p < 2250.0 || p >= 6750.0 {
                let edge = select(p, p - 9000.0, p >= 6750.0);
                return clamp(0.5 + edge / span, 0.0, 1.0);
            }
            return clamp(0.5 - (p - 4500.0) / span, 0.0, 1.0);
        }
        let lo = phase - span * 0.5;
        let hi = phase + span * 0.5;
        let lo_period = floor(lo / 9000.0);
        let hi_period = floor(hi / 9000.0);
        let measure = 4500.0 * (hi_period - lo_period)
            + min(hi - hi_period * 9000.0, 4500.0)
            - min(lo - lo_period * 9000.0, 4500.0);
        return clamp(measure / span, 0.0, 1.0);
    }
    let cut_cdf = material_noise_cdf(cutoff);
    if cut_cdf > 0.999 { return -1.0; }
    // The empirical noise has bounded support. Conditioning only truncates
    // its lower end; a support box inside one band has exact 0/1 coverage.
    let noise_low = max(cutoff, -65536.0);
    let radial_width = max(span, 0.0);
    if radial_width + noise_scale * (65536.0 - noise_low) < 4500.0 {
        let lower = phase + noise_scale * noise_low - radial_width * 0.5;
        let upper = phase + noise_scale * 65536.0 + radial_width * 0.5;
        let lower_band = floor(lower / 4500.0);
        if lower_band == floor(upper / 4500.0) {
            return select(0.0, 1.0, (i32(lower_band) & 1) == 0);
        }
    }
    if span <= 0.0 || (span < min(1.0, noise_scale * 65.536) && noise_scale > 0.0) {
        let first = i32(floor((phase - 4000.0) / 9000.0));
        var coverage = 0.0;
        for (var b = first; b < first + 3; b++) {
            let lo = max((f32(b) * 9000.0 - phase) / noise_scale, cutoff);
            let hi = max((f32(b) * 9000.0 + 4500.0 - phase) / noise_scale, cutoff);
            coverage += (material_noise_cdf(hi) - material_noise_cdf(lo)) / (1.0 - cut_cdf);
        }
        return clamp(coverage, 0.0, 1.0);
    }
    // Whole 9 m periods have exactly half light stone, for any noise phase.
    // Only a remainder below 9 m needs integration: bounded even at orbit.
    let whole = floor(span / 9000.0) * 9000.0;
    let remainder = max(span - whole, 0.0);
    if remainder < 0.01 && whole > 0.0 { return 0.5; }
    // A partial period differs from its half-coverage mean by at most
    // min(r, period-r)/2. Positive mixtures of noise phases, including
    // conditional ones, keep that bound. Skip integration below 1/1024
    // absolute coverage error; the near/resolved path stays exact.
    if min(remainder, 9000.0 - remainder) <= span * (1.0 / 512.0) { return 0.5; }
    let start = phase - span * 0.5;
    let left = start - floor(start / 9000.0) * 9000.0;
    let first = i32(floor((left - 4000.0) / 9000.0));
    var cutoff_integral = 0.0;
    if remainder / noise_scale > 4096.0 {
        // Identical for all eight endpoints; narrow boxes never use it.
        cutoff_integral = material_noise_integral(cutoff);
    }
    var coverage = 0.0;
    for (var b = first; b < first + 4; b++) {
        let lo = f32(b) * 9000.0;
        coverage += rock_box_cdf(lo + 4500.0, left, remainder, noise_scale, cutoff, cut_cdf, cutoff_integral)
            - rock_box_cdf(lo, left, remainder, noise_scale, cutoff, cut_cdf, cutoff_integral);
    }
    return clamp((whole * 0.5 + remainder * coverage) / span, 0.0, 1.0);
}

fn material_noise_cdf(value: f32) -> f32 {
    let cdf = MATERIAL_NOISE_CDF;
    let x = clamp(value / 65536.0 * 16.0 + 16.0, 0.0, 32.0);
    let i = min(u32(x), 31u);
    return mix(cdf[i], cdf[i + 1u], x - f32(i));
}

fn snow_material_coverage(resolved: f32, deviation: f32, threshold: f32,
    altitude: i32, dry: bool, depth: i32) -> vec4<f32> {
    let snow = material_noise_cdf((threshold - resolved) / deviation);
    let rock = 1.0 - snow;
    if dry { return vec4<f32>(snow, rock, 0.0, 0.0); }
    if material_weathered_skin {
        let stone = rock * weathered_stone_coverage(resolved, deviation, (threshold - resolved) / deviation);
        let dirt = select(0.0, rock * 0.125, depth < 1);
        let remaining = select(1.0, 0.875, depth < 1);
        return vec4<f32>(snow, stone * remaining, (rock - stone) * remaining, dirt);
    }
    // Integrate the existing alternating 4.5m stone bands too: replacing
    // unresolved snow with a single rock colour would bias their mean.
    // Radial support filters exposed rock conditioned on this snow threshold;
    // snow probability itself keeps the existing outcrop-noise coverage.
    let band_weight = smoothstep(1.125, 2.25, material_radial_span);
    var filtered_stone = -1.0;
    if band_weight > 0.0 {
        let phase = f32(rem_floor(altitude, 9000)) + resolved * (3000.0 / 65536.0);
        let coverage = rock_band_coverage(phase, material_radial_span * 1000.0,
            deviation, (threshold - resolved) / deviation);
        if coverage >= 0.0 { filtered_stone = rock * coverage; }
    }
    var stone = 0.0;
    if band_weight < 1.0 || filtered_stone < 0.0 {
        let mean_altitude = f32(altitude) + resolved * (3000.0 / 65536.0);
        let first = i32(floor((mean_altitude - 4000.0) / 4500.0));
        for (var b = first; b < first + 4; b++) {
            if (b & 1) != 0 { continue; }
            let lo = max(threshold, (f32(b) * 4500.0 - f32(altitude)) * (65536.0 / 3000.0));
            let hi = (f32(b + 1) * 4500.0 - f32(altitude)) * (65536.0 / 3000.0);
            if hi > lo {
                stone += material_noise_cdf((hi - resolved) / deviation)
                    - material_noise_cdf((lo - resolved) / deviation);
            }
        }
    }
    stone = clamp(stone, 0.0, rock);
    if filtered_stone >= 0.0 { stone = mix(stone, filtered_stone, band_weight); }
    let dirt = select(0.0, rock * 0.125, depth < 1);
    let remaining = select(1.0, 0.875, depth < 1);
    return vec4<f32>(snow, stone * remaining, (rock - stone) * remaining, dirt);
}

// Material of a solid ground cell by material style (`ground_material`).
fn ground_material(p: vec3<i32>, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
    material_mix = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    material_fleck_base = M_AIR;
    material_coverage = -1.0;
    material_coverage_ids = vec2<u32>(M_DARK_STONE, M_STONE);
    if terrain.style.x == 1 { return lunar_material(p, surface, depth, slope, layer); }
    if terrain.style.x == 3 { return rules_material(p, surface, top_height, depth, slope, layer); }
    if terrain.style.x == 2 {
        if depth == 0 { return u32(terrain.style.y); }
        if depth <= terrain.header.z { return u32(terrain.style.z); }
        return u32(terrain.style.w);
    }
    return earthlike_material(p, surface, top_height, depth, slope, layer);
}

// The first rule whose conditions hold, else the fallback (`rules_material`).
fn rules_material(p: vec3<i32>, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
    let wet = landform_moisture(p) >> 8u;
    let erosion = i32(surface << 24u) >> 24u;
    let altitude = layer * terrain.header.y;
    let seed = bitcast<u32>(terrain.header.w);
    let count = min(u32(max(terrain.materials.x, 0)), 16u);
    for (var index = 0u; index < count; index++) {
        let r = terrain.rules[index];
        if top_height < r.height_slope.x || top_height > r.height_slope.y
            || slope < r.height_slope.z || slope > r.height_slope.w
            || depth < r.depth_moisture.x || depth > r.depth_moisture.y
            || wet < r.depth_moisture.z || wet > r.depth_moisture.w
            || erosion < r.surface_bands.x || erosion > r.surface_bands.y { continue; }
        let salt = index * 0x9E3779B9u;
        if r.surface_bands.z > 0 && (div_floor(landform_strata(p, altitude), r.surface_bands.z) & 1) != r.surface_bands.w { continue; }
        if r.head.z > 0 && noise(p, u32(r.head.z), seed ^ 0x68E31DA4u ^ salt) <= r.head.w { continue; }
        if r.head.y > 0 {
            if i32(hash3(p.x, p.y, p.z ^ (layer * 0x9e37), 0x5f356495u ^ salt) & 255u) >= r.head.y { continue; }
            return u32(r.head.x) | select(0u, M_SPECK, depth == 0);
        }
        return u32(r.head.x);
    }
    return u32(terrain.materials.y);
}

// Lunar materials (`lunar_material`): 1 regolith, 2 mare, 3 ejecta, 4 rock,
// 5 basalt, 6 anorthosite.
fn lunar_material(p: vec3<i32>, surface: u32, depth: i32, slope: i32, layer: i32) -> u32 {
    let ejecta = (surface & 0x7fu) << 1u;
    let mare = (surface & 0x80u) != 0u;
    let h = hash3(p.x, p.y, p.z ^ (layer * 0x9e37), 0x2545F491u);
    if depth == 0 {
        if slope >= 12 || (slope >= 6 && (h & 3u) == 0u) { return 4u; }
        if ejecta > (h & 0xffu) { return 3u; }
        return select(1u, 2u, mare);
    }
    if depth < terrain.header.z { return select(1u, 2u, mare); }
    return select(6u, 5u, mare);
}

// Earthlike materials (`earthlike_material`).
fn earthlike_material(p: vec3<i32>, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
    let dirt = terrain.header.z;
    let steep = slope >= terrain.shape.y;
    let wet = landform_moisture(p);
    // Hash every domain axis: on a face one of them is nearly constant.
    let h = hash3(p.x, p.y, p.z ^ (layer * 0x9e37), 0x2545F491u);
    let altitude = layer * terrain.header.y;
    if top_height < terrain.levels.w {
        // Low basins: meadow with mud and sand patches (position hash only,
        // never aligned with height contours) over silt, gravel and stone.
        if depth == 0 {
            let s = hash3(p.x, p.y, p.z, 0x5f356495u);
            if (s & 15u) == 0u { return M_DIRT | M_SPECK; }
            if ((s >> 4u) & 31u) == 0u { return M_SAND | M_SPECK; }
            return M_GRASS;
        }
        if depth < dirt * 2 {
            return select(M_CLAY, M_GRAVEL, (h & 3u) == 0u);
        }
        return M_STONE;
    }
    let snowline = terrain.levels.z + mul_fine(terrain.levels.z / 4, wet - FINE_ONE / 2);
    // Alpine weight: 0 below the rockline, NOISE_ONE at the snowline.
    let rockline = snowline - terrain.levels.z / 3;
    let band = max(snowline - rockline, 256);
    let alpine = (clamp(top_height - rockline, 0, band) / 256) * NOISE_ONE / (band / 256);
    // Rock patches (~50 m and ~6 m octaves), also breaking up snow edges.
    // Noise is clamped to +-NOISE_ONE, so below the rockline on gentler slopes no
    // outcrop can reach the rock fringe: skip it (same result).
    let seed = bitcast<u32>(terrain.header.w);
    var outcrop = -NOISE_ONE;
    var resolved_outcrop = f32(outcrop);
    var outcrop_deviation = 0.0;
    var outcrop_support = 1.0;
    if alpine > 0 || slope >= 5 {
        // Preserve the original integer samples and canonical classification.
        let broad = noise(p, 12u, seed ^ 0x1B56C4E9u);
        let fine = noise(p, 9u, seed ^ 0x6A09E667u) / 3;
        outcrop = broad + fine;
        resolved_outcrop = f32(outcrop);
        if material_footprint > 1.6 {
            let broad_support = material_noise_support(12u, material_footprint);
            let fine_support = material_noise_support(9u, material_footprint);
            outcrop_support = min(broad_support, fine_support);
            if outcrop_support < 1.0 {
                resolved_outcrop = f32(broad) * broad_support + f32(fine) * fine_support;
                outcrop_deviation = sqrt((1.0 - broad_support * broad_support)
                    + (1.0 - fine_support * fine_support) / 9.0);
            }
            if outcrop_deviation > 0.0 && top_height > snowline && depth < dirt {
                let resolved = resolved_outcrop;
                let deviation = outcrop_deviation;
                // Match truncating integer division at negative slopes too.
                let q = 6 - slope;
                let threshold = select(f32((q - 1) * 8192) + 0.5, f32(q * 8192) - 0.5, q > 0);
                let dry = wet < FINE_ONE / 10 * 3 && !steep;
                material_mix_ids = vec4<u32>(M_SNOW, select(M_STONE, M_SAND, dry), M_DARK_STONE, M_DIRT);
                material_mix = snow_material_coverage(resolved, deviation, threshold, altitude, dry, depth);
            }
        }
    }
    // Snow does not hold on faces steeper than ~37 degrees: rock streaks the snowfields.
    if top_height > snowline && depth < dirt && slope + outcrop / 8192 < 6 { return M_SNOW; }
    if wet < FINE_ONE / 10 * 3 {
        if depth < dirt && !steep { return M_SAND; }
        let band = rem_floor(div_floor(landform_strata(p, altitude), 2100), 5);
        return select(M_SANDSTONE, M_CLAY, band == 1 || band == 3);
    }
    // Rock shows through the turf in the outcrop patches, which grow up the
    // alpine band below the snowline and on hillsides over ~32 degrees; scree and
    // bare soil fringe them. Deeper cells keep the strata.
    var exposed = outcrop + 2 * alpine - NOISE_ONE;
    if slope >= 5 { exposed += NOISE_ONE / 2; }
    let gully = i32(surface << 24u) >> 24u;
    if slope >= 2 { exposed += gully * (NOISE_ONE / 256); }
    if steep || (exposed > 0 && depth < dirt) {
        let rock = select(M_DARK_STONE, M_STONE, (div_floor(altitude + scale_q16(outcrop, 3000), 4500) & 1) == 0);
        if depth < 1 { material_fleck_base = rock; }
        // Four samples per 4.5 m band retain its resolved contrast, matching
        // the noise support gate. No coverage work on fully resolved rock.
        let band_weight = max(smoothstep(1.125, 2.25, material_radial_span), 1.0 - outcrop_support);
        // Snow/rock coverage already includes its conditioned stone bands;
        // the shader consumes that vector instead of this separate metadata.
        if material_weathered_skin && depth < dirt && material_mix.x < 0.0 {
            let exposure = f32(NOISE_ONE - 2 * alpine - select(0, NOISE_ONE / 2, slope >= 5)) + 0.5;
            let cutoff = select(-65536.0, (exposure - resolved_outcrop) / max(outcrop_deviation, 0.0001), !steep);
            material_coverage = weathered_stone_coverage(resolved_outcrop, outcrop_deviation, cutoff);
        } else if material_footprint > 0.0 && band_weight > 0.0 && material_mix.x < 0.0 {
            // Exposure conditions the phase distribution on non-steep patches.
            let exposure = f32(NOISE_ONE - 2 * alpine - select(0, NOISE_ONE / 2, slope >= 5)) + 0.5;
            let cutoff = select(-65536.0, (exposure - resolved_outcrop) / max(outcrop_deviation, 0.0001), !steep);
            let phase = f32(rem_floor(altitude, 9000)) + resolved_outcrop * (3000.0 / 65536.0);
            let coverage = rock_band_coverage(phase, material_radial_span * 1000.0, outcrop_deviation, cutoff);
            if coverage >= 0.0 {
                material_coverage = mix(select(0.0, 1.0, rock == M_STONE), coverage, band_weight);
            }
        }
        if depth < 1 && (h & 7u) == 0u { return M_DIRT; }
        return rock;
    }
    if depth == 0 {
        if exposed > -NOISE_ONE / 16 || (gully < -64 && slope >= 3) { return M_GRAVEL; }
        if exposed > -NOISE_ONE / 8 { return M_DIRT; }
        return M_GRASS;
    }
    if depth < dirt { return M_DIRT; }
    if depth < dirt * 3 && (h & 3u) == 0u { return M_GRAVEL; }
    return select(M_DARK_STONE, M_STONE, (div_floor(landform_strata(p, altitude), 12000) & 1) == 0);
}
