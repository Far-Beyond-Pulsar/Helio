// Integer terrain field. Bit-identical mirror of src/field.rs; keep in sync.
const HEIGHT_ONE: i32 = 1000;
const Q12: i32 = 65536; // noise unit (Q16)
const WARP_OCTAVES: u32 = 6u;

struct Octave { shift: u32, amplitude: i32, seed: u32, kind: u32 }
struct FieldConstants {
    header: vec4<i32>,  // reference cells, octave count, layer mm, dirt depth
    levels: vec4<i32>,  // basin floor, lowland, snowline, basin threshold (mm)
    misc: vec4<i32>,    // warp amplitude, mountain bias, steep slope (cells), seed
    scale: vec4<i32>,   // domain scale (Q24), level offset, grid cells, pad
    bounds: array<vec4<i32>, 6>, // per-level finer-surface excess (level cells)
    octaves: array<Octave, 32>,
}

fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    var v = (bitcast<u32>(x) * 0x8da6b343u) ^ (bitcast<u32>(y) * 0xd8163841u)
        ^ (bitcast<u32>(z) * 0xcb1ab31fu) ^ seed;
    v ^= v >> 16u;
    v *= 0x7feb352du;
    v ^= v >> 15u;
    v *= 0x846ca68bu;
    return v ^ (v >> 16u);
}

fn grad(hash: u32, x: i32, y: i32, z: i32) -> i32 {
    let h = hash & 15u;
    let u = select(y, x, h < 8u);
    let v = select(select(z, x, h == 12u || h == 14u), y, h < 4u);
    return select(-u, u, (h & 1u) == 0u) + select(-v, v, (h & 2u) == 0u);
}

fn mul16(a: i32, w: i32) -> i32 {
    return (a * (w >> 8u) + ((a * (w & 255)) >> 8u)) >> 8u;
}

fn fade(t: i32) -> i32 {
    let tu = bitcast<u32>(t);
    let t2 = (tu * tu) >> 16u;
    let t3 = (t2 * tu) >> 16u;
    let inner = 6 * i32(t2) - 15 * t + 10 * Q12;
    return mul16(i32(t3), inner);
}

fn lerp_q12(a: i32, b: i32, w: i32) -> i32 {
    return a + mul16(b - a, w);
}

fn to_q12(v: i32, shift: u32) -> i32 {
    if shift >= 16u { return v >> (shift - 16u); }
    return v << (16u - shift);
}

fn noise(p: vec3<i32>, shift: u32, seed: u32) -> i32 {
    let mask = (1 << shift) - 1;
    let c = p >> vec3<u32>(shift);
    let f = vec3<i32>(to_q12(p.x & mask, shift), to_q12(p.y & mask, shift), to_q12(p.z & mask, shift));
    let w = vec3<i32>(fade(f.x), fade(f.y), fade(f.z));
    let g000 = grad(hash3(c.x, c.y, c.z, seed), f.x, f.y, f.z);
    let g100 = grad(hash3(c.x + 1, c.y, c.z, seed), f.x - Q12, f.y, f.z);
    let g010 = grad(hash3(c.x, c.y + 1, c.z, seed), f.x, f.y - Q12, f.z);
    let g110 = grad(hash3(c.x + 1, c.y + 1, c.z, seed), f.x - Q12, f.y - Q12, f.z);
    let g001 = grad(hash3(c.x, c.y, c.z + 1, seed), f.x, f.y, f.z - Q12);
    let g101 = grad(hash3(c.x + 1, c.y, c.z + 1, seed), f.x - Q12, f.y, f.z - Q12);
    let g011 = grad(hash3(c.x, c.y + 1, c.z + 1, seed), f.x, f.y - Q12, f.z - Q12);
    let g111 = grad(hash3(c.x + 1, c.y + 1, c.z + 1, seed), f.x - Q12, f.y - Q12, f.z - Q12);
    let x00 = lerp_q12(g000, g100, w.x);
    let x10 = lerp_q12(g010, g110, w.x);
    let x01 = lerp_q12(g001, g101, w.x);
    let x11 = lerp_q12(g011, g111, w.x);
    let y0 = lerp_q12(x00, x10, w.y);
    let y1 = lerp_q12(x01, x11, w.y);
    return clamp(lerp_q12(y0, y1, w.z), -Q12, Q12);
}

fn scale_q12(n: i32, amplitude: i32) -> i32 {
    return n * (amplitude >> 16u) + ((n * ((amplitude & 0xffff) >> 4u)) >> 12u);
}

// Face bases, matching grid::FACE_BASIS.
fn face_axis(face: u32, axis: u32) -> vec3<i32> {
    switch face * 3u + axis {
        case 0u: { return vec3<i32>(1, 0, 0); }
        case 1u: { return vec3<i32>(0, 0, -1); }
        case 2u: { return vec3<i32>(0, 1, 0); }
        case 3u: { return vec3<i32>(-1, 0, 0); }
        case 4u: { return vec3<i32>(0, 0, 1); }
        case 5u: { return vec3<i32>(0, 1, 0); }
        case 6u: { return vec3<i32>(0, 1, 0); }
        case 7u: { return vec3<i32>(1, 0, 0); }
        case 8u: { return vec3<i32>(0, 0, -1); }
        case 9u: { return vec3<i32>(0, -1, 0); }
        case 10u: { return vec3<i32>(1, 0, 0); }
        case 11u: { return vec3<i32>(0, 0, 1); }
        case 12u: { return vec3<i32>(0, 0, 1); }
        case 13u: { return vec3<i32>(1, 0, 0); }
        case 14u: { return vec3<i32>(0, 1, 0); }
        case 15u: { return vec3<i32>(0, 0, -1); }
        case 16u: { return vec3<i32>(-1, 0, 0); }
        default: { return vec3<i32>(0, 1, 0); }
    }
}

fn mul_q24(a: u32, r: u32) -> u32 {
    let a1 = a >> 16u;
    let a0 = a & 0xffffu;
    let r1 = r >> 16u;
    let r0 = r & 0xffffu;
    return ((a1 * r1) << 8u) + ((a1 * r0 + a0 * r1) >> 8u) + ((a0 * r0) >> 24u);
}

// Level cell centre in half cells of the reference grid (grid independent).
fn domain_point(face: u32, i: i32, j: i32, level: u32) -> vec3<i32> {
    let reference = field.header.x;
    let scale = bitcast<u32>(field.scale.x);
    let half = 1u << level;
    let u = i32(mul_q24((bitcast<u32>(i) << (level + 1u)) + half, scale)) - reference;
    let v = i32(mul_q24((bitcast<u32>(j) << (level + 1u)) + half, scale)) - reference;
    return face_axis(face, 0u) * reference + face_axis(face, 1u) * u + face_axis(face, 2u) * v;
}

fn octave_resolved(o: Octave, level: u32) -> bool {
    return o.kind <= 1u || o.shift >= level + 3u;
}

fn terrain_height(p: vec3<i32>, level_in: u32) -> i32 {
    let count = u32(field.header.y);
    let level = level_in + u32(field.scale.y);
    var warp = vec3<i32>(0);
    for (var index = 0u; index < WARP_OCTAVES; index++) {
        let o = field.octaves[index];
        let n = scale_q12(noise(p, o.shift, o.seed), o.amplitude);
        let axis = o.kind - 4u;
        if axis == 0u { warp.x += n; } else if axis == 1u { warp.y += n; } else { warp.z += n; }
    }
    let q = p + warp;
    var continent = 0;
    var mask = 0;
    var ridged = 0;
    var ridge_weight = Q12 - 1;
    var detail = 0;
    for (var index = WARP_OCTAVES; index < count; index++) {
        let o = field.octaves[index];
        if !octave_resolved(o, level) { continue; }
        let n = noise(select(q, p, o.kind == 7u), o.shift, o.seed);
        if o.kind == 0u {
            continent += scale_q12(n, o.amplitude);
        } else if o.kind == 1u {
            mask += scale_q12(n, o.amplitude);
        } else if o.kind == 2u {
            let r = u32(clamp(Q12 - abs(n), 0, Q12 - 1));
            let r2 = (r * r) >> 16u;
            let v = i32((r2 * u32(ridge_weight)) >> 16u);
            ridge_weight = clamp(v * 2, Q12 / 4, Q12 - 1);
            ridged += scale_q12(v, o.amplitude);
        } else {
            detail += scale_q12(n, o.amplitude);
        }
    }
    let c = continent;
    var base: i32;
    if c < 0 {
        let t = min(-c, Q12);
        base = scale_q12(t, field.levels.x) + scale_q12(Q12 - t, field.levels.y / 8);
    } else {
        let t = min(c * 2, Q12);
        base = scale_q12(t, field.levels.y);
    }
    let land = clamp(c * 3, 0, Q12);
    let region = clamp((mask - field.misc.y) * 3, 0, Q12);
    let mountains = scale_q12(land, scale_q12(region, ridged));
    let wet = clamp(Q12 + c * 2, Q12 / 8, Q12);
    return base + mountains + scale_q12(wet, detail);
}

fn div_floor(a: i32, b: i32) -> i32 {
    let q = a / b;
    return select(q, q - 1, (a % b != 0) && ((a < 0) != (b < 0)));
}

fn top_cells(height: i32, level: u32) -> i32 {
    return div_floor(height, field.header.z) >> level;
}

fn terrain_kind(top: i32, k: i32) -> u32 {
    return select(0u, 1u, k < top);
}

fn moisture(p: vec3<i32>) -> i32 {
    let o = max(field.octaves[6].shift, 2u) - 1u;
    return (noise(p, o, bitcast<u32>(field.misc.w) ^ 0x51ED270Bu) + Q12) / 2;
}

const M_AIR: u32 = 0u;
const M_GRASS: u32 = 1u;
const M_DIRT: u32 = 2u;
const M_STONE: u32 = 3u;
const M_SAND: u32 = 4u;
const M_SNOW: u32 = 5u;
const M_GRAVEL: u32 = 7u;
const M_SANDSTONE: u32 = 8u;
const M_DARK_STONE: u32 = 9u;
const M_CLAY: u32 = 12u;

fn rem_floor(a: i32, b: i32) -> i32 {
    let r = a % b;
    return select(r, r + b, r < 0);
}

// Strata altitude (mm): layers undulate +-8 m over ~100 m, so cuts through
// them never show flat rings (`strata` in field.rs).
fn strata(p: vec3<i32>, altitude: i32) -> i32 {
    return altitude + scale_q12(noise(p, 11u, bitcast<u32>(field.misc.w) ^ 0x9B05688Cu), 8000);
}

// Ground slope in the 8x8 column block, eighths of a cell per cell
// (`block_slope` in field.rs).
fn block_slope_of(t_x0: i32, t_x7: i32, t_y0: i32, t_y7: i32) -> i32 {
    return max(abs(t_x7 - t_x0), abs(t_y7 - t_y0)) * 8 / 7;
}

fn ground_material(p: vec3<i32>, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
    let dirt = field.header.w;
    let steep = slope >= field.misc.z;
    let wet = moisture(p);
    // Hash every domain axis: on a face one of them is nearly constant.
    let h = hash3(p.x, p.y, p.z ^ (layer * 0x9e37), 0x2545F491u);
    let altitude = layer * field.header.z;
    if top_height < field.levels.w {
        // Low basins: meadow with mud and sand patches (position hash only,
        // never aligned with height contours) over silt, gravel and stone.
        if depth == 0 {
            let s = hash3(p.x, p.y, p.z, 0x5f356495u);
            if (s & 15u) == 0u { return M_DIRT; }
            if ((s >> 4u) & 31u) == 0u { return M_SAND; }
            return M_GRASS;
        }
        if depth < dirt * 2 {
            return select(M_CLAY, M_GRAVEL, (h & 3u) == 0u);
        }
        return M_STONE;
    }
    let snowline = field.levels.z + scale_q12(wet - Q12 / 2, field.levels.z / 4);
    // Alpine weight: 0 below the rockline, Q12 at the snowline.
    let rockline = snowline - field.levels.z / 3;
    let band = max(snowline - rockline, 256);
    let alpine = (clamp(top_height - rockline, 0, band) / 256) * Q12 / (band / 256);
    // Rock patches (~50 m and ~6 m octaves), also breaking up snow edges.
    // Noise is clamped to +-Q12, so below the rockline on gentler slopes no
    // outcrop can reach the rock fringe: skip it (same result).
    let seed = bitcast<u32>(field.misc.w);
    var outcrop = -Q12;
    if alpine > 0 || slope >= 5 {
        outcrop = noise(p, 10u, seed ^ 0x1B56C4E9u) + noise(p, 7u, seed ^ 0x6A09E667u) / 3;
    }
    // Snow does not hold on faces steeper than ~37 degrees: rock streaks the snowfields.
    if top_height > snowline && depth < dirt && slope + outcrop / 8192 < 6 { return M_SNOW; }
    if wet < Q12 * 3 / 10 {
        if depth < dirt && !steep { return M_SAND; }
        let band = rem_floor(div_floor(strata(p, altitude), 2100), 5);
        return select(M_SANDSTONE, M_CLAY, band == 1 || band == 3);
    }
    // Rock shows through the turf in the outcrop patches, which grow up the
    // alpine band below the snowline and on hillsides over ~32 degrees; scree and
    // bare soil fringe them. Deeper cells keep the strata.
    var exposed = outcrop + 2 * alpine - Q12;
    if slope >= 5 { exposed += Q12 / 2; }
    if steep || (exposed > 0 && depth < dirt) {
        if depth < 1 && (h & 7u) == 0u { return M_DIRT; }
        return select(M_DARK_STONE, M_STONE, (div_floor(altitude + scale_q12(outcrop, 3000), 4500) & 1) == 0);
    }
    if depth == 0 {
        if exposed > -Q12 / 16 { return M_GRAVEL; }
        if exposed > -Q12 / 8 { return M_DIRT; }
        return M_GRASS;
    }
    if depth < dirt { return M_DIRT; }
    if depth < dirt * 3 && (h & 3u) == 0u { return M_GRAVEL; }
    return select(M_DARK_STONE, M_STONE, (div_floor(strata(p, altitude), 12000) & 1) == 0);
}

fn bound_margin(level: u32) -> i32 {
    return field.bounds[level >> 2u][level & 3u];
}
