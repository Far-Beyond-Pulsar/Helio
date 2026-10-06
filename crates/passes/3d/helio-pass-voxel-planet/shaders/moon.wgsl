// The moon terrain program; mirror of src/moon.rs.
struct CraterOctave { shift: u32, depth: i32, density: u32, seed: u32 }
struct TerrainConstants {
    header: vec4<i32>,  // crater octaves, layer mm, regolith cells, plane
    shape: vec4<i32>,   // highland shift, highland mm, mare shift, mare threshold (Q16)
    surface: vec4<i32>, // mare depth mm, rim / depth (Q16), fresh share (Q16), seed
    radius: vec4<u32>,  // domain radius, largest crater radius (Q19 cells), pad
    craters: array<CraterOctave, 12>,
}

const MOON_REACH: i32 = 67108864; // squared ejecta reach in crater radii (Q24)

// 2^60 / d for d in [2^29, 2^30) (`moon::recip_q30`).
fn moon_recip_q30(d: u32) -> u32 {
    var y = 3031741621u - mul_shr(d, 2021161080u, 30u);
    for (var step = 0u; step < 4u; step++) {
        y = mul_shr(y, 2u * Q30 - mul_shr(d, y, 30u), 30u);
    }
    return y;
}

// r2 / rad2 in Q24 (`moon::squared_ratio`).
fn moon_squared_ratio(r2: u32, rad2: u32) -> u32 {
    let bits = 32u - countLeadingZeros(rad2);
    return mul_shr(r2, moon_recip_q30(rad2 << (30u - bits)), 6u + bits);
}

// Polynomial smooth minimum over a width `k` (`moon::smooth_min`).
fn moon_smooth_min(a: i32, b: i32, k: i32) -> i32 {
    let m = min(a, b);
    let gap = abs(a - b);
    if k <= 0 || gap >= k { return m; }
    let n = (k - gap) << 8u;
    let q = ((n / k) << 8u) + ((n % k) << 8u) / k;
    return m - scale_q16(mul16(q, q), k) / 4;
}

fn moon_resolved(shift: u32, level: u32) -> bool {
    return shift >= level + 5u; // landform::RESOLVED_SHIFT
}

// One crater octave (`moon::crater_octave`): height (mm), freshest ejecta.
fn moon_crater(o: CraterOctave, p: vec3<i32>, up: vec3<i32>) -> vec2<i32> {
    let s = o.shift;
    let c0 = p >> vec3<u32>(s);
    var height = 0;
    var ejecta = 0u;
    for (var index = 0; index < 27; index++) {
        let cell = c0 + vec3<i32>(index % 3 - 1, (index / 3) % 3 - 1, index / 9 - 1);
        let a = hash3(cell.x, cell.y, cell.z, o.seed);
        if (a & 0xffffu) >= o.density { continue; }
        let b = hash3(cell.x, cell.y, cell.z, o.seed ^ 0x6C8E9CF5u);
        let jitter = vec3<i32>(i32(b & 1023u), i32((b >> 10u) & 1023u), i32((b >> 20u) & 1023u));
        let centre = (cell << vec3<u32>(s)) + (jitter << vec3<u32>(s - 10u));
        // Only centres within half a cell of the surface.
        if terrain.header.w != 0 {
            if abs(centre.y) >= (1 << (s - 1u)) { continue; }
        } else {
            let ac = vec3<u32>(abs(centre));
            let len2 = mul_shr(ac.x, ac.x, 30u) + mul_shr(ac.y, ac.y, 30u) + mul_shr(ac.z, ac.z, 30u);
            let r2 = mul_shr(terrain.radius.x, terrain.radius.x, 30u);
            if u32(abs(i32(len2) - i32(r2))) >= (terrain.radius.x >> (30u - s)) { continue; }
        }
        // Squared horizontal distance in Q28 lattice cells (`crater_octave`).
        var d = centre - p;
        if s >= 19u { d = d >> vec3<u32>(s - 19u); } else { d = d << vec3<u32>(19u - s); }
        if max(max(abs(d.x), abs(d.y)), abs(d.z)) >= (1 << 19u) { continue; }
        let along = mul_shr_signed(d.x, up.x, 20u) + mul_shr_signed(d.y, up.y, 20u) + mul_shr_signed(d.z, up.z, 20u);
        let ad = vec3<u32>(abs(d));
        let len2 = mul_shr(ad.x, ad.x, 10u) + mul_shr(ad.y, ad.y, 10u) + mul_shr(ad.z, ad.z, 10u);
        let aa = mul_shr(u32(abs(along)), u32(abs(along)), 30u);
        let r2 = select(0u, len2 - aa, len2 > aa);
        let size = 36045 + i32(((a >> 16u) * 29491u) >> 16u);
        let radius = max(mul16(i32(terrain.radius.y), size), 1);
        let rad2 = max(mul_shr(u32(radius), u32(radius), 10u), 1u);
        if r2 >= 4u * rad2 { continue; }
        let x2 = i32(moon_squared_ratio(r2, rad2));
        let depth = scale_q16(size, o.depth);
        let rim = scale_q16(terrain.surface.y, depth);
        let bowl = mul_fine(x2, depth + rim) - depth;
        let t = (MOON_REACH - x2) / 3;
        let ejecta_height = mul_fine(mul_fine(mul_fine(t, t), t), rim);
        height += moon_smooth_min(bowl, ejecta_height, rim / 4);
        if (hash3(cell.x, cell.y, cell.z, o.seed ^ 0x1B873593u) & 0xffffu) < u32(terrain.surface.z) {
            ejecta = max(ejecta, 255u - u32((x2 >> 16u) * 255 / (MOON_REACH >> 16u)));
        }
    }
    return vec2<i32>(height, i32(ejecta));
}

// Height and surface word (`moon::height_parts`).
fn moon_parts(p: vec3<i32>, level: u32) -> vec2<i32> {
    let seed = bitcast<u32>(terrain.surface.w);
    let mare = clamp((noise_fine(p, u32(terrain.shape.z), seed ^ 0x2545F491u) - (terrain.shape.w << 8u)) * 4, 0, FINE_ONE);
    var highland = 0;
    for (var o = 0u; o < 3u; o++) {
        let shift = u32(max(terrain.shape.x - i32(o), 0));
        if moon_resolved(shift, level) {
            highland += mul_fine(terrain.shape.y / (1 << o), noise_fine(p, shift, seed ^ 0x51ED270Bu ^ o));
        }
    }
    var height = mul_fine(highland, FINE_ONE - mare / 2) - mul_fine(terrain.surface.x, mare);
    var up = vec3<i32>(0, 1073741824, 0);
    if terrain.header.w == 0 { up = unit_q30(p); }
    var ejecta = 0;
    for (var index = 0; index < terrain.header.x; index++) {
        let o = terrain.craters[index];
        if moon_resolved(o.shift, level) {
            let c = moon_crater(o, p, up);
            height += c.x;
            ejecta = max(ejecta, c.y);
        }
    }
    return vec2<i32>(height, (ejecta >> 1u) | (select(0, 1, mare >= FINE_ONE / 2) << 7u));
}

// Surface word of the last height evaluated in this invocation: generation
// asks for it right after the column's height.
var<private> moon_surface: i32 = 0;
var<private> moon_surface_at: vec4<i32> = vec4<i32>(0x7fffffff);

fn terrain_height(p: vec3<i32>, level: u32) -> i32 {
    let parts = moon_parts(p, level);
    moon_surface = parts.y;
    moon_surface_at = vec4<i32>(p, i32(level));
    return parts.x;
}

fn terrain_surface(p: vec3<i32>, level: u32, height: i32) -> u32 {
    if any(moon_surface_at != vec4<i32>(p, i32(level))) { _ = terrain_height(p, level); }
    return u32(moon_surface);
}

// Materials (`moon::ground_material`): 1 regolith, 2 mare, 3 ejecta, 4 rock,
// 5 basalt, 6 anorthosite.
fn ground_material(p: vec3<i32>, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
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
