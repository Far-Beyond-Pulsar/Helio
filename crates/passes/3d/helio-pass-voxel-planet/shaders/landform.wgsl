// The landform terrain program; mirror of src/landform.rs.
const LANDFORM_WARP: u32 = 6u; // leading domain-warp octaves (two per axis)

struct LandformOctave { shift: u32, amplitude: i32, seed: u32, kind: u32 }
struct TerrainConstants {
    header: vec4<i32>, // octave count, layer mm, dirt depth (cells), seed
    levels: vec4<i32>, // basin floor, lowland, snowline, basin threshold (mm)
    shape: vec4<i32>,  // mountain mask bias, steep slope (cells), pad, pad
    octaves: array<LandformOctave, 32>,
}

fn landform_resolved(o: LandformOctave, level: u32) -> bool {
    return o.kind <= 1u || o.shift >= level + 3u;
}

fn terrain_height(p: vec3<i32>, level: u32) -> i32 {
    let count = u32(terrain.header.x);
    var warp = vec3<i32>(0);
    for (var index = 0u; index < LANDFORM_WARP; index++) {
        let o = terrain.octaves[index];
        let n = scale_q16(noise(p, o.shift, o.seed), o.amplitude);
        let axis = o.kind - 4u;
        if axis == 0u { warp.x += n; } else if axis == 1u { warp.y += n; } else { warp.z += n; }
    }
    let q = p + warp;
    var continent = 0;
    var mask = 0;
    var ridged = 0;
    var ridge_weight = NOISE_ONE - 1;
    var detail = 0;
    for (var index = LANDFORM_WARP; index < count; index++) {
        let o = terrain.octaves[index];
        if !landform_resolved(o, level) { continue; }
        let n = noise(select(q, p, o.kind == 7u), o.shift, o.seed);
        if o.kind == 0u {
            continent += scale_q16(n, o.amplitude);
        } else if o.kind == 1u {
            mask += scale_q16(n, o.amplitude);
        } else if o.kind == 2u {
            let r = u32(clamp(NOISE_ONE - abs(n), 0, NOISE_ONE - 1));
            let r2 = (r * r) >> 16u;
            let v = i32((r2 * u32(ridge_weight)) >> 16u);
            ridge_weight = clamp(v * 2, NOISE_ONE / 4, NOISE_ONE - 1);
            ridged += scale_q16(v, o.amplitude);
        } else {
            detail += scale_q16(n, o.amplitude);
        }
    }
    let c = continent;
    var base: i32;
    if c < 0 {
        let t = min(-c, NOISE_ONE);
        base = scale_q16(t, terrain.levels.x) + scale_q16(NOISE_ONE - t, terrain.levels.y / 8);
    } else {
        let t = min(c * 2, NOISE_ONE);
        base = scale_q16(t, terrain.levels.y);
    }
    let land = clamp(c * 3, 0, NOISE_ONE);
    let region = clamp((mask - terrain.shape.x) * 3, 0, NOISE_ONE);
    let mountains = scale_q16(land, scale_q16(region, ridged));
    let wet = clamp(NOISE_ONE + c * 2, NOISE_ONE / 8, NOISE_ONE);
    return base + mountains + scale_q16(wet, detail);
}

fn landform_moisture(p: vec3<i32>) -> i32 {
    let o = max(terrain.octaves[6].shift, 2u) - 1u;
    return (noise(p, o, bitcast<u32>(terrain.header.w) ^ 0x51ED270Bu) + NOISE_ONE) / 2;
}

// Strata altitude (mm): layers undulate +-8 m over ~100 m, so cuts through
// them never show flat rings (`strata` in landform.rs).
fn landform_strata(p: vec3<i32>, altitude: i32) -> i32 {
    return altitude + scale_q16(noise(p, 11u, bitcast<u32>(terrain.header.w) ^ 0x9B05688Cu), 8000);
}

fn ground_material(p: vec3<i32>, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
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
    let snowline = terrain.levels.z + scale_q16(wet - NOISE_ONE / 2, terrain.levels.z / 4);
    // Alpine weight: 0 below the rockline, NOISE_ONE at the snowline.
    let rockline = snowline - terrain.levels.z / 3;
    let band = max(snowline - rockline, 256);
    let alpine = (clamp(top_height - rockline, 0, band) / 256) * NOISE_ONE / (band / 256);
    // Rock patches (~50 m and ~6 m octaves), also breaking up snow edges.
    // Noise is clamped to +-NOISE_ONE, so below the rockline on gentler slopes no
    // outcrop can reach the rock fringe: skip it (same result).
    let seed = bitcast<u32>(terrain.header.w);
    var outcrop = -NOISE_ONE;
    if alpine > 0 || slope >= 5 {
        outcrop = noise(p, 10u, seed ^ 0x1B56C4E9u) + noise(p, 7u, seed ^ 0x6A09E667u) / 3;
    }
    // Snow does not hold on faces steeper than ~37 degrees: rock streaks the snowfields.
    if top_height > snowline && depth < dirt && slope + outcrop / 8192 < 6 { return M_SNOW; }
    if wet < NOISE_ONE * 3 / 10 {
        if depth < dirt && !steep { return M_SAND; }
        let band = rem_floor(div_floor(landform_strata(p, altitude), 2100), 5);
        return select(M_SANDSTONE, M_CLAY, band == 1 || band == 3);
    }
    // Rock shows through the turf in the outcrop patches, which grow up the
    // alpine band below the snowline and on hillsides over ~32 degrees; scree and
    // bare soil fringe them. Deeper cells keep the strata.
    var exposed = outcrop + 2 * alpine - NOISE_ONE;
    if slope >= 5 { exposed += NOISE_ONE / 2; }
    if steep || (exposed > 0 && depth < dirt) {
        if depth < 1 && (h & 7u) == 0u { return M_DIRT; }
        return select(M_DARK_STONE, M_STONE, (div_floor(altitude + scale_q16(outcrop, 3000), 4500) & 1) == 0);
    }
    if depth == 0 {
        if exposed > -NOISE_ONE / 16 { return M_GRAVEL; }
        if exposed > -NOISE_ONE / 8 { return M_DIRT; }
        return M_GRASS;
    }
    if depth < dirt { return M_DIRT; }
    if depth < dirt * 3 && (h & 3u) == 0u { return M_GRAVEL; }
    return select(M_DARK_STONE, M_STONE, (div_floor(landform_strata(p, altitude), 12000) & 1) == 0);
}
