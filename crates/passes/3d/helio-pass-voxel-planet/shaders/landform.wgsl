// The landform terrain program; mirror of src/landform.rs.
const LANDFORM_WARP: u32 = 6u; // leading domain-warp octaves (two per axis)

struct LandformOctave { shift: u32, amplitude: i32, seed: u32, kind: u32 }
struct TerrainConstants {
    header: vec4<i32>, // octave count, layer mm, dirt depth (cells), seed
    levels: vec4<i32>, // basin floor, lowland, snowline, basin threshold (mm)
    shape: vec4<i32>,  // mountain mask bias, steep slope (cells), pad, pad
    octaves: array<LandformOctave, 32>,
    ridge_suffix: array<vec4<i32>, 66>, // signed conditional suffix means
}

fn landform_resolved(o: LandformOctave, level: u32) -> bool {
    return o.kind <= 1u || o.shift >= level + 3u;
}

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
    if shift >= effective_level + 3u { return FINE_ONE; }
    if shift == effective_level + 2u { return FINE_ONE / 2; }
    return 0;
}

// The false mode preserves the canonical operation order. Only generation
// compiles the display capability; climate, shade and verify_field stay exact.
fn terrain_height_mode(p: vec3<i32>, level: u32, display: bool) -> i32 {
    let count = u32(terrain.header.x);
    var warp = vec3<i32>(0);
    for (var index = 0u; index < LANDFORM_WARP; index++) {
        let o = terrain.octaves[index];
        let n = mul_fine(o.amplitude, noise_fine(p, o.shift, o.seed));
        let axis = o.kind - 4u;
        if axis == 0u { warp.x += n; } else if axis == 1u { warp.y += n; } else { warp.z += n; }
    }
    let q = p + warp;
    var continent = 0;
    var mask = 0;
    var ridged = 0;
    var ridge_weight = FINE_ONE - 1;
    var detail = 0;
    var ridge_gain = FINE_ONE;
    var ridge_row = 0u;
    for (var index = LANDFORM_WARP; index < count; index++) {
        let o = terrain.octaves[index];
        if RIDGE_DISPLAY_GENERATION && display && o.kind == 2u {
            // Expand B[k](w)=(1-s)*F[k](w)+s*(A[k]*v+B[k+1](next(w))).
            // Repeated lattice shifts may have several partial octaves.
            if ridge_gain != 0 {
                let support = ridge_display_support(o.shift, level);
                if support != FINE_ONE {
                    let mean_gain = mul_fine(ridge_gain, FINE_ONE - support);
                    ridged += mul_fine(ridge_suffix_mean(ridge_row, ridge_weight), mean_gain);
                    ridge_gain = mul_fine(ridge_gain, support);
                }
                if ridge_gain != 0 {
                    let n = noise_fine(q, o.shift, o.seed);
                    let r = clamp(FINE_ONE - abs(n), 0, FINE_ONE - 1);
                    let v = mul_fine(mul_fine(r, r), ridge_weight);
                    ridged += mul_fine(mul_fine(o.amplitude, v), ridge_gain);
                    ridge_weight = clamp(v * 2, FINE_ONE / 4, FINE_ONE - 1);
                }
            }
            ridge_row += 1u;
            continue;
        }
        if !landform_resolved(o, level) { continue; }
        if o.kind == 0u {
            continent += mul_fine(noise_fine(q, o.shift, o.seed), o.amplitude << 8u);
        } else if o.kind == 1u {
            mask += mul_fine(noise_fine(q, o.shift, o.seed), o.amplitude << 8u);
        } else if o.kind == 2u {
            let n = noise_fine(q, o.shift, o.seed);
            let r = clamp(FINE_ONE - abs(n), 0, FINE_ONE - 1);
            let v = mul_fine(mul_fine(r, r), ridge_weight);
            ridge_weight = clamp(v * 2, FINE_ONE / 4, FINE_ONE - 1);
            ridged += mul_fine(o.amplitude, v);
        } else {
            detail += scale_q16(noise(select(q, p, o.kind == 7u), o.shift, o.seed), o.amplitude);
        }
    }
    let c = continent;
    var base: i32;
    if c < 0 {
        let t = min(-c, FINE_ONE);
        base = mul_fine(terrain.levels.x, t) + mul_fine(terrain.levels.y / 8, FINE_ONE - t);
    } else {
        let t = min(c * 2, FINE_ONE);
        base = mul_fine(terrain.levels.y, t);
    }
    let land = clamp(c * 3, 0, FINE_ONE);
    let region = clamp((mask - (terrain.shape.x << 8u)) * 3, 0, FINE_ONE);
    let mountains = mul_fine(mul_fine(ridged, region), land);
    let wet = clamp(FINE_ONE + c * 2, FINE_ONE / 8, FINE_ONE);
    return base + mountains + mul_fine(detail, wet);
}

fn terrain_height(p: vec3<i32>, level: u32) -> i32 {
    return terrain_height_mode(p, level, false);
}

fn terrain_display_height(p: vec3<i32>, level: u32) -> i32 {
    return terrain_height_mode(p, level, terrain.shape.z != 0);
}

fn landform_moisture(p: vec3<i32>) -> i32 {
    let o = max(terrain.octaves[6].shift, 2u) - 1u;
    return (noise_fine(p, o, bitcast<u32>(terrain.header.w) ^ 0x51ED270Bu) + FINE_ONE) / 2;
}

// Strata altitude (mm): layers undulate +-8 m over ~100 m, so cuts through
// them never show flat rings (`strata` in landform.rs).
fn landform_strata(p: vec3<i32>, altitude: i32) -> i32 {
    return altitude + scale_q16(noise(p, 11u, bitcast<u32>(terrain.header.w) ^ 0x9B05688Cu), 8000);
}

// Material noise is defined in the fixed 5 cm half-reference domain.
// Four samples per lattice spacing retain contrast; below two samples the
// unresolved octave contributes its coverage instead of aliased class noise.
fn material_noise_support(shift: u32, pixel: f32) -> f32 {
    if pixel <= 0.0 { return 1.0; }
    let wavelength = 0.05 * f32(1u << shift);
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
    // Integrate the existing alternating 4.5m stone bands too: replacing
    // unresolved snow with a single rock colour would bias their mean.
    let mean_altitude = f32(altitude) + resolved * (3000.0 / 65536.0);
    let first = i32(floor((mean_altitude - 4000.0) / 4500.0));
    var stone = 0.0;
    for (var b = first; b < first + 4; b++) {
        if (b & 1) != 0 { continue; }
        let lo = max(threshold, (f32(b) * 4500.0 - f32(altitude)) * (65536.0 / 3000.0));
        let hi = (f32(b + 1) * 4500.0 - f32(altitude)) * (65536.0 / 3000.0);
        if hi > lo {
            stone += material_noise_cdf((hi - resolved) / deviation)
                - material_noise_cdf((lo - resolved) / deviation);
        }
    }
    stone = clamp(stone, 0.0, rock);
    let dirt = select(0.0, rock * 0.125, depth < 1);
    let remaining = select(1.0, 0.875, depth < 1);
    return vec4<f32>(snow, stone * remaining, (rock - stone) * remaining, dirt);
}

fn ground_material(p: vec3<i32>, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
    material_snow_mix = vec4<f32>(-1.0, 0.0, 0.0, 0.0);
    material_rock_base_id = M_AIR;
    material_stone_coverage = -1.0;
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
        let broad = noise(p, 10u, seed ^ 0x1B56C4E9u);
        let fine = noise(p, 7u, seed ^ 0x6A09E667u) / 3;
        outcrop = broad + fine;
        resolved_outcrop = f32(outcrop);
        if material_footprint > 1.6 {
            let broad_support = material_noise_support(10u, material_footprint);
            let fine_support = material_noise_support(7u, material_footprint);
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
                material_rock_id = select(M_STONE, M_SAND, dry);
                material_snow_mix = snow_material_coverage(resolved, deviation, threshold, altitude, dry, depth);
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
    if steep || (exposed > 0 && depth < dirt) {
        let rock = select(M_DARK_STONE, M_STONE, (div_floor(altitude + scale_q16(outcrop, 3000), 4500) & 1) == 0);
        if depth < 1 { material_rock_base_id = rock; }
        // Four samples per 4.5 m band retain its resolved contrast, matching
        // the noise support gate. No coverage work on fully resolved rock.
        let band_weight = max(smoothstep(1.125, 2.25, material_radial_span), 1.0 - outcrop_support);
        if material_footprint > 0.0 && band_weight > 0.0 {
            // Exposure conditions the phase distribution on non-steep patches.
            let exposure = f32(NOISE_ONE - 2 * alpine - select(0, NOISE_ONE / 2, slope >= 5)) + 0.5;
            let cutoff = select(-65536.0, (exposure - resolved_outcrop) / max(outcrop_deviation, 0.0001), !steep);
            let phase = f32(rem_floor(altitude, 9000)) + resolved_outcrop * (3000.0 / 65536.0);
            let coverage = rock_band_coverage(phase, material_radial_span * 1000.0, outcrop_deviation, cutoff);
            if coverage >= 0.0 {
                material_stone_coverage = mix(select(0.0, 1.0, rock == M_STONE), coverage, band_weight);
            }
        }
        if depth < 1 && (h & 7u) == 0u { return M_DIRT; }
        return rock;
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
