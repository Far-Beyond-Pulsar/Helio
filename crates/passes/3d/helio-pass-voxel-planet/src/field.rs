//! Deterministic integer terrain field shared bit-for-bit with `field.wgsl`.
//!
//! Heights are fixed point (1/256 base cell). Every operation is wrapping
//! two's-complement integer arithmetic, so CPU queries, collision and GPU
//! generation agree exactly. Additive octaves finer than a level's cells are
//! omitted at that level: coarse levels are band-limited point samples of the
//! same field rather than an independent smooth replacement.
use crate::grid::Grid;
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use serde::{Deserialize, Serialize};

/// Heights are integer millimetres above the datum (sea level).
pub const HEIGHT_ONE: i32 = 1000;
/// Noise output scale (Q16).
const ONE: i32 = 65_536;

pub mod material {
    pub const AIR: u32 = 0;
    pub const GRASS: u32 = 1;
    pub const DIRT: u32 = 2;
    pub const STONE: u32 = 3;
    pub const SAND: u32 = 4;
    pub const SNOW: u32 = 5;
    pub const WATER: u32 = 6;
    pub const GRAVEL: u32 = 7;
    pub const SANDSTONE: u32 = 8;
    pub const DARK_STONE: u32 = 9;
    pub const WOOD: u32 = 10;
    pub const LEAVES: u32 = 11;
    pub const CLAY: u32 = 12;
    pub const BRICK: u32 = 13;
    pub const PLANKS: u32 = 14;
    pub const COBBLE: u32 = 15;
    pub const COUNT: u32 = 16;
}

/// Authored landform parameters in metres. Converted per grid into
/// [`FieldConstants`]; the same recipe produces a similar planet at every
/// supported voxel size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Landform {
    pub seed: u32,
    /// Wavelength of continents.
    pub continent_km: f64,
    /// Ocean floor depth and typical lowland height.
    pub ocean_depth_m: f64,
    pub lowland_m: f64,
    pub mountain_m: f64,
    pub mountain_km: f64,
    pub hill_m: f64,
    pub hill_km: f64,
    /// Amplitude of metre-scale roughness as a fraction of wavelength.
    pub roughness: f64,
    pub warp_km: f64,
    pub snowline_m: f64,
}

impl Default for Landform {
    fn default() -> Self {
        Self {
            seed: 7,
            continent_km: 3_000.0,
            ocean_depth_m: 2_400.0,
            lowland_m: 180.0,
            mountain_m: 2_600.0,
            mountain_km: 90.0,
            hill_m: 140.0,
            hill_km: 9.0,
            roughness: 0.035,
            warp_km: 40.0,
            snowline_m: 2_300.0,
        }
    }
}

pub const OCTAVES: usize = 32;
pub const WARP_OCTAVES: usize = 6;

/// One additive octave: lattice shift, amplitude (height units) and kind.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Octave {
    pub shift: u32,
    pub amplitude: i32,
    pub seed: u32,
    /// 0 = continent, 1 = mountain mask, 2 = ridged, 3 = hills/detail, 4 = warp.
    pub kind: u32,
}

/// GPU-mirrored constants (std430/uniform compatible, 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct FieldConstants {
    /// reference cells, octave count, layer thickness (mm), dirt depth (cells).
    pub header: [i32; 4],
    /// basin floor, lowland, snowline, basin threshold (mm).
    pub levels: [i32; 4],
    /// warp amplitude (domain units), mountain mask bias, steep slope (cells/cell), seed.
    pub misc: [i32; 4],
    /// domain scale (Q24), level offset, grid cells, pad.
    pub scale: [i32; 4],
    /// Per-level conservative excess (level cells) of any equal-or-finer
    /// level's surface inside a level cell over that cell's own top.
    pub bounds: [[i32; 4]; 6],
    pub octaves: [Octave; OCTAVES],
}

impl FieldConstants {
    pub fn new(grid: &Grid, land: &Landform) -> Self {
        let units = |metres: f64| (metres * f64::from(HEIGHT_ONE)).round() as i32;
        // Lattice spacing for a wavelength, in reference half cells.
        let half = crate::grid::REFERENCE_VOXEL * 0.5;
        let shift = |metres: f64| ((metres / half).log2().round().clamp(1.0, 29.0)) as u32;
        let mut octaves = Vec::new();
        let mut seed = land.seed.wrapping_mul(0x9E37_79B9);
        let mut next_seed = || {
            seed = seed.wrapping_add(0x6D2B_79F5);
            seed
        };
        // Domain warp: two octaves per axis, amplitude in domain units, always
        // evaluated with 16-bit noise so the displacement is continuous.
        let warp_units = land.warp_km * 1_000.0 * 0.15 / half;
        for axis in 0..3 {
            for o in 0..2 {
                let w = land.warp_km * 1_000.0 / f64::from(1u32 << o);
                octaves.push(Octave {
                    shift: shift(w),
                    amplitude: (warp_units / f64::from(1u32 << o)).round() as i32,
                    seed: next_seed(),
                    kind: 4 + axis as u32,
                });
            }
        }
        for o in 0..4 {
            let w = land.continent_km * 1_000.0 / f64::from(1u32 << o);
            octaves.push(Octave {
                shift: shift(w),
                amplitude: ONE >> o,
                seed: next_seed(),
                kind: 0,
            });
        }
        for o in 0..2 {
            let w = land.mountain_km * 12_000.0 / f64::from(1u32 << o);
            octaves.push(Octave {
                shift: shift(w),
                amplitude: ONE >> o,
                seed: next_seed(),
                kind: 1,
            });
        }
        let mut amplitude = land.mountain_m;
        for o in 0..7 {
            let w = land.mountain_km * 1_000.0 / f64::from(1u32 << o);
            octaves.push(Octave {
                shift: shift(w),
                amplitude: units(amplitude),
                seed: next_seed(),
                kind: 2,
            });
            amplitude *= 0.47;
        }
        let mut amplitude = land.hill_m;
        let mut w = land.hill_km * 1_000.0;
        while w > 700.0 {
            octaves.push(Octave {
                shift: shift(w),
                amplitude: units(amplitude),
                seed: next_seed(),
                kind: 3,
            });
            amplitude *= 0.5;
            w *= 0.5;
        }
        let mut w = 512.0;
        while w >= 1.9 && octaves.len() < OCTAVES {
            octaves.push(Octave {
                shift: shift(w),
                amplitude: units(w * land.roughness),
                seed: next_seed(),
                // Metre-scale detail is not warped (kind 7).
                kind: 7,
            });
            w *= 0.5;
        }
        assert!(octaves.len() <= OCTAVES, "too many octaves");
        let count = octaves.len() as i32;
        let mut table = [Octave::default(); OCTAVES];
        table[..octaves.len()].copy_from_slice(&octaves);
        let mut constants = Self {
            header: [
                grid.reference_cells(),
                count,
                grid.layer_mm() as i32,
                ((0.7 / grid.voxel_size()).round() as i32).max(1),
            ],
            levels: [
                units(-land.ocean_depth_m),
                units(land.lowland_m),
                units(land.snowline_m),
                units(-8.0),
            ],
            misc: [0, ONE / 5, 2, land.seed as i32],
            scale: [grid.domain_scale() as i32, grid.level_offset() as i32, grid.cells(), 0],
            bounds: [[0; 4]; 6],
            octaves: table,
        };
        let margins = constants.bound_margins(grid);
        for (level, m) in margins.iter().enumerate() {
            constants.bounds[level / 4][level % 4] = *m;
        }
        constants
    }

    /// Conservative per-level surface excess, in level cells (see `bounds`).
    ///
    /// A finer level adds octaves that this level omits (each bounded by its
    /// amplitude) and the resolved field varies inside the cell by at most its
    /// Lipschitz constant times the half diagonal. The noise gradient bound
    /// `G` is 1.5x the measured maximum of the fixed-point gradient noise
    /// (5.3 per lattice spacing); `bound_margins_hold_for_sampled_cells`
    /// checks the result against exhaustive samples.
    pub fn bound_margins(&self, grid: &Grid) -> [i32; 24] {
        const G: f64 = 8.0;
        let count = self.header[1] as usize;
        let ridged_sum: f64 = self.octaves[..count].iter().filter(|o| o.kind == 2).map(|o| f64::from(o.amplitude.abs())).sum();
        let detail_sum: f64 = self.octaves[..count].iter().filter(|o| o.kind == 3 || o.kind == 7).map(|o| f64::from(o.amplitude.abs())).sum();
        // Domain warp Lipschitz constant (dimensionless).
        let warp = self.octaves[..WARP_OCTAVES]
            .iter()
            .filter(|o| o.kind == 4)
            .map(|o| f64::from(o.amplitude) * G / 2f64.powi(o.shift as i32))
            .sum::<f64>();
        let ratio = f64::from(grid.reference_cells()) / f64::from(grid.cells());
        let mut out = [0i32; 24];
        for level in 0..24u32 {
            let effective = level + self.scale[1] as u32;
            let mut dropped = 0.0;
            let mut lipschitz = 0.0;
            let mut unwarped = 0.0;
            for o in &self.octaves[WARP_OCTAVES..count] {
                let amplitude = match o.kind {
                    0 => f64::from(self.levels[0].abs()) + 2.0 * f64::from(self.levels[1].abs()) + 3.0 * ridged_sum + 2.0 * detail_sum,
                    1 => 3.0 * ridged_sum,
                    2 => 2.0 * f64::from(o.amplitude.abs()),
                    _ => f64::from(o.amplitude.abs()),
                } / f64::from(ONE) * f64::from(ONE);
                if o.kind >= 2 && o.shift < effective + 3 {
                    dropped += f64::from(o.amplitude.abs());
                } else if o.kind == 7 {
                    unwarped += amplitude * G / 2f64.powi(o.shift as i32);
                } else {
                    lipschitz += amplitude * G / 2f64.powi(o.shift as i32);
                }
            }
            // Half diagonal of a level cell in reference half cells.
            let half_diagonal = 2f64.powi(level as i32 + 1) * ratio * std::f64::consts::SQRT_2 * 0.5;
            let excess_mm = dropped + (lipschitz * (1.0 + warp) + unwarped) * half_diagonal;
            let cell_mm = f64::from(self.header[2]) * 2f64.powi(level as i32);
            out[level as usize] = ((excess_mm / cell_mm).ceil() as i64 + 2).clamp(2, 1 << 20) as i32;
        }
        out
    }
}

#[inline]
pub fn hash3(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut v = (x as u32).wrapping_mul(0x8da6_b343)
        ^ (y as u32).wrapping_mul(0xd816_3841)
        ^ (z as u32).wrapping_mul(0xcb1a_b31f)
        ^ seed;
    v ^= v >> 16;
    v = v.wrapping_mul(0x7feb_352d);
    v ^= v >> 15;
    v = v.wrapping_mul(0x846c_a68b);
    v ^ (v >> 16)
}

#[inline]
fn grad(hash: u32, x: i32, y: i32, z: i32) -> i32 {
    let h = hash & 15;
    let u = if h < 8 { x } else { y };
    let v = if h < 4 {
        y
    } else if h == 12 || h == 14 {
        x
    } else {
        z
    };
    (if h & 1 == 0 { u } else { u.wrapping_neg() }).wrapping_add(if h & 2 == 0 {
        v
    } else {
        v.wrapping_neg()
    })
}

/// `a * w >> 16` for `|a| < 2^19`, `0 <= w <= 65536` with 32-bit intermediates.
#[inline]
fn mul16(a: i32, w: i32) -> i32 {
    a.wrapping_mul(w >> 8)
        .wrapping_add(a.wrapping_mul(w & 255) >> 8)
        >> 8
}

#[inline]
fn fade(t: i32) -> i32 {
    // 6t^5 - 15t^4 + 10t^3 in Q16.
    let tu = t as u32;
    let t2 = tu.wrapping_mul(tu) >> 16;
    let t3 = t2.wrapping_mul(tu) >> 16;
    let inner = (6 * t2 as i32).wrapping_sub(15 * t).wrapping_add(10 * ONE);
    mul16(t3 as i32, inner)
}

#[inline]
fn lerp(a: i32, b: i32, w: i32) -> i32 {
    a.wrapping_add(mul16(b.wrapping_sub(a), w))
}

/// Gradient noise on a lattice of spacing `2^shift` domain units, with
/// 16-bit fractions and output in about [-65536, 65536].
pub fn noise(p: IVec3, shift: u32, seed: u32) -> i32 {
    let mask = (1i32 << shift) - 1;
    let c = [p.x >> shift, p.y >> shift, p.z >> shift];
    let f = [p.x & mask, p.y & mask, p.z & mask].map(|v| {
        if shift >= 16 {
            v >> (shift - 16)
        } else {
            v << (16 - shift)
        }
    });
    let w = f.map(fade);
    let corner = |dx: i32, dy: i32, dz: i32| {
        grad(
            hash3(c[0].wrapping_add(dx), c[1].wrapping_add(dy), c[2].wrapping_add(dz), seed),
            f[0] - dx * ONE,
            f[1] - dy * ONE,
            f[2] - dz * ONE,
        )
    };
    let x00 = lerp(corner(0, 0, 0), corner(1, 0, 0), w[0]);
    let x10 = lerp(corner(0, 1, 0), corner(1, 1, 0), w[0]);
    let x01 = lerp(corner(0, 0, 1), corner(1, 0, 1), w[0]);
    let x11 = lerp(corner(0, 1, 1), corner(1, 1, 1), w[0]);
    let y0 = lerp(x00, x10, w[1]);
    let y1 = lerp(x01, x11, w[1]);
    lerp(y0, y1, w[2]).clamp(-ONE, ONE)
}

/// `n * amplitude / 65536` without overflow (`|n| <= 2^17`, `|amplitude| < 2^27`).
#[inline]
pub fn scale(n: i32, amplitude: i32) -> i32 {
    n.wrapping_mul(amplitude >> 16)
        .wrapping_add(n.wrapping_mul((amplitude & 0xffff) >> 4) >> 12)
}

/// Additive detail finer than about four level cells is omitted at `level`
/// (`level` already includes the grid's reference offset).
#[inline]
fn resolved(o: &Octave, level: u32) -> bool {
    o.kind <= 1 || o.shift >= level + 3
}

/// Terrain surface height (height units above the datum) of a level cell
/// column whose centre is at domain point `p`.
pub fn height(k: &FieldConstants, p: IVec3, level: u32) -> i32 {
    let count = k.header[1] as usize;
    let level = level + k.scale[1] as u32;
    // The first six octaves are the domain warp (two per axis). The warp is a
    // coordinate transform, so every level evaluates it.
    let mut warp = [0i32; 3];
    for o in &k.octaves[..WARP_OCTAVES] {
        let axis = (o.kind - 4) as usize;
        warp[axis] = warp[axis].wrapping_add(scale(noise(p, o.shift, o.seed), o.amplitude));
    }
    let q = p + IVec3::from_array(warp);
    let mut continent = 0i32;
    let mut mask = 0i32;
    let mut ridged = 0i32;
    let mut ridge_weight = ONE - 1;
    let mut detail = 0i32;
    for o in &k.octaves[WARP_OCTAVES..count] {
        if !resolved(o, level) {
            continue;
        }
        let n = noise(if o.kind == 7 { p } else { q }, o.shift, o.seed);
        match o.kind {
            0 => continent = continent.wrapping_add(scale(n, o.amplitude)),
            1 => mask = mask.wrapping_add(scale(n, o.amplitude)),
            2 => {
                let r = (ONE - n.abs()).clamp(0, ONE - 1) as u32;
                let r2 = r.wrapping_mul(r) >> 16;
                let v = (r2.wrapping_mul(ridge_weight as u32) >> 16) as i32;
                ridge_weight = v.wrapping_mul(2).clamp(ONE / 4, ONE - 1);
                ridged = ridged.wrapping_add(scale(v, o.amplitude));
            }
            _ => detail = detail.wrapping_add(scale(n, o.amplitude)),
        }
    }
    // Continents: c in about [-1.9, 1.9] (Q16); shape basin/lowland transition.
    let c = continent;
    let base = if c < 0 {
        // Continental shelf then deep ocean.
        let t = (-c).min(ONE);
        scale(t, k.levels[0]).wrapping_add(scale(ONE - t, k.levels[1] / 8))
    } else {
        let t = (c * 2).min(ONE);
        scale(t, k.levels[1])
    };
    // Mountains rise only on land, inside the mountain-region mask.
    let land = (c * 3).clamp(0, ONE);
    let region = (mask.wrapping_sub(k.misc[1]) * 3).clamp(0, ONE);
    let mountains = scale(land, scale(region, ridged));
    // Land detail fades out under deep water.
    let wet = (ONE + c * 2).clamp(ONE / 8, ONE);
    base.wrapping_add(mountains).wrapping_add(scale(wet, detail))
}

/// Cell layer index of the first air cell above the column surface.
#[inline]
pub fn top_cells(k: &FieldConstants, height: i32, level: u32) -> i32 {
    height.div_euclid(k.header[2]) >> level
}

/// Canonical terrain kind at a level cell before edits: 0 air, 1 solid.
#[inline]
pub fn terrain_kind(top: i32, k: i32) -> u32 {
    u32::from(k < top)
}

/// Moisture in Q12 [0, 4096] from very low-frequency noise at `p`.
pub fn moisture(k: &FieldConstants, p: IVec3) -> i32 {
    let o = k.octaves[6].shift.saturating_sub(1).max(1);
    (noise(p, o, (k.misc[3] as u32) ^ 0x51ED_270B) + ONE) / 2
}

/// Material of a solid ground cell. `top_height` is the column height (mm),
/// `depth` cells below the column top (0 = exposed top cell), `slope` the
/// largest top difference (cells) to a neighbour inside the same 8x8 column
/// block, `layer` the base layer index of the cell.
pub fn ground_material(
    c: &FieldConstants,
    p: IVec3,
    top_height: i32,
    depth: i32,
    slope: i32,
    layer: i32,
) -> u32 {
    use material::*;
    let dirt = c.header[3];
    let steep = slope >= c.misc[2];
    let wet = moisture(c, p);
    // Hash every domain axis: on a face one of them is nearly constant.
    let h = hash3(p.x, p.y, p.z ^ layer.wrapping_mul(0x9e37), 0x2545_F491);
    let altitude = layer.wrapping_mul(c.header[2]);
    if top_height < c.levels[3] {
        // Low basins: meadow with mud and sand patches over silt, gravel
        // and stone. Surface variation hashes position only, so it never
        // lines up with height contours.
        if depth == 0 {
            let s = hash3(p.x, p.y, p.z, 0x5f35_6495);
            return if s & 15 == 0 {
                DIRT
            } else if (s >> 4) & 31 == 0 {
                SAND
            } else {
                GRASS
            };
        }
        return if depth < dirt * 2 {
            if h & 3 == 0 { GRAVEL } else { CLAY }
        } else {
            STONE
        };
    }
    let snowline = c.levels[2] + scale(wet - ONE / 2, c.levels[2] / 4);
    if top_height > snowline && depth < dirt && !steep {
        return SNOW;
    }
    if wet < ONE * 3 / 10 {
        // Dry lands: sand over banded sandstone and clay.
        if depth < dirt && !steep {
            return SAND;
        }
        let band = altitude.div_euclid(2_100).rem_euclid(5);
        return if band == 1 || band == 3 { CLAY } else { SANDSTONE };
    }
    if steep {
        return if depth < 1 && h & 7 == 0 {
            DIRT
        } else if altitude.div_euclid(900) & 1 == 0 {
            STONE
        } else {
            DARK_STONE
        };
    }
    if depth == 0 {
        GRASS
    } else if depth < dirt {
        DIRT
    } else if depth < dirt * 3 && h & 3 == 0 {
        GRAVEL
    } else if altitude.div_euclid(1_300) & 1 == 0 {
        STONE
    } else {
        DARK_STONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_is_bounded_and_continuous_at_lattice_points() {
        let mut max = 0;
        for x in -300..300 {
            let p = IVec3::new(x * 37, x * -11 + 5, x * 3);
            let v = noise(p, 6, 99);
            max = max.max(v.abs());
            let a = noise(IVec3::new(64 * x, 0, 0), 6, 5);
            let b = noise(IVec3::new(64 * x + 1, 0, 0), 6, 5);
            assert!((a - b).abs() < 6400, "{a} {b}");
        }
        assert!(max > 16000 && max <= 65536, "{max}");
    }

    #[test]
    fn height_range_is_planetary_and_levels_agree_on_large_scale() {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        let k = FieldConstants::new(&grid, &Landform::default());
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        let n = grid.cells();
        for s in 0..400 {
            let i = (s * 7919 % 400) * (n / 400);
            let j = (s * 104_729 % 400) * (n / 400);
            let p = grid.domain_point((s % 6) as u8, i, j, 0);
            let h0 = height(&k, p, 0);
            let h12 = height(&k, p, 12);
            lo = lo.min(h0);
            hi = hi.max(h0);
            // Band limiting only removes octaves shorter than the level.
            let metres = f64::from((h0 - h12).abs()) / 1000.0;
            assert!(metres < 400.0, "{metres}");
        }
        let lo_m = f64::from(lo) / 1000.0;
        let hi_m = f64::from(hi) / 1000.0;
        assert!(lo_m < -200.0 && hi_m > 100.0 && hi_m < 6_000.0, "{lo_m} {hi_m}");
    }
}



#[cfg(test)]
mod continuity {
    use super::*;
    #[test]
    fn noise_is_continuous_at_fine_steps() {
        for shift in [8u32, 12, 16, 19, 20, 24] {
            let mut worst = 0;
            for x in -600_000..-300_000 {
                let p = IVec3::new(x * 3 + 7, 99_614_720, -5_975_683);
                let d = (noise(p, shift, 12345) - noise(p + IVec3::X * 2, shift, 12345)).abs();
                worst = worst.max(d);
            }
            // Two domain units at the measured gradient bound (5.3 per lattice
            // spacing) plus rounding.
            let bound = (2.0 * 5.3 * 65536.0 / f64::from(1u32 << shift)).ceil() as i32 + 8;
            assert!(worst <= bound, "shift {shift}: {worst} > {bound}");
        }
    }
}
