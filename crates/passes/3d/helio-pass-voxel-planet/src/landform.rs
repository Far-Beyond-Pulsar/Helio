//! The built-in landform generator (`helio.landform`): continents, ocean
//! basins, ridged mountain ranges, hills and metre-scale roughness, with
//! meadows, dry lands, rock outcrops, strata and snow. Also the flat
//! generator (`helio.flat`). `landform.wgsl` and `flat.wgsl` mirror them.
//!
//! Every operation is wrapping two's-complement integer arithmetic, so CPU
//! and GPU agree to the bit. Additive octaves finer than a column footprint
//! are omitted: coarse levels are band-limited point samples of the same
//! field rather than an independent smooth replacement.
use crate::grid::Grid;
use crate::noise::{hash3, mul_fine, noise, noise_fine, scale, FINE_ONE, ONE};
use crate::terrain::{material, GeneratorInfo, TerrainField, TerrainGenerator, TerrainProgram, HEIGHT_ONE};
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;

pub const ID: &str = "helio.landform";
pub const VERSION: u32 = 1;
pub(crate) const DISPLAY_PROGRAM: &str = "helio.landform/1-ridge-envelope/1";
pub const FLAT_ID: &str = "helio.flat";
pub const FLAT_VERSION: u32 = 1;

/// Landform settings in metres (the generator's settings JSON). The same
/// settings produce a similar world at every supported voxel size.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Landform {
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
            continent_km: 3_000.0,
            ocean_depth_m: 2_400.0,
            lowland_m: 180.0,
            mountain_m: 2_400.0,
            mountain_km: 20.0,
            hill_m: 140.0,
            hill_km: 9.0,
            roughness: 0.035,
            warp_km: 40.0,
            snowline_m: 3_000.0,
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

/// `TerrainConstants` of `landform.wgsl` (uniform layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct LandformConstants {
    /// octave count, layer thickness (mm), dirt depth (cells), seed.
    pub header: [i32; 4],
    /// basin floor, lowland, snowline, basin threshold (mm).
    pub levels: [i32; 4],
    /// mountain mask bias, steep slope (cells/cell), pad, pad.
    pub shape: [i32; 4],
    pub octaves: [Octave; OCTAVES],
}

impl LandformConstants {
    pub fn new(grid: &Grid, land: &Landform, seed: u32) -> Self {
        let units = |metres: f64| (metres * f64::from(HEIGHT_ONE)).round() as i32;
        // Lattice spacing for a wavelength, in reference half cells.
        let half = crate::grid::REFERENCE_VOXEL * 0.5;
        let shift = |metres: f64| ((metres / half).log2().round().clamp(1.0, 29.0)) as u32;
        let mut octaves = Vec::new();
        let mut state = seed.wrapping_mul(0x9E37_79B9);
        let mut next_seed = || {
            state = state.wrapping_add(0x6D2B_79F5);
            state
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
        Self {
            header: [
                count,
                grid.layer_mm() as i32,
                ((0.7 / grid.voxel_size()).round() as i32).max(1),
                seed as i32,
            ],
            levels: [
                units(-land.ocean_depth_m),
                units(land.lowland_m),
                units(land.snowline_m),
                units(-8.0),
            ],
            shape: [ONE / 20, 16, 0, 0],
            octaves: table,
        }
    }

    /// Conservative per-level surface excess, in level cells (see
    /// [`TerrainField::bound_margins`]).
    ///
    /// A finer level adds octaves that this level omits (each bounded by its
    /// amplitude) and the resolved field varies inside the cell by at most its
    /// Lipschitz constant times the half diagonal. The noise gradient bound
    /// `G` is 1.5x the measured maximum of the fixed-point gradient noise
    /// (5.3 per lattice spacing); `check_field` samples the result.
    pub fn bound_margins(&self, grid: &Grid) -> [i32; 24] {
        const G: f64 = 8.0;
        let count = self.header[0] as usize;
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
            let effective = level + grid.level_offset();
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
            let cell_mm = f64::from(self.header[1]) * 2f64.powi(level as i32);
            out[level as usize] = ((excess_mm / cell_mm).ceil() as i64 + 2).clamp(2, 1 << 20) as i32;
        }
        out
    }

    /// Conservative lowest and highest surface height (height units).
    pub fn height_range(&self) -> (i32, i32) {
        let mut sum = i64::from(self.levels[1].abs());
        for o in &self.octaves[WARP_OCTAVES..self.header[0] as usize] {
            if o.kind >= 2 {
                sum += i64::from(o.amplitude.abs());
            }
        }
        let hi = sum + 10 * i64::from(HEIGHT_ONE);
        let lo = -hi - i64::from(self.levels[0].abs());
        (lo.max(i64::from(i32::MIN / 2)) as i32, hi.min(i64::from(i32::MAX / 2)) as i32)
    }
}

/// Additive detail finer than about four level cells is omitted at `level`
/// (`level` counts reference cells).
#[inline]
fn resolved(o: &Octave, level: u32) -> bool {
    o.kind <= 1 || o.shift >= level + 3
}

/// Surface height (height units above the datum) of the column centred at
/// domain point `p` with a `2^level` reference cell footprint.
pub fn height(k: &LandformConstants, p: IVec3, level: u32) -> i32 {
    let count = k.header[0] as usize;
    // The warp, continents, mountain regions and ridges scale up to
    // kilometres, so they use the fine (Q24) noise: 16-bit noise is constant
    // over metres at these wavelengths and its steps, multiplied by the
    // mountains, would cut terraces between neighbouring columns.
    //
    // The first six octaves are the domain warp (two per axis). The warp is a
    // coordinate transform, so every level evaluates it.
    let mut warp = [0i32; 3];
    for o in &k.octaves[..WARP_OCTAVES] {
        let axis = (o.kind - 4) as usize;
        warp[axis] = warp[axis].wrapping_add(mul_fine(o.amplitude, noise_fine(p, o.shift, o.seed)));
    }
    let q = p + IVec3::from_array(warp);
    let mut continent = 0i32;
    let mut mask = 0i32;
    let mut ridged = 0i32;
    let mut ridge_weight = FINE_ONE - 1;
    let mut detail = 0i32;
    for o in &k.octaves[WARP_OCTAVES..count] {
        if !resolved(o, level) {
            continue;
        }
        match o.kind {
            0 => continent = continent.wrapping_add(mul_fine(noise_fine(q, o.shift, o.seed), o.amplitude << 8)),
            1 => mask = mask.wrapping_add(mul_fine(noise_fine(q, o.shift, o.seed), o.amplitude << 8)),
            2 => {
                let n = noise_fine(q, o.shift, o.seed);
                let r = (FINE_ONE - n.abs()).clamp(0, FINE_ONE - 1);
                let v = mul_fine(mul_fine(r, r), ridge_weight);
                ridge_weight = (v * 2).clamp(FINE_ONE / 4, FINE_ONE - 1);
                ridged = ridged.wrapping_add(mul_fine(o.amplitude, v));
            }
            _ => detail = detail.wrapping_add(scale(noise(if o.kind == 7 { p } else { q }, o.shift, o.seed), o.amplitude)),
        }
    }
    // Continents: c in about [-1.9, 1.9] (Q24); shape basin/lowland transition.
    let c = continent;
    let base = if c < 0 {
        // Continental shelf then deep ocean.
        let t = (-c).min(FINE_ONE);
        mul_fine(k.levels[0], t).wrapping_add(mul_fine(k.levels[1] / 8, FINE_ONE - t))
    } else {
        let t = (c * 2).min(FINE_ONE);
        mul_fine(k.levels[1], t)
    };
    // Mountains rise only on land, inside the mountain-region mask.
    let land = (c * 3).clamp(0, FINE_ONE);
    let region = ((mask - (k.shape[0] << 8)) * 3).clamp(0, FINE_ONE);
    let mountains = mul_fine(mul_fine(ridged, region), land);
    // Land detail fades out under deep water.
    let wet = (FINE_ONE + c * 2).clamp(FINE_ONE / 8, FINE_ONE);
    base.wrapping_add(mountains).wrapping_add(mul_fine(detail, wet))
}

/// Moisture in Q24 [0, FINE_ONE] from very low-frequency noise at `p`
/// (fine, so dry-land edges follow smooth curves).
pub fn moisture(k: &LandformConstants, p: IVec3) -> i32 {
    let o = k.octaves[6].shift.saturating_sub(1).max(1);
    (noise_fine(p, o, (k.header[3] as u32) ^ 0x51ED_270B) + FINE_ONE) / 2
}

/// Strata altitude (mm): layers undulate +-8 m over ~100 m, so cuts
/// through them never show flat rings.
fn strata(c: &LandformConstants, p: IVec3, altitude: i32) -> i32 {
    altitude + scale(noise(p, 11, (c.header[3] as u32) ^ 0x9B05_688C), 8_000)
}

/// Material of a solid ground cell. `top_height` is the column height (mm),
/// `depth` cells below the column top (0 = exposed top cell), `slope` the
/// ground slope across the cell's 8x8 column block in eighths of a cell per
/// cell, `layer` the base layer index of the cell.
pub fn ground_material(
    c: &LandformConstants,
    p: IVec3,
    top_height: i32,
    depth: i32,
    slope: i32,
    layer: i32,
) -> u32 {
    use crate::terrain::material::*;
    let dirt = c.header[2];
    let steep = slope >= c.shape[1];
    let wet = moisture(c, p);
    // Hash every domain axis: on a face one of them is nearly constant.
    let h = hash3(p.x, p.y, p.z ^ layer.wrapping_mul(0x9e37), 0x2545_F491);
    let altitude = layer.wrapping_mul(c.header[1]);
    if top_height < c.levels[3] {
        // Low basins: meadow with mud and sand patches over silt, gravel
        // and stone. Surface variation hashes position only, so it never
        // lines up with height contours.
        if depth == 0 {
            let s = hash3(p.x, p.y, p.z, 0x5f35_6495);
            return if s & 15 == 0 {
                DIRT | SPECK
            } else if (s >> 4) & 31 == 0 {
                SAND | SPECK
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
    let snowline = c.levels[2] + mul_fine(c.levels[2] / 4, wet - FINE_ONE / 2);
    // Alpine weight: 0 below the rockline, ONE at the snowline.
    let rockline = snowline - c.levels[2] / 3;
    let band = (snowline - rockline).max(256);
    let alpine = ((top_height - rockline).clamp(0, band) / 256) * ONE / (band / 256);
    // Rock patches (~50 m and ~6 m octaves), also breaking up snow edges.
    // Noise is clamped to +-ONE, so below the rockline on gentler slopes no
    // outcrop can reach the rock fringe: skip it (same result).
    let seed = c.header[3] as u32;
    let outcrop = if alpine > 0 || slope >= 5 {
        noise(p, 10, seed ^ 0x1B56_C4E9) + noise(p, 7, seed ^ 0x6A09_E667) / 3
    } else {
        -ONE
    };
    // Snow does not hold on faces steeper than ~37 degrees: rock streaks the snowfields.
    if top_height > snowline && depth < dirt && slope + outcrop / 8192 < 6 {
        return SNOW;
    }
    if wet < FINE_ONE / 10 * 3 {
        // Dry lands: sand over banded sandstone and clay.
        if depth < dirt && !steep {
            return SAND;
        }
        let band = strata(c, p, altitude).div_euclid(2_100).rem_euclid(5);
        return if band == 1 || band == 3 { CLAY } else { SANDSTONE };
    }
    // Rock shows through the turf in the outcrop patches, which grow up the
    // alpine band below the snowline and on hillsides over ~32 degrees; scree and
    // bare soil fringe them. Deeper cells keep the strata.
    let mut exposed = outcrop + 2 * alpine - ONE;
    if slope >= 5 {
        exposed += ONE / 2;
    }
    if steep || (exposed > 0 && depth < dirt) {
        return if depth < 1 && h & 7 == 0 {
            DIRT
        } else if (altitude + scale(outcrop, 3_000)).div_euclid(4_500) & 1 == 0 {
            STONE
        } else {
            DARK_STONE
        };
    }
    if depth == 0 {
        if exposed > -ONE / 16 {
            GRAVEL
        } else if exposed > -ONE / 8 {
            DIRT
        } else {
            GRASS
        }
    } else if depth < dirt {
        DIRT
    } else if depth < dirt * 3 && h & 3 == 0 {
        GRAVEL
    } else if strata(c, p, altitude).div_euclid(12_000) & 1 == 0 {
        STONE
    } else {
        DARK_STONE
    }
}

/// Builds [`LandformField`]s from [`Landform`] settings.
pub struct LandformGenerator;

impl TerrainGenerator for LandformGenerator {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: ID.into(),
            version: VERSION,
            name: "Landform".into(),
            description: "Continents, ocean basins, mountain ranges and hills with meadows, dry lands, rock, strata and snow.".into(),
            settings_component: Some("VoxelLandformComponent".into()),
        }
    }
    fn build(&self, grid: &Grid, seed: u64, settings: &str) -> Result<Arc<dyn TerrainField>, String> {
        let land: Landform = if settings.trim().is_empty() {
            Landform::default()
        } else {
            serde_json::from_str(settings).map_err(|e| format!("invalid landform settings: {e}"))?
        };
        let positive = [land.continent_km, land.mountain_km, land.hill_km];
        let finite = [land.ocean_depth_m, land.lowland_m, land.mountain_m, land.hill_m, land.roughness, land.warp_km, land.snowline_m];
        if positive.iter().any(|v| !v.is_finite() || *v <= 0.0) || finite.iter().any(|v| !v.is_finite()) {
            return Err("landform settings must be finite, with positive wavelengths".into());
        }
        Ok(Arc::new(LandformField::new(grid, &land, (seed ^ (seed >> 32)) as u32)))
    }
}

/// A [`Landform`] on one grid.
pub struct LandformField {
    constants: LandformConstants,
    bounds: [i32; 24],
    render_bounds: [i32; 24],
    ridge_suffix: Option<[[i32; 4]; 66]>,
}

impl LandformField {
    pub fn new(grid: &Grid, land: &Landform, seed: u32) -> Self {
        let constants = LandformConstants::new(grid, land, seed);
        let bounds = constants.bound_margins(grid);
        // Unsupported recipes retain the canonical path. In particular, do
        // not saturate already-invalid huge finite amplitudes into new terrain.
        let ridge_suffix = crate::ridge_envelope::bake_ridge_suffix(grid, &constants).ok();
        let render_bounds = if ridge_suffix.is_some() {
            crate::ridge_envelope::render_bounds(grid, &constants, bounds)
        } else { bounds };
        Self { bounds, render_bounds, ridge_suffix, constants }
    }
    pub fn constants(&self) -> &LandformConstants {
        &self.constants
    }
}

impl TerrainField for LandformField {
    fn height(&self, p: IVec3, level: u32) -> i32 {
        height(&self.constants, p, level)
    }
    fn ground_material(&self, p: IVec3, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
        ground_material(&self.constants, p, top_height, depth, slope, layer)
    }
    fn height_range(&self) -> (i32, i32) {
        self.constants.height_range()
    }
    fn bound_margins(&self) -> [i32; 24] {
        self.bounds
    }
    fn render_bound_margins(&self) -> [i32; 24] {
        self.render_bounds
    }
    fn program(&self) -> TerrainProgram {
        let mut canonical = self.constants;
        canonical.shape[2] = i32::from(self.ridge_suffix.is_some());
        let mut constants = bytemuck::bytes_of(&canonical).to_vec();
        constants.extend_from_slice(bytemuck::cast_slice(&self.ridge_suffix.unwrap_or([[0; 4]; 66])));
        TerrainProgram {
            key: Cow::Borrowed(DISPLAY_PROGRAM),
            wgsl: Cow::Borrowed(include_str!("../shaders/landform.wgsl")),
            constants,
        }
    }
}

/// Settings of the flat generator: level ground at `height_m` with a
/// surface layer over soil over rock.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Flat {
    pub height_m: f64,
    pub soil_depth_m: f64,
    /// Material names (see [`material::NAMES`]).
    pub surface: String,
    pub soil: String,
    pub rock: String,
}

impl Default for Flat {
    fn default() -> Self {
        Self { height_m: 0.0, soil_depth_m: 1.0, surface: "Grass".into(), soil: "Dirt".into(), rock: "Stone".into() }
    }
}

/// Builds [`FlatField`]s from [`Flat`] settings.
pub struct FlatGenerator;

impl TerrainGenerator for FlatGenerator {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: FLAT_ID.into(),
            version: FLAT_VERSION,
            name: "Flat".into(),
            description: "Level ground: a surface layer over soil over rock.".into(),
            settings_component: Some("VoxelFlatTerrainComponent".into()),
        }
    }
    fn build(&self, grid: &Grid, _seed: u64, settings: &str) -> Result<Arc<dyn TerrainField>, String> {
        let flat: Flat = if settings.trim().is_empty() {
            Flat::default()
        } else {
            serde_json::from_str(settings).map_err(|e| format!("invalid flat terrain settings: {e}"))?
        };
        if !flat.height_m.is_finite() || flat.height_m.abs() > 1.0e6 || !flat.soil_depth_m.is_finite() || flat.soil_depth_m < 0.0 {
            return Err("flat terrain height and soil depth must be finite (height within 1000 km)".into());
        }
        let id = |name: &str| material::from_name(name).ok_or_else(|| format!("unknown flat terrain material {name:?}"));
        let (surface, soil, rock) = (id(&flat.surface)?, id(&flat.soil)?, id(&flat.rock)?);
        let layer = grid.layer_mm() as i32;
        // Whole layers, so the surface is exactly one cell boundary.
        let height = ((flat.height_m * f64::from(HEIGHT_ONE)).round() as i32).div_euclid(layer) * layer;
        let depth = (flat.soil_depth_m / grid.voxel_size()).round() as i32;
        Ok(Arc::new(FlatField { constants: [height, surface as i32, soil as i32, depth, rock as i32, 0, 0, 0] }))
    }
}

/// `TerrainConstants` of `flat.wgsl`: height, surface, soil, soil depth
/// (cells), rock.
pub struct FlatField {
    constants: [i32; 8],
}

impl TerrainField for FlatField {
    fn height(&self, _p: IVec3, _level: u32) -> i32 {
        self.constants[0]
    }
    fn ground_material(&self, _p: IVec3, _top_height: i32, depth: i32, _slope: i32, _layer: i32) -> u32 {
        let c = &self.constants;
        (if depth == 0 { c[1] } else if depth <= c[3] { c[2] } else { c[4] }) as u32
    }
    fn height_range(&self) -> (i32, i32) {
        (self.constants[0], self.constants[0])
    }
    fn bound_margins(&self) -> [i32; 24] {
        [2; 24]
    }
    fn program(&self) -> TerrainProgram {
        TerrainProgram {
            key: Cow::Borrowed("helio.flat/1"),
            wgsl: Cow::Borrowed(include_str!("../shaders/flat.wgsl")),
            constants: bytemuck::cast_slice(&self.constants).to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn climate_bounds_cover_both_signs_of_canonical_height_change() {
        for voxel in [0.1, 0.3, 1.0] {
            let grid = Grid::new(6_371_000.0, voxel).unwrap();
            let k = LandformConstants::new(&grid, &Landform::default(), 7);
            let bounds = k.bound_margins(&grid);
            let mut rng = 0x2545_F491_4F6C_DD1Du64;
            let mut next = || {
                rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng
            };
            for _ in 0..5000 {
                let level = 1 + (next() % u64::from(grid.levels() - 1)) as u32;
                let face = (next() % 6) as u8;
                let cells = (grid.cells() >> level).max(1) as u64;
                let i = (next() % cells) as i32;
                let j = (next() % cells) as i32;
                let coarse = height(&k, grid.domain_point(face, i, j, level), level + grid.level_offset());
                let top = coarse.div_euclid(k.header[1]) >> level;
                let fi = (i << level) + (next() % (1u64 << level)) as i32;
                let fj = (j << level) + (next() % (1u64 << level)) as i32;
                let canonical = i64::from(height(&k, grid.domain_point(face, fi, fj, 0), grid.level_offset()));
                let cell_mm = i64::from(k.header[1]) * (1i64 << level);
                let lo = i64::from(top - bounds[level as usize]) * cell_mm;
                let hi = i64::from(top + bounds[level as usize]) * cell_mm;
                assert!((lo..=hi).contains(&canonical), "{voxel} m level {level}: {canonical} outside {lo}..={hi}");
            }
        }
    }

    #[test]
    fn height_range_is_planetary_and_levels_agree_on_large_scale() {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        let k = LandformConstants::new(&grid, &Landform::default(), 7);
        let (range_lo, range_hi) = k.height_range();
        let mut lo = i32::MAX;
        let mut hi = i32::MIN;
        let n = grid.cells();
        let offset = grid.level_offset();
        for s in 0..400 {
            let i = (s * 7919 % 400) * (n / 400);
            let j = (s * 104_729 % 400) * (n / 400);
            let p = grid.domain_point((s % 6) as u8, i, j, 0);
            let h0 = height(&k, p, offset);
            let h12 = height(&k, p, 12 + offset);
            lo = lo.min(h0);
            hi = hi.max(h0);
            // Band limiting only removes octaves shorter than the level.
            let metres = f64::from((h0 - h12).abs()) / 1000.0;
            assert!(metres < 400.0, "{metres}");
        }
        let lo_m = f64::from(lo) / 1000.0;
        let hi_m = f64::from(hi) / 1000.0;
        assert!(lo_m < -200.0 && hi_m > 100.0 && hi_m < 6_000.0, "{lo_m} {hi_m}");
        assert!(range_lo <= lo && hi <= range_hi);
    }

    /// Neighbouring columns never step by a voxel layer: the second
    /// difference of heights across three adjacent 0.1 m columns stays below
    /// 100 mm. Low-frequency weights multiply kilometres of relief, so with
    /// 16-bit noise their quantization cut straight terraces up to a metre
    /// high across the land (1791 of these samples). What remains is the
    /// integer domain warp moving a steep slope by one extra domain unit
    /// (0.05 m), under 80 mm.
    #[test]
    fn the_field_has_no_steps_between_neighbouring_columns() {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        let k = LandformConstants::new(&grid, &Landform::default(), 7);
        let n = grid.cells() as u64;
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let offset = grid.level_offset();
        let (mut worst, mut steps) = (0, 0);
        for s in 0..20_000 {
            let face = (s % 6) as u8;
            let (i, j) = ((next() % (n - 4)) as i32, (next() % (n - 4)) as i32);
            // Along i and along j.
            for d in [(1, 0), (0, 1)] {
                let h = |t: i32| height(&k, grid.domain_point(face, i + d.0 * t, j + d.1 * t, 0), offset);
                let second = (h(0) - 2 * h(1) + h(2)).abs();
                worst = worst.max(second);
                steps += usize::from(second >= 100);
            }
        }
        assert!(steps == 0, "{steps} steps, worst second difference {worst} mm");
    }

    #[test]
    fn flat_ground_is_one_whole_layer_with_its_materials() {
        let grid = Grid::plane(crate::grid::Shape::Plane, 1024.0, 0.1).unwrap();
        let settings = r#"{"height_m": 2.34, "soil_depth_m": 0.5, "surface": "Sand", "rock": "dark_stone"}"#;
        let field = FlatGenerator.build(&grid, 0, settings).unwrap();
        assert_eq!(field.height(IVec3::ZERO, 0), 2_300);
        assert_eq!(field.ground_material(IVec3::ZERO, 2_300, 0, 0, 22), material::SAND);
        assert_eq!(field.ground_material(IVec3::ZERO, 2_300, 5, 0, 17), material::DIRT);
        assert_eq!(field.ground_material(IVec3::ZERO, 2_300, 6, 0, 16), material::DARK_STONE);
        assert!(FlatGenerator.build(&grid, 0, r#"{"rock": "Air"}"#).is_err());
    }
}
