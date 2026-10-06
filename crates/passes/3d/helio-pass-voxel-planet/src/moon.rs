//! The built-in moon generator (`helio.moon`): cratered highlands and dark
//! basalt maria, regolith over bedrock, no water or vegetation. `moon.wgsl`
//! mirrors it; like every generator it uses only the integer noise library,
//! so CPU and GPU agree to the bit.
//!
//! Craters come in octaves, each half the size of the previous one. An
//! octave has one candidate crater per cell of a 3D lattice, at a hashed
//! point; it exists with the octave's density when that point lies within
//! half a cell of the surface (sphere or plane), so craters are seamless
//! across cube edges. Each has a parabolic bowl, a sharp rim and an ejecta
//! blanket out to twice its radius; young craters (`fresh_share`) show
//! bright ejecta, recorded in the surface word with the mare share.
use crate::grid::{Grid, DOMAIN_UNIT};
use crate::landform::RESOLVED_SHIFT;
use crate::noise::{hash3, mul16, mul_fine, mul_shr, mul_shr_signed, noise_fine, scale, unit_q30, FINE_ONE, Q30};
use crate::terrain::{GeneratorInfo, MaterialAppearance, TerrainAppearance, TerrainField, TerrainGenerator, TerrainProgram, HEIGHT_ONE, MATERIALS};
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::sync::Arc;

pub const ID: &str = "helio.moon";
pub const VERSION: u32 = 1;
const PROGRAM: &str = "helio.moon/1";
/// Largest number of crater octaves.
pub const CRATER_OCTAVES: usize = 12;

/// Moon materials: ids into its material table ([`TerrainField::appearance`]).
pub mod material {
    pub const REGOLITH: u32 = 1;
    pub const MARE: u32 = 2;
    pub const EJECTA: u32 = 3;
    pub const ROCK: u32 = 4;
    pub const BASALT: u32 = 5;
    pub const ANORTHOSITE: u32 = 6;
}

/// Moon settings in metres (the generator's settings JSON).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Moon {
    /// Height and wavelength of the rolling highlands.
    pub highland_m: f64,
    pub highland_km: f64,
    /// Rough share of the surface in dark basalt maria, their size, and how
    /// far they lie below the highlands.
    pub mare_share: f64,
    pub mare_km: f64,
    pub mare_depth_m: f64,
    /// Diameter of the largest craters, and the number of crater octaves
    /// (each half the size of the previous one).
    pub crater_km: f64,
    pub crater_octaves: u32,
    /// Share of lattice cells holding a crater at the largest size, and its
    /// growth per octave (small craters are more common).
    pub crater_density: f64,
    pub crater_density_growth: f64,
    /// Depth of a simple crater as a fraction of its diameter; craters over
    /// 15 km are shallower.
    pub crater_depth_ratio: f64,
    /// Rim height as a fraction of the crater depth.
    pub crater_rim_ratio: f64,
    /// Share of craters young enough to show bright ejecta.
    pub fresh_share: f64,
    /// Regolith depth over the bedrock.
    pub regolith_m: f64,
}

impl Default for Moon {
    fn default() -> Self {
        Self {
            highland_m: 1_500.0,
            highland_km: 250.0,
            mare_share: 0.3,
            mare_km: 900.0,
            mare_depth_m: 1_200.0,
            crater_km: 40.0,
            crater_octaves: 10,
            crater_density: 0.3,
            crater_density_growth: 1.25,
            crater_depth_ratio: 0.2,
            crater_rim_ratio: 0.3,
            fresh_share: 0.15,
            regolith_m: 4.0,
        }
    }
}

/// One crater octave: lattice shift, depth (mm) of its largest craters,
/// density (Q16 share of cells) and seed.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct CraterOctave {
    pub shift: u32,
    pub depth: i32,
    pub density: u32,
    pub seed: u32,
}

/// `TerrainConstants` of `moon.wgsl` (uniform layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct MoonConstants {
    /// crater octave count, layer (mm), regolith depth (cells), plane (1).
    pub header: [i32; 4],
    /// highland shift, highland amplitude (mm), mare shift, mare threshold (Q16).
    pub shape: [i32; 4],
    /// mare depth (mm), rim / depth (Q16), fresh share (Q16), seed.
    pub surface: [i32; 4],
    /// domain radius (0 on a plane), largest crater radius (Q19 of a lattice
    /// cell), pad, pad.
    pub radius: [u32; 4],
    pub craters: [CraterOctave; CRATER_OCTAVES],
}

/// Crater radius of the largest crater of a cell (Q19 of the lattice cell,
/// 0.3): with the ejecta at twice the radius, the 3x3x3 cells around a
/// point hold every crater reaching it.
const RADIUS_Q19: u32 = 157_286;
/// Squared ejecta reach in crater radii (Q24).
const REACH: i32 = 4 << 24;

/// `2^60 / d` for `d` in `[2^29, 2^30)`: Newton from a linear guess, four
/// fixed steps on exact 64-bit products.
#[inline]
fn recip_q30(d: u32) -> u32 {
    let mut y = 3_031_741_621u32 - mul_shr(d, 2_021_161_080, 30);
    for _ in 0..4 {
        y = mul_shr(y, 2 * Q30 - mul_shr(d, y, 30), 30);
    }
    y
}

/// `r2 / rad2` in Q24 (`r2 < 4 rad2`, `rad2 > 0`).
#[inline]
fn squared_ratio(r2: u32, rad2: u32) -> u32 {
    let bits = 32 - rad2.leading_zeros();
    mul_shr(r2, recip_q30(rad2 << (30 - bits)), 6 + bits)
}

/// Polynomial smooth minimum of `a` and `b` over a width `k` (mm).
#[inline]
fn smooth_min(a: i32, b: i32, k: i32) -> i32 {
    let m = a.min(b);
    let gap = (a - b).abs();
    if k <= 0 || gap >= k {
        return m;
    }
    // (k - gap) / k in Q16, exactly (two division steps in 32 bits).
    let n = (k - gap) << 8;
    let q = ((n / k) << 8) + ((n % k) << 8) / k;
    m - scale(mul16(q, q), k) / 4
}

impl MoonConstants {
    pub fn new(grid: &Grid, moon: &Moon, seed: u32) -> Self {
        let units = |metres: f64| (metres * f64::from(HEIGHT_ONE)).round() as i32;
        let shift = |metres: f64| ((metres / DOMAIN_UNIT).log2().round().clamp(10.0, 30.0)) as u32;
        let mut state = seed.wrapping_mul(0x9E37_79B9);
        let mut next_seed = || {
            state = state.wrapping_add(0x6D2B_79F5);
            state
        };
        let mut craters = [CraterOctave::default(); CRATER_OCTAVES];
        let mut count = 0;
        for o in 0..(moon.crater_octaves as usize).min(CRATER_OCTAVES) {
            let diameter = moon.crater_km * 1_000.0 / f64::from(1u32 << o);
            // A cell holds one crater of at most 0.3 cells radius.
            let cell = diameter / (2.0 * f64::from(RADIUS_Q19) / 524_288.0);
            if cell < DOMAIN_UNIT * 1024.0 {
                break;
            }
            let depth = moon.crater_depth_ratio * diameter * (15_000.0 / diameter).min(1.0).powf(0.7);
            let density = (moon.crater_density * moon.crater_density_growth.powi(o as i32)).clamp(0.0, 0.95);
            craters[count] = CraterOctave {
                shift: shift(cell),
                depth: units(depth),
                density: (density * 65_536.0) as u32,
                seed: next_seed(),
            };
            count += 1;
        }
        let threshold = crate::landform::noise_quantile(1.0 - moon.mare_share.clamp(0.0, 1.0));
        Self {
            header: [count as i32, grid.layer_mm() as i32, ((moon.regolith_m / grid.voxel_size()).round() as i32).max(1), i32::from(grid.is_plane())],
            shape: [
                shift(moon.highland_km * 1_000.0) as i32,
                units(moon.highland_m),
                shift(moon.mare_km * 1_000.0) as i32,
                (threshold * 65_536.0).round() as i32,
            ],
            surface: [
                units(moon.mare_depth_m),
                (moon.crater_rim_ratio * 65_536.0).round() as i32,
                (moon.fresh_share.clamp(0.0, 1.0) * 65_536.0).round() as i32,
                seed as i32,
            ],
            radius: [grid.sphere_constants()[2], RADIUS_Q19, 0, 0],
            craters,
        }
    }

    fn octaves(&self) -> &[CraterOctave] {
        &self.craters[..self.header[0] as usize]
    }
}

/// Detail finer than about four level cells is omitted at `level`.
fn resolved(shift: u32, level: u32) -> bool {
    shift >= level + RESOLVED_SHIFT
}

/// One crater octave at `p` (vertical `up`, Q30): height (mm) and the
/// freshest ejecta reaching `p` (0..255).
fn crater_octave(k: &MoonConstants, o: &CraterOctave, p: IVec3, up: IVec3) -> (i32, u32) {
    let s = o.shift;
    let c0 = IVec3::new(p.x >> s, p.y >> s, p.z >> s);
    let (mut height, mut ejecta) = (0i32, 0u32);
    for index in 0..27 {
        let cell = c0 + IVec3::new(index % 3 - 1, (index / 3) % 3 - 1, index / 9 - 1);
        let a = hash3(cell.x, cell.y, cell.z, o.seed);
        if a & 0xffff >= o.density {
            continue;
        }
        let b = hash3(cell.x, cell.y, cell.z, o.seed ^ 0x6C8E_9CF5);
        let jitter = IVec3::new((b & 1023) as i32, ((b >> 10) & 1023) as i32, ((b >> 20) & 1023) as i32);
        let centre = IVec3::new(
            (cell.x << s) + (jitter.x << (s - 10)),
            (cell.y << s) + (jitter.y << (s - 10)),
            (cell.z << s) + (jitter.z << (s - 10)),
        );
        // Only centres within half a cell of the surface.
        if k.header[3] != 0 {
            if centre.y.abs() >= 1 << (s - 1) {
                continue;
            }
        } else {
            let len2 = mul_shr(centre.x.unsigned_abs(), centre.x.unsigned_abs(), 30)
                + mul_shr(centre.y.unsigned_abs(), centre.y.unsigned_abs(), 30)
                + mul_shr(centre.z.unsigned_abs(), centre.z.unsigned_abs(), 30);
            let r2 = mul_shr(k.radius[0], k.radius[0], 30);
            if (len2 as i32 - r2 as i32).unsigned_abs() >= k.radius[0] >> (30 - s) {
                continue;
            }
        }
        // Squared horizontal distance in Q28 lattice cells (offsets in Q19,
        // exact 64-bit squares); nothing beyond a cell reaches `p`.
        let d = centre - p;
        let q = |v: i32| if s >= 19 { v >> (s - 19) } else { v << (19 - s) };
        let d = IVec3::new(q(d.x), q(d.y), q(d.z));
        if d.abs().max_element() >= 1 << 19 {
            continue;
        }
        // The centre may lie half a cell off the surface: the vertical part
        // cancels, so it keeps 10 more bits (Q29).
        let along = mul_shr_signed(d.x, up.x, 20) + mul_shr_signed(d.y, up.y, 20) + mul_shr_signed(d.z, up.z, 20);
        let square = |v: i32| mul_shr(v.unsigned_abs(), v.unsigned_abs(), 10);
        let along2 = mul_shr(along.unsigned_abs(), along.unsigned_abs(), 30);
        let r2 = (square(d.x) + square(d.y) + square(d.z)).saturating_sub(along2);
        // Radius 0.55 to 1 of the largest; depth in proportion.
        let size = 36_045 + (((a >> 16) * 29_491) >> 16) as i32;
        let radius = mul16(k.radius[1] as i32, size).max(1);
        let rad2 = mul_shr(radius as u32, radius as u32, 10).max(1);
        if r2 >= 4 * rad2 {
            continue;
        }
        // (r / radius)^2 in Q24, below 4 (the ejecta reach): a full-precision
        // reciprocal, so large craters move smoothly between columns.
        let x2 = squared_ratio(r2, rad2) as i32;
        let depth = scale(size, o.depth);
        let rim = scale(k.surface[1], depth);
        // The bowl rises as x^2; the ejecta falls as t^3 from the rim to
        // twice the radius; a smooth minimum rounds the rim between them.
        let bowl = mul_fine(x2, depth + rim) - depth;
        let t = (REACH - x2) / 3;
        let ejecta_height = mul_fine(mul_fine(mul_fine(t, t), t), rim);
        height += smooth_min(bowl, ejecta_height, rim / 4);
        if (hash3(cell.x, cell.y, cell.z, o.seed ^ 0x1B87_3593) & 0xffff) < k.surface[2] as u32 {
            ejecta = ejecta.max(255 - ((x2 >> 16) * 255 / (REACH >> 16)) as u32);
        }
    }
    (height, ejecta)
}

/// Height (mm) and surface word of the column at `p` (`level` counts
/// reference cells): ejecta freshness in bits 0..7, mare in bit 7.
pub fn height_parts(k: &MoonConstants, p: IVec3, level: u32) -> (i32, u32) {
    // Fine (Q24) noise: 16-bit noise at these wavelengths is constant over
    // metres, and its steps times kilometres of relief cut terraces.
    let seed = k.surface[3] as u32;
    let mare = ((noise_fine(p, k.shape[2] as u32, seed ^ 0x2545_F491) - (k.shape[3] << 8)) * 4).clamp(0, FINE_ONE);
    let mut highland = 0;
    for (o, amplitude) in [(0u32, k.shape[1]), (1, k.shape[1] / 2), (2, k.shape[1] / 4)] {
        let shift = (k.shape[0] as u32).saturating_sub(o);
        if resolved(shift, level) {
            highland += mul_fine(amplitude, noise_fine(p, shift, seed ^ 0x51ED_270B ^ o));
        }
    }
    let mut height = mul_fine(highland, FINE_ONE - mare / 2) - mul_fine(k.surface[0], mare);
    let up = if k.header[3] != 0 { IVec3::new(0, Q30 as i32, 0) } else { unit_q30(p) };
    let mut ejecta = 0;
    for o in k.octaves() {
        if resolved(o.shift, level) {
            let (h, e) = crater_octave(k, o, p, up);
            height += h;
            ejecta = ejecta.max(e);
        }
    }
    (height, (ejecta >> 1) | (u32::from(mare >= FINE_ONE / 2) << 7))
}

/// Material of a solid moon cell (see [`TerrainField::ground_material`]).
pub fn ground_material(k: &MoonConstants, p: IVec3, surface: u32, depth: i32, slope: i32, layer: i32) -> u32 {
    use material::*;
    let ejecta = (surface & 0x7f) << 1;
    let mare = surface & 0x80 != 0;
    let h = hash3(p.x, p.y, p.z ^ layer.wrapping_mul(0x9e37), 0x2545_F491);
    if depth == 0 {
        if slope >= 12 || (slope >= 6 && h & 3 == 0) {
            return ROCK;
        }
        // Bright rays thin out away from the crater.
        if ejecta > (h & 0xff) {
            return EJECTA;
        }
        return if mare { MARE } else { REGOLITH };
    }
    if depth < k.header[2] {
        return if mare { MARE } else { REGOLITH };
    }
    if mare { BASALT } else { ANORTHOSITE }
}

/// Builds [`MoonField`]s from [`Moon`] settings.
pub struct MoonGenerator;

impl TerrainGenerator for MoonGenerator {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: ID.into(),
            version: VERSION,
            name: "Moon".into(),
            description: "Cratered highlands and dark basalt maria: regolith over bedrock, bright young ejecta, no water or vegetation.".into(),
            settings_component: Some("VoxelMoonComponent".into()),
        }
    }
    fn build(&self, grid: &Grid, seed: u64, settings: &str) -> Result<Arc<dyn TerrainField>, String> {
        let moon: Moon = if settings.trim().is_empty() {
            Moon::default()
        } else {
            serde_json::from_str(settings).map_err(|e| format!("invalid moon settings: {e}"))?
        };
        let lengths = [moon.highland_km, moon.mare_km, moon.crater_km, moon.crater_density_growth];
        let finite = [moon.highland_m, moon.mare_share, moon.mare_depth_m, moon.crater_density, moon.crater_depth_ratio, moon.crater_rim_ratio, moon.fresh_share, moon.regolith_m];
        if lengths.iter().any(|v| !v.is_finite() || *v <= 0.0) || finite.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("moon settings must be finite and non-negative, with positive sizes".into());
        }
        if moon.highland_m > 20_000.0 || moon.mare_depth_m > 20_000.0 || moon.crater_km > 2_000.0 {
            return Err("moon relief and craters must stay within 20 km and 2000 km".into());
        }
        let seed = (seed ^ (seed >> 32)) as u32;
        Ok(Arc::new(MoonField { constants: MoonConstants::new(grid, &moon, seed), grid: *grid }))
    }
}

pub struct MoonField {
    constants: MoonConstants,
    grid: Grid,
}

impl MoonField {
    pub fn constants(&self) -> &MoonConstants {
        &self.constants
    }

    /// Height excess of the octaves omitted at `level` (mm), and the
    /// resolved field's Lipschitz constant (mm per domain unit).
    fn level_bounds(&self, level: u32) -> (f64, f64) {
        let k = &self.constants;
        let (mut dropped, mut lipschitz) = (0.0, 0.0);
        let highland = f64::from(k.shape[1]);
        for (o, amplitude) in [(0u32, highland), (1, highland / 2.0), (2, highland / 4.0)] {
            let shift = (k.shape[0] as u32).saturating_sub(o);
            if resolved(shift, level) {
                lipschitz += amplitude * 8.0 / 2f64.powi(shift as i32);
            } else {
                dropped += amplitude;
            }
        }
        lipschitz += f64::from(k.surface[0]) * 4.0 * 8.0 / 2f64.powi(k.shape[2]);
        let rim = f64::from(k.surface[1]) / 65_536.0;
        for o in k.octaves() {
            // Up to two overlapping craters per octave reach a point.
            let relief = 2.0 * f64::from(o.depth) * (1.0 + rim);
            if resolved(o.shift, level) {
                // Bowl slope 2 (depth + rim) per radius, radius >= 0.165 cells.
                lipschitz += relief * 13.0 / 2f64.powi(o.shift as i32);
            } else {
                dropped += relief;
            }
        }
        (dropped, lipschitz)
    }
}

impl TerrainField for MoonField {
    fn height(&self, p: IVec3, level: u32) -> i32 {
        height_parts(&self.constants, p, level).0
    }
    fn surface(&self, p: IVec3, level: u32, _height: i32) -> u32 {
        height_parts(&self.constants, p, level).1
    }
    fn ground_material(&self, p: IVec3, surface: u32, _top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
        ground_material(&self.constants, p, surface, depth, slope, layer)
    }
    fn height_range(&self) -> (i32, i32) {
        let k = &self.constants;
        let rim = f64::from(k.surface[1]) / 65_536.0;
        let craters: f64 = k.octaves().iter().map(|o| 2.0 * f64::from(o.depth) * (1.0 + rim)).sum();
        let highland = 1.75 * f64::from(k.shape[1]);
        let lo = -(highland + f64::from(k.surface[0]) + craters) - 10_000.0;
        let hi = highland + craters + 10_000.0;
        (lo.max(f64::from(i32::MIN / 2)) as i32, hi.min(f64::from(i32::MAX / 2)) as i32)
    }
    fn bound_margins(&self) -> [i32; 24] {
        let grid = &self.grid;
        let ratio = f64::from(grid.reference_cells()) / f64::from(grid.cells());
        let cell = crate::grid::REFERENCE_VOXEL / DOMAIN_UNIT;
        std::array::from_fn(|level| {
            let (dropped, lipschitz) = self.level_bounds(level as u32 + grid.level_offset());
            let half_diagonal = 2f64.powi(level as i32) * cell * ratio * std::f64::consts::SQRT_2 * 0.5 * 1.05;
            let excess_mm = dropped + lipschitz * half_diagonal;
            let cell_mm = f64::from(self.constants.header[1]) * 2f64.powi(level as i32);
            ((excess_mm / cell_mm).ceil() as i64 + 2).clamp(2, 1 << 20) as i32
        })
    }
    fn appearance(&self) -> TerrainAppearance {
        moon_appearance()
    }
    fn program(&self) -> TerrainProgram {
        TerrainProgram {
            key: Cow::Borrowed(PROGRAM),
            wgsl: Cow::Borrowed(include_str!("../shaders/moon.wgsl")),
            constants: bytemuck::bytes_of(&self.constants).to_vec(),
        }
    }
}

/// The moon's material table: greys of regolith, mare, fresh ejecta,
/// boulders, basalt and anorthosite (sRGB).
pub fn moon_appearance() -> TerrainAppearance {
    use material::*;
    let mut table = TerrainAppearance::default();
    let colour = |rgb: [u8; 3], roughness: f32| MaterialAppearance {
        colour: [f32::from(rgb[0]) / 255.0, f32::from(rgb[1]) / 255.0, f32::from(rgb[2]) / 255.0, roughness],
        ..Default::default()
    };
    table.materials = [MaterialAppearance::default(); MATERIALS];
    table.materials[REGOLITH as usize] = colour([142, 140, 135], 0.95);
    table.materials[MARE as usize] = colour([84, 84, 86], 0.95);
    table.materials[EJECTA as usize] = colour([196, 194, 188], 0.93);
    table.materials[ROCK as usize] = colour([112, 110, 106], 0.85);
    table.materials[BASALT as usize] = colour([58, 59, 63], 0.8);
    table.materials[ANORTHOSITE as usize] = colour([168, 166, 158], 0.85);
    // Fresh ejecta and boulders thin out into the surrounding regolith.
    table.materials[EJECTA as usize].speck_host = Some(REGOLITH as u8);
    table.detail = [0.0, 0.12, 0.08, 0.0];
    table
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planet::{Planet, PlanetRecipe};
    use crate::terrain::TerrainSource;

    fn moon(shape: crate::grid::Shape) -> Planet {
        Planet::new(PlanetRecipe {
            shape,
            radius_m: 1_737_400.0,
            plane_size_m: 4_000.0,
            terrain: TerrainSource { generator: ID.into(), ..Default::default() },
            ..Default::default()
        })
        .unwrap()
    }

    /// Craters exist at every size and the bounds hold on samples.
    #[test]
    fn craters_shape_the_surface_within_its_bounds() {
        for shape in [crate::grid::Shape::Sphere, crate::grid::Shape::Plane] {
            let planet = moon(shape);
            crate::terrain::check_field(&planet, 2_000).unwrap();
            let k = match planet.field().program().constants.len() {
                n if n == std::mem::size_of::<MoonConstants>() => *bytemuck::from_bytes::<MoonConstants>(&planet.field().program().constants),
                n => panic!("constants {n}"),
            };
            assert!(k.header[0] >= 6, "crater octaves {}", k.header[0]);
            // Along a line the field dips into bowls: heights spread.
            let g = planet.grid();
            let n = g.cells();
            let (mut lo, mut hi, mut fresh) = (i32::MAX, i32::MIN, 0);
            for t in 0..4_000 {
                let face = if g.is_plane() { crate::grid::PLANE_FACE } else { 2 };
                let p = g.domain_point(face, n / 4 + t * 3, n / 2, 0);
                let (h, s) = height_parts(&k, p, g.level_offset());
                lo = lo.min(h);
                hi = hi.max(h);
                fresh += usize::from(s & 0xff > 0);
            }
            assert!(hi - lo > 1_000, "{shape:?}: relief {lo}..{hi}");
            eprintln!("{shape:?}: relief {lo}..{hi} mm, {fresh} samples on fresh ejecta");
        }
    }

    /// No steps between neighbouring columns (the bowls and rims are
    /// continuous; the rim's crease is a slope change, not a step).
    #[test]
    fn the_moon_has_no_steps_between_neighbouring_columns() {
        let planet = moon(crate::grid::Shape::Sphere);
        let g = planet.grid();
        let n = g.cells();
        let mut worst = 0;
        for t in 0..20_000 {
            let (i, j) = (n / 5 + t * 131, n / 3 + t * 17);
            let h = |d: i32| planet.field().height(g.domain_point(2, i + d, j, 0), g.level_offset());
            worst = worst.max((h(0) - 2 * h(1) + h(2)).abs());
        }
        assert!(worst < 100, "worst second difference {worst} mm");
    }
}
