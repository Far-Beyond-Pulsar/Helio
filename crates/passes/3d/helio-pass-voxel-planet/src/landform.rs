//! The terrain-stack interpreter of `helio.terrain` ([`crate::layers`]).
//!
//! A stack compiles to one octave table sorted from coarsest to finest,
//! each octave tagged with its layer (`kind | layer << 8`), plus a table of
//! layers. [`height_parts`] runs the octaves into per-layer accumulators,
//! then composes the layers in stack order with their masks; every world
//! (planet, moon, plane) runs this same code, and `landform.wgsl` mirrors
//! it, so changing layers never recompiles shaders.
//!
//! Every operation is wrapping two's-complement integer arithmetic, so CPU
//! and GPU agree to the bit. Additive octaves finer than a column footprint
//! are omitted: coarse levels are band-limited point samples of the same
//! field rather than an independent smooth replacement.
use crate::grid::Grid;
use crate::layers::{Caves, Overhangs};
use crate::noise::{fade, hash3, lerp, mul16, mul_fine, mul_shr, mul_shr_signed, noise, noise_fine, noise_fine_grad, scale, sin_turns, unit_q30, FINE_ONE, ONE, Q30};
use crate::terrain::{MaterialAppearance, TerrainAppearance, TerrainField, TerrainProgram, HEIGHT_ONE, MATERIALS};
use bytemuck::{Pod, Zeroable};
use glam::IVec3;
use std::borrow::Cow;

/// Program key: one WGSL source for every stack; generation compiles the
/// ridge-envelope display variant of it.
pub(crate) const DISPLAY_PROGRAM: &str = "helio.terrain/1-stack-ridge-envelope";

/// Volumetric terms of `landform.wgsl` (`TerrainConstants::volume`).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct LandformVolume {
    /// flags (1 caves, 2 overhangs), region shift, region threshold (Q16), depth (mm).
    pub caves: [i32; 4],
    /// tunnel shift, tunnel half width (Q16), cavern shift, cavern threshold (Q16).
    pub shapes: [i32; 4],
    /// overhang amplitude (mm), shift, region shift, region threshold (Q16).
    pub overhangs: [i32; 4],
    /// tunnel radius (mm), cavern size (mm), cover (mm), layer (mm).
    pub sizes: [i32; 4],
}

/// Quantile of [`noise`] (in units of [`ONE`]), from 400k samples: symmetric,
/// standard deviation 0.27, slightly lighter tails than a Gaussian
/// (`tests::noise_distribution_quantiles`).
pub(crate) fn noise_quantile(p: f64) -> f64 {
    const TABLE: [(f64, f64); 6] = [(0.5, 0.0), (0.75, 0.1937), (0.9, 0.3533), (0.95, 0.446), (0.99, 0.5851), (1.0, 0.75)];
    let (sign, p) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };
    let mut value = TABLE[5].1;
    for w in TABLE.windows(2) {
        if p <= w[1].0 {
            value = w[0].1 + (w[1].1 - w[0].1) * (p - w[0].0) / (w[1].0 - w[0].0);
            break;
        }
    }
    sign * value
}

const SEED_CAVE_REGION: u32 = 0xA511_E9B3;
const SEED_TUNNEL_A: u32 = 0x63D8_3595;
const SEED_TUNNEL_B: u32 = 0x2B1F_4C7A;
const SEED_CAVERN: u32 = 0x9E37_79B1;
const SEED_OVERHANG: u32 = 0x7F4A_7C15;
const SEED_OVERHANG_REGION: u32 = 0x4CF5_AD43;

impl LandformVolume {
    pub fn new(grid: &Grid, caves: &Caves, overhangs: &Overhangs) -> Self {
        let shift = |metres: f64| ((metres / crate::grid::DOMAIN_UNIT).log2().round().clamp(1.0, 30.0)) as i32;
        let mm = |metres: f64| (metres.max(0.0) * 1000.0).round().min(f64::from(i32::MAX / 4)) as i32;
        // Threshold above which `share` of the noise lies.
        let threshold = |share: f64| (noise_quantile(1.0 - share.clamp(0.0, 1.0)) * f64::from(ONE)).round() as i32;
        let layer = grid.layer_mm() as i32;
        // The level-0 band must hold the caves, the overhangs and the
        // column's own relief within 256 bricks (2048 cells).
        let depth = mm(caves.depth_m).min(1_700 * layer);
        let mut flags = 0;
        if caves.enabled && depth > 0 && caves.tunnel_radius_m.max(caves.cavern_wavelength_m) > 0.0 {
            flags |= 1;
        }
        let overhang = if overhangs.enabled { mm(overhangs.height_m).min(200 * layer) } else { 0 };
        if overhang > 0 {
            flags |= 2;
        }
        // Tunnels are where two noises are both near zero; their radius is
        // about the half width over the noise slope (~2 per wavelength).
        let width = (2.0 * caves.tunnel_radius_m / caves.tunnel_wavelength_m.max(1e-3) * f64::from(ONE)).round() as i32;
        Self {
            caves: [flags, shift(caves.region_km * 1000.0), threshold(caves.share), depth],
            shapes: [shift(caves.tunnel_wavelength_m), width.clamp(0, ONE), shift(caves.cavern_wavelength_m), threshold(caves.cavern_share)],
            overhangs: [overhang, shift(overhangs.wavelength_m), shift(overhangs.region_km * 1000.0), threshold(overhangs.share)],
            sizes: [mm(caves.tunnel_radius_m), mm(caves.cavern_wavelength_m / 4.0), mm(caves.cover_m), layer],
        }
    }

    fn seed(&self, seed: u32, salt: u32) -> u32 {
        seed ^ salt
    }

    /// Whether tunnels and caverns are resolved at `level` (their size is at
    /// least one level cell). Caverns under a rock cover also need a cell
    /// within that cover: coarser cells could not show them from outside,
    /// and their columns keep the heightfield's relief and filtering.
    fn caves_at(&self, level: u32) -> (bool, bool) {
        if self.caves[0] & 1 == 0 {
            return (false, false);
        }
        let layer = self.sizes[3];
        let cavern = if self.sizes[2] > 0 { self.sizes[1].min(self.sizes[2]) } else { self.sizes[1] };
        ((self.sizes[0] >> level) >= layer, (cavern >> level) >= layer)
    }

    fn cave_region(&self, p: IVec3, seed: u32) -> bool {
        noise(p, self.caves[1] as u32, self.seed(seed, SEED_CAVE_REGION)) > self.caves[2]
    }

    /// Overhang amplitude (mm) of the column at `p`, 0 where it is smaller
    /// than two level cells.
    fn overhang_amplitude(&self, p: IVec3, level: u32, seed: u32) -> i32 {
        if self.caves[0] & 2 == 0 {
            return 0;
        }
        let n = noise(p, self.overhangs[2] as u32, self.seed(seed, SEED_OVERHANG_REGION));
        let ramp = (n.wrapping_sub(self.overhangs[3]).wrapping_mul(4)).clamp(0, ONE);
        let a = scale(ramp, self.overhangs[0]);
        if (a >> level) < 2 * self.sizes[3] { 0 } else { a }
    }

    /// Level cells below and above the heightfield top that may differ.
    pub fn extent(&self, p: IVec3, level: u32, seed: u32) -> (i32, i32) {
        let layer = self.sizes[3];
        let (tunnels, caverns) = self.caves_at(level);
        let mut below = 0;
        if (tunnels || caverns) && self.cave_region(p, seed) {
            below = (self.caves[3] / layer >> level) + 2;
        }
        let a = self.overhang_amplitude(p, level, seed);
        let mut above = 0;
        if a > 0 {
            above = (a / layer >> level) + 2;
            below = below.max(above);
        }
        (below, above)
    }

    /// Kind of layer `k` of the column at `p` (heightfield top `top`), with
    /// 3D domain point `q`.
    pub fn cell(&self, p: IVec3, q: IVec3, level: u32, top: i32, k: i32, seed: u32) -> u32 {
        let layer = self.sizes[3];
        let mut solid = k < top;
        let a = self.overhang_amplitude(p, level, seed);
        if a > 0 {
            // Height of the cell centre over the heightfield top against a
            // 3D displacement: the surface folds into overhangs and arches.
            let cell = layer.wrapping_shl(level);
            let d = k.wrapping_sub(top).wrapping_mul(cell).wrapping_add(cell / 2);
            let s = scale(noise(q, self.overhangs[1] as u32, self.seed(seed, SEED_OVERHANG)), a);
            solid = d < s;
        }
        let (tunnels, caverns) = self.caves_at(level);
        if solid && k < top && (tunnels || caverns) && self.cave_region(p, seed) {
            let cell = layer.wrapping_shl(level);
            let depth = top.wrapping_sub(k).wrapping_mul(cell).wrapping_sub(cell / 2);
            if depth <= self.caves[3] {
                let w = self.shapes[1];
                if tunnels
                    && noise(q, self.shapes[0] as u32, self.seed(seed, SEED_TUNNEL_A)).abs() < w
                    && noise(q, self.shapes[0] as u32, self.seed(seed, SEED_TUNNEL_B)).abs() < w
                {
                    solid = false;
                } else if caverns && depth >= self.sizes[2] && noise(q, self.shapes[2] as u32, self.seed(seed, SEED_CAVERN)) > self.shapes[3] {
                    solid = false;
                }
            }
        }
        u32::from(solid)
    }

    /// Extra level cells a finer column's top may rise over a coarse one.
    fn rise_cells(&self, level: u32) -> i32 {
        if self.caves[0] & 2 == 0 { 0 } else { (self.overhangs[0] / self.sizes[3] >> level) + 2 }
    }
}

/// Octave table size, warp included.
pub const OCTAVES: usize = 48;
/// Leading domain-warp octaves (two per axis, zero without a Warp layer).
pub const WARP_OCTAVES: usize = 6;
/// Layers of a stack.
pub const LAYERS: usize = 8;

/// Octave kinds (low byte of [`Octave::kind`]; the layer index is above).
pub const CONTINENT: u32 = 0;
/// Mountain region mask.
pub const REGION: u32 = 1;
pub const RIDGE: u32 = 2;
/// Warped fBm (fine noise).
pub const HILLS: u32 = 3;
/// Domain warp, one kind per axis (4, 5, 6).
pub const WARP: u32 = 4;
/// Unwarped metre-scale fBm (16-bit noise).
pub const ROUGHNESS: u32 = 7;
/// Erosion gullies.
pub const EROSION: u32 = 8;
/// Crater lattice; its density (Q16) is the seed's low 16 bits.
pub const CRATER: u32 = 9;
/// Basin mask (fine noise, unwarped).
pub const BASIN: u32 = 10;

/// One octave: lattice shift, amplitude (height units; Q16 weight for the
/// masks) and seed, and `kind | layer << 8`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct Octave {
    pub shift: u32,
    pub amplitude: i32,
    pub seed: u32,
    pub kind: u32,
}

impl Octave {
    #[inline]
    pub fn class(&self) -> u32 {
        self.kind & 0xff
    }
    #[inline]
    pub fn layer(&self) -> usize {
        ((self.kind >> 8) & (LAYERS as u32 - 1)) as usize
    }
}

/// One layer of the compiled stack: its kind, mask and two parameters.
///
/// | kind | `a` | `b` |
/// |---|---|---|
/// | continents | ocean floor (height units, negative) | lowland height |
/// | mountains | region mask bias (Q16) | |
/// | erosion | saturation slope (height units per gradient span) | |
/// | craters | rim over depth (Q16) | share of fresh craters (Q16) |
/// | basins | depth (height units) | mask threshold (Q24) |
/// | plateau | height (height units, whole layers) | |
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct StackLayer {
    pub kind: u32,
    /// 0 everywhere, 1 land, 2 above deep sea.
    pub mask: u32,
    pub a: i32,
    pub b: i32,
}

impl StackLayer {
    pub const WARP: u32 = 1;
    pub const CONTINENTS: u32 = 2;
    pub const MOUNTAINS: u32 = 3;
    pub const HILLS: u32 = 4;
    pub const ROUGHNESS: u32 = 5;
    pub const EROSION: u32 = 6;
    pub const CRATERS: u32 = 7;
    pub const BASINS: u32 = 8;
    pub const PLATEAU: u32 = 9;
}

/// Material styles (`LandformConstants::style[0]`).
pub const STYLE_EARTHLIKE: i32 = 0;
pub const STYLE_LUNAR: i32 = 1;
pub const STYLE_LAYERED: i32 = 2;

/// `TerrainConstants` of `landform.wgsl` (uniform layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Pod, Zeroable)]
pub struct LandformConstants {
    /// octave count (warp slots included), layer thickness (mm), soil
    /// depth (cells), seed.
    pub header: [i32; 4],
    /// moisture shift, continents layer + 1 (0: none, all land), snowline,
    /// low-basin height (mm).
    pub levels: [i32; 4],
    /// ridge display layer + 1 (0: none), steep slope (cells/cell), ridge
    /// display flag, vertical axis (0 radial, 1 the plane's +Y).
    pub shape: [i32; 4],
    /// material style, then the Layered surface, soil and rock ids.
    pub style: [i32; 4],
    /// layer count, sum of the erosion amplitudes (height units), domain
    /// radius (0 on planes), warp octaves (0 or 6).
    pub stack: [i32; 4],
    pub layers: [StackLayer; LAYERS],
    pub octaves: [Octave; OCTAVES],
}

/// Crater radius of the largest crater of a cell (Q19 of the lattice cell,
/// 0.3): with the ejecta at twice the radius, the 3x3x3 cells around a
/// point hold every crater reaching it.
pub const CRATER_RADIUS_Q19: u32 = 157_286;
/// Squared ejecta reach in crater radii (Q24).
const REACH: i32 = 4 << 24;

impl LandformConstants {
    fn stack_layers(&self) -> &[StackLayer] {
        &self.layers[..(self.stack[0].max(0) as usize).min(LAYERS)]
    }

    fn table(&self) -> &[Octave] {
        &self.octaves[WARP_OCTAVES..(self.header[0].max(WARP_OCTAVES as i32) as usize).min(OCTAVES)]
    }

    /// The continents layer, if the stack has one.
    fn continents(&self) -> Option<usize> {
        (self.levels[1] > 0).then(|| (self.levels[1] - 1) as usize & (LAYERS - 1))
    }

    /// Largest change (height units) each layer's own term can make.
    fn magnitudes(&self) -> [f64; LAYERS] {
        let mut m = [0.0; LAYERS];
        for o in self.table() {
            let a = f64::from(o.amplitude).abs();
            m[o.layer()] += match o.class() {
                CONTINENT | REGION | BASIN => 0.0,
                // Up to two overlapping craters per octave reach a point.
                CRATER => 2.0 * a * (1.0 + f64::from(self.layers[o.layer()].a) / 65_536.0),
                _ => a,
            };
        }
        for (l, layer) in self.stack_layers().iter().enumerate() {
            m[l] += match layer.kind {
                StackLayer::CONTINENTS => f64::from(layer.a).abs() + f64::from(layer.b).abs(),
                StackLayer::BASINS | StackLayer::PLATEAU => f64::from(layer.a).abs(),
                _ => 0.0,
            };
        }
        m
    }

    /// Per layer, the largest change the layer makes as composed: its own
    /// term, and for basins also flattening the layers before them by half.
    fn effects(&self) -> [f64; LAYERS] {
        let m = self.magnitudes();
        let mut before = 0.0;
        let mut e = m;
        for (l, layer) in self.stack_layers().iter().enumerate() {
            if layer.kind == StackLayer::BASINS {
                e[l] += before / 2.0;
            }
            before += e[l];
        }
        e
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
        let magnitude = self.magnitudes();
        let effect = self.effects();
        let layers = self.stack_layers();
        let masked = |mask: u32| layers.iter().enumerate().filter(|(_, l)| l.mask == mask).map(|(l, _)| effect[l]).sum::<f64>();
        // The continents scale the base, the land mask (x3) and the deep-sea
        // fade (x2) of the layers they mask.
        let shape = self.continents().map_or(0.0, |l| f64::from(layers[l].a).abs() + 2.0 * f64::from(layers[l].b).abs());
        let continent = shape + 3.0 * masked(1) + 2.0 * masked(2);
        // Domain warp Lipschitz constant (dimensionless).
        let warp = self.octaves[..self.stack[3] as usize]
            .iter()
            .map(|o| f64::from(o.amplitude.abs()) * G / 2f64.powi(o.shift as i32))
            .sum::<f64>();
        let ratio = f64::from(grid.reference_cells()) / f64::from(grid.cells());
        let mut out = [0i32; 24];
        for level in 0..24u32 {
            let effective = level + grid.level_offset();
            let mut dropped = 0.0;
            let mut lipschitz = 0.0;
            let mut unwarped = 0.0;
            for o in self.table() {
                let a = f64::from(o.amplitude.abs());
                let per = 1.0 / 2f64.powi(o.shift as i32);
                let relief = match o.class() {
                    CRATER => 2.0 * a * (1.0 + f64::from(self.layers[o.layer()].a) / 65_536.0),
                    _ => a,
                };
                match o.class() {
                    CONTINENT => lipschitz += continent * G * per,
                    REGION => lipschitz += 3.0 * magnitude[o.layer()] * G * per,
                    BASIN => unwarped += 4.0 * effect[o.layer()] * G * per,
                    _ if !resolved(o, effective) => dropped += relief,
                    RIDGE => lipschitz += 2.0 * a * G * per,
                    HILLS => lipschitz += a * G * per,
                    // Gullies: the phase turns 2 pi STRIPES per lattice
                    // spacing, plus the turning of their direction.
                    EROSION => unwarped += a * 19.0 * per,
                    // Bowl slope 2 (depth + rim) per radius, radius >= 0.165 cells.
                    CRATER => unwarped += relief * 13.0 * per,
                    _ => unwarped += a * G * per,
                }
            }
            // Half diagonal of a level cell in domain units (sphere cells
            // are at most 1.05 times a face-centre cell).
            let cell = crate::grid::REFERENCE_VOXEL / crate::grid::DOMAIN_UNIT;
            let half_diagonal = 2f64.powi(level as i32) * cell * ratio * std::f64::consts::SQRT_2 * 0.5 * 1.05;
            let excess_mm = dropped + (lipschitz * (1.0 + warp) + unwarped) * half_diagonal;
            let cell_mm = f64::from(self.header[1]) * 2f64.powi(level as i32);
            out[level as usize] = ((excess_mm / cell_mm).ceil() as i64 + 2).clamp(2, 1 << 20) as i32;
        }
        out
    }

    /// Conservative lowest and highest surface height (height units).
    pub fn height_range(&self) -> (i32, i32) {
        let layers = self.stack_layers();
        let effect = self.effects();
        let plateau: f64 = layers.iter().filter(|l| l.kind == StackLayer::PLATEAU).map(|l| f64::from(l.a)).sum();
        let basins = layers.iter().any(|l| l.kind == StackLayer::BASINS);
        let mut spread = 0.0;
        for (l, layer) in layers.iter().enumerate() {
            // Basins may flatten a plateau: then it is spread, not an offset.
            if layer.kind != StackLayer::PLATEAU || basins {
                spread += effect[l];
            }
        }
        let centre = if basins { 0.0 } else { plateau };
        let pad = if spread > 0.0 { 10.0 * f64::from(HEIGHT_ONE) } else { 0.0 };
        let clamp = |v: f64| v.clamp(f64::from(i32::MIN / 2), f64::from(i32::MAX / 2)) as i32;
        (clamp((centre - spread - pad).floor()), clamp((centre + spread + pad).ceil()))
    }
}

/// Smallest lattice shift resolved at `level`, above the level: a lattice
/// of `2^(level + 5)` domain units spans four level cells.
pub const RESOLVED_SHIFT: u32 = 5;

/// Additive detail finer than about four level cells is omitted at `level`
/// (`level` counts reference cells); masks are always evaluated.
#[inline]
fn resolved(o: &Octave, level: u32) -> bool {
    matches!(o.class(), CONTINENT | REGION | BASIN) || o.shift >= level + RESOLVED_SHIFT
}

/// Gradients are in height units per `2^GRAD_SHIFT` domain units (3.3 km):
/// gentle slopes keep about 20 bits, which gully directions need.
pub const GRAD_SHIFT: u32 = 18;
/// Gully stripes per erosion lattice spacing.
const STRIPES: i32 = 2;

/// A lattice-spacing derivative per gradient span.
#[inline]
fn per_span(d: i32, shift: u32) -> i32 {
    if shift >= GRAD_SHIFT { d >> (shift - GRAD_SHIFT) } else { d << (GRAD_SHIFT - shift) }
}

#[inline]
fn per_span3(d: IVec3, shift: u32) -> IVec3 {
    IVec3::new(per_span(d.x, shift), per_span(d.y, shift), per_span(d.z, shift))
}

#[inline]
fn mul_fine3(v: IVec3, b: i32) -> IVec3 {
    IVec3::new(mul_fine(v.x, b), mul_fine(v.y, b), mul_fine(v.z, b))
}

/// Q24 weight of a clamp's derivative inside `(lo, hi)`, fading to zero over
/// `FINE_ONE / 8` at both edges. Gullies are steered by a continuous
/// gradient: a clamp's derivative switches on and off at its edges, and
/// the direction jump would shift the gully phase into a cliff.
#[inline]
fn soft_inside(x: i32, lo: i32, hi: i32) -> i32 {
    ((x - lo) * 8).clamp(0, FINE_ONE).min(((hi - x) * 8).clamp(0, FINE_ONE))
}

/// One erosion octave at `p`: gully stripes along the downhill direction of
/// `g` (height units per span w.r.t. `p`), blended over the octave's 3D
/// lattice with a random phase per corner, and their gradient (which steers
/// the finer octaves). `up` is the Q30 unit vertical.
///
/// The phase turns up to `2 sqrt(3) STRIPES` times across a lattice cell, so
/// direction and offsets keep full precision (Q30 unit vectors, exact 64-bit
/// products): a 1e-5 direction error would already shift a 40 m gully by
/// centimetres between neighbouring columns.
/// `saturation` is the slope (height units per span) of full depth.
fn erosion_octave(saturation: i32, o: &Octave, p: IVec3, g: IVec3, up: IVec3) -> (i32, IVec3) {
    // Across the slope, horizontal: t = up x g, |t| the horizontal slope.
    let t = IVec3::new(
        mul_shr_signed(up.y, g.z, 30) - mul_shr_signed(up.z, g.y, 30),
        mul_shr_signed(up.z, g.x, 30) - mul_shr_signed(up.x, g.z, 30),
        mul_shr_signed(up.x, g.y, 30) - mul_shr_signed(up.y, g.x, 30),
    );
    let tn = unit_q30(t);
    if tn == IVec3::ZERO {
        return (0, IVec3::ZERO);
    }
    let slope = (mul_shr_signed(t.x, tn.x, 30) + mul_shr_signed(t.y, tn.y, 30) + mul_shr_signed(t.z, tn.z, 30)) as u32;
    let saturation = saturation.max(1) as u32;
    let strength = (((slope.min(saturation) >> 5) << 16) / (saturation >> 5).max(1)) as i32;
    let s = o.shift;
    let c = IVec3::new(p.x >> s, p.y >> s, p.z >> s);
    let mask = (1i32 << s) - 1;
    let w = [p.x, p.y, p.z].map(|v| fade(if s >= 16 { (v & mask) >> (s - 16) } else { (v & mask) << (16 - s) }));
    let mut cos = [0i32; 8];
    let mut sin = [0i32; 8];
    for index in 0..8usize {
        let corner = c + IVec3::new((index & 1) as i32, ((index >> 1) & 1) as i32, (index >> 2) as i32);
        // Offset from the corner along tn, in Q16 lattice spacings.
        let d = p - IVec3::new(corner.x << s, corner.y << s, corner.z << s);
        let along = mul_shr_signed(d.x, tn.x, s + 14) + mul_shr_signed(d.y, tn.y, s + 14) + mul_shr_signed(d.z, tn.z, s + 14);
        let phase = along * STRIPES + (hash3(corner.x, corner.y, corner.z, o.seed) & 0xffff) as i32;
        // Gully profile pi/2 |cos(phase / 2)| - 1: V-shaped valleys between
        // rounded ridges, zero mean (levels that omit it keep their height)
        // and within the amplitude.
        cos[index] = mul16(sin_turns((phase >> 1) + 16_384).abs(), 102_944) - ONE;
        sin[index] = sin_turns(phase);
    }
    let tri = |v: [i32; 8]| {
        let y0 = lerp(lerp(v[0], v[1], w[0]), lerp(v[2], v[3], w[0]), w[1]);
        let y1 = lerp(lerp(v[4], v[5], w[0]), lerp(v[6], v[7], w[0]), w[1]);
        lerp(y0, y1, w[2])
    };
    let value = scale(mul16(tri(cos), strength), o.amplitude);
    // The steering derivative is the smooth cos(phase) one, so finer octaves
    // see no direction flip at valley floors: -A s sin(phase) 2 pi STRIPES /
    // lattice along tn (2 pi STRIPES = 3217 / 256).
    let m = scale(mul16(tri(sin), strength), o.amplitude);
    let magnitude = per_span(-((m * 3_217) >> 8), s);
    (value, IVec3::new(mul_shr_signed(magnitude, tn.x, 30), mul_shr_signed(magnitude, tn.y, 30), mul_shr_signed(magnitude, tn.z, 30)))
}

/// Ocean floor to lowland base of the continents at `c` (Q24): shelf then
/// deep ocean below the coast, lowlands rising inland.
#[inline]
fn continent_base(c: i32, floor: i32, lowland: i32) -> i32 {
    if c < 0 {
        let t = (-c).min(FINE_ONE);
        mul_fine(floor, t).wrapping_add(mul_fine(lowland / 8, FINE_ONE - t))
    } else {
        mul_fine(lowland, (c * 2).min(FINE_ONE))
    }
}

/// Land weight (rises from the coast) and deep-sea fade at continent value
/// `c`; without continents `c` is `FINE_ONE`, all land.
#[inline]
fn land_masks(c: i32) -> (i32, i32) {
    ((c * 3).clamp(0, FINE_ONE), (FINE_ONE + c * 2).clamp(FINE_ONE / 8, FINE_ONE))
}

#[inline]
fn masked(mask: u32, x: i32, land: i32, wet: i32) -> i32 {
    match mask {
        1 => mul_fine(x, land),
        2 => mul_fine(x, wet),
        _ => x,
    }
}

/// Mountain region weight of a mountain layer from its mask noise.
#[inline]
fn region_weight(region: i32, bias: i32) -> i32 {
    ((region - (bias << 8)) * 3).clamp(0, FINE_ONE)
}

/// Per-layer accumulators of [`height_parts`] and their gradients.
struct Accumulators {
    /// continents: Q24 value; mountains: ridged sum; basins: Q24 mask noise;
    /// other layers: height.
    value: [i32; LAYERS],
    /// mountains: Q24 region noise.
    region: [i32; LAYERS],
    /// mountains: ridge weight of the next octave.
    weight: [i32; LAYERS],
    dvalue: [IVec3; LAYERS],
    dregion: [IVec3; LAYERS],
    dweight: [IVec3; LAYERS],
}

/// Steering gradient (height units per span, w.r.t. the warped point) of
/// the continents and mountains resolved so far, masked as composed, with
/// every kink softened ([`soft_inside`]).
fn steering(k: &LandformConstants, s: &Accumulators) -> IVec3 {
    let (c, dc) = k.continents().map_or((FINE_ONE, IVec3::ZERO), |l| (s.value[l], s.dvalue[l]));
    let (land, wet) = land_masks(c);
    let dland = mul_fine3(dc * 3, soft_inside(c * 3, 0, FINE_ONE));
    let dwet = mul_fine3(dc * 2, soft_inside(FINE_ONE + c * 2, FINE_ONE / 8, FINE_ONE));
    let mut g = IVec3::ZERO;
    for (l, layer) in k.stack_layers().iter().enumerate() {
        let (x, dx) = match layer.kind {
            StackLayer::CONTINENTS => {
                let d = if c < 0 {
                    mul_fine3(mul_fine3(-dc, soft_inside(-c, 0, FINE_ONE)), layer.a - layer.b / 8)
                } else {
                    mul_fine3(mul_fine3(dc * 2, soft_inside(c * 2, 0, FINE_ONE)), layer.b)
                };
                (continent_base(c, layer.a, layer.b), d)
            }
            StackLayer::MOUNTAINS => {
                let m = (s.region[l] - (layer.a << 8)) * 3;
                let region = m.clamp(0, FINE_ONE);
                let dregion = mul_fine3(s.dregion[l] * 3, soft_inside(m, 0, FINE_ONE));
                (mul_fine(s.value[l], region), mul_fine3(s.dvalue[l], region) + mul_fine3(dregion, s.value[l]))
            }
            _ => continue,
        };
        g += match layer.mask {
            1 => mul_fine3(dx, land) + mul_fine3(dland, x),
            2 => mul_fine3(dx, wet) + mul_fine3(dwet, x),
            _ => dx,
        };
    }
    g
}

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

/// One crater octave at `p` (vertical `up`, Q30): height (mm) and the
/// freshest ejecta reaching `p` (0..255).
///
/// An octave has one candidate crater per cell of a 3D lattice, at a hashed
/// point; it exists with the octave's density when that point lies within
/// half a cell of the surface (sphere or plane), so craters are seamless
/// across cube edges. Each has a parabolic bowl, a rim and an ejecta
/// blanket out to twice its radius.
fn crater_octave(k: &LandformConstants, layer: &StackLayer, o: &Octave, p: IVec3, up: IVec3) -> (i32, u32) {
    let s = o.shift;
    let density = o.seed & 0xffff;
    let radius_domain = k.stack[2] as u32;
    let c0 = IVec3::new(p.x >> s, p.y >> s, p.z >> s);
    let (mut height, mut ejecta) = (0i32, 0u32);
    for index in 0..27 {
        let cell = c0 + IVec3::new(index % 3 - 1, (index / 3) % 3 - 1, index / 9 - 1);
        let a = hash3(cell.x, cell.y, cell.z, o.seed);
        if a & 0xffff >= density {
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
        if k.shape[3] != 0 {
            if centre.y.abs() >= 1 << (s - 1) {
                continue;
            }
        } else {
            let len2 = mul_shr(centre.x.unsigned_abs(), centre.x.unsigned_abs(), 30)
                + mul_shr(centre.y.unsigned_abs(), centre.y.unsigned_abs(), 30)
                + mul_shr(centre.z.unsigned_abs(), centre.z.unsigned_abs(), 30);
            let r2 = mul_shr(radius_domain, radius_domain, 30);
            if (len2 as i32 - r2 as i32).unsigned_abs() >= radius_domain >> (30 - s) {
                continue;
            }
        }
        // Squared horizontal distance in Q28 lattice cells; nothing beyond
        // a cell reaches `p`. Offsets are Q(19 + e) cells: Q19 for fine
        // lattices, whole domain units (e = s - 19) for coarse ones, whose
        // squares are exact 64-bit products. Rounding the offsets of a large
        // crater to Q19 would move its steep walls by centimetres between
        // neighbouring columns.
        let e = s.saturating_sub(19);
        let d = centre - p;
        let d = if s >= 19 { d } else { IVec3::new(d.x << (19 - s), d.y << (19 - s), d.z << (19 - s)) };
        if d.abs().max_element() >= 1 << (19 + e) {
            continue;
        }
        // The centre may lie half a cell off the surface: the vertical part
        // cancels, so it keeps 10 more bits (Q29).
        let along = mul_shr_signed(d.x, up.x, 20 + e) + mul_shr_signed(d.y, up.y, 20 + e) + mul_shr_signed(d.z, up.z, 20 + e);
        let square = |v: i32| mul_shr(v.unsigned_abs(), v.unsigned_abs(), 10 + 2 * e);
        let along2 = mul_shr(along.unsigned_abs(), along.unsigned_abs(), 30);
        let r2 = (square(d.x) + square(d.y) + square(d.z)).saturating_sub(along2);
        // Radius 0.55 to 1 of the largest; depth in proportion.
        let size = 36_045 + (((a >> 16) * 29_491) >> 16) as i32;
        let radius = mul16(CRATER_RADIUS_Q19 as i32, size).max(1);
        let rad2 = mul_shr(radius as u32, radius as u32, 10).max(1);
        if r2 >= 4 * rad2 {
            continue;
        }
        // (r / radius)^2 in Q24, below 4 (the ejecta reach): a full-precision
        // reciprocal, so large craters move smoothly between columns.
        let x2 = squared_ratio(r2, rad2) as i32;
        let depth = scale(size, o.amplitude);
        let rim = scale(layer.a, depth);
        // The bowl rises as x^2; the ejecta falls as t^3 from the rim to
        // twice the radius; a smooth minimum rounds the rim between them.
        let bowl = mul_fine(x2, depth + rim) - depth;
        let t = (REACH - x2) / 3;
        let ejecta_height = mul_fine(mul_fine(mul_fine(t, t), t), rim);
        height += smooth_min(bowl, ejecta_height, rim / 4);
        if (hash3(cell.x, cell.y, cell.z, o.seed ^ 0x1B87_3593) & 0xffff) < layer.b as u32 {
            ejecta = ejecta.max(255 - ((x2 >> 16) * 255 / (REACH >> 16)) as u32);
        }
    }
    (height, ejecta)
}

/// Surface height (height units above the datum) of the column centred at
/// domain point `p` with a `2^level` reference cell footprint.
pub fn height(k: &LandformConstants, p: IVec3, level: u32) -> i32 {
    height_parts(k, p, level).0
}

/// Surface word of a column (`terrain_surface`), by material style.
/// Earthlike: the erosion term as a signed byte of the erosion amplitude
/// sum (negative: gully floors, positive: the ribs between them). Lunar:
/// the freshest ejecta (bits 0..6) and the basin flag (bit 7).
pub fn surface(k: &LandformConstants, p: IVec3, level: u32) -> u32 {
    height_parts(k, p, level).1
}

/// [`height`] and [`surface`].
pub fn height_parts(k: &LandformConstants, p: IVec3, level: u32) -> (i32, u32) {
    // Erosion follows the slope of the larger terrain: when one of its
    // octaves is resolved here, the large octaves carry their gradients.
    let erosion = k.table().iter().any(|o| o.class() == EROSION && resolved(o, level));
    // The warp, masks, ridges and hills scale up to kilometres, so they use
    // the fine (Q24) noise: 16-bit noise is constant over metres at these
    // wavelengths and its steps, multiplied by the relief, would cut
    // terraces between neighbouring columns.
    //
    // The warp is a coordinate transform, so every level evaluates it.
    // `jacobian[a]` is the gradient of warp axis `a` (Q24, dimensionless).
    let mut warp = [0i32; 3];
    let mut jacobian = [IVec3::ZERO; 3];
    for o in &k.octaves[..(k.stack[3].max(0) as usize).min(WARP_OCTAVES)] {
        let axis = ((o.kind - WARP) as usize).min(2);
        if erosion {
            let (n, d) = noise_fine_grad(p, o.shift, o.seed);
            warp[axis] = warp[axis].wrapping_add(mul_fine(o.amplitude, n));
            let to_q24 = |v: i32| {
                let m = mul_fine(v, o.amplitude);
                if o.shift <= 24 { m << (24 - o.shift) } else { m >> (o.shift - 24) }
            };
            jacobian[axis] += IVec3::new(to_q24(d.x), to_q24(d.y), to_q24(d.z));
        } else {
            warp[axis] = warp[axis].wrapping_add(mul_fine(o.amplitude, noise_fine(p, o.shift, o.seed)));
        }
    }
    let q = p + IVec3::from_array(warp);
    // Gradients w.r.t. q map to p through (I + J)^T.
    let unwarp = |g: IVec3| {
        let column = |j: usize| g[j] + mul_fine(g.x, jacobian[0][j]) + mul_fine(g.y, jacobian[1][j]) + mul_fine(g.z, jacobian[2][j]);
        IVec3::new(column(0), column(1), column(2))
    };
    let up = if k.shape[3] != 0 { IVec3::new(0, Q30 as i32, 0) } else { unit_q30(p) };
    let mut s = Accumulators {
        value: [0; LAYERS],
        region: [0; LAYERS],
        weight: [FINE_ONE - 1; LAYERS],
        dvalue: [IVec3::ZERO; LAYERS],
        dregion: [IVec3::ZERO; LAYERS],
        dweight: [IVec3::ZERO; LAYERS],
    };
    let mut eroded = 0i32;
    let mut deroded = IVec3::ZERO;
    let mut ejecta = 0u32;
    for o in k.table() {
        if !resolved(o, level) {
            continue;
        }
        let l = o.layer();
        match o.class() {
            CONTINENT | REGION => {
                let (n, d) = if erosion { noise_fine_grad(q, o.shift, o.seed) } else { (noise_fine(q, o.shift, o.seed), IVec3::ZERO) };
                let v = mul_fine(n, o.amplitude << 8);
                let dv = mul_fine3(per_span3(d, o.shift), o.amplitude << 8);
                if o.class() == CONTINENT {
                    s.value[l] = s.value[l].wrapping_add(v);
                    s.dvalue[l] += dv;
                } else {
                    s.region[l] = s.region[l].wrapping_add(v);
                    s.dregion[l] += dv;
                }
            }
            RIDGE => {
                let (n, d) = if erosion { noise_fine_grad(q, o.shift, o.seed) } else { (noise_fine(q, o.shift, o.seed), IVec3::ZERO) };
                let r = (FINE_ONE - n.abs()).clamp(0, FINE_ONE - 1);
                let rr = mul_fine(r, r);
                let v = mul_fine(rr, s.weight[l]);
                if erosion {
                    // The crest's sign flip is softened over |n| < 1/4.
                    let dr = mul_fine3(per_span3(-d, o.shift), (n * 4).clamp(-FINE_ONE, FINE_ONE));
                    let dv = mul_fine3(mul_fine3(dr, r) * 2, s.weight[l]) + mul_fine3(s.dweight[l], rr);
                    s.dweight[l] = mul_fine3(dv * 2, soft_inside(v * 2, FINE_ONE / 4, FINE_ONE - 1));
                    s.dvalue[l] += mul_fine3(dv, o.amplitude);
                }
                s.weight[l] = (v * 2).clamp(FINE_ONE / 4, FINE_ONE - 1);
                s.value[l] = s.value[l].wrapping_add(mul_fine(o.amplitude, v));
            }
            HILLS => s.value[l] = s.value[l].wrapping_add(mul_fine(o.amplitude, noise_fine(q, o.shift, o.seed))),
            EROSION => {
                let g = unwarp(steering(k, &s)) + deroded;
                let (e, de) = erosion_octave(k.layers[l].a, o, p, g, up);
                s.value[l] = s.value[l].wrapping_add(e);
                eroded = eroded.wrapping_add(e);
                deroded += de;
            }
            CRATER => {
                let (h, e) = crater_octave(k, &k.layers[l], o, p, up);
                s.value[l] = s.value[l].wrapping_add(h);
                ejecta = ejecta.max(e);
            }
            BASIN => s.value[l] = s.value[l].wrapping_add(noise_fine(p, o.shift, o.seed)),
            _ => s.value[l] = s.value[l].wrapping_add(scale(noise(p, o.shift, o.seed), o.amplitude)),
        }
    }
    // Compose the layers in stack order.
    let c = k.continents().map_or(FINE_ONE, |l| s.value[l]);
    let (land, wet) = land_masks(c);
    let mut h = 0i32;
    let mut basin = 0i32;
    for (l, layer) in k.stack_layers().iter().enumerate() {
        let x = match layer.kind {
            StackLayer::CONTINENTS => continent_base(c, layer.a, layer.b),
            StackLayer::MOUNTAINS => mul_fine(s.value[l], region_weight(s.region[l], layer.a)),
            StackLayer::BASINS => {
                // Basins flatten what lies below them in the stack by half
                // and sink it by their depth.
                let m = masked(layer.mask, (s.value[l].wrapping_sub(layer.b).wrapping_mul(4)).clamp(0, FINE_ONE), land, wet);
                basin = basin.max(m);
                h = mul_fine(h, FINE_ONE - m / 2).wrapping_sub(mul_fine(layer.a, m));
                continue;
            }
            StackLayer::PLATEAU => layer.a,
            StackLayer::HILLS | StackLayer::ROUGHNESS | StackLayer::EROSION | StackLayer::CRATERS => s.value[l],
            _ => 0,
        };
        h = h.wrapping_add(masked(layer.mask, x, land, wet));
    }
    let surface = match k.style[0] {
        STYLE_EARTHLIKE => ((eroded * 127 / k.stack[1].max(1)).clamp(-127, 127) as u32) & 0xff,
        STYLE_LUNAR => (ejecta >> 1) | (u32::from(basin >= FINE_ONE / 2) << 7),
        _ => 0,
    };
    (h, surface)
}

/// Moisture in Q24 [0, FINE_ONE] from very low-frequency noise at `p`
/// (fine, so dry-land edges follow smooth curves).
pub fn moisture(k: &LandformConstants, p: IVec3) -> i32 {
    (noise_fine(p, k.levels[0].max(1) as u32, (k.header[3] as u32) ^ 0x51ED_270B) + FINE_ONE) / 2
}

/// Strata altitude (mm): layers undulate +-8 m over ~100 m, so cuts
/// through them never show flat rings.
fn strata(c: &LandformConstants, p: IVec3, altitude: i32) -> i32 {
    altitude + scale(noise(p, 13, (c.header[3] as u32) ^ 0x9B05_688C), 8_000)
}

/// Earthlike materials: meadows, dry lands, rock outcrops, strata and snow
/// above the snowline; gravel in erosion gullies.
fn earthlike_material(
    c: &LandformConstants,
    p: IVec3,
    surface: u32,
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
        noise(p, 12, seed ^ 0x1B56_C4E9) + noise(p, 9, seed ^ 0x6A09_E667) / 3
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
    // Erosion on slopes: rock shows on the ribs between gullies; sediment
    // fills their floors.
    let gully = i32::from(surface as u8 as i8);
    if slope >= 2 {
        exposed += gully * (ONE / 256);
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
        if exposed > -ONE / 16 || (gully < -64 && slope >= 3) {
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

/// Material of a solid ground cell by the stack's material style.
/// `top_height` is the column height (mm), `depth` cells below the column
/// top (0 = exposed top cell), `slope` the ground slope across the cell's
/// 8x8 column block in eighths of a cell per cell, `layer` the base layer
/// index of the cell.
pub fn ground_material(c: &LandformConstants, p: IVec3, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
    match c.style[0] {
        STYLE_LUNAR => lunar_material(c, p, surface, depth, slope, layer),
        STYLE_LAYERED => (if depth == 0 { c.style[1] } else if depth <= c.header[2] { c.style[2] } else { c.style[3] }) as u32,
        _ => earthlike_material(c, p, surface, top_height, depth, slope, layer),
    }
}

/// Lunar materials: ids into [`lunar_appearance`].
pub mod lunar {
    pub const REGOLITH: u32 = 1;
    pub const MARE: u32 = 2;
    pub const EJECTA: u32 = 3;
    pub const ROCK: u32 = 4;
    pub const BASALT: u32 = 5;
    pub const ANORTHOSITE: u32 = 6;
}

/// Regolith over bedrock, dark basin plains, bright young ejecta thinning
/// out away from the crater, boulders on steep slopes.
fn lunar_material(c: &LandformConstants, p: IVec3, surface: u32, depth: i32, slope: i32, layer: i32) -> u32 {
    use lunar::*;
    let ejecta = (surface & 0x7f) << 1;
    let mare = surface & 0x80 != 0;
    let h = hash3(p.x, p.y, p.z ^ layer.wrapping_mul(0x9e37), 0x2545_F491);
    if depth == 0 {
        if slope >= 12 || (slope >= 6 && h & 3 == 0) {
            return ROCK;
        }
        if ejecta > (h & 0xff) {
            return EJECTA;
        }
        return if mare { MARE } else { REGOLITH };
    }
    if depth < c.header[2] {
        return if mare { MARE } else { REGOLITH };
    }
    if mare { BASALT } else { ANORTHOSITE }
}

/// The lunar material table: greys of regolith, mare, fresh ejecta,
/// boulders, basalt and anorthosite (sRGB).
pub fn lunar_appearance() -> TerrainAppearance {
    use lunar::*;
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

/// A compiled terrain stack on one grid.
pub struct LandformField {
    constants: LandformConstants,
    volume: LandformVolume,
    bounds: [i32; 24],
    render_bounds: [i32; 24],
    ridge_suffix: Option<[[i32; 4]; 66]>,
}

impl LandformField {
    pub fn new(grid: &Grid, constants: LandformConstants, volume: LandformVolume) -> Self {
        let mut bounds = constants.bound_margins(grid);
        // Stacks without a supported display layer keep the canonical path.
        // In particular, do not saturate invalid huge amplitudes into new
        // terrain.
        let ridge_suffix = crate::ridge_envelope::bake_ridge_suffix(grid, &constants).ok();
        let mut render_bounds = if ridge_suffix.is_some() {
            crate::ridge_envelope::render_bounds(grid, &constants, bounds)
        } else {
            bounds
        };
        // Overhangs raise a finer column's highest solid cell over its
        // heightfield top.
        for level in 0..24u32 {
            let rise = volume.rise_cells(level);
            bounds[level as usize] = bounds[level as usize].saturating_add(rise);
            render_bounds[level as usize] = render_bounds[level as usize].saturating_add(rise);
        }
        Self { bounds, render_bounds, ridge_suffix, constants, volume }
    }
    pub fn constants(&self) -> &LandformConstants {
        &self.constants
    }
    pub fn volume(&self) -> &LandformVolume {
        &self.volume
    }
    fn seed(&self) -> u32 {
        self.constants.header[3] as u32
    }
}

impl TerrainField for LandformField {
    fn height(&self, p: IVec3, level: u32) -> i32 {
        height(&self.constants, p, level)
    }
    fn ground_material(&self, p: IVec3, surface: u32, top_height: i32, depth: i32, slope: i32, layer: i32) -> u32 {
        ground_material(&self.constants, p, surface, top_height, depth, slope, layer)
    }
    fn surface(&self, p: IVec3, level: u32, _height: i32) -> u32 {
        surface(&self.constants, p, level)
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
    fn extent(&self, p: IVec3, level: u32) -> (i32, i32) {
        self.volume.extent(p, level, self.seed())
    }
    fn cell(&self, p: IVec3, q: IVec3, level: u32, top: i32, k: i32) -> u32 {
        self.volume.cell(p, q, level, top, k, self.seed())
    }
    fn volume_bounds(&self) -> (i32, i32) {
        let flags = self.volume.caves[0];
        (if flags & 1 != 0 { self.volume.caves[3] } else { 0 }, if flags & 2 != 0 { self.volume.overhangs[0] } else { 0 })
    }
    fn appearance(&self) -> TerrainAppearance {
        if self.constants.style[0] == STYLE_LUNAR { lunar_appearance() } else { TerrainAppearance::default() }
    }
    fn program(&self) -> TerrainProgram {
        let mut canonical = self.constants;
        canonical.shape[2] = i32::from(self.ridge_suffix.is_some());
        let mut constants = bytemuck::bytes_of(&canonical).to_vec();
        constants.extend_from_slice(bytemuck::cast_slice(&self.ridge_suffix.unwrap_or([[0; 4]; 66])));
        constants.extend_from_slice(bytemuck::bytes_of(&self.volume));
        TerrainProgram {
            key: Cow::Borrowed(DISPLAY_PROGRAM),
            wgsl: Cow::Borrowed(include_str!("../shaders/landform.wgsl")),
            constants,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::TerrainLayers;
    use crate::terrain::{generated_kind, top_cells, TerrainField};

    fn earth(grid: &Grid) -> LandformConstants {
        TerrainLayers::earth().compile(grid, 7).unwrap().0
    }

    #[test]
    #[ignore = "measurement"]
    fn noise_distribution_quantiles() {
        let mut v: Vec<i32> = Vec::new();
        let mut rng = 0x1234_5678_9abc_def0u64;
        let mut next = || {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng
        };
        for _ in 0..400_000 {
            let p = IVec3::new((next() % 1_000_000) as i32, (next() % 1_000_000) as i32, (next() % 1_000_000) as i32);
            v.push(noise(p, 10, next() as u32));
        }
        v.sort_unstable();
        let q = |f: f64| f64::from(v[((v.len() - 1) as f64 * f) as usize]) / f64::from(ONE);
        let mean = v.iter().map(|x| f64::from(*x)).sum::<f64>() / v.len() as f64;
        let sd = (v.iter().map(|x| (f64::from(*x) - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt() / f64::from(ONE);
        let near = v.iter().filter(|x| x.abs() < ONE / 20).count() as f64 / v.len() as f64;
        eprintln!("sd {sd:.4}; q50 {:.4} q75 {:.4} q90 {:.4} q95 {:.4} q99 {:.4}; P(|n| < 0.05) {near:.4}", q(0.5), q(0.75), q(0.9), q(0.95), q(0.99));
    }

    /// Caves and overhangs change cells only inside the declared extent;
    /// without them the stack is a pure heightfield.
    #[test]
    fn caves_and_overhangs_stay_inside_their_extent() {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        let v2 = TerrainLayers::earth().field(&grid, 7).unwrap();
        let v1 = TerrainLayers::earth().heightfield().field(&grid, 7).unwrap();
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = || {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; rng
        };
        let (mut cave, mut overhang, mut volumetric) = (0, 0, 0);
        for _ in 0..1500 {
            let face = (next() % 6) as u8;
            let (i, j) = ((next() % grid.cells() as u64) as i32, (next() % grid.cells() as u64) as i32);
            let p = grid.domain_point(face, i, j, 0);
            let top = top_cells(&grid, v2.height(p, grid.level_offset()), 0);
            assert_eq!(v1.extent(p, 0), (0, 0));
            let (below, above) = v2.extent(p, 0);
            volumetric += usize::from(below + above > 0);
            for _ in 0..24 {
                // Inside and around the extent.
                let k = top - below - 4 + (next() % (below + above + 8) as u64) as i32;
                let kind = generated_kind(&grid, &v2, face, i, j, k, 0, top);
                assert_eq!(generated_kind(&grid, &v1, face, i, j, k, 0, top), u32::from(k < top));
                if k < top - below || k >= top + above {
                    assert_eq!(kind, u32::from(k < top), "outside the extent the heightfield holds");
                }
                cave += usize::from(kind == 0 && k < top);
                overhang += usize::from(kind == 1 && k >= top);
            }
        }
        eprintln!("{volumetric}/1500 volumetric columns, {cave} cave and {overhang} overhang samples");
        assert!(volumetric > 100 && cave > 0 && overhang > 0);
    }

    #[test]
    fn climate_bounds_cover_both_signs_of_canonical_height_change() {
        for voxel in [0.1, 0.3, 1.0] {
            let grid = Grid::new(6_371_000.0, voxel).unwrap();
            let k = earth(&grid);
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
        let k = earth(&grid);
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
        let k = earth(&grid);
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
}
