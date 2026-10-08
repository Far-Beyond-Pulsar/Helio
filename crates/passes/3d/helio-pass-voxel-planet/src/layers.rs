//! Terrain as an ordered stack of layers (`helio.terrain`): every world,
//! from an Earth-like planet to a cratered moon or a flat block world, is a
//! list of layer kinds with parameters plus a material style. Presets
//! ([`TerrainLayers::earth`], [`TerrainLayers::moon`], [`TerrainLayers::flat`])
//! are just data; a game builds its own stacks (or randomizes them from a
//! seed) through the same settings.
//!
//! The stack compiles to one octave table sorted from coarsest to finest
//! (each octave tagged with its layer) that a single interpreter evaluates
//! on CPU and GPU (`landform.rs`, `landform.wgsl`): changing layers never
//! recompiles shaders, and erosion octaves see the slope of everything
//! coarser than themselves, in any layer, at every level.
use crate::grid::{Grid, DOMAIN_UNIT};
use crate::landform::{
    noise_quantile, LandformConstants, LandformField, LandformVolume, Octave, PackedRule, StackLayer, BASIN, CONTINENT, CRATER, CRATER_RADIUS_Q19, EROSION,
    GRAD_SHIFT, HILLS, LAYERS, OCTAVES, REGION, RIDGE, ROUGHNESS, RULES, STYLE_EARTHLIKE, STYLE_LAYERED, STYLE_LUNAR, STYLE_RULES, WARP, WARP_OCTAVES,
};
use crate::noise::{FINE_ONE, ONE};
use crate::terrain::{material, GeneratorInfo, TerrainField, TerrainGenerator, TerrainSource, HEIGHT_ONE};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

pub const ID: &str = "helio.terrain";
/// Output version, recorded with saved edits (one version is registered).
pub const VERSION: u32 = 1;

/// What a layer adds to the terrain.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerKind {
    /// Rolling fBm relief, warped with the domain (hills, highlands).
    #[default]
    Hills,
    /// Bends the domain of the layers after it (coastlines, ridge lines).
    /// At most one, before every other layer.
    Warp,
    /// Continents and ocean basins: `height_m` ocean depth, `base_m`
    /// lowland height; provides the land masks. At most one.
    Continents,
    /// Ridged mountain ranges inside regions (`coverage`, `region_km`).
    Mountains,
    /// Metre-scale fBm detail, not warped; `ratio` is its amplitude as a
    /// fraction of each wavelength.
    Roughness,
    /// Gullies down the slope of the coarser terrain, branching; full depth
    /// on slopes steeper than `ratio` (rise over run).
    Erosion,
    /// Crater octaves from `scale_km` (largest diameter) down; `coverage`
    /// is the density of the largest, `persistence` its growth per octave,
    /// `ratio` depth over diameter, `ratio2` rim over depth, `ratio3` the
    /// share of young craters with bright ejecta.
    Craters,
    /// Smooth low plains (maria, lakes beds) covering `coverage` of the
    /// surface, `height_m` deep.
    Basins,
    /// A constant height: `height_m` (a flat world is one plateau).
    Plateau,
}

/// Where a layer applies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerMask {
    #[default]
    Everywhere,
    /// Land only (rises from the coast; needs a Continents layer).
    Land,
    /// Everywhere but fading out under deep sea.
    AboveDeepSea,
}

/// One layer of a [`TerrainLayers`] stack. Fields mean what the layer's
/// kind says ([`LayerKind`]); unused ones are ignored.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Layer {
    pub kind: LayerKind,
    pub enabled: bool,
    pub mask: LayerMask,
    /// Main height (m): amplitude of the first octave, depth of basins or
    /// oceans, plateau height.
    pub height_m: f64,
    /// Secondary height (m): the lowlands of Continents.
    pub base_m: f64,
    /// Wavelength of the first octave or the largest feature (km).
    pub scale_km: f64,
    pub octaves: u32,
    /// Amplitude ratio from one octave to the next (craters: density growth).
    pub persistence: f64,
    /// Share of the surface covered (mountain regions, basins, crater density).
    pub coverage: f64,
    /// Size of the regions a layer occupies (mountain ranges), km.
    pub region_km: f64,
    pub ratio: f64,
    pub ratio2: f64,
    pub ratio3: f64,
}

impl Default for Layer {
    fn default() -> Self {
        Self {
            kind: LayerKind::Hills,
            enabled: true,
            mask: LayerMask::Everywhere,
            height_m: 100.0,
            base_m: 0.0,
            scale_km: 10.0,
            octaves: 4,
            persistence: 0.5,
            coverage: 0.5,
            region_km: 100.0,
            ratio: 0.0,
            ratio2: 0.0,
            ratio3: 0.0,
        }
    }
}

impl Layer {
    /// A layer of `kind` with that kind's defaults: added alone to a stack,
    /// it shows (craters with depth and rims, mountains with their ranges).
    /// The Earth preset is made of these.
    pub fn new(kind: LayerKind) -> Self {
        use LayerKind::*;
        let base = Self { kind, ..Self::default() };
        match kind {
            Hills => Self { height_m: 140.0, scale_km: 9.0, ..base },
            Warp => Self { scale_km: 40.0, ..base },
            Continents => Self { height_m: 2_400.0, base_m: 180.0, scale_km: 3_000.0, ..base },
            Mountains => Self { height_m: 2_400.0, scale_km: 20.0, octaves: 7, persistence: 0.47, region_km: 240.0, coverage: 0.45, ..base },
            Roughness => Self { scale_km: 0.512, octaves: 9, ratio: 0.035, ..base },
            Erosion => Self { height_m: 40.0, scale_km: 1.6, octaves: 6, ratio: 0.5, ..base },
            Craters => Self { scale_km: 40.0, octaves: 8, coverage: 0.3, persistence: 1.25, ratio: 0.2, ratio2: 0.3, ratio3: 0.15, ..base },
            Basins => Self { height_m: 1_200.0, scale_km: 900.0, coverage: 0.3, ..base },
            Plateau => Self { height_m: 400.0, ..base },
        }
    }
}

/// How surface and buried cells get their materials.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialStyle {
    /// Meadows, dry lands, rock outcrops, strata and snow above `snowline_m`;
    /// gravel in erosion gullies.
    #[default]
    Earthlike,
    /// Regolith over bedrock, dark mare plains, bright young ejecta.
    Lunar,
    /// A `surface` layer over `soil_depth_m` of `soil` over `rock`.
    Layered,
    /// The first of the [`MaterialRule`]s that holds, else `rock`: a game's
    /// own biomes (snow above a height, rock on steep slopes, sediment in
    /// gullies, strata, patches).
    Rules,
}

/// One material rule: the material of a ground cell where every condition
/// holds (ranges are inclusive). Rules are tried in order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MaterialRule {
    /// Material name (see [`material::NAMES`]).
    pub material: String,
    /// Column height above the datum (m).
    pub min_height_m: f64,
    pub max_height_m: f64,
    /// Ground slope, rise over run (1 is 45 degrees).
    pub min_slope: f64,
    pub max_slope: f64,
    /// Depth below the column top (m); 0 is the exposed surface.
    pub min_depth_m: f64,
    pub max_depth_m: f64,
    /// Moisture, 0 (dry) to 1 (wet), varying over the continents' scale.
    pub min_moisture: f64,
    pub max_moisture: f64,
    /// Erosion, -1 (gully floors) to 1 (the ribs between gullies).
    pub min_erosion: f64,
    pub max_erosion: f64,
    /// Patches: noise blobs of this size (km; 0 none) covering `patch_share`.
    pub patch_km: f64,
    pub patch_share: f64,
    /// Strata: bands of this thickness (m; 0 none), the odd ones or the even.
    pub band_m: f64,
    pub odd_bands: bool,
    /// Single-cell specks: this share of the cells (1 for all cells).
    pub speck_share: f64,
}

impl Default for MaterialRule {
    fn default() -> Self {
        Self {
            material: "Stone".into(),
            min_height_m: -1.0e6,
            max_height_m: 1.0e6,
            min_slope: 0.0,
            max_slope: 1.0e3,
            min_depth_m: 0.0,
            max_depth_m: 1.0e6,
            min_moisture: 0.0,
            max_moisture: 1.0,
            min_erosion: -1.0,
            max_erosion: 1.0,
            patch_km: 0.0,
            patch_share: 0.5,
            band_m: 0.0,
            odd_bands: false,
            speck_share: 1.0,
        }
    }
}

impl MaterialRule {
    /// `material` everywhere (narrow it with the fields).
    pub fn new(material: &str) -> Self {
        Self { material: material.into(), ..Self::default() }
    }

    fn validate(&self, index: usize) -> Result<(), String> {
        let values = [
            self.min_height_m, self.max_height_m, self.min_slope, self.max_slope, self.min_depth_m, self.max_depth_m, self.min_moisture,
            self.max_moisture, self.min_erosion, self.max_erosion, self.patch_km, self.patch_share, self.band_m, self.speck_share,
        ];
        if values.iter().any(|v| !v.is_finite()) || self.patch_km < 0.0 || self.band_m < 0.0 {
            return Err(format!("material rule {index} has a non-finite or negative size"));
        }
        if material::from_name(&self.material).is_none() {
            return Err(format!("material rule {index}: unknown material {:?}", self.material));
        }
        Ok(())
    }

    fn pack(&self, grid: &Grid) -> PackedRule {
        let mm = |m: f64| (m * f64::from(HEIGHT_ONE)).clamp(-2.0e9, 2.0e9).round() as i32;
        let eighths = |s: f64| (s * 8.0).clamp(-1.0e9, 1.0e9).floor() as i32;
        let cells = |m: f64| (m / grid.voxel_size()).clamp(-1.0e9, 1.0e9).round() as i32;
        let q16 = |v: f64| (v.clamp(0.0, 1.0) * 65_536.0).round() as i32;
        let byte = |v: f64| (v.clamp(-1.0, 1.0) * 127.0).round() as i32;
        let patch = if self.patch_km > 0.0 {
            let shift = ((self.patch_km * 1_000.0 / DOMAIN_UNIT).log2().round().clamp(1.0, 30.0)) as i32;
            (shift, (noise_quantile(1.0 - self.patch_share.clamp(0.0, 1.0)) * f64::from(ONE)).round() as i32)
        } else {
            (0, 0)
        };
        let speck = if self.speck_share >= 1.0 { 0 } else { ((self.speck_share.max(0.0) * 256.0).round() as i32).clamp(1, 255) };
        PackedRule {
            head: [material::from_name(&self.material).unwrap_or(material::STONE) as i32, speck, patch.0, patch.1],
            height_slope: [mm(self.min_height_m), mm(self.max_height_m), eighths(self.min_slope), eighths(self.max_slope)],
            depth_moisture: [cells(self.min_depth_m), cells(self.max_depth_m), q16(self.min_moisture), q16(self.max_moisture)],
            surface_bands: [byte(self.min_erosion), byte(self.max_erosion), mm(self.band_m).max(0), i32::from(self.odd_bands)],
        }
    }
}

/// Generated caves: tunnels and caverns inside cave regions.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Caves {
    pub enabled: bool,
    /// Deepest cave cell below the local surface.
    pub depth_m: f64,
    /// Rough share of the land inside cave regions and their size.
    pub share: f64,
    pub region_km: f64,
    /// Tunnel radius and the wavelength of their winding.
    pub tunnel_radius_m: f64,
    pub tunnel_wavelength_m: f64,
    /// Cavern size and the rough share of the cave volume they open.
    pub cavern_wavelength_m: f64,
    pub cavern_share: f64,
    /// Rock kept over every cave: tunnels and caverns close towards it
    /// instead of breaking the surface, except at entrances.
    pub cover_m: f64,
    /// Rough share of a cave region where tunnels open to the surface, in
    /// zones about `entrance_spacing_m` across.
    pub entrance_share: f64,
    pub entrance_spacing_m: f64,
}

impl Default for Caves {
    fn default() -> Self {
        Self {
            enabled: true,
            depth_m: 120.0,
            share: 0.45,
            region_km: 6.0,
            tunnel_radius_m: 2.5,
            tunnel_wavelength_m: 160.0,
            cavern_wavelength_m: 160.0,
            cavern_share: 0.04,
            cover_m: 6.0,
            entrance_share: 0.05,
            entrance_spacing_m: 80.0,
        }
    }
}

/// Generated overhangs and arches: the surface displaced in 3D.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Overhangs {
    pub enabled: bool,
    /// Largest displacement over the heightfield.
    pub height_m: f64,
    pub wavelength_m: f64,
    /// Size and rough share of the regions with overhangs.
    pub region_km: f64,
    pub share: f64,
}

impl Default for Overhangs {
    fn default() -> Self {
        Self { enabled: true, height_m: 6.0, wavelength_m: 24.0, region_km: 3.0, share: 0.3 }
    }
}

/// A world's terrain: an ordered layer stack, volumetric features and a
/// material style (the `helio.terrain` settings JSON).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TerrainLayers {
    pub layers: Vec<Layer>,
    pub caves: Caves,
    pub overhangs: Overhangs,
    pub materials: MaterialStyle,
    /// Earthlike: flat ground above this height is snow.
    pub snowline_m: f64,
    /// Depth of the soil (Earthlike, Layered) or regolith (Lunar).
    pub soil_depth_m: f64,
    /// Layered: material names (see [`material::NAMES`]); `rock` is also
    /// the Rules style's fallback.
    pub surface: String,
    pub soil: String,
    pub rock: String,
    /// Rules style: tried in order (at most 16).
    pub rules: Vec<MaterialRule>,
}

impl Default for TerrainLayers {
    fn default() -> Self {
        Self::earth()
    }
}

impl TerrainLayers {
    /// Continents and ocean basins, ridged mountain ranges with branching
    /// erosion gullies, hills and metre-scale roughness; caves, overhangs,
    /// meadows, dry lands, rock, strata and snow.
    pub fn earth() -> Self {
        use LayerKind::*;
        Self {
            layers: vec![
                Layer::new(Warp),
                Layer::new(Continents),
                Layer { mask: LayerMask::Land, ..Layer::new(Mountains) },
                Layer::new(Erosion),
                Self::earth_hills(),
                Layer { mask: LayerMask::AboveDeepSea, ..Layer::new(Roughness) },
            ],
            caves: Caves::default(),
            overhangs: Overhangs::default(),
            materials: MaterialStyle::Earthlike,
            snowline_m: 3_000.0,
            soil_depth_m: 0.7,
            surface: "Grass".into(),
            soil: "Dirt".into(),
            rock: "Stone".into(),
            rules: Vec::new(),
        }
    }

    /// Earth's hills: 140 m over 9 km down to knolls of ~30 m over a
    /// kilometre, so the land near the eye has shape (four octaves at half
    /// persistence left it sloping ~3 %).
    fn earth_hills() -> Layer {
        Layer { mask: LayerMask::AboveDeepSea, octaves: 5, persistence: 0.6, ..Layer::new(LayerKind::Hills) }
    }

    /// A dry world of dunes and mesas, with materials from rules: dune sand
    /// on gentle ground, sandstone and clay strata on slopes, gravel in the
    /// gullies, scattered stone patches and dark stone specks.
    pub fn desert() -> Self {
        use LayerKind::*;
        let mut earth = Self::earth();
        earth.layers.retain(|l| l.kind != Continents);
        // The desert's hills stay gentle under its dunes.
        for l in earth.layers.iter_mut().filter(|l| l.kind == Hills) {
            *l = Layer { mask: l.mask, ..Layer::new(Hills) };
        }
        for l in &mut earth.layers {
            l.mask = LayerMask::Everywhere;
        }
        earth.layers.insert(1, Layer::new(Plateau));
        let rule = |material: &str, f: &dyn Fn(&mut MaterialRule)| {
            let mut r = MaterialRule::new(material);
            f(&mut r);
            r
        };
        Self {
            materials: MaterialStyle::Rules,
            rules: vec![
                rule("Gravel", &|r| {
                    r.max_depth_m = 0.3;
                    r.max_erosion = -0.4;
                    r.min_slope = 0.2;
                }),
                rule("Sand", &|r| {
                    r.max_depth_m = 1.5;
                    r.max_slope = 0.6;
                }),
                rule("DarkStone", &|r| {
                    r.max_depth_m = 0.0;
                    r.speck_share = 0.06;
                }),
                rule("Stone", &|r| {
                    r.patch_km = 0.05;
                    r.patch_share = 0.15;
                }),
                rule("Clay", &|r| {
                    r.band_m = 2.5;
                    r.odd_bands = true;
                }),
            ],
            rock: "Sandstone".into(),
            snowline_m: 1.0e5,
            ..earth
        }
    }

    /// Cratered highlands and dark basalt plains over regolith, no caves.
    /// The basins come after the highlands, which they flatten.
    pub fn moon() -> Self {
        use LayerKind::*;
        Self {
            layers: vec![
                Layer { kind: Hills, height_m: 1_500.0, scale_km: 250.0, octaves: 3, persistence: 0.5, ..Layer::default() },
                Layer { kind: Basins, height_m: 1_200.0, scale_km: 900.0, coverage: 0.3, ..Layer::default() },
                Layer {
                    kind: Craters,
                    scale_km: 40.0,
                    octaves: 10,
                    coverage: 0.3,
                    persistence: 1.25,
                    ratio: 0.2,
                    ratio2: 0.3,
                    ratio3: 0.15,
                    ..Layer::default()
                },
            ],
            caves: Caves { enabled: false, ..Caves::default() },
            overhangs: Overhangs { enabled: false, ..Overhangs::default() },
            materials: MaterialStyle::Lunar,
            soil_depth_m: 4.0,
            ..Self::earth()
        }
    }

    /// Level ground: one plateau, grass over a metre of dirt over stone.
    pub fn flat() -> Self {
        Self::flat_at(0.0)
    }

    /// [`Self::flat`] at `height_m` (rounded down to whole voxel layers).
    pub fn flat_at(height_m: f64) -> Self {
        Self {
            layers: vec![Layer { kind: LayerKind::Plateau, height_m, ..Layer::default() }],
            caves: Caves { enabled: false, ..Caves::default() },
            overhangs: Overhangs { enabled: false, ..Overhangs::default() },
            materials: MaterialStyle::Layered,
            soil_depth_m: 1.0,
            ..Self::earth()
        }
    }

    /// Without caves and overhangs: a pure heightfield.
    pub fn heightfield(mut self) -> Self {
        self.caves.enabled = false;
        self.overhangs.enabled = false;
        self
    }

    /// The stack's errors, before it is compiled.
    pub fn validate(&self) -> Result<(), String> {
        let enabled: Vec<&Layer> = self.layers.iter().filter(|l| l.enabled).collect();
        if enabled.len() > LAYERS {
            return Err(format!("a terrain has at most {LAYERS} enabled layers"));
        }
        for (index, l) in enabled.iter().enumerate() {
            let values = [l.height_m, l.base_m, l.scale_km, l.persistence, l.coverage, l.region_km, l.ratio, l.ratio2, l.ratio3];
            if values.iter().any(|v| !v.is_finite()) {
                return Err(format!("layer {index} ({:?}) has a non-finite value", l.kind));
            }
            if l.kind != LayerKind::Plateau && l.scale_km <= 0.0 {
                return Err(format!("layer {index} ({:?}) needs a positive scale", l.kind));
            }
            if l.height_m.abs() > 1.0e5 || l.base_m.abs() > 1.0e5 || l.scale_km > 1.0e5 {
                return Err(format!("layer {index} ({:?}) exceeds 100 km", l.kind));
            }
            if l.kind == LayerKind::Warp && index != 0 {
                return Err("a Warp layer must come first".into());
            }
            if l.mask != LayerMask::Everywhere && !enabled.iter().any(|o| o.kind == LayerKind::Continents) {
                return Err(format!("layer {index} ({:?}) is masked by land without a Continents layer", l.kind));
            }
        }
        for kind in [LayerKind::Warp, LayerKind::Continents] {
            if enabled.iter().filter(|l| l.kind == kind).count() > 1 {
                return Err(format!("a terrain has at most one {kind:?} layer"));
            }
        }
        let c = &self.caves;
        let o = &self.overhangs;
        let volume = [c.depth_m, c.share, c.region_km, c.tunnel_radius_m, c.tunnel_wavelength_m, c.cavern_wavelength_m, c.cavern_share, c.cover_m, c.entrance_share, c.entrance_spacing_m, o.height_m, o.wavelength_m, o.region_km, o.share];
        if volume.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err("cave and overhang settings must be finite and non-negative".into());
        }
        if c.region_km <= 0.0 || c.tunnel_wavelength_m <= 0.0 || c.cavern_wavelength_m <= 0.0 || c.entrance_spacing_m <= 0.0 || o.wavelength_m <= 0.0 || o.region_km <= 0.0 {
            return Err("cave and overhang wavelengths must be positive".into());
        }
        if !self.snowline_m.is_finite() || !self.soil_depth_m.is_finite() || self.soil_depth_m < 0.0 {
            return Err("snowline and soil depth must be finite (soil depth non-negative)".into());
        }
        for name in [&self.surface, &self.soil, &self.rock] {
            if material::from_name(name).is_none() {
                return Err(format!("unknown material {name:?}"));
            }
        }
        if self.rules.len() > RULES {
            return Err(format!("a terrain has at most {RULES} material rules"));
        }
        for (index, rule) in self.rules.iter().enumerate() {
            rule.validate(index)?;
        }
        Ok(())
    }

    /// The interpreter's constants and volumetric terms on `grid`.
    pub fn compile(&self, grid: &Grid, seed: u32) -> Result<(LandformConstants, LandformVolume), String> {
        self.validate()?;
        let units = |metres: f64| (metres * f64::from(HEIGHT_ONE)).round() as i32;
        let shift = |metres: f64| ((metres / DOMAIN_UNIT).log2().round().clamp(1.0, 30.0)) as u32;
        let mut state = seed.wrapping_mul(0x9E37_79B9);
        let mut next_seed = || {
            state = state.wrapping_add(0x6D2B_79F5);
            state
        };
        let mut warp = [Octave::default(); WARP_OCTAVES];
        for (index, w) in warp.iter_mut().enumerate() {
            // Zero-amplitude placeholders without a Warp layer.
            *w = Octave { shift: 20, amplitude: 0, seed: 0, kind: WARP + index as u32 / 2 };
        }
        let mut octaves: Vec<Octave> = Vec::new();
        let mut layers = [StackLayer::default(); LAYERS];
        let mut erosion_sum = 0i32;
        let mut display = 0;
        let mut continents = 0;
        let mut warps = 0;
        for (index, l) in self.layers.iter().filter(|l| l.enabled).enumerate() {
            let tag = (index as u32) << 8;
            let mask = match l.mask {
                LayerMask::Everywhere => 0,
                LayerMask::Land => 1,
                LayerMask::AboveDeepSea => 2,
            };
            let metres = l.scale_km * 1_000.0;
            let octave = |o: u32| metres / f64::from(1u32 << o);
            let mut layer = StackLayer { kind: 0, mask, a: 0, b: 0 };
            match l.kind {
                LayerKind::Warp => {
                    // Two octaves per axis, amplitude 15% of the wavelength.
                    let amplitude = metres * 0.15 / DOMAIN_UNIT;
                    for axis in 0..3 {
                        for o in 0..2 {
                            warp[axis * 2 + o] = Octave {
                                shift: shift(octave(o as u32)),
                                amplitude: (amplitude / f64::from(1u32 << o)).round() as i32,
                                seed: next_seed(),
                                kind: WARP + axis as u32,
                            };
                        }
                    }
                    warps = WARP_OCTAVES as i32;
                    layer.kind = StackLayer::WARP;
                }
                LayerKind::Continents => {
                    layer.kind = StackLayer::CONTINENTS;
                    layer.a = units(-l.height_m.abs());
                    layer.b = units(l.base_m);
                    continents = index as i32 + 1;
                    for o in 0..4 {
                        octaves.push(Octave { shift: shift(octave(o)), amplitude: ONE >> o, seed: next_seed(), kind: tag | CONTINENT });
                    }
                }
                LayerKind::Mountains => {
                    layer.kind = StackLayer::MOUNTAINS;
                    // Region mask: two fine octaves; the bias keeps `coverage`.
                    layer.a = (noise_quantile(1.0 - l.coverage.clamp(0.0, 1.0)) * f64::from(ONE) * 1.2).round() as i32;
                    let region = l.region_km * 1_000.0;
                    for o in 0..2 {
                        octaves.push(Octave { shift: shift(region / f64::from(1u32 << o)), amplitude: ONE >> o, seed: next_seed(), kind: tag | REGION });
                    }
                    let mut amplitude = l.height_m;
                    for o in 0..l.octaves.min(12) {
                        octaves.push(Octave { shift: shift(octave(o)), amplitude: units(amplitude), seed: next_seed(), kind: tag | RIDGE });
                        amplitude *= l.persistence;
                    }
                    // The first mountain layer with a bakeable ridge chain
                    // draws coarse levels from its ridge envelope.
                    if display == 0 && (1..=crate::ridge_envelope::RIDGES as u32).contains(&l.octaves) {
                        display = index as i32 + 1;
                    }
                }
                LayerKind::Hills | LayerKind::Roughness => {
                    let (kind, class) = if l.kind == LayerKind::Hills { (StackLayer::HILLS, HILLS) } else { (StackLayer::ROUGHNESS, ROUGHNESS) };
                    layer.kind = kind;
                    let mut amplitude = l.height_m;
                    for o in 0..l.octaves.min(16) {
                        let w = octave(o);
                        let a = if l.kind == LayerKind::Roughness { w * l.ratio } else { amplitude };
                        octaves.push(Octave { shift: shift(w), amplitude: units(a), seed: next_seed(), kind: tag | class });
                        amplitude *= l.persistence;
                    }
                }
                LayerKind::Erosion => {
                    layer.kind = StackLayer::EROSION;
                    layer.a = ((l.ratio.max(0.01) * f64::from(1u32 << GRAD_SHIFT) * DOMAIN_UNIT * 1_000.0).round() as i32).clamp(32, 1 << 26);
                    let mut amplitude = l.height_m;
                    for o in 0..l.octaves.min(8) {
                        if amplitude > 0.0 && octave(o) > 1.0 {
                            octaves.push(Octave { shift: shift(octave(o)), amplitude: units(amplitude), seed: next_seed(), kind: tag | EROSION });
                            erosion_sum += units(amplitude);
                        }
                        amplitude *= l.persistence;
                    }
                }
                LayerKind::Craters => {
                    layer.kind = StackLayer::CRATERS;
                    layer.a = (l.ratio2.clamp(0.0, 4.0) * 65_536.0).round() as i32;
                    layer.b = (l.ratio3.clamp(0.0, 1.0) * 65_536.0).round() as i32;
                    for o in 0..l.octaves.min(12) {
                        let diameter = octave(o);
                        // A cell holds one crater of at most 0.3 cells radius.
                        let cell = diameter / (2.0 * f64::from(CRATER_RADIUS_Q19) / 524_288.0);
                        if cell < DOMAIN_UNIT * 1024.0 {
                            break;
                        }
                        let depth = l.ratio.max(0.0) * diameter * (15_000.0 / diameter).min(1.0).powf(0.7);
                        let density = (l.coverage * l.persistence.powi(o as i32)).clamp(0.0, 0.95);
                        // Crater octaves carry their density in the seed's
                        // low 16 bits (the hash seed is the rest).
                        let seed = (next_seed() & !0xffff) | ((density * 65_535.0) as u32 & 0xffff);
                        octaves.push(Octave { shift: shift(cell), amplitude: units(depth), seed, kind: tag | CRATER });
                    }
                }
                LayerKind::Basins => {
                    layer.kind = StackLayer::BASINS;
                    layer.a = units(l.height_m.abs());
                    layer.b = ((noise_quantile(1.0 - l.coverage.clamp(0.0, 1.0)) * f64::from(FINE_ONE)).round()) as i32;
                    octaves.push(Octave { shift: shift(metres), amplitude: ONE, seed: next_seed(), kind: tag | BASIN });
                }
                LayerKind::Plateau => {
                    layer.kind = StackLayer::PLATEAU;
                    layer.a = units(l.height_m).div_euclid(grid.layer_mm() as i32) * grid.layer_mm() as i32;
                }
            }
            layers[index] = layer;
        }
        // Coarsest first, stable: each octave follows only coarser terrain
        // (erosion is steered by what precedes it), and the ridge weight
        // chains keep their order.
        octaves.sort_by(|a, b| b.shift.cmp(&a.shift));
        if WARP_OCTAVES + octaves.len() > OCTAVES {
            return Err(format!("the layers have {} octaves; at most {}", octaves.len(), OCTAVES - WARP_OCTAVES));
        }
        let mut table = [Octave::default(); OCTAVES];
        table[..WARP_OCTAVES].copy_from_slice(&warp);
        table[WARP_OCTAVES..WARP_OCTAVES + octaves.len()].copy_from_slice(&octaves);
        let soil = ((self.soil_depth_m / grid.voxel_size()).round() as i32).max(1);
        let id = |name: &str| material::from_name(name).unwrap_or(material::STONE) as i32;
        // Moisture follows the continents' scale (or 3000 km without them).
        let moisture = self
            .layers
            .iter()
            .find(|l| l.enabled && l.kind == LayerKind::Continents)
            .map_or(3_000.0, |l| l.scale_km)
            * 1_000.0;
        let style = match self.materials {
            MaterialStyle::Earthlike => STYLE_EARTHLIKE,
            MaterialStyle::Lunar => STYLE_LUNAR,
            MaterialStyle::Layered => STYLE_LAYERED,
            MaterialStyle::Rules => STYLE_RULES,
        };
        let mut rules = [PackedRule::default(); RULES];
        for (packed, rule) in rules.iter_mut().zip(&self.rules) {
            *packed = rule.pack(grid);
        }
        let constants = LandformConstants {
            header: [(WARP_OCTAVES + octaves.len()) as i32, grid.layer_mm() as i32, soil, seed as i32],
            levels: [shift(moisture / 2.0) as i32, continents, units(self.snowline_m), units(-8.0)],
            shape: [display, 16, 0, i32::from(grid.is_plane())],
            style: [style, id(&self.surface), id(&self.soil), id(&self.rock)],
            stack: [self.layers.iter().filter(|l| l.enabled).count() as i32, erosion_sum.max(1), grid.sphere_constants()[2] as i32, warps],
            materials: [self.rules.len() as i32, id(&self.rock), 0, 0],
            layers,
            octaves: table,
            rules,
        };
        Ok((constants, LandformVolume::new(grid, &self.caves, &self.overhangs)))
    }
}

impl TerrainLayers {
    /// The compiled field on `grid`.
    pub fn field(&self, grid: &Grid, seed: u32) -> Result<LandformField, String> {
        let (constants, volume) = self.compile(grid, seed)?;
        Ok(LandformField::new(grid, constants, volume))
    }

    /// A world terrain source using this stack.
    pub fn source(&self, seed: u64) -> TerrainSource {
        TerrainSource { generator: ID.into(), version: 0, seed, settings: serde_json::to_string(self).expect("terrain layers serialize") }
    }
}

/// Builds [`LandformField`]s from [`TerrainLayers`] settings.
pub struct LayersGenerator;

impl TerrainGenerator for LayersGenerator {
    fn info(&self) -> GeneratorInfo {
        GeneratorInfo {
            id: ID.into(),
            version: VERSION,
            name: "Terrain layers".into(),
            description: "An ordered stack of terrain layers (continents, mountains, erosion, hills, craters, basins, plateaus) with caves, overhangs and a material style.".into(),
            settings_component: Some("VoxelTerrainLayersComponent".into()),
        }
    }
    fn build(&self, grid: &Grid, seed: u64, settings: &str) -> Result<Arc<dyn TerrainField>, String> {
        let layers: TerrainLayers = if settings.trim().is_empty() {
            TerrainLayers::default()
        } else {
            serde_json::from_str(settings).map_err(|e| format!("invalid terrain layers: {e}"))?
        };
        Ok(Arc::new(layers.field(grid, (seed ^ (seed >> 32)) as u32)?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::Shape;
    use crate::planet::{Planet, PlanetRecipe};
    use crate::terrain::check_field;
    use glam::IVec3;

    fn planet(shape: Shape, radius_m: f64, stack: &TerrainLayers) -> Planet {
        Planet::new(PlanetRecipe { shape, radius_m, plane_size_m: 4_000.0, terrain: stack.source(7), ..Default::default() }).unwrap()
    }

    #[test]
    fn flat_ground_is_one_whole_layer_with_its_materials() {
        let grid = Grid::plane(Shape::Plane, 1024.0, 0.1).unwrap();
        let stack = TerrainLayers { soil_depth_m: 0.5, surface: "Sand".into(), rock: "dark_stone".into(), ..TerrainLayers::flat_at(2.34) };
        let field = LayersGenerator.build(&grid, 0, &serde_json::to_string(&stack).unwrap()).unwrap();
        assert_eq!(field.height(IVec3::ZERO, 0), 2_300);
        assert_eq!(field.height_range(), (2_300, 2_300));
        assert_eq!(field.bound_margins(), [2; 24]);
        assert_eq!(field.ground_material(IVec3::ZERO, 0, 2_300, 0, 0, 22), material::SAND);
        assert_eq!(field.ground_material(IVec3::ZERO, 0, 2_300, 5, 0, 17), material::DIRT);
        assert_eq!(field.ground_material(IVec3::ZERO, 0, 2_300, 6, 0, 16), material::DARK_STONE);
        assert!(LayersGenerator.build(&grid, 0, r#"{"rock": "Air"}"#).is_err());
    }

    /// Craters exist at every size, basins set the surface word, and the
    /// bounds hold on samples, on spheres and planes.
    #[test]
    fn a_moon_is_craters_and_basins_within_its_bounds() {
        let stack = TerrainLayers::moon();
        for shape in [Shape::Sphere, Shape::Plane] {
            let planet = planet(shape, 1_737_400.0, &stack);
            check_field(&planet, 2_000).unwrap();
            let g = planet.grid();
            let n = g.cells();
            let (mut lo, mut hi, mut fresh, mut mare) = (i32::MAX, i32::MIN, 0, 0);
            for t in 0..4_000 {
                let face = if g.is_plane() { crate::grid::PLANE_FACE } else { 2 };
                let p = g.domain_point(face, n / 4 + t * 3, n / 2, 0);
                let h = planet.field().height(p, g.level_offset());
                let s = planet.field().surface(p, g.level_offset(), h);
                lo = lo.min(h);
                hi = hi.max(h);
                fresh += usize::from(s & 0x7f > 0);
                mare += usize::from(s & 0x80 != 0);
            }
            assert!(hi - lo > 1_000, "{shape:?}: relief {lo}..{hi}");
            eprintln!("{shape:?}: relief {lo}..{hi} mm, {fresh} samples on fresh ejecta, {mare} on basins");
            assert_eq!(planet.field().appearance(), crate::landform::lunar_appearance());
        }
    }

    /// No steps between neighbouring columns (the bowls and rims are
    /// continuous; the rim's crease is a slope change, not a step).
    #[test]
    fn the_moon_has_no_steps_between_neighbouring_columns() {
        let planet = planet(Shape::Sphere, 1_737_400.0, &TerrainLayers::moon());
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

    /// Stacks the presets never make: two mountain layers, craters on land,
    /// basins flattening a plateau, layers masked by the coast. Their
    /// bounds hold all the same.
    #[test]
    fn custom_stacks_keep_their_bounds() {
        use LayerKind::*;
        let base = TerrainLayers::earth();
        let mut alien = base.clone();
        alien.layers.insert(3, Layer { kind: Mountains, mask: LayerMask::AboveDeepSea, height_m: 900.0, scale_km: 4.0, octaves: 5, region_km: 60.0, coverage: 0.3, ..Layer::default() });
        alien.layers.push(Layer { kind: Craters, mask: LayerMask::Land, scale_km: 12.0, octaves: 4, coverage: 0.4, persistence: 1.3, ratio: 0.15, ratio2: 0.4, ratio3: 0.5, ..Layer::default() });
        alien.layers.retain(|l| l.kind != Roughness);
        let mesa = TerrainLayers {
            layers: vec![
                Layer { kind: Plateau, height_m: 300.0, ..Layer::default() },
                Layer { kind: Hills, height_m: 80.0, scale_km: 2.0, octaves: 5, persistence: 0.55, ..Layer::default() },
                Layer { kind: Basins, height_m: 250.0, scale_km: 20.0, coverage: 0.4, ..Layer::default() },
                Layer { kind: Erosion, height_m: 15.0, scale_km: 0.8, octaves: 4, persistence: 0.5, ratio: 0.3, ..Layer::default() },
            ],
            materials: MaterialStyle::Layered,
            surface: "Sand".into(),
            soil: "Clay".into(),
            rock: "Sandstone".into(),
            ..base.clone()
        };
        for (name, stack) in [("alien", alien), ("mesa", mesa)] {
            for shape in [Shape::Sphere, Shape::Plane] {
                let planet = planet(shape, 2_000_000.0, &stack);
                let worst = check_field(&planet, 3_000).unwrap_or_else(|e| panic!("{name} {shape:?}: {e}"));
                eprintln!("{name} {shape:?}: worst {worst:?}");
            }
        }
    }

    /// Rules apply in order: the first that holds picks the material.
    #[test]
    fn material_rules_pick_the_first_rule_that_holds() {
        use crate::landform::ground_material;
        let grid = Grid::new(1_000_000.0, 0.1).unwrap();
        let (k, _) = TerrainLayers::desert().compile(&grid, 7).unwrap();
        let gully = (-100i32 as u32) & 0xff;
        let id = |m: u32| m & material::ID;
        let mut seen = std::collections::BTreeMap::new();
        for n in 0..4_000 {
            let p = grid.domain_point(2, 1_000 + n * 977, 2_000 + n * 131, 0);
            // A gully floor on a slope: gravel, the first rule.
            assert_eq!(ground_material(&k, p, gully, 500_000, 0, 4, 5_000), material::GRAVEL);
            // Gentle ground near the surface: sand.
            assert_eq!(ground_material(&k, p, 0, 500_000, 3, 2, 5_000), material::SAND);
            // Deeper: stone patches, clay bands or the sandstone fallback.
            let deep = id(ground_material(&k, p, 0, 500_000, 40, 2, 4_960 + n % 40));
            assert!([material::STONE, material::CLAY, material::SANDSTONE].contains(&deep), "{deep}");
            *seen.entry(deep).or_insert(0) += 1;
            // Steep exposed ground: dark stone specks among the others.
            *seen.entry(1000 + id(ground_material(&k, p, 0, 500_000, 0, 10, 5_000))).or_insert(0) += 1;
        }
        eprintln!("{seen:?}");
        assert!(seen.len() >= 5, "patches, bands, fallback and specks all occur: {seen:?}");
        let mut many = TerrainLayers::desert();
        many.rules = vec![MaterialRule::default(); 17];
        assert!(many.compile(&grid, 7).is_err());
        many.rules = vec![MaterialRule::new("Lava")];
        assert!(many.compile(&grid, 7).is_err());
    }

    #[test]
    fn invalid_stacks_are_rejected() {
        use LayerKind::*;
        let grid = Grid::new(1_000_000.0, 0.1).unwrap();
        let with = |layers: Vec<Layer>| TerrainLayers { layers, ..TerrainLayers::earth() };
        let hills = Layer { kind: Hills, ..Layer::default() };
        assert!(with(vec![hills.clone(); LAYERS + 1]).compile(&grid, 1).is_err());
        assert!(with(vec![hills.clone(), Layer { kind: Warp, ..Layer::default() }]).compile(&grid, 1).is_err());
        assert!(with(vec![Layer { mask: LayerMask::Land, ..hills.clone() }]).compile(&grid, 1).is_err());
        assert!(with(vec![Layer { scale_km: f64::NAN, ..hills.clone() }]).compile(&grid, 1).is_err());
        assert!(with(vec![Layer { kind: Roughness, octaves: 16, ..Layer::default() }; 4]).compile(&grid, 1).is_err(), "too many octaves");
        // Disabled layers do not count.
        let mut many = vec![Layer { enabled: false, ..hills.clone() }; LAYERS + 4];
        many.push(hills);
        assert!(with(many).compile(&grid, 1).is_ok());
    }

    /// Settings round-trip through JSON; empty settings are the Earth preset.
    #[test]
    fn presets_round_trip_and_empty_settings_are_the_earth() {
        let grid = Grid::new(6_371_000.0, 0.1).unwrap();
        for stack in [TerrainLayers::earth(), TerrainLayers::moon(), TerrainLayers::flat()] {
            let json = serde_json::to_string(&stack).unwrap();
            assert_eq!(serde_json::from_str::<TerrainLayers>(&json).unwrap(), stack);
        }
        let empty = LayersGenerator.build(&grid, 7, "").unwrap();
        let earth = TerrainLayers::earth().field(&grid, 7).unwrap();
        assert_eq!(empty.program(), earth.program());
    }
}
