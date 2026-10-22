//! The flight's worlds (`HELIO_VOXEL_FLIGHT_PRESET=earth|moon|desert`):
//! each one's planet (layer stack, radius, features), its look (materials
//! past the built-in ones), sky, haze and ground cover, all through the
//! engine's generic APIs: the stress test exercises what voxel_craft does,
//! at Earth and Moon scale.
//!
//! * Earth: broadleaf and birch woods in the lowlands, conifers up to the
//!   tree line, boulders on rock and scree, bushes in the meadows, shrubs
//!   and rocks on dry land; the engine's meadow with tall stands in a
//!   breeze; Earth's air with cumulus and a ground haze.
//! * Moon (1737 km): no air, no plants: simple craters of 1 to 36 m with
//!   lumpy rims of the ground they hit (fresh ones bright, with blocks
//!   thrown out), boulders, and boulder fields on fresh ejecta.
//! * Desert: sparse shrubs, rocks and sandstone boulders on the dunes;
//!   dusty air with a few high clouds and a warm haze.

use helio_pass_voxel_planet::features::{FeatureSet, Spawn, MAX_SPACING};
use helio_pass_voxel_planet::flora::{self, Species};
use helio_pass_voxel_planet::layers::TerrainLayers;
use helio_pass_voxel_planet::meadow::Meadow;
use helio_pass_voxel_planet::planet::EARTH_RADIUS;
use helio_pass_voxel_planet::stamp::Stamp;
use helio_pass_voxel_planet::terrain::{material, MaterialAppearance, TerrainAppearance};

/// The Moon's mean radius (m).
const MOON_RADIUS: f64 = 1_737_400.0;

/// Materials the flight's worlds add past the built-in ones (ids 22..32):
/// conifer needles, birch bark, a dry shrub's grey-green leaves and wood.
const NEEDLES: u32 = 22;
const BIRCH_BARK: u32 = 23;
const SAGE: u32 = 24;
const DRY_WOOD: u32 = 25;

/// Variants grown of each kind of feature.
const VARIANTS: u64 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    Earth,
    Moon,
    Desert,
}

impl Preset {
    /// `HELIO_VOXEL_FLIGHT_PRESET` (Earth).
    pub fn from_env() -> Self {
        match std::env::var("HELIO_VOXEL_FLIGHT_PRESET").as_deref() {
            Ok("moon") => Preset::Moon,
            Ok("desert") => Preset::Desert,
            Ok("earth") | Err(_) => Preset::Earth,
            Ok(other) => panic!("unknown preset {other}"),
        }
    }

    pub fn stack(self) -> TerrainLayers {
        match self {
            Preset::Earth => TerrainLayers::earth(),
            Preset::Moon => TerrainLayers::moon(),
            Preset::Desert => TerrainLayers::desert(),
        }
    }

    pub fn radius_m(self) -> f64 {
        match self {
            Preset::Moon => MOON_RADIUS,
            Preset::Earth | Preset::Desert => EARTH_RADIUS,
        }
    }

    /// What grows and lies on the ground, grown from `seed` in cells of
    /// `voxel` metres.
    pub fn features(self, voxel: f64, seed: u64) -> FeatureSet {
        let mut set = Decor { set: FeatureSet::default(), voxel, seed };
        match self {
            Preset::Earth => set.earth(),
            Preset::Moon => set.moon(),
            Preset::Desert => set.desert(),
        }
        set.set
    }

    /// The world's look: the generator's (`base`) with the materials its
    /// features add.
    pub fn appearance(self, mut base: TerrainAppearance) -> TerrainAppearance {
        let unit = |c: [u8; 3]| c.map(|v| f32::from(v) / 255.0);
        let look = |c: [u8; 3], roughness: f32, patches: Option<[[u8; 3]; 3]>, transmission: f32| {
            let [r, g, b] = unit(c);
            MaterialAppearance { colour: [r, g, b, roughness], patches: patches.map(|p| p.map(unit)), transmission, ..MaterialAppearance::default() }
        };
        // Leaves vary by patches over the land, as the turf does.
        base.materials[material::LEAVES as usize].patches = Some([[86, 112, 50], [59, 102, 52], [40, 80, 40]].map(unit));
        base.materials[NEEDLES as usize] = look([42, 68, 48], 0.9, Some([[62, 84, 54], [42, 68, 48], [30, 54, 42]]), 0.5);
        base.materials[BIRCH_BARK as usize] = look([212, 208, 196], 0.8, None, 0.0);
        base.materials[SAGE as usize] = look([146, 150, 116], 0.92, Some([[170, 164, 124], [146, 150, 116], [116, 126, 96]]), 0.4);
        base.materials[DRY_WOOD as usize] = look([110, 94, 78], 0.9, None, 0.0);
        base
    }

    /// The air (none on the Moon), around a planet of `radius` m.
    pub fn sky(self, radius: f64) -> Option<helio_pass_sky::AtmosphereComponent> {
        use helio_pass_sky::{AtmosphereComponent, CloudLayer};
        let clouds = std::env::var_os("HELIO_VOXEL_FLIGHT_NO_CLOUDS").is_none();
        match self {
            Preset::Moon => None,
            // Earth's air with fair-weather cumulus.
            Preset::Earth => {
                let sky = AtmosphereComponent::earth().around_planet([0.0; 3], radius);
                Some(if clouds { sky.with_clouds(CloudLayer::cumulus()) } else { sky })
            }
            // Dusty air: more and redder aerosols, a sandy ground bouncing
            // light into it, a few high clouds.
            Preset::Desert => {
                let earth = AtmosphereComponent::earth();
                let sky = AtmosphereComponent {
                    mie_scattering: [9.0e-3, 8.0e-3, 6.5e-3],
                    mie_absorption: [6.0e-3, 7.0e-3, 9.0e-3],
                    mie_scale_height: 1.6,
                    ground_albedo: [0.45, 0.38, 0.28],
                    ..earth
                }
                .around_planet([0.0; 3], radius);
                let high = CloudLayer { bottom_km: 4.5, top_km: 6.0, coverage: 0.2, kind: 0.4, ..CloudLayer::cumulus() };
                Some(if clouds { sky.with_clouds(high) } else { sky })
            }
        }
    }

    /// A ground haze thinning with altitude over the planet (none on the
    /// Moon), unless `HELIO_VOXEL_FLIGHT_NO_HAZE`.
    pub fn haze(self, radius: f64) -> Option<helio_pass_volumetric_fog::GlobalFogComponent> {
        use helio_pass_volumetric_fog::GlobalFogComponent;
        if std::env::var_os("HELIO_VOXEL_FLIGHT_NO_HAZE").is_some() {
            return None;
        }
        let haze = match self {
            Preset::Moon => return None,
            Preset::Earth => GlobalFogComponent { mode: 1, extinction: 0.0006, height_falloff: 0.004, anisotropy: 0.6, ..Default::default() },
            // Dust: thicker, warm, hugging the ground.
            Preset::Desert => GlobalFogComponent { mode: 1, extinction: 0.0012, height_falloff: 0.003, anisotropy: 0.7, albedo: [0.95, 0.84, 0.68], ..Default::default() },
        };
        Some(haze.around_planet([0.0; 3], radius))
    }

    /// The light the sky gives where it has no air to scatter it (the
    /// renderer's ambient): none on the Moon. Airy worlds' skies light
    /// themselves.
    pub fn ambient(self) -> ([f32; 3], f32) {
        match self {
            Preset::Moon => ([0.0; 3], 0.0),
            Preset::Earth | Preset::Desert => ([0.45, 0.58, 0.8], 0.35),
        }
    }

    /// The camera's exposure: voxel_craft's under air; on the Moon a
    /// camera exposes for sunlit regolith against a black sky (no sky
    /// lights the shade, nothing dims the sun).
    pub fn exposure(self) -> f32 {
        match self {
            Preset::Moon => 3.5,
            Preset::Earth | Preset::Desert => 2.0,
        }
    }

    /// The spawn's ground is level within this many metres 30 m around it:
    /// on the Moon a plain between craters (the others keep the first land
    /// found, as before).
    pub fn level_spawn_m(self) -> Option<f64> {
        match self {
            Preset::Moon => Some(1.5),
            Preset::Earth | Preset::Desert => None,
        }
    }

    /// Foliage over the ground, unless `HELIO_VOXEL_FLIGHT_NO_FOLIAGE`:
    /// Earth's meadows with stands of tall grass in a fresh breeze.
    pub fn meadow(self) -> Option<Meadow> {
        if std::env::var_os("HELIO_VOXEL_FLIGHT_NO_FOLIAGE").is_some() {
            return None;
        }
        match self {
            Preset::Earth => Some(helio_pass_voxel_planet::meadow::meadow().with_tall_stands().with_wind([1.0, 0.35], 4.5, 0.8)),
            Preset::Moon | Preset::Desert => None,
        }
    }
}

/// A feature set being grown: stamps added by kind, spawns in order (the
/// widest spacing first).
struct Decor {
    set: FeatureSet,
    voxel: f64,
    seed: u64,
}

impl Decor {
    /// Metres as cells.
    fn cells(&self, m: f64) -> u32 {
        (m / self.voxel).round().max(1.0) as u32
    }

    /// Metres between candidates as a spacing (cells, at most the widest).
    fn spacing(&self, m: f64) -> u32 {
        self.cells(m).min(MAX_SPACING)
    }

    /// `stamps` into the set: their indices.
    fn add(&mut self, stamps: impl IntoIterator<Item = Stamp>) -> Vec<u32> {
        let first = self.set.stamps.len() as u32;
        self.set.stamps.extend(stamps);
        (first..self.set.stamps.len() as u32).collect()
    }

    /// `VARIANTS` stamps of `grow` from seeds of this world and `kind`.
    fn grown(&mut self, kind: u64, grow: impl Fn(u64) -> Stamp) -> Vec<u32> {
        let seed = self.seed;
        self.add((0..VARIANTS).map(|v| grow(seed ^ (kind << 40) ^ v.wrapping_mul(0x9e37_79b9))))
    }

    /// Trees of `species` of `wood` and `leaves`.
    fn trees(&mut self, kind: u64, species: &[(Species, u32, u32)]) -> Vec<u32> {
        let voxel = self.voxel;
        let mut all = Vec::new();
        for (n, &(s, wood, leaves)) in species.iter().enumerate() {
            all.extend(self.grown(kind + n as u64, |seed| flora::tree(s, wood, leaves, seed, voxel)));
        }
        all
    }

    /// Woods on meadows and moors below the tree line, broadleaf low and
    /// conifers high; boulders on rock, scree and snow; bushes in the
    /// meadows; shrubs and rocks on dry land.
    fn earth(&mut self) {
        let voxel = self.voxel;
        // Broadleaves give way to conifers over this height (m), and those
        // stop at the tree line, below the snow; both lines wander a few
        // hundred metres over half a kilometre (no contour lines).
        let (conifer_line, tree_line) = (1_100.0, 2_600.0);
        let lowland = [-1e9, conifer_line];
        let (lowland_wander, tree_line_wander) = ([150.0, 500.0], [200.0, 450.0]);
        let broadleaf = self.trees(1, &[(Species::Oak, material::WOOD, material::LEAVES), (Species::Oak, material::WOOD, material::LEAVES), (Species::DarkOak, material::WOOD, material::LEAVES)]);
        let birch = self.trees(4, &[(Species::Birch, BIRCH_BARK, material::LEAVES)]);
        let conifer = self.trees(5, &[(Species::Spruce, material::WOOD, NEEDLES), (Species::Pine, material::WOOD, NEEDLES)]);
        let boulders = self.grown(8, |seed| flora::boulder(if seed % 3 == 0 { material::DARK_STONE } else { material::STONE }, seed, voxel));
        let bushes = self.grown(9, |seed| flora::bush(material::LEAVES, seed, voxel));
        let shrubs = self.grown(10, |seed| flora::shrub(DRY_WOOD, SAGE, seed, voxel));
        let rocks = self.grown(11, |seed| flora::boulder_sized(material::SANDSTONE, seed, voxel, [0.3, 0.9]));
        let meadow = vec![material::GRASS, material::DIRT];
        let forest = |name: &str, stamps: Vec<u32>, field_scale_m: f64, field_threshold: f32, far: u32| Spawn {
            name: name.into(),
            stamps,
            spacing: self.spacing(9.0),
            density: 0.85,
            field_scale_m,
            field_threshold,
            surfaces: meadow.clone(),
            heights_m: lowland,
            height_wander_m: lowland_wander,
            max_relief: self.cells(3.0),
            far_material: far,
            ..Spawn::default()
        };
        self.set.spawns = vec![
            forest("broadleaf woods", broadleaf, 2_400.0, 0.05, material::LEAVES),
            forest("birch woods", birch, 900.0, 0.45, material::LEAVES),
            Spawn { heights_m: [conifer_line, tree_line], height_wander_m: tree_line_wander, ..forest("conifers", conifer, 1_600.0, -0.25, NEEDLES) },
            Spawn { name: "boulders".into(), stamps: boulders, spacing: self.spacing(7.0), density: 0.1, surfaces: vec![material::STONE, material::DARK_STONE, material::GRAVEL, material::SNOW, material::GRASS], ..Spawn::default() },
            Spawn { name: "dry rocks".into(), stamps: rocks, spacing: self.spacing(7.0), density: 0.08, surfaces: vec![material::SAND, material::SANDSTONE], ..Spawn::default() },
            Spawn { name: "dry shrubs".into(), stamps: shrubs, spacing: self.spacing(5.0), density: 0.18, field_scale_m: 300.0, field_threshold: -0.4, surfaces: vec![material::SAND], max_relief: self.cells(1.0), ..Spawn::default() },
            Spawn { name: "bushes".into(), stamps: bushes, spacing: self.spacing(3.0), density: 0.12, field_scale_m: 200.0, field_threshold: -0.2, surfaces: meadow.clone(), heights_m: [-1e9, tree_line], height_wander_m: tree_line_wander, max_relief: self.cells(1.0), ..Spawn::default() },
        ];
    }

    /// Simple craters from 36 m down to a metre, many more small than
    /// large, rimmed with the ground they hit (a third of them fresh:
    /// bright ejecta and thrown blocks); boulders, thick on fresh ejecta.
    fn moon(&mut self) {
        let voxel = self.voxel;
        let crater = |rim: u32, rocks: u32, blocks: u32, diameter: [f64; 2]| move |seed: u64| flora::crater(rim, rocks, blocks, seed, voxel, diameter);
        let mut sizes = Vec::new();
        for (n, (diameter, spacing, density, blocks)) in [([14.0, 36.0], 100.0, 0.3, 14), ([5.0, 14.0], 40.0, 0.35, 8), ([1.2, 5.0], 12.0, 0.35, 0)].into_iter().enumerate() {
            let kind = 20 + 4 * n as u64;
            let highland = self.grown(kind, crater(material::REGOLITH, material::BOULDER, blocks / 2, diameter));
            let mare = self.grown(kind + 1, crater(material::MARE, material::BASALT, blocks / 2, diameter));
            let fresh = self.grown(kind + 2, crater(material::EJECTA, material::BOULDER, blocks, diameter));
            sizes.push((diameter, spacing, density, highland, mare, fresh));
        }
        let boulders = self.grown(40, |seed| flora::boulder_sized(material::BOULDER, seed, voxel, [0.3, 1.6]));
        let blocks = self.grown(41, |seed| flora::boulder_sized(material::BOULDER, seed, voxel, [0.1, 0.45]));
        let mut spawns = Vec::new();
        for (diameter, spacing, density, highland, mare, fresh) in sizes {
            let craters = |name: String, stamps: Vec<u32>, surfaces: Vec<u32>, density: f32| Spawn {
                name,
                stamps,
                spacing: self.spacing(spacing),
                density,
                surfaces,
                // Gentle ground: a crater's rim stays level.
                max_relief: self.cells(diameter[1] * 0.12),
                ..Spawn::default()
            };
            let size = format!("{:.0}-{:.0} m", diameter[0], diameter[1]);
            // Fresh craters are rarer: a third of the young-looking ones.
            spawns.push(craters(format!("fresh craters {size}"), fresh, vec![material::REGOLITH, material::MARE, material::EJECTA], density * 0.3));
            spawns.push(craters(format!("highland craters {size}"), highland, vec![material::REGOLITH, material::EJECTA], density));
            spawns.push(craters(format!("mare craters {size}"), mare, vec![material::MARE], density));
        }
        spawns.push(Spawn { name: "boulders".into(), stamps: boulders, spacing: self.spacing(6.0), density: 0.08, ..Spawn::default() });
        spawns.push(Spawn { name: "boulder fields".into(), stamps: blocks, spacing: self.spacing(2.5), density: 0.12, surfaces: vec![material::EJECTA], ..Spawn::default() });
        self.set.spawns = spawns;
    }

    /// Sandstone boulders, rocks and sparse shrubs over the dunes.
    fn desert(&mut self) {
        let voxel = self.voxel;
        let sandstone = self.grown(50, |seed| flora::boulder_sized(material::SANDSTONE, seed, voxel, [1.5, 3.5]));
        let rocks = self.grown(51, |seed| flora::boulder_sized(if seed % 3 == 0 { material::DARK_STONE } else { material::STONE }, seed, voxel, [0.25, 0.9]));
        let shrubs = self.grown(52, |seed| flora::shrub(DRY_WOOD, SAGE, seed, voxel));
        self.set.spawns = vec![
            Spawn { name: "sandstone boulders".into(), stamps: sandstone, spacing: self.spacing(36.0), density: 0.25, field_scale_m: 900.0, field_threshold: -0.2, ..Spawn::default() },
            Spawn { name: "rocks".into(), stamps: rocks, spacing: self.spacing(8.0), density: 0.07, ..Spawn::default() },
            Spawn { name: "shrubs".into(), stamps: shrubs, spacing: self.spacing(6.0), density: 0.12, field_scale_m: 450.0, field_threshold: -0.35, max_relief: self.cells(1.0), ..Spawn::default() },
        ];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every preset's features are a valid set of its planet.
    #[test]
    fn presets_build_valid_planets() {
        for preset in [Preset::Earth, Preset::Moon, Preset::Desert] {
            let recipe = helio_pass_voxel_planet::PlanetRecipe { radius_m: preset.radius_m(), terrain: preset.stack().source(7), features: preset.features(0.1, 7), ..Default::default() };
            helio_pass_voxel_planet::Planet::new(recipe).unwrap_or_else(|e| panic!("{preset:?}: {e}"));
        }
    }
}
