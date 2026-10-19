//! Decorating the terrain: trees, cacti and structures, as the voxel planet's
//! features (`helio_pass_voxel_planet::features`): a library of stamps grown
//! once from the world seed (tree variants of each species, structures with
//! their foundations, cacti) and spawns saying where each grows (biome by
//! the ground's surface, height, flatness, forest fields). The planet
//! evaluates them like its terrain, wherever and at whatever level it is
//! drawn: nothing is decided around the player or kept in memory, forests
//! show to the horizon (their far forms), and what players dig or build
//! applies over them.

use crate::blocks::Block;
use crate::config::WORLD_SEED;
use crate::structures::Kind;
use crate::trees::{Rng, Species, Tree};
use helio_pass_voxel_planet::features::{FeatureSet, Spawn};
use helio_pass_voxel_planet::stamp::Stamp;
use helio_pass_voxel_planet::terrain::material;

/// Columns a side of the square each structure, jungle giant, tree and
/// cactus may stand in (crowns reach past theirs: forests interlock).
const STRUCTURE_SPACING: u32 = 96;
const JUNGLE_SPACING: u32 = 16;
const TREE_SPACING: u32 = 8;
const CACTUS_SPACING: u32 = 5;
/// Tree variants grown of each species.
const VARIANTS: u64 = 6;
/// Conifers take over from broadleaves above this height (m).
const TREE_LINE_M: f64 = 55.0;

/// The world's decoration.
pub fn features() -> FeatureSet {
    let mut set = FeatureSet::default();
    let mut rng = Rng::new(WORLD_SEED ^ 0xdec0_7a7e);
    let add = |set: &mut FeatureSet, stamps: Vec<Stamp>| -> Vec<u32> {
        let first = set.stamps.len() as u32;
        set.stamps.extend(stamps);
        (first..set.stamps.len() as u32).collect()
    };
    let structures = |kinds: &[(Kind, usize)], rng: &mut Rng| -> Vec<Stamp> { kinds.iter().flat_map(|&(kind, n)| (0..n).map(|_| structure(kind, rng.u64())).collect::<Vec<_>>()).collect() };
    let temperate = add(&mut set, structures(&[(Kind::Cottage, 4), (Kind::Tower, 2), (Kind::Ruins, 2), (Kind::Well, 2)], &mut rng));
    let desert = add(&mut set, structures(&[(Kind::Pyramid, 3), (Kind::Well, 2)], &mut rng));
    let mountain = add(&mut set, structures(&[(Kind::Ruins, 3)], &mut rng));
    let jungle = add(&mut set, trees(&[Species::Jungle]));
    let broadleaf = add(&mut set, trees(&[Species::Oak, Species::Oak, Species::DarkOak]));
    let birch = add(&mut set, trees(&[Species::Birch]));
    let conifer = add(&mut set, trees(&[Species::Spruce, Species::Pine]));
    let cacti = add(&mut set, (2..5).map(|h| {
        let mut s = Stamp::new();
        s.fill([0, 0, 0], [0, h - 1, 0], Block::Cactus.material());
        s
    }).collect());
    let lowland = [-1e9, TREE_LINE_M];
    let structure = |name: &str, stamps: Vec<u32>, surfaces: &[u32], density: f32| Spawn {
        name: name.into(),
        stamps,
        spacing: STRUCTURE_SPACING,
        density,
        surfaces: surfaces.to_vec(),
        max_relief: 7,
        ..Spawn::default()
    };
    let forest = |name: &str, stamps: Vec<u32>, spacing: u32, field_scale_m: f64, field_threshold: f32, leaves: Block| Spawn {
        name: name.into(),
        stamps,
        spacing,
        density: 0.85,
        field_scale_m,
        field_threshold,
        surfaces: vec![material::GRASS, material::DIRT],
        heights_m: lowland,
        max_relief: 3,
        far_material: leaves.material(),
        ..Spawn::default()
    };
    set.spawns = vec![
        structure("temperate structures", temperate, &[material::GRASS, material::DIRT, material::CLAY], 0.8),
        structure("desert structures", desert, &[material::SAND, material::SANDSTONE], 0.8),
        structure("mountain ruins", mountain, &[material::SNOW, material::STONE, material::GRAVEL], 0.35),
        Spawn { density: 0.6, ..forest("jungle", jungle, JUNGLE_SPACING, 420.0, 0.35, Block::JungleLeaves) },
        forest("broadleaf forest", broadleaf, TREE_SPACING, 260.0, -0.1, Block::OakLeaves),
        forest("birch woods", birch, TREE_SPACING, 300.0, 0.2, Block::BirchLeaves),
        Spawn { surfaces: vec![material::GRASS, material::DIRT, material::SNOW], heights_m: [TREE_LINE_M, 1e9], ..forest("conifers", conifer, TREE_SPACING, 200.0, -0.3, Block::SpruceLeaves) },
        Spawn { name: "cacti".into(), stamps: cacti, spacing: CACTUS_SPACING, density: 0.12, surfaces: vec![material::SAND], max_relief: 2, ..Spawn::default() },
    ];
    set
}

/// `VARIANTS` trees of each of `species`, standing on dirt (the roots).
fn trees(species: &[Species]) -> Vec<Stamp> {
    species
        .iter()
        .enumerate()
        .flat_map(|(n, &s)| (0..VARIANTS).map(move |v| (n as u64, s, v)))
        .map(|(n, s, v)| {
            let mut stamp = Tree::grow(s, WORLD_SEED ^ (n << 32) ^ (v * 0x9e37_79b9) ^ s as u64).stamp();
            stamp.set([0, -1, 0], Block::Dirt.material());
            stamp
        })
        .collect()
}

/// A structure on its site: the room it needs cleared (its footprint and
/// a margin, five layers over it), a cobble foundation under it, the
/// building on the first layer over the ground.
fn structure(kind: Kind, seed: u64) -> Stamp {
    let blueprint = kind.build(seed);
    let [hx, hz] = blueprint.half;
    let h = hx.max(hz) + 2;
    let mut stamp = Stamp::new();
    stamp.hollow([-h, 0, -h], [h, blueprint.height + 5, h]);
    stamp.fill([-h + 1, -5, -h + 1], [h - 1, -1, h - 1], Block::Cobblestone.material());
    stamp.place(&blueprint.stamp, [0, 0, 0]);
    stamp
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::recipe::recipe;
    use helio_pass_voxel_planet::{BrushOp, BrushShape, Cell, Planet};

    /// The decoration is a valid feature set of the world: trees grow
    /// around the middle of a face (their cells read as their blocks), and
    /// a dig through one reads as dug.
    #[test]
    fn trees_grow_and_dig_like_ground() {
        let mut planet = Planet::new(recipe()).unwrap();
        let n = planet.grid().cells();
        let logs = [Block::OakLog, Block::BirchLog, Block::SpruceLog, Block::JungleLog].map(Block::material);
        let mut trunk = None;
        'search: for i in (n / 2 - 200..n / 2 + 200).step_by(3) {
            for j in (n / 2 - 200..n / 2 + 200).step_by(3) {
                let top = planet.terrain_top(2, i, j);
                for k in top..top + 3 {
                    if logs.contains(&planet.material(Cell::new(2, i, j, k))) && planet.solid(Cell::new(2, i, j, k)) {
                        trunk = Some(Cell::new(2, i, j, k));
                        break 'search;
                    }
                }
            }
        }
        let trunk = trunk.expect("no tree grows near the face's middle");
        let grid = *planet.grid();
        planet.apply(helio_pass_voxel_planet::Brush { center: grid.cell_center(trunk).to_array(), radius: 0.5, shape: BrushShape::Cube, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
        assert!(!planet.solid(trunk));
    }
}
