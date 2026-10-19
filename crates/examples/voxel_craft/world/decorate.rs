//! Decorating the terrain: trees, cacti and structures, as the voxel planet's
//! features (`helio_pass_voxel_planet::features`).
//!
//! The planet is divided into square regions of `REGION_BLOCKS` columns per
//! face. A region's decoration is a pure function of the world seed, the
//! region and the terrain (never edits or other regions), so a region
//! unloaded far behind the player comes back the same when they return,
//! and what players dig or build applies over it.

use crate::blocks::Block;
use crate::config::{REGION_BLOCKS, WORLD_SEED};
use crate::structures::Kind;
use crate::terrain::climate::{forest_density, forest_kind, hash, unit};
use crate::terrain::{self, Biome};
use crate::trees::{Rng, Species, Tree};
use helio_pass_voxel_planet::features::Region;
use helio_pass_voxel_planet::stamp::{column_box, Stamp};
use helio_pass_voxel_planet::terrain::material;
use helio_pass_voxel_planet::{Brush, Planet};

/// Chance a region holds a structure.
const STRUCTURE_CHANCE: f64 = 0.09;

/// Whether `region` lies where features stand (clear of the face's edges).
pub fn decorated(planet: &Planet, region: Region) -> bool {
    terrain::buildable(planet, region.x * REGION_BLOCKS + REGION_BLOCKS / 2, region.y * REGION_BLOCKS + REGION_BLOCKS / 2)
}

/// `region`'s features: the brushes that place them, in order.
pub fn decorate(planet: &Planet, region: Region) -> Vec<Brush> {
    let seed = WORLD_SEED;
    let Region { face, x: rx, y: ry } = region;
    let (i0, j0) = (rx * REGION_BLOCKS, ry * REGION_BLOCKS);
    let mut rng = Rng::new(hash(seed, i64::from(face), i64::from(rx), i64::from(ry)));
    let mut brushes = Vec::new();
    // Footprints taken: (i, j, radius).
    let mut taken: Vec<(i32, i32, i32)> = Vec::new();
    let grid = *planet.grid();

    // A structure, near the region's middle.
    if unit(hash(seed ^ 0xabc, i64::from(face), i64::from(rx), i64::from(ry))) < STRUCTURE_CHANCE {
        let (ci, cj) = (i0 + REGION_BLOCKS / 2 + rng.int(-5, 5), j0 + REGION_BLOCKS / 2 + rng.int(-5, 5));
        if let Some(g) = terrain::ground(planet, face, ci, cj) {
            if let Some(kind) = Kind::for_biome(Biome::of(g.material), rng.f64()) {
                let blueprint = kind.build(rng.u64()).oriented(rng.int(0, 3) as u32, rng.f64() < 0.5);
                let [hx, hz] = blueprint.half;
                let reach = hx.max(hz) + 1;
                // The ground under the footprint's corners and middle.
                let tops: Vec<i32> = [(0, 0), (hx, hz), (-hx, hz), (hx, -hz), (-hx, -hz)].into_iter().map(|(di, dj)| planet.terrain_top(face, ci + di, cj + dj)).collect();
                let (low, high) = (*tops.iter().min().unwrap(), *tops.iter().max().unwrap());
                if high - low <= 7 {
                    let floor = (tops.iter().sum::<i32>() as f64 / tops.len() as f64).round() as i32;
                    let columns = 2 * reach + 3;
                    // Clear what stands in the way, set a foundation, build.
                    brushes.push(column_box(&grid, face, ci, cj, floor, columns, blueprint.height + 6, None));
                    brushes.push(column_box(&grid, face, ci, cj, floor - 5, columns - 2, 5, Some(Block::Cobblestone.material())));
                    brushes.extend(blueprint.stamp.brushes(&grid, face, ci, cj, floor));
                    taken.push((ci, cj, reach + 3));
                }
            }
        }
    }

    // Trees and cacti.
    let (mid_i, mid_j) = (i0 + REGION_BLOCKS / 2, j0 + REGION_BLOCKS / 2);
    let density = forest_density(seed, face, mid_i, mid_j);
    let kind = forest_kind(seed, face, mid_i, mid_j);
    let attempts = 3 + (density * 22.0) as i32;
    let mut trunks: Vec<(i32, i32)> = Vec::new();
    let mut growth = Stamp::new();
    for _ in 0..attempts {
        let (i, j) = (i0 + rng.int(3, REGION_BLOCKS - 4), j0 + rng.int(3, REGION_BLOCKS - 4));
        if taken.iter().any(|&(ti, tj, r)| (ti - i).abs() <= r && (tj - j).abs() <= r) {
            continue;
        }
        // Trunks keep apart: close ones would merge their crowns.
        if trunks.iter().any(|&(ti, tj)| (ti - i).abs() < 4 && (tj - j).abs() < 4) {
            continue;
        }
        let Some(g) = terrain::ground(planet, face, i, j) else { continue };
        if terrain::relief(planet, face, i, j, 2) > 3 {
            continue;
        }
        // Offsets from the region's first column on the ground's layer.
        let at = [i - i0, g.cell.k + 1, j - j0];
        match Biome::of(g.material) {
            Biome::Desert => {
                // Cacti on sand, spread thin.
                if g.material == material::SAND && rng.f64() < 0.08 + density * 0.1 {
                    growth.fill(at, [at[0], at[1] + rng.int(2, 4) - 1, at[2]], Block::Cactus.material());
                    trunks.push((i, j));
                }
            }
            Biome::Rock => {}
            biome => {
                let height_m = f64::from(g.cell.k) * grid.voxel_size();
                let Some(species) = pick_species(biome, height_m, kind, density, &mut rng) else { continue };
                // A trunk of a big tree takes room: keep clear of the rest.
                let reach = if species == Species::Jungle { 8 } else { 3 };
                if trunks.iter().any(|&(ti, tj)| (ti - i).abs() < reach && (tj - j).abs() < reach) {
                    continue;
                }
                let mut tree = Tree::grow(species, hash(seed, i64::from(i), i64::from(j), 77)).stamp();
                // Crowns stop at hillsides: no leaves inside the terrain.
                let logs = [Block::OakLog, Block::BirchLog, Block::SpruceLog, Block::JungleLog].map(Block::material);
                tree.retain(|p, m| m.is_some_and(|m| logs.contains(&m)) || planet.terrain_top(face, i + p[0], j + p[2]) <= g.cell.k + 1 + p[1]);
                growth.place(&tree, at);
                // Dirt under the trunk, as the roots would make it.
                if g.material == material::GRASS {
                    growth.set([at[0], at[1] - 1, at[2]], Block::Dirt.material());
                }
                trunks.push((i, j));
            }
        }
    }
    brushes.extend(growth.brushes(&grid, face, i0, j0, 0));
    brushes
}

/// The tree a column grows, if any.
fn pick_species(biome: Biome, height_m: f64, kind: f64, density: f64, rng: &mut Rng) -> Option<Species> {
    // Sparse where the forest field is thin; none on open meadows.
    if rng.f64() > 0.35 + density {
        return None;
    }
    if biome == Biome::Snow || height_m > 55.0 {
        return Some(if rng.f64() < 0.65 { Species::Spruce } else { Species::Pine });
    }
    Some(match kind {
        k if k < 0.30 => {
            if rng.f64() < 0.18 {
                Species::DarkOak
            } else {
                Species::Oak
            }
        }
        k if k < 0.55 => {
            if rng.f64() < 0.7 {
                Species::Birch
            } else {
                Species::Oak
            }
        }
        k if k < 0.78 => {
            if rng.f64() < 0.4 {
                Species::Jungle
            } else {
                Species::Oak
            }
        }
        _ => {
            if rng.f64() < 0.55 {
                Species::Spruce
            } else {
                Species::Birch
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::recipe::recipe;
    use helio_pass_voxel_planet::{BrushOp, BrushShape, Cell};

    /// A region's features are a function of the seed and the terrain: the
    /// same brushes when decided again, after its neighbours' features loaded
    /// and players dug over it, and after it was unloaded and loaded back.
    #[test]
    fn regions_come_out_the_same_whenever_they_are_decided() {
        let mut planet = Planet::new(recipe()).unwrap();
        let n = planet.grid().cells() / REGION_BLOCKS;
        let centre = Region { face: 2, x: n / 2, y: n / 2 };
        let around = centre.around(2, n + 1);
        let first: Vec<Vec<Brush>> = around.iter().map(|r| decorate(&planet, *r)).collect();
        assert!(first.iter().map(Vec::len).sum::<usize>() > 0, "nothing grows around the centre");
        for (region, brushes) in around.iter().zip(&first) {
            planet.load_features(region.key(), brushes.clone()).unwrap();
        }
        let grid = *planet.grid();
        let at = Cell::new(2, centre.x * REGION_BLOCKS + 16, centre.y * REGION_BLOCKS + 16, planet.terrain_top(2, centre.x * REGION_BLOCKS + 16, centre.y * REGION_BLOCKS + 16));
        planet.apply(helio_pass_voxel_planet::Brush { center: grid.cell_center(at).to_array(), radius: 6.0, shape: BrushShape::Sphere, op: BrushOp::Remove, material: 0, height: 0.0 }).unwrap();
        for (region, brushes) in around.iter().zip(&first) {
            assert_eq!(&decorate(&planet, *region), brushes, "{region:?} decided again");
        }
        planet.unload_features(centre.key());
        assert_eq!(decorate(&planet, centre), first[0]);
    }
}
