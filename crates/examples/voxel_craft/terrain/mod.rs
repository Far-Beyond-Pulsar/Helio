//! The terrain: its recipe, surface queries and biomes.

pub mod climate;
pub mod recipe;

use crate::blocks::Block;
use helio_pass_voxel_planet::terrain::material;
use helio_pass_voxel_planet::{Cell, Planet};

/// The topmost solid block of a column.
#[derive(Clone, Copy, Debug)]
pub struct Ground {
    pub cell: Cell,
    /// The material id of the block.
    pub material: u32,
}

impl Ground {
    /// The cell above the ground (where things are built).
    pub fn above(&self) -> Cell {
        Cell::new(self.cell.face, self.cell.i, self.cell.j, self.cell.k + 1)
    }
}

/// Faces' edge margin (cells): features keep this far from a cube edge, where
/// a column's neighbours would be on another face.
pub const EDGE_MARGIN: i32 = 80;

/// Whether `(i, j)` is far enough from the face's edges to build on.
pub fn buildable(planet: &Planet, i: i32, j: i32) -> bool {
    let n = planet.grid().cells();
    i >= EDGE_MARGIN && j >= EDGE_MARGIN && i < n - EDGE_MARGIN && j < n - EDGE_MARGIN
}

/// The topmost solid cell of column `(i, j)` of `face` at or below its
/// generated top (caves and edits included), if within `reach` cells.
pub fn ground(planet: &Planet, face: u8, i: i32, j: i32) -> Option<Ground> {
    let top = planet.column_top(face, i, j, 0);
    // Cells over the generated top may be built; start from there.
    let mut k = top + 24;
    let floor = top - 64;
    while k > floor {
        let cell = Cell::new(face, i, j, k);
        if planet.solid(cell) {
            return Some(Ground { cell, material: planet.material(cell) });
        }
        k -= 1;
    }
    None
}

/// Heights of the ground around a column, for slope tests: the largest
/// difference from the column's own top among its neighbours `d` cells away.
pub fn relief(planet: &Planet, face: u8, i: i32, j: i32, d: i32) -> i32 {
    let here = planet.column_top(face, i, j, 0);
    [(d, 0), (-d, 0), (0, d), (0, -d)]
        .into_iter()
        .map(|(di, dj)| (planet.column_top(face, i + di, j + dj, 0) - here).abs())
        .max()
        .unwrap_or(0)
}

/// What kind of place a ground block says it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Biome {
    /// Grass: meadows and forests.
    Temperate,
    /// Sand and sandstone.
    Desert,
    /// Snow caps and cold slopes.
    Snow,
    /// Bare rock and scree.
    Rock,
}

impl Biome {
    pub fn of(ground_material: u32) -> Biome {
        match ground_material {
            material::GRASS | material::DIRT | material::CLAY => Biome::Temperate,
            material::SAND | material::SANDSTONE => Biome::Desert,
            material::SNOW => Biome::Snow,
            _ => Biome::Rock,
        }
    }
}

/// The block a material id is, when it is one of the game's.
pub fn block_of(material: u32) -> Option<Block> {
    Block::from_material(material)
}
