//! Stamps: small voxel models (a tree, a house, a well), cells of material
//! ids or air around an anchor. Features place them procedurally
//! (`crate::features`, turned and mirrored by each placement); an edit
//! places one as brushes ([`Stamp::brushes`]).
//!
//! Offsets are `[x, y, z]` with +y up the planet's radial axis and `y = 0`
//! the layer the anchor is in; x runs along the face's i axis and z along
//! its j axis.
//!
//! Placing one as edits is a few hundred brushes, not one a cell: vertical
//! runs of a material (a trunk, a wall) become one box brush
//! ([`run_brushes`]), and every brush is history the planet keeps and
//! renderers follow.

use crate::edits::{Brush, BrushOp, BrushShape};
use crate::grid::{Cell, Grid};
use rustc_hash::FxHashMap;

/// Cells of material ids (`Some`) or air (`None`) at offsets from an anchor.
/// Serialized as its cells in order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stamp {
    cells: FxHashMap<[i32; 3], Option<u32>>,
}

impl serde::Serialize for Stamp {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut cells: Vec<([i32; 3], Option<u32>)> = self.cells().collect();
        cells.sort_unstable();
        serde::Serialize::serialize(&cells, serializer)
    }
}

impl<'de> serde::Deserialize<'de> for Stamp {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let cells: Vec<([i32; 3], Option<u32>)> = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self { cells: cells.into_iter().collect() })
    }
}

impl Stamp {
    pub fn new() -> Self {
        Self::default()
    }

    /// Material `material` at `at` (over what was there).
    pub fn set(&mut self, at: [i32; 3], material: u32) {
        self.cells.insert(at, Some(material));
    }

    /// Air at `at`: whatever the ground or a neighbour put there goes.
    pub fn clear(&mut self, at: [i32; 3]) {
        self.cells.insert(at, None);
    }

    /// Fills the box between `a` and `b` (inclusive) with `material`.
    pub fn fill(&mut self, a: [i32; 3], b: [i32; 3], material: u32) {
        for at in box_cells(a, b) {
            self.set(at, material);
        }
    }

    /// Clears the box between `a` and `b` (inclusive).
    pub fn hollow(&mut self, a: [i32; 3], b: [i32; 3]) {
        for at in box_cells(a, b) {
            self.clear(at);
        }
    }

    /// The four walls of the box between `a` and `b` (its outer cells on
    /// the x and z sides), of `material`.
    pub fn walls(&mut self, a: [i32; 3], b: [i32; 3], material: u32) {
        let (lo, hi) = (min3(a, b), max3(a, b));
        for at in box_cells(lo, hi) {
            if at[0] == lo[0] || at[0] == hi[0] || at[2] == lo[2] || at[2] == hi[2] {
                self.set(at, material);
            }
        }
    }

    /// Every cell of `other` over this stamp's, offset by `at`.
    pub fn place(&mut self, other: &Stamp, at: [i32; 3]) {
        for (p, m) in &other.cells {
            self.cells.insert([p[0] + at[0], p[1] + at[1], p[2] + at[2]], *m);
        }
    }

    /// What the stamp holds at `at`: `None` where it holds nothing,
    /// `Some(None)` air.
    pub fn get(&self, at: [i32; 3]) -> Option<Option<u32>> {
        self.cells.get(&at).copied()
    }

    /// The stamp's cells, in no order.
    pub fn cells(&self) -> impl Iterator<Item = ([i32; 3], Option<u32>)> + '_ {
        self.cells.iter().map(|(p, m)| (*p, *m))
    }

    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The smallest and largest offsets it holds (`None` when empty).
    pub fn bounds(&self) -> Option<([i32; 3], [i32; 3])> {
        let mut cells = self.cells.keys();
        let first = *cells.next()?;
        Some(cells.fold((first, first), |(lo, hi), p| (min3(lo, *p), max3(hi, *p))))
    }

    /// The brushes that place the stamp with its origin at cell `(i, j, k)`
    /// of `face`: its air first (so it clears what stands there), then its
    /// materials, vertical runs as one brush each.
    pub fn brushes(&self, grid: &Grid, face: u8, i: i32, j: i32, k: i32) -> Vec<Brush> {
        let at = |p: [i32; 3]| Cell::new(face, i + p[0], j + p[2], k + p[1]);
        let (air, solid): (Vec<_>, Vec<_>) = self.cells.iter().map(|(p, m)| (at(*p), *m)).partition(|(_, m)| m.is_none());
        let mut out = run_brushes(grid, air);
        out.extend(run_brushes(grid, solid));
        out
    }
}

/// The brush that makes `cell` of `material`, or air (`None`): a cube of the
/// cell exactly.
pub fn cell_brush(grid: &Grid, cell: Cell, material: Option<u32>) -> Brush {
    let (op, material) = material.map_or((BrushOp::Remove, 0), |m| (BrushOp::Add, m));
    Brush { center: grid.cell_center(cell).to_array(), radius: grid.voxel_size() * 0.5, shape: BrushShape::Cube, op, material, height: 0.0 }
}

/// A flat-topped box brush over `columns` x `columns` columns (odd, centred
/// on column `(i, j)` of `face`), `layers` tall from layer `k`, of
/// `material` or air: one brush fills or clears what would take thousands
/// of cells (a building's footprint).
pub fn column_box(grid: &Grid, face: u8, i: i32, j: i32, k: i32, columns: i32, layers: i32, material: Option<u32>) -> Brush {
    let voxel = grid.voxel_size();
    let centre = grid.position(face, [f64::from(i) + 0.5, f64::from(j) + 0.5, f64::from(k) + f64::from(layers) * 0.5]);
    let (op, material) = material.map_or((BrushOp::Remove, 0), |m| (BrushOp::Add, m));
    Brush { center: centre.to_array(), radius: f64::from(columns) * voxel * 0.5, shape: BrushShape::Cube, op, material, height: f64::from(layers) * voxel * 0.5 }
}

/// Brushes for `cells`, a vertical run of one material (or of air) in a
/// column made one box brush. Later cells of the list win where it names a
/// cell twice.
pub fn run_brushes(grid: &Grid, cells: Vec<(Cell, Option<u32>)>) -> Vec<Brush> {
    let mut unique: FxHashMap<Cell, Option<u32>> = FxHashMap::default();
    for (cell, m) in cells {
        unique.insert(cell, m);
    }
    let mut cells: Vec<(Cell, Option<u32>)> = unique.into_iter().collect();
    cells.sort_by_key(|(c, m)| (c.face, c.i, c.j, *m, c.k));
    let mut out = Vec::new();
    let mut run: Option<(Cell, Option<u32>, i32)> = None;
    let flush = |run: Option<(Cell, Option<u32>, i32)>, out: &mut Vec<Brush>| {
        if let Some((start, m, length)) = run {
            out.push(if length == 1 { cell_brush(grid, start, m) } else { column_box(grid, start.face, start.i, start.j, start.k, 1, length, m) });
        }
    };
    for (cell, m) in cells {
        run = match run {
            Some((start, rm, length)) if rm == m && start.face == cell.face && start.i == cell.i && start.j == cell.j && start.k + length == cell.k => Some((start, rm, length + 1)),
            other => {
                flush(other, &mut out);
                Some((cell, m, 1))
            }
        };
    }
    flush(run, &mut out);
    out
}

fn min3(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

fn max3(a: [i32; 3], b: [i32; 3]) -> [i32; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

fn box_cells(a: [i32; 3], b: [i32; 3]) -> impl Iterator<Item = [i32; 3]> {
    let (lo, hi) = (min3(a, b), max3(a, b));
    (lo[0]..=hi[0]).flat_map(move |x| (lo[1]..=hi[1]).flat_map(move |y| (lo[2]..=hi[2]).map(move |z| [x, y, z])))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Planet, PlanetRecipe};

    /// A placed stamp reads back cell for cell: its materials where it set
    /// them, air where it cleared, in a handful of run brushes.
    #[test]
    fn a_placed_stamp_reads_back_cell_for_cell() {
        let mut planet = Planet::new(PlanetRecipe::default()).unwrap();
        let grid = *planet.grid();
        let (face, i, j) = (2u8, grid.cells() / 2, grid.cells() / 2);
        let k = planet.column_top(face, i, j, 0) + 1;
        let mut house = Stamp::new();
        house.fill([-3, -1, -2], [3, -1, 2], 7);
        house.walls([-3, 0, -2], [3, 3, 2], 12);
        house.hollow([0, 0, -2], [0, 1, -2]);
        house.fill([-2, 0, -1], [2, 3, 1], 7);
        house.hollow([-2, 0, -1], [2, 3, 1]);
        let brushes = house.brushes(&grid, face, i, j, k);
        assert!(brushes.len() * 2 < house.len(), "{} brushes for {} cells", brushes.len(), house.len());
        for brush in brushes {
            planet.apply(brush).unwrap();
        }
        for (p, m) in house.cells() {
            let cell = Cell::new(face, i + p[0], j + p[2], k + p[1]);
            assert_eq!(planet.solid(cell), m.is_some(), "{p:?}");
            if let Some(m) = m {
                assert_eq!(planet.material(cell), m, "{p:?}");
            }
        }
    }
}
