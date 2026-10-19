//! Block edits and the brushes that make them.
//!
//! A block is one cell of the voxel planet; a one-block cube brush at its
//! centre is exactly that cell. Edits are applied in order, later ones over
//! earlier ones.

use crate::blocks::Block;
use crate::config::VOXEL_M;
use helio_pass_voxel_planet::{Brush, BrushOp, BrushShape, Cell, Grid};

/// Set one cell to a block, or to air (`None`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BlockEdit {
    pub cell: Cell,
    pub block: Option<Block>,
}

impl BlockEdit {
    pub fn set(cell: Cell, block: Block) -> Self {
        Self { cell, block: Some(block) }
    }

    pub fn clear(cell: Cell) -> Self {
        Self { cell, block: None }
    }

    /// The brush that makes this edit.
    pub fn brush(&self, grid: &Grid) -> Brush {
        let (op, material) = match self.block {
            Some(block) => (BrushOp::Add, block.material()),
            None => (BrushOp::Remove, 0),
        };
        Brush { center: grid.cell_center(self.cell).to_array(), radius: VOXEL_M * 0.5, shape: BrushShape::Cube, op, material, height: 0.0 }
    }
}

/// A flat-topped box brush: `columns` x `columns` cells (odd, centred on
/// `(i, j)`) and `layers` tall from layer `k`. One brush fills or clears
/// what would take thousands of blocks (a structure's footprint).
pub fn box_brush(grid: &Grid, face: u8, i: i32, j: i32, k: i32, columns: i32, layers: i32, block: Option<Block>) -> Brush {
    let half_columns = f64::from(columns) * VOXEL_M * 0.5;
    let half_height = f64::from(layers) * VOXEL_M * 0.5;
    // The box's centre: its middle column, half way up.
    let centre = grid.position(face, [f64::from(i) + 0.5, f64::from(j) + 0.5, f64::from(k) + f64::from(layers) * 0.5]);
    let (op, material) = match block {
        Some(block) => (BrushOp::Add, block.material()),
        None => (BrushOp::Remove, 0),
    };
    Brush { center: centre.to_array(), radius: half_columns, shape: BrushShape::Cube, op, material, height: half_height }
}
