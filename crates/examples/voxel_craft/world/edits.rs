//! Block edits and the brushes that make them.
//!
//! A block is one cell of the voxel planet; a one-block cube brush at its
//! centre is exactly that cell. Edits are applied in order, later ones over
//! earlier ones.

use crate::blocks::Block;
use helio_pass_voxel_planet::stamp::cell_brush;
use helio_pass_voxel_planet::{Brush, Cell, Grid};

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
        cell_brush(grid, self.cell, self.block.map(Block::material))
    }
}
