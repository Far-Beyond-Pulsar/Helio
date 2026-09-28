//! Shape edits of a voxel world: the generic, persistent destruction and
//! construction journal every terrain generator understands.
//!
//! Brushes are ordered: a later brush wins where they overlap. Coordinates
//! are world metres relative to the terrain's origin, so the same journal
//! applies at every voxel size. Per-sample edits live in the payload chunks
//! instead (see `VoxelSampleEdit`).

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoxelBrushShape {
    Sphere,
    /// An axis-aligned cube in the world's own cell axes.
    Cube,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VoxelBrushOp {
    /// Carve cells to air.
    Remove,
    /// Fill cells with `material`.
    Add,
    /// Recolour existing solid cells with `material`.
    Paint,
}

/// One journal entry.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VoxelBrushEdit {
    pub center: [f64; 3],
    pub radius: f64,
    pub shape: VoxelBrushShape,
    pub op: VoxelBrushOp,
    /// Terrain material for `Add` and `Paint` (the generator's material
    /// table); ignored by `Remove`.
    #[serde(default)]
    pub material: u32,
}

impl VoxelBrushEdit {
    /// Finite centre, positive finite radius.
    pub fn validate(&self) -> Result<(), String> {
        if !self.center.iter().all(|v| v.is_finite()) || !(self.radius.is_finite() && self.radius > 0.0) {
            return Err("brush centre and radius must be finite, radius positive".into());
        }
        Ok(())
    }
}
