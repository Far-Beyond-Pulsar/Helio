//! Baked-lighting PVS (potentially-visible-set) data shape.
//!
//! Split out of `helio-bake` because that crate has a hard (non-optional)
//! dependency on `nebula`, Helio's heavyweight offline baker, and is itself
//! only pulled into `helio` behind the optional `"bake"` feature. The host
//! `Renderer` still needs to publish a `baked_pvs` resource-registry slot of
//! a *consistent type* whether or not baking is compiled in (an
//! `Option<BakedPvsRef<'_>>` that is always `None` when `"bake"` is off) — so
//! this crate holds only the plain data shape, no baking logic and no
//! dependency on `nebula`, and is always available regardless of the
//! `"bake"` feature.

/// Owned CPU-side PVS data stored in `helio_bake::BakedData`.
///
/// Published as a zero-copy [`BakedPvsRef`] into the resource registry each frame.
pub struct BakedPvsData {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub cell_count: u32,
    pub words_per_cell: u32,
    pub bits: Vec<u64>,
}

/// Borrowed reference into the pre-baked PVS bitfield grid.
///
/// Zero-copy view — `bits` borrows directly from the `BakedData` owned by
/// `helio-bake`'s `BakeInjectPass`. Valid for the duration of the frame.
#[derive(Clone, Copy)]
pub struct BakedPvsRef<'a> {
    pub world_min: [f32; 3],
    pub world_max: [f32; 3],
    pub grid_dims: [u32; 3],
    pub cell_size: f32,
    pub cell_count: u32,
    pub words_per_cell: u32,
    /// Packed bitfield: `bits[from * words_per_cell + to/64] >> (to%64) & 1 == 1` means
    /// cell `to` is potentially visible from cell `from`.
    pub bits: &'a [u64],
}

impl<'a> BakedPvsRef<'a> {
    /// Returns `true` if cell `to_cell` is potentially visible from cell `from_cell`.
    #[inline]
    pub fn is_visible(&self, from_cell: usize, to_cell: usize) -> bool {
        let idx = from_cell * self.words_per_cell as usize + to_cell / 64;
        if idx >= self.bits.len() {
            return true;
        } // conservative default
        (self.bits[idx] >> (to_cell % 64)) & 1 == 1
    }

    /// Returns the grid-cell index at world position `p`, or `None` if out of bounds.
    #[inline]
    pub fn cell_at(&self, p: [f32; 3]) -> Option<usize> {
        let [gx, gy, gz] = self.grid_dims;
        let dx = ((p[0] - self.world_min[0]) / self.cell_size) as i32;
        let dy = ((p[1] - self.world_min[1]) / self.cell_size) as i32;
        let dz = ((p[2] - self.world_min[2]) / self.cell_size) as i32;
        if dx < 0 || dy < 0 || dz < 0 || dx >= gx as i32 || dy >= gy as i32 || dz >= gz as i32 {
            return None;
        }
        Some(dx as usize + dy as usize * gx as usize + dz as usize * gx as usize * gy as usize)
    }
}
